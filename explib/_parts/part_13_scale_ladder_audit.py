

def _finite_int_ratio(numerator, denominator):
    """Exact integer ratio as a finite float, or None when unrepresentable.

    Python's integer true division is exact for values whose quotient fits a
    float, but converting a huge numerator/denominator to float (or forming a
    quotient above ``float.max``) raises ``OverflowError``.  Scale planning is
    allowed to declare token budgets/parameter counts far beyond training
    hardware; the audit must report the budget as unverifiable instead of
    crashing on a syntactically valid plan.
    """
    try:
        numerator = int(numerator)
        denominator = int(denominator)
    except (TypeError, ValueError, OverflowError):
        return None
    if denominator == 0:
        return None
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(value):
        return None
    # A nonzero integer ratio can underflow to 0.0 (e.g. 1 / 10**400);
    # reporting that as a measured zero would make the budget comparison look
    # defined.  None keeps it explicitly unverifiable.
    if value == 0.0 and numerator != 0:
        return None
    return float(value)


def _finite_float_ratio(numerator, denominator):
    """Finite quotient of two finite positive measurements, or ``None``.

    A crossover panel can declare finite token budgets whose ratio is beyond
    float64 (e.g. 1.7e308 and 1e-300).  Dividing them yields ``inf``; that is
    an unverifiable exposure ratio, not a measured "x inf" confound.
    """
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if not (math.isfinite(numerator) and math.isfinite(denominator)
                and denominator > 0.0):
            return None
        value = numerator / denominator
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return float(value) if math.isfinite(value) else None


def scale_ladder_audit(tiers, *, policy=None, tolerance=None):
    """Pre-flight audit of the scale ladder's budget and transfer assumptions.

    A width/depth comparison is not controlled if the larger model also saw a
    different tokens-per-parameter budget or silently inherited an untested
    learning-rate transfer.  This audit does not choose those values; it makes
    them machine-checkable and persistent before a panel consumes GPU time.

    `tiers` is a list of {name, cfg, params}: `cfg` carries steps / batch_size
    / seq_len / n_train_tokens / lr / warmup / seeds, `params` is the measured
    parameter count (optional; without it the budget per parameter is
    unverifiable).  A budget mismatch is a disclosure warning (a scale study
    may deliberately change budget), never a silent "matched" claim.
    """
    if policy is None:
        policy = dict(SCALE_TRANSFER_POLICY)
    elif not isinstance(policy, dict):
        # `dict(123)`/`dict(True)` raised an unrelated TypeError deep in the
        # audit; a policy is a declared mapping, not an iterable of pairs.
        raise ValueError(f'policy must be a mapping or None, got {type(policy).__name__}')
    else:
        policy = dict(policy)
    _tol_raw = (policy.get('tokens_per_param_tolerance', 0.25)
                if tolerance is None else tolerance)
    if isinstance(_tol_raw, (bool, np.bool_)):
        raise ValueError(f'tokens_per_param_tolerance must be a non-negative real number, got {_tol_raw!r}')
    try:
        tol = float(_tol_raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tokens_per_param_tolerance must be a non-negative real number, got {_tol_raw!r}') from None
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f'tokens_per_param_tolerance must be a finite non-negative number, got {_tol_raw!r}')

    def _as_int(value, label, *, allow_none=False):
        if value is None and allow_none:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{label} must be an integer, got {value!r}')
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            as_float = float(value)
            if not math.isfinite(as_float) or as_float != math.floor(as_float):
                raise ValueError(f'{label} must be an integer, got {value!r}')
            return int(as_float)
        raise ValueError(f'{label} must be an integer, got {value!r}')

    problems, warnings, rows = [], [], []
    try:
        family_m = _as_int(policy.get('family_m', 4), 'family_m')
        if family_m <= 0:
            raise ValueError('family_m must be positive')
    except (TypeError, ValueError, OverflowError) as e:
        family_m = None
        problems.append(f'policy.family_m is invalid ({e}); the family-resolution floor cannot be planned')
    for i, tier in enumerate(tiers or []):
        if not isinstance(tier, dict):
            problems.append(f'tier #{i} is not a dict')
            continue
        name = str(tier.get('name') or f'T{i}')
        _cfg_raw = tier.get('cfg')
        if _cfg_raw is not None and not isinstance(_cfg_raw, dict):
            # A truthy scalar/list here used to reach `cfg.get(...)` and crash
            # the whole pre-flight audit; it is a malformed declaration, not a
            # usable scale recipe.
            problems.append(f'{name}: cfg is not an object ({type(_cfg_raw).__name__})')
            continue
        cfg = _cfg_raw or {}
        row = {'name': name}
        # Optional corpus/tokenizer provenance: the ladder's cross-tier
        # contrasts (paired log PPL ratios, tokens-per-param) are only defined
        # when every tier measured the SAME corpus and tokenisation.  A tier
        # that carries no fingerprint is disclosed as unverifiable, never
        # assumed to match.
        for _fpk in ('corpus_fp', 'fit_fp', 'vocab'):
            _v = cfg.get(_fpk)
            if _v is None:
                _v = tier.get(_fpk)
            if _v is None:
                continue
            if _fpk == 'vocab':
                # Vocab width is an exact tokenizer identity: int(1.5) or
                # int(True) would silently attest a different tokenisation.
                try:
                    _v = _as_int(_v, f'{name}.vocab')
                except (TypeError, ValueError, OverflowError):
                    problems.append(f'{name}: vocab={_v!r} is not an integer')
                    continue
                if _v <= 0:
                    problems.append(f'{name}: vocab={_v!r} must be positive')
                    continue
            row[_fpk] = _v
        try:
            steps = _as_int(cfg['steps'], 'steps')
            batch = _as_int(cfg['batch_size'], 'batch_size')
            seq = _as_int(cfg['seq_len'], 'seq_len')
            # `or 0` would turn False/'' into a legal zero count, hiding a
            # flag where an exact integer budget is required.
            _tok_raw = cfg.get('n_train_tokens')
            declared_tokens = _as_int(_tok_raw if _tok_raw is not None else 0,
                                      'n_train_tokens')
            # Booleans are flags, not learning rates: float(True) == 1.0
            # would silently certify a shared LR transfer.
            lr = _finite_scalar(cfg.get('lr'))
            _warm_raw = cfg.get('warmup')
            warmup = _as_int(_warm_raw if _warm_raw is not None else 0, 'warmup')
            # Optimizer identity / betas / eps / clipping / schedule shape /
            # augmentation / precision are NOT captured by lr+warmup.  A scale
            # ladder must either prove they are shared or disclose that the
            # cross-scale effect is confounded with an optimizer change.
            _kn_raw = cfg.get('train_knobs') if 'train_knobs' in cfg else tier.get('train_knobs')
            train_knobs = None
            if _kn_raw is not None:
                if not isinstance(_kn_raw, dict):
                    problems.append(f'{name}: train_knobs is not an object '
                                    f'({type(_kn_raw).__name__})')
                else:
                    train_knobs = _canon_setting(_kn_raw)
            _seeds_raw = cfg.get('seeds')
            if _seeds_raw is None:
                _seed_iter = []
            elif isinstance(_seeds_raw, (str, bytes, dict)):
                raise ValueError('seeds must be an iterable of integers')
            else:
                try:
                    _seed_iter = list(_seeds_raw)
                except TypeError:
                    raise ValueError('seeds must be an iterable of integers') from None
            seeds = [_as_int(s, 'seed') for s in _seed_iter]
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            problems.append(f'{name}: incomplete scale cfg ({type(e).__name__}: {e})')
            continue
        _neg_seeds = sorted({s for s in seeds if s < 0})
        if _neg_seeds:
            problems.append(f'{name}: seed(s) {_neg_seeds[:5]} are negative; a seed id '
                            'is a non-negative draw identity')
        _dup_seeds = sorted(s for s, c in collections.Counter(seeds).items() if c > 1)
        if _dup_seeds:
            problems.append(f'{name}: duplicate seed(s) {_dup_seeds[:5]} in the declared '
                            'plan; each seed contributes one paired draw, so duplicates '
                            'would silently shrink the realised n')
        _seed_set = frozenset(seeds)
        if min(steps, batch, seq) <= 0:
            problems.append(f'{name}: steps/batch_size/seq_len must be positive')
            continue
        if declared_tokens < 0:
            problems.append(f'{name}: n_train_tokens={declared_tokens} must be non-negative')
        if warmup < 0:
            problems.append(f'{name}: warmup={warmup} must be non-negative')
        if lr is None or not (math.isfinite(lr) and lr > 0):
            problems.append(f'{name}: lr={cfg.get("lr")!r} is not a positive finite number')
        effective_tokens = steps * batch * seq
        # Compare the integer difference to the exact 1% bound.  Multiplying a
        # large declared token budget by 0.01 converts it to float and can
        # overflow even though the integer comparison is well defined.
        if (declared_tokens > 0
                and abs(effective_tokens - declared_tokens) * 100
                > declared_tokens):
            problems.append(
                f'{name}: declared n_train_tokens={declared_tokens:,} differs from '
                f'steps*batch_size*seq_len={effective_tokens:,} by more than 1%')
        params = tier.get('params')
        try:
            params = _as_int(params, 'params', allow_none=True)
        except (TypeError, ValueError, OverflowError):
            problems.append(f'{name}: params={params!r} is not an integer')
            params = None
        if params is not None and params <= 0:
            problems.append(f'{name}: params must be positive')
            params = None
        _tokens_per_param = (_finite_int_ratio(effective_tokens, params)
                             if params else None)
        if params is not None and _tokens_per_param is None:
            warnings.append(
                f'{name}: tokens-per-parameter is not representable as a '
                'finite float; budget matching is unverifiable')
        row.update({
            'steps': steps, 'batch_size': batch, 'seq_len': seq,
            'declared_tokens': (declared_tokens or None),
            'effective_tokens': int(effective_tokens),
            'tokens_per_param': _tokens_per_param,
            'params': params, 'lr': lr, 'warmup': warmup,
            'warmup_fraction': _finite_int_ratio(warmup, steps) if steps else None,
            'n_seeds': len(_seed_set),
            'seeds': sorted(_seed_set),
            'train_knobs': train_knobs,
            'p_floor': signflip_p_floor(len(_seed_set)),
            'min_seeds_for_family_significance': (signflip_min_n_for_holm(family_m, 0.05)
                                                  if family_m is not None else None),
        })
        if params is None:
            warnings.append(f'{name}: params missing — tokens-per-parameter and budget matching are unverifiable')
        rows.append(row)
    if len(rows) >= 2:
        tpp = [r['tokens_per_param'] for r in rows if r.get('tokens_per_param')]
        budget_matched = None
        tpp_ratio = None
        if len(tpp) == len(rows):
            _lo, _hi = min(tpp), max(tpp)
            if _lo > 0:
                try:
                    tpp_ratio = float(_hi / _lo)
                except (OverflowError, ZeroDivisionError):
                    tpp_ratio = None
                if tpp_ratio is not None and not math.isfinite(tpp_ratio):
                    tpp_ratio = None
            else:
                tpp_ratio = None
            budget_matched = bool(tpp_ratio is not None
                                  and tpp_ratio <= 1.0 + tol)
            if tpp_ratio is None:
                warnings.append(
                    'tokens-per-parameter is not matched across the ladder '
                    '(the max/min ratio exceeds the representable float range); '
                    'a cross-scale effect is confounded with a different '
                    'exposure budget')
            elif not budget_matched:
                warnings.append(
                    'tokens-per-parameter is not matched across the ladder '
                    f'(max/min={tpp_ratio:.2f} > {1.0 + tol:.2f}); a cross-scale '
                    'effect is confounded with a different exposure budget')
        lrs = {r['lr'] for r in rows if r.get('lr') is not None}
        lr_constant = (len(lrs) == 1) if lrs else None
        _transfer = str(policy.get('lr_transfer') or '')
        if lr_constant and _transfer in ('', 'constant_declared_not_validated', 'constant_lr_not_validated'):
            warnings.append(
                'learning rate is held constant across widths without a validated '
                'transfer rule (policy=%r); scale contrasts carry an optimisation-'
                'transfer confound, not a matched optimizer control' % _transfer)
        elif lr_constant is False:
            # Non-constant LR is legitimate only as a DECLARED per-tier
            # transfer; it is never an optimizer-matched control, so it must
            # stay visible regardless of the policy label.
            warnings.append(
                'learning rate DIFFERS across tiers (%s; policy=%r) — the '
                'cross-scale contrast is confounded with an LR-transfer effect, '
                'not a matched optimizer control'
                % (sorted(lrs), policy.get('lr_transfer')))
        # Optimizer / betas / eps / clipping / schedule shape / augmentation /
        # precision: compare the per-tier descriptor when declared.  A
        # mismatch invalidates the "same optimizer" premise; a partial or
        # absent descriptor is unverifiable and never a silent pass.
        _kn_rows = [r for r in rows if r.get('train_knobs') is not None]
        if not _kn_rows:
            train_knobs_binding = 'unverifiable'
            warnings.append(
                'no tier carries a train_knobs descriptor: optimizer / betas / '
                'eps / grad-clip / schedule shape / augmentation / precision '
                'cannot be shown equal across the ladder')
        elif len(_kn_rows) != len(rows):
            train_knobs_binding = 'partial'
            warnings.append(
                'only %d/%d tiers carry a train_knobs descriptor; the shared '
                'optimizer/augmentation recipe is unverifiable for the rest'
                % (len(_kn_rows), len(rows)))
        else:
            _kn_vals = {r['train_knobs'] for r in _kn_rows}
            if len(_kn_vals) > 1:
                train_knobs_binding = 'mixed'
                problems.append(
                    'training knobs differ across tiers (optimizer / betas / '
                    'eps / grad-clip / schedule shape / augmentation / '
                    'precision) — the cross-scale effect is confounded with a '
                    'different training recipe')
            else:
                train_knobs_binding = 'shared'
        warm = [r['warmup_fraction'] for r in rows if r.get('warmup_fraction') is not None]
        warm_ratio = (_finite_float_ratio(max(warm), min(warm))
                      if len(warm) == len(rows) and min(warm) > 0 else None)
        if (warm_ratio is None and len(warm) == len(rows) and min(warm) > 0):
            warnings.append(
                'warmup fractions differ by more than the representable float '
                'range across scales; the schedule is not transfer-matched')
        elif warm_ratio is not None and warm_ratio > 2.0:
            warnings.append(f'warmup fraction varies {warm_ratio:.2f}x across scales')
        _family_alpha = 0.05
        _family_n = (signflip_min_n_for_holm(family_m, _family_alpha)
                     if family_m is not None else None)
        for r in rows:
            if r.get('n_seeds') and r.get('p_floor') is not None:
                # Power is a property of the DECLARED family, not of the raw
                # sign-flip test alone.  A tier with n=6 can reach the raw
                # floor 0.031 but not a Holm floor of 0.094 for family_m=3;
                # reporting descriptive_only=False there would promise a
                # family-level claim the design cannot support.
                _fam_floor = (holm_min_attainable_p(r['n_seeds'], family_m,
                                                    _family_alpha)
                              if family_m is not None else None)
                r['p_floor_family_holm'] = _fam_floor
                _basis_floor = (_fam_floor if _fam_floor is not None
                                and math.isfinite(_fam_floor) else r['p_floor'])
                r['descriptive_only'] = bool(_basis_floor > _family_alpha)
                r['descriptive_only_basis'] = (
                    'holm_family' if (_fam_floor is not None
                                      and math.isfinite(_fam_floor))
                    else 'raw_signflip')
                r['seeds_sufficient_for_family'] = (
                    None if _family_n is None else bool(r['n_seeds'] >= _family_n))
        # Corpus/tokenizer binding across the ladder: a MISMATCH is a hard
        # problem (the cross-tier log-ratios the ladder exists to compute are
        # not defined), while missing fingerprints stay a warning (the audit
        # cannot attest, but old panels predate the field).
        binding = {}
        for _fpk, _what in (('corpus_fp', 'corpus fingerprint'),
                            ('fit_fp', 'tokenizer-fit fingerprint'),
                            ('vocab', 'vocab size')):
            _vals = [r.get(_fpk) for r in rows if r.get(_fpk) is not None]
            if len(_vals) >= 2 and len(set(map(str, _vals))) > 1:
                problems.append(
                    f'tiers do not share one {_what}: '
                    + ', '.join(f'{r["name"]}={r[_fpk]!r}' for r in rows if r.get(_fpk) is not None)
                    + ' — cross-tier PPL/log-ratio contrasts are not defined across '
                      'different tokenisations/corpora; run each tier on the same '
                      'corpus/tokenizer or report the tiers separately')
                binding[_fpk] = 'mixed'
            elif len(_vals) == len(rows) and len(rows) >= 2:
                binding[_fpk] = 'shared'
            elif len(_vals) >= 1:
                binding[_fpk] = 'partial'
                warnings.append(f'only {len(_vals)}/{len(rows)} tiers carry a {_what}; '
                                'the cross-tier corpus binding is unverifiable for the rest')
            else:
                binding[_fpk] = 'unverifiable'
        # Cross-tier paired contrasts need the SAME seed identities, not just
        # the same count.  Ragged completion is allowed, but the shared set is
        # the only paired evidence; a tier with extra seeds cannot enter the
        # paired test for those seeds.
        _declared_sets = [set(r.get('seeds') or []) for r in rows]
        _sets_present = [s for s in _declared_sets if s]
        if not _sets_present:
            seed_set_shared = None
            _seed_common = []
            warnings.append('no tier declares seeds — the cross-tier paired '
                            'contrast and its family-resolution floor are unverifiable')
        elif len(_sets_present) != len(rows):
            seed_set_shared = False
            # A tier with no declared seed plan cannot contribute a paired
            # draw.  Intersecting only the tiers that did declare seeds would
            # fabricate a common set (e.g. S2 empty + S1 0..7 -> "8 common
            # seeds") and could certify an unpaired ladder as sufficient.
            _seed_common = []
            warnings.append('only %d/%d tiers declare seeds; cross-tier pairing is '
                            'unverifiable for the rest' % (len(_sets_present), len(rows)))
        else:
            _seed_common = sorted(set.intersection(*_declared_sets))
            seed_set_shared = len({frozenset(s) for s in _declared_sets}) == 1
            if not seed_set_shared:
                warnings.append(
                    'tiers do not share one seed set (common seeds: %s; per-tier n: %s) — '
                    'cross-tier paired contrasts can use only the common seeds, and extra '
                    'seeds are not replication of the other tiers'
                    % (_seed_common or 'none',
                       ', '.join('%s=%d' % (r['name'], r['n_seeds']) for r in rows)))
        # A cross-tier paired contrast uses ONLY the seed-set intersection.
        # Per-tier n therefore overstates the paired evidence when one tier
        # added seeds the other never ran.  Make the paired design's own
        # resolution floor explicit instead of borrowing a tier's raw n.
        _n_common = len(_seed_common)
        _common_floor = (signflip_p_floor(_n_common) if _n_common > 0 else float('nan'))
        _common_fam_floor = (holm_min_attainable_p(_n_common, family_m,
                                                   _family_alpha)
                             if (_n_common > 0 and family_m is not None)
                             else None)
        _common_sufficient = (None if _family_n is None
                              else bool(_n_common >= _family_n))
        if _common_sufficient is False:
            warnings.append(
                'the cross-tier paired contrast has only %d common seed(s), '
                'but the declared %d-comparison Holm family needs n>=%d for '
                'its exact sign-flip floor to reach alpha=%g; per-tier extra '
                'seeds cannot repair this — repeat the missing seeds in every '
                'tier or shrink the family'
                % (_n_common, int(family_m), int(_family_n), _family_alpha))
        out = {'status': ('fail' if problems else ('warn' if warnings else 'ok')),
               'budget_matched': (budget_matched if len(rows) >= 2 else None),
               'seed_set_shared': seed_set_shared,
               'seed_set_intersection': _seed_common,
               'n_common_seeds': _n_common,
               'family_m': family_m,
               'family_alpha': _family_alpha,
               'family_min_seeds': _family_n,
               'common_seed_paired_p_floor': _common_floor,
               'common_seed_min_attainable_p_holm': _common_fam_floor,
               'common_seeds_sufficient_for_family': _common_sufficient,
               'tokens_per_param_ratio': tpp_ratio,
               'tokens_per_param_tolerance': tol,
               'lr_constant': (lr_constant if len(rows) >= 2 else None),
               'lr_values': sorted(lrs) if len(rows) >= 2 else None,
               'lr_transfer_policy': policy.get('lr_transfer'),
               'train_knobs_binding': train_knobs_binding,
               'corpus_binding': binding,
               'tiers': rows, 'policy': policy,
               'problems': problems, 'warnings': warnings}
    else:
        warnings.append('fewer than two complete tiers — no cross-scale comparison can be checked')
        out = {'status': ('fail' if problems else 'warn'), 'budget_matched': None,
               'corpus_binding': {},
               'tiers': rows, 'policy': policy, 'problems': problems,
               'warnings': warnings}
    return out


def loglog_regression(points, *, alpha=0.05):
    """log10(y) ~ log10(x) 回归，带斜率 CI / p / R² / 留一稳定性。

    用于规模律读数（如 crossover tokens ~ d_model）。`points` 接受
    [(x, y)] 或 [{label, x, y}]；只保留有限正数。点数 k>=3 时给出斜率的
    t 区间与双侧 p；k=2 时直线是恒等拟合，只报斜率并显式标注「无残差
    自由度、无 R²」。x 本身的测量误差未计入（观测性拟合，不是因果估计）。
    """
    alpha = _validated_alpha(alpha, context='loglog_regression alpha')
    if isinstance(points, (str, bytes, bytearray, dict)):
        raise ValueError('loglog_regression points must be an iterable of '
                         f'point records, not a {type(points).__name__}')
    rows = []
    for i, p in enumerate(points if points is not None else []):
        if isinstance(p, dict):
            x, y, label = p.get('x'), p.get('y'), p.get('label', i)
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            x, y, label = p[0], p[1], i
        else:
            continue
        if isinstance(x, (bool, np.bool_, str, bytes, bytearray, dict,
                          complex, np.complexfloating)) or isinstance(
                y, (bool, np.bool_, str, bytes, bytearray, dict,
                    complex, np.complexfloating)):
            # Numeric strings are labels, not measurements (same contract as
            # _finite_scalar); a mapping/string point is malformed, not a
            # zero-effect observation.
            continue
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0:
            rows.append((str(label), fx, fy))
    out = {'alpha': float(alpha), 'n_points': len(rows), 'labels': [r[0] for r in rows],
           'note': 'observational log-log fit; x measurement error is not modelled'}
    if len(rows) < 2:
        out.update({'status': 'insufficient_points'})
        return out
    x = np.log10(np.asarray([r[1] for r in rows], dtype=float))
    y = np.log10(np.asarray([r[2] for r in rows], dtype=float))
    xbar, ybar = float(x.mean()), float(y.mean())
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0.0:
        out.update({'status': 'degenerate_x'})
        return out
    slope = float(((x - xbar) * (y - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    pred = slope * x + intercept
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    try:
        _coef = 10 ** intercept
    except OverflowError:
        _coef = float('inf')
    out.update({'status': 'ok', 'slope': slope, 'exponent': slope, 'intercept': intercept,
                'law': 'y = %.6g * x^%.6g' % (_coef, slope)})
    k = len(rows)
    if k >= 3:
        sigma2 = ss_res / (k - 2)
        se = float(math.sqrt(sigma2 / sxx))
        tc = _t_critical(1.0 - float(alpha), k - 2)
        # A zero-residual fit has an infinite t statistic, not an undefined
        # p-value: report p=0 for a non-zero slope (p=1 for an exactly flat
        # line).  ``nan`` would hide a perfectly recovered power law.
        if se > 0.0:
            _p = student_t_two_sided_p(slope / se, k - 2)
        else:
            _p = 1.0 if abs(slope) <= 1e-15 else 0.0
        out.update({'se': se, 'ci': [slope - tc * se, slope + tc * se],
                    'p': _p,
                    'df': k - 2, 'r2': (1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
                    'residual_sigma_log10': float(math.sqrt(sigma2))})
        loo = []
        for i in range(k):
            xx = np.delete(x, i)
            yy = np.delete(y, i)
            xb, yb = float(xx.mean()), float(yy.mean())
            den = float(((xx - xb) ** 2).sum())
            if den <= 0.0:
                # Duplicate x values can leave no scale spread after dropping
                # one point; report the undefined fit instead of raising in a
                # report builder.
                loo.append({'label': rows[i][0], 'slope': None, 'note': 'degenerate x-spread after dropping this point'})
                continue
            s = float(((xx - xb) * (yy - yb)).sum() / den)
            loo.append({'label': rows[i][0], 'slope': s})
        out['leave_one_out'] = loo
        _loo_slopes = [v['slope'] for v in loo if v['slope'] is not None]
        out['leave_one_out_slope_range'] = [float(min(_loo_slopes)), float(max(_loo_slopes))] if _loo_slopes else None
    else:
        out.update({'r2': None, 'ci': None, 'p': None,
                    'note': out['note'] + '; k=2 gives an exact line with no residual degrees of freedom, so no CI/R2/p is reported'})
    return out


def crossover_scaling_fit(entries, *, min_points=4, volume_tol=0.25,
                          d_spread_min=1.2):
    """Fit log10(crossover tokens) ~ log10(d_model) only when the points can
    support it.  This replaces fitting a line
    through whatever panels happen to exist.

    `entries` is an iterable of dicts (or a dict label -> dict) with keys:
      d                 model width (required)
      crossover_tokens  observed crossover position in tokens, or None
      censored          True when no crossover was observed in the window
      train_tokens      total training tokens of that panel (required)
      reconstructed     True for a synthesized trajectory
      label             display name (defaults to the dict key/index)

    Refusal rules (each returns `status` + `reasons`, never a slope):
      * reconstructed trajectories are excluded outright;
      * any censored point refuses the fit (dropping it estimates the slope on
        a selected, faster-crossing subset -- informative censoring);
      * fewer than `min_points` distinct widths after de-duplication;
      * widths span less than `d_spread_min` (a degenerate x axis);
      * training volume is not comparable: max/min > 1 + volume_tol.  Scale
        panels with 4k vs 8k steps are not points on one d-axis;
      * duplicate widths: keep the panel with the most training tokens (the
        token-matched arm) and report the dropped duplicates.

    When it does fit, the slope/CI/p/R^2/leave-one-out all come from
    `loglog_regression`; the claim level is fixed to `descriptive` because the
    scale panels are observational, not randomized.
    """
    # Gate parameters: a bool/fractional point-count or tolerance would
    # silently change which fits are allowed.  Refuse instead of coercing.
    _min_pts = _as_seed_int(min_points)
    if _min_pts is None or _min_pts < 2:
        raise ValueError(f'min_points must be an integer >= 2, got {min_points!r}')
    min_points = _min_pts
    for _name, _value, _check in (
            ('volume_tol', volume_tol, lambda v: math.isfinite(v) and v >= 0.0),
            ('d_spread_min', d_spread_min, lambda v: math.isfinite(v) and v > 1.0)):
        if isinstance(_value, (bool, np.bool_)):
            raise ValueError(f'{_name} must be a real number, got {_value!r}')
        try:
            _fv = float(_value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{_name} must be a real number, got {_value!r}') from None
        if not _check(_fv):
            raise ValueError(f'{_name} has an invalid value {_value!r}')
    volume_tol = float(volume_tol)
    d_spread_min = float(d_spread_min)
    rows = []
    if isinstance(entries, (str, bytes, bytearray)):
        raise ValueError('crossover_scaling_fit entries must be an iterable '
                         'of panel records, not a string')
    if isinstance(entries, dict):
        entries = [(k, v) for k, v in entries.items()]
    else:
        entries = list(entries) if entries is not None else []
    for i, item in enumerate(entries):
        if isinstance(item, dict):
            label, e = item.get('label', str(i)), item
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
            label, e = str(item[0]), item[1]
        else:
            continue
        if not isinstance(e, dict):
            continue
        if e.get('reconstructed') or e.get('synthesized'):
            continue
        dur = e.get('d')
        tok = e.get('crossover_tokens')
        tr = e.get('train_tokens')
        if any(isinstance(v, (bool, np.bool_, str, bytes, bytearray, dict,
                              complex, np.complexfloating))
               for v in (dur, tok, tr)):
            # Numeric strings are not measurements; a mapping/string field is
            # malformed, not a positive width or token budget.
            continue
        try:
            fd = float(dur)
            ftok = float(tok) if tok is not None else None
            ftr = float(tr) if tr is not None else None
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(fd) and fd > 0.0):
            continue
        rows.append({'label': str(label), 'd': fd,
                     'crossover_tokens': (ftok if ftok is not None and math.isfinite(ftok) and ftok > 0.0 else None),
                     'train_tokens': (ftr if ftr is not None and math.isfinite(ftr) and ftr > 0.0 else None),
                     'censored': bool(e.get('censored') or e.get('mean_crossed') is False)})
    out = {'status': 'insufficient_points', 'n_input': len(rows),
           'claim_level': 'descriptive', 'min_points': int(min_points),
           'volume_tolerance': float(volume_tol), 'points': [], 'excluded': [],
           'note': 'observational scale fit; the panels are not randomized, so a slope is a descriptive trend, not a scaling law'}
    if not rows:
        out['reasons'] = ['no usable crossover panels']
        return out
    censored = [r['label'] for r in rows if r['censored']]
    missing = [r['label'] for r in rows if r['crossover_tokens'] is None or r['train_tokens'] is None]
    usable = [r for r in rows if not r['censored'] and r['crossover_tokens'] is not None and r['train_tokens'] is not None]
    out['censored_panels'] = censored
    out['incomplete_panels'] = missing
    if censored:
        out.update({'status': 'censored',
                    'reasons': [f'{len(censored)} panel(s) never crossed inside their training window ({censored}); dropping them would fit the slope on the faster-crossing subset only. Report the censored positions as lower bounds or run longer, then refit.']})
        return out
    if missing:
        out.update({'status': 'incomplete',
                    'reasons': [f'panel(s) lack a crossover position or a training-token count ({missing})']})
        return out
    # One point per width: the token-matched arm wins.
    by_d = {}
    for r in usable:
        by_d.setdefault(r['d'], []).append(r)
    dedup = {}
    dropped = []
    for d, rs in by_d.items():
        rs = sorted(rs, key=lambda r: (-r['train_tokens'], r['label']))
        dedup[d] = rs[0]
        dropped.extend(x['label'] for x in rs[1:])
    if dropped:
        out['deduped_by_d'] = sorted(dropped)
    pts = [dedup[d] for d in sorted(dedup)]
    out['points'] = [{k: p[k] for k in ('label', 'd', 'crossover_tokens', 'train_tokens')} for p in pts]
    out['excluded'] = sorted(set(censored) | set(missing) | set(dropped))
    if len(pts) < int(min_points):
        out.update({'status': 'insufficient_points',
                    'reasons': [f'{len(pts)} distinct width(s) after de-duplication; at least {int(min_points)} are required for a log-log fit (n=2 is a tautology, n=3 has 1 residual df)']})
        return out
    dmin, dmax = min(p['d'] for p in pts), max(p['d'] for p in pts)
    if dmax / dmin < float(d_spread_min):
        out.update({'status': 'degenerate_x',
                    'reasons': [f'width span {dmin:g}..{dmax:g} is only x{dmax / dmin:.2f}; a slope over less than x{float(d_spread_min):.2f} is not identified']})
        return out
    tmin = min(p['train_tokens'] for p in pts)
    tmax = max(p['train_tokens'] for p in pts)
    volume_ratio = _finite_float_ratio(tmax, tmin)
    # Compare with the ratio, not `tmin * (1 + tol)`: the product can overflow
    # for a finite-but-large tmin and turn a real mismatch into "not greater".
    if volume_ratio is None or volume_ratio > 1.0 + float(volume_tol):
        _ratio_txt = (f'x{volume_ratio:.2f}' if volume_ratio is not None
                      else 'a ratio beyond the representable float range')
        out.update({'status': 'training_volume_confounded',
                    'reasons': [f'training volume ranges {tmin:.3g}..{tmax:.3g} tokens ({_ratio_txt}); the d-axis is confounded with training length. '
                                'Add token-matched arms or model log(tokens) as a covariate before reporting a d slope.'],
                    'volume_ratio': volume_ratio})
        return out
    # A crossover cannot happen after the training run ended.  A panel whose
    # crossover position exceeds its own training budget is internally
    # inconsistent (mis-copied artifact, wrong token unit, or a censored point
    # mislabelled as observed); fitting it would fabricate a scaling slope.
    impossible = [p['label'] for p in pts
                  if p['crossover_tokens'] > p['train_tokens']]
    if impossible:
        out.update({
            'status': 'inconsistent_points',
            'inconsistent_panels': impossible,
            'reasons': [f'panel(s) report a crossover at more tokens than they '
                        f'trained on ({impossible}); an observed crossover must '
                        'lie within [1, train_tokens] — correct the artifact or '
                        'mark the panel censored']})
        return out
    fit = loglog_regression([{'x': p['d'], 'y': p['crossover_tokens'], 'label': p['label']} for p in pts])
    out.update(fit)
    out['status'] = fit.get('status', 'ok')
    out['volume_ratio'] = volume_ratio
    out['slope_per_doubling'] = (float(fit['slope']) * math.log10(2.0)) if isinstance(fit.get('slope'), (int, float)) else None
    out['claim_level'] = 'descriptive'
    out['note'] = ('observational log-log fit over uncensored, token-comparable panels; '
                   'report the CI and the leave-one-out range, and do not call it a scaling law')
    return out


# `mt` (matched-variant set) and `mr` (mlp_match_ref) can themselves contain
# underscores (e.g. `full_rope`, `csa_dynamic`), so those two values are
# captured lazily up to the next literal key rather than by `[^_]+`.
_RUN_CFG_RE = re.compile('^steps(?P<steps>[^_]+)_sl(?P<sl>[^_]+)_bs(?P<bs>[^_]+)_nt(?P<nt>[^_]+)_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)_d(?P<d>[^_]+)_L(?P<L>[^_]+)_H(?P<H>[^_]+)_Dh(?P<Dh>[^_]+)_v(?P<v>[^_]+)_wu(?P<wu>[^_]+)_cl(?P<cl>[^_]+)_dlm(?P<dlm>[^_]+)_mt(?P<mt>.*?)_mr(?P<mr>.*?)_ee(?P<ee>[^_]+)_es(?P<es>[^_]+)_det(?P<det>[^_]+)_df(?P<df>.+?)(?:_tk(?P<tk>[0-9a-f]+))?(?:_amp(?P<amp>[^_]+))?(?:_eb(?P<eb>[01]))?(?:_ae(?P<ae>[^_]+))?_cs(?P<cs>.+)$')

def run_cfg_knobs(fp):
    """Best-effort decode of a `_run_fp` string into its named training knobs.

    Used only for the audit display: it lets the report SHOW that two arms
    share optimizer / schedule / training length / data, instead of asserting it.
    `{}` means the string predates the current layout or is malformed.  The
    authoritative controlled-variable gate is still whole-string equality in
    `pair_reason` — this decoder never substitutes for it."""
    if not isinstance(fp, str):
        return {}
    m = _RUN_CFG_RE.match(fp)
    return m.groupdict() if m else {}

def variant_presence(outdir):
    out = {}
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return out
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — variant presence cannot be established')
        return out
    if not isinstance(raw, dict):
        # A valid JSON list/string is not a panel summary.  Calling .items()
        # on it used to crash the whole report instead of excluding one panel.
        print(f'[stats] WARNING: {sp} is a {type(raw).__name__}, not an object — variant presence cannot be established')
        return out
    kept = {}
    refused = {}
    for _k, r in raw.items():
        if not isinstance(r, dict):
            continue
        v = r.get('variant')
        if v is None:
            v = _k.split('::')[0]
        if not isinstance(v, str) or not v:
            # An unhashable/non-string arm id cannot be a presence key; this
            # is the same malformed-record contract as read_panel/report
            # pairing.  Skip it instead of crashing inside a dict lookup.
            print(f'[stats] WARNING: {sp}: record {_k!r} carries a '
                  f'non-string/empty arm id ({v!r}) — ignored for presence')
            continue
        if ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and (_as_seed_int(r.get('seed')) is not None):
            kept[v] = kept.get(v, 0) + 1
        else:
            refused[v] = refused.get(v, 0) + 1
    for v in sorted(set(kept) | set(refused)):
        if v not in refused:
            continue
        if v in kept:
            out[v] = f'present with {kept[v]} measurable record(s) (+{refused[v]} refused) — no shared usable seed'
        else:
            out[v] = f'present but ALL {refused[v]} record(s) refused as unmeasured (synthesized / error / nan ppl / no seed)'
    return out

def format_variant_presence(presence, variants):
    parts = [f'`{v}`: {presence[v]}' for v in variants if v in presence]
    if not parts:
        return None
    return '; '.join(parts) + ' — omitted rather than quoting a cross-panel delta'

def ppl_by_seed(outdir, full=False):
    _by_id = read_panel(outdir)
    out = {}
    if not _by_id:
        return out
    sp = os.path.join(outdir, 'summary.json')
    _seen_seed, _collide = ({}, {})
    for _v, _t, _s in _by_id:
        if (_v, _s) in _seen_seed and _seen_seed[_v, _s] != _t:
            _collide.setdefault(_v, set()).update({_seen_seed[_v, _s], _t})
        _seen_seed[_v, _s] = _t
    if _collide:
        for _v, _tags in sorted(_collide.items()):
            print(f'[stats] WARNING: {sp}: variant `{_v}` holds SEVERAL protocol conditions for one seed (tags {sorted(_tags)}). This reader keeps ONE record per (variant, seed) — the flat map CANNOT represent the others.  A paired contrast built from it would silently quote a single condition; use `ppl_by_seed_grouped` to read the panel per condition.')
    for (_v, _t, _s), (_k, r) in _by_id.items():
        _slot = out.setdefault(_v, {})
        if _s not in _slot:
            if full:
                _slot[_s] = r if 'protocol_tag' in r else {**r, 'protocol_tag': _t}
            else:
                _slot[_s] = r['ppl']
    return out

def _record_identity(rec, key, legacy_ok):
    seed = _as_seed_int(rec.get('seed'))
    if seed is None:
        return None
    var = rec.get('variant')
    if var is None:
        var = str(key).split('::', 1)[0]
    return (str(var), record_tag(rec, key, legacy_ok), seed)

def _as_seed_int(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        # np.float32/np.float64 are Real but not all are Python ``float``;
        # int() would silently truncate 1.5 to 1, turning an invalid seed
        # into a different valid one.
        f = float(v)
        return int(f) if math.isfinite(f) and f.is_integer() else None
    try:
        i = int(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(v, (str, bytes)):
        # Numeric strings such as "3" are accepted by the legacy readers;
        # int("3.0") already fails above, so no truncation can slip through.
        return i
    # Objects outside numbers.Real (notably Decimal and 0-d NumPy arrays)
    # can still define __int__ and would be silently truncated by int(v).
    # Check the numeric value itself before accepting the integer.
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return i
    if not math.isfinite(f) or f != float(i):
        return None
    return i

def _seed_value(v, *, where='seed'):
    """Validate one seed against the domain every RNG backend can accept.

    ``_as_seed_int`` is the generic integer reader used for counts and
    parameter totals (which can legitimately exceed 2**32).  Seeds are
    different: NumPy's global seeder rejects values outside [0, MAX_SEED], so
    allowing one here would only fail later (or, worse, be remapped by one
    backend but not another).  Booleans and non-integral floats are rejected
    for the same reason as in ``normalize_seed_list``.
    """
    s = _as_seed_int(v)
    if s is None:
        raise ValueError(f'{where} must be an integer, got {v!r}')
    if s < 0 or s > MAX_SEED:
        raise ValueError(
            f'{where} must be in [0, {MAX_SEED}] (the NumPy/PyTorch/DataLoader '
            f'common seed domain), got {s}')
    return s


def _other_seed(seed):
    """A deterministic seed that differs from ``seed`` without leaving range."""
    return (int(seed) + 1) % (MAX_SEED + 1)


def _exact_count(v, *, minimum=0):
    """Exact non-negative integer count (token/document counts, not seeds).

    Fractional/boolean values must not be truncated into a different pooled
    denominator, and a negative count is not a count at all.
    """
    i = _as_seed_int(v)
    if i is None or i < int(minimum):
        return None
    return i

def read_panel(outdir):
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return {}
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — this panel is EXCLUDED')
        return {}
    if not isinstance(raw, dict):
        # Parsing is not enough: the summary contract is a key -> record map.
        # A list/string would otherwise fail later inside .items() and take
        # down every consumer (ppl_by_seed, aggregate, report building).
        print(f'[stats] WARNING: {sp} is a {type(raw).__name__}, not an object — this panel is EXCLUDED')
        return {}
    _legacy_ok = legacy_warm_tags(raw)
    _by_id = {}
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('seed') is None:
            continue
        if not ppl_is_usable(r.get('ppl')) or r.get('synthesized'):
            continue
        _id = _record_identity(r, _k, _legacy_ok)
        if _id is None:
            continue
        if _id in _by_id:
            print(f'[stats] WARNING: {sp}: two measurable records for variant={_id[0]!r} protocol={_id[1]!r} seed={_id[2]} (keys {_by_id[_id][0]!r}, {_k!r}) — the PPL is AMBIGUOUS and no averaging can resolve it; the earlier key is used, the later one is EXCLUDED')
            continue
        _by_id[_id] = (_k, r)
    return _by_id

def ppl_by_seed_grouped(outdir):
    grouped = {}
    for (_v, _t, _s), (_k, _r) in read_panel(outdir).items():
        grouped.setdefault((_v, _t), {})[_s] = _r
    return grouped

# ---- 对比实验 / 消融 / 统计可靠性的声明式审计（零 GPU）----
# 每个「创新点」按留一法（leave-one-out）写出「带模块」与「去模块」的对照
# 臂。审计对每条对照：(1) 用 `pair_reason` 验证两臂严格同 run_cfg——优化器、
# 学习率调度、权重衰减、训练步数/ token、warmup、comp 正则、模型形状、数据
# 指纹与确定性档全部一致，只有层架构（variant）不同；(2) 逐种子配对给出
# mean±std、d_z 与精确符号翻转 p；(3) 用 |mean| <= std 标出「差异落在种子噪声内」、
# 用 n<3 标出欠功效；(4) 汇总每个创新点是否至少有一条完整留一对照。
# 符号约定：Δ = 「带模块」−「去模块」，PPL 越低越好，故 Δ<0 = 模块有用。
ABLATION_DESIGN = {
    'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'dynamic vs fixed blocking (CSA)'),
        ('hybrid_dynamic', 'hybrid_fixed', 'dynamic vs fixed blocking (CSA+HCA hybrid); the dynamic HCA layers train their boundary gates with comp_lambda_mult=4 (comp_lambda_effective 0.05->0.2) — part of the declared dynamic-HCA recipe, kept visible in train_objective and by the single-factor design gate (design_notes: single-factor contract)'),
        ('hybrid_csa_dyn', 'hybrid_fixed', 'dynamic vs fixed CSA blocking inside the hybrid (HCA fixed)'),
        ('hybrid_hca_dyn', 'hybrid_fixed', 'dynamic vs fixed HCA blocking inside the hybrid (CSA fixed); the dynamic HCA layers train their boundary gates with comp_lambda_mult=4 (comp_lambda_effective 0.05->0.2) — part of the declared dynamic-HCA recipe, kept visible in train_objective and by the single-factor design gate (design_notes: single-factor contract)'),
    ],
    'learned_selection': [
        ('csa_fixed', 'csa_fix_randidx', 'learned top-k vs random selection (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_randidx', 'learned top-k vs random selection (dynamic blocking)'),
    ],
    'content_pooling': [
        ('csa_fixed', 'csa_fix_zerocont', 'pooled block content vs zero content (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_zerocont', 'pooled block content vs zero content (dynamic blocking)'),
    ],
    # Presence is not alignment.  Zeroing the content changes BOTH whether the
    # block carries any content and whether that content is aligned with the
    # block's own tokens.  The shift control keeps the same non-zero content
    # table, shape and norm but moves each block's content back one block, so
    # the pair isolates *alignment* from mere presence/magnitude.
    'content_alignment': [
        ('csa_fixed', 'csa_fix_shiftcont', 'block-aligned vs previous-block pooled content (fixed blocking); norm-matched misalignment control'),
        ('csa_dynamic', 'csa_dyn_shiftcont', 'block-aligned vs previous-block pooled content (dynamic blocking); norm-matched misalignment control'),
    ],
    'attention_sink': [
        ('csa_fixed', 'csa_fix_nosink', 'sink on vs off (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_nosink', 'sink on vs off (dynamic blocking)'),
        ('full_sink', 'full', 'sink on vs off (dense control)'),
    ],
    # On/off changes BOTH whether there is a trained sink and whether the extra
    # sink parameter exists.  The capacity-matched placebo keeps the same
    # parameter shape/count/read budget but freezes a fixed, non-learned sink
    # vector, so `treatment - placebo` isolates the *learned* component while
    # `treatment - absent` remains the total (mechanism + capacity) effect.
    'attention_sink_learning': [
        ('csa_fixed', 'csa_fix_randomsink', 'learned sink vs frozen random sink (fixed blocking); parameter-matched placebo'),
        ('csa_dynamic', 'csa_dyn_randomsink', 'learned sink vs frozen random sink (dynamic blocking); parameter-matched placebo'),
        ('full_sink', 'full_randomsink', 'learned sink vs frozen random sink (dense control); parameter-matched placebo'),
    ],
    'compression_regularizer': [
        ('csa_dynamic', 'csa_dyn_nocomp', 'compression regularizer on vs off (dynamic blocking; the loss-term ablation; note this also freezes delta_logit because comp_reg was its only gradient source)'),
    ],
    'task_learned_boundary': [
        ('csa_dynamic_task', 'csa_dynamic', 'sampled cut policy trained by the LM loss vs hard threshold + comp_reg only'),
    ],
    'local_fusion': [
        ('csa_dyn_fuse', 'csa_dynamic', 'learned local fusion on vs off (CSA)'),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn', 'learned local fusion on vs off (hybrid)'),
    ],
    # On/off changes BOTH whether the depth-wise fusion convolution is trained
    # AND whether its parameters exist.  The capacity-matched placebo keeps the
    # same conv shape/count, freezes a non-learned kernel, and is constructed
    # without consuming the global RNG, so `treatment - placebo` isolates the
    # *learned* fusion component.  `treatment - absent` remains the total
    # (mechanism + capacity) effect.
    'local_fusion_learning': [
        ('csa_dyn_fuse', 'csa_dyn_randfuse', 'learned fusion vs frozen random fusion (CSA); parameter-matched placebo'),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn_randfuse', 'learned fusion vs frozen random fusion (hybrid); parameter-matched placebo'),
    ],
    'hybrid_hierarchy': [
        ('hybrid_csa_dyn', 'csa_dynamic', 'CSA+HCA hybrid vs CSA-only (dynamic blocking)'),
    ],
}

# ---- 2x2 factorial interaction design (comparative / ablation science) ----
# A leave-one-out family answers "does the module help on its own".  It cannot
# answer whether two modules overlap: a second module can help alone while
# adding little once the first is already present.  These 2x2 factorials use
# arms that RUN_ABL already schedules, so the interaction is measured on the
# same seeds/data/recipe without a new training run.  Cell keys are
# (factor1 level, factor2 level), both oriented control -> active:
#
#   interaction = (log PPL_11 - log PPL_10)
#               - (log PPL_01 - log PPL_00)
#
# i.e. the effect of factor2 when factor1 is active minus its effect when
# factor1 is at the control level.  Both simple effects are expected to be
# negative (the active level lowers PPL), so a POSITIVE interaction means the
# two interventions are sub-additive / redundant and a NEGATIVE one means
# synergy.  The log scale makes the contrast a ratio of ratios and prevents a
# higher baseline PPL from mechanically producing a larger absolute effect.
#
# Interactions are exploratory: they get their own declared Holm family and
# never enter the frozen confirmatory or the leave-one-out correction family.
_FACTORIAL_CELL_KEYS = (('control', 'control'), ('active', 'control'),
                        ('control', 'active'), ('active', 'active'))
FACTORIAL_DESIGN = {
    'dynamic_blocking_x_learned_selection': {
        'factors': ('dynamic_blocking', 'learned_selection'),
        'levels': {'dynamic_blocking': ('fixed', 'dynamic'),
                   'learned_selection': ('random', 'learned')},
        'cells': {
            ('control', 'control'): 'csa_fix_randidx',
            ('active', 'control'): 'csa_dyn_randidx',
            ('control', 'active'): 'csa_fixed',
            ('active', 'active'): 'csa_dynamic',
        },
        'hypothesis': 'dynamic_blocking_x_learned_selection',
        'note': ('Does learned selection still help once adaptive boundaries are '
                 'present, or do both mechanisms chase the same block-relevance '
                 'signal?'),
    },
    'dynamic_blocking_x_content_pooling': {
        'factors': ('dynamic_blocking', 'content_pooling'),
        'levels': {'dynamic_blocking': ('fixed', 'dynamic'),
                   'content_pooling': ('zero_content', 'pooled_content')},
        'cells': {
            ('control', 'control'): 'csa_fix_zerocont',
            ('active', 'control'): 'csa_dyn_zerocont',
            ('control', 'active'): 'csa_fixed',
            ('active', 'active'): 'csa_dynamic',
        },
        'hypothesis': 'dynamic_blocking_x_content_pooling',
        'note': ('Adaptive boundaries produce more homogeneous blocks; does '
                 'pooled block content still add as much as it does under fixed '
                 'blocks?'),
    },
    'dynamic_blocking_x_content_alignment': {
        'factors': ('dynamic_blocking', 'content_alignment'),
        'levels': {'dynamic_blocking': ('fixed', 'dynamic'),
                   'content_alignment': ('misaligned', 'aligned')},
        'cells': {
            ('control', 'control'): 'csa_fix_shiftcont',
            ('active', 'control'): 'csa_dyn_shiftcont',
            ('control', 'active'): 'csa_fixed',
            ('active', 'active'): 'csa_dynamic',
        },
        'hypothesis': 'dynamic_blocking_x_content_alignment',
        'note': ('Does block-content alignment still matter once boundaries are '
                 'adaptive?  Both levels use the same shifted/aligned content '
                 'tables, so the interaction is measured against the matched '
                 'placebo rather than against zero content.'),
    },
    'dynamic_blocking_x_attention_sink': {
        'factors': ('dynamic_blocking', 'attention_sink'),
        'levels': {'dynamic_blocking': ('fixed', 'dynamic'),
                   'attention_sink': ('no_sink', 'sink')},
        'cells': {
            ('control', 'control'): 'csa_fix_nosink',
            ('active', 'control'): 'csa_dyn_nosink',
            ('control', 'active'): 'csa_fixed',
            ('active', 'active'): 'csa_dynamic',
        },
        'hypothesis': 'dynamic_blocking_x_attention_sink',
        'note': ('Both a sink and adaptive blocks can absorb off-context mass; '
                 'the interaction asks whether their benefits overlap.'),
    },
    # The on/off interaction above cannot say whether a *learned* sink is what
    # overlaps with adaptive boundaries, because on/off also changes capacity.
    # This factorial uses the frozen-random placebo at both blocking levels, so
    # the second axis is learned-vs-placebo at identical parameter count.
    'dynamic_blocking_x_attention_sink_learning': {
        'factors': ('dynamic_blocking', 'attention_sink_learning'),
        'levels': {'dynamic_blocking': ('fixed', 'dynamic'),
                   'attention_sink_learning': ('random', 'learned')},
        'cells': {
            ('control', 'control'): 'csa_fix_randomsink',
            ('active', 'control'): 'csa_dyn_randomsink',
            ('control', 'active'): 'csa_fixed',
            ('active', 'active'): 'csa_dynamic',
        },
        'hypothesis': 'dynamic_blocking_x_attention_sink_learning',
        'note': ('Does the learned-sink advantage over an identical-capacity '
                 'frozen placebo shrink once boundaries are adaptive?  The '
                 'comparison is capacity-matched on both axes.'),
    },
}

# ---- 预先注册的最小实际效应量（SESOI）------------------------------------
# 「不显著」不是「等价」：精确符号翻转检验不能证明模块可以删掉，只能说明
# 当前种子数下没有检出差异。反过来，大 n 下一个 0.05% 的 PPL 差异也可以
# 「显著」，却不构成「模块有用」的实际证据。消融要回答的是**实际**问题，
# 因此每条声明对照都需要一个在看结果之前就固定的最小实际效应量（SESOI），
# 用配对 TOST 把「等价/可删」与「不确定」分开。
#
# 这里的 margin 是**相对于 without 臂主终点 PPL 的百分比**，不是用观察到的
# 种子方差反推的（后者会让噪声大的面板更容易「等价」）。1% 是本研究对
# 架构级消融预先采用的实际显著性阈值：PPL 改善小于 1% 时，相对训练配方、
# 评测切片和实现细节的差异不足以支撑「该模块有用/无用」的强主张。它不是
# 冻结确认清单的一部分（confirmatory manifest 不动），只约束探索性消融的
# **措辞与决策**；具体面板可以传入更小/更大的已声明 margin，但不能在看到
# 结果后改。`audit_gates.py sesoi` 把「每个声明组件都有 margin」变成门。
ABLATION_SESOI = {
    'dynamic_blocking': {
        'relative_pct': 1.0,
        'rationale': 'A boundary-policy change must move PPL by >=1% to be treated as practically useful; smaller shifts are within the study\'s recipe/evaluation variation.',
    },
    'learned_selection': {
        'relative_pct': 1.0,
        'rationale': 'Learned top-k selection is only practically useful if it beats the equal-budget random control by >=1% PPL.',
    },
    'content_pooling': {
        'relative_pct': 1.0,
        'rationale': 'Pooled block content must change PPL by >=1% versus zero-content blocks to support a content-pooling claim.',
    },
    'content_alignment': {
        'relative_pct': 1.0,
        'rationale': 'Moving the same pooled content one block back must change PPL by >=1% to support the stronger claim that content must be aligned with its own block, not merely present.',
    },
    'attention_sink': {
        'relative_pct': 1.0,
        'rationale': 'A sink contribution below 1% PPL is treated as practically removable at this study scale.',
    },
    'attention_sink_learning': {
        'relative_pct': 1.0,
        'rationale': ('A learned sink that does not beat the same-shape frozen '
                      'random sink by >=1% PPL is treated as an extra-capacity '
                      'effect rather than evidence for a learned sink mechanism.'),
    },
    'compression_regularizer': {
        'relative_pct': 1.0,
        'rationale': 'The compression loss term must deliver >=1% PPL to justify its extra tuning surface.',
    },
    'task_learned_boundary': {
        'relative_pct': 1.0,
        'rationale': 'The task-learned cut policy must beat the threshold gate by >=1% PPL to justify the score-function machinery.',
    },
    'local_fusion': {
        'relative_pct': 1.0,
        'rationale': 'The learned fusion convolution must move PPL by >=1% versus identity fusion to be practically relevant.',
    },
    'local_fusion_learning': {
        'relative_pct': 1.0,
        'rationale': ('A learned fusion convolution that does not beat the same-'
                      'shape frozen random kernel by >=1% PPL is treated as an '
                      'extra-capacity effect rather than evidence for a learned '
                      'fusion mechanism.'),
    },
    'hybrid_hierarchy': {
        'relative_pct': 1.0,
        'rationale': 'The CSA+HCA hierarchy must beat CSA-only by >=1% PPL to support a complementary-layer claim.',
    },
}

# ---- 2x2 交互的预注册 SESOI ------------------------------------------------
# 交互「不显著」同样不是「可加」：精确符号翻转只能说明当前种子数下没有
# 检出交互，不能证明交互在实际上为零。反过来，一个超过 SESOI 的交互才是
# 子可加/协同的实际证据。这里把与消融 SESOI 相同的逻辑移到 2x2 交互：
# 在看结果之前固定一个 log ratio-of-ratios 边界，用配对 TOST 把
# 「实际可加（等价零交互）」与「不确定」分开。
#
# 单位是自然对数交互尺度 I=log((PPL11/PPL10)/(PPL01/PPL00))：
#   relative_pct=1.0  <=>  margin = log(1.01)  <=>  比值比在 [1/1.01, 1.01]
# 这比「绝对 log 差」更可解释，也不会让高 PPL 基线机械地获得更大的区间。
# 每个声明的 factorial 必须有 margin；`audit_gates.py interaction_sesoi`
# 在定义点检查覆盖、单位和正性。缺 margin 时交互仍报告，但只能标
# not_declared，不能声称可加或可删。
INTERACTION_SESOI = {
    'dynamic_blocking_x_learned_selection': {
        'relative_pct': 1.0,
        'rationale': ('A learned-selection benefit that changes by <1% of its '
                      'ratio-of-ratios when dynamic blocking is active is treated '
                      'as practically additive at this study scale.'),
    },
    'dynamic_blocking_x_content_pooling': {
        'relative_pct': 1.0,
        'rationale': ('A content-pooling benefit that changes by <1% of its '
                      'ratio-of-ratios across blocking regimes is treated as '
                      'practically additive.'),
    },
    'dynamic_blocking_x_content_alignment': {
        'relative_pct': 1.0,
        'rationale': ('A content-alignment benefit that changes by <1% of its '
                      'ratio-of-ratios across blocking regimes is treated as '
                      'practically additive.'),
    },
    'dynamic_blocking_x_attention_sink': {
        'relative_pct': 1.0,
        'rationale': ('A sink benefit that changes by <1% of its ratio-of-ratios '
                      'across blocking regimes is treated as practically additive.'),
    },
    'dynamic_blocking_x_attention_sink_learning': {
        'relative_pct': 1.0,
        'rationale': ('A learned-sink advantage over the identical-capacity '
                      'frozen placebo that changes by <1% of its ratio-of-ratios '
                      'across blocking regimes is treated as practically additive.'),
    },
}

# ---- 交互覆盖范围：留一法不能替代 2x2 --------------------------------------
# 留一法只回答「单独移除 X 会怎样」，不能回答「X 和 Y 同时存在时收益是否
# 重叠」。本仓库已声明三组 dynamic_blocking × core module 的 2x2；但核心模块
# 之间的三对交互（selection × pooling、selection × sink、pooling × sink）
# 没有任何 2x2 臂，因此不能把它们写成「互不重叠」或「可加」。这里把
# 「哪些组件对属于本研究要解释的交互范围」显式声明出来；覆盖审计把
# tested / missing / waived 逐对点名。未测的交互不是零交互，也不自动
# 关闭单个已测交互的结论，只关闭「整个核心模块族的交互已被刻画」这一
# 更强的项目级主张。
#
# 只声明核心机制族：hybrid × dynamic 的轴不同（见 design_notes §59.3），
# compression_regularizer / task_learned_boundary / local_fusion 也各自
# 缺少可识别的 2x2 双关臂；把它们硬塞进同一个交互族只会制造不可解释的
# 空槽位。范围外的组件由审计单独列为 outside_scope，而不是静默算作已覆盖。
INTERACTION_SCOPE = {
    'content_alignment_control': {
        'factors': ('dynamic_blocking', 'content_alignment'),
        'rationale': ('The matched content placebo has its own 2x2 against '
                      'dynamic blocking: aligned vs shifted content must be '
                      'compared under both boundary regimes, not only under '
                      'the fixed-block reference.'),
    },
    'attention_sink_learning_control': {
        'factors': ('dynamic_blocking', 'attention_sink_learning'),
        'rationale': ('The capacity-matched sink placebo has its own 2x2 '
                      'against dynamic blocking, so the learned-vs-placebo '
                      'interaction is not inferred from the on/off factorial '
                      'that also changes parameter count.'),
    },
    'core_dynamic_modules': {
        'factors': ('dynamic_blocking', 'learned_selection',
                    'content_pooling', 'attention_sink'),
        'rationale': ('Every pair in this set changes a different read/selection '
                      'mechanism under the same csa_dynamic recipe; their '
                      'overlap is the interaction claim the paper can plausibly '
                      'need. Pairs without a scheduled double-ablation arm are '
                      'reported as missing, not assumed additive.'),
    },
}

# 主方法 vs 参数对齐基线的对比也属于「受控对比」：它回答的是方法本身是否
# 优于同参数量、同优化器、同训练规模、同数据的稠密基线，而不是只回答「消融臂
# 之间谁高谁低」。两条基线都来自 `variant_mlp_ratio` 的参数对齐（MLP 加宽
# 到与 csa_dynamic 相同），因此 `param_controlled` 是这条对比的硬读数。
def declared_ablation_pairs(design=None):
    """Every declared leave-one-out pair, in design order, without duplicates.

    Reports used to hard-code a second pair list, which drifted from
    ``ABLATION_DESIGN`` (for example the task-learned-boundary contrast was
    audited but never printed).  A report must display every contrast the
    completeness audit reasons about, so both now read the same declaration.
    """
    design = ABLATION_DESIGN if design is None else design
    out, seen = [], set()
    if not isinstance(design, dict):
        return out
    for contrasts in design.values():
        if not isinstance(contrasts, (list, tuple)):
            continue
        for item in contrasts:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            pair = (item[0], item[1])
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out

BASELINE_DESIGN = {
    'compression_vs_matched_dense': [
        ('csa_dynamic', 'full_sw128_matched', 'dynamic CSA vs parameter-matched windowed dense'),
        ('csa_dynamic', 'full_matched', 'dynamic CSA vs parameter-matched full dense'),
    ],
}

# ---- 参数匹配基线的预注册 SESOI -------------------------------------------
# 方法优于参数匹配稠密基线也需要实际显著性边界。一个在 n 很大时「显著」但
# 只比稠密基线低 0.1% PPL 的差异，不足以支撑「压缩读取代价换来了实际收益」
# 的主张；反过来，未检出差异也不能由「非显著」升级为「等价」。这个映射与
# ABLATION_SESOI 分开声明，因为 baseline 回答的是方法 vs 强基线，而不是
# 模块留一消融。margin 同样在结果之前固定，单位是 without 臂 PPL 的百分比。
BASELINE_SESOI = {
    'compression_vs_matched_dense': {
        'relative_pct': 1.0,
        'rationale': ('A dynamic-CSA PPL advantage below 1% over a parameter-'
                      'matched dense baseline is treated as within the study\'s '
                      'recipe/evaluation variation and cannot support a practical '
                      'superiority claim.'),
    },
}
DEFAULT_METHODOLOGY_SESOI = {**ABLATION_SESOI, **BASELINE_SESOI}

# ---- 单因素设计契约（design-time gate）------------------------------------
# ABLATION_DESIGN / BASELINE_DESIGN 的每条对照都是「应该只有一个因素在变」的
# 受控对比。run_cfg / train_knobs / hyperparams / objective / params 的配对门
# 都在**落盘记录**上把关（`pair_reason` 一族），但没有任何门检查**臂定义本
# 身**：一条对照的两个臂如果在 `make_layer_cfgs` 里多差了一个旋钮，所有记录
# 级门都会照样放行——被多改的那个旋钮就是那条对照的「因素」，没有第二个门
# 能看见它。这里把每条对照**允许**差的 AttnCfg 键集合预先声明出来，
# `single_factor_design_audit` 在定义点逐层 diff 两个臂的实际配置：
#   * 实际差集 != 声明键集（多一个或少一个）都算违例——多一个是把第二
#     个因素悄悄带进「单因素」对照；少一个是干预被改没了（键还在声明里，
#     臂已经不差它），两种漂移都要在跑 GPU 之前被抓住；
#   * 键集相等的对照按声明内容分级：`kind` 在差集里 = 换注意力算子本身
#     （另有 mlp 加宽的参数对齐在 cfg 之外），属于 architecture_level；
#     其余是 knob 级单因素；
#   * 带外因素（attention cfg 之外的 mlp_ratio 参数对齐）声明在
#     `CONTRAST_OUTSIDE_CFG_FACTORS` 里，由记录级 `_param_control` 关口。
# 键集必须显式枚举、缺声明即 fail-closed：以后往 ABLATION_DESIGN 里加对照
# 时，忘记声明键集本身就是错误，而不是被默认放行。
ABLATION_INTERVENTION_KEYS = {
    'dynamic_blocking': {
        # 「动态分块」因素束 = 边界门(learned gate) + 分块策略本身。
        ('csa_dynamic', 'csa_fixed'): frozenset({'dynamic', 'chunking', 'target_block_tokens'}),
        ('hybrid_csa_dyn', 'hybrid_fixed'): frozenset({'dynamic', 'chunking', 'target_block_tokens'}),
        # HCA 动态层的两个额外键都要在契约里写明（见 CONTRAST_OUTSIDE_CFG_FACTORS）：
        #   block_size —— 动态分段不读它（blocks_from_cuts 用 target_block_tokens/
        #                min_block/max_block），是 inert 字段的显式差，防止将来被
        #                当成「不一致的臂定义」误修；
        #   comp_lambda_mult —— HCA 动态边界门以 4x 权重训练（comp_lambda_effective
        #                0.05→0.2），这是动态 HCA 配方的一部分，不是隐藏因素；
        #                它同时进 train_objective，记录级 objective 门会继续把
        #                dynamic-vs-fixed 标为 system-level，本契约只负责不漏报。
        ('hybrid_dynamic', 'hybrid_fixed'): frozenset({'dynamic', 'chunking', 'target_block_tokens',
                                                      'block_size', 'comp_lambda_mult'}),
        ('hybrid_hca_dyn', 'hybrid_fixed'): frozenset({'dynamic', 'chunking', 'target_block_tokens',
                                                      'block_size', 'comp_lambda_mult'}),
    },
    'learned_selection': {
        ('csa_fixed', 'csa_fix_randidx'): frozenset({'indexer_mode'}),
        ('csa_dynamic', 'csa_dyn_randidx'): frozenset({'indexer_mode'}),
    },
    'content_pooling': {
        ('csa_fixed', 'csa_fix_zerocont'): frozenset({'content_mode'}),
        ('csa_dynamic', 'csa_dyn_zerocont'): frozenset({'content_mode'}),
    },
    'content_alignment': {
        ('csa_fixed', 'csa_fix_shiftcont'): frozenset({'content_mode'}),
        ('csa_dynamic', 'csa_dyn_shiftcont'): frozenset({'content_mode'}),
    },
    'attention_sink': {
        ('csa_fixed', 'csa_fix_nosink'): frozenset({'use_sink'}),
        ('csa_dynamic', 'csa_dyn_nosink'): frozenset({'use_sink'}),
        ('full_sink', 'full'): frozenset({'full_sink'}),
    },
    'attention_sink_learning': {
        ('csa_fixed', 'csa_fix_randomsink'): frozenset({'sink_mode'}),
        ('csa_dynamic', 'csa_dyn_randomsink'): frozenset({'sink_mode'}),
        ('full_sink', 'full_randomsink'): frozenset({'sink_mode'}),
    },
    'compression_regularizer': {
        ('csa_dynamic', 'csa_dyn_nocomp'): frozenset({'comp_lambda_mult'}),
    },
    'task_learned_boundary': {
        ('csa_dynamic_task', 'csa_dynamic'): frozenset({'boundary_policy'}),
    },
    'local_fusion': {
        ('csa_dyn_fuse', 'csa_dynamic'): frozenset({'fuse_mode'}),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn'): frozenset({'fuse_mode'}),
    },
    'local_fusion_learning': {
        ('csa_dyn_fuse', 'csa_dyn_randfuse'): frozenset({'fuse_mode'}),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn_randfuse'): frozenset({'fuse_mode'}),
    },
    'hybrid_hierarchy': {
        # kind 互换（部分层 csa→hca）= 换算子，architecture_level。
        ('hybrid_csa_dyn', 'csa_dynamic'): frozenset({'kind', 'chunking', 'dynamic',
                                                     'block_size', 'target_block_tokens'}),
    },
}
BASELINE_INTERVENTION_KEYS = {
    'compression_vs_matched_dense': {
        # 方法 vs 参数对齐稠密基线：kind 互换 + 动态分块束 + （window 臂的）窗口；
        # mlp 加宽在 attention cfg 之外，由 `_param_control` 在记录级把关。
        ('csa_dynamic', 'full_sw128_matched'): frozenset({'kind', 'chunking', 'dynamic',
                                                          'target_block_tokens', 'window'}),
        ('csa_dynamic', 'full_matched'): frozenset({'kind', 'chunking', 'dynamic',
                                                   'target_block_tokens'}),
    },
}
# 显式带外因素（不在 AttnCfg diff 里，但有别的门）：报告用，键集审计不查它们。
CONTRAST_OUTSIDE_CFG_FACTORS = {
    ('hybrid_dynamic', 'hybrid_fixed'): 'HCA dynamic layers train the boundary gate with comp_lambda_mult=4 (comp_lambda_effective 0.05->0.2); block_size differs but is inert under dynamic segmentation; the objective difference is reported by the record-level objective gate.',
    ('hybrid_hca_dyn', 'hybrid_fixed'): 'HCA dynamic layers train the boundary gate with comp_lambda_mult=4 (comp_lambda_effective 0.05->0.2); block_size differs but is inert under dynamic segmentation; the objective difference is reported by the record-level objective gate.',
    ('csa_dynamic', 'full_sw128_matched'): 'the dense arm widens mlp_ratio to parameter-match csa_dynamic (verified at record time by _param_control / init_core_fp).',
    ('csa_dynamic', 'full_matched'): 'the dense arm widens mlp_ratio to parameter-match csa_dynamic (verified at record time by _param_control / init_core_fp).',
    ('hybrid_csa_dyn', 'csa_dynamic'): 'attention-kind swap (csa->hca on half the layers); capacity/objective readouts stay with the record-level gates.',
}

# Contrasts that are explicitly NOT single-factor.  A multi-family key set is
# only accepted when the design says so here; otherwise it is a second factor
# that slipped into a contrast declared as a single-factor ablation.  These two
# dynamic-HCA pairs move the segmentation bundle together with the HCA
# boundary-gate objective multiplier (and an inert `block_size` field), so they
# keep the system-level reading from design_notes §22.4 and must never appear
# in `single_factor_pairs` merely because their key set was enumerated.
CONTRAST_SYSTEM_LEVEL_DECLARATIONS = {
    ('hybrid_dynamic', 'hybrid_fixed'):
        'dynamic segmentation bundle + HCA boundary-gate comp_lambda_mult=4 '
        '(and the inert block_size field): a system-level dynamic-vs-fixed '
        'readout, not a knob-level single factor',
    ('hybrid_hca_dyn', 'hybrid_fixed'):
        'dynamic segmentation bundle + HCA boundary-gate comp_lambda_mult=4 '
        '(and the inert block_size field): a system-level dynamic-vs-fixed '
        'readout, not a knob-level single factor',
}

# Declared single-factor intervention families over the attention cfg: a diff
# whose keys all sit inside ONE family isolates that knob (and nothing else).
# `kind` is deliberately NOT a family — swapping the attention operator is an
# architecture change (dense/sparse machinery, projections, sink semantics,
# plus the mlp-ratio parameter matching), i.e. multi-factor by construction.
# The seeded-random control's `random_salt` belongs to the `indexer_mode`
# family: the salt is the DEFINITION of the per-(seed, layer) pseudo-random
# draw, not a second intervention (design_notes: seeded random control).
SINGLE_FACTOR_CFG_FAMILIES = (
    frozenset(),  # identical attention cfgs — the factor lives outside (e.g. mlp_ratio)
    frozenset({'indexer_mode', 'random_salt'}),
    frozenset({'qk_norm', 'rope', 'rope_dim'}),
    # content-adaptive segmentation is one mechanism family: the boundary
    # policy/geometry moves together and the selector stays fixed.
    frozenset({'dynamic', 'chunking', 'target_block_tokens'}),
    frozenset({'index_topk'}),
    frozenset({'use_sink'}),
    frozenset({'full_sink'}),
    frozenset({'content_mode'}),
    frozenset({'sink_mode'}),
    frozenset({'boundary_policy'}),
    frozenset({'fuse_mode'}),
    frozenset({'comp_lambda_mult'}),
)

def cfg_keys_single_factor(keys):
    """True iff a config-diff key set isolates ONE declared intervention family.

    Used by the frozen-plan preflight and the audit gates: a comparison may
    carry a mechanism claim only when its attention-cfg diff is inside one
    family (no `kind` swap), the two arms' actual training objectives match,
    and the parameter totals are matched — the three places a "single factor"
    can silently become two.
    """
    ks = frozenset(str(k) for k in keys)
    return 'kind' not in ks and any(ks <= fam for fam in SINGLE_FACTOR_CFG_FAMILIES)

def _cfg_values_differ(value_a, value_b):
    """Whether two config-field values differ, with safe NumPy/torch truth.

    ``value_a != value_b`` returns an elementwise array/tensor for vector
    fields, so using it directly in ``if`` raises for any field with more than
    one element.  A config declaration may legitimately carry such a field
    (e.g. a mask) beside the scalar dose axis; an ambiguous comparison is
    treated as different so the gate fails closed instead of crashing or
    silently dropping a possible second factor.
    """
    try:
        differs = (value_a != value_b)
    except Exception:
        return True
    if isinstance(differs, np.ndarray):
        try:
            return bool(differs.any())
        except Exception:
            return True
    if torch.is_tensor(differs):
        try:
            return bool(differs.any().item())
        except Exception:
            return True
    try:
        return bool(differs)
    except Exception:
        return True


def layer_cfg_diff(ca, cb):
    """Field-by-field diff between two per-layer attention configs.

    Returns `{key: (value_a, value_b)}` over the dataclass fields where the
    two configs disagree.  `single_factor_design_audit` diffs EVERY declared
    contrast through this helper, so the key set a contrast is allowed to
    differ in is a verified property of `make_layer_cfgs` — not an
    assumption in prose.  Mixed config types (a patched rope config vs a base
    config) are still comparable: the fields of the union of both classes are
    compared with getattr, so a field only one class carries can never hide.
    """
    keys = []
    for klass in (type(ca), type(cb)):
        for f in _dc_mod.fields(klass):
            if f.name not in keys:
                keys.append(f.name)
    out = {}
    for name in keys:
        va, vb = getattr(ca, name, None), getattr(cb, name, None)
        if _cfg_values_differ(va, vb):
            out[name] = (va, vb)
    return out

def _design_pair_cfg_keys(variant_a, variant_b, factory, n_layers):
    """Union (over layers) of the config keys in which two arms differ.

    Returns `(keys, per_layer)` where `per_layer` maps layer index -> diff
    dict (layers with no diff are omitted).  Any construction failure in either
    arm -- including a factory that returns a non-sequence or a non-config
    object -- is returned as the exception object so the caller can report it
    as a violation instead of crashing the audit.
    """
    per_layer = {}
    keys = set()
    try:
        cfgs_a = factory(n_layers, variant_a)
        cfgs_b = factory(n_layers, variant_b)
        if isinstance(cfgs_a, (str, bytes, bytearray)) or isinstance(
                cfgs_b, (str, bytes, bytearray)):
            raise TypeError('factory returned a string, not a sequence of layer configs')
        cfgs_a, cfgs_b = list(cfgs_a), list(cfgs_b)
        if len(cfgs_a) != len(cfgs_b):
            raise ValueError(
                f'arms have different layer counts ({len(cfgs_a)} vs {len(cfgs_b)})')
        for i, (x, y) in enumerate(zip(cfgs_a, cfgs_b)):
            d = layer_cfg_diff(x, y)
            if d:
                per_layer[i] = d
                keys.update(d)
    except Exception as e:  # fail closed: an unbuildable arm is a design defect
        return e, per_layer
    return sorted(keys), per_layer

def single_factor_design_audit(design=None, *, baseline_design=None, factory=None,
                               intervention_keys=None, baseline_intervention_keys=None,
                               n_layers=6, extra_pairs=None, verbose=True):
    """Zero-GPU design-time gate: every declared contrast must be single-factor
    AT THE DEFINITION SITE, within its declared key set.

    The record-level gates (`pair_reason` / `train_knobs` / `hyperparams` /
    `objective` / `_param_control`) verify that two arms of a declared
    contrast share the training recipe, the optimizer, the MLP width, the
    objective and the parameter count.  None of them can see the *attention*
    configs, because differing there is the whole point of a contrast: the
    single-factor property lives in the DIFF.  This audit recomputes that
    diff for every declared pair and checks it against the declared
    intervention key set:

      * an UNDECLARED extra key is a second factor smuggled into a
        single-factor contrast (fail);
      * a DECLARED key that no longer differs means the intervention
        silently disappeared — the contrast now compares two identical arms
        (fail);
      * a declared pair with NO entry in the key declaration is refused:
        adding a contrast without declaring its factor keys is the exact
        drift this gate exists to catch (fail-closed, no default-allow);
      * `kind` in the diff means the attention operator itself is swapped —
        the contrast is reported as `architecture_level` (multi-factor by
        declaration, together with the mlp-ratio parameter matching in
        `CONTRAST_OUTSIDE_CFG_FACTORS`), never as a knob-level single factor;
      * a diff whose keys span more than ONE declared family is only accepted
        as `system_level` when the contrast is listed in
        `CONTRAST_SYSTEM_LEVEL_DECLARATIONS`.  Enumerating the extra keys in
        the intervention map is not enough: the contrast is then multi-factor
        by construction, and silently counting it as a knob-level single
        factor would let a two-factor ablation back a one-factor claim.

    Every reported level is therefore machine-derived from the actual diff:
    `single_factor` iff the keys sit inside one family, `architecture_level`
    for operator swaps, `system_level` for explicitly declared bundles.

    Returns the report dict; `status` is 'ok' iff no violation was found.
    Nothing is written to disk and no model is built — this is pure config
    arithmetic, safe in the frozen-confirmation window.
    """
    design = ABLATION_DESIGN if design is None else design
    if baseline_design is None:
        baseline_design = BASELINE_DESIGN
    intervention_keys = ABLATION_INTERVENTION_KEYS if intervention_keys is None else intervention_keys
    baseline_intervention_keys = (BASELINE_INTERVENTION_KEYS
                                  if baseline_intervention_keys is None else baseline_intervention_keys)
    factory = make_layer_cfgs if factory is None else factory
    _factory_name = getattr(factory, '__name__', repr(factory))
    report = {'factory': _factory_name, 'n_layers': int(n_layers),
              'pairs': {}, 'violations': [], 'undeclared': [], 'warnings': [],
              'checked_pairs': 0, 'single_factor_pairs': [],
              'architecture_level_pairs': [], 'system_level_pairs': []}

    def _audit_one(comp, a, b, declared_map, kind):
        label = f'{comp}: {a} vs {b}'
        declared = declared_map.get(comp, {}).get((a, b)) if isinstance(declared_map, dict) else None
        if declared is None:
            # fail closed: an undeclared contrast has no verified key contract.
            report['undeclared'].append(
                f'{label}: no intervention-key declaration for this contrast — declare the '
                f'exact AttnCfg keys it is allowed to differ in (ABLATION_INTERVENTION_KEYS) '
                f'before it can count as single-factor')
            return
        actual, per_layer = _design_pair_cfg_keys(a, b, factory, n_layers)
        if isinstance(actual, Exception):
            report['violations'].append(f'{label}: arm construction failed ({type(actual).__name__}: {actual})')
            return
        declared_set = {str(k) for k in declared}
        actual_set = set(actual)
        extra = sorted(actual_set - declared_set)
        missing = sorted(declared_set - actual_set)
        family_ok = cfg_keys_single_factor(actual_set)
        if 'kind' in actual_set:
            level = 'architecture_level'
        elif (a, b) in CONTRAST_SYSTEM_LEVEL_DECLARATIONS:
            level = 'system_level'
        elif family_ok:
            level = 'single_factor'
        else:
            level = 'multi_factor'
        entry = {'component': comp, 'a': a, 'b': b, 'kind': kind,
                 'declared_keys': sorted(declared_set), 'actual_keys': sorted(actual_set),
                 'extra_keys': extra, 'missing_keys': missing, 'level': level,
                 'single_factor_family': bool(family_ok),
                 'per_layer': {str(i): {k: [v0, v1] for k, (v0, v1) in d.items()}
                               for i, d in per_layer.items()}}
        if CONTRAST_OUTSIDE_CFG_FACTORS.get((a, b)):
            entry['outside_cfg_factors'] = CONTRAST_OUTSIDE_CFG_FACTORS[(a, b)]
        if CONTRAST_SYSTEM_LEVEL_DECLARATIONS.get((a, b)):
            entry['system_level_note'] = CONTRAST_SYSTEM_LEVEL_DECLARATIONS[(a, b)]
        if extra:
            report['violations'].append(
                f'{label}: UNDECLARED factor key(s) {extra} in the config diff — a '
                f'single-factor contrast may differ only in {sorted(declared_set)}')
        if missing:
            report['violations'].append(
                f'{label}: declared key(s) {missing} no longer differ — the intervention '
                f'disappeared; either the arm definition drifted or the declaration is stale')
        if level == 'multi_factor':
            report['violations'].append(
                f'{label}: the config diff keys {sorted(actual_set)} span more than one '
                f'declared single-factor family — this is a multi-factor contrast.  Either '
                f'remove the second factor from the arm definition or declare the contrast '
                f'multi-factor/system-level in CONTRAST_SYSTEM_LEVEL_DECLARATIONS before '
                f'it can be reported as an ablation')
        if not extra and not missing:
            if level == 'single_factor':
                report['single_factor_pairs'].append(label)
            elif level == 'architecture_level':
                report['architecture_level_pairs'].append(label)
            elif level == 'system_level':
                report['system_level_pairs'].append(label)
        report['pairs'][label] = entry
        report['checked_pairs'] += 1

    if isinstance(design, dict):
        for comp, contrasts in design.items():
            if not isinstance(contrasts, (list, tuple)):
                continue
            for item in contrasts:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                _audit_one(str(comp), str(item[0]), str(item[1]), intervention_keys, 'component')
    if isinstance(baseline_design, dict):
        for comp, contrasts in baseline_design.items():
            if not isinstance(contrasts, (list, tuple)):
                continue
            for item in contrasts:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                _audit_one(str(comp), str(item[0]), str(item[1]), baseline_intervention_keys, 'baseline')
    # caller-supplied pairs (e.g. the frozen rope family checked through the
    # v7 factory by audit_gates / v12 plan): same fail-closed contract, the
    # declaration map is passed with them.
    if extra_pairs:
        for comp, a, b, declared in extra_pairs:
            declared_map = {comp: {(a, b): frozenset(declared)}}
            _audit_one(str(comp), str(a), str(b), declared_map, 'extra')
    report['status'] = 'fail' if (report['violations'] or report['undeclared']) else 'ok'
    if verbose:
        if report['violations'] or report['undeclared']:
            print('[single-factor] REFUSED — declared contrasts are not single-factor at the definition site:')
            for _v in report['violations'] + report['undeclared']:
                print(f'  - {_v}')
        else:
            print(f"[single-factor] OK: {report['checked_pairs']} declared contrast(s) verified; "
                  f"{len(report['single_factor_pairs'])} knob-level single-factor, "
                  f"{len(report['architecture_level_pairs'])} architecture-level, "
                  f"{len(report['system_level_pairs'])} system-level (declared multi-factor)")
    return report


# ---- single-factor initialization contract ---------------------------------
# The cfg-diff gate above answers "does the arm definition differ in exactly
# the declared attention fields?".  It cannot see whether ADDING an optional
# parameter shifted the global RNG stream: an `nn.Conv1d` constructed only in
# the fusion treatment consumes torch randomness, so every later block
# parameter is drawn differently and the on/off contrast silently becomes
# "fusion + a different initialization".  This audit builds the two declared
# single-factor arms at the same seed and requires every parameter that is not
# explained by the declared intervention to be bit-identical.
#
# Intervention-specific parameters are whitelisted by the declared cfg keys:
#   fuse_mode                 -> fuse_conv.*
#   use_sink/full_sink/sink_mode -> sink
#   dynamic/chunking/target_block_tokens -> delta_logit (the learned gate)
# Everything else (W_q/W_o/W_aKV/W_bKV/W_aZ/W_bZ/W_DQ/W_DK/W_w/W_kvhead/...)
# must match exactly; an unexpected extra/missing tensor is also a violation.
def single_factor_initialization_audit(design=None, *, factory=None,
                                       n_layers=2, seed=1234,
                                       intervention_keys=None,
                                       baseline_design=None,
                                       baseline_intervention_keys=None,
                                       verbose=False):
    """Design-time check that a single-factor contrast changes only its factor.

    For every pair the cfg gate labels ``single_factor``, construct both arms
    through the real factory at the same seed and compare their parameter
    tensors.  Parameters named by the declared intervention are allowed to
    differ (or to exist on only one side); any other difference means the
    experiment also changed the initialization of the shared trunk/blocks.

    Returns a report dict; ``status`` is ``'ok'`` only when every checked pair
    is clean.  The audit uses a tiny CPU model and never reads a result file.
    """
    design = ABLATION_DESIGN if design is None else design
    factory = make_layer_cfgs if factory is None else factory
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        raise ValueError(
            f'single_factor_initialization_audit: n_layers must be a positive '
            f'integer, got {n_layers!r}')
    n_layers = int(_nl)
    _seed = _as_seed_int(seed)
    if _seed is None or _seed < 0:
        raise ValueError(
            f'single_factor_initialization_audit: seed must be a non-negative '
            f'integer, got {seed!r}')
    seed = int(_seed)
    cfg_rep = single_factor_design_audit(
        design, baseline_design=baseline_design, factory=factory,
        intervention_keys=intervention_keys,
        baseline_intervention_keys=baseline_intervention_keys,
        n_layers=n_layers, verbose=False)
    out = {'kind': 'single_factor_initialization', 'status': 'ok',
           'seed': seed, 'n_layers': n_layers, 'pairs': {},
           'violations': [], 'design_status': cfg_rep.get('status'),
           'note': ('non-intervention parameters must be bit-identical at the '
                    'same seed; optional-module construction must not consume '
                    'the global RNG stream')}
    # Fail closed when the definition-site audit rejects the design.  The loop
    # below only inspects pairs the cfg gate labels `single_factor`; an
    # undeclared extra key or a multi-factor contrast is therefore absent from
    # `single_factor_pairs`, and without this propagation the initialization
    # report would say "ok" for precisely the designs the caller must refuse.
    if cfg_rep.get('status') != 'ok':
        out['violations'].append(
            'the definition-site single-factor audit failed; initialization '
            'cannot certify a contrast whose cfg diff is undeclared or '
            'multi-factor')

    def _allowed_param(name, keys):
        leaf = name.rsplit('.', 1)[-1]
        if 'fuse_mode' in keys and 'fuse_conv' in name:
            return True
        if (keys & {'use_sink', 'full_sink', 'sink_mode'}) and leaf == 'sink':
            return True
        if (keys & {'dynamic', 'chunking', 'target_block_tokens'}
                and leaf == 'delta_logit'):
            return True
        return False

    _state_cache = {}

    def _state(arm):
        try:
            _ckey = ('arm', arm)
            if _ckey in _state_cache:
                return _state_cache[_ckey]
        except TypeError:
            _ckey = None
        _prev = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            model = SmallGPT(32, 16, n_layers, 2, 8, 16,
                             factory(n_layers, arm))
            _state = {n: p.detach().clone()
                      for n, p in model.named_parameters()}
        finally:
            torch.set_rng_state(_prev)
        if _ckey is not None:
            _state_cache[_ckey] = _state
        return _state

    for label in cfg_rep.get('single_factor_pairs') or []:
        rec = cfg_rep.get('pairs', {}).get(label) or {}
        a, b = rec.get('a'), rec.get('b')
        keys = set(rec.get('actual_keys') or [])
        entry = {'a': a, 'b': b, 'declared_keys': sorted(keys)}
        out['pairs'][label] = entry
        try:
            sa, sb = _state(a), _state(b)
        except Exception as e:
            entry['status'] = 'unconstructable'
            out['violations'].append(
                f'{label}: cannot build both arms for the initialization '
                f'check ({type(e).__name__}: {e})')
            continue
        common = sorted(set(sa) & set(sb))
        extra_a = sorted(set(sa) - set(sb))
        extra_b = sorted(set(sb) - set(sa))
        unexpected_extra = [n for n in extra_a + extra_b
                            if not _allowed_param(n, keys)]
        changed = [n for n in common
                   if not torch.equal(sa[n], sb[n])]
        unexpected_changed = [n for n in changed
                              if not _allowed_param(n, keys)]
        entry['n_common'] = len(common)
        entry['changed_params'] = changed
        entry['unexpected_changed'] = unexpected_changed
        entry['unexpected_extra_or_missing'] = unexpected_extra
        if unexpected_changed:
            out['violations'].append(
                f'{label}: non-intervention parameter(s) {unexpected_changed[:6]} '
                'differ at the same seed; the contrast also changes the shared '
                'initialization')
        if unexpected_extra:
            out['violations'].append(
                f'{label}: parameter set differs outside the declared '
                f'intervention ({unexpected_extra[:6]})')
        entry['status'] = ('ok' if not (unexpected_changed or unexpected_extra)
                           else 'confounded')
    if out['violations']:
        out['status'] = 'fail'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
