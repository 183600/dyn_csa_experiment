"""Variance decomposition for the crossed seed x evaluation-unit design.

The confirmatory test is the per-seed exact sign-flip on a paired endpoint.
That test says whether the mean effect is distinguishable from zero; it does
not say *where the variability comes from*.  With a dozen seeds and a fixed
evaluation slice, the same mean can rest on a very stable seed panel or on one
unstable seed; the same slice can be homogeneous or dominated by a handful of
units.  A two-way random-effects decomposition makes that visible and turns
"add seeds or add documents?" into a measured question instead of an
intuition.

The model is the balanced crossed design

    y[s, u] = mu + seed[s] + unit[u] + residual[s, u]

with independent zero-mean components.  Components are estimated by the
method of moments (two-way ANOVA), not by a fitted mixed model: no optimizer,
no convergence failure, and every reported number is an explicit function of
the cell means.  Negative method-of-moments estimates are retained as ``raw``
and truncated to zero for the reported ``estimate``; this is the standard
ANOVA convention and is named in the output.

Nothing here is a test.  It does not recompute p/Holm, does not change a
frozen verdict, and does not turn units into an arm randomisation.
"""


def _vc_matrix(values, *, name='values'):
    """Validate a balanced ``[n_seeds, n_units]`` measurement matrix."""
    if _contains_bool_value(values):
        raise ValueError(f'{name} contains a boolean; a flag is not a measurement')
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not a rectangular numeric matrix: {exc}') from None
    if raw.ndim != 2:
        raise ValueError(f'{name} must be a 2-D [n_seeds, n_units] matrix, '
                         f'got {raw.ndim}-D')
    if raw.dtype.kind not in ('i', 'u', 'f', 'O'):
        # Datetime/timedelta/void/string/boolean/complex arrays can be
        # coerced by ``astype(float)``; they are labels or clocks, not
        # measurements, so refuse them instead of silently scoring them.
        raise ValueError(f'{name} contains non-numeric values')
    if raw.dtype.kind == 'O':
        for v in raw.reshape(-1):
            if _finite_scalar(v) is None:
                raise ValueError(f'{name} contains a non-finite/non-numeric value')
    try:
        arr = raw.astype(float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} contains non-numeric values') from None
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f'{name} must have at least 2 seeds and 2 units, '
                         f'got {arr.shape[0]} x {arr.shape[1]}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains a non-finite value')
    return arr


def _vc_components_normalised(arr):
    """ANOVA table and method-of-moments components on the given scale."""
    S, D = arr.shape
    grand = float(np.mean(arr))
    row = np.mean(arr, axis=1)
    col = np.mean(arr, axis=0)
    ss_seed = float(D * np.sum((row - grand) ** 2))
    ss_unit = float(S * np.sum((col - grand) ** 2))
    resid = arr - row[:, None] - col[None, :] + grand
    ss_resid = float(np.sum(resid ** 2))
    # Rounding can leave a tiny negative sum of squares for an exactly
    # additive panel; clamp only the numerical dust, never a real value.
    if ss_resid < 0.0:
        if ss_resid > -1e-12 * max(1.0, float(np.sum(arr ** 2))):
            ss_resid = 0.0
        else:
            raise ValueError('residual sum of squares is negative beyond rounding')
    df_seed = S - 1
    df_unit = D - 1
    df_resid = df_seed * df_unit
    ms_seed = ss_seed / df_seed
    ms_unit = ss_unit / df_unit
    ms_resid = ss_resid / df_resid
    return {
        'grand_mean': grand,
        'anova': {
            'ss_seed': ss_seed, 'df_seed': df_seed, 'ms_seed': ms_seed,
            'ss_unit': ss_unit, 'df_unit': df_unit, 'ms_unit': ms_unit,
            'ss_residual': ss_resid, 'df_residual': df_resid,
            'ms_residual': ms_resid,
        },
        'raw': {
            'seed': (ms_seed - ms_resid) / D,
            'unit': (ms_unit - ms_resid) / S,
            'residual': ms_resid,
        },
    }


def _vc_satterthwaite_df(anova):
    """Satterthwaite df for the two-way mean variance when it is positive."""
    ms_s = float(anova['ms_seed'])
    ms_u = float(anova['ms_unit'])
    ms_e = float(anova['ms_residual'])
    num = ms_s + ms_u - ms_e
    den = (ms_s ** 2 / max(1, int(anova['df_seed']))
           + ms_u ** 2 / max(1, int(anova['df_unit']))
           + ms_e ** 2 / max(1, int(anova['df_residual'])))
    if not (math.isfinite(num) and math.isfinite(den)) or num <= 0.0 or den <= 0.0:
        return None
    df = (num * num) / den
    return float(df) if math.isfinite(df) and df > 0.0 else None


def _vc_interval(mean, variance, conf, df):
    """Mean +/- t * sqrt(variance), or ``None`` when undefined."""
    if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
        return None
    if not isinstance(variance, (int, float)) or not math.isfinite(float(variance)):
        return None
    variance = float(variance)
    if variance < 0.0:
        if variance > -1e-12 * max(1.0, abs(float(mean))):
            variance = 0.0
        else:
            return None
    if variance == 0.0:
        return {'low': float(mean), 'high': float(mean),
                'halfwidth': 0.0, 'df': df, 'conf': conf,
                'degenerate': True}
    if not (math.isfinite(float(conf)) and 0.0 < float(conf) < 1.0):
        return None
    if df is None or not (math.isfinite(float(df)) and float(df) > 0.0):
        return None
    tc = _t_critical(float(conf), float(df))
    if not math.isfinite(tc):
        return None
    half = float(tc * math.sqrt(variance))
    if not math.isfinite(half):
        return None
    return {'low': float(mean) - half, 'high': float(mean) + half,
            'halfwidth': half, 'df': float(df), 'conf': float(conf),
            'degenerate': False}


def crossed_variance_components(values, *, conf=0.95):
    """Decompose a balanced seed x unit panel into variance components.

    Parameters
    ----------
    values : array-like, shape ``[n_seeds, n_units]``
        A balanced measurement panel (PPL, NLL, a paired difference, ...).
        Every cell must be finite; a missing cell is refused rather than
        silently dropped, because dropping one seed-unit pair destroys the
        balance the ANOVA identity depends on.
    conf : float
        Confidence level for the mean interval and the predictive intervals.

    Returns
    -------
    dict
        ``status`` is ``'ok'`` when at least one component is positive,
        ``'not_estimable'`` for a constant panel (no variance to decompose).
        The result carries the ANOVA table, components, shares, ICCs, the
        variance of the panel mean (both superpopulation and unit-conditional),
        an effective sample size, predictive intervals for a new
        seed/unit/cell, and the next-measurement allocation readout.  Negative
        method-of-moments estimates appear in ``components.*.raw`` and are
        truncated in ``components.*.estimate``; they are never hidden.
    """
    arr = _vc_matrix(values)
    S, D = arr.shape
    N = S * D
    if isinstance(conf, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'conf must be a probability in (0,1), got {conf!r}')
    try:
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        conf = float('nan')
    if not (math.isfinite(conf) and 0.0 < conf < 1.0):
        raise ValueError(f'conf must be a probability in (0,1), got {conf!r}')

    # A PPL/NLL panel should be O(1); still, normalise an extreme finite
    # panel so squaring does not manufacture inf.  Shares, ICCs and effective
    # N are scale-free; absolute components are rescaled only when the product
    # is representable.
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1e100 or (0.0 < max_abs < 1e-100):
        scale = max_abs
        work = arr / scale
    else:
        scale = 1.0
        work = arr
    core = _vc_components_normalised(work)
    anova = core['anova']
    raw = core['raw']
    est = {k: max(0.0, raw[k]) for k in ('seed', 'unit', 'residual')}
    total = est['seed'] + est['unit'] + est['residual']
    raw_total = raw['seed'] + raw['unit'] + raw['residual']
    scale2 = scale * scale
    absolute_ok = bool(math.isfinite(scale2) and scale2 > 0.0)

    def _scale_mean(value):
        if value is None or not absolute_ok:
            return None
        value = float(value) * scale
        return value if math.isfinite(value) else None

    def _scale_var(value):
        if value is None or not absolute_ok:
            return None
        value = float(value) * scale2
        return value if math.isfinite(value) else None

    warnings = []
    negative = [k for k in ('seed', 'unit', 'residual') if raw[k] < 0.0]
    if negative:
        warnings.append(
            'negative method-of-moments estimate(s) for %s were truncated to '
            'zero; the raw values are retained and the components are '
            'imprecise when a variance is near zero' % ', '.join(negative))

    out_components = {}
    for key in ('seed', 'unit', 'residual'):
        out_components[key] = {
            'raw': _scale_var(raw[key]),
            'estimate': _scale_var(est[key]),
            'share': (float(est[key] / total) if total > 0.0 else None),
        }
    grand = _scale_mean(core['grand_mean'])

    def _var_of_mean(include_units=True):
        v = est['seed'] / S + est['residual'] / N
        if include_units:
            v += est['unit'] / D
        return v

    var_mean = _var_of_mean(True)
    var_mean_cond = _var_of_mean(False)
    status = 'ok' if (total > 0.0 and absolute_ok) else 'not_estimable'
    if total <= 0.0:
        warnings.append('all estimated variance components are zero; the '
                        'panel is constant and no sampling variance can be '
                        'reported')
    if not absolute_ok:
        warnings.append('the panel magnitude exceeds the safe squaring range; '
                        'absolute variance components are withheld and only '
                        'scale-free readouts are reported')

    df_mean = _vc_satterthwaite_df(anova)
    if df_mean is None:
        df_mean = float(max(1, min(S, D) - 1))
    if status == 'ok':
        mean_ci = _vc_interval(grand, _scale_var(var_mean), conf, df_mean)
        mean_ci_cond = _vc_interval(grand, _scale_var(var_mean_cond), conf,
                                    max(1.0, S - 1.0))
    else:
        mean_ci = mean_ci_cond = None

    eff_n = None
    design_effect = None
    if status == 'ok' and var_mean > 0.0:
        eff_n = total / var_mean
        design_effect = N / eff_n

    predictions = {}
    if status == 'ok':
        # Targets are deliberately different and named: a new cell is the
        # broadest replicate; a new seed on the SAME units drops the unit main
        # effect (the frozen slice is fixed); a new unit on the same seeds
        # keeps the training panel fixed.
        specs = {
            'new_cell': (est['seed'] + est['unit'] + est['residual'],
                         max(1.0, min(S, D) - 1.0)),
            'new_seed_same_units': (est['seed'] + est['residual'] / D,
                                    max(1.0, S - 1.0)),
            'new_unit_same_seeds': (est['unit'] + est['residual'] / S,
                                    max(1.0, D - 1.0)),
        }
        for label, (variance, df) in specs.items():
            interval = _vc_interval(grand, _scale_var(variance), conf, df)
            if interval is not None:
                predictions[label] = interval

    allocation = None
    if status == 'ok' and total > 0.0:
        s_est, u_est = est['seed'], est['unit']
        if s_est <= 0.0 and u_est <= 0.0:
            allocation = {
                'status': 'not_decidable',
                'reason': ('both seed and unit components are zero; residual '
                           'noise cannot be reduced by reallocating cells')}
        else:
            # For a fixed number of CELLS N = S*D, minimising
            # s/S + u/D + e/(S*D) gives S/D = s/u.  (The sqrt(s/u) rule
            # belongs to a fixed S+D budget, which this design does not hold.)
            # The next-cell comparison uses the exact finite differences,
            # including the residual term that an extra cell on either axis
            # also reduces.
            e_est = est['residual']
            ratio = (math.inf if u_est <= 0.0
                     else (s_est / u_est if s_est > 0.0 else 0.0))
            seed_gain = ((s_est + e_est / D) / float(S * (S + 1))
                         if S > 0 else 0.0)
            unit_gain = ((u_est + e_est / S) / float(D * (D + 1))
                         if D > 0 else 0.0)
            allocation = {
                'status': 'ok',
                'current': {'n_seeds': int(S), 'n_units': int(D),
                            'n_cells': int(N),
                            'seed_to_unit_ratio': float(S / D)},
                'recommended_seed_to_unit_ratio': (
                    float(ratio) if math.isfinite(ratio) else None),
                'recommended_seed_to_unit_ratio_note': (
                    'infinite: the estimated unit component is zero, so the '
                    'fixed-cell optimum puts all cells on the seed axis; the '
                    'next-cell advice still includes the residual term, which '
                    'additional units can reduce' if math.isinf(ratio) else None),
                'next_cell': ('seed' if seed_gain > unit_gain else 'unit'),
                'next_cell_note': (
                    'one more seed removes about %.6g of the mean variance; '
                    'one more unit removes about %.6g (same component scale; '
                    'both finite differences include the residual term)'
                    % (seed_gain, unit_gain)),
                'basis': ('for a fixed number of cells the variance-minimising '
                          'seed:unit ratio is seed_component / unit_component; '
                          'the next-cell advice compares the exact finite '
                          'differences (s+e/D)/(S(S+1)) and (u+e/S)/(D(D+1)) '
                          'at the observed design'),
            }

    out = {
        'kind': 'crossed_variance_components',
        'status': status,
        'design': 'balanced crossed random effects',
        'model': 'y[s,u] = mu + seed[s] + unit[u] + residual[s,u]',
        'n_seeds': int(S), 'n_units': int(D), 'n_cells': int(N),
        'grand_mean': grand,
        'component_scale': {
            'scale': scale,
            'absolute_available': absolute_ok,
            'note': ('absolute components are in squared input units when '
                     'absolute_available is true; otherwise the panel was '
                     'rescaled and only shares/ratios are meaningful'),
        },
        'anova': {k: (_scale_var(v) if k.startswith(('ss_', 'ms_')) else v)
                  for k, v in anova.items()},
        'components': out_components,
        'total_variance': _scale_var(total),
        'total_variance_raw': _scale_var(raw_total),
        'icc': {k: (float(est[k] / total) if total > 0.0 else None)
                for k in ('seed', 'unit', 'residual')},
        'mean_variance': {
            'seed': _scale_var(est['seed'] / S),
            'unit': _scale_var(est['unit'] / D),
            'residual': _scale_var(est['residual'] / N),
            'total': _scale_var(var_mean),
            'conditional_on_units': _scale_var(var_mean_cond),
            'se': (math.sqrt(_scale_var(var_mean))
                   if _scale_var(var_mean) is not None else None),
            'se_conditional_on_units': (
                math.sqrt(_scale_var(var_mean_cond))
                if _scale_var(var_mean_cond) is not None else None),
        },
        'mean_ci': mean_ci,
        'mean_ci_conditional_on_units': mean_ci_cond,
        'satterthwaite_df': (float(df_mean) if df_mean is not None else None),
        'effective_n': {
            'iid_cells': int(N),
            'equivalent': (float(eff_n) if eff_n is not None else None),
            'design_effect': (float(design_effect)
                              if design_effect is not None else None),
            'note': ('equivalent = total_variance / var(mean); design_effect '
                     '= n_cells / equivalent.  Values below 2 mean the cells '
                     'behave like far fewer independent observations'),
        },
        'prediction_intervals': predictions,
        'allocation': allocation,
        'negative_components': negative,
        'warnings': warnings,
        'policy': {
            'estimand': ('variance of one cell and of the S x D panel mean '
                         'under a balanced crossed random-effects model'),
            'components': ('method-of-moments ANOVA; negative estimates are '
                           'reported raw and truncated to zero for inference'),
            'mean_ci': ('Satterthwaite t interval for the superpopulation '
                        'mean (both seeds and units random); '
                        'mean_ci_conditional_on_units fixes the observed unit '
                        'effects and covers only seed/residual sampling'),
            'prediction_intervals': ('new_cell = a new seed on a new unit; '
                                     'new_seed_same_units = retrain one seed '
                                     'on the frozen unit slice; '
                                     'new_unit_same_seeds = one new unit with '
                                     'the frozen seed panel'),
            'not_a_test': ('descriptive variance decomposition; it does not '
                           'recompute p/Holm, does not enter the frozen family, '
                           'and does not change any verdict'),
            'approximation': ('components are imprecise with few seeds/units; '
                              'the intervals are t approximations, not a '
                              'fitted hierarchical Bayes model'),
        },
    }
    if S < 3 or D < 3:
        out['warnings'].append('fewer than three seeds or units: the variance '
                               'components are very imprecise')
    return out


def paired_delta_variance_components(nll_a, nll_b, n_tokens, *, conf=0.95):
    """Decompose a paired per-token NLL difference panel.

    ``nll_a`` and ``nll_b`` are ``[n_seeds, n_units]`` NLL sums and
    ``n_tokens`` is the length-``n_units`` token count of each unit.  The cell
    analysed is ``(nll_a - nll_b) / n_tokens`` (NLL per token; negative means
    arm a has lower NLL).  The equal-cell mean is reported next to the
    token-pooled mean so a length-weighted endpoint is not confused with the
    equal-unit estimand the variance components describe.
    """
    A = _vc_matrix(nll_a, name='nll_a')
    B = _vc_matrix(nll_b, name='nll_b')
    if A.shape != B.shape:
        raise ValueError(f'nll_a and nll_b shapes differ: {A.shape} vs {B.shape}')
    if _contains_bool_value(n_tokens):
        raise ValueError('n_tokens contains a boolean')
    try:
        tok_raw = np.asarray(n_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'n_tokens is not a numeric vector: {exc}') from None
    if tok_raw.dtype.kind not in ('i', 'u', 'f', 'O'):
        raise ValueError('n_tokens contains non-numeric values')
    try:
        tok = tok_raw.astype(float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('n_tokens is not a numeric vector') from None
    if tok.ndim != 1 or tok.shape[0] != A.shape[1]:
        raise ValueError(f'n_tokens must have length {A.shape[1]}, got {tok.shape}')
    if not np.all(np.isfinite(tok)) or np.any(tok <= 0.0):
        raise ValueError('n_tokens must be finite and strictly positive')
    delta = (A - B) / tok[None, :]
    out = crossed_variance_components(delta, conf=conf)
    out['kind'] = 'paired_delta_variance_components'
    out['delta_unit'] = ('NLL per token, arm a minus arm b '
                         '(negative = arm a better)')
    out['mean_delta_nll_per_token_equal_units'] = out.get('grand_mean')
    per_seed = np.sum(A - B, axis=1) / float(np.sum(tok))
    out['token_pooled_delta_nll_per_token'] = float(np.mean(per_seed))
    out['weighting_note'] = (
        'variance components are estimated on the equal-width cell matrix '
        '(one per-token delta per seed x unit); the token-pooled mean weights '
        'units by length and is the primary endpoint, so the two point '
        'estimates need not be identical when unit lengths vary')
    return out
