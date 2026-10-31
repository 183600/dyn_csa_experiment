# ---- 跨 panel 合成的效应尺度敏感性：百分比不是 log ratio ----
#
# `cross_panel_synthesis` 把每个 panel 的逐种子效应写成
# `100 * (with - without) / without_mean`（without 臂 PPL 的百分比），
# 再把它们交给 `meta_analyze_effects`。后者文档化的输入是**无量纲 log
# PPL 比**（`paired_log_ratio`），因为 PPL 是平均 token NLL 的指数：
# 只有 log PPL 的差是可加的，也是乘性尺度上对称的。把百分比直接喂给
# inverse-variance 元分析意味着：
#
# 1. 权重来自百分比尺度的 within-panel 方差，而不是 log 尺度的方差；
# 2. 合并出的「-x%」不是合并 log 效应反变换后的百分比（Jensen 偏差）；
# 3. 尺度选择本身可能改变合并方向或「区间是否排除 0」——
#    这正是 §105 在单对比上已经指出、§124 在跨 panel 上仍留作
#    「未枚举选择」的自由度。
#
# 本片段不把 log 追认为唯一「正确」尺度（§105 的边界仍成立）：它把两个
# 估计量并列计算，报告方向/零排除是否一致，并把「两个尺度都支持」作为
# `scale_robust_claim` 的必要条件。合成块的 percent 口径保持不变；
# 只在两尺度分歧时收回 `claim_allowed` 并写入具名原因，绝不产生新 p 值、
# 不进入 Holm、不改变任何 panel 的 verified、冻结判定或 CODE_SEMANTICS。

CROSS_PANEL_SCALE_SENSITIVITY_POLICY = {
    'version': 'cross-panel-scale-sensitivity-1',
    'percent_scale': ('100 * (with_ppl - without_ppl) / without_mean; the '
                      'historical synthesis input'),
    'log_scale': ('the shared meta-analysis documents `paired_log_ratio`.  '
                  'When every included panel records per-seed arm PPLs the '
                  'true `log(with_i) - log(without_i)` is used; otherwise the '
                  'exact transform of the recorded percent effect, '
                  '`log1p(percent/100) = log(with_i / without_mean)`, is used '
                  'for ALL panels so one coherent log scale is pooled.  The '
                  'field `log_ratio_basis` names which one.'),
    'back_transform': '100 * (exp(log_mean) - 1)',
    'direction_gate': ('the pooled percent mean and the pooled log mean must '
                       'have the same sign'),
    'precision_gate': ('where both pooled intervals exist, they must agree on '
                       'whether zero is excluded at alpha'),
    'claim_gate': ('a scale-robust common effect requires both scales to be '
                   'poolable, agree on direction and agree on zero exclusion'),
    'poolability_gate': ('the two scales must also agree on the heterogeneity '
                         'gate (poolable); a percent-poolable family whose log '
                         'meta-analysis is heterogeneous is not scale-robust'),
    'role': ('descriptive effect-scale sensitivity; it creates no p-value, '
             'enters no Holm family, and changes no panel `verified` flag, '
             'frozen manifest entry or confirmed verdict'),
}

_CROSS_PANEL_SCALE_DISAGREEMENTS = ('direction_discordant',
                                    'precision_discordant',
                                    'poolability_discordant')


def _cpss_finite(value):
    """Finite float or ``None``; booleans/strings are not measurements."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _cpss_ci(value):
    """A finite two-sided interval, or ``None`` (strict-JSON safe)."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo = _cpss_finite(value[0])
    hi = _cpss_finite(value[1])
    return [lo, hi] if lo is not None and hi is not None else None


def _cpss_sign(value):
    """-1/0/1 for a finite number, else ``None``."""
    number = _cpss_finite(value)
    if number is None:
        return None
    return (number > 0.0) - (number < 0.0)


def _cpss_excludes_zero(ci):
    """Whether a finite interval lies strictly on one side of zero."""
    ci = _cpss_ci(ci)
    if ci is None:
        return None
    return bool(ci[0] > 0.0 or ci[1] < 0.0)


def _cpss_meta_summary(meta):
    """Pull the pooled point/interval out of a meta-analysis record.

    The random-effects estimate is the headline when it exists; the
    fixed-effect estimate is the fallback, and the record names which one was
    used.  Missing/non-finite values stay ``None`` rather than becoming NaN or
    infinity in a strict-JSON report.
    """
    meta = meta if isinstance(meta, dict) else {}
    re_block = meta.get('random_effects')
    fe_block = meta.get('fixed_effect')
    re_block = re_block if isinstance(re_block, dict) else {}
    fe_block = fe_block if isinstance(fe_block, dict) else {}
    source = 'random_effects'
    mean = _cpss_finite(re_block.get('mean'))
    se = _cpss_finite(re_block.get('se'))
    ci = _cpss_ci(re_block.get('ci'))
    if mean is None and ci is None:
        source = 'fixed_effect'
        mean = _cpss_finite(fe_block.get('mean'))
        se = _cpss_finite(fe_block.get('se'))
        ci = _cpss_ci(fe_block.get('ci'))
    if mean is None and ci is None:
        source = None
    return {'status': meta.get('status'),
            'poolable': meta.get('poolable') is True,
            'mean': mean, 'se': se, 'ci': ci,
            'estimate_source': source,
            'heterogeneity': (meta.get('heterogeneity')
                              if isinstance(meta.get('heterogeneity'), dict)
                              else None)}


def _cpss_back_transform_pct(log_value):
    """``100 * (exp(x) - 1)`` with overflow reported as ``None``."""
    x = _cpss_finite(log_value)
    if x is None:
        return None
    try:
        value = 100.0 * math.expm1(x)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def cross_panel_scale_contrast(block, *, alpha=0.05, min_n=3):
    """Recompute one contrast on the percent and log-ratio scales.

    ``block`` is a contrast block produced by ``cross_panel_synthesis`` (it
    carries ``panels`` with per-seed percent effects and ``meta`` with the
    percent-scale pooled estimate).

    The log scale has two possible bases, and the record names the one it
    used in ``log_ratio_basis``:

    * ``paired_per_seed`` — every included panel recorded per-seed with/without
      PPLs, so the true ``log(with_i) - log(without_i)`` is used;
    * ``panel_mean_without`` — some panel lacks those vectors, so ALL panels
      use the exact transform of the recorded percent effect,
      ``log1p(percent_effect/100) = log(with_i / without_mean)``.  This is a
      log ratio to the panel's MEAN baseline, not the paired log ratio.

    Mixing the two bases would pool different estimands, so a single basis is
    chosen for the whole family.  Under the mean-baseline basis a percent
    effect at or below -100 implies a non-positive with-arm PPL; that log
    ratio is undefined and the contrast is reported ``not_estimable`` instead
    of fabricating one.  Under the paired basis the recorded PPLs are positive
    by construction, so a percent effect below -100 can still have a valid
    paired log ratio.

    The result is a deterministic, strict-JSON record.  It estimates two
    *different* quantities; neither is declared "the correct" one.
    """
    alpha = _validated_alpha(alpha, context='cross_panel_scale_contrast alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    block = block if isinstance(block, dict) else {}
    panels = block.get('panels')
    panels = panels if isinstance(panels, dict) else {}
    pct_meta = block.get('meta')
    pct = _cpss_meta_summary(pct_meta)
    corpus = block.get('corpus')
    corpus_fps = (corpus.get('fps') if isinstance(corpus, dict) else None)
    corpus_fps = (list(corpus_fps)
                  if isinstance(corpus_fps, (list, tuple)) else [])
    panel_order = [str(panel) for panel in panels]
    out = {
        'kind': 'cross_panel_scale_contrast',
        'version': CROSS_PANEL_SCALE_SENSITIVITY_POLICY['version'],
        'status': 'not_estimable',
        'reasons': [],
        'alpha': float(alpha),
        'min_n': int(min_n),
        'panel_order': panel_order,
        'percent': dict(pct, scale='relative_percent_of_without_arm_ppl'),
        'log_ratio': {
            'scale': 'log(with_ppl) - log(without_ppl)',
            'status': 'not_estimable', 'poolable': False,
            'mean': None, 'se': None, 'ci': None,
            'estimate_source': None, 'heterogeneity': None,
            'back_transformed_relative_pct': None,
            'back_transformed_ci_pct': None,
        },
        'direction_agree': None,
        'zero_exclusion_agree': None,
        'percent_excludes_zero': _cpss_excludes_zero(pct.get('ci')),
        'log_excludes_zero': None,
        'jensen_gap_pct': None,
        'claim_allowed_on_both_scales': False,
        'log_meta_input': {'points': None},
        'log_ratio_basis': None,
        'per_seed_ppl_panels': [],
        'note': CROSS_PANEL_SCALE_SENSITIVITY_POLICY['claim_gate'],
    }
    if not panels:
        out['reasons'].append('no per-panel percent effects to transform')
        return out
    # Pick ONE log basis for the whole family.  Mixing true paired log ratios
    # with the mean-baseline transform would pool two different estimands.
    _paired_panels = []
    for _panel in panel_order:
        _row = panels.get(_panel)
        _le = _row.get('log_effects') if isinstance(_row, dict) else None
        if _le is None:
            continue
        _eff = _row.get('effects')
        if (not isinstance(_le, (list, tuple))
                or not isinstance(_eff, (list, tuple))
                or len(_le) != len(_eff)
                or any(_cpss_finite(x) is None for x in _le)):
            out['reasons'].append(
                f'{_panel}: recorded log_effects is malformed or does not '
                'match the percent effect length')
            return out
        _paired_panels.append(_panel)
    _paired_all = bool(panel_order) and len(_paired_panels) == len(panel_order)
    out['log_ratio_basis'] = ('paired_per_seed' if _paired_all
                              else 'panel_mean_without')
    out['per_seed_ppl_panels'] = (_paired_panels if _paired_all else [])
    out['log_ratio']['scale'] = (
        'log(with_ppl) - log(without_ppl) (per-seed paired PPLs)'
        if _paired_all else
        'log1p(percent_effect/100) = log(with_ppl / without_mean) '
        '(panel mean baseline)')
    points = []
    for index, panel in enumerate(panel_order):
        row = panels.get(panel)
        if not isinstance(row, dict):
            out['reasons'].append(f'{panel}: panel row is not an object')
            return out
        effects = row.get('effects')
        seeds = row.get('seeds')
        if not isinstance(effects, (list, tuple)) or not effects:
            out['reasons'].append(f'{panel}: no per-seed percent effects')
            return out
        if not isinstance(seeds, (list, tuple)):
            out['reasons'].append(f'{panel}: no per-seed seed identity')
            return out
        try:
            norm_seeds = [_as_seed_int(s) for s in seeds]
        except (TypeError, ValueError, OverflowError):
            out['reasons'].append(f'{panel}: seed identity is not an integer list')
            return out
        if any(s is None for s in norm_seeds):
            out['reasons'].append(f'{panel}: seed identity is not an integer list')
            return out
        if len(norm_seeds) != len(effects) or len(set(norm_seeds)) != len(norm_seeds):
            out['reasons'].append(
                f'{panel}: seed identity length/duplication does not match the '
                'percent effects')
            return out
        if _paired_all:
            # Validate the percent side anyway, then use the true paired log
            # ratios.  A percent effect below -100% is possible here when the
            # per-seed without PPL is above the panel mean; it is not an
            # invalid log ratio because the paired PPLs are positive.
            for i, effect in enumerate(effects):
                if _cpss_finite(effect) is None:
                    out['reasons'].append(
                        f'{panel}: effects[{i}] is not a finite percent effect')
                    return out
            log_effects = [float(x) for x in row['log_effects']]
        else:
            log_effects = []
            for i, effect in enumerate(effects):
                value = _cpss_finite(effect)
                if value is None:
                    out['reasons'].append(
                        f'{panel}: effects[{i}] is not a finite percent effect')
                    return out
                # log1p needs 1 + effect/100 > 0.  Dividing first keeps the
                # comparison meaningful even for effects near the float limit.
                ratio = value / 100.0
                if not math.isfinite(ratio) or ratio <= -1.0:
                    out['reasons'].append(
                        f'{panel}: effects[{i}]={value!r}% implies a '
                        'non-positive with-arm PPL under the panel mean '
                        'baseline; log ratio is undefined')
                    return out
                log_value = math.log1p(ratio)
                if not math.isfinite(log_value):
                    out['reasons'].append(
                        f'{panel}: effects[{i}]={value!r}% has no finite log '
                        'ratio')
                    return out
                log_effects.append(float(log_value))
        points.append({
            'label': panel, 'scale': 1.0,
            'effects': log_effects, 'seeds': [int(s) for s in norm_seeds],
            'corpus_fp': (corpus_fps[index]
                          if index < len(corpus_fps) else None),
            'complete_pairs': True,
        })
    out['log_meta_input'] = {'points': points}
    log_meta = _cps_pool(points, alpha=alpha, min_n=min_n,
                         pair_detail_limit=0)
    log_summary = _cpss_meta_summary(log_meta)
    log_block = out['log_ratio']
    for key in ('status', 'poolable', 'mean', 'se', 'ci',
                'estimate_source', 'heterogeneity'):
        log_block[key] = log_summary.get(key)
    if log_summary.get('mean') is not None:
        log_block['back_transformed_relative_pct'] = _cpss_back_transform_pct(
            log_summary['mean'])
        ci = _cpss_ci(log_summary.get('ci'))
        if ci is not None:
            back_ci = [_cpss_back_transform_pct(ci[0]),
                       _cpss_back_transform_pct(ci[1])]
            log_block['back_transformed_ci_pct'] = (
                back_ci if all(x is not None for x in back_ci) else None)
    out['log_excludes_zero'] = _cpss_excludes_zero(log_summary.get('ci'))
    # Direction / zero-exclusion agreement.  A missing estimate is not
    # agreement: it is reported as None and blocks the scale-robust claim.
    pct_sign = _cpss_sign(pct.get('mean'))
    log_sign = _cpss_sign(log_summary.get('mean'))
    if pct_sign is not None and log_sign is not None:
        out['direction_agree'] = bool(pct_sign == log_sign)
    pct_zero = out['percent_excludes_zero']
    log_zero = out['log_excludes_zero']
    if pct_zero is not None and log_zero is not None:
        out['zero_exclusion_agree'] = bool(pct_zero == log_zero)
    percent_mean = _cpss_finite(pct.get('mean'))
    back_pct = _cpss_finite(
        log_block.get('back_transformed_relative_pct'))
    if percent_mean is not None and back_pct is not None:
        gap = percent_mean - back_pct
        out['jensen_gap_pct'] = gap if math.isfinite(gap) else None
    # The two scales are only comparable when both meta analyses actually
    # estimated a pooled point.  `meta_analyze_effects` can return status=ok
    # with poolable=False (heterogeneous), which is a legitimate descriptive
    # estimate; the claim gate then stays closed because poolable is false.
    if pct.get('status') != 'ok' or log_summary.get('status') != 'ok':
        out['reasons'].append(
            'at least one scale did not produce an estimable meta-analysis')
        return out
    if percent_mean is None or log_summary.get('mean') is None:
        out['reasons'].append(
            'at least one scale has no pooled point estimate')
        return out
    if out['direction_agree'] is False:
        out['status'] = 'direction_discordant'
        out['reasons'].append(
            'the pooled percent mean and the pooled log mean have opposite '
            'signs; the common-effect direction depends on the effect scale')
    elif out['zero_exclusion_agree'] is False:
        out['status'] = 'precision_discordant'
        out['reasons'].append(
            'the pooled percent and log intervals disagree on whether zero is '
            'excluded; the significance of the common effect depends on the '
            'effect scale')
    elif out['zero_exclusion_agree'] is None:
        out['reasons'].append(
            'at least one pooled interval is unavailable; zero-exclusion '
            'agreement cannot be attested')
        return out
    elif bool(pct.get('poolable')) != bool(log_summary.get('poolable')):
        out['status'] = 'poolability_discordant'
        out['reasons'].append(
            'the percent and log meta-analyses disagree on the heterogeneity '
            'gate (poolable); the common-effect admissibility depends on the '
            'effect scale')
    else:
        out['status'] = 'concordant'
    out['claim_allowed_on_both_scales'] = bool(
        pct.get('poolable') is True
        and log_summary.get('poolable') is True
        and out['direction_agree'] is True
        and out['zero_exclusion_agree'] is True)
    return out


def cross_panel_scale_sensitivity(synthesis, *, alpha=0.05, min_n=3):
    """Aggregate the per-contrast scale-sensitivity readout of a synthesis.

    ``synthesis`` is the report returned by ``cross_panel_synthesis`` (or a
    bare dict with the same ``components`` structure).  Every contrast that
    carries a stored ``scale_sensitivity`` block is summarised; a contrast
    without one is recomputed from its stored ``panels``/``meta`` so a
    consumer that only kept the synthesis block still gets the readout.

    The summary is descriptive.  It names which components are scale-robust
    and which ones are only supported on one of the two estimands; it does
    not add a test, a p-value or a Holm member.
    """
    alpha = _validated_alpha(alpha,
                             context='cross_panel_scale_sensitivity alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    synthesis = synthesis if isinstance(synthesis, dict) else {}
    components = synthesis.get('components')
    out = {
        'kind': 'cross_panel_scale_sensitivity',
        'version': CROSS_PANEL_SCALE_SENSITIVITY_POLICY['version'],
        'status': 'not_verifiable_locally',
        'policy': dict(CROSS_PANEL_SCALE_SENSITIVITY_POLICY),
        'alpha': float(alpha), 'min_n': int(min_n),
        'components': {}, 'n_contrasts': 0,
        'concordant_components': [],
        'scale_sensitive_components': [],
        'not_estimable_components': [],
        'note': ('percent and log-ratio are different estimands; a common '
                 'effect is only scale-robust when both are poolable, have the '
                 'same direction and agree on zero exclusion.  This block does '
                 'not change any panel verdict.'),
    }
    if not isinstance(components, dict) or not components:
        out['reason'] = 'the synthesis carries no component contrasts'
        return out
    for comp, comp_block in components.items():
        contrasts = (comp_block.get('contrasts')
                     if isinstance(comp_block, dict) else None)
        if not isinstance(contrasts, dict):
            continue
        comp_out = {'contrasts': {}, 'status': 'not_measured'}
        for key, block in contrasts.items():
            if not isinstance(block, dict):
                continue
            if block.get('status') == 'not_measured':
                record = {'status': 'not_measured',
                          'reasons': ['the contrast was not measured']}
            else:
                record = block.get('scale_sensitivity')
                if not isinstance(record, dict):
                    try:
                        record = cross_panel_scale_contrast(
                            block, alpha=alpha, min_n=min_n)
                    except (TypeError, ValueError, OverflowError) as exc:
                        record = {
                            'kind': 'cross_panel_scale_contrast',
                            'status': 'not_estimable',
                            'reasons': [f'{type(exc).__name__}: {exc}']}
            comp_out['contrasts'][str(key)] = record
            out['n_contrasts'] += 1
        statuses = [r.get('status') for r in comp_out['contrasts'].values()]
        if any(s in _CROSS_PANEL_SCALE_DISAGREEMENTS for s in statuses):
            comp_out['status'] = 'scale_sensitive'
            out['scale_sensitive_components'].append(comp)
        elif 'concordant' in statuses:
            comp_out['status'] = 'concordant'
            out['concordant_components'].append(comp)
        elif any(s not in (None, 'not_measured') for s in statuses):
            comp_out['status'] = 'not_estimable'
            out['not_estimable_components'].append(comp)
        out['components'][str(comp)] = comp_out
    for key in ('concordant_components', 'scale_sensitive_components',
                'not_estimable_components'):
        out[key] = sorted(set(out[key]))
    if out['scale_sensitive_components']:
        out['status'] = 'warn'
    elif out['not_estimable_components'] and not out['concordant_components']:
        out['status'] = 'not_estimable'
    elif out['not_estimable_components']:
        out['status'] = 'warn'
    elif out['concordant_components']:
        out['status'] = 'ok'
    elif out['n_contrasts']:
        out['status'] = 'not_estimable'
    return out
