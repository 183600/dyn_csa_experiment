# ---- 跨 panel 效应合成：可合并性门 + 不按显著性选 panel 的随机效应读数 ----
#
# `ablation_coverage_across_panels` 回答的是「某组件在哪些 panel 通过」，
# 它刻意不做合并。这个片段补上结果分析层缺的那一步：把**所有可估计**的
# panel 效应放在同一个尺度上合并，并先证明它们可合并。两个关键点：
#
# 1. 不能只合并「verified」的 panel。显著性筛选会条件在 p<alpha 上，把
#    合并均值推向有利方向（winner's curse 的跨 panel 版本）。合成必须纳入
#    每个有受控逐种子效应的 panel，无论其 p 值；只合并 verified 的读数只
#    作为**描述性选择偏差敏感性**并列展示，不能当作合并估计。
# 2. 合并前必须有可合并性契约：同一组件、同一臂对、同一效应尺度/符号、
#    同一语料/分词器、每 panel n>=min_n、逐种子配对完整、种子身份可核验。
#    共享训练种子的 panel 是重复测量，不是独立研究；只能走配对公共效应，
#    或者明确拒绝合并。缺身份的 panel 不能默认独立。
#
# 统计核心复用 `meta_analyze_effects`（DerSimonian–Laird τ²、Cochran Q、
# I²、固定/随机效应区间、预测区间、共享种子诊断），因此不引入第二套
# 元分析实现。本块不新增 p 值、不进入 Holm 族、不改变任何 panel 的
# `verified`/`confirmed` 或冻结判定。

CROSS_PANEL_SYNTHESIS_POLICY = {
    'version': 'cross-panel-synthesis-1',
    'effect_scale': 'relative_percent_of_without_arm_ppl',
    'sign_convention': ('with - without, in percent of the without-arm PPL; '
                        'the declared predicted_sign says which direction is '
                        'the hypothesis'),
    'min_panels_for_pooled_claim': 3,
    'i2_ceiling_percent': 50.0,
    'q_alpha': 0.10,
    'include_only_verified': False,
    'selection_bias_readout': True,
    'require_declared_seeds': True,
    'require_complete_pairs': True,
    'require_corpus_binding': True,
    'role': ('descriptive cross-panel random-effects synthesis; it never '
             'changes a panel p-value, Holm family, verified flag, frozen '
             'manifest or confirmed verdict'),
}


class _CrossPanelSynthesisError(ValueError):
    """A contrast cannot enter the synthesis; the message is the reason."""


# Per-pair detail is quadratic in the number of panels.  The claim/status
# computation still uses every pair; only the rendered detail is capped.
# A caller may override it with policy['max_pair_details'] without changing
# the declared synthesis policy itself.
CROSS_PANEL_SYNTHESIS_MAX_PAIR_DETAILS = 1024


def _cps_number(value, name, *, positive=False):
    """Finite float for a synthesis input; bool/str are not measurements."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)) or value is None:
        raise _CrossPanelSynthesisError(f'{name} is not a number')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _CrossPanelSynthesisError(f'{name} is not a finite number') from None
    if not math.isfinite(out):
        raise _CrossPanelSynthesisError(f'{name} is not finite')
    if positive and out <= 0.0:
        raise _CrossPanelSynthesisError(f'{name} must be > 0')
    return out


def _cps_finite_or_none(value):
    """Finite float for display, else None (strict-JSON safe)."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _cps_in_names(value, name):
    """Whether ``name`` is listed in a report's component-name sequence."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return False
    try:
        return str(name) in {str(x) for x in value}
    except (TypeError, ValueError):
        return False


def _cps_sign_or_none(value):
    """-1/0/1 declaration only; booleans/strings are not measurements."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)) or value is None:
        return None
    try:
        number = _as_seed_int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if number in (-1, 0, 1) else None


def _cps_entry_effect(entry):
    """Per-seed relative effect for one controlled contrast entry.

    The effect is ``100 * (with - without) / without_mean`` so panels with
    different PPL levels share one scale.  The per-seed values are rebuilt
    from the recorded deltas, and the recorded aggregate is cross-checked
    instead of trusted: a stale `relative_effect_pct` would silently change
    the synthesis input while every derived number stayed self-consistent.

    Every reported number is a finite measurement (or ``None`` when the
    original-unit SD/SE is not representable).  A relative effect that
    overflows float64 is not a measurement either: the panel is excluded with
    a named reason instead of leaking ``inf`` into a strict-JSON report.
    """
    if not isinstance(entry, dict):
        raise _CrossPanelSynthesisError('contrast entry is not an object')
    if not bool(entry.get('controlled')):
        raise _CrossPanelSynthesisError('contrast is not controlled')
    deltas = entry.get('deltas')
    if not isinstance(deltas, (list, tuple)) or not deltas:
        raise _CrossPanelSynthesisError('no per-seed deltas recorded')
    vals = []
    for i, d in enumerate(deltas):
        vals.append(_cps_number(d, f'deltas[{i}]'))
    seeds_raw = entry.get('seeds')
    if seeds_raw is None:
        raise _CrossPanelSynthesisError('no per-seed seed identity recorded')
    if isinstance(seeds_raw, (str, bytes, bytearray, dict)):
        raise _CrossPanelSynthesisError('seeds is not a seed list')
    try:
        seeds = [_as_seed_int(s) for s in seeds_raw]
    except TypeError:
        raise _CrossPanelSynthesisError('seeds is not a seed list') from None
    if any(s is None for s in seeds):
        raise _CrossPanelSynthesisError('seed identity contains a non-integer')
    if len(seeds) != len(vals) or len(set(seeds)) != len(seeds):
        raise _CrossPanelSynthesisError(
            'seed identity length/duplication does not match the deltas')
    if not bool(entry.get('fully_paired')):
        raise _CrossPanelSynthesisError(
            'paired panel is not complete (a dropped seed makes a complete-case '
            'subset, not a full panel)')
    without_mean = _cps_number(entry.get('without_mean'), 'without_mean',
                               positive=True)
    effects = []
    for i, v in enumerate(vals):
        # Divide before multiplying by 100: 100*1e308 overflows even when the
        # final relative effect (1e308/1e10*100) is representable.
        effect = (v / without_mean) * 100.0
        if not math.isfinite(effect):
            raise _CrossPanelSynthesisError(
                f'deltas[{i}]/without_mean is not representable as a finite '
                'relative effect')
        effects.append(float(effect))
    n = len(effects)
    # `np.mean`/`np.std` square deviations and sum the vector.  Finite inputs
    # near the float limit (or far below sqrt(tiny)) can therefore report
    # inf/zero for a real spread.  Reuse the repository's scaled helper; an
    # SD/SE that still cannot be represented in the input unit is withheld as
    # None, not emitted as Infinity.
    _arr = np.asarray(effects, dtype=float)
    _ms = _scaled_mean_std_if_extreme(_arr, ddof=1) if n > 1 else None
    if _ms is not None:
        mean, sd = float(_ms[0]), float(_ms[1])
    else:
        mean = float(np.mean(_arr))
        sd = float(np.std(_arr, ddof=1)) if n > 1 else None
    mean = _cps_finite_or_none(mean)
    if mean is None:
        raise _CrossPanelSynthesisError(
            'relative effect mean is not representable as a finite number')
    sd = _cps_finite_or_none(sd)
    se = (float(sd / math.sqrt(n))
          if sd is not None and n > 1 else None)
    se = _cps_finite_or_none(se)
    recorded = entry.get('relative_effect_pct')
    if recorded is not None:
        rec = _cps_number(recorded, 'relative_effect_pct')
        if not math.isclose(mean, rec, rel_tol=1e-3, abs_tol=1e-4):
            raise _CrossPanelSynthesisError(
                'recorded relative_effect_pct is inconsistent with the '
                'per-seed deltas/without_mean')
    out = {'effects': [float(x) for x in effects], 'seeds': [int(s) for s in seeds],
           'n': int(n), 'mean': mean, 'sd': sd, 'se': se,
           'without_mean': without_mean, 'complete_pairs': True}
    for key in ('with', 'without', 'kind'):
        if key in entry:
            out[key] = entry.get(key)
    if 'practical_decision' in entry and entry.get('practical_decision') is not None:
        _pd = entry.get('practical_decision')
        out['practical_decision'] = _pd if isinstance(_pd, str) else str(_pd)
    if 'predicted_sign' in entry:
        out['predicted_sign'] = _cps_sign_or_none(entry.get('predicted_sign'))
    if 'observed_sign' in entry:
        out['observed_sign'] = _cps_sign_or_none(entry.get('observed_sign'))
    for key in ('direction_consistent', 'capacity_confounded',
                'param_controlled', 'param_capacity_controlled', 'seed_driven'):
        if key in entry:
            out[key] = (None if entry.get(key) is None
                        else bool(entry.get(key)))
    if 'p_exact_signflip' in entry:
        out['p_exact_signflip'] = _cps_finite_or_none(
            entry.get('p_exact_signflip'))
    return out


def _cps_pool(points, *, alpha, min_n, pair_detail_limit=None,
              _prevalidated_points=False, _point_row_cache=None):
    """Run the shared meta-analysis, returning a status instead of raising."""
    try:
        meta = meta_analyze_effects(
            points, alpha=alpha, min_seeds=min_n,
            _seed_pair_detail_limit=pair_detail_limit,
            _prevalidated_points=_prevalidated_points,
            _point_row_cache=_point_row_cache)
    except (TypeError, ValueError, OverflowError) as exc:
        return {'status': 'not_estimable', 'poolable': False,
                'reason': f'{type(exc).__name__}: {exc}'}
    meta = dict(meta)
    meta.pop('policy', None)
    return meta


def _cps_claim(meta, *, direction_ok, independence_status, corpus_ok,
               n_panels=None, min_panels=3,
               i2_ceiling_percent=50.0, q_alpha=0.10):
    """Decide whether a cross-panel common effect may be claimed.

    Separate from `meta.poolable` because the latter is about heterogeneity
    and within-panel completeness; independence of the panels is a second
    axis.  A shared-seed family can still support a paired common effect (the
    meta block carries `paired_common_effect`), but it must not be described
    as an independent replication.
    """
    reasons = []
    meta = meta if isinstance(meta, dict) else {}
    if meta.get('status') != 'ok':
        reasons.append(meta.get('status') or 'meta-analysis not estimable')
    if meta.get('poolable') is not True:
        reasons.append('meta-analysis poolability gate is not satisfied')
    if direction_ok is not True:
        reasons.append('the declared direction is not uniform across panels')
    if corpus_ok is not True:
        reasons.append('corpus/tokenizer identity is not shared')
    _n_ok = True
    if n_panels is not None:
        _n = _as_seed_int(n_panels)
        if _n is None or _n < int(min_panels):
            _n_ok = False
            reasons.append(
                f'fewer than {int(min_panels)} pooled panels '
                f'({n_panels!r})')
    # `meta.poolable` is the historical default gate (I2<=50, Q p>=alpha).
    # A caller may declare a stricter synthesis policy; the claim must obey
    # that policy too, otherwise a contrast labelled `heterogeneous` could
    # still carry `claim_allowed=True`.
    _het_ok = True
    _het = meta.get('heterogeneity') or {}
    if meta.get('status') == 'ok':
        _i2 = _cps_finite_or_none(_het.get('I2_percent'))
        if _i2 is None and _het.get('I2_percent') is not None:
            _het_ok = False
            reasons.append('I2 is not a finite heterogeneity measurement')
        elif _i2 is not None and _i2 > float(i2_ceiling_percent):
            _het_ok = False
            reasons.append(
                f'I2={_i2:.6g}% exceeds the synthesis ceiling '
                f'{float(i2_ceiling_percent):.6g}%')
        _qp = _cps_finite_or_none(_het.get('p'))
        if _qp is None and _het.get('p') is not None:
            _het_ok = False
            reasons.append('Q p-value is not a finite heterogeneity measurement')
        elif _qp is not None and _qp < float(q_alpha):
            _het_ok = False
            reasons.append(
                f'Q p={_qp:.6g} is below the synthesis heterogeneity floor '
                f'{float(q_alpha):.6g}')
    dep = meta.get('seed_dependence')
    if dep == 'disjoint_seed_sets':
        if independence_status == 'independent_replication':
            tier = 'independent_replication'
        elif independence_status in ('new_seeds_only', 'insufficient_panels'):
            tier = 'new_seeds_shared_evaluation'
            reasons.append(
                'training seeds are disjoint but the evaluation slices are '
                f'shared or unattested ({independence_status})')
        else:
            tier = 'unattested_evaluation'
            reasons.append(
                'training seeds are disjoint but panel independence is '
                f'unattested ({independence_status})')
    elif dep == 'shared_seeds_paired':
        tier = 'paired_same_seeds'
        reasons.append(
            'panels share training seeds; the claim is a paired common effect '
            'on the same draws, not an independent replication')
    elif dep == 'shared_seeds_unpaired':
        tier = 'shared_seeds_unpaired'
        reasons.append('panels share training seeds without a powered paired '
                       'common effect')
    else:
        tier = 'unattested'
        reasons.append('seed identity is unattested')
    # A paired same-seed common effect is a legitimate but different claim;
    # block the *pooled common effect* label but keep the paired readout.  A
    # disjoint-seed family needs its evaluation independence attested.
    allowed = (meta.get('poolable') is True
               and direction_ok is True
               and corpus_ok is True
               and _n_ok is True
               and _het_ok is True
               and dep in ('disjoint_seed_sets', 'shared_seeds_paired')
               and (dep != 'disjoint_seed_sets'
                    or independence_status in ('independent_replication',
                                               'insufficient_panels')))
    if dep == 'shared_seeds_paired':
        allowed = False
    return {'claim_allowed': bool(allowed), 'evidence_tier': tier,
            'reasons': reasons}


def _cps_leave_one_out(points, *, alpha, min_n, full_mean):
    """Re-pool with each panel removed; is the sign dominated by one panel?"""
    out = {}
    if len(points) < 2:
        return out
    # Validate the family once and reuse each point's mean/SD/SE row across
    # the k subsets.  Re-running full validation and per-point NumPy reductions
    # for every omission made this O(k^2) in a large per-seed payload.  The
    # row cache is only valid without work-scale normalisation (ordinary
    # magnitudes), because an omitted extreme point can change the scale.
    _work_points = points
    _row_cache = {}
    _prevalidated = False
    try:
        _validated = _scale_points(points)
        _effect_abs = [abs(float(e)) for _p in _validated
                       for e in _p['effects']]
        _effect_max = max(_effect_abs, default=0.0)
        _effect_min = min((a for a in _effect_abs if a > 0.0), default=0.0)
        _needs_scale = bool(math.isfinite(_effect_max) and _effect_max > 0.0
                            and (_effect_max < _EFFECT_SCALE_FLOOR
                                 or _effect_min > 1e100))
        _work_points = _validated
        _prevalidated = True
        if not _needs_scale:
            _full = _cps_pool(_validated, alpha=alpha, min_n=min_n,
                              pair_detail_limit=0, _prevalidated_points=True)
            if isinstance(_full.get('per_point'), list):
                _row_cache = {r['label']: r for r in _full['per_point']
                              if isinstance(r, dict) and 'label' in r}
    except Exception:
        _work_points = points
        _row_cache = {}
        _prevalidated = False
    for i, p in enumerate(_work_points):
        rest = _work_points[:i] + _work_points[i + 1:]
        if len(rest) < 2:
            out[str(p.get('label'))] = {'status': 'insufficient_panels'}
            continue
        meta = _cps_pool(rest, alpha=alpha, min_n=min_n,
                         pair_detail_limit=0,
                         _prevalidated_points=_prevalidated,
                         _point_row_cache=_row_cache)
        mean = None
        if meta.get('status') == 'ok':
            re_block = meta.get('random_effects') or {}
            fe_block = meta.get('fixed_effect') or {}
            mean = re_block.get('mean')
            if mean is None:
                mean = fe_block.get('mean')
            mean = _cps_finite_or_none(mean)
        out[str(p.get('label'))] = {
            'status': meta.get('status'),
            'pooled_mean': mean,
            'same_sign_as_full': (None if mean is None
                                  or _cps_finite_or_none(full_mean) in (None, 0.0)
                                  else bool(float(mean) * float(full_mean) > 0.0)),
            'poolable': meta.get('poolable'),
        }
    return out


def _cps_selection_readout(points, rows, *, alpha, min_n, predicted_sign):
    """Descriptive all-panel vs verified-only synthesis.

    This is deliberately NOT a second estimate to choose from.  It quantifies
    the direction and size of the bias induced by pooling only panels whose
    p-value happened to clear alpha.
    """
    verified = [p for p, r in zip(points, rows) if r.get('verified') is True]
    out = {'n_all': len(points), 'n_verified': len(verified),
           'selection_on_significance': True,
           'note': ('verified-only pooling conditions on p<alpha and is shown '
                    'only to expose the sign/size of that selection; it is not '
                    'a valid common-effect estimate')}
    if len(verified) < 2:
        out['status'] = 'not_estimable'
        out['reason'] = 'fewer than two verified panels'
        return out
    meta = _cps_pool(verified, alpha=alpha, min_n=min_n,
                     pair_detail_limit=0)
    out['status'] = meta.get('status')
    out['poolable'] = meta.get('poolable')
    out['heterogeneity'] = meta.get('heterogeneity')
    re_block = meta.get('random_effects') or {}
    fe_block = meta.get('fixed_effect') or {}
    out['random_effects'] = re_block
    out['fixed_effect'] = fe_block
    verified_mean = _cps_finite_or_none(re_block.get('mean'))
    if verified_mean is None:
        verified_mean = _cps_finite_or_none(fe_block.get('mean'))
    all_meta = _cps_pool(points, alpha=alpha, min_n=min_n,
                         pair_detail_limit=0)
    all_mean = (_cps_finite_or_none(
                    (all_meta.get('random_effects') or {}).get('mean'))
                if all_meta.get('status') == 'ok' else None)
    if all_mean is None and all_meta.get('status') == 'ok':
        all_mean = _cps_finite_or_none(
            (all_meta.get('fixed_effect') or {}).get('mean'))
    if verified_mean is not None and all_mean is not None:
        # A "benefit" is positive when the effect has the declared sign; the
        # reported effect is signed with-without, so multiply by the sign.
        sign = _cps_sign_or_none(predicted_sign)
        if sign in (-1, 1):
            benefit_all = sign * float(all_mean)
            benefit_verified = sign * float(verified_mean)
            out['benefit_all_panels'] = benefit_all
            out['benefit_verified_only'] = benefit_verified
            difference = _cps_finite_or_none(benefit_verified - benefit_all)
            out['verified_minus_all_benefit'] = difference
            out['verified_only_more_favourable'] = bool(
                benefit_verified > benefit_all)
        else:
            out['verified_minus_all_effect'] = _cps_finite_or_none(
                float(verified_mean) - float(all_mean))
    return out


def cross_panel_synthesis(reports, design=None, *, alpha=0.05, min_n=3,
                          components=None, policy=None):
    """Quantitative, commensurability-gated synthesis of panel effects.

    ``reports`` maps panel name -> a ``methodology_audit`` report.  For every
    declared component and arm contrast, panels that measured the same
    ``(with, without)`` pair contribute their per-seed relative effect
    (percent of the without-arm PPL).  The function then:

      * excludes a panel with a named reason instead of dropping it silently
        (missing deltas/seeds, incomplete pairing, n<min_n, stale aggregate,
        duplicate contrast entry);
      * requires one direction declaration and one corpus/tokenizer identity
        across included panels;
      * reuses `meta_analyze_effects` for fixed/random-effects pooling,
        Cochran Q, I² and the prediction interval;
      * classifies panel dependence (independent draws, shared seeds, shared
        evaluation) and refuses to call a dependent family an independent
        replication;
      * reports a verified-only pooled estimate as a *descriptive selection
        sensitivity*, never as the headline estimate.

    A pooled common-effect claim requires at least three usable panels, low
    heterogeneity (I²<=50%, non-significant Q), complete paired panels,
    declared seed identities, a shared corpus/tokenizer and independently
    attested evaluation slices.  Otherwise the per-panel effects and the
    descriptive pooled value remain visible and the reason is reported.

    It reads only the reports it is given, starts no run, reads no test split
    and changes no p-value, Holm family, `verified` flag or frozen verdict.
    """
    policy = dict(CROSS_PANEL_SYNTHESIS_POLICY) if policy is None else dict(policy)
    alpha = _validated_alpha(alpha, context='cross_panel_synthesis alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    _min_panels = _as_seed_int(policy.get('min_panels_for_pooled_claim', 3))
    if _min_panels is None or _min_panels < 1:
        raise ValueError('policy.min_panels_for_pooled_claim must be a positive integer')
    _i2_cap = _cps_number(policy.get('i2_ceiling_percent', 50.0),
                          'policy.i2_ceiling_percent')
    _q_alpha = _cps_number(policy.get('q_alpha', 0.10), 'policy.q_alpha')
    if _i2_cap < 0.0:
        raise ValueError('policy.i2_ceiling_percent must be >= 0')
    if not (0.0 < _q_alpha < 1.0):
        raise ValueError('policy.q_alpha must be in (0, 1)')
    _raw_max_pair_details = policy.get(
        'max_pair_details', CROSS_PANEL_SYNTHESIS_MAX_PAIR_DETAILS)
    if _raw_max_pair_details is None:
        _max_pair_details = None
    else:
        _max_pair_details = _as_seed_int(_raw_max_pair_details)
        if _max_pair_details is None or _max_pair_details < 0:
            raise ValueError(
                'policy.max_pair_details must be None or a non-negative integer')
    if design is None:
        design = ABLATION_DESIGN
    if not isinstance(design, dict):
        raise ValueError('design must be a mapping of component -> contrast list')
    # JSON object keys are strings, and a mixed int/str panel mapping made
    # `sorted(reports)` raise before any analysis.  Normalise the public
    # boundary once so every downstream label/audit lookup uses the same key.
    design = {str(k): v for k, v in design.items()}
    if not isinstance(reports, dict):
        reports = {}
    else:
        reports = {str(k): _normalise_component_report(v)
                   for k, v in reports.items()}
    if components is None:
        wanted = set(design)
    elif isinstance(components, (str, bytes, bytearray)):
        wanted = {str(components)}
    else:
        try:
            wanted = {str(c) for c in components}
        except TypeError:
            raise ValueError(
                'components must be an iterable of names or a single name, '
                f'got {type(components).__name__}') from None
    out = {'kind': 'cross_panel_synthesis',
           'status': 'not_verifiable_locally',
           'policy': policy, 'alpha': float(alpha), 'min_n': int(min_n),
           'components': {}, 'warnings': [],
           'poolable_components': [], 'heterogeneous_components': [],
           'dependent_components': [], 'selection_sensitive_components': [],
           'note': ('cross-panel synthesis includes every estimable panel, '
                    'regardless of significance; verified-only pooling is '
                    'reported only as a selection-bias sensitivity.  It does '
                    'not change any panel verdict.')}
    if not reports:
        out['reason'] = 'no panel reports were supplied'
        return out
    any_measured = False
    for comp in sorted(design):
        if comp not in wanted:
            continue
        contrasts = design.get(comp) or []
        comp_block = {'contrasts': {}, 'status': 'not_measured'}
        # (with, without) -> list of (panel, entry)
        found = {}
        for panel in sorted(reports):
            rep = reports.get(panel)
            if not isinstance(rep, dict):
                continue
            _comp_map = rep.get('components')
            if not isinstance(_comp_map, dict):
                continue
            entries = _comp_map.get(comp)
            if not isinstance(entries, (list, tuple)):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                w, wo = entry.get('with'), entry.get('without')
                if not isinstance(w, str) or not isinstance(wo, str):
                    continue
                found.setdefault((w, wo), []).append((panel, entry))
        declared_pairs = []
        for spec in contrasts:
            if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                declared_pairs.append((str(spec[0]), str(spec[1])))
        # A component not declared with a machine-readable arm pair is still
        # reported from what the panels actually measured, so an older design
        # table cannot hide a contrast.
        for pair in sorted(found):
            if pair not in declared_pairs:
                declared_pairs.append(pair)
        for pair in sorted(set(declared_pairs)):
            rows = found.get(pair) or []
            key = f'{pair[0]} - {pair[1]}'
            block = {'with': pair[0], 'without': pair[1],
                     'component': comp, 'status': 'not_measured',
                     'claim_allowed': False, 'evidence_tier': None,
                     'panels': {}, 'excluded': {}, 'reasons': []}
            if not rows:
                comp_block['contrasts'][key] = block
                continue
            any_measured = True
            entries = []
            _panel_counts = {}
            for panel, _entry in rows:
                _panel_counts[panel] = int(_panel_counts.get(panel, 0)) + 1
            for panel, entry in rows:
                if _panel_counts.get(panel, 0) > 1:
                    block['excluded'][panel] = (
                        'duplicate contrast entries for the same arm pair in '
                        'one panel')
                    continue
                try:
                    eff = _cps_entry_effect(entry)
                except _CrossPanelSynthesisError as exc:
                    block['excluded'][panel] = str(exc)
                    continue
                if eff['n'] < min_n:
                    block['excluded'][panel] = (
                        f'n={eff["n"]} < min_n={min_n}')
                    continue
                eff['panel'] = panel
                rep = reports.get(panel) or {}
                eff['verified'] = _cps_in_names(
                    rep.get('verified_components'), comp)
                _abl = rep.get('ablation_completeness')
                eff['incomplete_ablation'] = bool(
                    isinstance(_abl, dict) and _cps_in_names(
                        _abl.get('incomplete_components'), comp))
                eff['direction_conflict'] = _cps_in_names(
                    rep.get('direction_conflict_components'), comp)
                eff['practically_harmful'] = _cps_in_names(
                    rep.get('practically_harmful_components'), comp)
                _draw_map = rep.get('component_draws')
                draws = (_draw_map.get(comp)
                         if isinstance(_draw_map, dict) else None)
                if isinstance(draws, dict):
                    # Only the identity fields the independence audit consumes
                    # are retained; a verbatim copy would bloat every report
                    # with the full per-panel record.
                    eff['draws'] = _json_clean({
                        k: draws.get(k) for k in
                        ('seeds', 'val_fps', 'eval_fps', 'holdout_row_fps',
                         'code_values', 'train_knobs_values', 'corpus_fps')
                        if k in draws})
                else:
                    eff['draws'] = {}
                entries.append(eff)
            block['panels'] = {
                e['panel']: {k: e.get(k) for k in
                             ('n', 'seeds', 'effects', 'mean', 'sd', 'se',
                              'verified', 'incomplete_ablation',
                              'direction_conflict', 'practically_harmful',
                              'predicted_sign', 'observed_sign',
                              'capacity_confounded', 'practical_decision',
                              'p_exact_signflip', 'seed_driven', 'draws')}
                for e in entries}
            if not entries:
                block['status'] = 'not_estimable'
                block['reasons'].append('no panel contributes a usable estimate')
                comp_block['contrasts'][key] = block
                continue
            if len(entries) < 2:
                block['status'] = 'insufficient_panels'
                block['reasons'].append('a cross-panel synthesis needs >=2 panels')
            # Corpus identity: one corpus_fp per included panel, shared across
            # panels.  A panel with several corpora inside one contrast is not
            # a single study and is excluded rather than averaged.
            corpus_fps = []
            _kept_entries = []
            for e in entries:
                fps = sorted({_draw_value(x) for x in
                              _identity_values(e['draws'].get('corpus_fps'))
                              if isinstance(x, str) and x.strip()})
                if len(fps) > 1:
                    block['excluded'][e['panel']] = (
                        'panel contrast spans multiple corpus fingerprints')
                    block['panels'].pop(e['panel'], None)
                    continue
                _kept_entries.append(e)
                corpus_fps.append(fps[0] if fps else None)
            # `list.remove` compares by value: two panels can have identical
            # contrast entries but different draw identities, and removing
            # one would silently delete the other panel.  Keep the filter
            # index-aligned with the entries it actually retains.
            entries = _kept_entries
            # Direction is a property of the panels that actually enter the
            # pool.  Recompute it after corpus exclusions: an excluded panel
            # with a missing/opposite declaration must not veto (or supply)
            # the direction of the surviving family.
            preds = sorted({e.get('predicted_sign') for e in entries
                            if e.get('predicted_sign') in (-1, 0, 1)},
                           key=str)
            pred_missing = any(e.get('predicted_sign') not in (-1, 0, 1)
                               for e in entries)
            direction_ok = (bool(entries) and not pred_missing
                            and len(preds) == 1 and preds[0] in (-1, 1))
            block['direction'] = {
                'predicted_signs': preds,
                'missing_or_invalid': bool(pred_missing),
                'consistent': bool(direction_ok),
                'note': ('one declared direction is required; 0/None means the '
                         'hypothesis did not declare one'),
            }
            if entries:
                corpus_ok = (all(x is not None for x in corpus_fps)
                             and len(set(corpus_fps)) == 1)
            else:
                corpus_ok = False
            block['corpus'] = {'fps': corpus_fps,
                               'shared': bool(corpus_ok)}
            if not entries:
                block['status'] = 'not_estimable'
                block['reasons'].append('no panel remains after corpus checks')
                comp_block['contrasts'][key] = block
                continue
            if not direction_ok:
                block['status'] = 'refused_direction_mismatch'
                comp_block['contrasts'][key] = block
                continue
            points = [{'label': e['panel'], 'scale': 1.0,
                       'effects': list(e['effects']), 'seeds': list(e['seeds']),
                       'corpus_fp': (corpus_fps[i] if i < len(corpus_fps) else None),
                       'complete_pairs': True}
                      for i, e in enumerate(entries)]
            block['meta_input'] = {'points': points}
            meta = _cps_pool(points, alpha=alpha, min_n=min_n,
                             pair_detail_limit=_max_pair_details)
            block['meta'] = {
                'status': meta.get('status'),
                'fixed_effect': meta.get('fixed_effect'),
                'random_effects': meta.get('random_effects'),
                'heterogeneity': meta.get('heterogeneity'),
                'prediction_interval': meta.get('prediction_interval'),
                'seed_dependence': meta.get('seed_dependence'),
                'seed_overlap': meta.get('seed_overlap'),
                'paired_common_effect': meta.get('paired_common_effect'),
                'poolable': meta.get('poolable'),
                'corpus_binding': meta.get('corpus_binding'),
                'underpowered_points': meta.get('underpowered_points'),
                'note': meta.get('note') if isinstance(meta, dict) else None,
            }
            if meta.get('status') != 'ok':
                block['status'] = 'not_estimable'
                block['claim_allowed'] = False
                block['evidence_tier'] = 'not_estimable'
                block['reasons'].append(meta.get('reason')
                                        or 'meta-analysis is not estimable')
                comp_block['contrasts'][key] = block
                continue
            # Panel dependence: component_draws are optional; missing identity
            # is unattested, never independent.
            draws = {e['panel']: e['draws'] for e in entries}
            ind = replication_independence_audit(
                draws, pair_detail_limit=_max_pair_details)
            block['independence'] = ind
            claim = _cps_claim(meta, direction_ok=direction_ok,
                               independence_status=ind.get('status'),
                               corpus_ok=corpus_ok,
                               n_panels=len(entries),
                               min_panels=_min_panels,
                               i2_ceiling_percent=_i2_cap,
                               q_alpha=_q_alpha)
            block['claim_allowed'] = claim['claim_allowed']
            block['evidence_tier'] = claim['evidence_tier']
            block['reasons'].extend(claim['reasons'])
            if any(e.get('direction_conflict') for e in entries):
                block['claim_allowed'] = False
                block['reasons'].append(
                    'a panel reports a correction-significant opposite direction')
            if any(e.get('practically_harmful') for e in entries):
                block['claim_allowed'] = False
                block['reasons'].append(
                    'a panel reports removal as practically harmful')
            if any(e.get('incomplete_ablation') for e in entries):
                block['reasons'].append(
                    'at least one included panel has an incomplete ablation; the '
                    'effect is still a measured contrast, not the whole innovation')
            if any(e.get('capacity_confounded') is True for e in entries):
                block['reasons'].append(
                    'at least one panel is capacity-confounded; the synthesis is '
                    'a system-level effect, not a matched-capacity mechanism effect')
            full_mean = _cps_finite_or_none(
                (meta.get('random_effects') or {}).get('mean')
                if meta.get('random_effects') else None)
            if full_mean is None:
                full_mean = _cps_finite_or_none(
                    (meta.get('fixed_effect') or {}).get('mean'))
            block['leave_one_panel_out'] = _cps_leave_one_out(
                points, alpha=alpha, min_n=min_n, full_mean=full_mean)
            if block['leave_one_panel_out']:
                _same = [v.get('same_sign_as_full')
                         for v in block['leave_one_panel_out'].values()]
                block['sign_stable_leave_one_out'] = (
                    None if not _same or any(x is None for x in _same)
                    else bool(all(_same)))
            else:
                block['sign_stable_leave_one_out'] = None
            if policy.get('selection_bias_readout'):
                block['verified_only_sensitivity'] = _cps_selection_readout(
                    points, entries, alpha=alpha, min_n=min_n,
                    predicted_sign=(preds[0] if preds else None))
                _vb = block['verified_only_sensitivity'].get(
                    'verified_only_more_favourable')
                if _vb is True:
                    block['reasons'].append(
                        'the verified-only subset looks more favourable than the '
                        'all-panel estimate (significance selection)')
                    out['selection_sensitive_components'].append(comp)
            het = meta.get('heterogeneity') or {}
            i2 = het.get('I2_percent')
            qp = het.get('p')
            if (i2 is not None and float(i2) > _i2_cap) or (
                    qp is not None and math.isfinite(float(qp))
                    and float(qp) < _q_alpha):
                block['status'] = 'heterogeneous'
                out['heterogeneous_components'].append(comp)
            elif block['claim_allowed']:
                block['status'] = 'poolable'
                out['poolable_components'].append(comp)
            elif len(entries) < _min_panels:
                block['status'] = 'insufficient_panels'
            elif claim['evidence_tier'] in (
                    'paired_same_seeds', 'shared_seeds_unpaired',
                    'new_seeds_shared_evaluation', 'unattested_evaluation',
                    'unattested'):
                block['status'] = 'dependent_draws'
                out['dependent_components'].append(comp)
            else:
                block['status'] = 'not_poolable'
            comp_block['contrasts'][key] = block
        statuses = [b.get('status') for b in comp_block['contrasts'].values()]
        if 'poolable' in statuses:
            comp_block['status'] = 'poolable'
        elif any(s not in (None, 'not_measured', 'not_estimable')
                 for s in statuses):
            comp_block['status'] = 'measured_not_poolable'
        elif 'not_estimable' in statuses:
            comp_block['status'] = 'not_estimable'
        out['components'][comp] = comp_block
    for comp in out['components']:
        for b in out['components'][comp]['contrasts'].values():
            for reason in b.get('reasons') or []:
                msg = f'{comp} [{b.get("with")} - {b.get("without")}]: {reason}'
                if msg not in out['warnings']:
                    out['warnings'].append(msg)
    out['n_components'] = len(out['components'])
    out['n_poolable_components'] = len(set(out['poolable_components']))
    out['n_contrasts'] = sum(len(c.get('contrasts') or {})
                             for c in out['components'].values())
    for list_key in ('poolable_components', 'heterogeneous_components',
                     'dependent_components', 'selection_sensitive_components'):
        out[list_key] = sorted(set(out[list_key]))
    if any_measured:
        out['status'] = 'warn' if out['warnings'] else 'ok'
    else:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no panel carries a controlled per-seed contrast for '
                         'any declared component; the synthesis has nothing to '
                         'pool')
    return out
