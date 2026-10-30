

def factorial_interaction_audit(summary=None, design=None, *, alpha=0.05,
                                min_n=3, recs=None, hypotheses=None,
                                sesoi=None, look_history=None):
    """Paired 2x2 interaction tests for the declared ablation factorials.

    The estimand is the *change* in a paired effect: for each seed,
    ``(log PPL_11 - log PPL_10) - (log PPL_01 - log PPL_00)``.  A positive
    value means the second factor adds less when the first is active
    (sub-additive), a negative value means synergy.  The exact sign-flip test
    and the seed-level confidence interval therefore operate on the
    interaction itself, not on a comparison of two separate significance
    verdicts.

    The interaction family is declared separately and is EXPLORATORY: it never
    changes the leave-one-out or frozen confirmatory Holm families.  Missing
    or uncontrolled factorials keep a dummy p=1 slot, so an incomplete 2x2
    panel cannot shrink the correction and buy significance.

    ``sesoi`` maps factorial name -> pre-registered margin (default
    ``INTERACTION_SESOI``; pass ``{}`` to disable the practical layer).  When
    a margin exists, a paired TOST on the interaction gets its own Holm family
    and yields an ``additive`` verdict only if the interval lies inside the
    margin.  A non-significant interaction without that test stays
    ``indeterminate``; it is never evidence of additivity.

    ``look_history`` (a ``look_history.json`` object) applies the same exact
    max-over-looks correction to the interaction family: a factorial measured
    after seed top-ups is judged by the sequential sign-flip p, not only by
    the final-n p, and the equivalence (TOST) p is Bonferroni-corrected over
    looks.  Without a history the historical final-look behaviour is kept.
    """
    alpha = _validated_alpha(alpha, context='factorial_interaction_audit alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'factorial_interaction_audit: min_n must be a non-negative integer, got {min_n!r}')
    min_n = int(_min_n)
    design = FACTORIAL_DESIGN if design is None else design
    hypotheses = (INTERACTION_HYPOTHESIS_DESIGN
                  if hypotheses is None else hypotheses)
    try:
        sesoi_specs = _normalise_interaction_sesoi_design(
            INTERACTION_SESOI if sesoi is None else sesoi)
        sesoi_error = None
    except ValueError as e:
        sesoi_specs = {}
        sesoi_error = f'{type(e).__name__}: {e}'
    _look_series = looks_by_contrast(look_history or {})
    _look_invalid = ((look_history or {}).get('invalid')
                     if isinstance(look_history, dict) else
                     (None if look_history is None else
                      f'look_history is a {type(look_history).__name__}, not an object'))
    empty_extra = {'equivalent_interactions': [],
                   'hypothesis_supported_interactions': [],
                   'sub_sesoi_interactions': [],
                   'indeterminate_interactions': [],
                   'not_measured_interactions': [],
                   'not_declared_interactions': [],
                   'look_adjusted_interactions': [],
                   'downgraded_interactions': [],
                   'look_history': {'n_look_adjusted': 0,
                                    'invalid': _look_invalid},
                   'sesoi': {'declared': sorted(sesoi_specs), 'error': sesoi_error}}
    if not isinstance(design, dict) or not design:
        return {'kind': 'factorial_interaction', 'status': 'not_applicable',
                'factorials': {}, 'family': {},
                'significant_interactions': [],
                'direction_conflict_interactions': [], 'warnings': [],
                'note': 'no factorial design is declared', **empty_extra}
    if recs is None:
        recs, _amb = _variant_seed_map(summary or {})
    startup_warnings = []
    if not isinstance(recs, dict):
        return {'kind': 'factorial_interaction', 'status': 'not_applicable',
                'factorials': {}, 'family': {},
                'significant_interactions': [],
                'direction_conflict_interactions': [],
                'warnings': [f'records must be a mapping, got {type(recs).__name__}'],
                'note': 'no usable factorial records were supplied', **empty_extra}
    present = set()
    for _key in recs:
        if not (isinstance(_key, tuple) and len(_key) == 3):
            startup_warnings.append(
                f'record key {_key!r} is not a (variant, protocol, seed) triple; ignored')
            continue
        try:
            present.add(_key[0])
        except TypeError:
            startup_warnings.append(
                f'record key {_key!r} has an unhashable arm id; ignored')
    arm_sets = []
    for _name, _spec in design.items():
        if not isinstance(_spec, dict):
            continue
        _cells = _spec.get('cells')
        if not isinstance(_cells, dict):
            continue
        _raw = [_cells.get(k) for k in _FACTORIAL_CELL_KEYS]
        if any(not isinstance(_a, str) or not _a.strip() for _a in _raw):
            continue
        _arms = [a.strip() for a in _raw]
        if len(set(_arms)) == 4:
            arm_sets.append(set(_arms))
    all_arms = set().union(*arm_sets) if arm_sets else set()
    shared = set.intersection(*arm_sets) if arm_sets else set()
    if arm_sets and not (present & all_arms):
        return {'kind': 'factorial_interaction', 'status': 'not_applicable',
                'factorials': {}, 'family': {},
                'significant_interactions': [],
                'direction_conflict_interactions': [],
                'warnings': startup_warnings,
                'note': ('this panel contains none of the arms declared by a '
                         '2x2 factorial; no interaction is measured here'),
                **empty_extra}
    out = {'kind': 'factorial_interaction', 'status': 'ok',
           'alpha': float(alpha), 'min_n': min_n, 'factorials': {},
           'family': {}, 'significant_interactions': [],
           'direction_conflict_interactions': [],
           'equivalent_interactions': [],
           'hypothesis_supported_interactions': [],
           'sub_sesoi_interactions': [],
           'indeterminate_interactions': [],
           'not_measured_interactions': [],
           'not_declared_interactions': [],
           'look_adjusted_interactions': [],
           'downgraded_interactions': [],
           'look_history': {
               'n_look_adjusted': 0,
               'invalid': _look_invalid,
               'policy': ('each new seed set aggregated for this panel is one '
                          'look; the interaction family uses the exact '
                          'max-over-looks sign-flip p and the TOST p is '
                          'Bonferroni-corrected over looks'),
           },
           'sesoi': {
               'declared': sorted(sesoi_specs),
               'error': sesoi_error,
               'policy': {
                   'scale': ('natural-log ratio-of-ratios; relative_pct p is '
                             'converted to log1p(p/100)'),
                   'test': ('paired TOST at alpha over the interaction '
                            'family; Holm-adjusted TOST p required'),
                   'non_significant_is_not_additivity': True,
                   'sub_sesoi_is_not_zero': True,
                   'equivalence_scope': ('practical additivity inside the '
                                         'declared margin, not exact zero '
                                         'interaction'),
               },
           },
           'warnings': list(startup_warnings),
           'claim_level': ('exploratory interaction family; never part of the '
                           'leave-one-out or frozen confirmatory correction'),
           'note': ('positive interaction = the second factor adds less when the '
                    'first is active (sub-additive); negative = synergy. A '
                    'non-significant interaction is not evidence that the '
                    'effects are additive; only a pre-registered SESOI + TOST '
                    'can support an additive_within_sesoi verdict.')}
    if sesoi_error is not None:
        out['status'] = 'fail'
        out['warnings'].append(
            f'interaction SESOI is malformed; no practical interaction claim '
            f'is allowed: {sesoi_error}')
    if _look_invalid:
        out['warnings'].append(
            f'interaction look history is unreadable or malformed '
            f'({_look_invalid}); the optional-stopping correction cannot be '
            'verified for the interaction family — do not read its naive p as '
            'look-adjusted')
    family = []
    family_naive = []
    tost_family = []
    interaction_values = {}

    def _unusable_factorial(name, reason):
        sname = str(name)
        rec = {'factors': None, 'cells': None, 'n_present_cells': 0,
               'status': 'incomplete',
               'interaction': {'available': False, 'reason': reason}}
        out['factorials'][sname] = rec
        out['warnings'].append(
            f'{name}: {reason}; its family slot is kept at p=1')
        family.append((sname, 1.0))
        family_naive.append((sname, 1.0))
        _sesoi_spec = sesoi_specs.get(sname)
        _margin = (_sesoi_spec.get('margin_log_ratio')
                   if isinstance(_sesoi_spec, dict) else None)
        if _margin is None:
            rec['practical_decision'] = 'not_declared'
            rec['sesoi'] = {
                'declared': False, 'margin_log_ratio': None,
                'decision': 'not_declared', 'claim': 'not_declared',
                'reason': 'no pre-registered interaction SESOI was supplied',
            }
            if sesoi_specs:
                tost_family.append((sname, 1.0))
        else:
            rec['practical_decision'] = 'not_measured'
            rec['sesoi'] = {
                'declared': True,
                'margin_log_ratio': float(_margin),
                'decision': 'not_measured', 'claim': 'not_measured',
                'reason': ('the factorial declaration is malformed; '
                           'additivity cannot be certified'),
            }
            tost_family.append((sname, 1.0))

    for name, spec in design.items():
        if not isinstance(spec, dict):
            _unusable_factorial(name, 'factorial declaration is not a mapping')
            continue
        cells = spec.get('cells')
        factors = spec.get('factors')
        if not isinstance(cells, dict):
            _unusable_factorial(name, 'cells must be a mapping')
            continue
        raw_arms = [cells.get(k) for k in _FACTORIAL_CELL_KEYS]
        if any(not isinstance(a, str) or not a.strip() for a in raw_arms):
            _unusable_factorial(
                name, 'the four cells must map to non-empty, hashable arm ids')
            continue
        arms = [a.strip() for a in raw_arms]
        extra_cells = sorted(str(k) for k in cells if k not in _FACTORIAL_CELL_KEYS)
        if extra_cells:
            out['warnings'].append(
                f'{name}: cell declaration has extra key(s): '
                + ', '.join(extra_cells))
        if len(set(arms)) != 4:
            _unusable_factorial(name, 'the four cells are not distinct arms')
            continue
        present_arms = set(arms) & present
        if not present_arms:
            continue
        # A partial panel that only contains arms shared by every declared
        # factorial does not identify THIS factorial.  A complete four-arm
        # panel does, and a panel with a design-specific arm is reported as
        # incomplete.  Comparing against the global intersection (rather than
        # skipping whenever all present arms are shared) keeps a single-design
        # audit measurable instead of falsely returning not_applicable.  With
        # exactly one declared factorial there is no ambiguity to resolve, so
        # even a partial panel must be reported as incomplete (and keep its
        # p=1 family slot) rather than silently vanish.
        if (len(present_arms) < 4 and present_arms <= shared
                and len(arm_sets) > 1):
            continue
        norm_cells = {k: arms[i] for i, k in enumerate(_FACTORIAL_CELL_KEYS)}
        rec = {'factors': list(factors) if isinstance(factors, (list, tuple)) else None,
               'cells': {f'{k[0]}|{k[1]}': norm_cells[k]
                         for k in _FACTORIAL_CELL_KEYS},
               'n_present_cells': len(present_arms), 'status': 'incomplete',
               'interaction': {'available': False}}
        out['factorials'][str(name)] = rec
        _sesoi_spec = sesoi_specs.get(str(name))
        _margin = (_sesoi_spec.get('margin_log_ratio')
                   if isinstance(_sesoi_spec, dict) else None)
        if len(present_arms) < 4:
            missing = sorted(set(arms) - present_arms)
            rec['missing_arms'] = missing
            rec['interaction']['reason'] = 'missing arm(s): ' + ', '.join(missing)
            out['warnings'].append(
                f'{name}: 2x2 interaction is incomplete (missing '
                + ', '.join(missing) + '); its family slot is kept at p=1')
            family.append((str(name), 1.0))
            family_naive.append((str(name), 1.0))
            if _margin is None:
                rec['practical_decision'] = 'not_declared'
                rec['sesoi'] = {
                    'declared': False, 'margin_log_ratio': None,
                    'decision': 'not_declared', 'claim': 'not_declared',
                    'reason': 'no pre-registered interaction SESOI was supplied',
                }
                if sesoi_specs:
                    tost_family.append((str(name), 1.0))
            else:
                rec['practical_decision'] = 'not_measured'
                rec['sesoi'] = {
                    'declared': True,
                    'margin_log_ratio': float(_margin),
                    'decision': 'not_measured', 'claim': 'not_measured',
                    'reason': 'missing 2x2 cell(s); additivity cannot be certified',
                }
                tost_family.append((str(name), 1.0))
            continue
        rows, skipped, param = _factorial_condition_rows(recs, norm_cells)
        rec['n_complete'] = len(rows)
        rec['seeds'] = [r['seed'] for r in rows]
        rec['dropped'] = skipped
        rec['param_control'] = param
        if not rows:
            rec['status'] = 'no_data'
            rec['interaction']['reason'] = (
                'no seed has four controlled cells with a usable primary PPL')
            out['warnings'].append(f'{name}: no complete 2x2 seed remains; '
                                   'its family slot is kept at p=1')
            family.append((str(name), 1.0))
            family_naive.append((str(name), 1.0))
            if _margin is None:
                rec['practical_decision'] = 'not_declared'
                rec['sesoi'] = {
                    'declared': False, 'margin_log_ratio': None,
                    'decision': 'not_declared', 'claim': 'not_declared',
                    'reason': 'no pre-registered interaction SESOI was supplied',
                }
                if sesoi_specs:
                    tost_family.append((str(name), 1.0))
            else:
                rec['practical_decision'] = 'not_measured'
                rec['sesoi'] = {
                    'declared': True,
                    'margin_log_ratio': float(_margin),
                    'decision': 'not_measured', 'claim': 'not_measured',
                    'reason': 'no controlled 2x2 seed; additivity cannot be certified',
                }
                tost_family.append((str(name), 1.0))
            continue
        st = paired_effect_stats([r['interaction_log_ppl'] for r in rows],
                                 alpha=alpha, min_n=min_n,
                                 seeds=[r['seed'] for r in rows])
        cell = {k: st.get(k) for k in
                ('n', 'n_input', 'n_dropped', 'dropped_seeds', 'complete_pairs',
                 'seeds', 'mean', 'std', 'std_defined', 'sem', 'cohens_dz',
                 'ci95_low', 'ci95_high', 'p_exact_signflip', 'p_floor',
                 'underpowered', 'resolution_limited', 'within_seed_noise',
                 'significant', 'effect_vs_noise', 'seed_driven', 'seed_robust')}
        m = cell.get('mean')
        if isinstance(m, (int, float)) and math.isfinite(float(m)) and -700.0 < float(m) < 700.0:
            cell['ratio_of_ratios'] = float(math.exp(float(m)))
            cell['percent_change'] = float(100.0 * (math.exp(float(m)) - 1.0))
        else:
            cell['ratio_of_ratios'] = None
            cell['percent_change'] = None
        simple = {}
        for key in rows[0]['simple_log_effects']:
            vals = [float(r['simple_log_effects'][key]) for r in rows]
            mean = float(np.mean(vals))
            simple[key] = {'mean_log_effect': mean,
                           'ratio': float(math.exp(mean)) if -700.0 < mean < 700.0 else None,
                           'n': len(vals),
                           'role': 'descriptive simple effect (not tested here)'}
        rec['simple_effects'] = simple
        hyp = hypotheses.get(str(name)) if isinstance(hypotheses, dict) else None
        pred = hyp.get('predicted_sign') if isinstance(hyp, dict) else None
        obs = _sign_of(st.get('mean')) if st.get('n') else None
        cell.update({
            'available': True,
            'definition': ('(log PPL_11 - log PPL_10) - '
                           '(log PPL_01 - log PPL_00); positive = the second '
                           'factor adds less when the first is active'),
            'hypothesis': hyp.get('statement') if isinstance(hyp, dict) else None,
            'predicted_sign': pred,
            'observed_sign': obs,
            'direction_consistent': (None if pred not in (-1, 0, 1) or obs is None
                                     else (pred == 0 or pred == obs)),
            'p_holm': None,
            'p_tost_holm': None,
            'significant_family': False,
            'role': 'exploratory interaction (separate Holm family)',
        })
        if param.get('capacity_controlled') is not True:
            cell['mechanism_claim_allowed'] = False
            cell['capacity_confound'] = (
                'unverifiable' if param.get('capacity_controlled') is None
                else 'parameter/breakdown mismatch')
            out['warnings'].append(
                f'{name}: the four cells do not clear the capacity control '
                f'({cell["capacity_confound"]}); the interaction is reported but '
                'cannot back a mechanism claim')
        else:
            cell['mechanism_claim_allowed'] = True
        rec['interaction'] = cell
        rec['status'] = 'ok'
        _vals = [float(r['interaction_log_ppl']) for r in rows]
        interaction_values[str(name)] = _vals
        _look_key = _interaction_look_key(str(name))
        _cur_cells = {f'{k[0]}|{k[1]}': norm_cells[k]
                      for k in _FACTORIAL_CELL_KEYS}
        _cur_tags = {str(r['seed']): str(r.get('tag') or '') for r in rows}
        _seq, _stale, _n_series = _interaction_look_analysis(
            _look_series, _look_key, [r['seed'] for r in rows], _vals,
            alpha=alpha, final_tags=_cur_tags, final_cells=_cur_cells)
        cell['look_analysis'] = _seq
        cell['look_adjusted'] = bool((_seq or {}).get('look_adjusted'))
        cell['n_looks'] = int((_seq or {}).get('n_looks') or 0)
        _adj_p = ((_seq or {}).get('p_sequential')
                  if isinstance(_seq, dict) else st.get('p_exact_signflip'))
        if not (isinstance(_adj_p, (int, float))
                and math.isfinite(float(_adj_p))):
            _adj_p = 1.0
        cell['p_look_adjusted'] = float(_adj_p)
        _raw_p = (float(st['p_exact_signflip'])
                  if (isinstance(st.get('p_exact_signflip'), (int, float))
                      and math.isfinite(float(st['p_exact_signflip'])))
                  else 1.0)
        if _stale:
            out['warnings'].append(
                f'{name}: {len(_stale)} recorded interaction look(s) are STALE '
                '(a seed delta, cell arm, or protocol tag changed after '
                're-running); they were excluded from the sequential test, so '
                'the correction covers only the remaining '
                f'{int((_seq or {}).get("n_looks") or 1)} look(s)')
        family.append((str(name), float(_adj_p)))
        family_naive.append((str(name), _raw_p))
        if cell['look_adjusted']:
            out['look_adjusted_interactions'].append(str(name))
            if _raw_p < alpha <= float(_adj_p):
                out['downgraded_interactions'].append(str(name))
                out['warnings'].append(
                    f'{name}: the naive final-look interaction p={_raw_p:.4g} '
                    f'would be significant, but the exact max-over-looks p '
                    f'over {cell["n_looks"]} looks is {float(_adj_p):.4g} '
                    f'(>= alpha={alpha:.3g}); a seed top-up does not make the '
                    'interaction confirmable')
        if _sesoi_spec is None:
            rec['practical_decision'] = 'not_declared'
            cell['practical_decision'] = 'not_declared'
            cell['sesoi'] = {
                'declared': False, 'margin_log_ratio': None,
                'decision': 'not_declared', 'claim': 'not_declared',
                'reason': 'no pre-registered interaction SESOI was supplied',
            }
            if sesoi_specs:
                tost_family.append((str(name), 1.0))
        else:
            _dir = pred if pred in (-1, 0, 1) else 0
            _pd = interaction_practical_decision(
                _vals, margin=float(_margin), alpha=alpha, min_n=min_n,
                direction=_dir, label=str(name))
            _pt = _pd.get('tost_p_tost')
            _pt_adj = (float(_pt) if (isinstance(_pt, (int, float))
                                      and math.isfinite(float(_pt)))
                       else 1.0)
            _k_looks = int(cell['n_looks'] or 1)
            if _k_looks > 1:
                # Equivalence is a positive claim and is equally exposed to
                # optional stopping; Bonferroni over the spent looks is valid
                # and conservative.
                _pt_adj = min(1.0, _pt_adj * _k_looks)
            _pd['p_tost_look_adjusted'] = _pt_adj
            _pd['n_looks'] = _k_looks
            rec['practical_decision'] = _pd['decision']
            cell['sesoi'] = _pd
            cell['practical_decision'] = _pd['decision']
            tost_family.append((str(name), _pt_adj))
    if not out['factorials']:
        if out['status'] != 'fail':
            out['status'] = 'incomplete' if out['warnings'] else 'not_applicable'
        if not out['warnings']:
            out['note'] = ('no complete factorial declaration could be measured '
                           'from the supplied records')
        return out
    if family:
        h = holm_adjust(family, alpha=alpha)
        hn = holm_adjust(family_naive, alpha=alpha) if family_naive else {}
        floors = []
        for name, _p in family:
            rec = out['factorials'].get(name) or {}
            cell = rec.get('interaction') or {}
            floors.append(cell.get('p_floor') if cell.get('available') else 1.0)
        out['family'] = {
            'name': 'declared 2x2 factorial interactions (exploratory; not pre-registered as confirmatory)',
            'm': len(family),
            'alpha': float(alpha),
            'min_seeds_for_family_significance': signflip_min_n_for_holm(len(family), alpha),
            'min_attainable_p_holm': holm_min_attainable_p_floors(floors, alpha=alpha),
            'n_look_adjusted': len(out['look_adjusted_interactions']),
            'p_adjustment': ('exact max-over-looks sign-flip p per interaction; '
                             'naive final-look p kept as p_holm_naive'),
            'note': ('missing/uncontrolled factorials keep a dummy p=1 slot; '
                     'this family never changes the leave-one-out or frozen '
                     'confirmatory correction'),
        }
        ht = holm_adjust(tost_family, alpha=alpha) if tost_family else {}
        out['sesoi']['family'] = {
            'm': len(tost_family),
            'alpha': float(alpha),
            'method': ('Holm step-down over the declared interaction TOSTs; '
                       'missing interactions keep a p=1 slot'),
            # TOST is a t-approximation and has no 2/2**n exact floor.
            'min_attainable_p_holm': None,
        }
        for name, _p in family:
            rec = out['factorials'].get(name) or {}
            cell = rec.get('interaction') or {}
            if not cell.get('available'):
                continue
            cell['p_holm'] = (h.get(name) or {}).get('p_holm')
            cell['p_holm_naive'] = (hn.get(name) or {}).get('p_holm')
            cell['p_tost_holm'] = (ht.get(name) or {}).get('p_holm')
            cell['significant_family'] = bool(
                cell.get('significant')
                and isinstance(cell.get('p_holm'), (int, float))
                and float(cell['p_holm']) < alpha)
            cell['significant_family_naive'] = bool(
                cell.get('significant')
                and isinstance(cell.get('p_holm_naive'), (int, float))
                and float(cell['p_holm_naive']) < alpha)
            if cell['significant_family']:
                out['significant_interactions'].append(name)
                if cell.get('direction_consistent') is False:
                    out['direction_conflict_interactions'].append(name)
            _spec = sesoi_specs.get(name)
            _vals = interaction_values.get(name)
            if _spec is not None and _vals:
                _pred = cell.get('predicted_sign')
                _dir = _pred if _pred in (-1, 0, 1) else 0
                _prev = cell.get('sesoi') or {}
                _pd = interaction_practical_decision(
                    _vals, margin=float(_spec['margin_log_ratio']),
                    alpha=alpha, min_n=min_n, direction=_dir,
                    p_holm=cell.get('p_holm'),
                    p_holm_tost=cell.get('p_tost_holm'),
                    label=name)
                _pd['p_tost_look_adjusted'] = _prev.get('p_tost_look_adjusted')
                _pd['n_looks'] = _prev.get('n_looks')
                rec['practical_decision'] = _pd['decision']
                cell['sesoi'] = _pd
                cell['practical_decision'] = _pd['decision']
    out['look_history']['n_look_adjusted'] = len(
        out['look_adjusted_interactions'])
    out['look_history']['look_adjusted_contrasts'] = list(
        out['look_adjusted_interactions'])
    out['look_history']['downgraded_naive_significant'] = list(
        out['downgraded_interactions'])
    # One classification pass over every declared factorial avoids duplicate
    # list entries across complete, incomplete and no-data branches.
    for name, rec in out['factorials'].items():
        _pd = rec.get('practical_decision')
        if _pd == 'additive':
            out['equivalent_interactions'].append(name)
        elif _pd == 'hypothesis_supported':
            out['hypothesis_supported_interactions'].append(name)
        elif _pd == 'sub_sesoi':
            out['sub_sesoi_interactions'].append(name)
        elif _pd == 'not_measured':
            out['not_measured_interactions'].append(name)
        elif _pd == 'not_declared':
            out['not_declared_interactions'].append(name)
        elif _pd == 'direction_conflict':
            # Already listed in direction_conflict_interactions (family-sig).
            pass
        else:
            out['indeterminate_interactions'].append(name)
    if (out['status'] != 'fail'
            and any(rec.get('status') != 'ok'
                    for rec in out['factorials'].values())):
        out['status'] = 'incomplete'
    return out



# ---- 预注册剂量/有序水平设计：dose-response 不是端点对比 --------------------
# 动机：机制面板经常扫 top-k、块大小、distractor 比例等有序水平。只挑两个
# 端点做配对检验，既丢掉了顺序信息，也把「事后选出的端点」伪装成
# dose-response。下面的门在**定义点**要求：至少三个水平、剂量单调、只改变
# 一个声明键、所有水平使用同一批种子与同一测量协议，并且方向/形状在读数
# 前声明。记录级检验则对每个种子的斜率做一次精确符号翻转；相邻水平方向
# 只作为形状门槛，不把多个相邻配对当成多个独立假设。

_DOSE_DIRECTION_SIGNS = {'increasing': 1, 'decreasing': -1}
_DOSE_RESPONSE_SHAPES = ('linear', 'monotone')


def _dose_direction_sign(direction, *, context='dose direction'):
    """Map a pre-registered ordered alternative to +1/-1.

    Only the canonical spellings are accepted (case/whitespace normalised).
    An 'auto' or post-hoc direction is deliberately not representable: the
    direction is part of the design, not a choice made after seeing the trend.
    """
    if not isinstance(direction, str):
        raise ValueError(f'{context} must be "increasing" or "decreasing", got {direction!r}')
    key = direction.strip().lower()
    if key not in _DOSE_DIRECTION_SIGNS:
        raise ValueError(
            f'{context} must be "increasing" or "decreasing", got {direction!r}')
    return _DOSE_DIRECTION_SIGNS[key]


def _dose_shape(shape, *, context='dose-response shape'):
    if not isinstance(shape, str):
        raise ValueError(f'{context} must be one of {list(_DOSE_RESPONSE_SHAPES)}, got {shape!r}')
    key = shape.strip().lower()
    if key not in _DOSE_RESPONSE_SHAPES:
        raise ValueError(
            f'{context} must be one of {list(_DOSE_RESPONSE_SHAPES)}, got {shape!r}')
    return key


def _dose_seed_list(raw, *, context):
    """Normalise a declared seed list, naming invalid/duplicate entries."""
    if isinstance(raw, (str, bytes, bytearray, dict)):
        raise ValueError(f'{context} must be an iterable of integer seeds, not {type(raw).__name__}')
    try:
        items = list(raw)
    except TypeError:
        raise ValueError(f'{context} must be an iterable of integer seeds') from None
    out, seen = [], set()
    for x in items:
        s = _as_seed_int(x)
        if s is None:
            raise ValueError(f'{context} contains a non-integer seed {x!r}')
        if s in seen:
            raise ValueError(f'{context} contains duplicate seed {s!r}')
        seen.add(s)
        out.append(s)
    return sorted(out)


def _normalise_dose_levels(levels, *, require_arm, context='dose_response'):
    """Normalise mapping/list level declarations to one canonical list.

    A mapping is ``label -> {dose, arm, seeds, protocol, values}``.  A list
    accepts the same objects plus an explicit ``label`` (falling back to the
    list index).  Structural errors raise; the public audit catches them and
    reports fail-closed rather than dropping a malformed level.
    """
    if levels is None:
        raise ValueError(f'{context}: levels must be supplied')
    rows = []
    if isinstance(levels, dict):
        items = list(levels.items())
        for label, spec in items:
            if not isinstance(spec, dict):
                raise ValueError(f'{context}: level {label!r} must be an object, '
                                 f'got {type(spec).__name__}')
            rec = dict(spec)
            rec.setdefault('label', label)
            rows.append(rec)
    elif isinstance(levels, (list, tuple)):
        for i, spec in enumerate(levels):
            if not isinstance(spec, dict):
                raise ValueError(f'{context}: level #{i} must be an object, '
                                 f'got {type(spec).__name__}')
            rec = dict(spec)
            rec.setdefault('label', str(i))
            rows.append(rec)
    else:
        raise ValueError(f'{context}: levels must be a mapping or list, '
                         f'got {type(levels).__name__}')
    out, seen_labels, seen_doses = [], set(), set()
    for i, rec in enumerate(rows):
        label = rec.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f'{context}: level #{i} needs a non-empty string label')
        label = label.strip()
        if label in seen_labels:
            raise ValueError(f'{context}: duplicate level label {label!r}')
        seen_labels.add(label)
        dose = _finite_scalar(rec.get('dose'))
        if dose is None:
            raise ValueError(f'{context}[{label}]: dose must be a finite number, '
                             f'got {rec.get("dose")!r}')
        if dose in seen_doses:
            raise ValueError(f'{context}: duplicate dose {dose!r}')
        seen_doses.add(dose)
        arm = rec.get('arm')
        if require_arm:
            if not isinstance(arm, str) or not arm.strip():
                raise ValueError(f'{context}[{label}]: arm must be a non-empty string')
            arm = arm.strip()
        elif arm is not None:
            arm = str(arm).strip()
            if not arm:
                raise ValueError(f'{context}[{label}]: arm must be non-empty when supplied')
        seeds = None
        if rec.get('seeds') is not None:
            seeds = _dose_seed_list(rec['seeds'], context=f'{context}[{label}].seeds')
        protocol = None
        if rec.get('protocol') is not None:
            _raw_protocol = rec['protocol']
            protocol = (_raw_protocol if isinstance(_raw_protocol, str)
                        else _canon_setting(_raw_protocol))
        out.append({'label': label, 'dose': float(dose), 'arm': arm,
                    'seeds': seeds, 'protocol': protocol,
                    'values': rec.get('values')})
    return out


def _dose_arm_value(arm, factory, n_layers, dose_key):
    """Extract one scalar dose-key value from every constructed layer.

    Returns ``(value, error)``.  A key absent on some layer, or a key that
    varies across layers, cannot define a single declared dose for the arm.
    """
    try:
        cfgs = factory(n_layers, arm)
        if isinstance(cfgs, (str, bytes, bytearray)):
            return None, 'factory returned a string, not a sequence of layer configs'
        cfgs = list(cfgs)
        if len(cfgs) != int(n_layers):
            return None, f'factory returned {len(cfgs)} layers, expected {n_layers}'
    except Exception as e:
        return None, f'arm construction failed ({type(e).__name__}: {e})'
    vals = []
    for i, cfg in enumerate(cfgs):
        try:
            if isinstance(cfg, dict):
                val = cfg.get(dose_key)
            else:
                val = getattr(cfg, dose_key, None)
        except Exception as e:
            return None, f'cannot read {dose_key!r} on layer {i} ({type(e).__name__}: {e})'
        f = _finite_scalar(val)
        if f is None:
            return None, f'{dose_key!r} is missing/non-numeric on layer {i}'
        vals.append(f)
    if not vals:
        return None, 'factory returned no layers'
    if len(set(vals)) != 1:
        return None, f'{dose_key!r} varies across layers ({vals!r})'
    return vals[0], None



def _dose_pair_cfg_keys(variant_a, variant_b, factory, n_layers):
    """Config diff for the ordered-ladder audit, including plain dict configs.

    The rest of the single-factor audits use ``layer_cfg_diff`` directly, which
    is dataclass-only.  A zero-GPU design declaration is often a small literal
    factory, so accepting both dataclasses and mappings keeps the dose gate
    usable without weakening the real-arm path.
    """
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
        per_layer = {}
        keys = set()
        for i, (x, y) in enumerate(zip(cfgs_a, cfgs_b)):
            if isinstance(x, dict) and isinstance(y, dict):
                d = {}
                for k in set(x) | set(y):
                    if k not in x or k not in y:
                        d[k] = (x.get(k), y.get(k))
                    elif _cfg_values_differ(x[k], y[k]):
                        d[k] = (x[k], y[k])
            else:
                d = layer_cfg_diff(x, y)
            if d:
                per_layer[i] = d
                keys.update(d)
    except Exception as e:
        return e, {}
    return sorted(keys), per_layer


def dose_response_design_audit(levels, *, direction, dose_key, factory=None,
                               n_layers=6, min_levels=3, min_seeds=3,
                               require_seeds=True, require_protocol=False,
                               shape='linear'):
    """Zero-GPU gate for a pre-registered ordered (dose-response) ladder.

    A trend test is only interpretable if the ordered levels are a single-axis
    manipulation of the same units under the same measurement protocol.  The
    gate constructs every declared arm through the module factory and checks:

      * at least three distinct, finite, sorted doses;
      * every level names a distinct arm and the arm diff from the baseline is
        exactly ``{dose_key}`` (not a bundle, not an absent intervention);
      * the numeric ``dose_key`` values are unique and monotonic in the declared
        dose order (inverse coding is allowed, a scrambled ladder is not);
      * every level declares the same seed set, with at least ``min_seeds``
        (by default a declaration must carry them; a legacy call may opt out);
      * if any level declares a protocol identity, every level declares the
        same one; ``require_protocol`` makes that mandatory.

    This does not test the outcome and cannot verify that the direction was
    written before the first look; it makes the declared design machine-checked
    so a post-hoc endpoint pair cannot be relabelled as a dose-response.
    """
    out = {'kind': 'dose_response_design', 'status': 'ok', 'direction': direction,
           'direction_sign': None, 'shape_hypothesis': shape,
           'dose_key': dose_key, 'dose_key_values': {},
           'n_levels': 0, 'labels': [], 'levels': {}, 'violations': [],
           'warnings': [], 'seed_identity': None, 'protocol_identity': None,
           'note': ('ordered levels are a design claim: the gate verifies the '
                    'single-axis ladder, not the outcome')}
    try:
        sign = _dose_direction_sign(direction, context='dose_response_design direction')
        out['direction_sign'] = int(sign)
    except ValueError as e:
        out['status'] = 'fail'
        out['violations'].append(str(e))
        sign = None
    try:
        out['shape_hypothesis'] = _dose_shape(
            shape, context='dose_response_design shape')
    except ValueError as e:
        out['status'] = 'fail'
        out['violations'].append(str(e))
    if not isinstance(dose_key, str) or not dose_key.strip():
        out['status'] = 'fail'
        out['violations'].append(
            f'dose_key must be a non-empty AttnCfg key name, got {dose_key!r}')
        dose_key = None
    else:
        dose_key = dose_key.strip()
        out['dose_key'] = dose_key
    _nl = _as_seed_int(n_layers)
    if _nl is None or _nl < 1:
        out['status'] = 'fail'
        out['violations'].append(f'n_layers must be a positive integer, got {n_layers!r}')
        n_layers = None
    else:
        n_layers = _nl
    _ml = _as_seed_int(min_levels)
    if _ml is None or _ml < 3:
        out['status'] = 'fail'
        out['violations'].append(
            f'min_levels must be an integer >= 3 (two points are an endpoint '
            f'contrast, not a dose-response), got {min_levels!r}')
        min_levels = 3
    else:
        min_levels = _ml
    _ms = _as_seed_int(min_seeds)
    if _ms is None or _ms < 1:
        out['status'] = 'fail'
        out['violations'].append(f'min_seeds must be a positive integer, got {min_seeds!r}')
        min_seeds = 1
    else:
        min_seeds = _ms
    if levels is None:
        out['note'] = 'no ordered levels were declared'
        if out['status'] != 'fail':
            out['status'] = 'not_applicable'
        return out
    try:
        rows = _normalise_dose_levels(levels, require_arm=True,
                                      context='dose_response_design')
    except ValueError as e:
        out['status'] = 'fail'
        out['violations'].append(str(e))
        return out
    if not rows:
        out['status'] = 'fail'
        out['violations'].append('dose_response_design: levels is empty')
        return out
    rows = sorted(rows, key=lambda r: r['dose'])
    out['n_levels'] = len(rows)
    out['labels'] = [r['label'] for r in rows]
    out['doses'] = [r['dose'] for r in rows]
    if len(rows) < min_levels:
        out['status'] = 'fail'
        out['violations'].append(
            f'{len(rows)} ordered level(s) declared; at least {min_levels} are '
            'required for a dose-response (use a named endpoint contrast for two)')
    arms = [r['arm'] for r in rows]
    if len(set(arms)) != len(arms):
        out['status'] = 'fail'
        out['violations'].append('the ordered levels must name distinct arms; '
                                 'reusing an arm makes two doses the same manipulation')
    for r in rows:
        out['levels'][r['label']] = {
            'dose': r['dose'], 'arm': r['arm'], 'seeds': r['seeds'],
            'protocol': r['protocol']}
    # Seed identity: a dose-response is paired within seed, so the seed set
    # must be the same at every level.  Missing declarations are a warning for
    # legacy callers and a hard failure for a pre-registered one.
    seed_sets = [r['seeds'] for r in rows]
    if all(s is not None for s in seed_sets):
        first = set(seed_sets[0])
        for r, s in zip(rows, seed_sets):
            if set(s) != first:
                out['status'] = 'fail'
                out['violations'].append(
                    f'{r["label"]}: seed set {s!r} differs from '
                    f'{rows[0]["label"]} {sorted(first)!r}; the ladder must be '
                    'paired within seed')
        out['seed_identity'] = 'declared'
        out['seeds'] = sorted(first)
        out['n_seeds'] = len(first)
        if len(first) < min_seeds:
            out['status'] = 'fail'
            out['violations'].append(
                f'{len(first)} seed(s) declared; at least {min_seeds} are required')
    else:
        if require_seeds:
            out['status'] = 'fail'
            out['violations'].append(
                'every ordered level must declare its seed set (or set '
                'require_seeds=False only for a legacy/descriptive audit)')
        else:
            out['warnings'].append(
                'seed sets were not declared on every level; within-seed pairing '
                'is unverified')
        out['seed_identity'] = 'unverified'
    # Measurement protocol: equal protocol ids prevent a dose ladder from
    # silently changing eval length/checkpoint/recipe between levels.
    protocols = [r['protocol'] for r in rows]
    if all(p is not None for p in protocols):
        first_p = protocols[0]
        for r, p in zip(rows, protocols):
            if p != first_p:
                out['status'] = 'fail'
                out['violations'].append(
                    f'{r["label"]}: protocol identity {p!r} differs from '
                    f'{rows[0]["label"]} {first_p!r}; the trend would mix '
                    'different measurements')
        out['protocol_identity'] = 'declared'
        out['protocols'] = protocols
    elif any(p is not None for p in protocols):
        out['status'] = 'fail'
        out['violations'].append(
            'protocol identity is declared on only some levels; it must be '
            'declared on all or none')
        out['protocol_identity'] = 'partial'
    else:
        if require_protocol:
            out['status'] = 'fail'
            out['violations'].append(
                'require_protocol=True but no level declares a protocol identity')
        else:
            out['warnings'].append(
                'no protocol identity was declared; equal eval/checkpoint/recipe '
                'semantics across doses are unverified')
        out['protocol_identity'] = 'unverified'
    # Config-level intervention fidelity.
    if factory is None:
        factory = make_layer_cfgs
    if not callable(factory):
        out['status'] = 'fail'
        out['violations'].append(f'factory must be callable, got {type(factory).__name__}')
        return out
    if dose_key is not None and n_layers is not None and len(set(arms)) == len(arms):
        baseline = rows[0]
        seen_signatures = {}
        numeric_values = []
        for r in rows:
            value, why = _dose_arm_value(r['arm'], factory, n_layers, dose_key)
            out['dose_key_values'][r['label']] = value if value is not None else None
            if value is None:
                out['violations'].append(
                    f'{r["label"]}: cannot extract a scalar {dose_key!r} '
                    f'({why}); an ordered ladder needs a numeric dose axis')
            else:
                numeric_values.append((r, value))
            if r is baseline:
                continue
            keys, per_layer = _dose_pair_cfg_keys(
                baseline['arm'], r['arm'], factory, n_layers)
            if isinstance(keys, Exception):
                out['violations'].append(
                    f'{r["label"]}: cannot construct the {baseline["arm"]} vs '
                    f'{r["arm"]} contrast ({type(keys).__name__}: {keys})')
                continue
            actual = set(keys)
            if actual != {dose_key}:
                extra = sorted(actual - {dose_key})
                missing = sorted({dose_key} - actual)
                detail = []
                if extra:
                    detail.append(f'extra key(s) {extra}')
                if missing:
                    detail.append(f'{dose_key!r} does not differ')
                out['violations'].append(
                    f'{r["label"]}: the declared dose axis is not isolated '
                    f'(actual AttnCfg diff {sorted(actual)!r}; '
                    + '; '.join(detail) + ')')
            sig = _canon_setting(per_layer)
            if sig in seen_signatures:
                out['violations'].append(
                    f'{r["label"]}: identical intervention signature to '
                    f'{seen_signatures[sig]}; the declared doses are not distinct')
            else:
                seen_signatures[sig] = r['label']
            out['levels'][r['label']]['cfg_diff_keys'] = sorted(actual)
            out['levels'][r['label']]['per_layer'] = {
                str(i): {k: [v0, v1] for k, (v0, v1) in d.items()}
                for i, d in per_layer.items()}
        if len(numeric_values) == len(rows):
            if len({v for _r, v in numeric_values}) != len(numeric_values):
                out['status'] = 'fail'
                out['violations'].append(
                    f'{dose_key!r} has duplicate values across declared doses: '
                    f'{[(r["label"], v) for r, v in numeric_values]!r}')
            else:
                diffs = [numeric_values[i + 1][1] - numeric_values[i][1]
                         for i in range(len(numeric_values) - 1)]
                if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
                    out['status'] = 'fail'
                    out['violations'].append(
                        f'{dose_key!r} values are not monotonic in the declared '
                        f'dose order: {[(r["label"], v) for r, v in numeric_values]!r}')
        elif numeric_values:
            out['violations'].append(
                f'{dose_key!r} could not be read as one finite scalar for every '
                'level; its order was not verified')
    if out['violations']:
        out['status'] = 'fail'
    return out


def dose_response_trend(levels, *, direction, alpha=0.05, min_n=3,
                        shape='linear'):
    """Exact paired trend test for a pre-registered ordered ladder.

    Each level supplies ``values`` as ``seed -> metric``.  For every seed with
    a finite value at every level, the per-seed slope across the declared doses
    is computed by OLS.  The primary test is the exact two-sided sign-flip on
    those slopes (one test per ordered family, not one per adjacent pair).
    The pre-registered direction gates a positive claim; ``shape='monotone'``
    additionally requires every adjacent *mean* step to move in that direction.
    A significant opposite-direction slope is reported as a direction conflict,
    never silently relabelled as support.

    This is a within-seed trend, not a token-level or between-seed test.  The
    caller must keep the measurement protocol fixed across doses (the design
    gate checks the declared identity); a significant trend still does not
    prove the mechanism, only that the ordered manipulation is associated with
    an ordered outcome.
    """
    alpha = _validated_alpha(alpha, context='dose_response_trend alpha')
    sign = _dose_direction_sign(direction, context='dose_response_trend direction')
    shape = _dose_shape(shape, context='dose_response_trend shape')
    _mn = _as_seed_int(min_n)
    if _mn is None or _mn < 0:
        raise ValueError(f'dose_response_trend: min_n must be a non-negative integer, got {min_n!r}')
    min_n = _mn
    out = {'kind': 'dose_response_trend', 'status': 'ok', 'direction': direction,
           'direction_sign': int(sign), 'shape_hypothesis': shape, 'alpha': alpha,
           'min_n': int(min_n), 'n_levels': 0, 'labels': [], 'doses': [],
           'n_complete_seeds': 0, 'seeds': [], 'excluded_seeds': [],
           'per_seed_slopes': {}, 'slope_stats': None, 'monotonicity': None,
           'claim_allowed': False, 'verdict': None, 'warnings': [],
           'violations': [],
           'note': ('one pre-registered ordered test; do not also select the '
                    'smallest adjacent-pair p and call it the dose-response')}
    try:
        rows = _normalise_dose_levels(levels, require_arm=False,
                                      context='dose_response_trend')
    except ValueError as e:
        out['status'] = 'invalid'
        out['violations'].append(str(e))
        return out
    if not rows:
        out['status'] = 'no_data'
        out['violations'].append('no levels were supplied')
        return out
    rows = sorted(rows, key=lambda r: r['dose'])
    out['n_levels'] = len(rows)
    out['labels'] = [r['label'] for r in rows]
    out['doses'] = [r['dose'] for r in rows]
    if len(rows) < 3:
        out['status'] = 'insufficient_levels'
        out['verdict'] = 'insufficient_levels'
        out['warnings'].append(
            f'{len(rows)} level(s) supplied; a two-point contrast is an '
            'endpoint comparison, not a dose-response')
        return out
    # Protocol identity is a confound gate, not a statistical one.
    protocols = [r['protocol'] for r in rows]
    if all(p is not None for p in protocols):
        out['protocols'] = protocols
        protocol_consistent = len(set(protocols)) == 1
        if not protocol_consistent:
            out['warnings'].append(
                'the levels carry different protocol identities; the trend is '
                'confounded and no claim is allowed')
    elif any(p is not None for p in protocols):
        protocol_consistent = None
        out['warnings'].append(
            'protocol identity is declared on only some levels; consistency is unverified')
    else:
        protocol_consistent = None
        out['warnings'].append(
            'no protocol identity was declared; equal measurement semantics '
            'across doses are unverified')
    out['protocol_consistent'] = protocol_consistent
    # Normalise the per-level seed values.
    vals_by_level = {}
    declared_sets = [r['seeds'] for r in rows]
    anchor = None
    if all(s is not None for s in declared_sets):
        _sets = [set(s) for s in declared_sets]
        if any(s != _sets[0] for s in _sets):
            out['status'] = 'invalid'
            out['violations'].append(
                'declared seed sets differ across levels; a paired within-seed '
                'trend is not defined')
            return out
        anchor = set(_sets[0])
        out['seeds_declared'] = True
    else:
        out['seeds_declared'] = False
        if any(s is not None for s in declared_sets):
            out['warnings'].append(
                'seed sets are declared on only some levels; the complete-case '
                'intersection is used and named below')
    for r in rows:
        raw = r.get('values')
        if not isinstance(raw, dict):
            out['status'] = 'invalid'
            out['violations'].append(
                f'{r["label"]}: values must be a seed -> metric mapping, '
                f'got {type(raw).__name__}')
            return out
        vals, bad, nonfinite = {}, [], []
        for raw_seed, raw_val in raw.items():
            s = _as_seed_int(raw_seed)
            if s is None:
                bad.append(f'non-integer seed {raw_seed!r}')
                continue
            if s in vals or s in nonfinite:
                bad.append(f'duplicate seed {s!r} after normalisation')
                continue
            f = _finite_scalar(raw_val)
            if f is None:
                nonfinite.append(s)
                continue
            vals[s] = f
        if bad:
            out['status'] = 'invalid'
            out['violations'].append(
                f'{r["label"]}: ' + '; '.join(bad[:8])
                + ('' if len(bad) <= 8 else f' (+{len(bad) - 8} more)'))
            return out
        if nonfinite:
            out['warnings'].append(
                f'{r["label"]}: non-finite metric for seed(s) {sorted(nonfinite)!r}; '
                'they are excluded as named incomplete cases')
        if anchor is not None:
            extra = sorted(set(vals) - anchor)
            if extra:
                out['warnings'].append(
                    f'{r["label"]}: values contain seed(s) outside the declared '
                    f'set: {extra!r}; they are not used')
            missing = sorted(anchor - set(vals))
            if missing:
                out['warnings'].append(
                    f'{r["label"]}: declared seed(s) without a finite value: {missing!r}')
        vals_by_level[r['label']] = vals
    if anchor is not None:
        universe = set(anchor)
    else:
        sets = [set(v) for v in vals_by_level.values()]
        universe = set.union(*sets)
    complete = []
    for s in sorted(universe):
        missing_levels = [r['label'] for r in rows if s not in vals_by_level[r['label']]]
        if not missing_levels:
            complete.append(s)
        else:
            out['excluded_seeds'].append(
                {'seed': s, 'reason': 'missing/non-finite at level(s): '
                 + ', '.join(missing_levels)})
    out['n_complete_seeds'] = len(complete)
    out['seeds'] = complete
    if not complete:
        out['status'] = 'no_complete_seeds'
        out['verdict'] = 'no_complete_seeds'
        return out
    x = np.asarray(out['doses'], dtype=float)
    xc = x - float(x.mean())
    sxx = float((xc ** 2).sum())
    if not (math.isfinite(sxx) and sxx > 0.0):
        out['status'] = 'invalid'
        out['violations'].append('doses have zero spread; a trend slope is undefined')
        return out
    # An arbitrary finite metric panel near the float limit can overflow the
    # raw OLS mean/deviation arithmetic even though the true per-seed slope is
    # representable.  Normalise only that extreme path; ordinary panels keep
    # the historical NumPy arithmetic bit-for-bit.
    _y_abs_max = max((abs(vals_by_level[r['label']][s])
                      for s in complete for r in rows), default=0.0)
    _y_extreme = bool(math.isfinite(_y_abs_max) and _y_abs_max > 0.0
                      and (_y_abs_max < 1e-100 or _y_abs_max > 1e100))
    slopes = []
    for s in complete:
        y = np.asarray([vals_by_level[r['label']][s] for r in rows], dtype=float)
        if _y_extreme:
            y_work = y / _y_abs_max
            y_mean = _scaled_mean_or_raw(y_work)
            if y_mean is None or not math.isfinite(float(y_mean)):
                slopes.append(None)
                continue
            slope = (float(((xc) * (y_work - float(y_mean))).sum() / sxx)
                     * _y_abs_max)
        else:
            y_mean = _scaled_mean_or_raw(y)
            if y_mean is None or not math.isfinite(float(y_mean)):
                slopes.append(None)
                continue
            slope = float(((xc) * (y - float(y_mean))).sum() / sxx)
        slopes.append(float(slope) if math.isfinite(slope) else None)
    slope_seeds = [s for s, v in zip(complete, slopes) if v is not None]
    slope_values = [v for v in slopes if v is not None]
    if len(slope_values) < len(complete):
        for s, v in zip(complete, slopes):
            if v is None:
                out['excluded_seeds'].append({
                    'seed': s,
                    'reason': ('per-seed trend slope is not representable as '
                               'a finite float')})
    out['per_seed_slopes'] = {
        str(s): float(v) for s, v in zip(complete, slopes) if v is not None}
    if not slope_values:
        out['status'] = 'not_estimable'
        out['verdict'] = 'not_estimable'
        out['slope_stats'] = None
        out['warnings'].append(
            'no per-seed trend slope is representable as a finite float; the '
            'ordered test cannot be formed')
        return out
    stats = paired_effect_stats(slope_values, alpha=alpha, min_n=min_n,
                                seeds=slope_seeds)
    out['slope_stats'] = stats
    mean_slope = stats.get('mean')
    p = stats.get('p_exact_signflip')
    p_f = _finite_scalar(p)
    p = p_f if p_f is not None else None
    out['p_exact_signflip'] = p
    out['p_floor'] = stats.get('p_floor')
    out['underpowered'] = bool(stats.get('underpowered'))
    out['resolution_limited'] = bool(stats.get('resolution_limited'))
    out['within_seed_noise'] = bool(stats.get('within_seed_noise'))
    direction_match = None
    if isinstance(mean_slope, (int, float)) and math.isfinite(float(mean_slope)):
        if abs(float(mean_slope)) <= 1e-15:
            direction_match = False
        else:
            direction_match = ((float(mean_slope) > 0) == (sign > 0))
    out['direction_match'] = direction_match
    # Adjacent mean steps answer the shape question.  Individual reversals are
    # reported as a seed count; they are not additional hypothesis tests.
    # Compare values directly instead of subtracting them, so a finite panel
    # whose adjacent difference exceeds float64 is not misclassified.
    steps = []
    strict_seeds = 0
    for s in complete:
        _all = True
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            _va = vals_by_level[a['label']][s]
            _vb = vals_by_level[b['label']][s]
            _all = _all and ((_vb > _va) if sign > 0 else (_vb < _va))
        if _all:
            strict_seeds += 1
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        _a_vals = [vals_by_level[a['label']][s] for s in complete]
        _b_vals = [vals_by_level[b['label']][s] for s in complete]
        ds = [bv - av for av, bv in zip(_a_vals, _b_vals)]
        if all(math.isfinite(d) for d in ds):
            md = _scaled_mean_or_raw(ds)
            if md is not None and not math.isfinite(float(md)):
                md = None
        else:
            # A pairwise difference overflowed even though both level means
            # are finite; mean(b) - mean(a) is the same estimand and stays
            # representable in that case.
            _mb = _scaled_mean_or_raw(_b_vals)
            _ma = _scaled_mean_or_raw(_a_vals)
            md = (float(_mb) - float(_ma)
                  if (_mb is not None and _ma is not None
                      and math.isfinite(float(_mb))
                      and math.isfinite(float(_ma)))
                  else None)
            if md is not None and not math.isfinite(md):
                md = None
        steps.append({
            'from': a['label'], 'to': b['label'],
            'dose_gap': float(b['dose'] - a['dose']),
            'mean_delta': (None if md is None else float(md)),
            'n': len(ds),
            'direction_match': (None if md is None
                                else bool(float(md) * sign > 0)),
            'zero': (None if md is None else bool(abs(float(md)) <= 1e-15)),
        })
    all_steps = bool(steps) and all(step['direction_match'] for step in steps)
    out['monotonicity'] = {
        'adjacent_steps': steps,
        'n_steps': len(steps),
        'n_steps_in_direction': int(sum(1 for step in steps if step['direction_match'])),
        'all_mean_steps_in_direction': all_steps,
        'strict_monotone_seeds': int(strict_seeds),
        'strict_monotone_seed_fraction': (float(strict_seeds) / len(complete)
                                          if complete else None),
    }
    _first_vals = [vals_by_level[rows[0]['label']][s] for s in complete]
    _last_vals = [vals_by_level[rows[-1]['label']][s] for s in complete]
    _ep_diffs = [lv - fv for fv, lv in zip(_first_vals, _last_vals)]
    if all(math.isfinite(d) for d in _ep_diffs):
        _ep_mean = _scaled_mean_or_raw(_ep_diffs)
        if _ep_mean is not None and not math.isfinite(float(_ep_mean)):
            _ep_mean = None
    else:
        _last_mean = _scaled_mean_or_raw(_last_vals)
        _first_mean = _scaled_mean_or_raw(_first_vals)
        _ep_mean = (float(_last_mean) - float(_first_mean)
                    if (_last_mean is not None and _first_mean is not None
                        and math.isfinite(float(_last_mean))
                        and math.isfinite(float(_first_mean)))
                    else None)
        if _ep_mean is not None and not math.isfinite(_ep_mean):
            _ep_mean = None
    out['endpoint_delta'] = {
        'first_to_last_mean': (None if _ep_mean is None else float(_ep_mean)),
        'per_seed': {
            str(s): (float(lv - fv) if math.isfinite(lv - fv) else None)
            for s, fv, lv in zip(complete, _first_vals, _last_vals)},
    }
    raw_sig = bool(p is not None and p < alpha)
    if out['n_complete_seeds'] < min_n:
        verdict = 'underpowered'
    elif protocol_consistent is False:
        verdict = 'protocol_conflict'
    elif out['resolution_limited']:
        verdict = 'resolution_limited'
    elif out['within_seed_noise']:
        verdict = 'within_seed_noise'
    elif not raw_sig:
        verdict = 'not_significant'
    elif direction_match is False:
        verdict = 'direction_conflict'
    elif shape == 'monotone' and not all_steps:
        verdict = 'shape_conflict'
    elif all_steps:
        verdict = 'monotone_trend_supported'
    else:
        verdict = 'linear_trend_supported'
    out['verdict'] = verdict
    out['claim_allowed'] = bool(
        verdict in ('monotone_trend_supported', 'linear_trend_supported')
        and direction_match is True and protocol_consistent is not False)
    out['claim_level'] = ('pre_registered_ordered_contrast'
                          if out['claim_allowed'] else None)
    if protocol_consistent is False:
        out['claim_allowed'] = False
        out['claim_level'] = None
    if shape == 'monotone' and not all_steps and raw_sig and direction_match:
        out['warnings'].append(
            'the pre-registered monotone shape is not supported by every '
            'adjacent mean step; the linear trend is reported without the '
            'stronger monotone claim')
    out['family_note'] = ('this is one ordered test; if several ordered '
                          'families are analyzed, correct their p-values as a '
                          'declared family instead of treating adjacent pairs '
                          'as independent')
    return out


_METHODOLOGY_STEPS_CACHE = None


def _methodology_step_sources():
    """Load the pre-split methodology_audit body fragments in order."""
    global _METHODOLOGY_STEPS_CACHE
    if _METHODOLOGY_STEPS_CACHE is None:
        _root = os.path.dirname(os.path.abspath(__file__))
        _steps_dir = os.path.join(_root, 'explib', 'methodology_steps')
        _loaded = []
        for _step_name in sorted(os.listdir(_steps_dir)):
            if _step_name.endswith('.py'):
                with open(os.path.join(_steps_dir, _step_name), encoding='utf-8') as _step_fh:
                    _loaded.append(_step_fh.read())
        _METHODOLOGY_STEPS_CACHE = _loaded
    return _METHODOLOGY_STEPS_CACHE


def methodology_audit(summary, design=None, *, baselines=None, alpha=0.05, min_n=3, outdir=None, write=True, verbose=True, sesoi=None, look_history=None, planned_cells=None):
    """Zero-GPU audit of controlled comparison, ablation completeness and
    statistical reliability for one panel's `summary` dict.

    For every declared leave-one-out contrast (and, by default, the primary
    parameter-matched dense baselines) it recomputes the per-seed paired delta
    under a strict same-run_cfg / same-train_knobs gate, the mean±std / d_z /
    exact sign-flip p, and flags underpowered (n<min_n), noise-level
    (|mean| <= std) and resolution-limited (the exact-test p floor is above
    alpha) results.

    Multiple comparisons are corrected with Holm step-down over the declared
    family.  A missing arm does NOT shrink the family (dummy p=1 entries keep
    the declared m), so an incomplete ablation cannot buy a lower p.  A
    component is `verified` only when a controlled contrast is significant
    AFTER that correction and its effect exceeds seed noise (and, when a SESOI
    is declared for it, the practical decision is `superior`); merely having
    n>=min_n is reported as `measured` (descriptive), never as verification.
    Per-component Holm results are also exported, because the report-facing
    claim is usually per innovation, while the global correction is the
    conservative cross-component reading.

    ``sesoi`` is the pre-registered smallest effect size of interest mapping
    (default: :data:`DEFAULT_METHODOLOGY_SESOI`, the union of
    :data:`ABLATION_SESOI` and :data:`BASELINE_SESOI`; pass ``{}`` to disable).
    For every component and parameter-matched baseline with a declared margin,
    a paired TOST classifies the contrast as ``superior`` / ``inferior`` /
    ``sub_sesoi`` / ``equivalent`` / indeterminate.  A non-significant result
    is never re-labelled equivalent without TOST, and a statistically
    significant effect below the margin is not counted as practically useful.

    Returns the report dict and, when `outdir` is given, writes
    `methodology_audit.json`.

    ``look_history`` (default: read ``<outdir>/look_history.json`` when it
    exists) records every aggregation look of the panel.  When a contrast was
    measured at more than one growing seed set, its significance is ALSO
    required to pass the exact sequential sign-flip p from
    :func:`sequential_signflip_p`; the naive final-n p is kept as
    ``p_exact_signflip`` / ``p_holm_global_naive`` for transparency.  This is
    the optional-stopping correction: topping up seeds after seeing a
    non-significant result cannot buy the final-n p for free.

    ``planned_cells`` is the run manifest's flat ``arm::seedN`` list.  When it
    is omitted and ``outdir`` holds a ``run_manifest.json``, that manifest's
    plan is loaded automatically.  The planned frame makes the missing-data
    block possible: each contrast gets a partial-identification bound and a
    tipping point for the seeds/cells the run-time cap, a failure or the
    control gate removed.  The block is descriptive and never changes a
    p-value, Holm family, ``verified`` verdict or the frozen plan; without a
    plan the audit says so instead of inventing one.

    This report is single-panel evidence.  Project-level replication and
    cross-panel contradictions are computed by
    :func:`ablation_coverage_across_panels`; do not read one panel's
    `verified_components` as an independently replicated result."""
    if planned_cells is None and outdir:
        try:
            _manifest_path = os.path.join(outdir, 'run_manifest.json')
            with open(_manifest_path, encoding='utf-8') as _mh:
                _manifest = json.load(_mh)
            if isinstance(_manifest, dict) and isinstance(_manifest.get('planned_cells'), list):
                planned_cells = list(_manifest['planned_cells'])
        except (OSError, ValueError, TypeError):
            planned_cells = None
    _step_env = dict(globals())
    _step_env.update(locals())
    for _step_src in _methodology_step_sources():
        exec(compile(_step_src, '<methodology_audit step>', 'exec'), _step_env)
    return _step_env['report']

def baseline_claim_gate(report):
    """Unified ladder for the claim "the method beats a well-tuned baseline".

    A significant matched-dense contrast is only interpretable after two weaker
    rungs have been cleared on the SAME panel: the declared method arm must
    beat the tuned count-model floor, and the dense baseline must have been
    selected by the declared equal-search tuning grid and actually trained at
    that selected configuration.  Keeping the three rungs explicit prevents the
    common overclaim in which a method beats an untuned baseline (or one arm of
    a multi-baseline family) and is reported as beating "the baseline".

    ``claim_allowed`` is True only when every applicable rung passes.  Missing
    evidence is ``unverified`` (warn), not a pass; explicit counter-evidence
    (method below the floor, or a matched baseline significantly better than
    the method) is ``failed``.  The gate never reads the test split and adds no
    measurement: it only combines artifacts the panel audit already computed.
    """
    out = {'kind': 'baseline_claim_gate', 'status': 'not_applicable',
           'applicable': False, 'claim_allowed': None, 'rungs': {},
           'reasons': [], 'warnings': [],
           'note': ('rungs: tuned count-model floor -> equal-search tuning -> '
                    'tuned matched-dense contrast; all must pass on the same '
                    'panel and rows')}
    if not isinstance(report, dict):
        out.update(status='fail', reasons=['methodology audit is not an object'])
        return out
    baselines = report.get('baselines')
    if not isinstance(baselines, dict) or not baselines:
        out['reasons'].append('this panel declares no matched-dense baseline contrast')
        return out
    out['applicable'] = True
    simple = report.get('simple_baseline') if isinstance(
        report.get('simple_baseline'), dict) else {}
    tuning = report.get('baseline_tuning') if isinstance(
        report.get('baseline_tuning'), dict) else {}
    floor_status = simple.get('method_arm_floor_status')
    floor_ok = (simple.get('tuned_floor_claim_allowed') is True
                and simple.get('comparable_to_panel') is True
                and simple.get('method_arms_beat_floor') is True)
    if floor_ok:
        _fs, _fd = 'passed', 'declared method arm(s) beat the tuned count floor'
    elif (floor_status == 'method_arm_does_not_beat_floor'
          or simple.get('panel_beats_simple_floor') is False
          or (floor_status == 'method_floor_contrast_unresolved'
              and simple.get('method_arms_beat_floor') is False)):
        _fs, _fd = 'failed', 'a declared method arm does not beat the tuned count floor'
    else:
        _fs, _fd = 'unverified', (simple.get('status') or 'missing')
    out['rungs']['tuned_count_floor'] = {'passed': _fs == 'passed',
                                         'status': _fs, 'detail': _fd}
    if tuning.get('tuned_claim_allowed') is True:
        _ts, _td = 'passed', 'equal-search tuning artifact is verified'
    elif tuning.get('tuning_verified') is True:
        _ts, _td = 'unverified', ('tuning artifact is verified but not adequate: '
                                  + '; '.join(tuning.get('tuned_claim_reasons') or []))
    else:
        _ts, _td = 'unverified', (tuning.get('status') or 'missing')
    out['rungs']['equal_search_tuning'] = {'passed': _ts == 'passed',
                                           'status': _ts, 'detail': _td}
    _tuned_v = list(report.get('tuned_verified_baselines') or [])
    _untuned_v = list(report.get('untuned_verified_baselines') or [])
    _conflict = list(report.get('direction_conflict_baselines') or [])
    if _conflict:
        _ms = 'failed'
        _md = ('matched baseline(s) significantly better than the method: '
               + ', '.join(map(str, _conflict)))
    elif _tuned_v and not _untuned_v:
        _ms, _md = 'passed', 'tuned matched-dense contrast is verified'
    elif _untuned_v:
        _ms = 'unverified'
        _md = ('verified only against an untuned baseline: '
               + ', '.join(map(str, _untuned_v)))
    else:
        _ms, _md = 'unverified', 'no verified method-vs-baseline contrast'
    out['rungs']['tuned_matched_dense'] = {'passed': _ms == 'passed',
                                           'status': _ms, 'detail': _md}
    out['claim_allowed'] = all(r['passed'] for r in out['rungs'].values())
    for _name, _rung in out['rungs'].items():
        if not _rung['passed']:
            out['reasons'].append(f"{_name}: {_rung['status']} ({_rung['detail']})")
    if out['claim_allowed']:
        out['status'] = 'ok'
    elif any(r['status'] == 'failed' for r in out['rungs'].values()):
        out['status'] = 'fail'
    else:
        out['status'] = 'warn'
    if out['status'] != 'ok':
        out['warnings'].append(
            'do not quote a "beats a tuned baseline" claim until every rung passes')
    return out
