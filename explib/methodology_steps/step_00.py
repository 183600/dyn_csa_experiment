alpha = _validated_alpha(alpha, context='methodology_audit alpha')

_min_n = _as_seed_int(min_n)

if _min_n is None or _min_n < 0:
    raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')

min_n = _min_n

_sesoi_specs = _normalise_sesoi_design(DEFAULT_METHODOLOGY_SESOI if sesoi is None else sesoi)

default_design = design is None

design = ABLATION_DESIGN if design is None else design

_design_type_problem = None

if not isinstance(design, dict):
    _design_type_problem = type(design).__name__
    design = {}

_baseline_type_problem = None

if baselines is not None and (not isinstance(baselines, dict)):
    _baseline_type_problem = type(baselines).__name__
    baselines = {}

_summary_type = None

if summary is not None and (not isinstance(summary, dict)):
    _summary_type = type(summary).__name__
    summary = {}

recs, amb = _variant_seed_map(summary)

present = sorted({v for v, _t, _s in recs})

if baselines is None:
    _base_arms = {x for _cs in BASELINE_DESIGN.values() for _a, _b, _n in _cs for x in (_a, _b)}
    baselines = BASELINE_DESIGN if default_design and _base_arms & set(present) else {}

if look_history is None and outdir is not None:
    look_history = load_look_history(outdir)

if look_history is not None and (not isinstance(look_history, dict)):
    report_look_invalid = f'look_history is a {type(look_history).__name__}, not an object'
    look_history = {'looks': []}
else:
    report_look_invalid = (look_history or {}).get('invalid') if isinstance(look_history, dict) else None

_look_series = looks_by_contrast(look_history or {})

design = _materialise_contrast_design(design)

baselines = _materialise_contrast_design(baselines)

try:
    _min_sig_n = signflip_min_n_for_holm(1, alpha) or 0
except (ValueError, ZeroDivisionError):
    _min_sig_n = 0

report = {'variants_present': present, 'n_measurable': len(recs), 'ambiguous': sorted(amb), 'min_n': int(min_n), 'alpha': float(alpha), 'min_seeds_for_significance': _min_sig_n, 'components': {}, 'baselines': {}, 'verified_components': [], 'measured_components': [], 'significant_components': [], 'nominally_significant_components': [], 'resolution_limited_components': [], 'underpowered_components': [], 'noise_level_components': [], 'unverified_components': [], 'system_level_only_components': [], 'equivalent_components': [], 'sub_sesoi_components': [], 'practically_indeterminate_components': [], 'practically_useful_components': [], 'practically_harmful_components': [], 'practically_inconclusive_components': [], 'sesoi': {}, 'look_history': {}, 'look_adjusted_contrasts': [], 'direction_conflict_components': [], 'hypotheses_missing': [], 'verified_baselines': [], 'measured_baselines': [], 'significant_baselines': [], 'nominally_significant_baselines': [], 'resolution_limited_baselines': [], 'underpowered_baselines': [], 'noise_level_baselines': [], 'unverified_baselines': [], 'system_level_only_baselines': [], 'direction_conflict_baselines': [], 'tuned_verified_baselines': [], 'untuned_verified_baselines': [], 'direction_unspecified_components': [], 'direction_unspecified_baselines': [], 'incomplete_components': [], 'controlled_variables': {}, 'component_draws': {}, 'ablation_completeness': {}, 'ablation_complete': None, 'arm_summary': {}, 'arms_below_min_seeds': [], 'component_seed_requirements': {}, 'baseline_seed_requirements': {}, 'hypotheses': {}, 'hypotheses_incomplete': [], 'hypothesis_audit': {}, 'metric_audit': {}, 'baseline_tuning': {}, 'simple_baseline': {}, 'baseline_claim_gate': {}, 'family': {}, 'warnings': []}

if _summary_type is not None:
    report['warnings'].append(f'summary is a {_summary_type}, not an object; no record-level contrast can be audited')

if _design_type_problem is not None:
    report['warnings'].append(f'design is a {_design_type_problem}, not a mapping; no component contrast can be audited')

if _baseline_type_problem is not None:
    report['warnings'].append(f'baselines is a {_baseline_type_problem}, not a mapping; no baseline contrast can be audited')

if report_look_invalid:
    report['warnings'].append(f'look history is unreadable or malformed ({report_look_invalid}); the optional-stopping/sequential correction cannot be verified for this panel — do not read its naive p as look-adjusted')

def _attach_look_analysis(st, key, final_seeds, final_deltas):
    """Attach the sequential p for this contrast from the recorded looks.

        A history look whose shared-seed deltas disagree with the current
        record is STALE (the seed was re-run): it is excluded and named, never
        silently averaged into the joint null.  The final look is always
        appended, so a manually edited summary still gets a two-look check
        when history exists.
        """
    if final_seeds is None or final_deltas is None:
        return None
    _final_seeds = list(final_seeds)
    _final_deltas = list(final_deltas)
    if not _final_seeds or not _final_deltas:
        return None

    def _look_rows(rec):
        raw_seeds = rec.get('seeds')
        raw_deltas = rec.get('deltas')
        if raw_seeds is None or raw_deltas is None:
            return None
        if isinstance(raw_seeds, (str, bytes, bytearray)) or isinstance(raw_deltas, (str, bytes, bytearray)):
            return None
        seeds = [_as_seed_int(s) for s in raw_seeds]
        if any((s is None for s in seeds)):
            return None
        deltas = [_finite_scalar(d) for d in raw_deltas]
        if any((d is None for d in deltas)):
            return None
        return (seeds, [float(d) for d in deltas])
    cur = {}
    for s, d in zip(_final_seeds, _final_deltas):
        si = _as_seed_int(s)
        if si is None:
            raise ValueError('final look seeds must be integers')
        cur[si] = float(d)
    series, stale = ([], [])
    seen = set()
    for lk in _look_series.get(key) or []:
        try:
            parsed = _look_rows(lk)
            if parsed is None:
                raise ValueError('non-numeric seed/delta')
            seeds, deltas = parsed
        except (TypeError, ValueError):
            stale.append({'look': lk.get('look'), 'reason': 'non-numeric seed/delta'})
            continue
        if len(seeds) != len(deltas) or not seeds:
            stale.append({'look': lk.get('look'), 'reason': 'seed/delta length mismatch or empty'})
            continue
        _rows = sorted(zip(seeds, deltas), key=lambda t: t[0])
        seeds = [s for s, _d in _rows]
        deltas = [d for _s, d in _rows]
        dm = dict(zip(seeds, deltas))
        bad = [s for s in seeds if s in cur and abs(dm[s] - cur[s]) > 1e-09]
        if bad:
            stale.append({'look': lk.get('look'), 'reason': 'seed delta changed across looks', 'seeds': sorted(bad)})
            continue
        sig = (tuple(seeds), tuple((round(v, 12) for v in deltas)))
        if sig in seen:
            continue
        seen.add(sig)
        series.append({'look': lk.get('look'), 'seeds': seeds, 'deltas': deltas})
    _final_order = sorted(cur)
    sig = (tuple(_final_order), tuple((round(cur[s], 12) for s in _final_order)))
    if sig not in seen:
        series.append({'look': series[-1].get('look', 0) + 1 if series else 1, 'seeds': _final_order, 'deltas': [cur[s] for s in _final_order], 'current': True})
    if not series:
        return None
    seq = sequential_signflip_p(series, alpha=alpha)
    seq['contrast'] = key
    seq['n_stale_looks'] = len(stale)
    seq['stale_looks'] = stale
    seq['look_adjusted'] = bool(len(series) > 1)
    if stale:
        report['warnings'].append('%s: %d recorded look(s) are STALE (a seed delta changed after re-running); they were excluded from the sequential test, so the correction covers only the remaining %d look(s)' % (key, len(stale), len(series)))
    if len(series) > 1:
        report['look_adjusted_contrasts'].append(key)
        if seq.get('p_sequential') is not None and math.isfinite(float(seq['p_sequential'])):
            _naive = st.get('p_exact_signflip')
            if isinstance(_naive, (int, float)) and math.isfinite(float(_naive)) and (float(_naive) < alpha) and (float(seq['p_sequential']) >= alpha):
                report['warnings'].append('%s: the naive final-look p=%.4g would be significant, but the sequential p over %d looks is %.4g (>= alpha=%.3g) — topping up seeds does not make the effect confirmable; report it as exploratory/unresolved' % (key, float(_naive), len(series), float(seq['p_sequential']), alpha))
    return seq

def _collect(design_group, kind):
    out = {}
    meta = {}
    if not isinstance(design_group, dict):
        report['warnings'].append(f'{kind} design is not a mapping; no contrast can be audited')
        return (out, meta)
    for comp, contrasts in design_group.items():
        entries = []
        arms = set()
        if isinstance(contrasts, (str, bytes, bytearray)):
            report['warnings'].append(f'{comp}: {kind} contrasts are a string, not a list of (with, without, note) tuples; the declaration is malformed and was not audited')
            _contrast_items = []
        else:
            try:
                _contrast_items = list(contrasts or ())
            except TypeError:
                report['warnings'].append(f'{comp}: {kind} contrasts are not an iterable; the declaration is malformed and was not audited')
                _contrast_items = []
        for _item in _contrast_items:
            if not isinstance(_item, (tuple, list)) or len(_item) < 2:
                report['warnings'].append(f'{comp}: malformed {kind} contrast {_item!r} (expected (with, without, note)); it was not audited')
                continue
            a, b = (_item[0], _item[1])
            if not isinstance(a, str) or not isinstance(b, str) or (not a.strip()) or (not b.strip()):
                report['warnings'].append(f'{comp}: malformed {kind} contrast {_item!r} (arm ids must be non-empty strings); it was not audited')
                continue
            note = _item[2] if len(_item) > 2 else ''
            arms.update((a, b))
            a_present, b_present = (a in present, b in present)
            conds = _paired_seed_deltas(recs, a, b, strict_hyper=kind == 'component')
            _cfg_a = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == a), None)
            _cfg_b = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == b), None)
            _rel, _param_ok = _param_control(recs, a, b)
            _param_detail = _param_control_detail(recs, a, b)
            _param_capacity_ok = _param_capacity_controlled(_param_detail)
            _contrast_measured = False
            for (ts, bs), (deltas, shared, skipped) in sorted(conds.items()):
                _skipped_seeds = {s for s, _why in skipped}
                _passed = [s for s in shared if s not in _skipped_seeds]
                _a_vals, _b_vals = ([], [])
                for s in _passed:
                    _av = _record_primary_ppl(recs[a, ts, s])
                    _bv = _record_primary_ppl(recs[b, bs, s])
                    if _av is None or _bv is None:
                        continue
                    _a_vals.append(float(_av))
                    _b_vals.append(float(_bv))
                st = paired_effect_stats(deltas, alpha=alpha, min_n=min_n, with_values=_a_vals, without_values=_b_vals)
                _obj_pairs = [_objective_diff(recs[a, ts, s], recs[b, bs, s]) for s in _passed]
                _obj = sorted({r for r, _ks in _obj_pairs if r})
                _obj_keys = sorted({k for _r, ks in _obj_pairs for k in ks})
                _expected_obj = OBJECTIVE_INTERVENTION_KEYS.get(comp, frozenset()) if kind == 'component' else frozenset()
                _obj_unexpected = [k for k in _obj_keys if k not in _expected_obj]
                _obj_bad = ['objective differs: ' + ', '.join(_obj_unexpected)] if _obj_unexpected else []
                st['objective_matched'] = False if _obj_keys else True if all((r is None for r, _ks in _obj_pairs)) else None
                st['objective_differences'] = _obj_bad or [x for x in _obj if x.startswith('unverifiable')]
                st['objective_expected_differences'] = sorted(set(_obj_keys) & set(_expected_obj))
                st['objective_differences_all'] = list(_obj_keys)
                if _obj_bad:
                    st['mechanism_claim_allowed'] = False
                    report.setdefault('objective_confounded', []).append('%s vs %s: %s — same run_cfg but different actual loss/optimizer objective; the contrast is a system-level comparison, not an isolated mechanism' % (a, b, '; '.join(_obj_bad)))
                _cc = _compute_control_detail(recs, a, b)
                st['compute_control'] = _cc
                st['compute_matched'] = _cc.get('compute_matched')
                if _cc.get('compute_matched') is False and isinstance(st.get('mean'), (int, float)) and math.isfinite(float(st['mean'])):
                    _better = a if float(st['mean']) < 0 else b
                    _more = _cc.get('more_compute_arm')
                    _bias = 'favors_claimed_winner' if _more == _better else 'conservative_against_claimed_winner'
                    st['compute_confound_direction'] = _bias
                    st['compute_ratio'] = max(_cc.get('time_ratio_a_over_b') or 1.0, 1.0 / (_cc.get('time_ratio_a_over_b') or 1.0)) if _cc.get('time_ratio_a_over_b') else max(_cc.get('proxy_ratio_a_over_b') or 1.0, 1.0 / (_cc.get('proxy_ratio_a_over_b') or 1.0))
                    if _bias == 'favors_claimed_winner' and float(st['compute_ratio']) > 1.5:
                        st['mechanism_claim_allowed'] = False
                        report.setdefault('compute_confounded', []).append('%s vs %s: the arm that wins by %.2f PPL also consumed about %.2fx the compute (%s); the performance difference is not separable from the compute difference' % (a, b, abs(float(st['mean'])), float(st['compute_ratio']), _cc.get('compute_matched_source')))
                if _a_vals and _b_vals:
                    _a_mean = float(np.mean(_a_vals))
                    _b_mean = float(np.mean(_b_vals))
                    _a_std = float(np.std(_a_vals, ddof=1)) if len(_a_vals) > 1 else 0.0
                    _b_std = float(np.std(_b_vals, ddof=1)) if len(_b_vals) > 1 else 0.0
                    _rel_effect = 100.0 * float(st['mean']) / _b_mean if _b_mean > 0 else float('nan')
                    _rel_std = 100.0 * float(st['std']) / _b_mean if _b_mean > 0 else float('nan')
                    if float(st['std']) > 0:
                        _noise_ratio = abs(float(st['mean'])) / float(st['std'])
                    elif abs(float(st['mean'])) <= 1e-12:
                        _noise_ratio = 0.0
                    else:
                        _noise_ratio = float('inf')
                else:
                    _a_mean = _b_mean = _a_std = _b_std = float('nan')
                    _rel_effect = _rel_std = _noise_ratio = float('nan')
                _hyp = _hypothesis_spec(comp, kind)
                _pred = (_hyp or {}).get('predicted_sign')
                _obs = _sign_of(st.get('mean')) if st.get('n') else None
                st.update({'with': a, 'without': b, 'note': note, 'kind': kind, 'tag_with': ts, 'tag_without': bs, 'n_shared': len(shared), 'n_skipped': len(skipped), 'n_paired': len(_passed), 'seeds': _passed, 'skipped_reasons': sorted({reason for _s, reason in skipped}), 'controlled': bool(deltas), 'fully_paired': bool(deltas) and (not skipped), 'run_cfg_equal': _cfg_a is not None and _cfg_a == _cfg_b, 'run_cfg_knobs': run_cfg_knobs(_cfg_a) or None, 'param_rel_diff': _rel, 'param_controlled': _param_ok, 'param_breakdown': _param_detail, 'param_capacity_controlled': _param_capacity_ok, 'capacity_confounded': _param_ok is False or _param_capacity_ok is False, 'with_mean': _a_mean, 'with_std': _a_std, 'without_mean': _b_mean, 'without_std': _b_std, 'relative_effect_pct': _rel_effect, 'relative_noise_pct': _rel_std, 'signal_to_noise': _noise_ratio, 'ci95_low': st.get('ci95_low'), 'ci95_high': st.get('ci95_high'), 'seed_driven': st.get('seed_driven'), 'seed_robust': st.get('seed_robust'), 'loo_means': st.get('loo_means'), 'mean_pm_std': '%+.4f +/- %.4f' % (float(st['mean']), float(st['std'])) if st.get('n') else 'n/a', 'hypothesis': (_hyp or {}).get('statement'), 'hypothesis_mechanism': (_hyp or {}).get('mechanism'), 'hypothesis_falsifier': (_hyp or {}).get('falsifier'), 'primary_metric': (_hyp or {}).get('primary_metric'), 'requires_param_match': bool((_hyp or {}).get('requires_param_match') is True), 'predicted_sign': _pred, 'observed_sign': _obs, 'direction_consistent': None if _pred is None or _obs is None else _pred == 0 or _pred == _obs})
                st['deltas'] = [float(x) for x in deltas]
                _look_key = _contrast_look_key(kind, comp, a, b, ts, bs)
                _seq = _attach_look_analysis(st, _look_key, _passed, deltas)
                if _seq is not None:
                    st['look_analysis'] = _seq
                    st['p_look_adjusted'] = _seq.get('p_sequential')
                else:
                    st['p_look_adjusted'] = st.get('p_exact_signflip')
                st['look_adjusted'] = bool((_seq or {}).get('look_adjusted'))
                if st['requires_param_match'] and _param_capacity_ok is not True:
                    st['param_match_unmet'] = True
                    st['mechanism_claim_allowed'] = False
                    if deltas:
                        report['warnings'].append('%s: the pre-registered hypothesis requires matched capacity, but `%s` vs `%s` does not clear the parameter/breakdown control (%s); the contrast may measure capacity rather than the declared mechanism' % (comp, a, b, 'unverifiable' if _param_capacity_ok is None else 'capacity differs'))
                st['paired_row_bootstrap_delta'] = paired_row_bootstrap_delta([(recs[a, ts, s], recs[b, bs, s]) for s in _passed])
                if deltas and comp in _sesoi_specs:
                    _margin, _why = _sesoi_margin_ppl(_sesoi_specs[comp], _b_mean)
                    if _margin is None:
                        st['sesoi'] = {'declared': True, 'margin_ppl': None, 'decision': 'not_declared', 'claim': 'not_declared', 'reason': _why, 'source': _sesoi_specs[comp]}
                        st['practical_decision'] = 'not_declared'
                    else:
                        _dir = _pred if _pred in (-1, 0, 1) else -1
                        _pd = ablation_practical_decision(deltas, margin=_margin, alpha=alpha, min_n=min_n, direction=_dir, label=f'{comp}: {a} - {b}')
                        _pd['declared'] = True
                        _pd['source'] = _sesoi_specs[comp]
                        st['sesoi'] = _pd
                        st['practical_decision'] = _pd['decision']
                entries.append(st)
                if deltas:
                    _contrast_measured = True
                    if skipped:
                        report['warnings'].append('%s: `%s` vs `%s` has %d controlled pair(s) but %d seed(s) rejected by the same-run_cfg/train_knobs gate (%s) — n counts only the controlled pairs; do not read the skipped seeds as a larger sample' % (comp, a, b, len(deltas), len(skipped), '; '.join(sorted({reason for _s, reason in skipped}))))
                    if st.get('seed_driven'):
                        report['warnings'].append('%s: `%s` vs `%s` (n=%d) is SEED-DRIVEN — dropping any one seed flips the sign of the mean, so the direction is not stable across seeds even if the aggregate p looks small; report the per-seed deltas and add seeds before treating it as a mechanism effect' % (comp, a, b, int(st.get('n') or 0)))
            if a_present and b_present:
                if not _contrast_measured:
                    _why = sorted({reason for _dl, _sh, _sk in conds.values() for _s, reason in _sk})
                    if _why:
                        report['warnings'].append('%s: `%s` vs `%s` both measured but NO seed clears the same-run_cfg/train_knobs gate — %s' % (comp, a, b, '; '.join(_why)))
                    else:
                        report['warnings'].append('%s: `%s` vs `%s` both measured but share no common protocol condition/seed — no contrast computed' % (comp, a, b))
            elif a_present != b_present:
                report['warnings'].append('%s: incomplete %s `%s` vs `%s` — missing `%s`' % (comp, kind, a, b, a if not a_present else b))
            else:
                report['warnings'].append('%s: neither `%s` nor `%s` measured — no %s for this %s yet' % (comp, a, b, 'leave-one-out' if kind == 'component' else 'baseline contrast', kind))
            if kind == 'baseline' and _param_ok is False:
                report['warnings'].append('%s: `%s` vs `%s` differ in parameter count by %.2f%% — NOT parameter-controlled, so this baseline claim is not counted as verified' % (comp, a, b, 100.0 * (_rel or 0.0)))
            elif kind == 'baseline' and _param_ok is None and a_present and b_present:
                report['warnings'].append('%s: `%s` vs `%s` carry no usable `params` — parameter control is UNVERIFIABLE, so this baseline claim is not counted as verified' % (comp, a, b))
            elif kind == 'component' and _param_ok is False:
                report['warnings'].append('%s: `%s` vs `%s` differ in parameter count by %.2f%% — part of this module effect may be a capacity (parameter-count) effect rather than the mechanism; report the parameter delta or add a parameter-matched control before claiming the mechanism is useful' % (comp, a, b, 100.0 * (_rel or 0.0)))
            elif _param_ok is True and _param_capacity_ok is False:
                _bad_keys = [k for k in ('non_embedding_controlled', 'active_controlled') if _param_detail.get(k) is False]
                report['warnings'].append('%s: `%s` vs `%s` match on TOTAL parameters but the recorded breakdown differs (%s) — the totals hide reallocated or dead capacity (MLP width / indexer projections), so this is not a capacity-matched control' % (comp, a, b, ', '.join((k.replace('_controlled', '') for k in _bad_keys)) or 'breakdown'))
        out[comp] = entries
        meta[comp] = {'arms': arms}
    return (out, meta)

comp_entries, comp_meta = _collect(design, 'component')

base_entries, base_meta = _collect(baselines, 'baseline')

for comp in design:
    _hyp = _hypothesis_spec(comp, 'component')
    if _hyp is None:
        report['hypotheses_missing'].append(comp)
    else:
        report['hypotheses'][comp] = dict(_hyp, kind='component')

for comp in baselines:
    _hyp = _hypothesis_spec(comp, 'baseline')
    if _hyp is None:
        report['hypotheses_missing'].append(comp)
    else:
        report['hypotheses'][comp] = dict(_hyp, kind='baseline')

_ha_comp = hypothesis_audit(HYPOTHESIS_DESIGN, 'component', names=list(design))

_ha_base = hypothesis_audit(BASELINE_HYPOTHESIS_DESIGN, 'baseline', names=list(baselines))

report['hypothesis_audit'] = {'component': _ha_comp, 'baseline': _ha_base}

for _ha in (_ha_comp, _ha_base):
    for _w in _ha.get('warnings') or []:
        report['warnings'].append(_w)
    report['hypotheses_incomplete'].extend((n for n in _ha.get('invalid') or [] if n not in report['hypotheses_incomplete']))
    report['hypotheses_incomplete'].extend((n for n in _ha.get('metric_unknown') or [] if n not in report['hypotheses_incomplete']))

_present_set = set(present)

_scoped_design = _scope_design(design, _present_set)

_scoped_baselines = _scope_design(baselines, _present_set)

if _scoped_design:
    report['ablation_completeness'] = ablation_completeness_audit(_scoped_design, _present_set)
    report['ablation_complete'] = bool(report['ablation_completeness'].get('complete'))
    report['incomplete_components'] = list(report['ablation_completeness'].get('incomplete_components') or [])
    report['missing_ablation_arms'] = dict(report['ablation_completeness'].get('missing_arms') or {})
    for _comp in report['ablation_completeness'].get('incomplete_components') or []:
        _miss = report['ablation_completeness'].get('missing_arms', {}).get(_comp) or []
        report['warnings'].append('%s: ablation is INCOMPLETE — missing arm(s) %s; this component cannot support a "useful" claim until every declared contrast is present' % (_comp, ', '.join(_miss) or '(unspecified)'))
else:
    report['ablation_completeness'] = {'mode': 'leave-one-out', 'complete': None, 'components': {}, 'complete_components': [], 'incomplete_components': [], 'missing_arms': {}}
    report['incomplete_components'] = []
    report['missing_ablation_arms'] = {}
    report['ablation_complete'] = None

def _slice_fp(r):
    return tuple((str(r.get(k) or '') for k in ('val_fp', 'eval_fp', 'holdout_row_fp'))) + (measurement_semantics_key(r),)

_arm_groups = {}

for (v, t, s), r in recs.items():
    _arm_groups.setdefault((v, str(t or ''), _slice_fp(r)), {})[s] = r

_base_counts = {}

for v, t, _fpk in _arm_groups:
    _base_counts[v, t] = _base_counts.get((v, t), 0) + 1

_base_seen = {}

for (v, t, _fpk), _by_seed in sorted(_arm_groups.items()):
    _recs_sorted = [_by_seed[s] for s in sorted(_by_seed)]
    _primary = [_record_primary_ppl(r) for r in _recs_sorted]
    _miss = sum((v is None for v in _primary))
    _nested_flags = {r.get('ppl_estimation_rows') == 'nested_validation_holdout' for r in _recs_sorted}
    _mixed_estimands = len(_nested_flags) > 1
    if _mixed_estimands:
        report['warnings'].append('%s: records mix nested-holdout and full-validation estimands; the arm summary falls back to full-validation PPL for the whole arm instead of averaging two different samples' % (v if not t else f'{v}::{t}'))
    if _miss or _mixed_estimands:
        _vals = [float(r['ppl']) for r in _recs_sorted]
        _estimand = 'full_validation_fallback'
        if _miss and (not _mixed_estimands):
            report['warnings'].append('%s: %d/%d record(s) declare the nested estimand but lack holdout accounting; arm summary falls back to full-validation PPL for the whole arm' % (v if not t else f'{v}::{t}', _miss, len(_recs_sorted)))
        _pooled, _nll_tot, _ntok_tot = _pooled_ppl_from_records(_recs_sorted, estimand=False)
    else:
        _vals = [float(x) for x in _primary]
        _estimand = 'nested_validation_holdout' if _nested_flags == {True} else 'full_validation'
        _pooled, _nll_tot, _ntok_tot = _pooled_ppl_from_records(_recs_sorted)
    _key = v if not t else '%s::%s' % (v, t)
    if _base_counts.get((v, t), 0) > 1:
        _idx = _base_seen.get((v, t), 0)
        _base_seen[v, t] = _idx + 1
        _key = '%s::slice%d' % (_key, _idx)
        report['warnings'].append('%s: records span different validation slices or code-semantics stamps; arm summary is split instead of pooling different documents/implementations' % _key)
    report['arm_summary'][_key] = {'variant': v, 'protocol': t, 'n_seeds': len(_vals), 'seeds': sorted(_by_seed), 'ppl_mean': float(np.mean(_vals)), 'ppl_mean_arithmetic': float(np.mean(_vals)), 'ppl_pooled': _pooled, 'ppl_primary': float(_pooled) if _pooled is not None else float(np.mean(_vals)), 'ppl_estimand': _estimand, 'ppl_primary_missing': int(_miss), 'nll_sum': _nll_tot, 'n_tokens': _ntok_tot, 'ppl_std': float(np.std(_vals, ddof=1)) if len(_vals) > 1 else 0.0, 'ppl_std_defined': bool(len(_vals) > 1), 'meets_min_seeds': len(_vals) >= int(min_n)}
    if len(_vals) < int(min_n):
        report['arms_below_min_seeds'].append(_key)

declared_m = sum((len(v) for v in _scoped_design.values())) + sum((len(v) for v in _scoped_baselines.values()))

all_entries = [e for group in (comp_entries, base_entries) for entries in group.values() for e in entries]

_family_with_missing = _family_size_with_missing(_scoped_design, comp_entries) + _family_size_with_missing(_scoped_baselines, base_entries)

m_family = max(declared_m, len(all_entries), _family_with_missing)

_holm_assign(all_entries, m_family, alpha, 'p_holm_global_naive')

_holm_assign(all_entries, m_family, alpha, 'p_holm_global', p_key='p_look_adjusted')

_holm_assign(all_entries, m_family, alpha, 'p_holm_family', p_key='p_look_adjusted')

for e in all_entries:
    _phn = e.get('p_holm_global_naive')
    e['significant_global_naive'] = bool(_phn is not None and float(_phn) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))

for e in all_entries:
    _ph = e.get('p_holm_global')
    e['significant_global'] = bool(_ph is not None and float(_ph) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
    e['significant_family'] = e['significant_global']

_sesoi_family = _apply_sesoi_holm(comp_entries, _scoped_design, base_entries, _scoped_baselines, _sesoi_specs, alpha=alpha, min_n=min_n)

report['interactions'] = factorial_interaction_audit(recs=recs, alpha=alpha, min_n=min_n, look_history=look_history)

for _iw in report['interactions'].get('warnings') or []:
    report['warnings'].append('interaction: ' + str(_iw))

report['family'] = {'name': ('ABLATION_DESIGN+BASELINE_DESIGN' if _scoped_baselines else 'ABLATION_DESIGN') + ' (exploratory; not pre-registered)', 'declared_contrasts': int(declared_m), 'effective_m': int(m_family), 'alpha': float(alpha), 'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha), 'note': 'Holm step-down over every declared contrast; missing arms keep their slot (dummy p=1). A panel below min_seeds_for_family_significance is resolution-limited, however large the observed effect.'}

def _publish(group_name, design_group, entries_by_comp, meta, kind, require_param_match):
    status_names = ('verified', 'system_level_only', 'direction_conflict', 'direction_unspecified', 'equivalent', 'sub_sesoi', 'practically_indeterminate', 'measured', 'significant', 'nominally_significant', 'resolution_limited', 'underpowered', 'noise_level', 'unverified')
    lists = {s: [] for s in status_names}
    for comp, contrasts in design_group.items():
        entries = entries_by_comp.get(comp, [])
        _c_items = contrasts if isinstance(contrasts, (list, tuple)) else []
        _n_comp = _declared_contrast_count({comp: contrasts})
        m_comp = max(_n_comp, _family_size_with_missing({comp: contrasts}, {comp: entries}))
        _holm_assign(entries, m_comp, alpha, 'p_holm_component')
        for e in entries:
            _phc = e.get('p_holm_component')
            e['significant_component'] = bool(_phc is not None and float(_phc) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
        status = _classify_group_status(entries, min_n=min_n, alpha=alpha, m_family=m_family, require_param_match=require_param_match, n_declared=_n_comp, declared_pairs=[(c[0], c[1]) for c in _c_items if isinstance(c, (tuple, list)) and len(c) >= 2 and isinstance(c[0], str) and isinstance(c[1], str) and c[0].strip() and c[1].strip()])
        lists[status].append(comp)
        if status == 'verified':
            lists['significant'].append(comp)
        if any((e.get('significant') for e in entries if e.get('controlled'))):
            lists['nominally_significant'].append(comp)
        if any((int(e.get('n') or 0) >= int(min_n) and e.get('controlled') for e in entries)):
            lists['measured'].append(comp)
        report[group_name][comp] = entries
        _arms = meta[comp]['arms']
        _cv = controlled_variables_report(recs, _arms)
        _cv.update({'arms_present': sorted((a for a in _arms if a in present)), 'param_matched': all((e.get('param_controlled') is True for e in entries)) if entries else None, 'objective_matched': all((e.get('objective_matched') is True for e in entries)) if entries else None, 'objective_differences': sorted({x for e in entries for x in e.get('objective_differences') or [] if 'differs' in str(x)})})
        if _cv.get('objective_differences'):
            report['warnings'].append('%s: arms share run_cfg but not the actual training objective (%s) — the contrast is a system-level comparison, not an isolated mechanism claim' % (comp, '; '.join(_cv['objective_differences'])))
        if _cv.get('code_equal') is False:
            report['warnings'].append('%s: arms were measured under different code-semantics stamps (%s) — the two numbers were produced by different implementations, so this is not a controlled pair' % (comp, _cv.get('code_values')))
        report['controlled_variables'][comp] = _cv
        if _cv.get('within_arm_mixed'):
            report['warnings'].append('%s: arm(s) %s mix DIFFERENT protocols within one arm across seeds — a seed-averaged claim would pool incompatible runs' % (comp, ', '.join(_cv['within_arm_mixed'])))
        if _cv.get('partially_recorded'):
            report['warnings'].append('%s: arm(s) %s carry the run_cfg/train_knobs/hyperparameter descriptor only on SOME seed records — equality across arms cannot be verified; the seeds without the descriptor are already excluded from the paired n' % (comp, ', '.join(_cv['partially_recorded'])))
        if _cv.get('train_knobs_defects'):
            _detail = '; '.join(('%s: %s' % (a, ', '.join(sorted({d['reason'] for d in ds}))) for a, ds in sorted(_cv['train_knobs_defects'].items())))
            report['warnings'].append('%s: present train_knobs descriptor(s) are incomplete or malformed (%s) — an incomplete descriptor cannot prove the optimizer/schedule/augmentation recipe was controlled, so the affected pairs are refused rather than treated as legacy metadata' % (comp, _detail))
        if _cv.get('hyperparam_differences'):
            report['warnings'].append('%s: structured hyperparameters differ across arms (%s) even where run_cfg matches — the comparison is not fully controlled' % (comp, ', '.join(_cv['hyperparam_differences'])))
        if _cv.get('seed_coverage_equal') is False:
            _cov = '; '.join(('%s={%s}' % (a, ','.join(('%s:%s' % (t or 'base', s) for t, s in _cv['seed_coverage'].get(a) or []))) for a in _cv.get('arms') or []))
            report['warnings'].append('%s: arms do NOT share the same seed/protocol coverage (%s) — arm means are over different seed sets and the extra/missing seeds must not be read as a paired contrast; pair only the shared seeds (n_paired) or restore the missing cells' % (comp, _cov))
        _req_key = 'component_seed_requirements' if kind == 'component' else 'baseline_seed_requirements'
        report[_req_key][comp] = {'declared_contrasts': int(_n_comp), 'min_seeds_for_component_significance': signflip_min_n_for_holm(_n_comp, alpha), 'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha)}
    for status in status_names:
        report[f'{status}_{group_name}'] = lists[status]
    return lists

_publish('components', design, comp_entries, comp_meta, 'component', require_param_match=False)

_publish('baselines', baselines, base_entries, base_meta, 'baseline', require_param_match=True)

report['component_draws'] = {}

for _group, _group_design, _group_meta in (('components', design, comp_meta), ('baselines', baselines, base_meta)):
    for _comp in _group_design:
        _arms = (_group_meta.get(_comp) or {}).get('arms') or []
        report['component_draws'][_comp] = component_draw_identity(recs, _arms)

_sesoi_components = {}

_sesoi_lists = {'useful': [], 'equivalent': [], 'sub_sesoi': [], 'harmful': [], 'inconclusive': [], 'not_measured': [], 'not_declared': []}

_present_arms = set(present)

for _group in ('components', 'baselines'):
    for _comp, _entries in (report.get(_group) or {}).items():
        if _comp not in _sesoi_specs:
            continue
        _declared_entries = [e for e in _entries if (e.get('sesoi') or {}).get('declared')]
        _dec = [e.get('practical_decision') for e in _declared_entries]
        _design_contrasts = ((design if _group == 'components' else baselines) or {}).get(_comp) or []
        _declared_pairs = []
        for _c in _design_contrasts:
            if isinstance(_c, (tuple, list)) and len(_c) >= 2:
                _declared_pairs.append((_c[0], _c[1]))
        _powered_pairs = {(e.get('with'), e.get('without')) for e in _entries if e.get('controlled') and (_as_seed_int(e.get('n')) or 0) >= int(min_n)}
        _all_declared_powered = bool(_declared_pairs) and set(_declared_pairs).issubset(_powered_pairs)
        if _group == 'components':
            _comp_complete = bool((((report.get('ablation_completeness') or {}).get('components') or {}).get(_comp) or {}).get('complete'))
        else:
            _comp_complete = bool(_declared_pairs) and all((_a in _present_arms and _b in _present_arms for _a, _b in _declared_pairs))
        if not _declared_entries:
            _status = 'not_measured'
        elif any((d == 'not_declared' for d in _dec)):
            _status = 'not_declared'
        elif any((d == 'inferior' for d in _dec)):
            _status = 'harmful'
        elif any((d == 'superior' for d in _dec)):
            _status = 'useful'
        elif _dec and all((d == 'equivalent' for d in _dec)) and _comp_complete and _all_declared_powered:
            _status = 'equivalent'
        elif _dec and all((d in ('sub_sesoi', 'equivalent') for d in _dec)) and any((d == 'sub_sesoi' for d in _dec)):
            _status = 'sub_sesoi'
        else:
            _status = 'inconclusive'
        _sesoi_lists[_status].append(_comp)
        _sesoi_components[_comp] = {'group': _group, 'margin_spec': _sesoi_specs[_comp], 'status': _status, 'ablation_complete': _comp_complete, 'all_declared_contrasts_powered': bool(_all_declared_powered), 'decisions': [{'with': e.get('with'), 'without': e.get('without'), 'n': (e.get('sesoi') or {}).get('n'), 'mean': (e.get('sesoi') or {}).get('mean'), 'margin_ppl': (e.get('sesoi') or {}).get('margin_ppl'), 'decision': e.get('practical_decision'), 'claim': (e.get('sesoi') or {}).get('claim'), 'p_tost': (e.get('sesoi') or {}).get('tost_p_tost'), 'p_tost_holm': (e.get('sesoi') or {}).get('tost_p_holm'), 'ci_tost': (e.get('sesoi') or {}).get('tost_ci'), 'reason': (e.get('sesoi') or {}).get('reason')} for e in _declared_entries]}
        if _dec and all((d == 'equivalent' for d in _dec)) and (not (_comp_complete and _all_declared_powered)):
            report['warnings'].append('%s: the measured contrast(es) are TOST-equivalent, but the declared ablation is INCOMPLETE or has a missing/underpowered contrast — equivalence on that partial panel is descriptive only and cannot support "remove the component"' % _comp)
        if _status == 'useful' and (not (_comp_complete and _all_declared_powered)):
            report['warnings'].append('%s: at least one contrast is practically useful, but the declared ablation is INCOMPLETE or has a missing/underpowered contrast — report this as a contrast-level result, not as "the component is useful" for the whole innovation' % _comp)
        _spec = _sesoi_specs[_comp]
        _margin_desc = '%.2f%% of the without-arm PPL' % float(_spec['relative_pct']) if _spec.get('relative_pct') is not None else '%.4g PPL (absolute)' % float(_spec['absolute_ppl'])
        if _status == 'sub_sesoi':
            report['warnings'].append('%s: the effect is statistically detectable but smaller than the pre-registered SESOI (%s); do not quote it as a practically useful module contribution' % (_comp, _margin_desc))
        elif _status == 'equivalent':
            report['warnings'].append('%s: TOST places the ablation effect inside the pre-registered SESOI (%s) after family adjustment — this is positive evidence of practical removability, not a non-significant null' % (_comp, _margin_desc))
        elif _status == 'harmful':
            report['warnings'].append('%s: the module-removal contrast is significant and practically HARMFUL in the declared direction (effect larger than the SESOI); the mechanism claim is contradicted, not merely unsupported' % _comp)
        elif _status == 'inconclusive':
            report['warnings'].append('%s: neither a practically useful effect (>SESOI) nor TOST equivalence inside the SESOI is established; report it as inconclusive rather than as a null or a win' % _comp)

report['sesoi'] = {'policy': 'pre-registered relative SESOI with paired TOST for ablation components and parameter-matched baselines; exploratory panels only (the frozen confirmatory manifest is unchanged)', 'alpha': float(alpha), 'min_n': int(min_n), 'declared_components': sorted(set(_sesoi_specs) & set(design)), 'declared_baselines': sorted(set(_sesoi_specs) & set(baselines)), 'family': _sesoi_family, 'components': _sesoi_components, 'useful_components': sorted(_sesoi_lists['useful']), 'equivalent_components': sorted(_sesoi_lists['equivalent']), 'sub_sesoi_components': sorted(_sesoi_lists['sub_sesoi']), 'harmful_components': sorted(_sesoi_lists['harmful']), 'inconclusive_components': sorted(_sesoi_lists['inconclusive']), 'not_measured_components': sorted(_sesoi_lists['not_measured']), 'not_declared_components': sorted(_sesoi_lists['not_declared'])}

report['practically_useful_components'] = list(report['sesoi']['useful_components'])

report['practically_harmful_components'] = list(report['sesoi']['harmful_components'])

report['practically_inconclusive_components'] = sorted(set(report['sesoi']['inconclusive_components']) | set(report['sesoi']['sub_sesoi_components']) | set(report['sesoi']['not_declared_components']))

if report.get('ablation_completeness'):
    report['ablation_completeness']['omitted_components'] = list(report.get('unverified_components') or [])
    if report['ablation_completeness'].get('complete') is None:
        report['ablation_complete'] = None
    else:
        report['ablation_complete'] = bool(report['ablation_completeness'].get('complete')) and (not report.get('unverified_components'))

report['metric_audit'] = metric_policy_audit(summary)

if report['metric_audit'].get('warnings'):
    report['warnings'].extend(report['metric_audit']['warnings'])

_tune_path = _baseline_artifact_path('baseline_tuning.json', outdir)

report['baseline_tuning'] = baseline_tuning_audit(path=_tune_path)

_bt = report['baseline_tuning']

_bt_claim_reasons = []

if not _bt.get('tuning_verified'):
    _bt_claim_reasons.append('equal-search tuning artifact not verified')

if _bt.get('grid_adequacy') not in (None, 'ok'):
    _bt_claim_reasons.append('grid_adequacy=%r' % (_bt.get('grid_adequacy'),))

if _bt.get('search_complete') is False:
    _bt_claim_reasons.append('search grid was truncated')

_bt_transfer = _bt.get('budget_transfer') or {}

if isinstance(_bt_transfer, dict) and _bt_transfer.get('equal') is False:
    _bt_claim_reasons.append('hyperparameters were selected at a different training horizon than the final records (search_steps=%r, final=%r)' % (_bt_transfer.get('search_steps'), _bt_transfer.get('final_steps_by_arm')))

if isinstance(_bt_transfer, dict) and _bt_transfer.get('tokens_equal') is False:
    _bt_claim_reasons.append('hyperparameters were selected on a different token exposure than the final records (trial=%r, final=%r)' % (_bt_transfer.get('trial_tokens_by_arm'), _bt_transfer.get('final_tokens_by_arm')))

_bt_stability = _bt.get('selection_stability') or {}

if any((isinstance(v, dict) and v.get('status') == 'ok' and (v.get('stable') is False) for v in _bt_stability.values())):
    _bt_claim_reasons.append('selected config is not leave-one-search-seed-out stable')

_bt['tuned_claim_allowed'] = bool(_bt.get('tuning_verified') and (not _bt_claim_reasons))

_bt['tuned_claim_reasons'] = _bt_claim_reasons

for _bt_w in _bt.get('warnings') or []:
    report['warnings'].append('baseline tuning: ' + str(_bt_w))

if _scoped_baselines:
    _panel_arms = set()
    for _cs in _scoped_baselines.values():
        for _a, _b, _n in _cs:
            _panel_arms.update((_a, _b))
    _cov = report['baseline_tuning'].get('coverage') or {}
    _sel_cfg = report['baseline_tuning'].get('selected_configs') or {}
    _uncovered = sorted((a for a in _panel_arms if a not in _cov))
    _panel_cfg_ok = {}
    for _arm in _panel_arms:
        _ref = _cov.get(_arm, _arm)
        _sel = _sel_cfg.get(_ref) or _sel_cfg.get(_arm)
        _arm_recs = [r for (v, _t, _s), r in recs.items() if v == _arm and isinstance(r, dict)]
        _hps = [r.get('hyperparams') for r in _arm_recs if isinstance(r.get('hyperparams'), dict)]
        if not isinstance(_sel, dict) or not _hps:
            _panel_cfg_ok[_arm] = False
            continue
        if len(_hps) != len(_arm_recs):
            _panel_cfg_ok[_arm] = False
            report['warnings'].append('%s: only %d/%d panel record(s) carry structured hyperparameters; the tuned-config claim is unverifiable for the rest' % (_arm, len(_hps), len(_arm_recs)))
            continue
        _ok = True
        for _kn in ('lr', 'weight_decay', 'warmup'):
            if _sel.get(_kn) is None:
                continue
            _vals = [h.get(_kn) for h in _hps]
            if any((v is None for v in _vals)):
                _ok = False
                break
            try:
                if any((float(v) != float(_sel[_kn]) for v in _vals)):
                    _ok = False
                    break
            except (TypeError, ValueError, OverflowError):
                _ok = False
                break
        _panel_cfg_ok[_arm] = _ok
    _mismatched = sorted((a for a in _panel_arms if not _panel_cfg_ok.get(a)))
    report['baseline_tuning']['panel_baseline_arms'] = sorted(_panel_arms)
    report['baseline_tuning']['panel_uncovered_arms'] = _uncovered
    report['baseline_tuning']['panel_arm_hyperparams_match'] = _panel_cfg_ok
    _tuned_comps = set()
    for _comp, _contrasts in _scoped_baselines.items():
        _arms = {a for _a, _b, _n in _contrasts for a in (_a, _b)}
        if _arms and _bt.get('tuned_claim_allowed') and all((_panel_cfg_ok.get(a) for a in _arms)):
            _tuned_comps.add(_comp)
    _verified = list(report.get('verified_baselines') or [])
    report['tuned_verified_baselines'] = [c for c in _verified if c in _tuned_comps]
    report['untuned_verified_baselines'] = [c for c in _verified if c not in _tuned_comps]
    _dir_conf = sorted((c for c in report.get('direction_conflict_baselines') or [] if c in set(_scoped_baselines)))
    _tuned_v = list(report['tuned_verified_baselines'])
    _untuned_v = list(report['untuned_verified_baselines'])
    if _dir_conf:
        _tuned_status = 'falsified'
    elif not _bt.get('tuning_verified'):
        _tuned_status = 'unverified'
    elif _tuned_v and (not _untuned_v) and (not _uncovered) and (not _mismatched):
        _tuned_status = 'supported'
    elif _tuned_v:
        _tuned_status = 'partial'
    else:
        _tuned_status = 'unresolved'
    _tuned_obs = -1 if _tuned_status == 'supported' else 1 if _tuned_status == 'falsified' else None
    report['baseline_tuning']['tuned_hypothesis'] = dict(TUNED_BASELINE_HYPOTHESIS_DESIGN['method_beats_tuned_baseline'], observed_status=_tuned_status, observed_sign=_tuned_obs, direction_consistent=None if _tuned_obs is None else bool(_tuned_obs == -1), tuned_verified_baselines=_tuned_v, untuned_verified_baselines=_untuned_v, direction_conflict_baselines=_dir_conf, panel_uncovered_arms=_uncovered, panel_arm_hyperparams_mismatch=_mismatched)
    if _tuned_status == 'falsified':
        report['warnings'].append('the tuned-baseline hypothesis is FALSIFIED: a matched dense baseline is significantly better than the declared method arm (%s), so the method cannot be quoted as beating a tuned baseline' % ', '.join(_dir_conf))
    elif _tuned_status == 'partial':
        report['warnings'].append('the tuned-baseline hypothesis is only PARTIAL: some verified baseline contrasts were not run at the selected tuned configuration (%s)' % ', '.join(_untuned_v))
    elif _tuned_status == 'unresolved':
        report['warnings'].append('the tuned-baseline hypothesis is UNRESOLVED: no declared method-vs-baseline contrast is verified after family correction at the current seed count')
    if _bt.get('tuning_verified') and (not _bt.get('tuned_claim_allowed')) and _verified:
        report['warnings'].append('baseline tuning is verified as an equal-search grid but does not support a "well-tuned baseline" claim (%s); the verified baseline contrasts stay in untuned_verified_baselines until the search is adequate' % '; '.join(_bt_claim_reasons))
    if not report['baseline_tuning'].get('tuning_verified'):
        report['warnings'].append('baseline claims have no verified equal-search tuning artifact (%s) — a win over an untuned baseline is not evidence that the method beats a well-tuned baseline; run baseline_tune.py and put baseline_tuning.json next to this panel' % (report['baseline_tuning'].get('status') or 'missing'))
    elif _uncovered:
        report['warnings'].append('the tuning artifact does not cover panel arm(s) %s — equal-search evidence for another arm does not transfer automatically; tune these arms (or declare an explicit inherits_tuning parent) before claiming a tuned-baseline win for them' % ', '.join(_uncovered))
    elif _mismatched:
        report['warnings'].append('panel arm(s) %s were not trained at the tuned config (recorded hyperparameters differ from baseline_tuning.selected_configs) — their significant contrasts stay UNTUNED-baseline evidence; run the final tuned configs before claiming a win over a tuned baseline' % ', '.join(_mismatched))

_sb_path = _baseline_artifact_path('simple_baseline.json', outdir)

report['simple_baseline'] = simple_baseline_audit(path=_sb_path)

for _sb_w in report['simple_baseline'].get('warnings') or []:
    report['warnings'].append('simple baseline: ' + str(_sb_w))
