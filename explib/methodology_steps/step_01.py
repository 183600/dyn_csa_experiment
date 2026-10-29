if _scoped_baselines:
    _sb = report['simple_baseline']
    _sb_claim_reasons = []
    if _sb.get('grid_adequacy') not in (None, 'ok'):
        _sb_claim_reasons.append('count-model grid adequacy=%r' % (_sb.get('grid_adequacy'),))
    if _sb.get('grid_boundary_hits'):
        _sb_claim_reasons.append('selected smoothing on a grid boundary (%s)' % ', '.join(map(str, _sb.get('grid_boundary_hits') or [])))
    if _sb.get('selection_tie_arms'):
        _sb_claim_reasons.append('selected smoothing tied with the runner-up for %s' % ', '.join(map(str, _sb.get('selection_tie_arms') or [])))
    if _sb.get('fit_truncated') is True:
        _fb = _sb.get('fit_budget') or {}
        _sb_claim_reasons.append("the count model was fitted on a truncated prefix (%r/%r tokens, max_fit_tokens=%r), so it is not a floor trained on the panel's full corpus" % (_fb.get('n_fit_tokens'), _fb.get('n_source_tokens'), _fb.get('max_fit_tokens')))
    _sb['tuned_floor_claim_allowed'] = bool(_sb.get('baseline_verified') and (not _sb_claim_reasons))
    _sb['tuned_floor_claim_reasons'] = _sb_claim_reasons
    if _sb.get('baseline_verified') and _sb_claim_reasons:
        report['warnings'].append('the count-model floor passed its structural audit but is not adequately tuned (%s) — a beats-floor claim is against a weakly tuned floor; extend the alpha grid and re-run before quoting it' % '; '.join(_sb_claim_reasons))
    if not _sb.get('baseline_verified'):
        report['warnings'].append('no verified tuned count-model floor (%s) — run baseline_tune.py simple and place simple_baseline.json next to this panel; a neural model that has not been shown to beat the trivial floor should not be interpreted as evidence for the mechanism' % (_sb.get('status') or 'missing'))
    else:
        _main_arms = set()
        for _cs in list(_scoped_design.values()) + list(_scoped_baselines.values()):
            for _a, _b, _n in _cs:
                _main_arms.add(_a)
                _main_arms.add(_b)
        _cands = [(k, v) for k, v in (report.get('arm_summary') or {}).items() if v.get('variant') in _main_arms]
        _base_cands = [(k, v) for k, v in _cands if not v.get('protocol')]
        if _base_cands:
            _cands = _base_cands
        if _cands:
            _ref_key, _ref = min(_cands, key=lambda kv: float(kv[1].get('ppl_primary', kv[1]['ppl_mean'])))
            _panel_val_fps = {r.get('val_fp') for r in recs.values() if r.get('val_fp')}
            _panel_eval_fps = {r.get('eval_fp') for r in recs.values() if r.get('eval_fp')}
            _panel_holdout_fps = {r.get('holdout_row_fp') for r in recs.values() if r.get('holdout_row_fp')}
            _sb_data = _sb.get('data') or {}
            _sb_val_fp = _sb_data.get('cache_val_fp') or _sb_data.get('val_fp')
            _sb_eval_fp = _sb_data.get('eval_fp')
            _sb_row_split = _sb_data.get('row_split') if isinstance(_sb_data.get('row_split'), dict) else {}
            _sb_nested = bool(_sb_data.get('nested_selection') is True or _sb_row_split)
            _sb_holdout_row_fp = _sb_row_split.get('evaluation_row_fp')
            _panel_nested = any((isinstance(r, dict) and r.get('ppl_estimation_rows') == 'nested_validation_holdout' for r in recs.values()))
            _comparable = None
            if _sb_val_fp and _panel_val_fps:
                _comparable = _sb_val_fp in _panel_val_fps
            if _comparable is True and _panel_nested and (not _sb_nested):
                _comparable = False
                report['warnings'].append('the panel uses the nested held-out estimand but the simple-baseline floor was scored on full validation; the beats-floor comparison is refused')
            if _comparable is True:
                if _sb_nested:
                    _tune_split = (report.get('baseline_tuning') or {}).get('selection_row_split') or {}
                    _sb_sel_row_fp = _sb_row_split.get('selection_row_fp')
                    if _tune_split.get('selection_row_fp') and _sb_sel_row_fp and (_tune_split.get('selection_row_fp') != _sb_sel_row_fp):
                        _comparable = False
                        report['warnings'].append('nested simple-baseline selection row fingerprint differs from the neural tuning selection rows — the floor and the panel were tuned on different documents; the beats-floor comparison is refused')
                    if _sb_holdout_row_fp and _panel_holdout_fps:
                        if _sb_holdout_row_fp not in _panel_holdout_fps:
                            _comparable = False
                            report['warnings'].append('nested simple-baseline holdout row fingerprint differs from the panel holdout rows — the floor and the panel scored different validation documents; the beats-floor comparison is refused')
                    else:
                        _comparable = False
                        report['warnings'].append('nested simple-baseline holdout rows cannot be matched to the panel (missing evaluation_row_fp or panel holdout_row_fp) — the floor and the panel may use different estimands; the beats-floor comparison is refused')
                elif _sb_eval_fp and _panel_eval_fps:
                    if _sb_eval_fp not in _panel_eval_fps:
                        _comparable = False
                        report['warnings'].append('simple-baseline eval row fingerprint differs from the panel final-eval rows — the floor and the panel scored different validation documents; the beats-floor comparison is refused')
            _panel_ppl = float(_ref.get('ppl_primary', _ref['ppl_mean']))
            _floor_ppl = float(_sb.get('best_ppl')) if _sb.get('best_ppl') is not None else None
            _panel_tokens_per_seed = None
            _n_seed = _as_seed_int(_ref.get('n_seeds'))
            _n_tok = _exact_count(_ref.get('n_tokens'), minimum=1)
            if _n_seed and _n_tok and (_n_tok % _n_seed == 0):
                _panel_tokens_per_seed = int(_n_tok // _n_seed)
            _floor_tokens = _exact_count(_sb_data.get('n_eval_tokens'), minimum=1)
            _token_match = None
            if _panel_tokens_per_seed is not None and _floor_tokens is not None:
                _token_match = bool(_panel_tokens_per_seed == _floor_tokens)
                if not _token_match:
                    report['warnings'].append("simple-baseline evaluated %s tokens but the panel's held-out estimate covers %s tokens per seed — the PPLs are not on the same sample; the beats-floor comparison is refused" % (f'{_floor_tokens:,}', f'{_panel_tokens_per_seed:,}'))
                    if _comparable is True:
                        _comparable = False
            _floor_cmp = None
            if _comparable is True and _floor_ppl is not None:
                _ref_variant = _ref.get('variant')
                _ref_protocol = _ref.get('protocol') or ''
                _seed_vals = [float(_ppl) for (v, t, _s), r in recs.items() if v == _ref_variant and (t or '') == _ref_protocol and ((_ppl := _record_primary_ppl(r)) is not None)]
                if _seed_vals:
                    _deltas = [x - _floor_ppl for x in _seed_vals]
                    _fst = paired_effect_stats(_deltas, alpha=alpha, min_n=int(report.get('min_n') or 3))
                    # paired_effect_stats already uses the guarded mean; do
                    # not re-average the raw deltas here (their naive sum can
                    # overflow even when the mean is representable).
                    _pt = _fst.get('mean')
                    if _pt is None or not math.isfinite(float(_pt)):
                        _pt = _scaled_mean_or_raw(_deltas)
                    if _pt < 0:
                        if _fst.get('significant'):
                            _verdict = 'beats_floor_significant'
                        elif _fst.get('underpowered') or _fst.get('resolution_limited'):
                            _verdict = 'beats_floor_pointwise_underpowered'
                        else:
                            _verdict = 'beats_floor_pointwise_not_significant'
                    else:
                        _verdict = 'does_not_beat_floor'
                    _floor_cmp = {'n': int(_fst.get('n') or 0), 'mean_delta': _pt, 'std': _fst.get('std'), 'sem': _fst.get('sem'), 'p_exact_signflip': _fst.get('p_exact_signflip'), 'p_floor': _fst.get('p_floor'), 'ci95_low': _fst.get('ci95_low'), 'ci95_high': _fst.get('ci95_high'), 'within_seed_noise': _fst.get('within_seed_noise'), 'underpowered': _fst.get('underpowered'), 'resolution_limited': _fst.get('resolution_limited'), 'significant': _fst.get('significant'), 'verdict': _verdict, 'note': 'delta = panel seed PPL - floor PPL; negative means the panel beats the floor. The seed-level test treats the floor as fixed; the paired row bootstrap (when per-row floor accounting exists) also covers evaluation-row uncertainty.'}
                    if _verdict == 'beats_floor_pointwise_not_significant':
                        report['warnings'].append('the panel is pointwise below the tuned count-model floor (%.4f vs %.4f) but the seed-level floor contrast is NOT significant (p=%.3g, n=%d) — do not describe this as beating the floor' % (_panel_ppl, _floor_ppl, float(_fst.get('p_exact_signflip') or float('nan')), int(_fst.get('n') or 0)))
                    elif _verdict == 'beats_floor_pointwise_underpowered':
                        report['warnings'].append('the panel is pointwise below the count-model floor but the floor contrast is resolution-limited at n=%d (p floor %.3g); add seeds before claiming a beats-floor win' % (int(_fst.get('n') or 0), float(_fst.get('p_floor') or float('nan'))))
            _method_arm_names = sorted({_a for _cs in _scoped_baselines.values() for _a, _b, _n in _cs})
            _floor_by_arm = {}
            _method_floor_status = 'unverified'
            if _comparable is True and _floor_ppl is not None:
                _by_variant = {}
                for _k, _v in _cands:
                    _by_variant.setdefault(_v.get('variant'), []).append((_k, _v))
                for _v, _vent in sorted(_by_variant.items(), key=lambda kv: str(kv[0])):
                    _vent = [kv for kv in _vent if not kv[1].get('protocol')] or _vent
                    _vk, _vs = min(_vent, key=lambda kv: float(kv[1].get('ppl_primary', kv[1].get('ppl_mean', float('inf')))))
                    _vp = _vs.get('protocol') or ''
                    _arm_recs = [r for (vv, tt, _s), r in recs.items() if vv == _v and (tt or '') == _vp and (_record_primary_ppl(r) is not None)]
                    _vals = [float(_record_primary_ppl(r)) for r in _arm_recs]
                    if not _vals:
                        continue
                    _deltas = [x - _floor_ppl for x in _vals]
                    _fst = paired_effect_stats(_deltas, alpha=alpha, min_n=int(report.get('min_n') or 3))
                    # paired_effect_stats already uses the guarded mean; do
                    # not re-average the raw deltas here (their naive sum can
                    # overflow even when the mean is representable).
                    _pt = _fst.get('mean')
                    if _pt is None or not math.isfinite(float(_pt)):
                        _pt = _scaled_mean_or_raw(_deltas)
                    if _pt < 0:
                        if _fst.get('significant'):
                            _av = 'beats_floor_significant'
                        elif _fst.get('underpowered') or _fst.get('resolution_limited'):
                            _av = 'beats_floor_pointwise_underpowered'
                        else:
                            _av = 'beats_floor_pointwise_not_significant'
                    else:
                        _av = 'does_not_beat_floor'
                    _floor_by_arm[_v] = {'n': int(_fst.get('n') or 0), 'protocol': _vp, 'mean_delta': _pt, 'std': _fst.get('std'), 'p_exact_signflip': _fst.get('p_exact_signflip'), 'p_floor': _fst.get('p_floor'), 'ci95_low': _fst.get('ci95_low'), 'ci95_high': _fst.get('ci95_high'), 'within_seed_noise': _fst.get('within_seed_noise'), 'underpowered': _fst.get('underpowered'), 'resolution_limited': _fst.get('resolution_limited'), 'verdict': _av}
                    _rep_nll = _sb.get('reported_nll_per_row')
                    _rep_tok = _sb.get('reported_n_tokens_per_row')
                    if isinstance(_rep_nll, list) and isinstance(_rep_tok, list) and _rep_nll and (len(_rep_nll) == len(_rep_tok)):

                        def _floor_record_for(rec):
                            if rec.get('ppl_estimation_rows') == 'nested_validation_holdout':
                                return {'ppl_estimation_rows': 'nested_validation_holdout', 'nll_per_row_holdout': _rep_nll, 'n_tokens_per_row_holdout': _rep_tok}
                            return {'nll_per_row': _rep_nll, 'n_tokens_per_row': _rep_tok}
                        _floor_by_arm[_v]['paired_row_bootstrap_delta'] = paired_row_bootstrap_delta([(r, _floor_record_for(r)) for r in _arm_recs])
                    else:
                        _floor_by_arm[_v]['paired_row_bootstrap_delta'] = {'status': 'unverifiable', 'reason': 'floor artifact lacks per-row accounting; the floor was treated as a fixed number'}
                _floor_named = []
                for _v in _method_arm_names:
                    _p = (_floor_by_arm.get(_v) or {}).get('p_exact_signflip')
                    try:
                        _p = float(_p)
                    except (TypeError, ValueError, OverflowError):
                        _p = float('nan')
                    _floor_named.append((_v, _p if math.isfinite(_p) else 1.0))
                _floor_holm = holm_adjust(_floor_named, alpha=alpha) if _floor_named else {}
                for _v in _method_arm_names:
                    _c = _floor_by_arm.get(_v)
                    if _c is None:
                        continue
                    _h = _floor_holm.get(_v)
                    _c['p_holm'] = (_h or {}).get('p_holm')
                    _c['declared_family_m'] = len(_floor_named)
                    _c['significant_holm'] = bool((_h or {}).get('reject') and _c.get('verdict') == 'beats_floor_significant')
                    if _c.get('verdict') == 'does_not_beat_floor':
                        _c['method_status'] = 'does_not_beat_floor'
                    elif _c.get('significant_holm'):
                        _c['method_status'] = 'beats_floor_holm'
                    elif _c.get('verdict') == 'beats_floor_significant':
                        _c['method_status'] = 'family_correction_not_met'
                    else:
                        _c['method_status'] = 'not_confirmed'
                _statuses = [(_floor_by_arm.get(_v) or {}).get('method_status') for _v in _method_arm_names]
                if _floor_cmp is not None and _ref.get('variant') in _floor_by_arm:
                    _floor_cmp['paired_row_bootstrap_delta'] = _floor_by_arm[_ref.get('variant')].get('paired_row_bootstrap_delta')
                if not _method_arm_names or any((s is None for s in _statuses)):
                    _method_floor_status = 'unverified'
                elif any((s == 'does_not_beat_floor' for s in _statuses)):
                    _method_floor_status = 'method_arm_does_not_beat_floor'
                elif all((s == 'beats_floor_holm' for s in _statuses)):
                    _method_floor_status = 'all_method_arms_beat_floor'
                else:
                    _method_floor_status = 'method_floor_contrast_unresolved'
            _method_arms_beat_floor = True if _method_floor_status == 'all_method_arms_beat_floor' else False if _method_floor_status == 'method_arm_does_not_beat_floor' else None
            if _method_floor_status == 'method_arm_does_not_beat_floor':
                _failed = sorted((v for v, c in _floor_by_arm.items() if c.get('verdict') == 'does_not_beat_floor'))
                report['warnings'].append('declared method arm(s) %s do NOT beat the tuned count-model floor (best_ppl=%.4f) — the mechanism claim lacks the most basic baseline, even if another panel arm clears it' % (', '.join(_failed), _floor_ppl))
            elif _method_floor_status == 'method_floor_contrast_unresolved':
                report['warnings'].append('the declared method arm(s) do not yet have a Holm-cleared beats-floor contrast — report the floor comparison as unconfirmed, not as a win')
            _floor_deltas = [(_floor_by_arm.get(_v) or {}).get('mean_delta') for _v in _method_arm_names if _finite_scalar((_floor_by_arm.get(_v) or {}).get('mean_delta')) is not None]
            _floor_obs = None
            if _floor_deltas:
                if all((float(v) < 0 for v in _floor_deltas)):
                    _floor_obs = -1
                elif all((float(v) > 0 for v in _floor_deltas)):
                    _floor_obs = 1
                else:
                    _floor_obs = 0
            if _method_floor_status == 'all_method_arms_beat_floor':
                _floor_h_status = 'supported'
            elif _method_floor_status == 'method_arm_does_not_beat_floor':
                _floor_h_status = 'falsified'
            elif _method_floor_status == 'method_floor_contrast_unresolved':
                _floor_h_status = 'unresolved'
            else:
                _floor_h_status = 'unverified'
            _floor_h = hypothesis_audit(FLOOR_HYPOTHESIS_DESIGN, 'floor')
            _floor_spec = dict(FLOOR_HYPOTHESIS_DESIGN['method_beats_tuned_count_floor'])
            _floor_spec.update({'observed_status': _floor_h_status, 'observed_sign': _floor_obs, 'direction_consistent': None if _floor_obs is None else bool(_floor_obs == -1), 'method_arm_status': _method_floor_status, 'audit_status': _floor_h.get('status')})
            report['simple_baseline'].update({'method_arm_floor_contrasts': _floor_by_arm, 'method_arm_floor_status': _method_floor_status, 'method_arms_beat_floor': _method_arms_beat_floor, 'panel_reference_arm': _ref_key, 'panel_reference_ppl': _panel_ppl, 'panel_reference_protocol': _ref.get('protocol'), 'panel_val_fp': sorted(_panel_val_fps)[:3] or None, 'panel_eval_fp': sorted(_panel_eval_fps)[:3] or None, 'panel_holdout_fp': sorted(_panel_holdout_fps)[:3] or None, 'simple_eval_fp': _sb_eval_fp, 'simple_holdout_row_fp': _sb_holdout_row_fp, 'simple_nested_selection': _sb_nested, 'panel_tokens_per_seed': _panel_tokens_per_seed, 'simple_eval_tokens': _floor_tokens, 'token_denominator_match': _token_match, 'comparable_to_panel': _comparable, 'floor_contrast': _floor_cmp, 'panel_beats_simple_floor': bool(_panel_ppl < _floor_ppl) if _comparable is True and _floor_ppl is not None else None, 'floor_hypothesis': _floor_spec})
            if _comparable is False:
                report['warnings'].append('simple-baseline artifact and this panel do not share a validation fingerprint — the floor comparison is NOT comparable and must not be quoted')
            elif _comparable is None:
                report['warnings'].append('the floor comparison is UNVERIFIABLE: panel records and/or the simple-baseline artifact lack a shared validation fingerprint; do not quote a beats-floor claim')
            elif report['simple_baseline'].get('panel_beats_simple_floor') is False:
                report['warnings'].append("the panel's best declared-arm PPL (%.4f, %s) does not beat the tuned count-model floor (%.4f, %s) — the mechanism claim lacks the most basic baseline" % (_panel_ppl, _ref_key, _floor_ppl, _sb.get('best_baseline') or 'count-model'))

if report['hypotheses_missing']:
    report['warnings'].append('no pre-registered hypothesis for: ' + ', '.join(sorted(set(report['hypotheses_missing']))) + ' — direction cannot be checked, so a significant result is at most measured')

report['baseline_claim_gate'] = baseline_claim_gate(report)

_bcg = report['baseline_claim_gate']

if _bcg.get('status') == 'fail':
    report['warnings'].append('baseline claim gate FAILED: ' + '; '.join(_bcg.get('reasons') or []))

_ix_report = report.get('interactions') or {}

_ix_cells = [rec.get('interaction') for rec in (_ix_report.get('factorials') or {}).values() if isinstance(rec, dict) and isinstance(rec.get('interaction'), dict)]

_look_entries = [e for e in all_entries if isinstance(e.get('look_analysis'), dict)]

_look_entries += [c for c in _ix_cells if isinstance(c.get('look_analysis'), dict)]

_ix_look_keys = {c['look_analysis'].get('contrast') for c in _ix_cells if isinstance(c.get('look_analysis'), dict) and c['look_analysis'].get('contrast')}

report['look_adjusted_contrasts'] = sorted(set(report['look_adjusted_contrasts']) | _ix_look_keys)

_look_keys = sorted({e['look_analysis'].get('contrast') for e in _look_entries if e['look_analysis'].get('contrast')})

_multi = [e for e in _look_entries if int(e['look_analysis'].get('n_looks') or 0) > 1]

_multi_keys = sorted({e['look_analysis'].get('contrast') for e in _multi if e['look_analysis'].get('contrast')})

_downgraded = set()

for e in _look_entries:
    if 'significant_global_naive' in e:
        if e.get('significant_global_naive') and (not e.get('significant_global')):
            _downgraded.add(e['look_analysis'].get('contrast'))
    elif e.get('significant_family_naive') and (not e.get('significant_family')):
        _downgraded.add(e['look_analysis'].get('contrast'))

_downgraded = sorted((k for k in _downgraded if k))

report['look_history'] = {'schema': LOOK_HISTORY_SCHEMA, 'panel_looks': int(max([int(e['look_analysis'].get('n_looks') or 0) for e in _look_entries] or [0])), 'n_contrasts_with_looks': len(_look_keys), 'n_contrasts_look_adjusted': len(_multi_keys), 'look_adjusted_contrasts': _multi_keys, 'downgraded_naive_significant': _downgraded, 'invalid': report_look_invalid, 'policy': 'each new seed set aggregated for this panel is one look; significance uses the exact max-over-looks sign-flip p, and the naive final-n p is kept only as a diagnostic'}

if _downgraded:
    report['warnings'].append('optional-stopping correction DOWNGRADED %d contrast(s): %s — the naive final-look family p was < alpha, but the exact max-over-looks p was not. Adding seeds after seeing a non-significant result is one of the looks, not a free rerun.' % (len(_downgraded), ', '.join(_downgraded[:5])))

if write and outdir:
    try:
        os.makedirs(outdir, exist_ok=True)
        atomic_write_json(os.path.join(outdir, 'methodology_audit.json'), report)
    except Exception as e:
        print('[audit] could not write methodology_audit.json: %s: %s' % (type(e).__name__, e))

if verbose:
    _print_methodology_audit(report)
