# ---- missing-data sensitivity: what the complete-case subset cannot see -----
# The decision rule already refuses to promote an incomplete paired panel.
# This step adds the quantitative readout that rule leaves out: a sharp
# partial-identification bound over a declared range of unobserved per-seed
# differences, the tipping value that would erase the observed sign / SESOI
# margin, and the cell-level coverage accounting.  It is descriptive: it does
# not touch significant/verified/Holm or any frozen verdict.

try:
    _missing_coverage = panel_missingness_audit(
        summary, planned_cells=planned_cells)
except (ValueError, TypeError) as _missing_exc:
    _missing_coverage = {
        'kind': 'panel_missingness_audit',
        'status': 'invalid',
        'reason': f'{type(_missing_exc).__name__}: {_missing_exc}',
        'plan_basis': None,
        'by_arm': {},
        'observed_seeds_union': sorted({s for (_v, _t, s) in recs}),
        'warnings': ['the planned-cell list is malformed; coverage is only '
                     'reported relative to the observed seed union'],
    }

if planned_cells:
    _missing_frame_basis = 'run_manifest_planned_cells'
else:
    _missing_frame_basis = 'observed_seed_union'
    _missing_coverage.setdefault('warnings', []).append(
        'no planned-cell list: the frame for each contrast is the union of '
        'observed seeds, so an unrun planned cell cannot be distinguished '
        'from a seed that was never scheduled')

_missing_rows = []
_missing_fragile = []
_missing_margin_not = []
_missing_no_observed = []
_missing_complete = []
for _missing_entry in all_entries:
    if not isinstance(_missing_entry, dict):
        continue
    _m_obs = [s for s in (_missing_entry.get('seeds') or [])]
    _m_skip = [s for s in (_missing_entry.get('skipped_seeds') or [])]
    _m_frame = set(_m_obs) | set(_m_skip)
    _m_planned_known = False
    if planned_cells:
        for _m_arm in (_missing_entry.get('with'), _missing_entry.get('without')):
            _m_plan = ((_missing_coverage.get('by_arm') or {}).get(_m_arm) or {}).get('planned_seeds')
            if _m_plan:
                _m_frame.update(_m_plan)
                _m_planned_known = True
    if not _m_planned_known:
        _m_frame.update(_missing_coverage.get('usable_seeds_union') or
                        _missing_coverage.get('observed_seeds_union') or [])
    _m_frame = sorted(_m_frame)
    _m_deltas = _missing_entry.get('deltas') or []
    _m_sesoi = _missing_entry.get('sesoi') or {}
    _m_margin = _finite_scalar(_m_sesoi.get('margin_ppl'))
    _m_margin = abs(float(_m_margin)) if _m_margin is not None else 0.0
    try:
        _m = missingness_sensitivity(
            _m_deltas, seeds=_m_obs, planned_seeds=_m_frame,
            excluded_seeds=_m_skip, margin=_m_margin, alpha=alpha)
    except (ValueError, TypeError) as _m_exc:
        _m = {
            'kind': 'missingness_sensitivity',
            'status': 'invalid',
            'reason': f'{type(_m_exc).__name__}: {_m_exc}',
            'n_planned': len(_m_frame), 'n_observed': len(_m_obs),
            'n_missing': None, 'n_excluded': len(_m_skip),
            'warnings': ['the stored seed accounting could not be reconciled '
                         'with the planned frame; no missing-data bound is given'],
        }
        report['warnings'].append(
            '%s `%s` vs `%s`: missing-data sensitivity is INVALID (%s); do '
            'not treat the complete-case estimate as a planned-panel estimate'
            % (_missing_entry.get('kind') or 'contrast',
               _missing_entry.get('with'), _missing_entry.get('without'), _m_exc))
    _missing_entry['missingness'] = _m
    _m_name = '%s::%s - %s' % (_missing_entry.get('kind') or 'contrast',
                               _missing_entry.get('with'),
                               _missing_entry.get('without'))
    _m_row = {
        'name': _m_name,
        'component': _missing_entry.get('component'),
        'with': _missing_entry.get('with'),
        'without': _missing_entry.get('without'),
        'status': _m.get('status'),
        'n_planned': _m.get('n_planned'),
        'n_observed': _m.get('n_observed'),
        'n_missing': _m.get('n_missing'),
        'n_excluded': _m.get('n_excluded'),
        'n_unobserved': _m.get('n_unobserved'),
        'missing_seeds': _m.get('missing_seeds'),
        'excluded_seeds': _m.get('excluded_seeds'),
        'robust_zero': _m.get('robust_zero'),
        'robust_margin': _m.get('robust_margin'),
        'unobserved_mean_for_zero': _m.get('unobserved_mean_for_zero'),
        'overall_mean_bounds': _m.get('overall_mean_bounds'),
    }
    _missing_rows.append(_m_row)
    if _m.get('status') == 'complete':
        _missing_complete.append(_m_name)
    elif _m.get('status') == 'no_observed_pairs':
        _missing_no_observed.append(_m_name)
    elif _m.get('status') == 'not_robust_within_range':
        _missing_fragile.append(_m_name)
    elif _m.get('status') == 'sign_robust_margin_not':
        _missing_margin_not.append(_m_name)

for _m_w in _missing_coverage.get('warnings') or []:
    report['warnings'].append('missingness: ' + str(_m_w))
if _missing_fragile:
    report['warnings'].append(
        'missing-data sensitivity: %d contrast(s) are NOT sign-robust within '
        'the observed delta range (%s). The complete-case direction is a '
        'selected subset, not an identified panel result; add the missing '
        'seeds or widen the declared sensitivity range before quoting it.'
        % (len(_missing_fragile), ', '.join(_missing_fragile[:5])))
if _missing_margin_not:
    report['warnings'].append(
        'missing-data sensitivity: %d contrast(s) keep their sign but not '
        'their SESOI margin under the observed delta range (%s); do not quote '
        'the margin-clearing claim as missing-robust.'
        % (len(_missing_margin_not), ', '.join(_missing_margin_not[:5])))

report['missingness'] = {
    'kind': 'panel_missingness',
    'policy': MISSINGNESS_POLICY,
    'frame_basis': _missing_frame_basis,
    'coverage': _missing_coverage,
    'n_contrasts': len(_missing_rows),
    'n_complete': len(_missing_complete),
    'n_no_observed_pairs': len(_missing_no_observed),
    'n_sign_not_robust': len(_missing_fragile),
    'n_margin_not_robust': len(_missing_margin_not),
    'complete_contrasts': _missing_complete,
    'no_observed_contrasts': _missing_no_observed,
    'sign_not_robust_contrasts': _missing_fragile,
    'margin_not_robust_contrasts': _missing_margin_not,
    'contrasts': _missing_rows,
    'role': ('descriptive partial-identification sensitivity for the planned '
             'but unobserved cells; it does not add a test, change a p-value, '
             'Holm family, verified verdict or any frozen decision'),
}

# ---- evaluation-trajectory stability: is the final sign a persistent regime? ----
# The frozen endpoint stays the fixed final evaluation.  This readout only asks
# whether the per-eval-step history on the same paired seeds tells the same
# story, and it never changes significant/verified/Holm or a frozen verdict.
_traj_rows = []
_traj_late = []
_traj_narrow = []
_traj_disagree = []
_traj_final_only = []
_traj_unavailable = []
_traj_invalid = []
_traj_short = []
_traj_subset_mixed = []
_traj_endpoint_mixed = []
_traj_scale_sensitive = []
for _traj_entry in all_entries:
    if not isinstance(_traj_entry, dict) or not _traj_entry.get('deltas'):
        continue
    _tr_name = '%s::%s - %s' % (_traj_entry.get('kind') or 'contrast',
                                _traj_entry.get('with'),
                                _traj_entry.get('without'))
    _tr_seeds = list(_traj_entry.get('seeds') or [])
    _tr_a, _tr_b = _traj_entry.get('with'), _traj_entry.get('without')
    _tr_ta, _tr_tb = _traj_entry.get('tag_with'), _traj_entry.get('tag_without')
    _tr_ha, _tr_hb = {}, {}
    _tr_endpoint_steps = set()
    _tr_subset_fps = set()
    for _tr_s in _tr_seeds:
        _tr_ra = recs.get((_tr_a, _tr_ta, _tr_s))
        _tr_rb = recs.get((_tr_b, _tr_tb, _tr_s))
        if isinstance(_tr_ra, dict) and _tr_ra.get('ppl_history') is not None:
            _tr_ha[_tr_s] = _tr_ra.get('ppl_history')
        if isinstance(_tr_rb, dict) and _tr_rb.get('ppl_history') is not None:
            _tr_hb[_tr_s] = _tr_rb.get('ppl_history')
        for _tr_rec in (_tr_ra, _tr_rb):
            if not isinstance(_tr_rec, dict):
                continue
            _tr_step = _as_seed_int(_tr_rec.get('steps_completed')
                                    if _tr_rec.get('steps_completed') is not None
                                    else _tr_rec.get('steps'))
            if _tr_step is not None:
                _tr_endpoint_steps.add(_tr_step)
            if _tr_rec.get('eval_subset_fp'):
                _tr_subset_fps.add(str(_tr_rec.get('eval_subset_fp')))
    _tr_endpoint = (next(iter(_tr_endpoint_steps))
                    if len(_tr_endpoint_steps) == 1 else None)
    try:
        _tr = paired_trajectory_stability(
            _tr_ha, _tr_hb, seeds=_tr_seeds,
            primary_delta=_traj_entry.get('mean'),
            endpoint_step=_tr_endpoint)
    except (ValueError, TypeError) as _tr_exc:
        _tr = {
            'kind': 'paired_trajectory_stability',
            'status': 'invalid',
            'reason': f'{type(_tr_exc).__name__}: {_tr_exc}',
            'warnings': ['the stored evaluation history is malformed; the '
                         'endpoint trajectory is not interpreted'],
        }
        _traj_invalid.append(_tr_name)
    _tr['history_subset_fps'] = sorted(_tr_subset_fps)
    _tr['endpoint_step_shared'] = _tr_endpoint
    if len(_tr_endpoint_steps) > 1:
        _tr['warnings'] = list(_tr.get('warnings') or []) + [
            'the paired records do not share one endpoint step (%d distinct '
            'values); the primary endpoint is not measured at a common '
            'training horizon' % len(_tr_endpoint_steps)]
        _traj_endpoint_mixed.append(_tr_name)
    if len(_tr_subset_fps) > 1:
        _tr['warnings'] = list(_tr.get('warnings') or []) + [
            'the paired records do not share one evaluation-subset '
            'fingerprint (%d distinct values); the per-eval-step history is '
            'not known to score the same rows across every seed/arm'
            % len(_tr_subset_fps)]
        _traj_subset_mixed.append(_tr_name)
    if _tr.get('final_step_is_endpoint') is False:
        _tr['warnings'] = list(_tr.get('warnings') or []) + [
            'the history trajectory ends before the primary endpoint step']
        _traj_short.append(_tr_name)
    _traj_entry['trajectory_stability'] = _tr
    _tr_row = {
        'name': _tr_name,
        'component': _traj_entry.get('component'),
        'status': _tr.get('status'),
        'n_seeds_included': _tr.get('n_seeds_included'),
        'n_shared_steps': _tr.get('n_shared_steps'),
        'final_step': _tr.get('final_step'),
        'final_sign': _tr.get('final_sign'),
        'primary_sign': _tr.get('primary_sign'),
        'primary_vs_history_sign': _tr.get('primary_vs_history_sign'),
        'primary_vs_history_log_sign': _tr.get('primary_vs_history_log_sign'),
        'scale_sensitive_final': _tr.get('scale_sensitive_final'),
        'n_seeds_final_sign_matches': _tr.get('n_seeds_final_sign_matches'),
        'n_seeds_tail_stable': _tr.get('n_seeds_tail_stable'),
        'tail_gap_slope_log_per_1k': _tr.get('tail_gap_slope_log_per_1k'),
        'gap_trend': _tr.get('gap_trend'),
        'endpoint_step': _tr.get('endpoint_step'),
        'final_step_is_endpoint': _tr.get('final_step_is_endpoint'),
        'history_subset_fps': _tr.get('history_subset_fps'),
        'skipped_seeds': _tr.get('skipped_seeds'),
    }
    _traj_rows.append(_tr_row)
    if _tr.get('status') == 'late_crossing':
        _traj_late.append(_tr_name)
    elif _tr.get('status') == 'final_point_only':
        _traj_final_only.append(_tr_name)
    elif _tr.get('status') == 'insufficient_history':
        _traj_unavailable.append(_tr_name)
    if _tr.get('gap_trend') == 'narrowing':
        _traj_narrow.append(_tr_name)
    if _tr.get('primary_vs_history_sign') is False:
        _traj_disagree.append(_tr_name)
    if _tr.get('scale_sensitive_final') is True:
        _traj_scale_sensitive.append(_tr_name)
    if _tr.get('status') in ('final_point_only', 'zero_final'):
        for _tr_w in _tr.get('warnings') or []:
            report['warnings'].append('trajectory (%s): %s' % (_tr_name, _tr_w))

if _traj_late:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) end on a late-crossing sign '
        'that differs from an earlier point in the declared tail (%s). The '
        'frozen final-step endpoint still stands, but do not describe the '
        'ordering as persistent without reporting the crossing.'
        % (len(_traj_late), ', '.join(_traj_late[:5])))
if _traj_narrow:
    report['warnings'].append(
        'endpoint trajectory: the history-series gap is narrowing over the '
        'declared tail for %d contrast(s) (%s); the final gap may shrink with '
        'more training, so do not quote it as a converged ordering.'
        % (len(_traj_narrow), ', '.join(_traj_narrow[:5])))
if _traj_short:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) have a balanced history that ends '
        'before the primary endpoint step (%s); the trajectory cannot '
        'corroborate the final endpoint.' % (len(_traj_short),
                                             ', '.join(_traj_short[:5])))
if _traj_scale_sensitive:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) have a final mean log-PPL ratio '
        'whose sign differs from the final mean raw PPL difference (%s); the '
        'effect is scale-sensitive across seeds, so the sign claim must name '
        'the estimand (log ratio vs PPL points).'
        % (len(_traj_scale_sensitive), ', '.join(_traj_scale_sensitive[:5])))
if _traj_endpoint_mixed:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) mix endpoint steps across their '
        'records (%s); the trajectory cannot be aligned to one reported '
        'training horizon.' % (len(_traj_endpoint_mixed),
                               ', '.join(_traj_endpoint_mixed[:5])))
if _traj_subset_mixed:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) mix evaluation-subset '
        'fingerprints across seeds/arms (%s); the history may not score the '
        'same rows everywhere, so read its step mean as descriptive only.'
        % (len(_traj_subset_mixed), ', '.join(_traj_subset_mixed[:5])))
if _traj_invalid:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) carry a malformed evaluation '
        'history (%s); the trajectory is marked invalid rather than silently '
        'omitted.' % (len(_traj_invalid), ', '.join(_traj_invalid[:5])))
if _traj_disagree:
    report['warnings'].append(
        'endpoint trajectory: %d contrast(s) have a history-series final sign '
        'that disagrees with the primary-endpoint sign (%s). The history is '
        'measured on the evaluation subset while the primary PPL uses the full '
        'fixed validation rows; inspect the sample difference before treating '
        'the trajectory as corroboration.'
        % (len(_traj_disagree), ', '.join(_traj_disagree[:5])))

report['trajectory_stability'] = {
    'kind': 'panel_trajectory_stability',
    'policy': TRAJECTORY_POLICY,
    'n_contrasts': len(_traj_rows),
    'n_stable': sum(1 for r in _traj_rows if r.get('status') == 'stable'),
    'n_late_crossing': len(_traj_late),
    'n_zero_final': sum(1 for r in _traj_rows if r.get('status') == 'zero_final'),
    'n_final_point_only': len(_traj_final_only),
    'n_gap_narrowing': len(_traj_narrow),
    'n_primary_history_disagreement': len(_traj_disagree),
    'n_insufficient_history': len(_traj_unavailable),
    'n_invalid': len(_traj_invalid),
    'n_history_short_of_endpoint': len(_traj_short),
    'n_mixed_history_subset': len(_traj_subset_mixed),
    'n_mixed_endpoint_step': len(_traj_endpoint_mixed),
    'n_scale_sensitive_final': len(_traj_scale_sensitive),
    'late_crossing_contrasts': _traj_late,
    'gap_narrowing_contrasts': _traj_narrow,
    'primary_history_disagreement_contrasts': _traj_disagree,
    'final_point_only_contrasts': _traj_final_only,
    'invalid_contrasts': _traj_invalid,
    'history_short_of_endpoint_contrasts': _traj_short,
    'mixed_history_subset_contrasts': _traj_subset_mixed,
    'mixed_endpoint_step_contrasts': _traj_endpoint_mixed,
    'scale_sensitive_final_contrasts': _traj_scale_sensitive,
    'contrasts': _traj_rows,
    'role': ('descriptive final-sign/trajectory consistency; the primary '
             'endpoint remains the fixed final evaluation and this block does '
             'not add a test, change p/Holm/verified or a frozen verdict'),
}

if write and outdir:
    try:
        os.makedirs(outdir, exist_ok=True)
        atomic_write_json(os.path.join(outdir, 'methodology_audit.json'), report)
    except Exception as e:
        print('[audit] could not write methodology_audit.json: %s: %s' % (type(e).__name__, e))

if verbose:
    _print_methodology_audit(report)
