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

if write and outdir:
    try:
        os.makedirs(outdir, exist_ok=True)
        atomic_write_json(os.path.join(outdir, 'methodology_audit.json'), report)
    except Exception as e:
        print('[audit] could not write methodology_audit.json: %s: %s' % (type(e).__name__, e))

if verbose:
    _print_methodology_audit(report)
