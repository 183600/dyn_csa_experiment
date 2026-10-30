# ---- endpoint-selection provenance: final step vs selected best step ------
# A quoted PPL is only comparable across arms when both arms name the same
# endpoint rule.  This block attaches the rule audit to the panel; it is
# descriptive/provenance only and does not touch any p-value, Holm family or
# frozen decision.

_endpoint_records = []
for (_ep_v, _ep_t, _ep_s), _ep_r in recs.items():
    if not isinstance(_ep_r, dict):
        continue
    _ep_ch = _ep_r.get('curve_health')
    _ep_ckpt = _ep_r.get('reported_checkpoint')
    if _ep_ckpt is None:
        _ep_ckpt = _ep_r.get('checkpoint')
    _endpoint_records.append({
        'arm': _ep_v,
        'tag': _ep_t or '',
        'seed': _ep_s,
        'ppl': _ep_r.get('ppl'),
        'steps': _ep_r.get('steps'),
        'steps_planned': _ep_r.get('steps_planned'),
        'steps_completed': _ep_r.get('steps_completed',
                                     _ep_r.get('steps_done')),
        'reported_step': _ep_r.get('reported_step'),
        'eval_step': _ep_r.get('eval_step'),
        'endpoint_step': _ep_r.get('endpoint_step'),
        'endpoint_selection_rule': _ep_r.get('endpoint_selection_rule'),
        'selection_rule': _ep_r.get('selection_rule'),
        'selection_split': _ep_r.get('selection_split'),
        'eval_split': _ep_r.get('eval_split'),
        'checkpoint': _ep_ckpt,
        'early_stop': _ep_r.get('early_stop'),
        'curve_health': _ep_ch,
        'ppl_estimation_rows': _ep_r.get('ppl_estimation_rows'),
        'best_step': _ep_r.get('best_step'),
        'best_val_ppl': _ep_r.get('best_val_ppl'),
        'final_minus_best': _ep_r.get('final_minus_best'),
    })

_endpoint_arms = {(_ep_v, _ep_t or '') for (_ep_v, _ep_t, _ep_s) in recs}
_endpoint_comparisons = []
for _ep_entry in all_entries:
    if not isinstance(_ep_entry, dict):
        continue
    _ep_a, _ep_b = _ep_entry.get('with'), _ep_entry.get('without')
    _ep_ta = _ep_entry.get('tag_with') or ''
    _ep_tb = _ep_entry.get('tag_without') or ''
    if not (isinstance(_ep_a, str) and isinstance(_ep_b, str)):
        continue
    # A missing arm is an ablation-completeness problem, not an endpoint-rule
    # mismatch.  Do not turn the endpoint block into a second failure for the
    # same missing records.
    if (_ep_a, _ep_ta) not in _endpoint_arms or (_ep_b, _ep_tb) not in _endpoint_arms:
        continue
    _endpoint_comparisons.append((_ep_a, _ep_ta, _ep_b, _ep_tb))
_endpoint_comparisons = sorted(set(_endpoint_comparisons))

try:
    _endpoint_report = endpoint_selection_audit(
        _endpoint_records, comparisons=_endpoint_comparisons)
except (ValueError, TypeError) as _ep_exc:
    _endpoint_report = {
        'kind': 'endpoint_selection_audit',
        'status': 'not_verifiable',
        'reason': f'{type(_ep_exc).__name__}: {_ep_exc}',
        'comparisons': {}, 'warnings': [], 'violations': [],
        'role': ('endpoint/checkpoint selection provenance only; no p-value '
                 'or frozen decision is changed'),
    }

report['endpoint_selection'] = _endpoint_report
if _endpoint_report.get('status') == 'fail':
    report['warnings'].append(
        'endpoint selection: the reported endpoint is not comparable across '
        'arms (%s)' % '; '.join(_endpoint_report.get('violations') or [])[:500])
elif _endpoint_report.get('status') == 'warn':
    _ep_named = [
        _ep_name for _ep_name, _ep_c in
        (_endpoint_report.get('comparisons') or {}).items()
        if isinstance(_ep_c, dict) and _ep_c.get('status') == 'warn']
    report['warnings'].append(
        'endpoint selection: %d comparison(s) quote a data-dependent or '
        'unequal-step endpoint (%s). Do not read a selected best step as the '
        'pre-registered final-step measurement; report the selection rule and '
        'its independent split.'
        % (len(_ep_named), ', '.join(_ep_named[:5])))
