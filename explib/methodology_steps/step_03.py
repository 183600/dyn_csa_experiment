# ---- comparative experiment family integrity: max-T, look-history, cross-panel ----
# This step adds correlation-aware family correction (max-T step-down), look-history
# sequential p-value tracking, and cross-panel comparability diagnostics for the
# declared family of contrasts.  It is descriptive: it does not change p_holm,
# verified verdict, or any frozen decision.

try:
    _family = report.get('family') or {}
    _family_name = _family.get('name', 'unknown')
    _alpha = float(_family.get('alpha', 0.05))
    _m_family = int(_family.get('effective_m', _family.get('declared_contrasts', 0)))
    _min_n = int(report.get('min_n', 3))
    all_entries = [e for group in (report.get('components') or {}, report.get('baselines') or {})
                   for entries in group.values() for e in entries]
    _m_declared = int(_family.get('declared_contrasts', len(all_entries)))

    # --- max-T family correction (correlation-aware step-down) ---
    # Holm is the declared correction.  The contrasts in an ablation/table panel
    # share reference arms, seeds and evaluation rows, so their deltas are correlated.
    # max-T uses the joint null distribution and is generally more powerful than Holm
    # for the same FWER.  This readout is a sensitivity readout: Holm remains the
    # declared correction and a max-T-only rejection is NOT promoted to `verified`.

    _joint_records = {}
    _joint_entries = {}
    for _jk_kind, _jk_group in (('components', report.get('components') or {}),
                                  ('baselines', report.get('baselines') or {})):
        for _jk_comp, _jk_entries in _jk_group.items():
            for _jk_i, _jk_e in enumerate(_jk_entries):
                _jk_key = '%s::%s::%d' % (_jk_kind, _jk_comp, _jk_i)
                _jk_ok = bool(_jk_e.get('controlled')
                              and _jk_e.get('fully_paired')
                              and _jk_e.get('deltas')
                              and int(_jk_e.get('n') or 0) >= 2)
                if _jk_ok:
                    _jk_reason = None
                elif not _jk_e.get('controlled'):
                    _jk_reason = 'no controlled paired delta'
                elif not _jk_e.get('fully_paired'):
                    _jk_reason = 'incomplete paired coverage'
                elif not _jk_e.get('deltas'):
                    _jk_reason = 'no per-seed deltas recorded'
                else:
                    _jk_reason = 'fewer than 2 paired seeds'
                _joint_records[_jk_key] = {
                    'deltas': list(_jk_e.get('deltas') or []),
                    'seeds': list(_jk_e.get('seeds') or []),
                    'eligible': _jk_ok,
                    'reason': _jk_reason,
                }
                _joint_entries[_jk_key] = _jk_e
    if _m_declared > 0:
        # maxt_joint_correction / signflip_min_n_for_holm live in the explib
        # globals that methodology_audit already injects via _step_env; the
        # step fragments are NOT importable modules (they execute inside the
        # audit's namespace), so no import is needed or possible here.
        _joint = maxt_joint_correction(_joint_records, alpha=_alpha, family_size=_m_declared)
    else:
        _joint = {
            'status': 'not_applicable', 'method': None, 'family_ready': False,
            'm_eligible': 0, 'family_size': 0, 'n_groups': 0, 'by_key': {},
            'excluded': {}, 'role': None,
            'warnings': ['no declared ablation/baseline contrast'],
        }
    for _jk_key, _jk_entry in _joint_entries.items():
        _jk_rec = (_joint.get('by_key') or {}).get(_jk_key)
        if _jk_rec and _jk_rec.get('status') == 'ok':
            _jk_entry['p_maxt'] = _jk_rec['p_maxt']
            _jk_entry['p_maxt_single'] = _jk_rec['p_maxt_single']
            _jk_entry['p_raw_signflip_maxt'] = _jk_rec['p_raw_signflip']
            _jk_entry['t_maxt'] = _jk_rec['t_stat']
            _jk_entry['maxt_rank'] = _jk_rec['rank']
            _jk_entry['reject_maxt'] = _jk_rec['reject_maxt']
            _jk_entry['significant_maxt'] = bool(
                _jk_rec['reject_maxt'] and _joint.get('family_ready'))
            _jk_entry['ci_simultaneous'] = _jk_rec['ci_simultaneous']
            _jk_entry['maxt_group'] = _jk_rec['group']
            _jk_entry['maxt_family_ready'] = bool(_joint.get('family_ready'))
            _jk_entry['maxt_status'] = _joint.get('status')
        else:
            _jk_entry['p_maxt'] = None
            _jk_entry['reject_maxt'] = False
            _jk_entry['significant_maxt'] = False
            _jk_entry['maxt_family_ready'] = bool(_joint.get('family_ready'))
            _jk_entry['maxt_status'] = (
                (_jk_rec or {}).get('status')
                or ('not_eligible' if _jk_key in (_joint.get('excluded') or {})
                    else 'not_computed'))
            if _jk_key in (_joint.get('excluded') or {}):
                _jk_entry['maxt_excluded_reason'] = _joint['excluded'][_jk_key]
    _joint_holm_only = sorted(
        _jk_key for _jk_key, _jk_e in _joint_entries.items()
        if _jk_e.get('significant_global') and not _jk_e.get('reject_maxt'))
    _joint_maxt_only = sorted(
        _jk_key for _jk_key, _jk_e in _joint_entries.items()
        if _jk_e.get('significant_maxt') and not _jk_e.get('significant_global'))
    report['family']['maxt'] = {
        'status': _joint.get('status'),
        'method': _joint.get('method'),
        'family_ready': bool(_joint.get('family_ready')),
        'm_eligible': _joint.get('m_eligible'),
        'family_size': _joint.get('family_size'),
        'n_groups': _joint.get('n_groups'),
        'alpha': float(_alpha),
        # NOTE: maxt_min_n_for_alpha(alpha) is the best-case design-floor
        # helper (dict with min_n/p_floor, part_40); signflip_min_n_for_holm
        # is a DIFFERENT function taking (m_comparisons, alpha) and returning
        # a bare int/None — passing alpha as m returns None.
        'best_case': maxt_min_n_for_alpha(_alpha),
        'role': _joint.get('role'),
        'n_maxt_reject': sum(1 for _r in (_joint.get('by_key') or {}).values()
                             if _r.get('reject_maxt')),
        'n_maxt_significant': sum(1 for _e in _joint_entries.values()
                                  if _e.get('significant_maxt')),
        'n_holm_reject': sum(1 for _e in _joint_entries.values()
                             if _e.get('significant_global')),
        'holm_only': _joint_holm_only,
        'maxt_only': _joint_maxt_only,
        'groups': _joint.get('groups'),
        'excluded': _joint.get('excluded'),
        'warnings': _joint.get('warnings'),
        'note': ('max-T is a correlation-aware sensitivity readout of the same '
                 'per-seed deltas; Holm remains the declared correction and a '
                 'max-T-only rejection is NOT promoted to `verified`'),
    }
    report['maxt_only_contrasts'] = _joint_maxt_only
    report['holm_only_contrasts'] = _joint_holm_only
    report['maxt_only_pairs'] = [
        '%s - %s (%s)' % ((_joint_entries[_k].get('with') or '?'),
                          (_joint_entries[_k].get('without') or '?'),
                          _joint_entries[_k].get('kind') or '?')
        for _k in _joint_maxt_only]
    report['holm_only_pairs'] = [
        '%s - %s (%s)' % ((_joint_entries[_k].get('with') or '?'),
                          (_joint_entries[_k].get('without') or '?'),
                          _joint_entries[_k].get('kind') or '?')
        for _k in _joint_holm_only]
    if _joint.get('family_ready') and _joint_maxt_only:
        report['warnings'].append(
            'the exact max-T family rejects %d contrast(s) that Holm does not '
            '(%s): the declared Holm reading is conservative for this correlated '
            'family, but a max-T-only rejection is a sensitivity readout, not a '
            'confirmed ablation result — pre-register max-t as the correction '
            'before using it for a claim'
            % (len(_joint_maxt_only), ', '.join(_joint_maxt_only[:3])))
    elif _joint_holm_only:
        report['warnings'].append(
            'Holm rejects %d contrast(s) that the max-T readout does not (%s); '
            'this can happen when the complete family spans several seed sets or '
            'a degenerate contrast is excluded — inspect the group table before '
            'quoting either correction'
            % (len(_joint_holm_only), ', '.join(_joint_holm_only[:3])))

    # --- look-history / optional-stopping correction ---
    # Records every aggregation look of the panel.  When a contrast was measured
    # at more than one growing seed set, its significance is required to pass the
    # exact sequential sign-flip p; the naive final-n p is kept as
    # p_exact_signflip / p_holm_global_naive for transparency.
    _look_entries = [e for e in all_entries if isinstance(e.get('look_analysis'), dict)]
    _look_keys = sorted({e['look_analysis'].get('contrast') for e in _look_entries if e['look_analysis'].get('contrast')})
    _multi = [e for e in _look_entries if int(e['look_analysis'].get('n_looks') or 0) > 1]
    _multi_keys = sorted({e['look_analysis'].get('contrast') for e in _multi if e['look_analysis'].get('contrast')})
    _downgraded = set()
    for e in _look_entries:
        if 'significant_global_naive' in e:
            if e.get('significant_global_naive') and (not e.get('significant_global')):
                _downgraded.add(e['look_analysis'].get('contrast'))
    _downgraded = sorted(_downgraded)
    if _downgraded:
        report['warnings'].append(
            'optional-stopping correction DOWNGRADED %d contrast(s): %s — '
            'the naive final-look family p was < alpha, but the exact max-over-looks p was not. '
            'Adding seeds after seeing a non-significant result is one of the looks, '
            'not a free rerun.' % (len(_downgraded), ', '.join(_downgraded[:5])))

    # --- cross-panel comparability: NOT available here ---
    # cross_panel_synthesis (part_63) synthesizes MULTIPLE panel reports and
    # has no single-panel entry-point form; it runs at the report level, not
    # inside one methodology_audit.  A per-panel cross-reference block was
    # drafted here against a function that does not exist
    # (cross_panel_synthesis_audit) and was unreachable dead code until the
    # namespace bug above was fixed, so it is removed rather than wired to a
    # wrong signature.  The real cross-panel gates remain
    # cross_panel_synthesis / ablation_coverage_across_panels at the report
    # layer.

except Exception as _e:
    report['warnings'].append(
        'comparative family integrity audit failed: %s: %s' % (type(_e).__name__, _e))