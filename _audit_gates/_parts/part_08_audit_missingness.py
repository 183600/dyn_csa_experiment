
def audit_missingness(path=None):
    """Replay the missing-data sensitivity block from a methodology report.

    The panel audit now attaches, for every declared contrast, the planned seed
    frame, the observed/missing/excluded accounting, the sharp overall-mean
    bounds over a declared delta range, and the tipping value that would erase
    the observed sign (or SESOI margin).  This gate recomputes those numbers
    from the stored per-seed deltas and planned frame, so a hand-edited
    ``methodology_audit.json`` cannot claim that a complete-case contrast is
    missing-data robust.

    A missing block is ``not_verifiable_locally`` (an older artifact), never a
    pass.  The gate never changes a p-value, Holm family, ``verified`` verdict
    or a frozen confirmation decision; it only checks that the descriptive
    robustness readout is internally reproducible.
    """
    L = _import_exp_lib()
    out = {'kind': 'missingness_audit', 'status': 'ok', 'path': path,
           'panels': {}, 'n_checked': 0, 'n_ok': 0, 'n_not_verifiable': 0,
           'problems': [], 'warnings': [],
           'note': ('descriptive partial-identification sensitivity: absence '
                    'of a block leaves missing-data robustness unverified; it '
                    'does not make the contrast wrong or significant')}

    def _finite(x):
        if isinstance(x, (bool, np.bool_, str, bytes, bytearray)) or x is None:
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    def _count(x):
        y = _finite(x)
        return int(y) if y is not None and y.is_integer() else None

    def _close(a, b, tol=1e-9):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))

    def _check_one(p):
        if not os.path.exists(p):
            return {'status': 'not_verifiable', 'reason': 'artifact not found',
                    'problems': [], 'warnings': []}
        data = _load(p)
        if not isinstance(data, dict):
            return {'status': 'fail', 'problems': [f'{p} is not a readable JSON object'],
                    'warnings': []}
        md = data.get('missingness')
        if not isinstance(md, dict):
            return {'status': 'not_verifiable',
                    'reason': 'methodology_audit.json has no missingness block (legacy artifact)',
                    'problems': [], 'warnings': []}
        problems, warnings = [], []
        rows = []
        for group_name in ('components', 'baselines'):
            group = data.get(group_name)
            if not isinstance(group, dict):
                continue
            for comp, entries in group.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    m = e.get('missingness')
                    if not isinstance(m, dict):
                        continue
                    label = '%s/%s:%s-%s' % (group_name, comp,
                                             e.get('with'), e.get('without'))
                    if m.get('status') == 'invalid':
                        warnings.append(f'{label}: stored block is marked invalid')
                        rows.append({'contrast': label, 'status': 'invalid'})
                        continue
                    deltas = m.get('observed_deltas')
                    if deltas is None:
                        deltas = e.get('deltas')
                    seeds = m.get('observed_seeds')
                    planned = m.get('planned_seeds')
                    excluded = m.get('excluded_seeds') or []
                    if not isinstance(seeds, list) or not isinstance(planned, list):
                        warnings.append(f'{label}: no stored seed frame; cannot replay')
                        rows.append({'contrast': label, 'status': 'not_verifiable'})
                        continue
                    _problems_before = len(problems)
                    try:
                        rec = L.missingness_sensitivity(
                            deltas or [], seeds=seeds, planned_seeds=planned,
                            excluded_seeds=excluded,
                            margin=m.get('margin') or 0.0,
                            delta_range=m.get('range'),
                            alpha=m.get('alpha', data.get('alpha', 0.05)),
                            min_n=m.get('min_n', data.get('min_n', 3)))
                    except (ValueError, TypeError) as exc:
                        problems.append(f'{label}: stored frame/deltas do not replay ({exc})')
                        rows.append({'contrast': label, 'status': 'fail'})
                        continue
                    checks = {
                        'n_planned': (m.get('n_planned'), rec.get('n_planned'), _count),
                        'n_observed': (m.get('n_observed'), rec.get('n_observed'), _count),
                        'n_missing': (m.get('n_missing'), rec.get('n_missing'), _count),
                        'n_excluded': (m.get('n_excluded'), rec.get('n_excluded'), _count),
                        'observed_mean': (m.get('observed_mean'), rec.get('observed_mean'), _finite),
                        'unobserved_mean_for_zero': (m.get('unobserved_mean_for_zero'), rec.get('unobserved_mean_for_zero'), _finite),
                        'overall_mean_bounds': (m.get('overall_mean_bounds'), rec.get('overall_mean_bounds'), None),
                        'robust_zero': (m.get('robust_zero'), rec.get('robust_zero'), None),
                        'robust_margin': (m.get('robust_margin'), rec.get('robust_margin'), None),
                    }
                    for key, (stored, fresh, parser) in checks.items():
                        if key == 'overall_mean_bounds':
                            ok = (isinstance(stored, (list, tuple)) and len(stored) == 2
                                  and isinstance(fresh, (list, tuple)) and len(fresh) == 2
                                  and _close(stored[0], fresh[0])
                                  and _close(stored[1], fresh[1]))
                        elif key in ('robust_zero', 'robust_margin'):
                            ok = stored is fresh
                        elif parser is _count:
                            ok = _count(stored) == fresh
                        else:
                            ok = _close(_finite(stored), fresh)
                        if not ok:
                            problems.append(
                                f'{label}: {key} is {stored!r}, recomputed {fresh!r}')
                    if m.get('status') != rec.get('status'):
                        problems.append(
                            f'{label}: status is {m.get("status")!r}, recomputed {rec.get("status")!r}')
                    rows.append({'contrast': label,
                                 'status': ('fail' if len(problems) > _problems_before else 'ok'),
                                 'n_planned': rec.get('n_planned'),
                                 'n_observed': rec.get('n_observed'),
                                 'n_missing': rec.get('n_missing'),
                                 'stored_status': m.get('status'),
                                 'recomputed_status': rec.get('status')})
        if not rows:
            return {'status': 'not_verifiable',
                    'reason': 'no contrast carries a missingness block',
                    'problems': [], 'warnings': []}
        return {'status': ('fail' if problems else ('warn' if warnings else 'ok')),
                'rows': rows, 'problems': problems, 'warnings': warnings}
    targets = []
    if path is not None:
        p = _audit_artifact(path)
        if os.path.isdir(p):
            p = os.path.join(p, 'methodology_audit.json')
        targets = [p]
    else:
        for d in _iter_result_dirs():
            p = os.path.join(REPO, d, 'methodology_audit.json')
            if os.path.exists(p):
                targets.append(p)
    if not targets:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no results_*/methodology_audit.json on disk; run '
                         '`python audit_gates.py methodology` after the panel '
                         'run, then replay the missingness block')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    for p in targets:
        rep = _check_one(p)
        out['panels'][os.path.relpath(p, REPO)] = rep
        out['n_checked'] += 1
        out['problems'].extend(rep.get('problems') or [])
        out['warnings'].extend(rep.get('warnings') or [])
        if rep.get('status') == 'ok':
            out['n_ok'] += 1
        elif rep.get('status') == 'not_verifiable':
            out['n_not_verifiable'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_ok'] != out['n_checked']:
        out['status'] = 'warn'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
