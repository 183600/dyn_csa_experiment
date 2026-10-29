# Replay the evaluation-trajectory stability block from a panel report.
# This fragment deliberately carries no module docstring: the audit CLI
# help text lives in part_05 and must remain the package __doc__.


def audit_trajectory(path=None):
    """
    Replay the evaluation-trajectory stability block from a panel report.

    ``methodology_audit`` records, for every paired contrast with an evaluation
    history, the balanced per-step paired log-PPL series and a small set of
    trajectory shape readouts (final sign, late crossing, gap trend, per-seed tail
    signs).  Those readouts are descriptive, but a hand-edited artifact could still
    claim "stable" for a trajectory that crosses at the last step.

    This gate recomputes the shape from the stored series and per-seed tail signs.
    A missing block in an older artifact is ``not_verifiable`` (never a pass); a
    block whose status/counts/signs cannot be reproduced is ``fail``.  The gate
    never changes a p-value, Holm family, ``verified`` verdict or frozen decision.
    """
    L = _import_exp_lib()
    out = {'kind': 'trajectory_stability_audit', 'status': 'ok', 'path': path,
           'panels': {}, 'n_checked': 0, 'n_ok': 0, 'n_not_verifiable': 0,
           'problems': [], 'warnings': [],
           'note': ('descriptive final-sign/trajectory consistency; the primary '
                    'endpoint remains the fixed final evaluation and a missing '
                    'block leaves stability unverified, not wrong')}

    def _finite(x):
        if isinstance(x, (bool, np.bool_, str, bytes, bytearray)) or x is None:
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    def _count(x, minimum=None):
        y = _finite(x)
        if y is None or not y.is_integer():
            return None
        y = int(y)
        return y if (minimum is None or y >= minimum) else None

    def _close(a, b, tol=1e-9):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))

    def _sign(x, zero):
        if x is None:
            return None
        if abs(x) <= zero:
            return 0
        return 1 if x > 0.0 else -1

    def _slope(rows):
        if len(rows) < 2:
            return None
        xs = [float(r['step']) for r in rows]
        ys = [float(r['mean_log_ratio']) for r in rows]
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        den = sum((x - xm) ** 2 for x in xs)
        if den <= 0.0:
            return None
        return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den * 1000.0

    def _shape(series, tail_points, zero):
        final = series[-1]
        tail = series[-min(tail_points, len(series)):]
        final_sign = final['sign']
        tail_steps = [r['step'] for r in tail]
        if final_sign == 0:
            status, stable = 'zero_final', False
        else:
            opposite = [r['step'] for r in tail[:-1]
                        if r['sign'] not in (None, 0, final_sign)]
            nonzero_earlier = [r['step'] for r in tail[:-1]
                               if r['sign'] not in (None, 0)]
            if opposite:
                status, stable = 'late_crossing', False
            elif not nonzero_earlier:
                status, stable = 'final_point_only', False
            else:
                status, stable = 'stable', True
        changes = []
        last = None
        for row in series:
            if row['sign'] in (None, 0):
                continue
            if last is not None and row['sign'] != last:
                changes.append(row['step'])
            last = row['sign']
        gap_slope = _slope(tail)
        if gap_slope is None or final_sign in (None, 0):
            trend = None
        else:
            gs = _sign(gap_slope, zero)
            trend = ('flat' if gs == 0 else
                     'widening' if gs == final_sign else 'narrowing')
        return {
            'status': status, 'tail_sign_stable': stable,
            'final_sign': final_sign, 'final_step': final['step'],
            'tail_steps': tail_steps,
            'n_sign_changes': len(changes),
            'last_sign_change_step': changes[-1] if changes else None,
            'gap_slope': gap_slope, 'gap_trend': trend,
            'n_seeds_final_sign_matches': None, 'n_seeds_tail_stable': None,
        }

    def _check_block(label, tr, problems):
        _before = len(problems)
        if not isinstance(tr, dict):
            problems.append(f'{label}: trajectory_stability is not an object')
            return False
        status = tr.get('status')
        if status in ('invalid', 'insufficient_history'):
            return True  # explicitly unavailable; nothing numeric to replay
        policy = tr.get('policy') if isinstance(tr.get('policy'), dict) else {}
        tail_points = _count(policy.get('tail_points'), minimum=2)
        if tail_points is None:
            problems.append(f'{label}: policy.tail_points is not an integer >= 2')
            return False
        zero = _finite(policy.get('zero_tol_log'))
        if zero is None or zero < 0.0:
            problems.append(f'{label}: policy.zero_tol_log is not a finite '
                            'non-negative number')
            return False
        series = tr.get('series')
        if not isinstance(series, list) or len(series) < 2:
            problems.append(f'{label}: series is not a list of at least 2 steps')
            return False
        parsed = []
        prev_step = None
        for i, row in enumerate(series):
            if not isinstance(row, dict):
                problems.append(f'{label}: series[{i}] is not an object')
                continue
            step = _count(row.get('step'), minimum=0)
            if step is None:
                problems.append(f'{label}: series[{i}].step is not a '
                                'non-negative integer')
                continue
            if prev_step is not None and step <= prev_step:
                problems.append(f'{label}: series steps are not strictly '
                                f'increasing at index {i}')
            prev_step = step
            mean_log = _finite(row.get('mean_log_ratio'))
            mean_delta = _finite(row.get('mean_delta_ppl'))
            if mean_log is None or mean_delta is None:
                problems.append(f'{label}: series[{i}] has a non-finite mean')
                continue
            if row.get('sign') != _sign(mean_log, zero):
                problems.append(f'{label}: series[{i}].sign does not match its '
                                'mean_log_ratio and zero_tol')
            n_seeds = _count(row.get('n_seeds'), minimum=1)
            if n_seeds is None:
                problems.append(f'{label}: series[{i}].n_seeds is not a '
                                'positive integer')
            parsed.append({'step': step, 'mean_log_ratio': mean_log,
                           'mean_delta_ppl': mean_delta,
                           'sign': _sign(mean_log, zero),
                           'n_seeds': n_seeds})
        if len(parsed) < 2:
            problems.append(f'{label}: fewer than 2 reusable series rows')
            return False
        shape = _shape(parsed, tail_points, zero)
        for key, stored in (
                ('status', tr.get('status')),
                ('tail_sign_stable', tr.get('tail_sign_stable')),
                ('final_sign', tr.get('final_sign')),
                ('final_step', tr.get('final_step')),
                ('tail_steps', tr.get('tail_steps')),
                ('n_sign_changes', tr.get('n_sign_changes')),
                ('last_sign_change_step', tr.get('last_sign_change_step')),
                ('gap_trend', tr.get('gap_trend'))):
            fresh = shape[key]
            if isinstance(fresh, list):
                ok = (isinstance(stored, list) and stored == fresh)
            elif isinstance(fresh, (int, float)) and not isinstance(fresh, bool):
                ok = _close(_finite(stored), _finite(fresh))
            else:
                ok = stored == fresh
            if not ok:
                problems.append(f'{label}: {key} is {stored!r}, recomputed {fresh!r}')
        if not _close(_finite(tr.get('tail_gap_slope_log_per_1k')),
                      _finite(shape['gap_slope'])):
            problems.append(f'{label}: tail_gap_slope_log_per_1k is '
                            f'{tr.get("tail_gap_slope_log_per_1k")!r}, '
                            f'recomputed {shape["gap_slope"]!r}')
        if _count(tr.get('n_shared_steps'), minimum=0) != len(parsed):
            problems.append(f'{label}: n_shared_steps does not match the '
                            'stored series length')
        if tr.get('shared_steps') != [r['step'] for r in parsed]:
            problems.append(f'{label}: shared_steps does not match the series')
        if not _close(_finite(tr.get('final_mean_log_ratio')),
                      _finite(parsed[-1]['mean_log_ratio'])):
            problems.append(f'{label}: final_mean_log_ratio does not match the '
                            'last series row')
        if not _close(_finite(tr.get('final_mean_delta_ppl')),
                      _finite(parsed[-1]['mean_delta_ppl'])):
            problems.append(f'{label}: final_mean_delta_ppl does not match the '
                            'last series row')
        delta_sign = _sign(parsed[-1]['mean_delta_ppl'], zero)
        if tr.get('final_delta_ppl_sign') != delta_sign:
            problems.append(f'{label}: final_delta_ppl_sign does not match the '
                            'last series row')
        want_scale = bool(shape['final_sign'] not in (None, 0)
                          and delta_sign not in (None, 0)
                          and shape['final_sign'] != delta_sign)
        if tr.get('scale_sensitive_final') is not want_scale:
            problems.append(f'{label}: scale_sensitive_final is inconsistent '
                            'with the final log/raw signs')

        per_seed = tr.get('per_seed_tail')
        if not isinstance(per_seed, list):
            problems.append(f'{label}: per_seed_tail is not a list')
        else:
            _matches = _tail_stable = 0
            for i, row in enumerate(per_seed):
                if not isinstance(row, dict) or not isinstance(row.get('tail_signs'), list):
                    problems.append(f'{label}: per_seed_tail[{i}] is malformed')
                    continue
                own = row['tail_signs']
                if any(x not in (-1, 0, 1) for x in own):
                    problems.append(f'{label}: per_seed_tail[{i}] has a sign '
                                    'outside {-1,0,1}')
                if own and own[-1] == shape['final_sign'] and shape['final_sign'] != 0:
                    _matches += 1
                nz = [x for x in own if x not in (None, 0)]
                if nz and all(x == nz[-1] for x in nz):
                    _tail_stable += 1
            if _count(tr.get('n_seeds_final_sign_matches'), minimum=0) != _matches:
                problems.append(f'{label}: n_seeds_final_sign_matches is '
                                f'{tr.get("n_seeds_final_sign_matches")!r}, '
                                f'recomputed {_matches}')
            if _count(tr.get('n_seeds_tail_stable'), minimum=0) != _tail_stable:
                problems.append(f'{label}: n_seeds_tail_stable is '
                                f'{tr.get("n_seeds_tail_stable")!r}, '
                                f'recomputed {_tail_stable}')
            if _count(tr.get('n_seeds_included'), minimum=0) != len(per_seed):
                problems.append(f'{label}: n_seeds_included does not match '
                                'per_seed_tail length')

        ep = tr.get('endpoint_step')
        final_is_ep = tr.get('final_step_is_endpoint')
        if ep is None:
            ok_ep = final_is_ep is None
        else:
            epi = _count(ep, minimum=0)
            if epi is None:
                problems.append(f'{label}: endpoint_step is not a non-negative '
                                'integer')
                ok_ep = False
            else:
                ok_ep = final_is_ep is (shape['final_step'] == epi)
        if not ok_ep:
            problems.append(f'{label}: final_step_is_endpoint is inconsistent '
                            'with endpoint_step/final_step')

        psign = tr.get('primary_sign')
        pd = tr.get('primary_delta_ppl')
        if pd is None:
            if psign is not None:
                problems.append(f'{label}: primary_sign is set without a '
                                'primary_delta_ppl')
        elif _finite(pd) is None:
            problems.append(f'{label}: primary_delta_ppl is not finite')
        elif psign != _sign(_finite(pd), zero):
            problems.append(f'{label}: primary_sign does not match '
                            'primary_delta_ppl and zero_tol')
        pv = tr.get('primary_vs_history_sign')
        pv_log = tr.get('primary_vs_history_log_sign')
        if psign in (None, 0) or delta_sign in (None, 0):
            ok_pv = pv is None
        else:
            ok_pv = pv is bool(psign == delta_sign)
        if psign in (None, 0) or shape['final_sign'] in (None, 0):
            ok_pv_log = pv_log is None
        else:
            ok_pv_log = pv_log is bool(psign == shape['final_sign'])
        if not ok_pv:
            problems.append(f'{label}: primary_vs_history_sign is inconsistent '
                            'with the stored raw-delta signs')
        if not ok_pv_log:
            problems.append(f'{label}: primary_vs_history_log_sign is '
                            'inconsistent with the stored log signs')
        return len(problems) == _before

    def _check_one(p):
        if not os.path.exists(p):
            return {'status': 'not_verifiable', 'reason': 'artifact not found',
                    'problems': [], 'warnings': []}
        data = _load(p)
        if not isinstance(data, dict):
            return {'status': 'fail', 'problems': [f'{p} is not a readable JSON object'],
                    'warnings': []}
        panel = data.get('trajectory_stability')
        if not isinstance(panel, dict):
            return {'status': 'not_verifiable',
                    'reason': ('methodology_audit.json has no '
                               'trajectory_stability block (legacy artifact)'),
                    'problems': [], 'warnings': []}
        problems, warnings, rows = [], [], []
        statuses = []
        trends = []
        pvs = []
        scales = []
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
                    tr = e.get('trajectory_stability')
                    if not isinstance(tr, dict):
                        continue
                    label = '%s/%s:%s-%s' % (group_name, comp,
                                             e.get('with'), e.get('without'))
                    ok = _check_block(label, tr, problems)
                    st = tr.get('status')
                    statuses.append(st)
                    trends.append(tr.get('gap_trend'))
                    pvs.append(tr.get('primary_vs_history_sign'))
                    scales.append(tr.get('scale_sensitive_final'))
                    rows.append({'contrast': label, 'status': st,
                                 'replayed': ok})
                    if st in ('insufficient_history', 'invalid'):
                        warnings.append(f'{label}: trajectory is {st}')
        if not rows:
            return {'status': 'not_verifiable',
                    'reason': 'no contrast carries a trajectory_stability block',
                    'problems': [], 'warnings': []}
        # Panel-level counts must agree with the rows (a hand-edited summary
        # cannot claim "all stable" while a row is a late crossing).
        n = len(rows)
        counts = {
            'n_contrasts': n,
            'n_stable': statuses.count('stable'),
            'n_late_crossing': statuses.count('late_crossing'),
            'n_zero_final': statuses.count('zero_final'),
            'n_final_point_only': statuses.count('final_point_only'),
            'n_insufficient_history': statuses.count('insufficient_history'),
            'n_invalid': statuses.count('invalid'),
            'n_gap_narrowing': trends.count('narrowing'),
            'n_primary_history_disagreement': sum(1 for x in pvs if x is False),
            'n_scale_sensitive_final': sum(1 for x in scales if x is True),
        }
        for key, fresh in counts.items():
            if _count(panel.get(key), minimum=0) != fresh:
                problems.append(f'panel: {key} is {panel.get(key)!r}, '
                                f'recomputed {fresh}')
        if not panel.get('policy'):
            warnings.append('panel has no stored policy; the tail/zero-tol '
                            'convention is unverifiable')
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
                         'run, then replay the trajectory block')
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
