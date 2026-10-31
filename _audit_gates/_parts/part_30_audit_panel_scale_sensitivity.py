# Panel-scale sensitivity audit (split fragment).
#
# ``cross_panel_synthesis`` reports the pooled percent effect and, from
# ``part_64``, the pooled log-ratio effect with a direction / zero-exclusion
# concordance readout.  The two are different estimands, so a stored
# ``concordant`` label cannot be trusted by inspection: this gate recomputes
# both sides from the contrast's own ``panels`` rows and ``meta_input`` points,
# replays every derived number (pooled means/SEs/intervals, heterogeneity,
# back-transformed percent, Jensen gap, agreement flags and the scale-robust
# claim), and checks that the synthesis did not keep a pooled claim the scale
# gate rejects.
#
# It is an internal replay.  It can show that the recorded scale readout follows
# from the recorded per-seed effects; it cannot show that those effects came
# from a particular run (stamped artifacts and the frozen lock do that), and it
# creates no p-value, no Holm member and no change to any panel verdict.  A
# missing block is ``not_verifiable_locally`` (an older or partial report),
# never a pass; a present-but-tampered block is ``fail``.


def _pss_forbidden_keys(value, *, path=''):
    """Keys that would smuggle a confirmatory verdict into a sensitivity block."""
    problems = []
    if isinstance(value, dict):
        for key in sorted(value):
            if key in ('confirmed', 'p_holm', 'verdict', 'decision'):
                problems.append(
                    f'{path}/{key}: a scale-sensitivity block may not carry a '
                    'confirmatory verdict')
            problems.extend(_pss_forbidden_keys(
                value[key], path=f'{path}/{key}'))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            problems.extend(_pss_forbidden_keys(item, path=f'{path}[{i}]'))
    return problems


def _pss_panel_log_problems(panel, row):
    """Check retained per-seed PPLs against the recorded percent effects.

    A stored ``log_effects`` vector is an input to the replay, so an edited
    vector would otherwise replay against itself.  When the producer retained
    per-seed arm PPLs, verify that their differences reproduce the percent
    effects (using their own mean baseline) and that ``log_effects`` is the
    true paired log ratio.  Missing/partial vectors are a named problem, not a
    silent fallback: the basis field must describe what is actually present.
    """
    problems = []
    if not isinstance(row, dict):
        return [f'{panel}: panel row is not an object']
    log_effects = row.get('log_effects')
    with_vals = row.get('per_seed_with')
    without_vals = row.get('per_seed_without')
    effects = row.get('effects')
    if log_effects is None and with_vals is None and without_vals is None:
        return problems
    if log_effects is None or with_vals is None or without_vals is None:
        return [f'{panel}: log_effects/per_seed_with/per_seed_without must be '
                'present together']
    for name, value in (('log_effects', log_effects),
                        ('per_seed_with', with_vals),
                        ('per_seed_without', without_vals),
                        ('effects', effects)):
        if not isinstance(value, (list, tuple)):
            problems.append(f'{panel}: {name} is not a list')
            return problems
    n = len(effects)
    if not (len(log_effects) == len(with_vals) == len(without_vals) == n):
        problems.append(f'{panel}: log/per-seed/percent vector lengths differ')
        return problems
    if n == 0:
        return problems
    checks = []
    for i in range(n):
        try:
            a = float(with_vals[i])
            b = float(without_vals[i])
            e = float(effects[i])
            lv = float(log_effects[i])
        except (TypeError, ValueError, OverflowError):
            problems.append(f'{panel}: non-numeric value at index {i}')
            return problems
        if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
            problems.append(f'{panel}: per-seed PPL at index {i} is not positive')
            return problems
        if not (math.isfinite(e) and math.isfinite(lv)):
            problems.append(f'{panel}: non-finite percent/log value at index {i}')
            return problems
        checks.append((a, b, e, lv))
    try:
        mean_without = _import_exp_lib()._scaled_mean_or_raw(
            [b for _a, b, _e, _lv in checks])
    except (TypeError, ValueError, OverflowError):
        mean_without = None
    if mean_without is None or not math.isfinite(mean_without) or mean_without <= 0.0:
        problems.append(f'{panel}: the retained PPL mean baseline is unavailable')
        return problems
    for i, (a, b, e, lv) in enumerate(checks):
        expected_pct = 100.0 * (a - b) / mean_without
        if not math.isclose(e, expected_pct, rel_tol=1e-9,
                            abs_tol=1e-9 * max(1.0, abs(expected_pct))):
            problems.append(
                f'{panel}: effects[{i}] does not equal 100*(with-without)/'
                'mean(without) from the retained PPLs')
        expected_log = math.log(a) - math.log(b)
        if not math.isclose(lv, expected_log, rel_tol=1e-9,
                            abs_tol=1e-12):
            problems.append(
                f'{panel}: log_effects[{i}] is not log(with)-log(without) '
                'from the retained PPLs')
    return problems


def audit_panel_scale_sensitivity(path=None):
    """Replay the cross-panel effect-scale sensitivity from its inputs."""
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'audit_report.json')
    out = {'kind': 'panel_scale_sensitivity_audit', 'status': 'ok',
           'path': p, 'source': None, 'components': {}, 'n_contrasts': 0,
           'n_checked': 0, 'n_ok': 0, 'n_fail': 0, 'n_missing': 0,
           'problems': [], 'warnings': [],
           'note': ('percent and log-ratio are different estimands; this gate '
                    'replays both from the recorded per-seed effects and '
                    'checks that a scale-discordant contrast does not keep a '
                    'pooled claim.  It creates no p-value and changes no panel '
                    'verdict.')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no audit report or synthesis artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    synthesis, where = _ps_find_block(data)
    if not isinstance(synthesis, dict):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the artifact carries no cross_panel_synthesis block '
                         'with a scale sensitivity to audit')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['source'] = where
    alpha = synthesis.get('alpha', 0.05)
    min_n = synthesis.get('min_n', 3)
    comps = synthesis.get('components')
    if not isinstance(comps, dict):
        out['status'] = 'fail'
        out['problems'].append('the synthesis block carries no components object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    for comp, comp_block in comps.items():
        if not isinstance(comp_block, dict):
            out['problems'].append(f'{comp}: component block is not an object')
            continue
        contrasts = comp_block.get('contrasts')
        if not isinstance(contrasts, dict):
            out['problems'].append(f'{comp}: contrasts is not an object')
            continue
        comp_out = {'contrasts': {}, 'status': comp_block.get('status')}
        for key, block in contrasts.items():
            label = f'{comp}::{key}'
            out['n_contrasts'] += 1
            if not isinstance(block, dict):
                out['problems'].append(f'{label}: contrast block is not an object')
                continue
            status = block.get('status')
            if status == 'not_measured':
                comp_out['contrasts'][str(key)] = {'status': 'not_measured'}
                continue
            _panels = block.get('panels')
            if isinstance(_panels, dict):
                for _panel, _row in _panels.items():
                    out['problems'].extend(
                        _pss_panel_log_problems(str(_panel), _row))
            stored = block.get('scale_sensitivity')
            if not isinstance(stored, dict):
                out['n_missing'] += 1
                out['warnings'].append(
                    f'{label}: no scale-sensitivity block (older or partial '
                    'synthesis); the percent-only claim is not scale-verified')
                comp_out['contrasts'][str(key)] = {'status': 'not_verifiable_locally'}
                continue
            out['n_checked'] += 1
            # Replay the percent side from meta_input as well, so a tampered
            # `meta` block cannot make the scale record self-consistent.
            points = (block.get('meta_input') or {}).get('points')
            pct_meta_stored = block.get('meta') if isinstance(block.get('meta'), dict) else {}
            if not isinstance(points, (list, tuple)):
                out['problems'].append(
                    f'{label}: pooled contrast has no meta_input points to replay')
                meta_for_replay = pct_meta_stored
            else:
                try:
                    pct_meta_replay = L._cps_pool(
                        list(points), alpha=alpha, min_n=min_n,
                        pair_detail_limit=0)
                except (TypeError, ValueError, OverflowError) as exc:
                    out['problems'].append(
                        f'{label}: percent-meta replay failed '
                        f'({type(exc).__name__}: {exc})')
                    meta_for_replay = pct_meta_stored
                else:
                    meta_for_replay = pct_meta_stored
                    for sub in ('status', 'poolable', 'fixed_effect',
                                'random_effects', 'heterogeneity'):
                        out['problems'].extend(
                            _ps_compare(pct_meta_stored.get(sub),
                                        pct_meta_replay.get(sub),
                                        path=f'{label}/meta/{sub}'))
            try:
                replay = L.cross_panel_scale_contrast(
                    dict(block, meta=meta_for_replay),
                    alpha=alpha, min_n=min_n)
            except (TypeError, ValueError, OverflowError) as exc:
                out['problems'].append(
                    f'{label}: scale replay failed ({type(exc).__name__}: {exc})')
                continue
            mismatch = _ps_compare(
                stored, replay, path=f'{label}/scale_sensitivity')
            out['problems'].extend(mismatch)
            out['problems'].extend(_pss_forbidden_keys(
                stored, path=f'{label}/scale_sensitivity'))
            # The producer must not keep the pooled claim when the replay says
            # the percent and log scales disagree.  This is the gate's point:
            # a self-consistent but unapplied sensitivity is not a gate.
            if replay.get('status') in ('direction_discordant',
                                        'precision_discordant',
                                        'poolability_discordant'):
                if block.get('claim_allowed') is not False:
                    out['problems'].append(
                        f'{label}: scale status {replay.get("status")!r} but '
                        f'claim_allowed is {block.get("claim_allowed")!r}')
                if status not in ('scale_sensitive', 'heterogeneous',
                                  'insufficient_panels', 'dependent_draws'):
                    out['problems'].append(
                        f'{label}: scale status {replay.get("status")!r} but '
                        f'contrast status is {status!r}')
            comp_out['contrasts'][str(key)] = {
                'status': replay.get('status'),
                'stored_status': stored.get('status'),
                'claim_allowed_on_both_scales':
                    replay.get('claim_allowed_on_both_scales'),
                'claim_allowed': block.get('claim_allowed'),
                'audit': 'replayed' if not mismatch else 'mismatch'}
        out['components'][str(comp)] = comp_out
    # The aggregate summary is derived from the per-block records; replay it
    # too so editing only the summary is still detected.
    stored_summary = synthesis.get('scale_sensitivity')
    try:
        replay_summary = L.cross_panel_scale_sensitivity(
            synthesis, alpha=alpha, min_n=min_n)
    except (TypeError, ValueError, OverflowError) as exc:
        out['problems'].append(
            f'summary: scale-sensitivity replay failed '
            f'({type(exc).__name__}: {exc})')
    else:
        if isinstance(stored_summary, dict):
            out['problems'].extend(_ps_compare(
                stored_summary, replay_summary, path='scale_sensitivity'))
            out['problems'].extend(_pss_forbidden_keys(
                stored_summary, path='scale_sensitivity'))
        else:
            out['n_missing'] += 1
            out['warnings'].append(
                'the synthesis has no aggregate scale_sensitivity block')
    if out['problems']:
        out['status'] = 'fail'
        out['n_fail'] = len(out['problems'])
    elif out['n_checked'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no measured contrast carries a scale-sensitivity '
                         'block; the report predates the gate or has none')
    elif out['n_missing']:
        out['status'] = 'warn'
    else:
        out['status'] = 'ok'
        out['n_ok'] = out['n_checked']
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
