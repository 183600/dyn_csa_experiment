def audit_winner_curse(path=None):
    """Result-analysis audit: the stored selection-bias block replays exactly.

    ``finaltest_stats`` records ``winner_curse`` from the frozen per-arm
    endpoint cells.  The block is exploratory by contract and must not carry a
    confirmatory field.  This gate rebuilds it from its own
    ``per_arm_seed_values`` and the declared policy, then compares the observed
    winner, bootstrap optimism, optimism-corrected endpoint, split-sample
    honest value, selection frequencies and selection regret.  A hand-edited
    corrected value, confidence interval or status therefore cannot survive a
    local replay.  A missing legacy block is ``not_verifiable_locally``; a
    legitimate ``not_estimable`` record without raw values is
    ``not_applicable``.  The gate is an internal replay, not a binding of the
    raw values to the original test slice.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'winner_curse_audit', 'status': 'ok', 'path': p,
           'n_arms': 0, 'n_seeds': 0, 'replayed': False,
           'problems': [], 'warnings': [],
           'note': ('exploratory selection-bias readout: no p-value, no FWER '
                    'control, no change to frozen p/Holm/confirmed; this is an '
                    'internal replay of per_arm_seed_values, not a binding of '
                    'those values to the original test slice')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    wc = ft.get('winner_curse') if isinstance(ft, dict) else None
    if wc is None:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the stats artifact carries no winner_curse block '
                         '(legacy/stale analysis artifact)')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(wc, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'winner_curse is a {type(wc).__name__}, not a JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    forbidden = sorted(k for k in
                       ('confirmed', 'confirmed_effective', 'p_holm', 'verified',
                        'verdict', 'holm', 'p_exact_signflip') if k in wc)
    if forbidden:
        out['status'] = 'fail'
        out['problems'].append(
            'winner_curse carries confirmatory field(s) '
            f'{forbidden}; a selection readout must not contain a frozen verdict')
    if wc.get('kind') != 'winner_curse_correction':
        out['status'] = 'fail'
        out['problems'].append(
            f"winner_curse.kind is {wc.get('kind')!r}, not "
            "'winner_curse_correction'")
    status = wc.get('status')
    if status not in L.WINNER_CURSE_STATUSES:
        out['status'] = 'fail'
        out['problems'].append(
            f'winner_curse.status {status!r} is not a declared status')
    values = wc.get('per_arm_seed_values')
    if status == 'not_estimable' and not isinstance(values, dict):
        if out['status'] != 'fail':
            out['status'] = 'not_applicable'
            out['reason'] = (wc.get('reason') or
                             'winner_curse is not estimable and stores no '
                             'per-arm values')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(values, dict) or not values:
        out['status'] = 'fail'
        out['problems'].append(
            'winner_curse has no non-empty per_arm_seed_values for replay')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    pol = wc.get('policy')
    if not isinstance(pol, dict) or not pol:
        out['status'] = 'fail'
        out['problems'].append('winner_curse lacks its declared policy block')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    n_boot = _int_field(pol, 'n_boot')
    boot_seed = _int_field(pol, 'bootstrap_seed')
    split_seed = _int_field(pol, 'split_seed')
    max_partitions = _int_field(pol, 'max_partitions')
    min_seeds = _int_field(pol, 'min_seeds')
    better = pol.get('better')
    lead_level = pol.get('lead_level')
    tie_tol = pol.get('tie_tol', 0.0)
    if (not isinstance(tie_tol, (int, float)) or isinstance(tie_tol, bool)
            or not math.isfinite(float(tie_tol)) or float(tie_tol) < 0.0):
        out['status'] = 'fail'
        out['problems'].append(
            'winner_curse policy.tie_tol is not a non-negative finite number')
    if (n_boot is None or boot_seed is None or split_seed is None
            or max_partitions is None or min_seeds is None):
        out['status'] = 'fail'
        out['problems'].append(
            'winner_curse policy lacks an integer n_boot/bootstrap_seed/'
            'split_seed/max_partitions/min_seeds')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if better not in ('min', 'max'):
        out['status'] = 'fail'
        out['problems'].append(
            f'winner_curse policy.better {better!r} is not min/max')
    try:
        replay = L.winner_curse_correction(
            values, better=better, n_boot=n_boot, seed=boot_seed,
            split_seed=split_seed, max_partitions=max_partitions,
            lead_level=lead_level, min_seeds=min_seeds, tie_tol=tie_tol)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'winner_curse_correction could not replay the stored block: '
            f'{type(exc).__name__}: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    out['n_arms'] = int(replay.get('n_arms') or 0)
    out['n_seeds'] = int(replay.get('n_seeds') or 0)
    out['replayed'] = True
    keys = ('status', 'version', 'better', 'n_arms', 'n_seeds', 'arms',
            'seeds', 'complete_coverage', 'dropped_seeds', 'policy',
            'observed_means', 'observed_winner', 'observed_runner_up',
            'observed_winner_value', 'observed_runner_up_value',
            'observed_gap', 'observed_tied_for_best', 'p_best',
            'observed_winner_p_best', 'bootstrap', 'split_sample',
            'consistency', 'status_reason', 'per_arm_seed_values')
    if replay.get('status') != 'not_estimable':
        _missing = [k for k in keys if k not in wc]
        if _missing:
            out['status'] = 'fail'
            out['problems'].append(
                'winner_curse is missing required field(s): '
                + ', '.join(_missing))
    bad = _arm_rank_mismatches(wc, replay, keys=keys)
    if bad:
        out['status'] = 'fail'
        out['problems'].append(
            'stored winner_curse does not replay from per_arm_seed_values: '
            + ', '.join(bad))
    if wc.get('status') == 'not_estimable' and replay.get('status') != 'not_estimable':
        out['status'] = 'fail'
        out['problems'].append(
            'stored status is not_estimable but the raw values replay as '
            'estimable; the block cannot be trusted')
    if wc.get('status') == 'not_estimable':
        if not wc.get('reason'):
            out['status'] = 'fail'
            out['problems'].append(
                'not_estimable winner_curse does not state a reason')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # Structural coherence of the stored block, independent of the replay.
    def _prob_vector(name, obj):
        if not isinstance(obj, dict) or not obj:
            out['status'] = 'fail'
            out['problems'].append(f'winner_curse.{name} is not a non-empty object')
            return
        vals = []
        for arm, value in obj.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                out['status'] = 'fail'
                out['problems'].append(
                    f'winner_curse.{name}[{arm!r}] is not a number')
                continue
            value = float(value)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                out['status'] = 'fail'
                out['problems'].append(
                    f'winner_curse.{name}[{arm!r}] is not a probability')
            vals.append(value)
        if vals and abs(sum(vals) - 1.0) > 1e-9:
            out['status'] = 'fail'
            out['problems'].append(
                f'winner_curse.{name} sums to {sum(vals)!r}, not 1')

    _prob_vector('p_best', wc.get('p_best'))
    ss = wc.get('split_sample') or {}
    _prob_vector('split_sample.selection_frequency',
                 ss.get('selection_frequency') if isinstance(ss, dict) else None)
    if not isinstance(wc.get('arms'), list) or not isinstance(wc.get('seeds'), list):
        out['status'] = 'fail'
        out['problems'].append('winner_curse lacks list arms/seeds')
    else:
        if wc.get('n_arms') != len(wc.get('arms')) or wc.get('n_seeds') != len(wc.get('seeds')):
            out['status'] = 'fail'
            out['problems'].append(
                'winner_curse n_arms/n_seeds disagree with the arms/seeds lists')
        _arms = set(str(a) for a in wc.get('arms'))
        for _name, _obj in (('p_best', wc.get('p_best')),
                            ('observed_means', wc.get('observed_means')),
                            ('per_arm_seed_values', wc.get('per_arm_seed_values')),
                            ('split_sample.selection_frequency',
                             (wc.get('split_sample') or {}).get('selection_frequency'))):
            if isinstance(_obj, dict) and set(str(a) for a in _obj) != _arms:
                out['status'] = 'fail'
                out['problems'].append(
                    f'winner_curse.{_name} does not cover exactly the stored arms')
    if isinstance(ss, dict):
        _np = ss.get('n_partitions')
        _nf = ss.get('n_folds')
        if _np is not None or _nf is not None:
            if (not isinstance(_np, int) or isinstance(_np, bool)
                    or not isinstance(_nf, int) or isinstance(_nf, bool)
                    or _np < 1 or _nf != 2 * _np):
                out['status'] = 'fail'
                out['problems'].append(
                    'winner_curse.split_sample n_partitions/n_folds are not a '
                    'positive two-direction split')
            elif _np > max_partitions:
                out['status'] = 'fail'
                out['problems'].append(
                    'winner_curse.split_sample n_partitions exceeds max_partitions')
    gap = wc.get('observed_gap')
    if gap is not None and (not isinstance(gap, (int, float))
                            or not math.isfinite(float(gap)) or float(gap) < 0.0):
        out['status'] = 'fail'
        out['problems'].append('winner_curse.observed_gap is not a non-negative finite number')
    if isinstance(ss, dict):
        for key in ('selection_regret_mean', 'selection_regret_ci95'):
            val = ss.get(key)
            if key.endswith('_mean'):
                if val is not None and (not isinstance(val, (int, float))
                                        or not math.isfinite(float(val))
                                        or float(val) < 0.0):
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'winner_curse.split_sample.{key} is not a non-negative finite number')
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                lo, hi = val[0], val[1]
                if (not isinstance(lo, (int, float)) or not isinstance(hi, (int, float))
                        or not math.isfinite(float(lo)) or not math.isfinite(float(hi))
                        or float(lo) > float(hi)):
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'winner_curse.split_sample.{key} is not an ordered finite interval')
    bs = wc.get('bootstrap') or {}
    if isinstance(bs, dict):
        _nv = bs.get('n_valid')
        if (not isinstance(_nv, int) or isinstance(_nv, bool)
                or _nv < 0 or _nv > n_boot):
            out['status'] = 'fail'
            out['problems'].append(
                'winner_curse.bootstrap.n_valid is not an integer in [0, n_boot]')
    if isinstance(bs, dict) and isinstance(bs.get('optimism_ci95'), (list, tuple)):
        pair = bs['optimism_ci95']
        if (len(pair) >= 2 and all(isinstance(x, (int, float)) for x in pair[:2])
                and float(pair[0]) > float(pair[1])):
            out['status'] = 'fail'
            out['problems'].append(
                'winner_curse.bootstrap.optimism_ci95 is not ordered')
    pbest = wc.get('observed_winner_p_best')
    lead = pol.get('lead_level')
    if status == 'selection_robust':
        if isinstance(pbest, (int, float)) and isinstance(lead, (int, float)):
            if float(pbest) < float(lead):
                out['status'] = 'fail'
                out['problems'].append(
                    'selection_robust requires the observed winner P(best) to '
                    'reach lead_level')
        if wc.get('observed_gap_exceeds_regret', (bs or {}).get('observed_gap_exceeds_regret')) is not True:
            out['status'] = 'fail'
            out['problems'].append(
                'selection_robust requires the observed gap to exceed the '
                'bootstrap selection regret')
    if status == 'winner_unstable':
        if isinstance(pbest, (int, float)) and isinstance(lead, (int, float)):
            if float(pbest) >= float(lead):
                out['status'] = 'fail'
                out['problems'].append(
                    'winner_unstable requires P(best) below lead_level')
    if status == 'tied_best':
        tied = wc.get('observed_tied_for_best')
        if not isinstance(tied, list) or len(tied) < 2:
            out['status'] = 'fail'
            out['problems'].append(
                'tied_best requires at least two arms sharing the best value')
    if status == 'selection_noise_dominated':
        if isinstance(pbest, (int, float)) and isinstance(lead, (int, float)):
            if float(pbest) < float(lead):
                out['status'] = 'fail'
                out['problems'].append(
                    'selection_noise_dominated requires P(best) to reach lead_level')
        if (bs or {}).get('observed_gap_exceeds_regret') is not False:
            out['status'] = 'fail'
            out['problems'].append(
                'selection_noise_dominated requires the observed gap to be '
                'below the bootstrap selection regret')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
