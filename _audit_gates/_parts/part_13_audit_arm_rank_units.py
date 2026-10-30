
def _aru_close(a, b, *, rel=1e-12, abs_=1e-12):
    """Structural equality with a numeric tolerance for replay comparisons.

    Part 12 defines the same replay comparison for arm-level rankings; both
    audits must accept exactly the same stored JSON, so this stays a thin
    alias rather than a second implementation that could drift.
    """
    return _arm_rank_close(a, b, rel=rel, abs_=abs_)


def _aru_axis_coherent(axis, *, arms, key):
    if not isinstance(axis, dict):
        return False, f'{key} is not an object'
    pb = axis.get('p_best')
    if not isinstance(pb, dict) or set(pb) != set(arms):
        return False, f'{key}.p_best does not cover exactly the stored arms'
    vals = []
    for arm in arms:
        v = pb[arm]
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0):
            return False, f'{key}.p_best[{arm!r}] is not a probability'
        vals.append(float(v))
    # With competition ranks several arms can share rank 1 in one replicate,
    # so the arm-wise probabilities sum to the expected number of (possibly
    # tied) winners: at least 1 and at most K.  Requiring exactly 1 would
    # reject a legitimate all-identical panel, whose p_best is 1 for every arm.
    total = sum(vals)
    if not (1.0 - 1e-9 <= total <= len(arms) + 1e-9):
        return False, (f'{key}.p_best sums to {total!r}; expected a value in '
                       f'[1, {len(arms)}] for (possibly tied) rank-1 arms')
    ci = axis.get('rank_ci95')
    if not isinstance(ci, dict) or set(ci) != set(arms):
        return False, f'{key}.rank_ci95 does not cover exactly the stored arms'
    for arm in arms:
        c = ci[arm]
        if (not isinstance(c, (list, tuple)) or len(c) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in c)
                or not (1 <= c[0] <= c[1] <= len(arms))):
            return False, f'{key}.rank_ci95[{arm!r}] is not a valid [lo, hi]'
    return True, None


def audit_arm_rank_units(path=None):
    """Result-analysis audit: the stored unit-sensitivity block replays.

    ``finaltest_stats`` records ``arm_rank_units`` next to ``arm_ranking``.
    It is the evaluation-unit axis of the same fixed-arm ranking: a paired
    two-axis bootstrap over seeds and frozen evaluation units.  This gate
    rebuilds the nested per-arm/seed NLL matrices from the block's own
    ``per_arm_seed_nll`` / ``per_unit_tokens`` and replays every derived
    readout, so a hand-edited ``P(best)``, rank interval or status cannot
    survive.  It also rejects confirmatory fields and checks that the declared
    statuses/bands are internally coherent.

    A missing legacy block is ``not_verifiable_locally`` (not a pass); a bare
    ``not_estimable`` record with no raw matrices is ``not_applicable``.  It
    is an internal replay: it cannot prove the matrices came from the original
    test slice, which is the job of the stamped-input/analysis-integrity
    gates.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'arm_rank_unit_audit', 'status': 'ok', 'path': p,
           'n_arms': 0, 'n_seeds': 0, 'n_units': 0, 'replayed': False,
           'problems': [], 'warnings': [],
           'note': ('exploratory evaluation-unit sensitivity of the fixed-arm '
                    'ranking: no p-value, no FWER control, no change to frozen '
                    'p/Holm/confirmed; this is an internal replay of the stored '
                    'per-unit NLL/token matrices, not a binding to the original '
                    'test slice')}
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
    rk = ft.get('arm_rank_units') if isinstance(ft, dict) else None
    if rk is None:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the stats artifact carries no arm_rank_units block '
                         '(legacy/stale analysis artifact)')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(rk, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'arm_rank_units is a {type(rk).__name__}, not a JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    forbidden = sorted(k for k in
                       ('confirmed', 'confirmed_effective', 'p_holm', 'verified',
                        'verdict', 'holm', 'p_exact_signflip') if k in rk)
    if forbidden:
        out['status'] = 'fail'
        out['problems'].append(
            'arm_rank_units carries confirmatory field(s) '
            f'{forbidden}; a ranking readout must not contain a frozen verdict')
    if rk.get('kind') != 'arm_rank_unit_sensitivity':
        out['status'] = 'fail'
        out['problems'].append(
            f"arm_rank_units.kind is {rk.get('kind')!r}, not "
            "'arm_rank_unit_sensitivity'")
    status = rk.get('status')
    if status not in L.ARM_RANK_UNIT_STATUSES:
        out['status'] = 'fail'
        out['problems'].append(
            f'arm_rank_units.status {status!r} is not a declared status')

    raw_nll = rk.get('per_arm_seed_nll')
    tokens = rk.get('per_unit_tokens')
    arms = rk.get('arms')
    seeds = rk.get('seeds')
    if status == 'not_estimable' and not isinstance(raw_nll, dict):
        if out['status'] != 'fail':
            out['status'] = 'not_applicable'
            out['reason'] = (rk.get('reason') or
                             'arm_rank_units is not estimable and stores no raw matrices')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if (not isinstance(raw_nll, dict) or not raw_nll
            or not isinstance(tokens, list) or not tokens
            or not isinstance(arms, list) or not arms
            or not isinstance(seeds, list) or not seeds):
        out['status'] = 'fail'
        out['problems'].append(
            'arm_rank_units lacks non-empty per_arm_seed_nll / per_unit_tokens / arms / seeds')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if (len(set(arms)) != len(arms)
            or any(not isinstance(a, str) or not a for a in arms)):
        out['status'] = 'fail'
        out['problems'].append('arm_rank_units.arms is not a list of unique non-empty strings')
    if (len(set(seeds)) != len(seeds)
            or any(isinstance(sd, bool) or not isinstance(sd, int) for sd in seeds)):
        out['status'] = 'fail'
        out['problems'].append('arm_rank_units.seeds is not a list of unique integers')
    if set(raw_nll) != set(arms):
        out['status'] = 'fail'
        out['problems'].append(
            'arm_rank_units.per_arm_seed_nll does not cover exactly the stored arms')
    if out['status'] == 'fail' and out['problems']:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _positive_int(key, default=None):
        raw = rk.get(key, default)
        if isinstance(raw, (bool, np.bool_)) or raw is None:
            return None
        if isinstance(raw, numbers.Integral):
            # Keep large exact integers exact instead of forcing them through
            # float64 (a legal bootstrap seed may exceed the float range).
            return int(raw)
        if isinstance(raw, float):
            return int(raw) if raw.is_integer() else None
        try:
            as_int = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            if raw != as_int:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        return as_int

    n_boot = _positive_int('n_boot')
    boot_seed = _positive_int('bootstrap_seed')
    min_seeds = _positive_int('min_seeds', 5)
    min_units = _positive_int('min_units', 8)
    tie_tol = rk.get('tie_tol', 0.0)
    for _name, _v in (('n_boot', n_boot), ('bootstrap_seed', boot_seed),
                      ('min_seeds', min_seeds), ('min_units', min_units)):
        if _v is None:
            out['status'] = 'fail'
            out['problems'].append(
                f'arm_rank_units lacks an integer {_name}')
    if out['status'] == 'fail' and out['problems']:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    better = rk.get('better')
    if better not in ('min', 'max'):
        out['status'] = 'fail'
        out['problems'].append(f'arm_rank_units.better {better!r} is not min/max')
    levels = {}
    for key in ('unit_level', 'seed_level', 'lead_level'):
        v = rk.get(key)
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0):
            out['status'] = 'fail'
            out['problems'].append(f'arm_rank_units.{key} is not a probability')
        else:
            levels[key] = float(v)

    # Rebuild the nested input from the block's own raw matrices.
    unit_ids = rk.get('unit_ids')
    if unit_ids is not None and (not isinstance(unit_ids, list)
                                 or len(unit_ids) != len(tokens)):
        out['status'] = 'fail'
        out['problems'].append('arm_rank_units.unit_ids is misaligned with tokens')
        unit_ids = None
    arm_units = {}
    try:
        for arm in arms:
            row = raw_nll.get(arm)
            if not isinstance(row, dict):
                raise ValueError(f'per_arm_seed_nll[{arm!r}] is not an object')
            if set(row) != {str(s) for s in seeds}:
                raise ValueError(
                    f'per_arm_seed_nll[{arm!r}] does not cover exactly the stored seeds')
            arm_units[arm] = {}
            for s in seeds:
                nll = row[str(s)]
                if not isinstance(nll, list) or len(nll) != len(tokens):
                    raise ValueError(
                        f'per_arm_seed_nll[{arm!r}][{s!r}] is misaligned with tokens')
                cell = {'nll': nll, 'tokens': tokens}
                if unit_ids is not None:
                    cell['unit_ids'] = unit_ids
                arm_units[arm][int(s)] = cell
    except (TypeError, ValueError, KeyError) as exc:
        out['status'] = 'fail'
        out['problems'].append(f'could not rebuild the raw matrices: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    pav = rk.get('per_arm_seed_values')
    # Only pass the endpoint cross-check when the stored block declares that
    # it performed it.  A block generated without ``per_arm_seed_values``
    # legitimately records ``endpoint_values_match='not_checked'``; forcing the
    # check during replay would make that block unverifiable rather than
    # replaying the same computation.
    _cross_check = (rk.get('endpoint_values_match') == 'checked')
    _all_numeric = False
    if isinstance(pav, dict):
        _all_numeric = all(
            isinstance(row, dict)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(float(x)) for x in row.values())
            for row in pav.values())
    try:
        replay = L.arm_rank_unit_sensitivity(
            arm_units, better=better, n_boot=n_boot, seed=boot_seed,
            min_seeds=min_seeds, min_units=min_units,
            unit_level=levels.get('unit_level'),
            seed_level=levels.get('seed_level'),
            lead_level=levels.get('lead_level'), tie_tol=tie_tol,
            unit_ids=unit_ids,
            per_arm_seed_values=(pav if (_all_numeric and _cross_check) else None),
            unit_granularity=rk.get('unit_granularity'))
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'arm_rank_unit_sensitivity could not replay the stored block: '
            f'{type(exc).__name__}: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    out['n_arms'] = int(replay.get('n_arms') or 0)
    out['n_seeds'] = int(replay.get('n_seeds') or 0)
    out['n_units'] = int(replay.get('n_units') or 0)
    out['replayed'] = True
    keys = ('status', 'better', 'n_boot', 'bootstrap_seed', 'unit_level',
            'seed_level', 'lead_level', 'tie_tol', 'min_seeds', 'min_units',
            'n_arms', 'n_seeds', 'n_units', 'arms', 'seeds', 'dropped_seeds',
            'unit_granularity',
            'complete_seed_coverage', 'unit_ids', 'population_claim_allowed',
            'per_unit_tokens', 'per_arm_seed_nll', 'per_arm_seed_values',
            'observed_means', 'observed_log_means', 'observed_scaled_means',
            'observed_rank',
            'observed_winner', 'observed_runner_up', 'observed_tied_for_best',
            'observed_gap', 'observed_gap_units', 'practical_tie',
            'seed_only', 'unit_only', 'two_way', 'gap_decomposition',
            'endpoint_values_match')
    if rk.get('status') != 'not_estimable':
        _missing = [k for k in keys if k not in rk]
        if _missing:
            out['status'] = 'fail'
            out['problems'].append(
                'arm_rank_units is missing required field(s): ' + ', '.join(_missing))
    bad = [k for k in keys if k in rk and not _aru_close(rk.get(k), replay.get(k))]
    if bad:
        out['status'] = 'fail'
        out['problems'].append(
            'stored arm_rank_units does not replay from its raw matrices: '
            + ', '.join(bad))
    if rk.get('status') == 'not_estimable' and replay.get('status') != 'not_estimable':
        out['status'] = 'fail'
        out['problems'].append(
            'stored status is not_estimable but the raw matrices replay as '
            'estimable; the block cannot be trusted')
    if rk.get('status') == 'not_estimable':
        if not rk.get('reason'):
            out['status'] = 'fail'
            out['problems'].append(
                'not_estimable arm_rank_units does not state a reason')
        if out['problems']:
            out['status'] = 'fail'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # Structural coherence independent of the replay.
    if not isinstance(arms, list) or len(set(arms)) != len(arms):
        out['status'] = 'fail'
        out['problems'].append('arm_rank_units.arms is not a unique list')
    else:
        if rk.get('n_arms') != len(arms) or rk.get('n_seeds') != len(seeds):
            out['status'] = 'fail'
            out['problems'].append(
                'arm_rank_units n_arms/n_seeds disagree with arms/seeds')
        for axis_key in ('seed_only', 'unit_only', 'two_way'):
            ok, why = _aru_axis_coherent(rk.get(axis_key), arms=arms, key=axis_key)
            if not ok:
                out['status'] = 'fail'
                out['problems'].append(why)
        winner = rk.get('observed_winner')
        orank = rk.get('observed_rank')
        if not isinstance(orank, dict) or winner not in orank:
            out['status'] = 'fail'
            out['problems'].append(
                'arm_rank_units has no observed_rank entry for observed_winner')
        elif orank.get(winner) != 1:
            out['status'] = 'fail'
            out['problems'].append(
                'observed_winner does not carry observed_rank 1')
        if rk.get('status') == 'stable':
            for axis_key in ('seed_only', 'unit_only', 'two_way'):
                axis = rk.get(axis_key) or {}
                level = (levels.get('seed_level') if axis_key == 'seed_only'
                         else levels.get('unit_level'))
                if axis.get('p_best', {}).get(winner, 0.0) < float(level or 0.0):
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'stable status but {axis_key} p_best for the observed '
                        'winner is below the declared band')
                if axis.get('p_best_argmax') != winner:
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'stable status but {axis_key} p_best_argmax is not the '
                        'observed winner')
    if out['problems']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
