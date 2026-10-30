
def _arm_rank_close(a, b, *, rel=1e-12, abs_=1e-12):
    """Structural equality with a numeric tolerance for replay comparisons."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return (isinstance(a, (bool, np.bool_))
                and isinstance(b, (bool, np.bool_))
                and bool(a) == bool(b))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        if (isinstance(a, (complex, np.complexfloating))
                or isinstance(b, (complex, np.complexfloating))):
            return False
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError, OverflowError):
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return fa == fb
        return abs(fa - fb) <= max(abs_, rel * max(1.0, abs(fa), abs(fb)))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_arm_rank_close(a[k], b[k], rel=rel, abs_=abs_) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_arm_rank_close(x, y, rel=rel, abs_=abs_) for x, y in zip(a, b))
    return a == b


def _arm_rank_mismatches(stored, replay, *, keys):
    bad = []
    for key in keys:
        if key not in stored:
            continue
        if not _arm_rank_close(stored.get(key), replay.get(key)):
            bad.append(key)
    return bad


def audit_arm_ranking(path=None):
    """Result-analysis audit: the stored arm-ranking block replays exactly.

    ``finaltest_stats`` records ``arm_ranking`` (paired seed-bootstrap ranking
    resolution) from the frozen per-arm endpoint cells.  This gate recomputes
    that block from its own stored ``per_arm_seed_values`` and compares every
    derived readout, so a hand-edited ``p_best`` / rank interval / winner /
    status cannot survive.  The block is exploratory by contract: it must not
    carry a confirmatory field (``confirmed`` / ``p_holm`` / ``verified`` or a
    verdict), and a missing legacy block is ``not_verifiable_locally`` rather
    than a pass.  A failed replay is a ``fail``; a legitimate
    ``not_estimable`` record without raw values is ``not_applicable``.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'arm_rank_audit', 'status': 'ok', 'path': p,
           'n_arms': 0, 'n_seeds': 0, 'replayed': False,
           'problems': [], 'warnings': [],
           'note': ('exploratory ranking resolution: no p-value, no FWER '
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
    rk = ft.get('arm_ranking') if isinstance(ft, dict) else None
    if rk is None:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the stats artifact carries no arm_ranking block '
                         '(legacy/stale analysis artifact)')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(rk, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'arm_ranking is a {type(rk).__name__}, not a JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    forbidden = sorted(k for k in
                       ('confirmed', 'confirmed_effective', 'p_holm', 'verified',
                        'verdict', 'holm', 'p_exact_signflip') if k in rk)
    if forbidden:
        out['status'] = 'fail'
        out['problems'].append(
            'arm_ranking carries confirmatory field(s) '
            f'{forbidden}; a ranking readout must not contain a frozen verdict')
    if rk.get('kind') != 'arm_rank_inference':
        out['status'] = 'fail'
        out['problems'].append(
            f"arm_ranking.kind is {rk.get('kind')!r}, not 'arm_rank_inference'")
    status = rk.get('status')
    if status not in L.ARM_RANK_STATUSES:
        out['status'] = 'fail'
        out['problems'].append(f'arm_ranking.status {status!r} is not a declared status')
    values = rk.get('per_arm_seed_values')
    if status == 'not_estimable' and not isinstance(values, dict):
        # A block recorded before any arm/endpoint existed has no raw values;
        # it is a legitimate refusal, not a replayable measurement.  Do not
        # turn "no ranking was computable" into a pass, and do not let the
        # downgrade erase an already-detected forbidden/kind/status problem.
        if out['status'] != 'fail':
            out['status'] = 'not_applicable'
            out['reason'] = (rk.get('reason') or
                             'arm_ranking is not estimable and stores no per-arm values')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(values, dict) or not values:
        out['status'] = 'fail'
        out['problems'].append(
            'arm_ranking has no non-empty per_arm_seed_values for replay')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _positive_int(key, default=None):
        raw = rk.get(key, default)
        if isinstance(raw, (bool, np.bool_)) or raw is None:
            return None
        try:
            as_int = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            if float(raw) != float(as_int):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        return as_int

    n_boot = _positive_int('n_boot')
    boot_seed = _positive_int('bootstrap_seed')
    min_seeds = _positive_int('min_seeds', 5)
    if n_boot is None or boot_seed is None or min_seeds is None:
        out['status'] = 'fail'
        out['problems'].append(
            'arm_ranking lacks an integer n_boot/bootstrap_seed/min_seeds')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    better = rk.get('better')
    if better not in ('min', 'max'):
        out['status'] = 'fail'
        out['problems'].append(f'arm_ranking.better {better!r} is not min/max')
    rank_level = rk.get('rank_level')
    lead_level = rk.get('lead_level')
    tie_tol = rk.get('tie_tol', 0.0)
    try:
        replay = L.arm_rank_inference(
            values, better=better, n_boot=n_boot, seed=boot_seed,
            rank_level=rank_level, lead_level=lead_level,
            tie_tol=tie_tol, min_seeds=min_seeds)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'arm_rank_inference could not replay the stored block: '
            f'{type(exc).__name__}: {exc}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    out['n_arms'] = int(replay.get('n_arms') or 0)
    out['n_seeds'] = int(replay.get('n_seeds') or 0)
    out['replayed'] = True
    keys = ('status', 'better', 'n_boot', 'bootstrap_seed', 'p_best_resolution', 'rank_level',
            'lead_level', 'tie_tol', 'min_seeds',
            'n_arms', 'n_seeds', 'arms', 'seeds', 'complete_coverage',
            'dropped_seeds', 'observed_means', 'observed_rank',
            'observed_winner', 'observed_runner_up', 'observed_tied_for_best',
            'observed_gap', 'top2_gap_ci95', 'top2_gap_resolved', 'top2_order_prob', 'p_best',
            'p_top2', 'mean_rank', 'rank_ci95', 'p_best_max', 'p_best_argmax',
            'observed_winner_p_best',
            'p_best_entropy_normalized', 'effective_best_candidates',
            'winner_stable_loo', 'practical_tie', 'n_distinct_bootstrap_winners',
            'pairwise_p_better', 'reason', 'status_reason', 'loo')
    # A stored estimable block must not hide a tampered field by deleting it;
    # only ``reason`` is legitimately absent from an estimable block.
    if rk.get('status') != 'not_estimable':
        _missing = [k for k in keys if k != 'reason' and k not in rk]
        if _missing:
            out['status'] = 'fail'
            out['problems'].append(
                'arm_ranking is missing required field(s): ' + ', '.join(_missing))
    bad = _arm_rank_mismatches(rk, replay, keys=keys)
    if bad:
        out['status'] = 'fail'
        out['problems'].append(
            'stored arm_ranking does not replay from per_arm_seed_values: '
            + ', '.join(bad))
    if rk.get('status') == 'not_estimable' and replay.get('status') != 'not_estimable':
        out['status'] = 'fail'
        out['problems'].append(
            'stored status is not_estimable but the raw values replay as '
            'estimable; the block cannot be trusted')
    if rk.get('status') == 'not_estimable':
        # A refusal that still carries its raw cells was replayed above; it is
        # verified, not downgraded.  Only the estimable derived fields (p_best,
        # rank intervals, ...) are legitimately absent.
        if not rk.get('reason'):
            out['status'] = 'fail'
            out['problems'].append(
                'not_estimable arm_ranking does not state a reason')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # Structural coherence of the stored block, independent of the replay.
    arms = rk.get('arms')
    seeds = rk.get('seeds')
    if not isinstance(arms, list) or not isinstance(seeds, list):
        out['status'] = 'fail'
        out['problems'].append('arm_ranking lacks list arms/seeds')
    else:
        if len(set(arms)) != len(arms):
            out['status'] = 'fail'
            out['problems'].append('arm_ranking.arms contains a duplicate')
        if rk.get('n_arms') != len(arms) or rk.get('n_seeds') != len(seeds):
            out['status'] = 'fail'
            out['problems'].append(
                'arm_ranking n_arms/n_seeds disagree with the arms/seeds lists')
        for arm in arms:
            row = values.get(arm)
            if not isinstance(row, dict):
                out['status'] = 'fail'
                out['problems'].append(f'per_arm_seed_values[{arm!r}] is not an object')
                continue
            if set(row) != {str(s) for s in seeds}:
                out['status'] = 'fail'
                out['problems'].append(
                    f'per_arm_seed_values[{arm!r}] does not cover exactly the stored seeds')
        pb = rk.get('p_best')
        if isinstance(pb, dict) and pb:
            vals = [float(v) for v in pb.values()]
            if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in vals):
                out['status'] = 'fail'
                out['problems'].append('p_best contains a non-probability')
            elif abs(sum(vals) - 1.0) > 1e-9:
                out['status'] = 'fail'
                out['problems'].append(
                    f'p_best sums to {sum(vals)!r}, not 1')
        else:
            out['status'] = 'fail'
            out['problems'].append('arm_ranking lacks a p_best object')
        rci = rk.get('rank_ci95')
        if isinstance(rci, dict):
            for arm, ci in rci.items():
                if (not isinstance(ci, (list, tuple)) or len(ci) != 2
                        or not all(isinstance(x, int) and not isinstance(x, bool)
                                   for x in ci)
                        or not (1 <= ci[0] <= ci[1] <= len(arms))):
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'rank_ci95[{arm!r}] is not a valid [lo, hi] in 1..K')
        else:
            out['status'] = 'fail'
            out['problems'].append('arm_ranking lacks rank_ci95')
        orank = rk.get('observed_rank')
        if isinstance(orank, dict):
            if any((not isinstance(v, int) or isinstance(v, bool)
                    or not (1 <= v <= len(arms))) for v in orank.values()):
                out['status'] = 'fail'
                out['problems'].append('observed_rank contains an out-of-range rank')
            winner = rk.get('observed_winner')
            if winner in orank and orank[winner] != 1:
                out['status'] = 'fail'
                out['problems'].append(
                    'observed_winner does not carry observed_rank 1')
        else:
            out['status'] = 'fail'
            out['problems'].append('arm_ranking lacks observed_rank')
        if rk.get('status') == 'pairwise_only' and len(arms) != 2:
            out['status'] = 'fail'
            out['problems'].append('pairwise_only status with a non-two-arm panel')
        if rk.get('status') == 'decisive':
            pl = rk.get('rank_level')
            if (not isinstance(pl, (int, float))
                    or float(rk.get('observed_winner_p_best', -1)) < float(pl)):
                out['status'] = 'fail'
                out['problems'].append(
                    'decisive status below the declared rank_level for the '
                    'observed winner')
            if rk.get('p_best_argmax') != rk.get('observed_winner'):
                out['status'] = 'fail'
                out['problems'].append(
                    'decisive status but p_best_argmax is not the observed winner')
            if rk.get('top2_gap_resolved') is not True:
                out['status'] = 'fail'
                out['problems'].append(
                    'decisive status without a top-two gap interval that excludes zero')
            if rk.get('winner_stable_loo') is not True:
                out['status'] = 'fail'
                out['problems'].append(
                    'decisive status without leave-one-seed-out winner stability')
    if out['problems']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
