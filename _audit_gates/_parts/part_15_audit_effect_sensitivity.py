def _es_close(a, b, rel=1e-9, abs_=1e-12):
    """Compare two finite numbers without turning a missing value into 0."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= max(abs_, rel * max(1.0, abs(fa), abs(fb)))


def _es_audit_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def audit_effect_sensitivity(path=None):
    """Replay the effect-scale / robust-location sensitivity block.

    ``finaltest_stats`` now records, next to every frozen primary comparison,
    an ``effect_sensitivity`` block: the exact sign-flip p on the raw PPL
    difference, the same test on the log-PPL ratio, the geometric ratio, and
    an exact Hodges-Lehmann interval for the paired pseudo-median.  The block
    is descriptive, but it is still a stored claim about the frozen data: a
    hand edit can make a raw-mean-only result look scale-robust without
    touching ``p_exact_signflip``.

    This gate rebuilds the block from its own ``per_seed`` with/without PPL
    pairs and compares every derived field.  It also cross-checks the raw
    deltas against the primary comparison's ``per_seed`` vector, so the two
    blocks cannot drift apart.  A missing block is ``not_verifiable_locally``
    (a legacy artifact), never a pass; a malformed or disagreeing block is
    ``fail``.  The audit is read-only and never changes a p-value, the Holm
    family or a frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'effect_sensitivity_audit', 'status': 'ok', 'path': p,
           'n_comparisons': 0, 'n_replayed': 0, 'n_not_replayable': 0,
           'comparisons': {}, 'problems': [], 'warnings': [],
           'note': ('rebuilds raw/log sign-flip p, geometric ratio and the '
                    'exact Hodges-Lehmann interval from stored per-seed PPL '
                    'pairs; a missing block is not a pass and a disagreement '
                    'is a fail; no frozen p/Holm/verdict is changed')}
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
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['n_comparisons'] = len(comps)
    _man = ft.get('manifest') if isinstance(ft.get('manifest'), dict) else {}
    _seed_whitelist = None
    _seeds = _man.get('seeds')
    if isinstance(_seeds, list):
        _wl = [_es_audit_int(s) for s in _seeds]
        if all(s is not None and 0 <= s <= int(L.MAX_SEED) for s in _wl):
            if len(set(_wl)) == len(_wl):
                _seed_whitelist = set(_wl)
            else:
                out['problems'].append('manifest.seeds contains duplicate seeds')
        else:
            out['problems'].append('manifest.seeds is not a list of exact seed integers')
    _confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision')
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            out['problems'].append(f'{name}: comparison is not a JSON object')
            continue
        es = comp.get('effect_sensitivity')
        if es is None:
            out['n_not_replayable'] += 1
            out['warnings'].append(
                f'{name}: no effect_sensitivity block (legacy artifact); '
                'the scale/location claim is not replayable')
            continue
        if not isinstance(es, dict):
            out['problems'].append(
                f'{name}: effect_sensitivity is a {type(es).__name__}, not an object')
            out['n_not_replayable'] += 1
            continue
        row = {'status': 'fail', 'problems': [], 'n': None, 'stored_status': es.get('status')}
        out['comparisons'][str(name)] = row
        if any(f in es for f in _confirm_fields):
            row['problems'].append(
                'effect_sensitivity carries confirmatory field(s) '
                f'{sorted(set(es) & set(_confirm_fields))}')
        if es.get('kind') != 'effect_scale_location_sensitivity':
            row['problems'].append(
                f"effect_sensitivity.kind is {es.get('kind')!r}, not "
                "'effect_scale_location_sensitivity'")
        status = es.get('status')
        if status not in L.EFFECT_SENSITIVITY_STATUSES:
            row['problems'].append(
                f'effect_sensitivity.status {status!r} is not a declared status')
        per_seed = es.get('per_seed')
        if not isinstance(per_seed, list) or not per_seed:
            if status == 'not_estimable' and not row['problems']:
                row['status'] = 'not_applicable'
                out['n_not_replayable'] += 1
                out['warnings'].append(
                    f'{name}: bare not_estimable effect_sensitivity has no '
                    'per_seed pairs; nothing to replay')
                continue
            row['problems'].append('effect_sensitivity.per_seed is missing or empty')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        a_vals, b_vals, seeds = [], [], []
        _seen = set()
        for i, entry in enumerate(per_seed):
            if not isinstance(entry, dict):
                row['problems'].append(f'per_seed[{i}] is not an object')
                continue
            s = _es_audit_int(entry.get('seed'))
            if s is None or s < 0 or s > int(L.MAX_SEED):
                row['problems'].append(f'per_seed[{i}].seed is not a seed integer')
                continue
            if s in _seen:
                row['problems'].append(f'per_seed[{i}] repeats seed {s}')
                continue
            if _seed_whitelist is not None and s not in _seed_whitelist:
                row['problems'].append(
                    f'per_seed[{i}].seed {s} is not in the frozen manifest seed list')
            av = entry.get('with')
            bv = entry.get('without')
            if (not isinstance(av, (int, float)) or isinstance(av, bool)
                    or not isinstance(bv, (int, float)) or isinstance(bv, bool)
                    or not math.isfinite(float(av)) or not math.isfinite(float(bv))
                    or float(av) <= 0.0 or float(bv) <= 0.0):
                row['problems'].append(
                    f'per_seed[{i}] with/without are not finite positive PPLs')
                continue
            _seen.add(s)
            seeds.append(s)
            a_vals.append(float(av))
            b_vals.append(float(bv))
        if row['problems']:
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        alpha = es.get('alpha', ft.get('alpha', 0.05))
        conf = es.get('conf', L.EFFECT_SENSITIVITY_POLICY['conf'])
        try:
            replay = L.effect_sensitivity(
                a_vals, b_vals, seeds=seeds, alpha=alpha, conf=conf)
        except Exception as exc:
            row['problems'].append(
                f'replay raised {type(exc).__name__}: {exc}')
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        row['n'] = int(replay['n'])
        if _es_audit_int(es.get('n')) != int(replay['n']):
            row['problems'].append(
                f"n={es.get('n')!r} does not equal the replayed n={replay['n']}")
        _ni = _es_audit_int(es.get('n_input'))
        _nd = _es_audit_int(es.get('n_dropped'))
        if _ni is None or _nd is None or _ni != int(replay['n']) + _nd:
            row['problems'].append(
                'n_input/n_dropped do not add up to the replayed valid-pair count')
        _dropped = es.get('dropped')
        if not isinstance(_dropped, list) or len(_dropped) != _nd:
            row['problems'].append(
                'dropped must be a list whose length equals n_dropped')
        else:
            _drop_seeds = []
            for i, entry in enumerate(_dropped):
                if not isinstance(entry, dict):
                    row['problems'].append(f'dropped[{i}] is not an object')
                    continue
                ds = _es_audit_int(entry.get('seed'))
                if ds is None:
                    row['problems'].append(f'dropped[{i}].seed is not a seed integer')
                else:
                    _drop_seeds.append(ds)
            if set(_drop_seeds) & set(seeds):
                row['problems'].append(
                    'a seed appears in both per_seed and dropped')
        _es_seeds = es.get('seeds')
        _ps_seeds = [p.get('seed') for p in per_seed]
        if not isinstance(_es_seeds, list) or _es_seeds != _ps_seeds:
            row['problems'].append(
                'effect_sensitivity.seeds does not match the per_seed order')
        if _ps_seeds != seeds:
            row['problems'].append(
                'per_seed order does not match the validated seed list')
        if replay['status'] != status:
            row['problems'].append(
                f"status={status!r} does not replay (got {replay['status']!r})")
        for key in ('alpha', 'conf'):
            if not _es_close(es.get(key), replay.get(key)):
                row['problems'].append(
                    f'{key}={es.get(key)!r} does not replay ({replay.get(key)!r})')
        _pairs = (
            ('raw', ('mean', 'p_exact_signflip', 'geometric_ratio')),
            ('log_ratio', ('mean', 'p_exact_signflip', 'geometric_ratio',
                           'relative_pct')),
        )
        for block, keys in _pairs:
            got = es.get(block) if isinstance(es.get(block), dict) else {}
            want = replay.get(block) or {}
            for key in keys:
                gv, wv = got.get(key), want.get(key)
                if gv is None and wv is None:
                    continue
                if gv is None or wv is None or not _es_close(gv, wv):
                    row['problems'].append(
                        f'{block}.{key}={gv!r} does not replay ({wv!r})')
        loc = es.get('location') if isinstance(es.get('location'), dict) else {}
        wloc = replay.get('location') or {}
        if loc.get('status') != wloc.get('status'):
            row['problems'].append(
                f"location.status={loc.get('status')!r} does not replay "
                f"({wloc.get('status')!r})")
        for key in ('estimate', 'attained_conf'):
            gv, wv = loc.get(key), wloc.get(key)
            if gv is None and wv is None:
                continue
            if gv is None or wv is None or not _es_close(gv, wv):
                row['problems'].append(
                    f'location.{key}={gv!r} does not replay ({wv!r})')
        gci, wci = loc.get('ci'), wloc.get('ci')
        if gci is None and wci is None:
            pass
        elif (not isinstance(gci, (list, tuple)) or len(gci) < 2
              or not isinstance(wci, (list, tuple)) or len(wci) < 2
              or not _es_close(gci[0], wci[0]) or not _es_close(gci[1], wci[1])):
            row['problems'].append(
                f'location.ci={gci!r} does not replay ({wci!r})')
        gcon = es.get('concordance') if isinstance(es.get('concordance'), dict) else {}
        wcon = replay.get('concordance') or {}
        for key in ('direction_agree', 'significance_agree', 'raw_significant',
                    'log_significant', 'location_excludes_zero', 'all_agree'):
            if gcon.get(key) != wcon.get(key):
                row['problems'].append(
                    f'concordance.{key}={gcon.get(key)!r} does not replay '
                    f'({wcon.get(key)!r})')
        # The block's own per-seed derived values must match its PPL pairs,
        # and the raw deltas must equal the primary comparison vector.
        _comp_raw = comp.get('per_seed')
        for i, entry in enumerate(per_seed):
            s = _es_audit_int(entry.get('seed'))
            av = float(entry.get('with'))
            bv = float(entry.get('without'))
            rd = av - bv
            lr = math.log(av) - math.log(bv)
            if entry.get('raw_delta') is None or not _es_close(entry.get('raw_delta'), rd):
                row['problems'].append(
                    f'per_seed[{i}].raw_delta does not equal with-without')
            if entry.get('log_ratio') is None or not _es_close(entry.get('log_ratio'), lr):
                row['problems'].append(
                    f'per_seed[{i}].log_ratio does not equal log(with/without)')
        _comp_seeds = comp.get('seeds')
        if isinstance(_comp_seeds, list):
            _cs = [_es_audit_int(s) for s in _comp_seeds]
            if any(s is None for s in _cs):
                row['problems'].append('comparison.seeds is not a list of integers')
            elif [int(s) for s in seeds] != [int(s) for s in _cs]:
                row['problems'].append(
                    'effect_sensitivity seeds do not equal the comparison seeds')
        if isinstance(_comp_raw, list):
            try:
                _cs = [_es_audit_int(s) for s in (comp.get('seeds') or [])]
                if any(s is None for s in _cs) or len(_cs) != len(_comp_raw):
                    raise ValueError('comparison seeds and per_seed are misaligned')
                _stored = {int(s): float(d) for s, d in zip(_cs, _comp_raw)}
                for s, av, bv in zip(seeds, a_vals, b_vals):
                    if s in _stored and not _es_close(_stored[s], av - bv):
                        row['problems'].append(
                            f'seed {s}: raw delta disagrees with the primary per_seed vector')
            except (TypeError, ValueError, OverflowError) as exc:
                row['problems'].append(
                    f'comparison per_seed vector is malformed ({exc})')
        if row['problems']:
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            row['status'] = 'fail'
            continue
        row['status'] = 'ok'
        out['n_replayed'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no comparison carries a replayable effect_sensitivity '
                         'block (legacy artifact); the scale/location claim is '
                         'not verifiable locally')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
