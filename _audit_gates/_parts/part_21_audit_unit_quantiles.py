def _uqa_int(value):
    """Exact integer for one stored count/label, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def audit_unit_quantiles(path=None):
    """Replay the equal-unit distribution of every paired PPL contrast.

    ``finaltest_stats`` stores the sufficient statistics (per-seed summed-NLL
    differences, token counts, unit ids, seeds and requested quantiles) next
    to the derived ``unit_quantile_profile``.  This gate rebuilds the whole
    block with the same :func:`exp_lib.unit_quantile_profile` implementation
    and compares every derived number.  A hand edit that makes the effect look
    broad-based, or that hides a harmed tail, without moving the sufficient
    statistics is therefore caught.

    A missing block is ``not_verifiable_locally`` (a legacy artifact), a bare
    ``not_estimable`` block is ``not_applicable``, and a malformed or
    disagreeing block is ``fail``.  The gate is read-only and never changes a
    p-value, the Holm family or a frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'unit_quantile_audit', 'status': 'ok', 'path': p,
           'n_comparisons': 0, 'n_replayed': 0, 'n_not_replayable': 0,
           'n_not_estimable': 0, 'n_blocks_seen': 0, 'n_ok': 0,
           'comparisons': {}, 'problems': [], 'warnings': [],
           'note': ('rebuilds the per-unit quantile profile from the stored '
                    'per-seed NLL differences, token counts, ids, seeds and '
                    'requested quantiles; a missing block is not a pass and a '
                    'disagreement is a fail; no frozen p/Holm/verdict is '
                    'changed')}
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
    confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision')
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            out['problems'].append(f'{name}: comparison is not a JSON object')
            continue
        dsm = comp.get('doc_sampling')
        uq = dsm.get('unit_quantile_profile') if isinstance(dsm, dict) else None
        if uq is None:
            out['n_not_replayable'] += 1
            out['warnings'].append(
                f'{name}: no unit_quantile_profile block (legacy artifact); '
                'the per-unit distribution is not replayable')
            continue
        row = {'status': 'fail', 'problems': [], 'n_units': None,
               'n_seeds': None, 'n_quantiles': None}
        out['comparisons'][str(name)] = row
        out['n_blocks_seen'] += 1
        if not isinstance(uq, dict):
            row['problems'].append(
                f'unit_quantile_profile is a {type(uq).__name__}, not an object')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if any(f in uq for f in confirm_fields):
            row['problems'].append(
                'unit_quantile_profile carries confirmatory field(s) '
                f'{sorted(set(uq) & set(confirm_fields))}')
        if uq.get('kind') != 'unit_quantile_profile':
            row['problems'].append(
                f"unit_quantile_profile.kind is {uq.get('kind')!r}, not "
                "'unit_quantile_profile'")
        status = uq.get('status')
        if status not in L.UNIT_QUANTILE_STATUSES:
            row['problems'].append(
                f'unit_quantile_profile.status {status!r} is not a declared status')
        if status == 'not_estimable':
            if not row['problems']:
                row['status'] = 'not_applicable'
                out['n_not_replayable'] += 1
                out['n_not_estimable'] += 1
                out['warnings'].append(
                    f'{name}: unit_quantile_profile is not_estimable '
                    f'({uq.get("reason") or "no reason recorded"}); nothing '
                    'to replay')
                continue
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if status != 'ok':
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        stats = uq.get('sufficient_statistics')
        if not isinstance(stats, dict):
            row['problems'].append(
                'unit_quantile_profile.sufficient_statistics is not an object')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        diff = stats.get('per_seed_nll_diff')
        toks = stats.get('n_tokens')
        ids = stats.get('unit_ids')
        seeds = stats.get('seeds')
        qs = stats.get('quantiles')
        if not isinstance(diff, list) or not diff or not all(
                isinstance(r, list) and r for r in diff):
            row['problems'].append(
                'sufficient_statistics.per_seed_nll_diff is missing or empty')
        if not isinstance(toks, list) or not toks:
            row['problems'].append(
                'sufficient_statistics.n_tokens is missing or empty')
        if not isinstance(ids, list) or not ids:
            row['problems'].append(
                'sufficient_statistics.unit_ids is missing or empty')
        if not isinstance(seeds, list) or not seeds:
            row['problems'].append(
                'sufficient_statistics.seeds is missing or empty')
        if not isinstance(qs, list) or not qs:
            row['problems'].append(
                'sufficient_statistics.quantiles is missing or empty')
        n_units = _uqa_int(uq.get('n_units'))
        n_seeds = _uqa_int(uq.get('n_seeds'))
        n_tokens = _uqa_int(uq.get('n_tokens'))
        if n_units is None or n_units < int(L.UNIT_QUANTILE_POLICY['min_units']):
            row['problems'].append('n_units is not an integer at or above the policy minimum')
        if n_seeds is None or n_seeds < 1:
            row['problems'].append('n_seeds is not a positive integer')
        if n_tokens is None or n_tokens < 1:
            row['problems'].append('n_tokens is not a positive integer')
        if isinstance(toks, list):
            tc = []
            for x in toks:
                if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                    row['problems'].append(
                        f'n_tokens contains {x!r}, not a positive integer')
                    tc = []
                    break
                tc.append(int(x))
            if tc and n_tokens is not None and sum(tc) != n_tokens:
                row['problems'].append('n_tokens does not equal the sum of token counts')
            if tc and n_units is not None and len(tc) != n_units:
                row['problems'].append('token counts are not aligned with n_units')
            if tc and isinstance(diff, list) and any(
                    isinstance(r, list) and len(r) != len(tc) for r in diff):
                row['problems'].append(
                    'a per-seed NLL-difference row is not aligned with n_units')
        if isinstance(diff, list) and n_seeds is not None and len(diff) != n_seeds:
            row['problems'].append('per_seed_nll_diff rows do not match n_seeds')
        if isinstance(diff, list):
            for ri, r in enumerate(diff):
                if not isinstance(r, list):
                    row['problems'].append(
                        f'per_seed_nll_diff[{ri}] is not a list')
                    break
                for v in r:
                    if isinstance(v, (bool, str, bytes, bytearray)):
                        row['problems'].append(
                            'per_seed_nll_diff contains a non-numeric value')
                        break
                    if not isinstance(v, (int, float)):
                        row['problems'].append(
                            'per_seed_nll_diff contains a non-numeric value')
                        break
                    if not math.isfinite(float(v)):
                        row['problems'].append(
                            'per_seed_nll_diff contains a non-finite value')
                        break
                else:
                    continue
                break
        if isinstance(ids, list) and n_units is not None:
            if len(ids) != n_units:
                row['problems'].append('unit_ids are not aligned with n_units')
            elif len(set(str(x) for x in ids)) != len(ids):
                row['problems'].append('unit_ids contain duplicates')
        if isinstance(seeds, list) and n_seeds is not None:
            norm = [_uqa_int(s) for s in seeds]
            if any(s is None for s in norm):
                row['problems'].append('seeds contains a non-integer label')
            elif len(set(norm)) != len(norm):
                row['problems'].append('seeds contains duplicate labels')
            if len(norm) != n_seeds:
                row['problems'].append('seeds length differs from n_seeds')
        if isinstance(qs, list):
            if any(isinstance(q, bool) or not isinstance(q, (int, float))
                   or not math.isfinite(float(q)) or not (0.0 < float(q) < 1.0)
                   for q in qs):
                row['problems'].append(
                    'quantiles contains a value outside (0, 1)')
            elif any(float(qs[i]) >= float(qs[i + 1]) for i in range(len(qs) - 1)):
                row['problems'].append('quantiles are not strictly increasing')
        dsm_units = _uqa_int(dsm.get('pooled_ci_n_units')) if isinstance(dsm, dict) else None
        if dsm_units is not None and n_units is not None and dsm_units != n_units:
            row['problems'].append(
                'pooled_ci_n_units does not match unit_quantile_profile.n_units')
        comp_pairs = comp.get('per_seed')
        if isinstance(comp_pairs, list) and comp_pairs and isinstance(seeds, list):
            comp_seeds = []
            ok_pairs = True
            for rec in comp_pairs:
                if not isinstance(rec, dict):
                    ok_pairs = False
                    break
                s = _uqa_int(rec.get('seed'))
                if s is None:
                    ok_pairs = False
                    break
                comp_seeds.append(s)
            if ok_pairs and set(comp_seeds) != set(_uqa_int(s) for s in seeds):
                row['problems'].append(
                    'unit_quantile_profile.seeds do not match the comparison per_seed seeds')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        policy = uq.get('policy') if isinstance(uq.get('policy'), dict) else {}
        tails = uq.get('tails') if isinstance(uq.get('tails'), dict) else {}
        conf = policy.get('conf')
        tf = tails.get('tail_fraction', policy.get('tail_fraction'))
        try:
            replay = L.unit_quantile_profile(
                diff, toks, unit_ids=ids, seeds=seeds, quantiles=qs,
                conf=conf, tail_fraction=tf)
        except Exception as exc:
            row['problems'].append(
                f'replay raised {type(exc).__name__}: {exc}')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if not isinstance(replay, dict) or replay.get('status') != 'ok':
            row['problems'].append('replay did not produce an ok block')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        # The block carries its own sufficient statistics, so the strongest
        # check is a full structural replay: every quantile, interval, tail
        # mean and derived flag must be rebuilt from the stored inputs.
        _lsa_same(uq, replay, 'unit_quantile_profile', row['problems'])
        row['n_units'] = n_units
        row['n_seeds'] = n_seeds
        row['n_quantiles'] = len(qs)
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
        else:
            row['status'] = 'ok'
            out['n_replayed'] += 1
            out['n_ok'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        if (out['n_not_estimable'] > 0
                and out['n_not_estimable'] == out['n_blocks_seen']
                and out['n_blocks_seen'] == out['n_comparisons']):
            out['status'] = 'not_applicable'
            out['reason'] = 'every stored unit_quantile_profile block is not_estimable'
        else:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'no unit_quantile_profile block could be replayed'
    elif out['n_ok'] == 0:
        out['status'] = 'not_applicable'
        out['reason'] = 'every stored unit_quantile_profile block is not_estimable'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
