def _lsa_close(got, want, rel=1e-9, abs_=1e-12):
    if got is None or want is None:
        return got is None and want is None
    try:
        a, b = float(got), float(want)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(abs_, rel * max(1.0, abs(a), abs(b)))


def _lsa_same(got, want, path, problems):
    """Strict structural comparison of a replayed block with the stored one."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            problems.append(f'{path}: stored value is not an object')
            return
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        if missing:
            problems.append(f'{path}: missing keys {missing}')
        if extra:
            problems.append(f'{path}: unexpected keys {extra}')
        for key in sorted(set(want) & set(got), key=str):
            _lsa_same(got[key], want[key], f'{path}.{key}', problems)
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            problems.append(f'{path}: list length/shape differs')
            return
        for i, (g, w) in enumerate(zip(got, want)):
            _lsa_same(g, w, f'{path}[{i}]', problems)
        return
    if isinstance(want, bool) or isinstance(got, bool):
        if got is not want:
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if isinstance(want, int) and isinstance(got, int):
        if int(got) != int(want):
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if not _lsa_close(got, want):
            problems.append(f'{path}: {got!r} != replayed {want!r}')
        return
    if got != want:
        problems.append(f'{path}: {got!r} != replayed {want!r}')


def _lsa_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def audit_length_strata(path=None):
    """Replay the post-hoc length post-stratification of every comparison.

    ``finaltest_stats`` stores the sufficient statistics (per-unit token
    counts, per-stratum unit indices and per-seed summed NLLs) next to the
    derived length-strata block.  This gate rebuilds the block from those
    statistics by treating each stratum as one aggregated unit and running the
    same :func:`exp_lib.length_stratified_effect` implementation, then compares
    every derived number.  A hand edit that makes the token-pooled effect look
    robust to length composition without moving the sufficient statistics is
    therefore caught.

    A missing block is ``not_verifiable_locally`` (a legacy artifact), a bare
    ``not_estimable`` block is ``not_applicable``, and a malformed or
    disagreeing block is ``fail``.  The gate is read-only and never changes a
    p-value, the Holm family or a frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'length_strata_audit', 'status': 'ok', 'path': p,
           'n_comparisons': 0, 'n_replayed': 0, 'n_not_replayable': 0,
           'n_not_estimable': 0, 'n_blocks_seen': 0, 'n_ok': 0, 'comparisons': {}, 'problems': [], 'warnings': [],
           'note': ('rebuilds the length post-stratification block from the '
                    'stored per-unit token counts, stratum membership and '
                    'per-seed summed NLLs; a missing block is not a pass and '
                    'a disagreement is a fail; no frozen p/Holm/verdict is '
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
        ls = dsm.get('length_strata') if isinstance(dsm, dict) else None
        if ls is None:
            out['n_not_replayable'] += 1
            out['warnings'].append(
                f'{name}: no length_strata block (legacy artifact); the '
                'composition-sensitivity claim is not replayable')
            continue
        row = {'status': 'fail', 'problems': [], 'n_units': None,
               'n_strata': None, 'n_seeds': None}
        out['comparisons'][str(name)] = row
        out['n_blocks_seen'] += 1
        if not isinstance(ls, dict):
            row['problems'].append(
                f'length_strata is a {type(ls).__name__}, not an object')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if any(f in ls for f in confirm_fields):
            row['problems'].append(
                'length_strata carries confirmatory field(s) '
                f'{sorted(set(ls) & set(confirm_fields))}')
        if ls.get('kind') != 'length_stratification':
            row['problems'].append(
                f"length_strata.kind is {ls.get('kind')!r}, not "
                "'length_stratification'")
        status = ls.get('status')
        if status not in L.LENGTH_STRATA_STATUSES:
            row['problems'].append(
                f'length_strata.status {status!r} is not a declared status')
        if status == 'not_estimable':
            if not row['problems']:
                row['status'] = 'not_applicable'
                out['n_not_replayable'] += 1
                out['n_not_estimable'] += 1
                out['warnings'].append(
                    f'{name}: length_strata is not_estimable '
                    f'({ls.get("reason") or "no reason recorded"}); nothing '
                    'to replay')
                continue
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if status != 'ok':
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        policy = ls.get('policy')
        if not isinstance(policy, dict):
            row['problems'].append('length_strata.policy is not an object')
        units = ls.get('units')
        if not isinstance(units, dict):
            row['problems'].append('length_strata.units is not an object')
        strata = ls.get('strata')
        if not isinstance(strata, list) or not strata:
            row['problems'].append('length_strata.strata is missing or empty')
        per_seed = ls.get('per_seed')
        if not isinstance(per_seed, list) or not per_seed:
            row['problems'].append('length_strata.per_seed is missing or empty')
        seeds = ls.get('seeds')
        if not isinstance(seeds, list) or not seeds:
            row['problems'].append('length_strata.seeds is missing or empty')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        # --- structural checks ---------------------------------------------
        n_units = _lsa_int(ls.get('n_units'))
        n_strata = _lsa_int(ls.get('n_strata'))
        n_seeds = _lsa_int(ls.get('n_seeds'))
        if n_units is None or n_units < 2:
            row['problems'].append('n_units is not an integer >= 2')
        if n_strata is None or n_strata < 2 or n_strata != len(strata):
            row['problems'].append(
                f'n_strata={n_strata!r} does not match the {len(strata)} strata')
        if n_seeds is None or n_seeds < 2 or n_seeds != len(per_seed):
            row['problems'].append(
                f'n_seeds={n_seeds!r} does not match the {len(per_seed)} '
                'per-seed rows')
        if isinstance(seeds, list):
            norm = [_lsa_int(s) for s in seeds]
            if any(s is None for s in norm):
                row['problems'].append('seeds contains a non-integer label')
            elif len(set(norm)) != len(norm):
                row['problems'].append('seeds contains duplicate labels')
            if len(norm) != len(per_seed):
                row['problems'].append('seeds length differs from per_seed')
            elif any(_lsa_int(r.get('seed')) != s for r, s in zip(per_seed, norm)):
                row['problems'].append('per_seed.seed does not match seeds order')
        token_counts = units.get('token_counts') if isinstance(units, dict) else None
        if not isinstance(token_counts, list) or len(token_counts) != n_units:
            row['problems'].append(
                'units.token_counts is missing or not aligned with n_units')
            token_counts = None
        else:
            tc = []
            for x in token_counts:
                if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                    row['problems'].append(
                        f'units.token_counts contains {x!r}, not a positive integer')
                    tc = []
                    break
                tc.append(int(x))
            token_counts = tc if tc else None
        if isinstance(units, dict):
            total = _lsa_int(units.get('n_tokens'))
            if total is None or token_counts is None or total != sum(token_counts):
                row['problems'].append(
                    'units.n_tokens does not equal the sum of token_counts')
        # Per-seed totals and stratum sums must partition each other.
        if token_counts is not None:
            seen = []
            total_tokens = float(sum(token_counts))
            for gi, s in enumerate(strata):
                if not isinstance(s, dict):
                    row['problems'].append(f'strata[{gi}] is not an object')
                    continue
                idx = s.get('unit_indices')
                if not isinstance(idx, list) or not idx:
                    row['problems'].append(f'strata[{gi}].unit_indices is missing')
                    continue
                if any(_lsa_int(i) is None or int(i) < 0 or int(i) >= n_units for i in idx):
                    row['problems'].append(
                        f'strata[{gi}].unit_indices contains an out-of-range index')
                    continue
                idx = [int(i) for i in idx]
                if len(set(idx)) != len(idx):
                    row['problems'].append(
                        f'strata[{gi}].unit_indices contains duplicates')
                seen.extend(idx)
                want_units = len(idx)
                want_tokens = sum(token_counts[i] for i in idx)
                if _lsa_int(s.get('n_units')) != want_units:
                    row['problems'].append(
                        f'strata[{gi}].n_units does not equal its index count')
                if _lsa_int(s.get('n_tokens')) != want_tokens:
                    row['problems'].append(
                        f'strata[{gi}].n_tokens does not equal its token sum')
                want_share = want_tokens / total_tokens if total_tokens else None
                if not _lsa_close(s.get('token_share'), want_share):
                    row['problems'].append(
                        f'strata[{gi}].token_share does not equal its token share')
                a_sum = s.get('nll_a_sum')
                b_sum = s.get('nll_b_sum')
                lr = s.get('per_seed_log_ratio')
                dp = s.get('per_seed_delta_ppl')
                if not isinstance(a_sum, list) or len(a_sum) != n_seeds:
                    row['problems'].append(f'strata[{gi}].nll_a_sum is misaligned')
                    continue
                if not isinstance(b_sum, list) or len(b_sum) != n_seeds:
                    row['problems'].append(f'strata[{gi}].nll_b_sum is misaligned')
                    continue
                if not isinstance(lr, list) or len(lr) != n_seeds:
                    row['problems'].append(f'strata[{gi}].per_seed_log_ratio is misaligned')
                    continue
                if not isinstance(dp, list) or len(dp) != n_seeds:
                    row['problems'].append(f'strata[{gi}].per_seed_delta_ppl is misaligned')
                    continue
                for si in range(n_seeds):
                    a_v = _lsa_int(a_sum[si]) if isinstance(a_sum[si], int) else a_sum[si]
                    try:
                        a_f = float(a_v)
                        b_f = float(b_sum[si])
                    except (TypeError, ValueError, OverflowError):
                        row['problems'].append(
                            f'strata[{gi}].nll sums contain a non-numeric value')
                        continue
                    if not (math.isfinite(a_f) and math.isfinite(b_f)):
                        row['problems'].append(
                            f'strata[{gi}].nll sums contain a non-finite value')
                        continue
                    want_lr = (a_f - b_f) / want_tokens
                    if not _lsa_close(lr[si], want_lr):
                        row['problems'].append(
                            f'strata[{gi}].per_seed_log_ratio[{si}] does not '
                            'replay from the stored NLL sums')
                    pa = L._safe_exp_ppl(a_f / want_tokens)
                    pb = L._safe_exp_ppl(b_f / want_tokens)
                    want_dp = None
                    if (pa is not None and pb is not None
                            and math.isfinite(pa) and math.isfinite(pb)
                            and math.isfinite(pa - pb)):
                        want_dp = float(pa - pb)
                    if not _lsa_close(dp[si], want_dp):
                        row['problems'].append(
                            f'strata[{gi}].per_seed_delta_ppl[{si}] does not '
                            'replay from the stored NLL sums')
            if len(seen) != n_units or sorted(seen) != list(range(n_units)):
                row['problems'].append(
                    'stratum unit_indices do not partition all evaluation units')
        # Per-seed totals must equal the sum of the stratum sufficient stats.
        if isinstance(strata, list) and all(
                isinstance(s, dict) and isinstance(s.get('nll_a_sum'), list)
                and isinstance(s.get('nll_b_sum'), list) for s in strata):
            for si, rec in enumerate(per_seed):
                if not isinstance(rec, dict):
                    row['problems'].append(f'per_seed[{si}] is not an object')
                    continue
                try:
                    a_tot = math.fsum(float(s['nll_a_sum'][si]) for s in strata)
                    b_tot = math.fsum(float(s['nll_b_sum'][si]) for s in strata)
                except (TypeError, ValueError, OverflowError, IndexError):
                    row['problems'].append(
                        f'per_seed[{si}] totals cannot be rebuilt')
                    continue
                if not _lsa_close(rec.get('nll_a_total'), a_tot):
                    row['problems'].append(
                        f'per_seed[{si}].nll_a_total != sum of stratum NLLs')
                if not _lsa_close(rec.get('nll_b_total'), b_tot):
                    row['problems'].append(
                        f'per_seed[{si}].nll_b_total != sum of stratum NLLs')
                if not _lsa_close(rec.get('log_ratio'), (a_tot - b_tot) / total_tokens):
                    row['problems'].append(
                        f'per_seed[{si}].log_ratio != token-pooled total')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        # --- full replay at the stratum level ------------------------------
        try:
            a_mat = [[float(s['nll_a_sum'][si]) for s in strata]
                     for si in range(n_seeds)]
            b_mat = [[float(s['nll_b_sum'][si]) for s in strata]
                     for si in range(n_seeds)]
            toks = [int(s['n_tokens']) for s in strata]
            ref = policy.get('reference_weights') if policy.get('reference') == 'custom' else policy.get('reference')
            replay = L.length_stratified_effect(
                a_mat, b_mat, toks,
                n_strata=len(strata), min_units_per_stratum=1,
                conf=policy.get('conf'), reference=ref,
                seeds=seeds)
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
        # Derived blocks: compare every field except the identities that
        # legitimately change when units are aggregated into strata.
        for key in ('overall', 'standardized', 'heterogeneity',
                    'sign_consistency', 'concentration'):
            _lsa_same(ls.get(key), replay.get(key), key, row['problems'])
        for gi, (stored, want) in enumerate(zip(strata, replay['strata'])):
            for key in ('n_tokens', 'log_ratio', 'ratio', 'log_ratio_ci95',
                        'delta_ppl', 'delta_ppl_ci95',
                        'n_seeds_with_delta_ppl'):
                _lsa_same(stored.get(key), want.get(key),
                          f'strata[{gi}].{key}', row['problems'])
        for key in ('version', 'n_strata', 'conf', 'reference',
                    'reference_weights', 'stratification', 'q_method'):
            _lsa_same((policy or {}).get(key), (replay.get('policy') or {}).get(key),
                      f'policy.{key}', row['problems'])
        # Cross-check the seed set against the comparison's own per-seed
        # vector when the latter exists; the two blocks describe one panel.
        comp_pairs = comp.get('per_seed')
        if isinstance(comp_pairs, list) and comp_pairs:
            comp_seeds = []
            ok_pairs = True
            for rec in comp_pairs:
                if not isinstance(rec, dict):
                    ok_pairs = False
                    break
                s = _lsa_int(rec.get('seed'))
                if s is None:
                    ok_pairs = False
                    break
                comp_seeds.append(s)
            if ok_pairs and set(comp_seeds) != set(seeds):
                row['problems'].append(
                    'length_strata.seeds does not match the comparison per_seed seeds')
        dsm_units = _lsa_int(dsm.get('pooled_ci_n_units')) if isinstance(dsm, dict) else None
        if dsm_units is not None and dsm_units != n_units:
            row['problems'].append(
                'pooled_ci_n_units does not match length_strata.n_units')
        row['n_units'] = n_units
        row['n_strata'] = n_strata
        row['n_seeds'] = n_seeds
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
            out['reason'] = 'every stored length_strata block is not_estimable'
        else:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'no length_strata block could be replayed'
    elif out['n_ok'] == 0:
        out['status'] = 'not_applicable'
        out['reason'] = 'every stored length_strata block is not_estimable'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
