def _poa_close(a, b, *, rel=1e-9, abs_=1e-9):
    """Finite-number equality with a declared tolerance."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(fa) and math.isfinite(fb)
            and abs(fa - fb) <= abs_ + rel * max(abs(fa), abs(fb)))


def _poa_compare(stored, replay, path, problems,
                *, skip=('upstream', 'sufficient_statistics')):
    """Recursively compare a stored readout with its replay.

    The replay is rebuilt from the stored influence series and declared policy
    inputs, so every derived number must match.  Input echoes are skipped;
    everything a reader would quote is compared.
    """
    if isinstance(replay, dict):
        if not isinstance(stored, dict):
            problems.append(f'{path}: stored value is not an object')
            return
        for key, value in replay.items():
            if key in skip:
                continue
            if key not in stored:
                problems.append(f'{path}.{key}: missing from the stored block')
                continue
            _poa_compare(stored[key], value, f'{path}.{key}', problems,
                         skip=skip)
        return
    if isinstance(replay, list):
        if not isinstance(stored, list) or len(stored) != len(replay):
            problems.append(f'{path}: stored list is missing or has a '
                            'different length')
            return
        for i, (a, b) in enumerate(zip(stored, replay)):
            _poa_compare(a, b, f'{path}[{i}]', problems, skip=skip)
        return
    if isinstance(replay, bool) or replay is None:
        if stored != replay:
            problems.append(f'{path}: stored {stored!r} != replayed {replay!r}')
        return
    if isinstance(replay, (int, float)):
        if isinstance(stored, bool) or not _poa_close(stored, replay):
            problems.append(f'{path}: stored {stored!r} != replayed {replay!r}')
        return
    if stored != replay:
        problems.append(f'{path}: stored {stored!r} != replayed {replay!r}')


def audit_pooled_order(path=None):
    """Replay the token-pooled order-dependence / effective-n readout.

    ``finaltest_stats`` stores, next to ``doc_sampling.pooled_order_inference``,
    the two centred cluster-influence series and every policy input.  This
    gate rebuilds the block with :func:`exp_lib.pooled_order_inference_from_influence`
    and compares the derived intervals, design effects, effective n and
    permutation p.  A missing block is ``not_verifiable_locally`` (legacy
    artifact), a bare ``not_estimable`` block is ``not_applicable``, and a
    malformed or disagreeing block is ``fail``.  The gate never changes a
    frozen p/Holm/verdict and never reads the test split.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'pooled_order_audit', 'status': 'ok', 'path': p,
           'n_comparisons': 0, 'n_replayed': 0, 'n_not_replayable': 0,
           'n_not_estimable': 0, 'n_blocks_seen': 0, 'n_ok': 0,
           'comparisons': {}, 'problems': [], 'warnings': [],
           'note': ('rebuilds the token-pooled order-dependence readout from '
                    'the stored influence series and policy inputs; a missing '
                    'block is not a pass and a disagreement is a fail; no '
                    'frozen p/Holm/verdict is changed')}
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
    confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision',
                      'p_exact_signflip')
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            out['problems'].append(f'{name}: comparison is not a JSON object')
            continue
        dsm = comp.get('doc_sampling')
        block = dsm.get('pooled_order_inference') if isinstance(dsm, dict) else None
        if block is None:
            if isinstance(dsm, dict) and dsm.get('pooled_delta_ppl_ci95') is not None:
                out['n_not_replayable'] += 1
                out['warnings'].append(
                    f'{name}: token-pooled cluster CI exists but no '
                    'doc_sampling.pooled_order_inference block (legacy '
                    'artifact); the pooled order diagnosis is not replayable')
            continue
        row = {'status': 'fail', 'problems': [], 'n_units': None}
        out['comparisons'][str(name)] = row
        out['n_blocks_seen'] += 1
        if not isinstance(block, dict):
            row['problems'].append(
                f'pooled_order_inference is a {type(block).__name__}, not an '
                'object')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if any(f in block for f in confirm_fields):
            row['problems'].append(
                'pooled_order_inference carries confirmatory field(s) '
                f'{sorted(set(block) & set(confirm_fields))}')
        if block.get('kind') != 'pooled_order_inference':
            row['problems'].append(
                f"pooled_order_inference.kind is {block.get('kind')!r}, not "
                "'pooled_order_inference'")
        status = block.get('status')
        if status not in L.POOLED_ORDER_STATUSES:
            row['problems'].append(
                f'pooled_order_inference.status {status!r} is not a declared '
                'status')
        if status == 'not_estimable':
            if not row['problems']:
                row['status'] = 'not_applicable'
                out['n_not_replayable'] += 1
                out['n_not_estimable'] += 1
                out['warnings'].append(
                    f'{name}: pooled_order_inference is not_estimable '
                    f'({block.get("reason") or "no reason recorded"}); nothing '
                    'to replay')
                continue
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if status != 'ok' or row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        # The surrounding doc_sampling block names the same evaluation
        # units.  A hand edit that changes its unit count while leaving the
        # influence series untouched must not replay as a pass.
        _block_units = _oda_int(block.get('n_units'))
        if isinstance(dsm, dict) and _block_units is not None:
            for _key in ('pooled_ci_n_units', 'n_docs'):
                _ds_units = _oda_int(dsm.get(_key))
                if _ds_units is not None and _ds_units != _block_units:
                    row['problems'].append(
                        f'doc_sampling.{_key} does not match '
                        'pooled_order_inference.n_units')
        stats = block.get('sufficient_statistics')
        if not isinstance(stats, dict):
            row['problems'].append(
                'pooled_order_inference.sufficient_statistics is not an object')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        required = ('ppl_influence', 'log_ratio_influence', 'delta_ppl',
                    'delta_log_ratio', 'n_batches', 'block_lengths', 'n_boot',
                    'n_perm', 'conf', 'alpha', 'seed')
        missing = [k for k in required if k not in stats]
        if missing:
            row['problems'].append(
                f'sufficient_statistics missing {missing}')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        try:
            replay = L.pooled_order_inference_from_influence(
                stats['ppl_influence'], stats['log_ratio_influence'],
                delta_ppl=stats['delta_ppl'],
                delta_log_ratio=stats['delta_log_ratio'],
                unit_ids=stats.get('unit_ids'),
                seeds=stats.get('seeds'),
                n_seeds=stats.get('n_seeds', block.get('n_seeds')),
                block_lengths=stats['block_lengths'],
                n_batches=stats['n_batches'],
                n_boot=stats['n_boot'], n_perm=stats['n_perm'],
                conf=stats['conf'], alpha=stats['alpha'],
                seed=stats['seed'],
                design_effect_tolerance=stats.get('design_effect_tolerance'),
                material_ci_inflation=stats.get('material_ci_inflation'))
        except Exception as exc:
            row['problems'].append(
                f'replay raised {type(exc).__name__}: {exc}')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        out['n_replayed'] += 1
        _poa_compare(block, replay, f'{name}.pooled_order_inference',
                     row['problems'])
        row['n_units'] = replay.get('n_units')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
        else:
            row['status'] = 'ok'
            out['n_ok'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        if (out['n_not_estimable'] > 0
                and out['n_not_estimable'] == out['n_blocks_seen']
                and out['n_blocks_seen'] == out['n_comparisons']):
            out['status'] = 'not_applicable'
            out['reason'] = ('every stored pooled_order_inference block is '
                             'not_estimable')
        else:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = ('no replayable token-pooled order-dependence '
                             'block in the artifact')
    elif out['n_ok'] == 0:
        out['status'] = 'not_applicable'
        out['reason'] = 'every stored pooled_order_inference block is not_estimable'
    elif out['n_not_replayable']:
        out['status'] = 'warnings'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
