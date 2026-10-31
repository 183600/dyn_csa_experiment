def _oda_int(value):
    """Exact integer from one stored count/label, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def audit_order_dependence(path=None):
    """Replay the order-dependence / effective-n readout of every comparison.

    ``finaltest_stats`` stores the ordered per-unit effect vector and every
    policy input (block lengths, batch counts, bootstrap/permutation draws,
    confidence level, alpha, seed and the two decision thresholds) next to the
    derived ``doc_sampling.order_inference`` block.  This gate rebuilds the
    whole block with the same :func:`exp_lib.ordered_unit_inference`
    implementation and compares every derived number.  A hand edit that
    widens the interval, hides a detected design effect, or changes the
    effective n without moving the sufficient statistics is therefore caught.

    A missing block is ``not_verifiable_locally`` (a legacy artifact), a bare
    ``not_estimable`` block is ``not_applicable``, and a malformed or
    disagreeing block is ``fail``.  The gate is read-only and never changes a
    p-value, the Holm family or a frozen verdict.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'order_dependence_audit', 'status': 'ok', 'path': p,
           'n_comparisons': 0, 'n_replayed': 0, 'n_not_replayable': 0,
           'n_not_estimable': 0, 'n_blocks_seen': 0, 'n_ok': 0,
           'comparisons': {}, 'problems': [], 'warnings': [],
           'note': ('rebuilds the ordered-unit inference from the stored '
                    'ordered values and policy inputs; a missing block is not '
                    'a pass and a disagreement is a fail; no frozen '
                    'p/Holm/verdict is changed')}
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
        oi = dsm.get('order_inference') if isinstance(dsm, dict) else None
        if oi is None:
            out['n_not_replayable'] += 1
            out['warnings'].append(
                f'{name}: no doc_sampling.order_inference block (legacy '
                'artifact); the ordered-unit diagnosis is not replayable')
            continue
        row = {'status': 'fail', 'problems': [], 'n_units': None}
        out['comparisons'][str(name)] = row
        out['n_blocks_seen'] += 1
        if not isinstance(oi, dict):
            row['problems'].append(
                f'order_inference is a {type(oi).__name__}, not an object')
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        if any(f in oi for f in confirm_fields):
            row['problems'].append(
                'order_inference carries confirmatory field(s) '
                f'{sorted(set(oi) & set(confirm_fields))}')
        if oi.get('kind') != 'ordered_unit_inference':
            row['problems'].append(
                f"order_inference.kind is {oi.get('kind')!r}, not "
                "'ordered_unit_inference'")
        status = oi.get('status')
        if status not in L.ORDER_DEPENDENCE_STATUSES:
            row['problems'].append(
                f'order_inference.status {status!r} is not a declared status')
        if status == 'not_estimable':
            if not row['problems']:
                row['status'] = 'not_applicable'
                out['n_not_replayable'] += 1
                out['n_not_estimable'] += 1
                out['warnings'].append(
                    f'{name}: order_inference is not_estimable '
                    f'({oi.get("reason") or "no reason recorded"}); nothing '
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
        stats = oi.get('sufficient_statistics')
        if not isinstance(stats, dict):
            row['problems'].append(
                'order_inference.sufficient_statistics is not an object')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        values = stats.get('values')
        ids = stats.get('unit_ids')
        blocks = stats.get('block_lengths')
        batches = stats.get('n_batches')
        n_units = _oda_int(oi.get('n_units'))
        if not isinstance(values, list) or not values:
            row['problems'].append(
                'sufficient_statistics.values is missing or empty')
        if not isinstance(blocks, list) or not blocks:
            row['problems'].append(
                'sufficient_statistics.block_lengths is missing or empty')
        if not isinstance(batches, list) or not batches:
            row['problems'].append(
                'sufficient_statistics.n_batches is missing or empty')
        if n_units is None or n_units < int(L.ORDER_DEPENDENCE_POLICY['min_units']):
            row['problems'].append(
                'n_units is not an integer at or above the policy minimum')
        if isinstance(values, list):
            for v in values:
                if isinstance(v, (bool, str, bytes, bytearray)):
                    row['problems'].append(
                        'sufficient_statistics.values contains a non-numeric value')
                    break
                if not isinstance(v, (int, float)):
                    row['problems'].append(
                        'sufficient_statistics.values contains a non-numeric value')
                    break
                try:
                    finite = math.isfinite(float(v))
                except (TypeError, ValueError, OverflowError):
                    # An exact integer beyond float64 cannot be converted to a
                    # float; a stored measurement that large is malformed, not
                    # a reason to let the audit raise and abort the report.
                    finite = False
                if not finite:
                    row['problems'].append(
                        'sufficient_statistics.values contains a non-finite value')
                    break
            if n_units is not None and len(values) != n_units:
                row['problems'].append(
                    'sufficient_statistics.values is not aligned with n_units')
        if isinstance(ids, list) and n_units is not None:
            if len(ids) != n_units:
                row['problems'].append('unit_ids are not aligned with n_units')
            elif len(set(str(x) for x in ids)) != len(ids):
                row['problems'].append('unit_ids contain duplicates')
        elif ids is not None:
            row['problems'].append('unit_ids is neither null nor a list')
        if isinstance(blocks, list):
            norm = [_oda_int(b) for b in blocks]
            if any(b is None for b in norm):
                row['problems'].append(
                    'block_lengths contains a non-integer entry')
            elif len(set(norm)) != len(norm):
                row['problems'].append('block_lengths contains duplicates')
            elif n_units is not None and any(not (2 <= b < n_units) for b in norm):
                row['problems'].append(
                    'block_lengths contains a value outside [2, n_units)')
        if isinstance(batches, list):
            norm = [_oda_int(g) for g in batches]
            if any(g is None for g in norm):
                row['problems'].append(
                    'n_batches contains a non-integer entry')
            elif len(set(norm)) != len(norm):
                row['problems'].append('n_batches contains duplicates')
            elif n_units is not None and any(
                    not (2 <= g <= n_units // 2) for g in norm):
                row['problems'].append(
                    'n_batches contains an infeasible count for n_units')
        if isinstance(dsm, dict):
            ds_units = _oda_int(dsm.get('pooled_ci_n_units'))
            if ds_units is not None and n_units is not None and ds_units != n_units:
                row['problems'].append(
                    'pooled_ci_n_units does not match order_inference.n_units')
            ds_docs = _oda_int(dsm.get('n_docs'))
            if ds_docs is not None and n_units is not None and ds_docs != n_units:
                row['problems'].append(
                    'doc_sampling.n_docs does not match order_inference.n_units')
        if row['problems']:
            out['n_not_replayable'] += 1
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        try:
            replay = L.ordered_unit_inference(
                values, unit_ids=ids,
                block_lengths=blocks,
                n_batches=(None if stats.get('n_batches_defaulted') is True
                           else (stats.get('raw_n_batches')
                                 if isinstance(stats.get('raw_n_batches'), list)
                                 else batches)),
                n_boot=stats.get('n_boot'), n_perm=stats.get('n_perm'),
                conf=stats.get('conf'), alpha=stats.get('alpha'),
                seed=stats.get('seed'),
                design_effect_tolerance=stats.get('design_effect_tolerance'),
                material_ci_inflation=stats.get('material_ci_inflation'))
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
        _lsa_same(oi, replay, 'order_inference', row['problems'])
        row['n_units'] = n_units
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
            out['reason'] = 'every stored order_inference block is not_estimable'
        else:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'no order_inference block could be replayed'
    elif out['n_ok'] == 0:
        out['status'] = 'not_applicable'
        out['reason'] = 'every stored order_inference block is not_estimable'
    else:
        out['status'] = 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
