def _uo_int(value):
    """Exact integer from one stored count/label, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _uo_problems_from_attestation(att, ids, *, unit_kind=None):
    """Structural/order checks for one stored attestation and id sequence."""
    try:
        return list(_import_exp_lib().unit_order_problems(
            att, ids, unit_kind=unit_kind))
    except Exception as exc:
        return [f'{type(exc).__name__}: {exc}']


def _uo_check_cell(cell):
    """Validate one evaluation cell's ``unit_order`` block.

    Returns ``(status, unit_kind, problems, legacy)`` where status is
    ``ok``/``fail``/``not_verifiable_locally``/``not_applicable``.
    """
    if not isinstance(cell, dict):
        return 'fail', None, ['cell is not a JSON object'], False
    if cell.get('article_granularity') == 'article':
        kind, ids_key, tok_key = ('article', 'article_ids',
                                  'n_tokens_per_article')
        expected = cell.get('n_articles')
    elif cell.get('doc_granularity') == 'document':
        kind, ids_key, tok_key = ('document', 'doc_ids',
                                  'n_tokens_per_doc')
        expected = cell.get('n_docs')
    else:
        return 'not_applicable', None, [], False
    ids = cell.get(ids_key)
    if not isinstance(ids, list) or not ids:
        return 'fail', kind, [f'{ids_key} is missing or empty'], False
    if len(set(str(x) for x in ids)) != len(ids):
        return 'fail', kind, [f'{ids_key} contains duplicate ids'], False
    att = cell.get('unit_order')
    if att is None:
        return 'not_verifiable_locally', kind, [], True
    if not isinstance(att, dict):
        return 'fail', kind, ['unit_order is not an object'], False
    perm = att.get('permutation')
    problems = []
    ordered_ids = None
    if not isinstance(perm, list) or len(perm) != len(ids):
        problems.append('unit_order.permutation is not aligned with the ids')
    elif any(_uo_int(x) is None for x in perm) or sorted(
            _uo_int(x) for x in perm) != list(range(len(ids))):
        problems.append('unit_order.permutation is not a permutation')
    else:
        ordered_ids = [str(ids[int(i)]) for i in perm]
        problems.extend(_uo_problems_from_attestation(
            att, ordered_ids, unit_kind=kind))
    n_units = _uo_int(att.get('n_units'))
    if n_units is not None and n_units != len(ids):
        problems.append(f'unit_order.n_units={n_units} != {len(ids)} ids')
    if expected is not None and _uo_int(expected) != len(ids):
        problems.append(f'{kind} count {expected!r} != {len(ids)} ids')
    toks = cell.get(tok_key)
    if isinstance(toks, list) and toks:
        try:
            total = int(sum(int(x) for x in toks))
        except (TypeError, ValueError, OverflowError):
            total = None
        if total is not None:
            if att.get('n_tokens') is not None and _uo_int(
                    att.get('n_tokens')) != total:
                problems.append(
                    f'unit_order.n_tokens={att.get("n_tokens")!r} != the '
                    f'cell token total {total}')
            n_tok = _uo_int(cell.get('n_tok'))
            if n_tok is not None and n_tok != total:
                problems.append(f'cell n_tok={n_tok} != unit token total '
                                f'{total}')
            pos = att.get('first_positions')
            if isinstance(pos, list):
                if any(_uo_int(x) is None or _uo_int(x) >= total
                       for x in pos):
                    problems.append(
                        'unit_order.first_positions contains a position '
                        'outside the scored token stream')
    return ('fail' if problems else 'ok'), kind, problems, False


def _uo_attestation_matches(uo, readout, *, path, problems):
    """Cross-check a readout's compact unit_order and its sufficient stats."""
    if not isinstance(uo, dict) or uo.get('status') != 'ok':
        problems.append(f'{path}: an order readout exists but the cell-level '
                        'unit_order attestation is not ok')
        return
    ro = readout.get('unit_order') if isinstance(readout, dict) else None
    if not isinstance(ro, dict):
        problems.append(f'{path}: readout.unit_order is missing')
        return
    if ro.get('status') != 'verified':
        problems.append(f'{path}: readout.unit_order.status is '
                        f'{ro.get("status")!r}, not verified')
    for key in ('sha256', 'basis', 'unit_kind'):
        if ro.get(key) != uo.get(key):
            problems.append(f'{path}: readout.unit_order.{key} disagrees with '
                            'the cell attestation')
    stats = readout.get('sufficient_statistics')
    full = stats.get('unit_order_attestation') if isinstance(stats, dict) else None
    if full is None:
        problems.append(f'{path}: sufficient_statistics.unit_order_attestation '
                        'is missing; a verified readout must be replayable')
        return
    if not isinstance(full, dict):
        problems.append(f'{path}: unit_order_attestation is not an object')
        return
    for key in ('sha256', 'basis', 'unit_kind', 'n_units', 'n_tokens',
                'permutation', 'first_positions'):
        if key in uo and full.get(key) != uo.get(key):
            problems.append(f'{path}: stored unit_order_attestation.{key} '
                            'disagrees with the cell attestation')
    perm = full.get('permutation')
    if not isinstance(perm, list):
        problems.append(f'{path}: stored unit_order_attestation.permutation '
                        'is missing')
    elif any(_uo_int(x) is None for x in perm) or sorted(
            _uo_int(x) for x in perm) != list(range(len(perm))):
        problems.append(f'{path}: stored unit_order_attestation.permutation '
                        'is not a permutation')
    pos = full.get('first_positions')
    if not isinstance(pos, list):
        problems.append(f'{path}: stored unit_order_attestation.'
                        'first_positions is missing')
    else:
        norm = [_uo_int(x) for x in pos]
        if any(x is None for x in norm) or any(
                b <= a for a, b in zip(norm, norm[1:])):
            problems.append(f'{path}: stored unit_order_attestation.'
                            'first_positions is not strictly increasing')


def _uo_audit_stats(data):
    """Audit the comparison-level ``doc_sampling`` order blocks."""
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    out = {'kind': 'unit_order_audit', 'artifact': 'stats', 'status': 'ok',
           'n_comparisons': 0, 'n_ok': 0, 'n_not_verifiable': 0,
           'n_legacy': 0, 'comparisons': {}, 'problems': [], 'warnings': []}
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        return out
    out['n_comparisons'] = len(comps)
    confirm_fields = ('confirmed', 'p_holm', 'verified', 'verdict', 'decision',
                      'p_exact_signflip')
    for name, comp in comps.items():
        row = {'status': 'ok', 'problems': [], 'warnings': []}
        out['comparisons'][str(name)] = row
        if not isinstance(comp, dict):
            row['problems'].append('comparison is not a JSON object')
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
            continue
        dsm = comp.get('doc_sampling')
        if not isinstance(dsm, dict):
            row['warnings'].append('no doc_sampling block')
            out['n_not_verifiable'] += 1
            continue
        uo = dsm.get('unit_order')
        readouts = []
        for key in ('order_inference', 'pooled_order_inference'):
            block = dsm.get(key)
            if block is None:
                continue
            if not isinstance(block, dict):
                row['problems'].append(f'{key} is not an object')
                continue
            readouts.append((key, block))
            if any(f in block for f in confirm_fields):
                row['problems'].append(
                    f'{key} carries confirmatory field(s) '
                    f'{sorted(set(block) & set(confirm_fields))}')
        if uo is None:
            if readouts:
                row['problems'].append(
                    'order readout exists without a doc_sampling.unit_order '
                    'attestation; a sorted-id sequence cannot be read as '
                    'corpus order')
            else:
                row['warnings'].append(
                    'legacy artifact: no doc_sampling.unit_order attestation')
                out['n_legacy'] += 1
        elif not isinstance(uo, dict):
            row['problems'].append('doc_sampling.unit_order is not an object')
        elif uo.get('status') == 'fail':
            row['problems'].append(
                'doc_sampling.unit_order.status=fail: '
                f'{uo.get("reason") or uo.get("problems") or "no reason recorded"}')
        elif uo.get('status') not in ('ok', 'not_applicable',
                                       'not_verifiable_locally'):
            row['problems'].append(
                f'doc_sampling.unit_order.status={uo.get("status")!r} is not '
                'a declared status')
        elif uo.get('status') == 'not_applicable':
            # Row-level panels have no id permutation: ppl_per_seq is already
            # in evaluation-panel order.  Such a readout must be unverified
            # (caller-asserted row order); a 'verified' claim would require a
            # document/article attestation that the cell does not carry.
            for key, block in readouts:
                ro = block.get('unit_order') or {}
                if ro.get('status') == 'verified':
                    row['problems'].append(
                        f'{key} claims a verified order but the cell is '
                        'row-level (no id permutation to verify)')
            if readouts:
                row['warnings'].append(
                    'row-level order readout: evaluation-panel order, no '
                    'document/article attestation')
                out['n_ok'] += 1
            else:
                row['warnings'].append(
                    'row-level cell: no document/article order attestation')
                out['n_not_verifiable'] += 1
        elif uo.get('status') != 'ok':
            if readouts:
                row['problems'].append(
                    f'doc_sampling.unit_order.status={uo.get("status")!r} but '
                    'an order readout is present')
            else:
                row['warnings'].append(
                    f'doc_sampling.unit_order.status={uo.get("status")!r}: '
                    f'{uo.get("reason") or "no reason recorded"}')
                out['n_not_verifiable'] += 1
        else:
            for key, block in readouts:
                _uo_attestation_matches(
                    uo, block, path=f'{name}.{key}',
                    problems=row['problems'])
        if row['problems']:
            row['status'] = 'fail'
            out['problems'].extend(f'{name}: {x}' for x in row['problems'])
        elif row['warnings']:
            row['status'] = 'warnings'
            out['warnings'].extend(f'{name}: {x}' for x in row['warnings'])
        else:
            out['n_ok'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0:
        out['status'] = ('not_verifiable_locally' if out['n_legacy']
                         else 'not_verifiable_locally')
        out['reason'] = ('no comparison carries a verifiable unit-order '
                         'attestation')
    elif out['warnings']:
        out['status'] = 'warnings'
    return out


def _uo_audit_records(data):
    """Audit ``by_len`` cells in a results summary or a single cell."""
    runs = data.get('runs') if isinstance(data.get('runs'), list) else None
    if runs is None and isinstance(data, dict):
        # A summary mapping is the other historical shape; a bare cell is
        # also accepted for direct checks.
        if any(k in data for k in ('doc_ids', 'article_ids', 'unit_order')):
            runs = [data]
        else:
            runs = [v for v in data.values() if isinstance(v, dict)]
    out = {'kind': 'unit_order_audit', 'artifact': 'records', 'status': 'ok',
           'n_cells': 0, 'n_ok': 0, 'n_legacy': 0, 'n_not_applicable': 0,
           'n_fail': 0, 'cells': {}, 'problems': [], 'warnings': []}
    if not runs:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the artifact carries no run records'
        return out
    for i, rec in enumerate(runs):
        if not isinstance(rec, dict):
            out['problems'].append(f'record[{i}] is not an object')
            continue
        by_len = rec.get('by_len')
        cells = (by_len.items() if isinstance(by_len, dict)
                 else [(None, rec)])
        key_base = rec.get('key') or rec.get('variant') or f'record[{i}]'
        for length, cell in cells:
            if not isinstance(cell, dict):
                continue
            status, kind, problems, legacy = _uo_check_cell(cell)
            if status == 'not_applicable':
                out['n_not_applicable'] += 1
                continue
            label = f'{key_base}@L{length}' if length is not None else str(key_base)
            out['n_cells'] += 1
            if status == 'fail':
                out['n_fail'] += 1
                out['cells'][label] = {'status': 'fail', 'problems': problems}
                out['problems'].extend(f'{label}: {x}' for x in problems)
            elif legacy:
                out['n_legacy'] += 1
                out['cells'][label] = {'status': 'not_verifiable_locally'}
                out['warnings'].append(
                    f'{label}: legacy cell has no unit_order attestation')
            else:
                out['n_ok'] += 1
                out['cells'][label] = {'status': 'ok', 'unit_kind': kind}
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no evaluated cell carries a unit_order attestation '
                         '(legacy artifact)')
    elif out['n_legacy'] and out['n_ok']:
        out['status'] = 'fail'
        out['problems'].append(
            f'{out["n_legacy"]} of {out["n_cells"]} cells are legacy while '
            'others carry unit_order; the evaluation order is inconsistent '
            'across the panel')
    elif out['n_legacy']:
        out['status'] = 'warnings'
        out['warnings'].append(
            f'{out["n_legacy"]} of {out["n_cells"]} cells are legacy and '
            'cannot be order-audited')
    return out


def audit_unit_order(path=None):
    """Audit first-occurrence unit order and its use by order readouts.

    The evaluation aggregator stores per-unit vectors in the canonical
    sorted-id layout and records corpus/first-occurrence order separately in
    ``unit_order``.  This gate checks that:

    * every evaluated document/article cell carries a well-formed attestation;
    * ``permutation`` reconstructs the analyzed order from the stored ids;
    * ``sha256`` recomputes over that ordered sequence;
    * ``first_positions`` is strictly increasing and inside the scored stream;
    * ``n_tokens`` equals the cell's token total; and
    * every ``order_inference``/``pooled_order_inference`` block used the
      attested order, recording ``verified`` plus a replayable full
      attestation in its sufficient statistics.

    A legacy cell/readout without the attestation is
    ``not_verifiable_locally``; a readout without it, or any digest/order
    disagreement, is ``fail``.  The gate never reads the test split and never
    changes a p-value, Holm family or frozen verdict.
    """
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    if os.path.isdir(p):
        candidates = ('test_stats.json', 'finaltest_stats.json',
                      'summary.json', 'analysis.json')
        found = None
        for name in candidates:
            candidate = os.path.join(p, name)
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            out = {'kind': 'unit_order_audit',
                   'status': 'not_verifiable_locally', 'path': os.path.abspath(p),
                   'reason': ('the directory carries no test_stats.json / '
                              'summary.json artifact'),
                   'problems': [], 'warnings': []}
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        p = found
    if not os.path.exists(p):
        out = {'kind': 'unit_order_audit', 'status': 'not_verifiable_locally',
               'path': p, 'reason': 'no artifact on disk',
               'problems': [], 'warnings': []}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if not isinstance(data, dict):
        out = {'kind': 'unit_order_audit', 'status': 'fail', 'path': p,
               'problems': [f'{p} exists but is not a readable JSON object'],
               'warnings': []}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    if isinstance(ft, dict) and isinstance(ft.get('comparisons'), dict):
        out = _uo_audit_stats(data)
    else:
        out = _uo_audit_records(data)
    out['path'] = p
    out.setdefault('problems', [])
    out.setdefault('warnings', [])
    out['note'] = ('checks the first-occurrence order attestation and its use '
                   'by order-dependent readouts; a sorted/shuffled vector is '
                   'not corpus order; no frozen p/Holm/verdict is changed')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
