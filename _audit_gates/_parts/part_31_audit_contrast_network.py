# ---------------------------------------------------------------------------
# 7e) contrast network：多臂对比的可组合性、共同参照与闭环一致性
# ---------------------------------------------------------------------------

_CONTRAST_NETWORK_FORBIDDEN = frozenset({
    'confirmed', 'p_holm', 'p_holm_global', 'p_holm_family',
    'p_holm_component', 'verdict', 'decision', 'significant',
    'significant_global', 'holm',
})


def _cn_gate_forbidden(obj, path='', depth=0):
    """Recursively name forbidden decision keys smuggled into the block.

    Iterative traversal, so an adversarially deep JSON object cannot put a
    ``verdict``/``p_holm`` key beyond the former depth-12 cutoff and still
    pass.  Exceeding the safety depth is itself reported (fail closed).
    """
    found = []
    stack = [(obj, path, int(depth))]
    while stack:
        node, p, d = stack.pop()
        if d > 512:
            found.append(f'{p or "<root>"}.<nesting-depth>')
            continue
        if isinstance(node, dict):
            for k, v in node.items():
                sub = '%s.%s' % (p, k) if p else str(k)
                if str(k) in _CONTRAST_NETWORK_FORBIDDEN:
                    found.append(sub)
                stack.append((v, sub, d + 1))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                stack.append((v, '%s[%d]' % (p, i), d + 1))
    return found


def _cn_gate_default_path():
    cands = []
    for _name in sorted(os.listdir(REPO)):
        if _name.startswith('analysis_'):
            cands.append(os.path.join(REPO, _name, 'methodology_audit.json'))
    for _name in _iter_result_dirs():
        cands.append(os.path.join(REPO, _name, 'methodology_audit.json'))
    existing = sorted({p for p in cands if os.path.exists(p)},
                      key=lambda p: os.path.getmtime(p), reverse=True)
    return existing[0] if existing else None


def _cn_gate_len(value):
    """Length of a JSON collection, or None for a malformed scalar.

    The gate reads arbitrary stored JSON; a hand-edited ``node_conflicts: 1``
    must become a named shape violation, not a TypeError before any verdict
    can be printed.
    """
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value)
    return None


def _cn_gate_shape_problems(block):
    problems = []
    if not isinstance(block, dict):
        return ['contrast_network is not an object']
    for key in ('edges', 'nodes', 'graph_components', 'declared_components',
                'reference', 'closure', 'indirect_contrasts'):
        if key in block and not isinstance(block.get(key), dict):
            problems.append(f'{key} is not an object')
    for key in ('node_conflicts', 'conflicting_edges', 'ranking_reasons',
                'warnings'):
        if key in block and not isinstance(block.get(key), list):
            problems.append(f'{key} is not a list')
    for key in ('n_entries', 'n_edges_declared', 'n_edges', 'n_nodes',
                'n_components'):
        if key in block:
            value = block.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f'{key} is not an integer')
    for key in ('node_conflicts', 'conflicting_edges'):
        value = block.get(key)
        if isinstance(value, list) and any(not isinstance(v, dict) for v in value):
            problems.append(f'{key} contains a non-object row')
    edges = block.get('edges')
    if isinstance(edges, dict):
        for eid, row in edges.items():
            if not isinstance(row, dict):
                problems.append(f'edge {eid!r} is not an object')
            elif not isinstance(row.get('seeds'), list):
                problems.append(f'edge {eid!r} has no seed list')
    indirect = block.get('indirect_contrasts')
    if isinstance(indirect, dict):
        for name, row in indirect.items():
            if not isinstance(row, dict):
                problems.append(f'indirect contrast {name!r} is not an object')
    return problems


def _cn_gate_json_safe(value):
    """Recursively replace non-finite floats so a replay verdict is valid JSON.

    The summary echoes stored fields for comparison.  A hand-edited NaN/Inf
    must be reported as a difference, not make ``json.dumps`` fail while
    printing the verdict that names it.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _cn_gate_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cn_gate_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_cn_gate_json_safe(v) for v in value)
    return value


def _cn_gate_summary(block):
    if not isinstance(block, dict):
        return None
    ref = block.get('reference') if isinstance(block.get('reference'), dict) else {}
    closure = block.get('closure') if isinstance(block.get('closure'), dict) else {}
    edges = block.get('edges') if isinstance(block.get('edges'), dict) else {}
    indirect = (block.get('indirect_contrasts')
                if isinstance(block.get('indirect_contrasts'), dict) else {})
    summary = {
        'status': block.get('status'),
        'n_entries': block.get('n_entries'),
        'n_edges_declared': block.get('n_edges_declared'),
        'n_edges': block.get('n_edges'),
        'n_nodes': block.get('n_nodes'),
        'n_components': block.get('n_components'),
        'n_ignored': _cn_gate_len(block.get('ignored_entries')),
        'n_excluded': _cn_gate_len(block.get('excluded_edges')),
        'reference_node': ref.get('node'),
        'reference_seed_support': ref.get('seed_support'),
        'reference_covers': ref.get('covers_declared_components'),
        'reference_missing': ref.get('missing_declared_components'),
        'closure_status': closure.get('status'),
        'closure_max_abs_residual': closure.get('max_abs_residual'),
        'closure_n_cycles': closure.get('n_cycles'),
        'cross_component_ranking_allowed': block.get(
            'cross_component_ranking_allowed'),
        'n_node_conflicts': _cn_gate_len(block.get('node_conflicts')),
        'n_conflicting_edges': _cn_gate_len(block.get('conflicting_edges')),
        'n_indirect': _cn_gate_len(block.get('indirect_contrasts')),
        'edges': {str(k): (v.get('log_ratio_mean') if isinstance(v, dict) else None)
                  for k, v in edges.items()},
        'edge_meta': {
            str(k): ([v.get('with'), v.get('without'), v.get('tag_with'),
                      v.get('tag_without'), v.get('component'),
                      v.get('kind'), v.get('seeds'), v.get('n'),
                      v.get('relative_effect_pct')]
                     if isinstance(v, dict) else None)
            for k, v in edges.items()
        },
        'indirect': {
            str(k): (v.get('log_ratio_mean') if isinstance(v, dict) else None)
            for k, v in indirect.items()
        },
        'indirect_meta': {
            str(k): ([v.get('from'), v.get('to'), v.get('path'),
                      v.get('seeds'), v.get('n'), v.get('source'),
                      v.get('claim_allowed'), v.get('relative_effect_pct')]
                     if isinstance(v, dict) else None)
            for k, v in indirect.items()
        },
    }
    return _cn_gate_json_safe(summary)


def _audit_contrast_network_impl(path=None):
    """Replay the multi-arm composability block of a methodology audit.

    The gate recomputes :func:`exp_lib.contrast_network_audit` from the stored
    ``components``/``baselines`` entries and compares the graph status, edges,
    reference, closure and ``cross_component_ranking_allowed`` to the stored
    block.  Missing block -> ``not_verifiable_locally``; any disagreement or a
    smuggled ``confirmed``/``p_holm``/``verdict``/``decision`` key -> ``fail``.

    The block being replayed is descriptive: a pass says the arm table composes
    and its common-reference/indirect rows were not tampered with.  It does not
    make an indirect row a measured contrast, and it does not change any
    p-value, Holm family or frozen verdict.
    """
    out = {'kind': 'audit_contrast_network', 'status': 'not_verifiable_locally',
           'path': None, 'violations': [], 'warnings': [],
           'stored': None, 'recomputed': None,
           'role': ('replay the multi-arm composability block only; no p-value, '
                    'Holm family, verified verdict or frozen decision is changed')}
    p = path
    if p is not None:
        p = os.fspath(p)
        if os.path.isdir(p):
            p = os.path.join(p, 'methodology_audit.json')
        p = _audit_artifact(p)
    else:
        p = _cn_gate_default_path()
    if not p:
        out['warnings'].append(
            'no methodology_audit.json found under analysis_*/ or results_*/; '
            'the contrast-network block cannot be replayed locally')
        return out
    out['path'] = p
    rep = _load(p)
    if rep is None:
        out['warnings'].append('%s is absent or not a JSON object' % p)
        return out
    stored = rep.get('contrast_network')
    if stored is None:
        out['warnings'].append(
            'the methodology report carries no contrast_network block; '
            'regenerate the audit with the current library')
        return out
    if not isinstance(stored, dict):
        out['status'] = 'fail'
        out['violations'].append('contrast_network is not an object')
        return out
    forbidden = _cn_gate_forbidden(stored)
    if forbidden:
        out['status'] = 'fail'
        out['violations'].append(
            'the descriptive contrast-network block carries decision keys: '
            + ', '.join(sorted(set(forbidden))))
        return out
    shape_problems = _cn_gate_shape_problems(stored)
    if shape_problems:
        out['status'] = 'fail'
        out['violations'].extend(shape_problems)
        return out
    L = _import_exp_lib()
    try:
        recomputed = L.contrast_network_audit(
            rep, alpha=rep.get('alpha', 0.05), min_n=rep.get('min_n', 3))
    except (TypeError, ValueError) as e:
        out['status'] = 'fail'
        out['violations'].append(
            'the stored report cannot be replayed (%s: %s)' % (type(e).__name__, e))
        return out
    out['stored'] = _cn_gate_summary(stored)
    out['recomputed'] = _cn_gate_summary(recomputed)
    if out['stored'] is None or out['recomputed'] is None:
        out['status'] = 'fail'
        out['violations'].append('the stored or recomputed summary is not an object')
        return out
    for key in ('status', 'n_entries', 'n_edges_declared',
                'n_edges', 'n_nodes', 'n_components', 'n_ignored', 'n_excluded',
                'reference_node', 'reference_seed_support',
                'reference_covers', 'reference_missing', 'closure_status',
                'closure_max_abs_residual', 'closure_n_cycles',
                'cross_component_ranking_allowed',
                'n_node_conflicts', 'n_conflicting_edges', 'n_indirect'):
        if out['stored'].get(key) != out['recomputed'].get(key):
            out['violations'].append(
                '%s: stored=%r recomputed=%r'
                % (key, out['stored'].get(key), out['recomputed'].get(key)))
    for key in ('edge_meta', 'indirect_meta'):
        if out['stored'].get(key) != out['recomputed'].get(key):
            out['violations'].append(
                '%s differs: stored=%r recomputed=%r'
                % (key, out['stored'].get(key), out['recomputed'].get(key)))
    for label, key in (('edge', 'edges'), ('indirect', 'indirect')):
        se = out['stored'].get(key) or {}
        re_ = out['recomputed'].get(key) or {}
        if set(se) != set(re_):
            out['violations'].append(
                '%s set differs: stored-only=%s recomputed-only=%s'
                % (label, sorted(set(se) - set(re_)),
                   sorted(set(re_) - set(se))))
        for eid in sorted(set(se) & set(re_)):
            a, b = se.get(eid), re_.get(eid)
            if a is None or b is None:
                continue
            try:
                if abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(a)), abs(float(b))):
                    out['violations'].append(
                        '%s %s log-ratio mean: stored=%r recomputed=%r'
                        % (label, eid, a, b))
            except (TypeError, ValueError, OverflowError):
                out['violations'].append(
                    '%s %s carries a non-numeric log ratio' % (label, eid))
    if stored.get('status') == 'ok' and recomputed.get('status') != 'ok':
        out['violations'].append(
            'the stored block claims status=ok but the replay does not')
    if (stored.get('cross_component_ranking_allowed') is True
            and recomputed.get('cross_component_ranking_allowed') is not True):
        out['violations'].append(
            'the stored block allows cross-component ranking, but the replay '
            'does not')
    if stored.get('status') in ('ok', 'warn') and not recomputed.get('edges'):
        out['violations'].append(
            'the stored block describes an edge graph that the replay could not '
            'build')
    if out['violations']:
        out['status'] = 'fail'
    else:
        out['status'] = 'ok'
    return out


def audit_contrast_network(path=None):
    """Replay the multi-arm composability block and print the JSON verdict."""
    out = _audit_contrast_network_impl(path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
