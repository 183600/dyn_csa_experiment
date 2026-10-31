# ---------------------------------------------------------------------------
# 7e) contrast network：多臂对比的可组合性、共同参照与闭环一致性
# ---------------------------------------------------------------------------

_CONTRAST_NETWORK_FORBIDDEN = frozenset({
    'confirmed', 'p_holm', 'p_holm_global', 'p_holm_family',
    'p_holm_component', 'verdict', 'decision', 'significant',
    'significant_global', 'holm',
})


def _cn_gate_forbidden(obj, path='', depth=0):
    """Recursively name forbidden decision keys smuggled into the block."""
    found = []
    if depth > 12:
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = '%s.%s' % (path, k) if path else str(k)
            if str(k) in _CONTRAST_NETWORK_FORBIDDEN:
                found.append(p)
            found.extend(_cn_gate_forbidden(v, p, depth + 1))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(_cn_gate_forbidden(v, '%s[%d]' % (path, i), depth + 1))
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


def _cn_gate_summary(block):
    if not isinstance(block, dict):
        return None
    ref = block.get('reference') if isinstance(block.get('reference'), dict) else {}
    closure = block.get('closure') if isinstance(block.get('closure'), dict) else {}
    edges = block.get('edges') if isinstance(block.get('edges'), dict) else {}
    return {
        'status': block.get('status'),
        'n_edges': block.get('n_edges'),
        'n_nodes': block.get('n_nodes'),
        'n_components': block.get('n_components'),
        'reference_node': ref.get('node'),
        'closure_status': closure.get('status'),
        'cross_component_ranking_allowed': block.get(
            'cross_component_ranking_allowed'),
        'n_node_conflicts': len(block.get('node_conflicts') or []),
        'n_conflicting_edges': len(block.get('conflicting_edges') or []),
        'n_indirect': len(block.get('indirect_contrasts') or {}),
        'edges': {str(k): (v.get('log_ratio_mean') if isinstance(v, dict) else None)
                  for k, v in edges.items()},
    }


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
    for key in ('status', 'n_edges', 'n_nodes', 'n_components',
                'reference_node', 'closure_status',
                'cross_component_ranking_allowed',
                'n_node_conflicts', 'n_conflicting_edges', 'n_indirect'):
        if out['stored'].get(key) != out['recomputed'].get(key):
            out['violations'].append(
                '%s: stored=%r recomputed=%r'
                % (key, out['stored'].get(key), out['recomputed'].get(key)))
    se, re_ = out['stored'].get('edges') or {}, out['recomputed'].get('edges') or {}
    if set(se) != set(re_):
        out['violations'].append(
            'edge set differs: stored-only=%s recomputed-only=%s'
            % (sorted(set(se) - set(re_)), sorted(set(re_) - set(se))))
    for eid in sorted(set(se) & set(re_)):
        a, b = se.get(eid), re_[eid]
        if a is None or b is None:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(a)), abs(float(b))):
                out['violations'].append(
                    'edge %s log-ratio mean: stored=%r recomputed=%r'
                    % (eid, a, b))
        except (TypeError, ValueError, OverflowError):
            out['violations'].append('edge %s carries a non-numeric log ratio' % eid)
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
