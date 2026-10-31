# ---- multi-arm comparison network: composability, common reference, closure ----
# A panel audit checks every declared contrast PAIR independently: the two arms
# must share run_cfg / code semantics / evaluation fingerprints and the paired
# per-seed deltas are recomputed.  That is necessary but not sufficient for a
# multi-arm ablation table.  Individually valid pairs can still be mutually
# incomposable when the same arm label is measured in two different runs:
#
#   * pair (A, B) may use B from one run/evaluation slice, while pair (B, C)
#     uses a DIFFERENT B.  Each pair passes its own gate, yet "A > B > C" is a
#     splice of two experiments, not one three-arm comparison;
#   * components verified against different controls cannot be ranked on one
#     scale unless the arm graph connects them;
#   * a path-derived (indirect) effect must not be reported as if it were a
#     measured contrast.
#
# This fragment adds the arm-graph readout.  Nodes are (arm, protocol tag)
# pairs; each controlled contrast is a directed edge ``without -> with`` whose
# per-seed weight is the paired log PPL ratio.  The audit:
#
#   1. verifies that every arm node has ONE per-seed PPL measurement across all
#      pairs that mention it (a conflicting value is a named node split, not a
#      silent average);
#   2. builds the connected components of the clean edge graph and names which
#      declared ablation components live in each one;
#   3. reports whether a single common reference covers every declared
#      component, which is the precondition for a cross-component ranking;
#   4. checks cycle closure on the exact per-seed log ratios.  Given a
#      consistent node this closure is an identity, so a non-zero residual is
#      evidence that the recorded arm is not the same measurement, NOT an
#      interaction test;
#   5. derives indirect contrasts only on paths whose edges share seeds and
#      whose nodes are consistent.  Those rows are labelled ``derived`` and
#      never enter a Holm family, a ``verified`` verdict or a frozen decision.
#
# The block is descriptive.  It does not add a p-value to any declared family,
# does not change an existing p/Holm/verified result, and does not read the
# test split.  It only prevents a multi-arm table from quoting an ordering or
# an indirect comparison that its own measurements do not compose.

CONTRAST_NETWORK_POLICY = {
    'version': 'contrast-network-1',
    'ppl_relative_tolerance': 1e-12,
    'log_tolerance': 1e-12,
    'min_shared_seeds': 2,
    'max_indirect_pairs': 128,
    'reference_rule': ('the clean-graph node with the largest degree; ties are '
                       'broken by node id.  A caller may name an explicit '
                       'reference arm instead'),
    'closure_note': ('with one consistent per-seed PPL value per arm node, a '
                     'cycle of log ratios sums to zero identically.  A non-zero '
                     'residual is a measurement-identity failure, not an '
                     'interaction estimate'),
    'role': ('multi-arm composability / common-reference description; it adds '
             'no p-value to a declared family and never changes p/Holm/'
             'verified or a frozen decision'),
}


def _cn_node_id(arm, tag):
    """Stable node id for an (arm, protocol tag) pair."""
    return str(arm) if not tag else '%s::%s' % (str(arm), str(tag))


def _cn_tag(entry, side):
    raw = entry.get('tag_%s' % side)
    return '' if raw is None else str(raw)


def _cn_same(a, b, tol):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= float(tol) * max(1.0, abs(a), abs(b))


def _cn_entries(entries):
    """Flatten a methodology report / grouped entries into a list of entries."""
    if isinstance(entries, dict):
        grouped = []
        if isinstance(entries.get('components'), dict):
            for _comp, _items in entries['components'].items():
                if isinstance(_items, (list, tuple)):
                    grouped.extend(_items)
        if isinstance(entries.get('baselines'), dict):
            for _comp, _items in entries['baselines'].items():
                if isinstance(_items, (list, tuple)):
                    grouped.extend(_items)
        if grouped:
            return grouped
        # A plain {name: [entries]} mapping is also accepted for direct use.
        for _items in entries.values():
            if isinstance(_items, (list, tuple)):
                grouped.extend(_items)
        return grouped
    if isinstance(entries, (list, tuple)):
        return list(entries)
    return []


def _cn_normalise_entry(entry):
    """(normalised edge, None) or (None, named reason)."""
    if not isinstance(entry, dict):
        return None, 'not an object'
    a, b = entry.get('with'), entry.get('without')
    if not isinstance(a, str) or not a.strip():
        return None, 'with arm is not a non-empty string'
    if not isinstance(b, str) or not b.strip():
        return None, 'without arm is not a non-empty string'
    if entry.get('controlled') is not True:
        return None, 'not a controlled pair'
    if entry.get('fully_paired') is not True:
        return None, 'not a fully paired contrast'
    raw_seeds = entry.get('seeds')
    raw_with = entry.get('per_seed_with')
    raw_without = entry.get('per_seed_without')
    if (not isinstance(raw_seeds, (list, tuple))
            or not isinstance(raw_with, (list, tuple))
            or not isinstance(raw_without, (list, tuple))):
        return None, 'per-seed arm PPL vectors are absent'
    if not (len(raw_seeds) == len(raw_with) == len(raw_without)):
        return None, 'seed / per-seed arm PPL lengths differ'
    seeds, with_vals, without_vals = [], [], []
    for _s, _w, _wo in zip(raw_seeds, raw_with, raw_without):
        _si = _as_seed_int(_s)
        _wf = _finite_scalar(_w)
        _wof = _finite_scalar(_wo)
        if _si is None:
            return None, 'seed is not an integer'
        if _wf is None or _wof is None or _wf <= 0.0 or _wof <= 0.0:
            return None, 'per-seed arm PPL is not a finite positive number'
        seeds.append(int(_si))
        with_vals.append(float(_wf))
        without_vals.append(float(_wof))
    if len(set(seeds)) != len(seeds):
        return None, 'duplicate seeds'
    if len(seeds) < int(CONTRAST_NETWORK_POLICY['min_shared_seeds']):
        return None, 'fewer than %d paired seeds' % int(
            CONTRAST_NETWORK_POLICY['min_shared_seeds'])
    # The stored raw deltas are a derived field.  If they disagree with the
    # per-seed PPLs, the summary is stale; a network that used either number
    # could silently contradict the other, so the edge is excluded by name.
    raw_deltas = entry.get('deltas')
    if raw_deltas is not None:
        if not isinstance(raw_deltas, (list, tuple)) or len(raw_deltas) != len(seeds):
            return None, 'stored delta list length differs from the seed count'
        for _s, _d, _w, _wo in zip(seeds, raw_deltas, with_vals, without_vals):
            _df = _finite_scalar(_d)
            if _df is None or not _cn_same(_df, _w - _wo,
                                           CONTRAST_NETWORK_POLICY['ppl_relative_tolerance']):
                return None, 'stored delta disagrees with per-seed arm PPL at seed %s' % _s
    lr = {s: math.log(w / wo) for s, w, wo in zip(seeds, with_vals, without_vals)}
    nid_a = _cn_node_id(a.strip(), _cn_tag(entry, 'with'))
    nid_b = _cn_node_id(b.strip(), _cn_tag(entry, 'without'))
    return {
        'with': a.strip(), 'without': b.strip(),
        'tag_with': _cn_tag(entry, 'with'), 'tag_without': _cn_tag(entry, 'without'),
        'node_with': nid_a, 'node_without': nid_b,
        'component': entry.get('component'),
        'kind': entry.get('kind'),
        'seeds': sorted(seeds),
        'with_values': {s: w for s, w in zip(seeds, with_vals)},
        'without_values': {s: wo for s, wo in zip(seeds, without_vals)},
        'log_ratio': lr,
        'note': entry.get('note'),
    }, None


def _cn_edge_id(edge):
    return '%s -> %s' % (edge['node_without'], edge['node_with'])


def _cn_edge_mean(edge):
    vals = [edge['log_ratio'][s] for s in edge['seeds']]
    return float(sum(vals) / len(vals))


def _cn_bfs_component(start, adj, seen):
    stack = [start]
    seen.add(start)
    nodes = []
    while stack:
        n = stack.pop()
        nodes.append(n)
        for nxt, _eid, _sign in adj.get(n, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return sorted(nodes)


def _cn_shortest_path(start, goal, adj):
    """Shortest undirected path as [(node, edge_id, sign)] steps, or None."""
    if start == goal:
        return []
    prev = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt, eid, sign in adj.get(cur, ()):
            if nxt in prev:
                continue
            prev[nxt] = (cur, eid, sign)
            if nxt == goal:
                steps = []
                node = goal
                while prev[node] is not None:
                    p, e, sg = prev[node]
                    steps.append((p, e, sg))
                    node = p
                steps.reverse()
                return steps
            queue.append(nxt)
    return None


def _cn_path_values(steps, edges, seeds):
    """Per-seed log ratios for a path on the intersection of edge seed sets."""
    common = set(seeds)
    for _p, eid, _sg in steps:
        common &= set(edges[eid]['log_ratio'])
    common = sorted(common)
    out = {}
    for s in common:
        out[s] = float(sum(sign * edges[eid]['log_ratio'][s]
                           for _p, eid, sign in steps))
    return out


def contrast_network_audit(entries, *, reference_arm=None, alpha=0.05, min_n=3):
    """Audit whether a panel's pairwise ablation contrasts compose into one graph.

    ``entries`` may be a :func:`methodology_audit` report (or its flattened
    component/baseline entries).  Only complete controlled pairs that carry
    both per-seed arm PPLs are eligible; a pair with a stale ``deltas`` field is
    excluded and named.

    The returned block reports nodes, clean edges, graph components, the
    reference arm, node-identity conflicts, cycle closure, indirect path
    contrasts, and ``cross_component_ranking_allowed``.  The latter is True only
    when at least two declared components have clean edges, every such edge
    belongs to one connected component, a common reference covers them, and no
    node is measured twice with different PPLs.  It is False/None otherwise,
    with named reasons.  Nothing here mutates an input entry or creates a
    ``verified``/Holm claim."""
    alpha = _validated_alpha(alpha, context='contrast_network_audit alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError('contrast_network_audit: min_n must be a non-negative '
                         'integer, got %r' % (min_n,))
    min_n = int(_min_n)
    out = {
        'kind': 'contrast_network_audit',
        'status': 'not_estimable',
        'policy': dict(CONTRAST_NETWORK_POLICY),
        'alpha': float(alpha), 'min_n': min_n,
        'n_entries': 0, 'n_edges_declared': 0, 'n_edges': 0,
        'n_nodes': 0, 'n_components': 0,
        'edges': {}, 'nodes': {}, 'graph_components': {},
        'declared_components': {},
        'reference': {'node': None, 'arm': None, 'tag': None, 'basis': None,
                      'graph_component': None, 'covers_declared_components': [],
                      'missing_declared_components': []},
        'cross_component_ranking_allowed': None,
        'ranking_reasons': [],
        'closure': {'status': 'not_applicable', 'n_cycles': 0,
                    'max_abs_residual': None,
                    'tolerance': float(CONTRAST_NETWORK_POLICY['log_tolerance']),
                    'cycles': [], 'warnings': []},
        'node_conflicts': [], 'conflicting_edges': [],
        'excluded_edges': {}, 'ignored_entries': [],
        'indirect_contrasts': {}, 'warnings': [],
        'role': CONTRAST_NETWORK_POLICY['role'],
    }
    raw_entries = _cn_entries(entries)
    out['n_entries'] = len(raw_entries)
    if not raw_entries:
        out['warnings'].append('no component/baseline entries to compose')
        return out
    normalised = []
    _data_defect = False
    for i, entry in enumerate(raw_entries):
        edge, reason = _cn_normalise_entry(entry)
        if edge is None:
            out['ignored_entries'].append({
                'index': i,
                'with': entry.get('with') if isinstance(entry, dict) else None,
                'without': entry.get('without') if isinstance(entry, dict) else None,
                'component': entry.get('component') if isinstance(entry, dict) else None,
                'reason': reason,
            })
            if reason and str(reason).startswith('stored delta disagrees'):
                _data_defect = True
            continue
        normalised.append(edge)
    out['n_edges_declared'] = len(normalised)
    if not normalised:
        out['warnings'].append(
            'no complete controlled contrast carries per-seed arm PPLs; the '
            'arm graph cannot be composed (this is not a pass)')
        return out
    # Duplicate edge identity: the same pair/tag tuple may be listed under two
    # components.  Identical values are deduplicated; contradictory values are
    # a hard identity conflict, because either number could be quoted.
    by_id = {}
    for edge in normalised:
        eid = _cn_edge_id(edge)
        if eid not in by_id:
            by_id[eid] = edge
            continue
        old = by_id[eid]
        same = (old['seeds'] == edge['seeds']
                and all(_cn_same(old['log_ratio'][s], edge['log_ratio'][s],
                                 CONTRAST_NETWORK_POLICY['log_tolerance'])
                        for s in old['seeds']))
        if same:
            continue
        out['conflicting_edges'].append({
            'edge': eid,
            'reason': 'the same arm pair/tag is recorded with different per-seed values',
            'components': sorted({str(old.get('component')), str(edge.get('component'))}),
        })
        by_id.pop(eid, None)
    # Node identity: one arm+tag must have one PPL per seed across every edge
    # that mentions it.  A difference means the label refers to two runs or two
    # evaluation slices; averaging them would fabricate a single arm.
    observations = {}
    for eid, edge in by_id.items():
        for side in ('with', 'without'):
            node = edge['node_%s' % side]
            vals = edge['%s_values' % side]
            for s in edge['seeds']:
                observations.setdefault((node, s), []).append((eid, vals[s]))
    split_nodes = set()
    conflicts_by_node = {}
    for (node, seed), rows in sorted(observations.items()):
        vals = [v for _eid, v in rows]
        if all(_cn_same(vals[0], v, CONTRAST_NETWORK_POLICY['ppl_relative_tolerance'])
               for v in vals[1:]):
            continue
        split_nodes.add(node)
        _base = vals[0]
        _max_rel = 0.0
        for v in vals[1:]:
            denom = max(1.0, abs(_base), abs(v))
            _max_rel = max(_max_rel, abs(v - _base) / denom)
        info = conflicts_by_node.setdefault(node, {
            'node': node, 'seeds': [],
            'max_relative_difference': 0.0,
            'reason': ('the same arm/protocol label has different per-seed PPLs '
                       'across pairs; it is not one measurement'),
        })
        info['seeds'].append({
            'seed': int(seed),
            'values': [{'edge': eid, 'ppl': float(v)} for eid, v in rows],
            'max_relative_difference': float(_max_rel),
        })
        info['max_relative_difference'] = max(
            info['max_relative_difference'], float(_max_rel))
    for node in sorted(conflicts_by_node):
        info = conflicts_by_node[node]
        info['seeds'].sort(key=lambda row: row['seed'])
        out['node_conflicts'].append(info)
    for node in sorted(split_nodes):
        out['warnings'].append(
            'arm node %s is measured more than once with different per-seed PPLs; '
            'every edge that mentions it is excluded from the composable graph' % node)
    clean = {}
    for eid, edge in by_id.items():
        if edge['node_with'] in split_nodes or edge['node_without'] in split_nodes:
            out['excluded_edges'][eid] = 'touches a split arm node'
            continue
        clean[eid] = edge
    out['n_edges'] = len(clean)
    for eid, edge in clean.items():
        mean = _cn_edge_mean(edge)
        try:
            rel = float(100.0 * (math.exp(mean) - 1.0))
        except OverflowError:
            rel = None
        out['edges'][eid] = {
            'with': edge['node_with'], 'without': edge['node_without'],
            'tag_with': edge['tag_with'], 'tag_without': edge['tag_without'],
            'component': edge.get('component'), 'kind': edge.get('kind'),
            'seeds': list(edge['seeds']), 'n': len(edge['seeds']),
            'log_ratio_mean': float(mean),
            'relative_effect_pct': rel,
        }
    if not clean:
        out['warnings'].append(
            'every candidate edge touches a split arm node or a conflicting '
            'duplicate; there is no composable graph')
        out['status'] = 'warn'
        _declared_names = sorted({str(e.get('component')) for e in normalised
                                  if e.get('kind') in (None, 'component')
                                  and e.get('component') is not None})
        if len(_declared_names) >= 2:
            out['cross_component_ranking_allowed'] = False
            reasons = []
            if split_nodes:
                reasons.append(
                    '%d arm node(s) are measured twice with different per-seed '
                    'PPLs (split arm node: %s)'
                    % (len(split_nodes), ', '.join(sorted(split_nodes))))
            if out['conflicting_edges']:
                reasons.append('%d duplicate edge(s) disagree'
                               % len(out['conflicting_edges']))
            reasons.append('no composable clean edge remains; the multi-arm '
                           'ordering is not identified')
            out['ranking_reasons'] = reasons
        return out
    # ---- undirected adjacency for connectivity and paths -------------------
    adj = {}
    for eid, edge in clean.items():
        a, b = edge['node_without'], edge['node_with']
        adj.setdefault(a, []).append((b, eid, 1.0))
        adj.setdefault(b, []).append((a, eid, -1.0))
    for node in adj:
        adj[node].sort(key=lambda t: (t[0], t[1]))
    seen = set()
    components = []
    for node in sorted(adj):
        if node in seen:
            continue
        members = _cn_bfs_component(node, adj, seen)
        components.append(members)
    components.sort(key=lambda nodes: (len(nodes), nodes[0]))
    comp_index = {}
    for cid, nodes in enumerate(components):
        for node in nodes:
            comp_index[node] = cid
    out['n_nodes'] = len(comp_index)
    out['n_components'] = len(components)
    degree = {node: len(adj.get(node, ())) for node in comp_index}
    for node in sorted(comp_index):
        out['nodes'][node] = {
            'arm': node.split('::', 1)[0],
            'tag': (node.split('::', 1)[1] if '::' in node else ''),
            'seeds': sorted({s for eid, e in clean.items()
                             if node in (e['node_with'], e['node_without'])
                             for s in e['seeds']}),
            'degree': int(degree[node]),
            'graph_component': int(comp_index[node]),
        }
    for cid, nodes in enumerate(components):
        cedges = sorted(eid for eid, e in clean.items()
                        if e['node_with'] in nodes and e['node_without'] in nodes)
        local_ref = sorted(nodes, key=lambda n: (-degree[n], n))[0]
        out['graph_components'][str(cid)] = {
            'nodes': list(nodes), 'edges': cedges,
            'declared_components': [], 'local_reference': local_ref,
        }
    # ---- declared ablation components --------------------------------------
    declared = {}
    for edge in normalised:
        comp = edge.get('component')
        if comp is None:
            continue
        if edge.get('kind') not in (None, 'component'):
            continue
        info = declared.setdefault(str(comp), {'nodes': set(), 'clean_nodes': set(),
                                               'clean_edges': set()})
        info['nodes'].update((edge['node_with'], edge['node_without']))
        eid = _cn_edge_id(edge)
        if eid in clean:
            info['clean_nodes'].update((edge['node_with'], edge['node_without']))
            info['clean_edges'].add(eid)
    for comp in sorted(declared):
        info = declared[comp]
        cids = sorted({comp_index[n] for n in info['clean_nodes'] if n in comp_index})
        out['declared_components'][comp] = {
            'nodes': sorted(info['nodes']),
            'clean_nodes': sorted(info['clean_nodes']),
            'graph_components': cids,
            'has_clean_edge': bool(info['clean_edges']),
            'connected': bool(cids) and len(cids) == 1,
            'clean_edges': sorted(info['clean_edges']),
        }
        if cids:
            for cid in cids:
                out['graph_components'][str(cid)]['declared_components'].append(comp)
    for cid in out['graph_components']:
        out['graph_components'][cid]['declared_components'] = sorted(
            set(out['graph_components'][cid]['declared_components']))
    # ---- common reference --------------------------------------------------
    ref_node = None
    ref_basis = None
    if reference_arm is not None:
        raw = reference_arm
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            raw = _cn_node_id(raw[0], raw[1])
        raw = str(raw)
        if raw in comp_index:
            ref_node, ref_basis = raw, 'explicit reference arm'
        else:
            matches = sorted(n for n in comp_index
                             if n == raw or n.split('::', 1)[0] == raw)
            if len(matches) == 1:
                ref_node, ref_basis = matches[0], 'explicit reference arm'
            elif matches:
                ref_node = sorted(matches, key=lambda n: (-degree[n], n))[0]
                ref_basis = 'explicit reference arm (ambiguous; highest degree)'
                out['warnings'].append(
                    'reference_arm %r matches several graph nodes %s; using %s'
                    % (raw, matches, ref_node))
    if ref_node is None:
        if reference_arm is not None:
            out['warnings'].append(
                'reference_arm %r is not a clean graph node; falling back to '
                'the highest-degree node' % (reference_arm,))
        ref_node = sorted(comp_index, key=lambda n: (-degree[n], n))[0]
        ref_basis = CONTRAST_NETWORK_POLICY['reference_rule']
    ref_cid = comp_index[ref_node]
    covers = []
    missing = []
    for comp, info in out['declared_components'].items():
        if not info['has_clean_edge']:
            missing.append(comp)
        elif info['clean_nodes'] and all(comp_index.get(n) == ref_cid
                                         for n in info['clean_nodes']):
            covers.append(comp)
        else:
            missing.append(comp)
    out['reference'] = {
        'node': ref_node,
        'arm': ref_node.split('::', 1)[0],
        'tag': (ref_node.split('::', 1)[1] if '::' in ref_node else ''),
        'basis': ref_basis,
        'graph_component': int(ref_cid),
        'covers_declared_components': sorted(covers),
        'missing_declared_components': sorted(missing),
        'seed_support': None,
    }
    # Connectivity is not enough for a PAIRED cross-component ranking: every
    # declared component must also be reachable from the reference on a common
    # seed set.  A graph can be connected through a shared arm while the two
    # sides were run on disjoint seeds (e.g. A-B on 0,1,2 and B-C on 3,4,5).
    # Those component effects can be reported separately, but their difference
    # is not a paired contrast.
    component_support = {}
    for comp, info in out['declared_components'].items():
        supports = []
        for node in info['clean_nodes']:
            if node == ref_node:
                continue
            steps = _cn_shortest_path(ref_node, node, adj)
            if steps is None:
                supports.append(set())
                continue
            common = None
            for _p, eid, _sg in steps:
                edge_seeds = set(clean[eid]['seeds'])
                common = edge_seeds if common is None else common & edge_seeds
            supports.append(common if common is not None else set())
        if supports:
            support = set(supports[0])
            for extra in supports[1:]:
                support &= extra
        else:
            support = set()
        info['seed_support'] = sorted(support)
        component_support[comp] = support
    if component_support:
        _global = set(next(iter(component_support.values())))
        for support in component_support.values():
            _global &= support
        out['reference']['seed_support'] = sorted(_global)
    # ---- cycle closure -----------------------------------------------------
    # BFS forest; every non-tree edge closes one fundamental cycle.  The
    # residual is computed from the recorded per-seed EDGE ratios, not from
    # node values, so a stale/duplicated edge can actually show up.
    tree_edges = set()
    visited = set()
    for root in sorted(adj):
        if root in visited:
            continue
        visited.add(root)
        queue = [root]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt, eid, _sign in adj.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                tree_edges.add(eid)
                queue.append(nxt)
    tree_adj = {}
    for eid in tree_edges:
        edge = clean[eid]
        a, b = edge['node_without'], edge['node_with']
        tree_adj.setdefault(a, []).append((b, eid, 1.0))
        tree_adj.setdefault(b, []).append((a, eid, -1.0))
    for node in tree_adj:
        tree_adj[node].sort(key=lambda t: (t[0], t[1]))
    closure_cycles = []
    max_abs = 0.0
    for eid in sorted(clean):
        if eid in tree_edges:
            continue
        edge = clean[eid]
        u, v = edge['node_without'], edge['node_with']
        steps = _cn_shortest_path(v, u, tree_adj)
        if steps is None:
            out['closure']['warnings'].append(
                'non-tree edge %s has no tree path; closure not formed' % eid)
            continue
        path_edges = [e for _p, e, _s in steps] + [eid]
        common = set(clean[eid]['seeds'])
        for pe in path_edges:
            common &= set(clean[pe]['seeds'])
        common = sorted(common)
        if len(common) < int(CONTRAST_NETWORK_POLICY['min_shared_seeds']):
            out['closure']['warnings'].append(
                'cycle closed by %s has fewer than %d shared seeds; residual '
                'not evaluated' % (eid, int(CONTRAST_NETWORK_POLICY['min_shared_seeds'])))
            continue
        # Traverse the cycle from v along the tree to u, then back to v on
        # eid.  Each step is (current_node, edge, direction); the direction is
        # re-derived from the edge endpoints so a malformed adjacency cannot
        # flip a term silently.
        residual = {}
        for s in common:
            total = 0.0
            for cur, pe, _sg in steps:
                ed = clean[pe]
                if ed['node_without'] == cur:
                    total += ed['log_ratio'][s]
                elif ed['node_with'] == cur:
                    total -= ed['log_ratio'][s]
                else:
                    total = float('nan')
                    break
            # Close the cycle along the non-tree edge from u back to v.
            ed = clean[eid]
            if ed['node_without'] == u and ed['node_with'] == v:
                total += ed['log_ratio'][s]
            elif ed['node_with'] == u and ed['node_without'] == v:
                total -= ed['log_ratio'][s]
            else:
                total = float('nan')
            residual[s] = total
        if any(not math.isfinite(v) for v in residual.values()):
            out['closure']['warnings'].append(
                'cycle closed by %s produced a non-finite residual; it was '
                'not evaluated' % eid)
            continue
        vals = [abs(v) for v in residual.values()]
        cyc_max = max(vals)
        max_abs = max(max_abs, cyc_max)
        closure_cycles.append({
            'non_tree_edge': eid,
            'edges': path_edges,
            'seeds': common,
            'residual_mean': float(sum(residual.values()) / len(residual)),
            'max_abs_residual': float(cyc_max),
            'n': len(common),
        })
    tol = float(CONTRAST_NETWORK_POLICY['log_tolerance'])
    if not closure_cycles:
        closure_status = 'not_applicable'
    elif max_abs <= tol:
        closure_status = 'coherent'
    else:
        closure_status = 'incoherent'
    out['closure'] = {
        'status': closure_status,
        'n_cycles': len(closure_cycles),
        'max_abs_residual': float(max_abs) if closure_cycles else None,
        'tolerance': tol,
        'cycles': closure_cycles,
        'warnings': out['closure'].get('warnings') or [],
    }
    # ---- indirect (path-derived) contrasts --------------------------------
    declared_nodes = set()
    for info in out['declared_components'].values():
        declared_nodes.update(info['clean_nodes'])
    candidates = []
    nodes_sorted = sorted(declared_nodes)
    for i, a in enumerate(nodes_sorted):
        for b in nodes_sorted[i + 1:]:
            direct = any(
                (e['node_without'] == a and e['node_with'] == b)
                or (e['node_without'] == b and e['node_with'] == a)
                for e in clean.values())
            if direct:
                continue
            # Report log(PPL_from / PPL_to): traverse from the comparison
            # endpoint back to `from`, so a negative value means `from` has
            # the lower PPL.  This matches the without -> with edge direction.
            steps = _cn_shortest_path(b, a, adj)
            if steps is None:
                continue
            candidates.append((a, b, steps))
    candidates.sort(key=lambda t: (t[0], t[1]))
    cap = int(CONTRAST_NETWORK_POLICY['max_indirect_pairs'])
    for a, b, steps in candidates[:cap]:
        vals = _cn_path_values(steps, clean, set().union(
            *[set(clean[e]['seeds']) for _p, e, _s in steps]))
        if len(vals) < int(CONTRAST_NETWORK_POLICY['min_shared_seeds']):
            continue
        seeds = sorted(vals)
        lr = [vals[s] for s in seeds]
        st = paired_effect_stats(lr, alpha=alpha, min_n=min_n, seeds=seeds)
        mean = float(st.get('mean')) if st.get('mean') is not None else None
        rel = None
        if mean is not None and math.isfinite(mean):
            try:
                rel = float(100.0 * (math.exp(mean) - 1.0))
            except OverflowError:
                rel = None
        out['indirect_contrasts']['%s -> %s' % (a, b)] = {
            'from': a, 'to': b,
            'path': [eid for _p, eid, _s in steps],
            'seeds': seeds, 'n': len(seeds),
            'log_ratio_mean': mean,
            'relative_effect_pct': rel,
            'p_exact_signflip': st.get('p_exact_signflip'),
            'resolution_limited': st.get('resolution_limited'),
            'underpowered': st.get('underpowered'),
            'source': 'derived_path',
            'claim_allowed': False,
            'role': ('descriptive path-derived contrast; it is not a measured '
                     'pair, adds no p-value to a declared family, and cannot '
                     'become verified here'),
        }
    if len(candidates) > cap:
        out['warnings'].append(
            'indirect contrasts are capped at %d; %d further connected pairs were '
            'not derived' % (cap, len(candidates) - cap))
    # ---- ranking gate ------------------------------------------------------
    reasons = []
    applicable = len(out['declared_components']) >= 2
    if not applicable:
        reasons.append('fewer than two declared ablation components; a '
                       'cross-component ranking claim is not applicable')
    if out['node_conflicts']:
        reasons.append('%d arm node(s) are measured twice with different '
                       'per-seed PPLs' % len(out['node_conflicts']))
    if out['conflicting_edges']:
        reasons.append('%d duplicate edge(s) disagree' % len(out['conflicting_edges']))
    if out['n_components'] > 1:
        reasons.append('the clean arm graph has %d disconnected components'
                       % out['n_components'])
    _unconnected = sorted(c for c, i in out['declared_components'].items()
                          if not i['has_clean_edge'] or not i['connected'])
    if _unconnected:
        reasons.append('declared component(s) are not in one clean connected '
                       'graph: %s' % ', '.join(_unconnected))
    if out['reference']['missing_declared_components']:
        reasons.append('the chosen reference does not cover declared '
                       'component(s): %s'
                       % ', '.join(out['reference']['missing_declared_components']))
    if _data_defect:
        reasons.append('at least one stored contrast delta is stale; the '
                       'composable graph is not based on a self-consistent '
                       'table')
    _support = out['reference'].get('seed_support')
    if applicable and (_support is None
                       or len(_support) < int(CONTRAST_NETWORK_POLICY['min_shared_seeds'])):
        reasons.append(
            'the declared component paths do not share at least %d common '
            'seed(s); their effects are on different draws and a paired '
            'cross-component ranking is not identified'
            % int(CONTRAST_NETWORK_POLICY['min_shared_seeds']))
    if closure_status == 'incoherent':
        reasons.append('cycle closure residual %.3g exceeds tolerance %.3g'
                       % (max_abs, tol))
    if applicable:
        out['cross_component_ranking_allowed'] = not reasons
    out['ranking_reasons'] = reasons
    # ---- status and warnings ----------------------------------------------
    if (out['node_conflicts'] or out['conflicting_edges']
            or closure_status == 'incoherent' or _data_defect):
        out['status'] = 'warn'
    elif out['n_components'] > 1:
        out['status'] = 'warn'
    else:
        out['status'] = 'ok'
    if _data_defect:
        out['warnings'].append(
            'at least one stored contrast delta disagrees with its own per-seed '
            'arm PPLs; that edge was excluded instead of letting the report '
            'quote two different effects')
    if out['n_components'] > 1:
        out['warnings'].append(
            'the arm graph is disconnected (%d components); a contrast in one '
            'component cannot be composed with an arm in another'
            % out['n_components'])
    if out['cross_component_ranking_allowed'] is False and applicable:
        out['warnings'].append(
            'cross-component ranking is NOT allowed: ' + '; '.join(reasons))
    return out
