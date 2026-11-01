# ---- 消融晶格：从「留一对比」到可识别的组件归因 -------------------------
#
# A leave-one-out family answers a *context-specific* question: how much does
# removing component i change PPL when every other component is already
# active?  A 2x2 factorial answers a *pairwise* question at one context.  Both
# are valid, but neither identifies a component's average marginal
# contribution when the components interact: the same component can have a
# negative marginal in one context and a positive marginal in another.
#
# The standard object that averages those context-specific marginals over all
# orders is the Shapley value on the subset lattice.  It is only defined when
# every subset of the declared component set was measured with a common seed
# set; a star design (baseline + one component removed) does not contain the
# 2^k cells, so it cannot be relabelled as "component importance".  This
# fragment therefore:
#
#   * computes exact Shapley values, removal (LOO) effects, pairwise
#     interactions at every context, and the order-2+ Möbius remainder from a
#     complete lattice;
#   * refuses to compute when cells or seeds are missing (no imputation, no
#     averaging over different seed subsets);
#   * checks the efficiency / dummy / symmetry axioms and the additive
#     decomposition as internal consistency gates;
#   * audits the *design*: which declared factorial scopes are complete
#     lattices, which claims they license, and how many cells a proposed
#     higher-order lattice would still need.
#
# Sign convention: v(S) = log PPL of the subset-S arm.  Delta = active minus
# inactive, so a negative Shapley/LOO effect means the component lowers PPL
# (is beneficial), and a negative pair interaction means synergy.  A positive
# interaction means the two components overlap (sub-additive / redundant).
# Nothing here changes a frozen p-value, Holm family, verified/confirmed flag,
# or reads a data split.

import hashlib as _al_hashlib
import itertools as _al_itertools
import json as _al_json
import math as _al_math
import os as _al_os
import time as _al_time

AL_LATTICE_KIND = 'ablation_lattice'
AL_LATTICE_SCHEMA = 1
AL_LATTICE_MAX_COMPONENTS = 6
AL_LATTICE_TOL = 1e-9

AL_LATTICE_POLICY = {
    'estimand': ('exact Shapley value on the complete subset lattice of the '
                 'declared components, plus removal (LOO), pairwise '
                 'interaction and order-2+ Mobius remainder'),
    'scale': ('natural-log PPL; effect = active subset minus inactive subset; '
              'negative = lower PPL / beneficial'),
    'completeness': ('all 2^k subsets must be present; missing cells are '
                     'reported and never imputed'),
    'seed_coverage': ('complete-case intersection across every cell; missing '
                      'seeds are reported and never averaged over a different '
                      'subset'),
    'interaction_sign': ('negative = synergy; positive = sub-additive / '
                         'redundant overlap; zero = additive in that context'),
    'claim_scope': ('within one declared component scope; overlapping scopes '
                    'cannot be merged into a global Shapley without a full '
                    'joint lattice'),
    'frozen_decision_changed': False,
}


def _al_finite(value, *, where, positive=False):
    """Finite number for an analysis input; no strings/bools/NaN/Inf."""
    if isinstance(value, (bool,)):
        raise ValueError(f'{where}: expected a finite number, got {value!r}')
    try:
        if hasattr(value, 'item') and not isinstance(value, (int, float)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ValueError(f'{where}: expected a finite number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{where}: expected a finite number, got {value!r}') from None
    if not _al_math.isfinite(out):
        raise ValueError(f'{where}: expected a finite number, got {value!r}')
    if positive and out <= 0.0:
        raise ValueError(
            f'{where}: expected a positive PPL, got {value!r}')
    return out


def _al_int(value, *, where, minimum=None):
    if isinstance(value, bool):
        raise ValueError(f'{where}: expected an integer, got {value!r}')
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: expected an integer, got {value!r}') from None
    try:
        if value != out:
            raise ValueError(f'{where}: expected an integer, got {value!r}')
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: expected an integer, got {value!r}') from None
    if minimum is not None and out < minimum:
        raise ValueError(
            f'{where}: expected an integer >= {minimum}, got {value!r}')
    return out


def _al_component(value, *, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where}: component names must be non-empty strings, '
                         f'got {value!r}')
    name = value.strip()
    if any(ch in name for ch in '+,|,'):
        raise ValueError(
            f'{where}: component name {name!r} contains a lattice separator')
    return name


def _al_arm(value, *, where):
    """Arm ids are opaque labels; only non-empty strings are required."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where}: arm id must be a non-empty string, '
                         f'got {value!r}')
    return value.strip()


def _al_components(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, (list, tuple)):
        raise ValueError('components must be a list/tuple of names')
    out = []
    for i, item in enumerate(value):
        name = _al_component(item, where=f'components[{i}]')
        if name in out:
            raise ValueError(f'components: duplicate component {name!r}')
        out.append(name)
    if not out:
        raise ValueError('components must not be empty')
    return out


def _al_split_key(key):
    """Normalise a subset key (frozenset/tuple/list/string) to a tuple."""
    if isinstance(key, str):
        text = key.strip()
        if not text:
            return ()
        return tuple(part.strip() for part in
                     text.replace('|', '+').replace(',', '+').split('+')
                     if part.strip())
    if isinstance(key, (frozenset, set)):
        return tuple(sorted(key))
    if isinstance(key, (tuple, list)):
        return tuple(key)
    raise ValueError(
        f'cells: subset key must be a string/tuple/frozenset, got '
        f'{type(key).__name__}')


def _al_normalise_cells(cells, components, *, scale):
    if not isinstance(cells, dict):
        raise ValueError(f'cells must be a mapping, got {type(cells).__name__}')
    norm = {}
    for key, per_seed in cells.items():
        raw_parts = _al_split_key(key)
        parts = tuple(_al_component(p, where=f'cells[{key!r}]')
                      for p in raw_parts)
        subset = frozenset(parts)
        if len(subset) != len(parts):
            raise ValueError(f'cells[{key!r}]: duplicate component in subset')
        if components is not None:
            unknown = sorted(set(subset) - set(components))
            if unknown:
                raise ValueError(f'cells[{key!r}]: components not declared: '
                                 + ', '.join(unknown))
        if subset in norm:
            raise ValueError(f'cells: duplicate subset {sorted(subset)!r} after '
                             'normalisation')
        if not isinstance(per_seed, dict):
            raise ValueError(
                f'cells[{key!r}]: values must map seed -> measurement, got '
                f'{type(per_seed).__name__}')
        vals = {}
        for seed_key, val in per_seed.items():
            seed = _as_seed_int(seed_key)  # noqa: F821 (part_13)
            if seed is None:
                raise ValueError(
                    f'cells[{key!r}]: seed key {seed_key!r} is not an integer')
            if seed in vals:
                raise ValueError(
                    f'cells[{key!r}]: duplicate seed {seed!r} after normalisation')
            vals[seed] = _al_finite(
                val, where=f'cells[{key!r}][seed={seed}]',
                positive=(scale == 'ppl'))
        norm[subset] = vals
    return norm


def _al_name(subset, components):
    return '+'.join(sorted(subset, key=components.index))


def _al_all_subsets(components):
    out = []
    for size in range(len(components) + 1):
        for combo in _al_itertools.combinations(components, size):
            out.append(frozenset(combo))
    return out


def _al_num(value):
    """Finite JSON number or None; strict-JSON writers must not see NaN."""
    if isinstance(value, (bool,)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _al_math.isfinite(out) else None


def _al_direction(mean):
    if mean is None:
        return None
    if mean > AL_LATTICE_TOL:
        return 1
    if mean < -AL_LATTICE_TOL:
        return -1
    return 0


def _al_stats(vector, seeds, *, alpha, min_n):
    """Paired seed-level summary of a lattice-derived per-seed effect."""
    if not vector:
        return {'n': 0, 'seeds': [], 'mean': None, 'std': None,
                'std_defined': False, 'sem': None, 'ci95_low': None,
                'ci95_high': None, 'p_exact_signflip': None, 'p_floor': None,
                'underpowered': True, 'resolution_limited': True,
                'within_seed_noise': True, 'significant': False,
                'effect_vs_noise': 'no data', 'seed_driven': None,
                'seed_robust': None, 'ratio': None, 'percent_change': None,
                'direction': None}
    st = paired_effect_stats(  # noqa: F821 (part_10)
        vector, alpha=alpha, min_n=min_n, seeds=list(seeds))
    keys = ('n', 'n_input', 'n_dropped', 'dropped_seeds', 'complete_pairs',
            'seeds', 'mean', 'std', 'std_defined', 'sem', 'ci95_low',
            'ci95_high', 'p_exact_signflip', 'p_floor', 'underpowered',
            'resolution_limited', 'within_seed_noise', 'significant',
            'effect_vs_noise', 'seed_driven', 'seed_robust')
    out = {k: st.get(k) for k in keys}
    for key in ('mean', 'std', 'sem', 'ci95_low', 'ci95_high',
                'p_exact_signflip', 'p_floor'):
        out[key] = _al_num(out.get(key))
    mean = out.get('mean')
    if mean is not None and -700.0 < mean < 700.0:
        out['ratio'] = _al_num(_al_math.exp(mean))
        out['percent_change'] = _al_num(100.0 * (_al_math.exp(mean) - 1.0))
    else:
        out['ratio'] = None
        out['percent_change'] = None
    out['direction'] = _al_direction(mean)
    return out


def _al_rank(values, *, tol=AL_LATTICE_TOL):
    """Average ranks, ascending; ties share the average rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]] - values[order[i]]) <= tol * max(
                1.0, abs(values[order[i]]), abs(values[order[j + 1]])):
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def _al_spearman(a, b):
    n = len(a)
    if n < 2:
        return None
    ra, rb = _al_rank(a), _al_rank(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va <= 0.0 or vb <= 0.0:
        return None
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    return _al_num(cov / _al_math.sqrt(va * vb))


def _al_shapley(values, components):
    """Exact Shapley values for one seed's lattice values."""
    k = len(components)
    out = {}
    facts = [_al_math.factorial(i) for i in range(k + 1)]
    for comp in components:
        others = [c for c in components if c != comp]
        phi = 0.0
        for size in range(k):
            weight = facts[size] * facts[k - size - 1] / facts[k]
            for combo in _al_itertools.combinations(others, size):
                subset = frozenset(combo)
                phi += weight * (values[subset | {comp}] - values[subset])
        out[comp] = float(phi)
    return out


def _al_cells_to_records(cells, components):
    records = []
    for subset in _al_all_subsets(components):
        if subset not in cells:
            continue
        per = cells[subset]
        records.append({
            'subset': [c for c in components if c in subset],
            'per_seed': {str(seed): per[seed] for seed in sorted(per)},
        })
    return records


def _al_cells_from_records(records, components, *, scale):
    if not isinstance(records, (list, tuple)):
        raise ValueError('inputs.cells must be a list of records')
    cells = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f'inputs.cells[{i}] must be an object')
        subset = rec.get('subset')
        if not isinstance(subset, (list, tuple)):
            raise ValueError(f'inputs.cells[{i}].subset must be a list')
        per_seed = rec.get('per_seed')
        if not isinstance(per_seed, dict):
            raise ValueError(f'inputs.cells[{i}].per_seed must be an object')
        parts = [_al_component(x, where=f'inputs.cells[{i}].subset')
                 for x in subset]
        if len(set(parts)) != len(parts):
            raise ValueError(
                f'inputs.cells[{i}].subset repeats a component')
        key = frozenset(parts)
        if key in cells:
            raise ValueError(f'inputs.cells: duplicate subset {sorted(key)!r}')
        vals = {}
        for seed_key, val in per_seed.items():
            seed = _as_seed_int(seed_key)  # noqa: F821 (part_13)
            if seed is None:
                raise ValueError(
                    f'inputs.cells[{i}]: seed {seed_key!r} is not an integer')
            if seed in vals:
                raise ValueError(
                    f'inputs.cells[{i}]: duplicate seed {seed!r}')
            vals[seed] = _al_finite(
                val, where=f'inputs.cells[{i}][seed={seed}]',
                positive=(scale == 'ppl'))
        cells[key] = vals
    return cells


def shapley_lattice_attribution(
        cells, *, components=None, scale='ppl', min_n=3, alpha=0.05,
        max_components=AL_LATTICE_MAX_COMPONENTS, tol=AL_LATTICE_TOL):
    """Exact component attribution on a complete ablation subset lattice.

    ``cells`` maps subset -> {seed: measurement}.  A subset key may be a
    tuple/frozenset of component names or a ``+``-joined string (the empty
    subset is ``''`` / ``()``).  With ``scale='ppl'`` the measurement is a
    positive perplexity and is log-transformed; with ``scale='log_ppl'`` it is
    already a log PPL and is used as-is.

    The function is deliberately **all-or-nothing**: every one of the 2^k
    subsets and a common seed set are required.  A missing cell is named in
    ``missing_subsets`` and the result is ``not_estimable``; the function
    never substitutes a lower-order contrast, imputes a cell, or computes
    Shapley values over a seed subset that differs per cell.  If a cell has
    fewer seeds than the others, the common intersection is used and
    ``seed_coverage.coverage_ok`` is false; if that intersection falls below
    ``min_n``, the result is ``not_estimable``.
    """
    if scale not in ('ppl', 'log_ppl'):
        raise ValueError("scale must be 'ppl' or 'log_ppl'")
    _alpha = _al_finite(alpha, where='alpha')
    if not (0.0 < _alpha < 1.0):
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    _min_n = _al_int(min_n, where='min_n', minimum=0)
    _max_k = _al_int(max_components, where='max_components', minimum=1)
    _tol = _al_finite(tol, where='tol')
    if _tol < 0.0:
        raise ValueError(f'tol must be non-negative, got {tol!r}')
    if isinstance(components, (list, tuple)) and len(components) == 0:
        # An empty component list is a degenerately small lattice, not a type
        # error: report not_estimable so a stored empty artifact can replay.
        comps = []
    else:
        comps = _al_components(components)
    norm = _al_normalise_cells(cells, comps, scale=scale)
    if comps is None:
        inferred = set()
        for subset in norm:
            inferred.update(subset)
        comps = sorted(inferred)
        # Re-check names now that the component set is explicit.
        comps = [_al_component(c, where='components') for c in comps]
    k = len(comps)
    base = {
        'kind': AL_LATTICE_KIND,
        'schema': AL_LATTICE_SCHEMA,
        'policy': dict(AL_LATTICE_POLICY),
        'components': list(comps),
        'n_components': k,
        'scale': scale,
        'alpha': float(_alpha),
        'min_n': int(_min_n),
        'max_components': int(_max_k),
        'tolerance': float(_tol),
        'complete': False,
        'expected_subsets': [],
        'missing_subsets': [],
        'seed_coverage': {},
        'cell_means': {},
        'total_effect': {},
        'shapley': {},
        'loo': {},
        'marginals': {},
        'interactions': [],
        'higher_order': {},
        'decomposition': {},
        'axioms': {},
        'rank_agreement': {},
        'warnings': [],
        'problems': [],
        'note': ('exact lattice attribution; not a frozen confirmatory test and '
                 'not a substitute for a pre-registered contrast'),
    }
    if k == 0:
        base['problems'].append('no components were declared')
        base['status'] = 'not_estimable'
        return base
    if k > _max_k:
        base['problems'].append(
            f'{k} components exceed max_components={_max_k}; the full lattice '
            f'would need {2 ** k} cells')
        base['status'] = 'not_estimable'
        return base
    expected = _al_all_subsets(comps)
    expected_names = [_al_name(s, comps) for s in expected]
    base['expected_subsets'] = expected_names
    missing = [s for s in expected if s not in norm]
    extra = [s for s in norm if s not in expected]
    base['missing_subsets'] = [_al_name(s, comps) for s in missing]
    base['complete'] = not missing and not extra
    union_seeds = sorted(set().union(*[set(v) for v in norm.values()])) \
        if norm else []
    common = sorted(set.intersection(*[set(v) for v in norm.values()])) \
        if norm else []
    missing_by_cell = {}
    for subset in expected:
        if subset not in norm:
            continue
        miss = sorted(set(union_seeds) - set(norm[subset]))
        if miss:
            missing_by_cell[_al_name(subset, comps)] = miss
    extra_by_cell = {}
    for subset in expected:
        if subset not in norm:
            continue
        extra_seeds = sorted(set(norm[subset]) - set(common))
        if extra_seeds:
            extra_by_cell[_al_name(subset, comps)] = extra_seeds
    base['seed_coverage'] = {
        'n_seeds_union': len(union_seeds),
        'n_common_seeds': len(common),
        'common_seeds': common,
        'coverage_ok': (bool(common) and not missing_by_cell
                        and not extra_by_cell),
        'missing_by_cell': missing_by_cell,
        'extra_by_cell': extra_by_cell,
        'policy': ('complete-case intersection across all cells; a seed is '
                   'usable only when every cell measured it'),
    }
    # Cell means on the common seed set (never a per-cell seed subset).
    if common:
        for subset in expected:
            if subset not in norm:
                continue
            vals = [(_al_math.log(norm[subset][s]) if scale == 'ppl'
                     else norm[subset][s]) for s in common]
            base['cell_means'][_al_name(subset, comps)] = {
                'mean': _al_num(sum(vals) / len(vals)),
                'n': len(vals),
                'seeds': common,
            }
    if missing:
        base['problems'].append(
            'the subset lattice is incomplete; missing cell(s): '
            + ', '.join(repr(x) for x in base['missing_subsets']))
    if extra:
        base['problems'].append(
            'the lattice contains subset(s) outside the declared components: '
            + ', '.join(repr(_al_name(s, comps)) for s in extra))
    if not common:
        base['problems'].append(
            'no seed is measured in every cell; a per-seed Shapley value is '
            'undefined and no cross-cell imputation is allowed')
    elif len(common) < _min_n:
        base['problems'].append(
            f'only {len(common)} common seed(s) across all cells; '
            f'min_n={_min_n} is required')
    if missing_by_cell and common:
        base['warnings'].append(
            'cell(s) are missing some seeds; the effects are computed on the '
            f'{len(common)}-seed complete-case intersection, not on each '
            "cell's own seed set: " + ', '.join(
                f'{name}: {miss}' for name, miss in sorted(missing_by_cell.items())))
    if base['problems']:
        base['status'] = 'not_estimable'
        return base
    # ---- per-seed lattice values ----------------------------------------
    seed_values = {}
    for seed in common:
        vals = {}
        for subset in expected:
            raw = norm[subset][seed]
            vals[subset] = (_al_math.log(raw) if scale == 'ppl' else raw)
        seed_values[seed] = vals
    empty = frozenset()
    full = frozenset(comps)

    # ---- total effect, Shapley, LOO -------------------------------------
    total_vec = [seed_values[s][full] - seed_values[s][empty] for s in common]
    base['total_effect'] = _al_stats(total_vec, common, alpha=_alpha,
                                     min_n=_min_n)
    phi_vecs = {c: [] for c in comps}
    loo_vecs = {c: [] for c in comps}
    for seed in common:
        vals = seed_values[seed]
        phi = _al_shapley(vals, comps)
        for comp in comps:
            phi_vecs[comp].append(phi[comp])
            loo_vecs[comp].append(vals[full] - vals[full - {comp}])
    phi_mean, loo_mean = {}, {}
    for comp in comps:
        phi_st = _al_stats(phi_vecs[comp], common, alpha=_alpha, min_n=_min_n)
        loo_st = _al_stats(loo_vecs[comp], common, alpha=_alpha, min_n=_min_n)
        phi_st['per_seed'] = {str(s): _al_num(phi_vecs[comp][i])
                              for i, s in enumerate(common)}
        loo_st['per_seed'] = {str(s): _al_num(loo_vecs[comp][i])
                              for i, s in enumerate(common)}
        base['shapley'][comp] = phi_st
        base['loo'][comp] = loo_st
        phi_mean[comp] = phi_st['mean'] or 0.0
        loo_mean[comp] = loo_st['mean'] or 0.0
    # ---- marginal context dependence ------------------------------------
    for comp in comps:
        contexts = []
        others = [x for x in comps if x != comp]
        for size in range(len(others) + 1):
            for combo in _al_itertools.combinations(others, size):
                contexts.append(frozenset(combo))
        by_context = []
        seed_means = {s: [] for s in common}
        for ctx in contexts:
            vals = []
            for seed in common:
                vals.append(seed_values[seed][ctx | {comp}]
                            - seed_values[seed][ctx])
                seed_means[seed].append(vals[-1])
            mean = _al_num(sum(vals) / len(vals)) if vals else None
            by_context.append({
                'context': [c for c in comps if c in ctx],
                'mean': mean,
                'n': len(vals),
            })
        means = [row['mean'] for row in by_context
                 if row['mean'] is not None]
        lo = min(means) if means else None
        hi = max(means) if means else None
        seed_sign_change = False
        for seed in common:
            sv = seed_means[seed]
            if any(x > _tol for x in sv) and any(x < -_tol for x in sv):
                seed_sign_change = True
                break
        base['marginals'][comp] = {
            'n_contexts': len(contexts),
            'min': lo,
            'max': hi,
            'range': (hi - lo) if (lo is not None and hi is not None) else None,
            'context_dependent': bool(
                (lo is not None and hi is not None
                 and lo < -_tol and hi > _tol) or seed_sign_change),
            'seed_sign_change': seed_sign_change,
            'by_context': by_context,
        }
    # ---- pairwise interactions at every context -------------------------
    for i, comp_i in enumerate(comps):
        for comp_j in comps[i + 1:]:
            others = [x for x in comps if x not in (comp_i, comp_j)]
            contexts = []
            for size in range(len(others) + 1):
                for combo in _al_itertools.combinations(others, size):
                    contexts.append(frozenset(combo))
            by_context = []
            seed_vals = {s: [] for s in common}
            for ctx in contexts:
                vals = []
                for seed in common:
                    v = seed_values[seed]
                    vals.append(v[ctx | {comp_i, comp_j}] - v[ctx | {comp_i}]
                                - v[ctx | {comp_j}] + v[ctx])
                    seed_vals[seed].append(vals[-1])
                by_context.append({
                    'context': [c for c in comps if c in ctx],
                    'value': _al_num(sum(vals) / len(vals)) if vals else None,
                    'n': len(vals),
                })
            means = [row['value'] for row in by_context
                     if row['value'] is not None]
            at_empty_vec = [seed_values[s][frozenset({comp_i, comp_j})]
                            - seed_values[s][frozenset({comp_i})]
                            - seed_values[s][frozenset({comp_j})]
                            + seed_values[s][empty] for s in common]
            empty_full = frozenset(comps) - {comp_i, comp_j}
            at_full_vec = [seed_values[s][full]
                           - seed_values[s][empty_full | {comp_i}]
                           - seed_values[s][empty_full | {comp_j}]
                           + seed_values[s][empty_full] for s in common]
            sign_consistent = bool(
                means and (all(x > _tol for x in means)
                           or all(x < -_tol for x in means)
                           or all(abs(x) <= _tol for x in means)))
            seed_sign_consistent = True
            for seed in common:
                sv = seed_vals[seed]
                if any(x > _tol for x in sv) and any(x < -_tol for x in sv):
                    seed_sign_consistent = False
                    break
            base['interactions'].append({
                'pair': [comp_i, comp_j],
                'at_empty': _al_stats(at_empty_vec, common, alpha=_alpha,
                                      min_n=_min_n),
                'at_full_minus_pair': _al_stats(at_full_vec, common,
                                                alpha=_alpha, min_n=_min_n),
                'range': (max(means) - min(means)) if means else None,
                'sign_consistent': sign_consistent,
                'seed_sign_consistent': seed_sign_consistent,
                'by_context': by_context,
            })
    # ---- order-2+ Möbius remainder --------------------------------------
    main_vecs = {c: [seed_values[s][frozenset({c})]
                     - seed_values[s][empty] for s in common] for c in comps}
    pair_empty = {}
    for i, comp_i in enumerate(comps):
        for comp_j in comps[i + 1:]:
            pair_empty[f'{comp_i}|{comp_j}'] = [
                seed_values[s][frozenset({comp_i, comp_j})]
                - seed_values[s][frozenset({comp_i})]
                - seed_values[s][frozenset({comp_j})]
                + seed_values[s][empty] for s in common]
    higher_vec = []
    for idx, seed in enumerate(common):
        total = total_vec[idx]
        lower = sum(main_vecs[c][idx] for c in comps) + sum(
            vals[idx] for vals in pair_empty.values())
        higher_vec.append(total - lower)
    base['higher_order'] = _al_stats(higher_vec, common, alpha=_alpha,
                                     min_n=_min_n)
    base['decomposition'] = {
        'main_effects_at_empty': {
            c: _al_stats(main_vecs[c], common, alpha=_alpha, min_n=_min_n)
            for c in comps},
        'pair_interactions_at_empty': {
            key: _al_stats(vals, common, alpha=_alpha, min_n=_min_n)
            for key, vals in pair_empty.items()},
        'higher_order_at_empty': base['higher_order'],
        'identity': ('total = sum(main effects at empty) + '
                     'sum(pair interactions at empty) + higher-order remainder'),
    }
    # ---- axioms and rank agreement --------------------------------------
    sum_phi = sum(phi_mean.values())
    total_mean = base['total_effect'].get('mean')
    gap = (sum_phi - total_mean) if total_mean is not None else None
    base['axioms'] = {
        'efficiency': {
            'sum_shapley': _al_num(sum_phi),
            'total_effect': total_mean,
            'gap': _al_num(gap),
            'pass': bool(gap is not None
                         and abs(gap) <= _tol * max(1.0, abs(total_mean))),
        },
        'dummy': {},
        'symmetry': {'symmetric_pairs': [], 'violations': []},
    }
    for comp in comps:
        all_marg = [row['mean'] for row in base['marginals'][comp]['by_context']
                    if row['mean'] is not None]
        dummy = bool(all_marg and all(abs(x) <= _tol for x in all_marg))
        phi_st = base['shapley'][comp]
        phi_val = phi_st.get('mean')
        base['axioms']['dummy'][comp] = {
            'all_marginals_zero': dummy,
            'shapley': phi_val,
            'pass': (not dummy) or (phi_val is not None
                                    and abs(phi_val) <= _tol),
        }
    for i, comp_i in enumerate(comps):
        for comp_j in comps[i + 1:]:
            others = [x for x in comps if x not in (comp_i, comp_j)]
            symmetric = True
            for size in range(len(others) + 1):
                for combo in _al_itertools.combinations(others, size):
                    ctx = frozenset(combo)
                    for seed in common:
                        v = seed_values[seed]
                        if abs(v[ctx | {comp_i}] - v[ctx | {comp_j}]) > _tol * max(
                                1.0, abs(v[ctx | {comp_i}]),
                                abs(v[ctx | {comp_j}])):
                            symmetric = False
                            break
                    if not symmetric:
                        break
                if not symmetric:
                    break
            if symmetric:
                base['axioms']['symmetry']['symmetric_pairs'].append(
                    [comp_i, comp_j])
                if abs((base['shapley'][comp_i]['mean'] or 0.0)
                       - (base['shapley'][comp_j]['mean'] or 0.0)) > _tol:
                    base['axioms']['symmetry']['violations'].append(
                        [comp_i, comp_j])
    phi_ranks = _al_rank([base['shapley'][c]['mean'] or 0.0 for c in comps])
    loo_ranks = _al_rank([base['loo'][c]['mean'] or 0.0 for c in comps])
    for idx, comp in enumerate(comps):
        base['shapley'][comp]['rank'] = phi_ranks[idx]
        base['loo'][comp]['rank'] = loo_ranks[idx]
    flips = []
    for i, comp_i in enumerate(comps):
        for comp_j in comps[i + 1:]:
            s_i = base['shapley'][comp_i]['mean'] or 0.0
            s_j = base['shapley'][comp_j]['mean'] or 0.0
            l_i = base['loo'][comp_i]['mean'] or 0.0
            l_j = base['loo'][comp_j]['mean'] or 0.0
            if (abs(s_i - s_j) <= _tol or abs(l_i - l_j) <= _tol):
                # A tie is not an ordering disagreement.
                continue
            if ((s_i < s_j) != (l_i < l_j)):
                flips.append([comp_i, comp_j])
    base['rank_agreement'] = {
        'shapley_ranks': {c: base['shapley'][c]['rank'] for c in comps},
        'loo_ranks': {c: base['loo'][c]['rank'] for c in comps},
        'spearman': _al_spearman(
            [base['shapley'][c]['mean'] or 0.0 for c in comps],
            [base['loo'][c]['mean'] or 0.0 for c in comps]),
        'order_flips': flips,
        'n_order_flips': len(flips),
        'note': ('Shapley averages marginals over contexts; LOO is the '
                 'removal marginal at the full context. A rank flip is a '
                 'context-dependent component, not an implementation bug.'),
    }
    context_dep = [c for c in comps if base['marginals'][c]['context_dependent']]
    if context_dep:
        base['warnings'].append(
            'marginal effect changes sign across contexts for: '
            + ', '.join(context_dep)
            + '; a single LOO number is not a context-free importance score')
    # The axioms are exact identities of the estimator.  A violation is an
    # implementation/numerical failure, not a surprising scientific result;
    # surface it as a failed analysis instead of a green artifact.
    if not base['axioms']['efficiency']['pass']:
        base['problems'].append(
            'efficiency axiom failed: sum(Shapley) does not equal the total '
            'effect')
    for comp, row in base['axioms']['dummy'].items():
        if not row['pass']:
            base['problems'].append(
                f'dummy axiom failed for {comp!r}: all marginals are zero but '
                'its Shapley value is not')
    if base['axioms']['symmetry']['violations']:
        base['problems'].append(
            'symmetry axiom failed for pair(s): '
            + ', '.join('+'.join(pair) for pair in
                        base['axioms']['symmetry']['violations']))
    if base['problems']:
        base['status'] = 'fail'
    else:
        base['status'] = 'warnings' if base['warnings'] else 'ok'
    return base


def ablation_lattice_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    text = _al_json.dumps(body, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False)
    return _al_hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_ablation_lattice(cells, *, components=None, scale='ppl', min_n=3,
                           alpha=0.05,
                           max_components=AL_LATTICE_MAX_COMPONENTS,
                           created_at=None):
    """Run the attribution and stamp its canonical content hash.

    ``inputs.cells`` is stored as a serialisable record list (JSON object keys
    cannot be subsets), so the artifact can be replayed without trusting any
    derived field.
    """
    out = shapley_lattice_attribution(
        cells, components=components, scale=scale, min_n=min_n, alpha=alpha,
        max_components=max_components)
    # Record exactly the inputs the replay needs.  Keep the pre-normalised
    # values so a replay cannot silently change a PPL through string parsing.
    comps = list(out['components'])
    norm = _al_normalise_cells(cells, comps if components is not None else None,
                               scale=scale)
    out['inputs'] = {
        'scale': scale,
        'components': comps,
        'min_n': int(out['min_n']),
        'alpha': float(out['alpha']),
        'max_components': int(out['max_components']),
        'cells': _al_cells_to_records(norm, comps),
    }
    out['created_at'] = created_at or _al_time.strftime('%Y-%m-%dT%H:%M:%S')
    out['content_sha256'] = ablation_lattice_content_sha(out)
    return out


def _al_equiv(a, b, *, tol=1e-9):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(
            1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_al_equiv(a[k], b[k], tol=tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _al_equiv(x, y, tol=tol) for x, y in zip(a, b))
    return a == b


def validate_ablation_lattice(artifact):
    """Recompute a stored lattice artifact from its recorded cells.

    Structural validity is separate from the analysis verdict: a valid
    ``not_estimable`` artifact (the design is incomplete) replays cleanly and
    is returned with ``replayed_status='not_estimable'``.  A changed cell,
    seed, Shapley value, interaction, axiom block or status is a ``fail``.
    """
    out = {'kind': 'ablation_lattice_validation', 'status': 'ok',
           'problems': [], 'warnings': [], 'replayed': False,
           'replayed_status': None}
    if not isinstance(artifact, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'artifact must be an object, got {type(artifact).__name__}')
        return out
    if artifact.get('kind') != AL_LATTICE_KIND:
        out['problems'].append(
            f"artifact.kind must be {AL_LATTICE_KIND!r}")
    if artifact.get('schema') != AL_LATTICE_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {AL_LATTICE_SCHEMA}')
    if artifact.get('status') not in ('ok', 'warnings', 'not_estimable',
                                      'fail'):
        out['problems'].append(
            f"artifact.status={artifact.get('status')!r} is not recognised")
    if artifact.get('policy') != AL_LATTICE_POLICY:
        out['problems'].append(
            'artifact.policy does not match this implementation')
    for field in ('problems', 'warnings'):
        value = artifact.get(field)
        if value is not None and not isinstance(value, (list, tuple)):
            out['problems'].append(
                f'artifact.{field} must be a list, got {type(value).__name__}')
    recorded = artifact.get('content_sha256')
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append(
            'content_sha256 is missing or not a 64-hex digest')
    else:
        try:
            want = ablation_lattice_content_sha(artifact)
        except (TypeError, ValueError, RecursionError) as exc:
            out['problems'].append(
                f'artifact body is not canonical JSON: '
                f'{type(exc).__name__}: {exc}')
        else:
            if recorded != want:
                out['problems'].append(
                    'content_sha256 does not reproduce from the artifact body')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs is missing or not an object')
        out['status'] = 'fail'
        return out
    try:
        raw_components = inputs.get('components')
        if isinstance(raw_components, (list, tuple)) and len(raw_components) == 0:
            components = []
        else:
            components = _al_components(raw_components)
        scale = inputs.get('scale')
        if scale not in ('ppl', 'log_ppl'):
            raise ValueError("inputs.scale must be 'ppl' or 'log_ppl'")
        cells = _al_cells_from_records(inputs.get('cells'), components,
                                       scale=scale)
        replay = shapley_lattice_attribution(
            cells, components=components, scale=scale,
            min_n=inputs.get('min_n'), alpha=inputs.get('alpha'),
            max_components=inputs.get('max_components',
                                      AL_LATTICE_MAX_COMPONENTS))
    except (TypeError, ValueError, KeyError) as exc:
        out['problems'].append(
            f'artifact.inputs cannot be replayed: {type(exc).__name__}: {exc}')
        out['status'] = 'fail'
        return out
    out['replayed'] = True
    out['replayed_status'] = replay.get('status')
    # Compare every derived block.  `inputs`, `created_at` and the hash are
    # provenance, not estimates; all other keys are replayable.
    skip = {'kind', 'schema', 'policy', 'inputs', 'created_at',
            'content_sha256', 'status', 'problems', 'warnings'}
    for key in sorted(set(replay) - skip):
        if not _al_equiv(artifact.get(key), replay.get(key)):
            out['problems'].append(
                f'artifact.{key} does not replay from its recorded cells')
    if artifact.get('status') != replay.get('status'):
        out['problems'].append(
            f"artifact.status={artifact.get('status')!r} does not match "
            f"the replayed status={replay.get('status')!r}")
    if not _al_equiv(artifact.get('problems'), replay.get('problems'),
                     tol=0.0):
        out['problems'].append('artifact.problems does not replay')
    if not _al_equiv(artifact.get('warnings'), replay.get('warnings'),
                     tol=0.0):
        out['problems'].append('artifact.warnings does not replay')
    if out['problems']:
        out['status'] = 'fail'
    return out


def write_ablation_lattice(outdir, artifact, *, overwrite=False):
    """Atomically persist ``ablation_lattice.json``; refuse silent replacement."""
    if isinstance(outdir, _al_os.PathLike):
        outdir = _al_os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_ablation_lattice(artifact)
    path = _al_os.path.join(outdir, 'ablation_lattice.json')
    out = {'kind': 'write_ablation_lattice', 'path': path, 'written': False,
           'unchanged': False, 'conflict': False, 'status': verdict['status'],
           'artifact_status': (artifact.get('status')
                               if isinstance(artifact, dict) else None),
           'problems': list(verdict['problems'])}
    if verdict['status'] != 'ok':
        return out
    if _al_os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                old = _al_json.load(fh)
        except Exception:
            old = None
        if (isinstance(old, dict)
                and old.get('content_sha256') == artifact.get('content_sha256')):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different ablation_lattice.json already exists; pass '
                'overwrite=True only after re-running the analysis')
            return out
    _al_os.makedirs(outdir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        _al_json.dump(artifact, fh, ensure_ascii=False, indent=1,
                      allow_nan=False)
        fh.write('\n')
    _al_os.replace(tmp, path)
    out['written'] = True
    return out


# ---- design side: which scopes are complete lattices, which claims licensed -
def ablation_lattice_design(components=None, *, arms_by_subset=None,
                            design=None, factorials=None, present=None,
                            n_seeds=5,
                            max_components=AL_LATTICE_MAX_COMPONENTS):
    """Audit the declared ablation design as a set of lattice scopes.

    Two modes are supported:

    * canonical (no ``components``): each declared 2x2 factorial is treated as
      a two-component lattice.  The audit reports whether its four cells are
      distinct declared arms present in the run panel, whether a pair-scope
      Shapley is therefore identifiable, and the structural limits of the
      leave-one-out family (it is a star, not a joint lattice).
    * proposed (``components`` + ``arms_by_subset``): report the exact missing
      subsets and the additional arms/cells a full 2^k lattice would require.
      A design is allowed to be incomplete; the point is to price and name the
      gap instead of silently treating a star as attribution.
    """
    _max_k = _al_int(max_components, where='max_components', minimum=1)
    _n_seeds = _al_int(n_seeds, where='n_seeds', minimum=1)
    out = {
        'kind': 'ablation_lattice_design',
        'schema': 1,
        'status': 'ok',
        'policy': dict(AL_LATTICE_POLICY),
        'mode': 'canonical',
        'max_components': int(_max_k),
        'n_seeds': int(_n_seeds),
        'scopes': {},
        'global': {},
        'licensed_claims': [
            'pairwise_interaction_within_a_complete_scope',
            'context_specific_loo_marginal',
        ],
        'unlicensed_claims': [
            'sum_of_leave_one_out_effects_as_component_importance',
            'cross_scope_shapley_without_a_joint_lattice',
            'global_component_importance_from_a_star_design',
        ],
        'problems': [],
        'warnings': [],
        'note': ('a star (baseline + one-component-removed) is not a subset '
                 'lattice; Shapley attribution requires every 2^k cell'),
    }
    if present is None:
        run = globals().get('RUN_ABL')
        present = set(run.get('variants', ())) if isinstance(run, dict) else set()
    else:
        try:
            present = set(present)
        except TypeError:
            out['status'] = 'fail'
            out['problems'].append('present must be an iterable of arm ids')
            return out
    if components is not None:
        comps = _al_components(components)
        if arms_by_subset is None:
            out['status'] = 'fail'
            out['problems'].append(
                'a proposed scope needs arms_by_subset (subset -> arm)')
            return out
        out['mode'] = 'proposed'
        if len(comps) > _max_k:
            out['status'] = 'not_estimable'
            out['problems'].append(
                f'{len(comps)} components exceed max_components={_max_k}')
            out['global'] = {'components': comps, 'n_components': len(comps),
                             'full_lattice_cells': 2 ** len(comps),
                             'shapley_identifiable': False}
            return out
        if not isinstance(arms_by_subset, dict):
            out['status'] = 'fail'
            out['problems'].append('arms_by_subset must be a mapping')
            return out
        expected = _al_all_subsets(comps)
        normal = {}
        for key, arm in arms_by_subset.items():
            parts = _al_split_key(key)
            subset = frozenset(_al_component(p, where='arms_by_subset')
                               for p in parts)
            if subset not in expected:
                out['problems'].append(
                    f'arms_by_subset[{key!r}]: not a subset of the declared '
                    f'components {comps}')
                continue
            try:
                arm = _al_arm(arm, where=f'arms_by_subset[{key!r}]')
            except ValueError as exc:
                out['problems'].append(str(exc))
                continue
            if subset in normal:
                out['problems'].append(
                    f'arms_by_subset: duplicate subset {sorted(subset)!r}')
                continue
            normal[subset] = arm
        used = {}
        for subset, arm in normal.items():
            if arm in used:
                out['problems'].append(
                    f'arms_by_subset: arm {arm!r} is assigned to both '
                    f'{sorted(used[arm])!r} and {sorted(subset)!r}')
            used[arm] = subset
        missing = [s for s in expected if s not in normal]
        extra = [s for s in normal if s not in expected]
        if extra:
            out['problems'].append(
                'arms_by_subset contains subset(s) outside the declared '
                'components')
        missing_names = [_al_name(s, comps) for s in missing]
        present_missing = sorted({normal[s] for s in normal
                                  if normal[s] not in present})
        identifiable = ((not missing) and (not present_missing)
                        and not out['problems'])
        if present_missing:
            out['warnings'].append(
                'arm(s) declared in the proposed lattice are absent from the '
                'run panel: ' + ', '.join(present_missing))
        out['scopes']['proposed'] = {
            'components': comps,
            'n_components': len(comps),
            'arms_by_subset': {
                _al_name(s, comps): normal[s] for s in expected if s in normal},
            'n_cells_expected': len(expected),
            'n_cells_present': len(normal),
            'missing_subsets': missing_names,
            'missing_arms_from_run_panel': present_missing,
            'complete': not missing,
            'measurable': identifiable,
            'shapley_identifiable': identifiable,
            'completion': {
                'additional_arms': len(missing),
                'additional_existing_arms_to_run': len(present_missing),
                'additional_cells': (len(missing) + len(present_missing))
                                    * _n_seeds,
                'n_seeds': _n_seeds,
                'missing_subsets': missing_names,
                'missing_arms_from_run_panel': present_missing,
                'note': ('one new arm per missing subset, plus the declared '
                         'arms not yet in the run panel; each must be run on '
                         'the same n_seeds to keep the lattice complete-case'),
            },
        }
        out['global'] = {
            'components': comps,
            'n_components': len(comps),
            'full_lattice_cells': len(expected),
            'n_cells_present': len(normal),
            'n_cells_missing': len(missing),
            'shapley_identifiable': identifiable,
            'additional_cells_at_n_seeds': (len(missing) + len(present_missing))
                                           * _n_seeds,
        }
        out['licensed_claims'] = (['pairwise_interaction_within_a_complete_scope',
                                   'context_specific_loo_marginal']
                                  + (['within_scope_shapley'] if identifiable
                                     else []))
        if out['problems']:
            out['status'] = 'fail'
        elif not identifiable:
            out['status'] = 'incomplete'
        return out
    # ---- canonical mode: declared 2x2 factorials ------------------------
    out['mode'] = 'canonical'
    factorials = (globals().get('FACTORIAL_DESIGN')
                  if factorials is None else factorials)
    design = (globals().get('ABLATION_DESIGN')
              if design is None else design)
    if not isinstance(factorials, dict) or not factorials:
        out['status'] = 'not_applicable'
        out['note'] = 'no 2x2 factorial scopes are declared'
        return out
    cell_keys = globals().get('_FACTORIAL_CELL_KEYS',
                              (('control', 'control'), ('active', 'control'),
                               ('control', 'active'), ('active', 'active')))
    scope_components = {}
    for name, spec in factorials.items():
        if not isinstance(spec, dict):
            out['problems'].append(f'{name}: factorial declaration is not an object')
            continue
        factors = spec.get('factors')
        cells = spec.get('cells')
        if (not isinstance(factors, (list, tuple)) or len(factors) != 2
                or not isinstance(cells, dict)):
            out['problems'].append(
                f'{name}: a two-component lattice needs factors[2] and cells')
            continue
        try:
            f1 = _al_component(factors[0], where=f'{name}.factors[0]')
            f2 = _al_component(factors[1], where=f'{name}.factors[1]')
        except ValueError as exc:
            out['problems'].append(f'{name}: {exc}')
            continue
        if f1 == f2:
            out['problems'].append(f'{name}: the two factors are identical')
            continue
        arms = [cells.get(key) for key in cell_keys]
        if any(not isinstance(a, str) or not a.strip() for a in arms):
            out['problems'].append(
                f'{name}: the four cells must map to non-empty arm ids')
            continue
        arms = [a.strip() for a in arms]
        if len(set(arms)) != 4:
            out['problems'].append(f'{name}: the four cells are not distinct arms')
            continue
        by_subset = {
            frozenset(): arms[0],
            frozenset({f1}): arms[1],
            frozenset({f2}): arms[2],
            frozenset({f1, f2}): arms[3],
        }
        missing_arms = [a for a in arms if a not in present]
        scope_components[name] = {f1, f2}
        out['scopes'][str(name)] = {
            'components': [f1, f2],
            'n_components': 2,
            'cells': {_al_name(s, [f1, f2]): a for s, a in by_subset.items()},
            'n_cells_expected': 4,
            'n_cells_present': 4,
            'missing_subsets': [],
            'missing_arms_from_run_panel': missing_arms,
            'complete': not missing_arms,
            'shapley_identifiable': not missing_arms,
        }
        if missing_arms:
            out['warnings'].append(
                f'{name}: arm(s) absent from the run panel: '
                + ', '.join(missing_arms)
                + '; the 2x2 lattice is declared but not measurable')
    if out['problems']:
        out['status'] = 'fail'
        return out
    if not out['scopes']:
        out['status'] = 'not_applicable'
        return out
    all_components = sorted(set().union(*scope_components.values()))
    counts = {}
    for comps in scope_components.values():
        for comp in comps:
            counts[comp] = counts.get(comp, 0) + 1
    overlap = {c: n for c, n in sorted(counts.items()) if n > 1}
    out['global'] = {
        'components': all_components,
        'n_components': len(all_components),
        'n_scopes': len(out['scopes']),
        'scope_overlap_counts': overlap,
        'join_lattice_cells': (2 ** len(all_components)
                               if len(all_components) <= _max_k else None),
        'shapley_identifiable': False,
        'reason': ('the canonical family is a set of overlapping 2x2 scopes, '
                   'not one joint subset lattice; cross-scope Shapley is not '
                   'identified without measuring every joint cell'),
    }
    out['licensed_claims'] = [
        'pairwise_interaction_within_a_complete_2x2_scope',
        'within_scope_two_component_shapley',
        'context_specific_loo_marginal',
    ]
    out['unlicensed_claims'] = [
        'sum_of_leave_one_out_effects_as_component_importance',
        'cross_scope_shapley_without_a_joint_lattice',
        'global_component_importance_from_a_star_design',
        'higher_order_interaction_beyond_a_declared_scope',
    ]
    # The star family is a structural fact, not a failure.  Report it so a
    # reader cannot mistake the LOO table for a Shapley decomposition.
    n_contrasts = 0
    if isinstance(design, dict):
        for pairs in design.values():
            if isinstance(pairs, (list, tuple)):
                n_contrasts += len(pairs)
    out['star_family'] = {
        'n_components': len(design) if isinstance(design, dict) else None,
        'n_contrasts': n_contrasts,
        'complete_lattice': False,
        'reason': ('leave-one-out contrasts measure each component only at the '
                   'full context; the 2^k subsets needed for Shapley are not '
                   'part of this family'),
    }
    out['status'] = 'warnings' if out['warnings'] else 'ok'
    return out


def factorial_lattice_attribution(rows, *, factors, arms, alpha=0.05,
                                  min_n=3):
    """Shapley/interaction block for one already-controlled 2x2 factorial.

    ``rows`` is the list returned by ``_factorial_condition_rows`` (one row
    per seed with ``ppl[arm]`` for all four arms).  The four cells are mapped
    to the empty / {factor1} / {factor2} / {factor1,factor2} lattice cells.
    This is the bridge that makes an existing 2x2 panel answer both questions:
    the interaction (does one factor still help when the other is active?) and
    the within-scope Shapley shares (which factor carries the joint effect?).
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('rows must be a non-empty list of 2x2 seed rows')
    if not isinstance(factors, (list, tuple)) or len(factors) != 2:
        raise ValueError('factors must be a two-item sequence')
    f1 = _al_component(factors[0], where='factors[0]')
    f2 = _al_component(factors[1], where='factors[1]')
    if f1 == f2:
        raise ValueError('factors must be distinct')
    if not isinstance(arms, (list, tuple)) or len(arms) != 4:
        raise ValueError('arms must be the four cell arm ids in '
                         'control/control, active/control, control/active, '
                         'active/active order')
    arm_ids = [_al_arm(a, where=f'arms[{i}]') for i, a in enumerate(arms)]
    if len(set(arm_ids)) != 4:
        raise ValueError('arms must be four distinct arm ids')
    cells = {frozenset(): {}, frozenset({f1}): {},
             frozenset({f2}): {}, frozenset({f1, f2}): {}}
    keys = [frozenset(), frozenset({f1}), frozenset({f2}), frozenset({f1, f2})]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('each row must be an object')
        seed = _as_seed_int(row.get('seed'))  # noqa: F821 (part_13)
        if seed is None:
            raise ValueError(f'row seed is not an integer: {row.get("seed")!r}')
        ppl = row.get('ppl')
        if not isinstance(ppl, dict):
            raise ValueError(f'row seed={seed} has no per-arm ppl mapping')
        for key, arm in zip(keys, arm_ids):
            if arm not in ppl:
                raise ValueError(
                    f'row seed={seed} has no PPL for arm {arm!r}')
            if seed in cells[key]:
                raise ValueError(
                    f'duplicate row for seed={seed} in the 2x2 factorial')
            cells[key][seed] = _al_finite(
                ppl[arm], where=f'row seed={seed} arm={arm}', positive=True)
    return shapley_lattice_attribution(
        cells, components=[f1, f2], scale='ppl', min_n=min_n, alpha=alpha)


def ablation_lattice_summary(artifact):
    """Compact block for reports/snapshots."""
    if not isinstance(artifact, dict):
        return {'kind': 'ablation_lattice_summary', 'status': 'missing',
                'ok': False}
    shap = artifact.get('shapley') if isinstance(artifact.get('shapley'), dict) else {}
    return {
        'kind': 'ablation_lattice_summary',
        'status': artifact.get('status'),
        'ok': artifact.get('status') in ('ok', 'warnings'),
        'n_components': artifact.get('n_components'),
        'complete': artifact.get('complete'),
        'missing_subsets': artifact.get('missing_subsets'),
        'n_common_seeds': (artifact.get('seed_coverage') or {}).get(
            'n_common_seeds'),
        'total_effect': (artifact.get('total_effect') or {}).get('mean'),
        'shapley': {c: (v or {}).get('mean') for c, v in shap.items()},
        'rank_agreement_spearman': (artifact.get('rank_agreement') or {}).get(
            'spearman'),
        'content_sha256': artifact.get('content_sha256'),
    }
