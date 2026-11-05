# ---- 消融归因的联合推断：Shapley 排名不是「每个分量各自一张 CI」 --------
#
# §136 给出了完整子集晶格上的精确 Shapley/LOO/交互归因，并对每个分量分别
# 报告 exact sign-flip p 与正态 CI。但「分量 i 比 j 更重要」是一个**分量
# 之间的比较**，不是「i 的 Shapley 是否非零」；把 k 张独立的 95% CI 并排
# 展示再按点估计排名，会同时犯两个错误：
#
#   * 多重比较：k 个分量、k(k-1)/2 个两两顺序，每个都用 alpha 判读；
#   * 相关性：所有 Shapley 值来自同一批 lattice cell 与同一批种子，
#     独立 CI 既不是联合区间，也不能直接用于顺序判定。
#
# 本片段补上归因层的联合推断：从完整晶格逐种子 Shapley 向量出发，用精确
# sign-flip max-T（Westfall-Young）给出分量的同时区间与族校正 p；对每个
# 分量对做逐种子差值检验，并用同一族校正决定「顺序是否被分辨」。只有在
# 所有相邻顺序都被分辨时才许可完整排名；只有 top-1 分量与所有其他分量
# 都被分辨时才许可「最重要分量」这一句话。
#
# 另一个容易忽略的事实：Shapley 值对值函数的单调变换**不是不变的**。
# 同一组 lattice cell，在 log PPL 与 raw PPL 上可以给出不同的分量顺序。
# 因此本片段同时报告两种尺度，把 raw PPL 作为敏感性读数，绝不把两者
# 平均成第三个「真值」；主尺度仍是 log PPL（相对/比值尺度），但结论
# 必须显式携带 `scale_sensitivity`。
#
# SCOPE: 这是**探索性归因读数**。它不新增训练、不读取任何 split、不重算
# PPL、不产生确认 p 值、不进入冻结 Holm 族，也不改变 `verified`/
# `confirmed`。它只回答「在这一个完整晶格、这一批种子、这两个尺度上，
# 分量顺序是否被数据分辨」。

import math as _arb_math
import time as _arb_time


AR_KIND = 'ablation_ranking'
AR_RESULT_KIND = 'ablation_ranking_result'
AR_SCHEMA = 1
AR_MAX_COMPONENTS = AL_LATTICE_MAX_COMPONENTS  # noqa: F821 (part_73)
AR_TOL = AL_LATTICE_TOL  # noqa: F821 (part_73)

AR_POLICY = {
    'version': 'ablation-ranking-1',
    'estimand': ('exact Shapley values of the declared components on the '
                 'complete subset lattice, plus their pairwise ordering'),
    'value_function': ('positive PPL per lattice cell; log PPL is the primary '
                       'relative scale, raw PPL is reported as a sensitivity '
                       'scale'),
    'sign': ('ascending Shapley; a more negative value means the component '
             'lowers PPL more, so rank 1 is the most beneficial component'),
    'component_family': ('all declared components, corrected jointly with '
                         'exact sign-flip max-T (Westfall-Young) when the '
                         'seed count is enumerable; Holm is a fail-safe '
                         'fallback, never a silent replacement'),
    'ordering_family': ('all unordered component pairs; a pair is separated '
                        'only when the family-adjusted two-sided p is <= '
                        'alpha and the mean difference has the claimed sign'),
    'ranking_rule': ('full ranking is licensed only when every pair is '
                     'separated and the dominance graph is acyclic; top-1 is '
                     'licensed only when one component is separated from '
                     'every other component'),
    'scale_rule': ('Shapley is not invariant under monotone transforms of the '
                   'value function; log PPL and raw PPL are reported side by '
                   'side and are never averaged or treated as two independent '
                   'tests'),
    'resolution_rule': ('the exact two-sided sign-flip floor is 2/2**n; if '
                        'that floor exceeds alpha no pair can be separated at '
                        'this seed count'),
    'claim_scope': ('within one complete declared lattice; never across '
                    'scopes, never causal, never a substitute for a '
                    'pre-registered contrast'),
    'frozen_decision_changed': False,
}

_AR_SCALES = ('log_ppl', 'ppl')
_AR_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'verified', 'verdict',
    'frozen_decision', 'confirmation_verdict', 'p_holm_final',
})
_AR_INPUT_KEYS = ('cells', 'components', 'alpha', 'min_n', 'primary_scale',
                  'scales', 'max_components')


def _arb_repr(value, *, limit=120):
    """Short, exception-safe repr for diagnostics on untrusted values.

    Python 3.12 refuses to stringify an integer with more than ~4300 digits.
    Formatting such an integer inside a validation error would replace the
    intended ``ValueError``/failure record with a second
    ``ValueError: Exceeds the limit ...``, turning a reported tamper into an
    uncaught crash.
    """
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _arb_p_floor(n_seeds):
    """Exact two-sided sign-flip floor ``2 / 2**n`` without int->float blowup.

    ``1 << n`` followed by true division raises ``OverflowError`` once n is
    above the float exponent range (n > 1023), even though the mathematical
    floor is simply 0.0.  The panel can legitimately grow to that many seeds
    when the same complete lattice is replayed at a larger scale, so compute
    the value directly in the float exponent domain.
    """
    n = int(n_seeds)
    if n <= 0:
        return None
    try:
        return _arb_math.ldexp(2.0, -n)
    except OverflowError:
        # ``len()`` cannot practically reach this, but an arbitrarily large
        # exponent must still mean "below the smallest representable float".
        return 0.0


def _arb_number(value, *, where, positive=False):
    """Finite JSON number for an analysis input; bool/None/str fail closed."""
    if hasattr(value, 'item') and not isinstance(value, (int, float)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bool, bytes, bytearray, str)) or value is None:
        raise ValueError(
            f'{where}: expected a finite number, got {_arb_repr(value)}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{where}: expected a finite number, got {_arb_repr(value)}'
        ) from None
    if not _arb_math.isfinite(out):
        raise ValueError(
            f'{where}: expected a finite number, got {_arb_repr(value)}')
    if positive and out <= 0.0:
        raise ValueError(
            f'{where}: expected a positive value, got {_arb_repr(value)}')
    return out


def _arb_int(value, *, where, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(
            f'{where}: expected an integer, got {_arb_repr(value)}')
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{where}: expected an integer, got {_arb_repr(value)}'
        ) from None
    try:
        if value != out:
            raise ValueError(
                f'{where}: expected an integer, got {_arb_repr(value)}')
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{where}: expected an integer, got {_arb_repr(value)}'
        ) from None
    try:
        # An integer that Python 3.12 cannot stringify cannot be written as a
        # JSON number (the writer uses ``allow_nan=False`` and the default
        # digit limit).  Reject it at the input boundary with a clear error
        # instead of failing later while hashing/writing the artifact.  A
        # merely large but representable value (e.g. 10**400) stays valid.
        str(out)
    except (ValueError, MemoryError):
        raise ValueError(
            f'{where}: integer is too large to represent as JSON') from None
    if minimum is not None and out < minimum:
        raise ValueError(
            f'{where}: expected >= {minimum}, got {_arb_repr(value)}')
    if maximum is not None and out > maximum:
        raise ValueError(
            f'{where}: expected <= {maximum}, got {_arb_repr(value)}')
    return out


def _arb_scale(value, *, where='scale'):
    if not isinstance(value, str) or value not in _AR_SCALES:
        raise ValueError(
            f'{where} must be one of {_AR_SCALES}, got {_arb_repr(value)}')
    return value


def _arb_scales(value, *, primary):
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        raise ValueError('scales must be an iterable of scale names')
    try:
        raw = list(value)
    except TypeError:
        raise ValueError('scales must be an iterable of scale names') from None
    if not raw:
        raise ValueError('scales must contain at least the primary scale')
    out = []
    for i, item in enumerate(raw):
        scale = _arb_scale(item, where=f'scales[{i}]')
        if scale not in out:
            out.append(scale)
    if _arb_scale(primary, where='primary_scale') not in out:
        raise ValueError('primary_scale must be one of scales')
    return tuple(out)


def _arb_forbidden_payload(obj):
    """Collect forbidden decision fields anywhere in an untrusted payload.

    Iterative (rather than recursive) so a deeply nested or self-referential
    artifact is reported as a tamper instead of raising ``RecursionError``
    before validation can return a failure object.  A ``seen`` set also keeps
    a cyclic in-memory mapping from looping forever.
    """
    found = []
    stack = [obj]
    seen = set()
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            for key, value in cur.items():
                if isinstance(key, str) and key in _AR_FORBIDDEN_KEYS:
                    found.append(key)
                stack.append(value)
        elif isinstance(cur, (list, tuple)):
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            stack.extend(cur)
    return found


def _arb_order(mean_i, mean_j, *, tol):
    """Ordering from point means; ties are explicit, never rounded away."""
    diff = float(mean_i) - float(mean_j)
    if diff < -tol * max(1.0, abs(float(mean_i)), abs(float(mean_j))):
        return -1
    if diff > tol * max(1.0, abs(float(mean_i)), abs(float(mean_j))):
        return 1
    return 0


def _arb_pair_name(a, b):
    return f'{a}|{b}'


def _arb_joint(vectors, *, alpha):
    """Joint family inference with exact max-T, plus an explicit Holm reading.

    ``vectors`` is ``{name: [per-seed delta]}``.  The exact max-T route gives
    simultaneous intervals; a degenerate row (constant non-zero) has no finite
    studentised max-T null, so the routine returns a named Holm fallback rather
    than hiding the failure.  Holm is always computed as well, so a reader can
    see whether the two family procedures agree instead of trusting one label.
    """
    names = list(vectors)
    raw_p = {}
    for name in names:
        st = paired_effect_stats(  # noqa: F821 (part_10)
            list(vectors[name]), alpha=alpha, min_n=1)
        raw_p[name] = st.get('p_exact_signflip')
    holm = holm_adjust([(name, raw_p[name]) for name in names],  # noqa: F821
                       alpha=alpha)
    holm_rows = {}
    for name in names:
        rec = dict(holm.get(name) or {})
        holm_rows[name] = {
            'p_holm': rec.get('p_holm'),
            'reject_holm': bool(rec.get('reject')),
            'rank_holm': rec.get('rank'),
        }
    try:
        block = maxt_signflip_adjust(  # noqa: F821 (part_40)
            {name: list(vectors[name]) for name in names}, alpha=alpha)
        contrasts = {}
        for name, rec in (block.get('contrasts') or {}).items():
            row = dict(rec)
            row.update(holm_rows.get(name) or {})
            row.setdefault('p_raw_signflip', raw_p.get(name))
            contrasts[name] = row
        return {
            'method': 'exact_signflip_maxt',
            'fallback': False,
            'fallback_reason': None,
            'contrasts': contrasts,
            'critical_value': block.get('critical_value'),
            'ci_level_requested': block.get('ci_level_requested'),
            'ci_tail_level': block.get('ci_tail_level'),
            'ci_note': block.get('ci_note'),
            'p_floor_single': block.get('p_floor_single'),
            'n_patterns': block.get('n_patterns'),
        }
    except Exception as exc:  # degenerate/serialisation/alloc cap
        reason = f'{type(exc).__name__}: {exc}'
    contrasts = {}
    for name in names:
        rec = holm_rows[name]
        contrasts[name] = {
            'p_raw_signflip': raw_p.get(name),
            'p_maxt': None,
            'p_holm': rec.get('p_holm'),
            'reject_maxt': bool(rec.get('reject_holm')),
            'reject_holm': bool(rec.get('reject_holm')),
            'mean': None,
            'sd': None,
            'sem': None,
            'ci_simultaneous': None,
            'rank': rec.get('rank_holm'),
        }
    return {
        'method': 'holm_fallback',
        'fallback': True,
        'fallback_reason': reason,
        'contrasts': contrasts,
        'critical_value': None,
        'ci_level_requested': None,
        'ci_tail_level': None,
        'ci_note': ('exact max-T was not computable for this family; Holm '
                    'over the same declared family is reported instead'),
        'p_floor_single': None,
        'n_patterns': None,
    }


def _arb_component_stats(vectors, seeds, *, alpha, min_n, joint):
    out = {}
    for name, vec in vectors.items():
        st = _al_stats(vec, seeds, alpha=alpha, min_n=min_n)  # noqa: F821
        jr = (joint.get('contrasts') or {}).get(name) or {}
        out[name] = {
            'mean': st.get('mean'),
            'std': st.get('std'),
            'sem': st.get('sem'),
            'ci95_low': st.get('ci95_low'),
            'ci95_high': st.get('ci95_high'),
            'p_raw_signflip': st.get('p_exact_signflip'),
            'p_floor': st.get('p_floor'),
            'resolution_limited': st.get('resolution_limited'),
            'p_maxt_component': jr.get('p_maxt'),
            'p_holm_component': jr.get('p_holm'),
            'reject_component': bool(jr.get('reject_maxt')),
            'ci_simultaneous': jr.get('ci_simultaneous'),
        }
    return out


def _arb_add_ranks(stats, comps, *, tol):
    """Attach ascending point ranks (lower/more negative = more beneficial)."""
    values = [stats[c]['mean'] for c in comps]
    if any(v is None for v in values):
        return stats
    ranks = _al_rank(values, tol=tol)  # noqa: F821
    for i, c in enumerate(comps):
        stats[c]['point_rank'] = ranks[i]
    return stats


def _arb_ranking(comps, vectors, stats, *, alpha, tol):
    """Pairwise dominance with a family correction and rank bounds.

    The dominance graph uses only separated pairs.  A cycle is reported as an
    inconsistency (it can happen when pairwise tests disagree with the mean
    ordering), not silently turned into a total order.
    """
    pair_rows = []
    diff_vectors = {}
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            name = _arb_pair_name(a, b)
            diff = [float(x) - float(y) for x, y in zip(vectors[a], vectors[b])]
            diff_vectors[name] = diff
            st = paired_effect_stats(diff, alpha=alpha, min_n=1)  # noqa: F821
            pair_rows.append({
                'a': a,
                'b': b,
                'name': name,
                'mean_diff': st.get('mean'),
                'p_raw_signflip': st.get('p_exact_signflip'),
                'p_floor': st.get('p_floor'),
            })
    holm = holm_adjust(  # noqa: F821 (part_09)
        [(row['name'], row['p_raw_signflip']) for row in pair_rows],
        alpha=alpha)
    joint = _arb_joint(diff_vectors, alpha=alpha)
    joint_rows = joint.get('contrasts') or {}
    separated = {}
    for row in pair_rows:
        name = row['name']
        jr = joint_rows.get(name) or {}
        hr = holm.get(name) or {}
        p_holm = hr.get('p_holm')
        p_maxt = jr.get('p_maxt') if joint.get('fallback') is False else None
        p_primary = p_maxt if p_maxt is not None else p_holm
        mean_diff = row.get('mean_diff')
        order = 0
        if mean_diff is not None:
            order = _arb_order(stats[row['a']]['mean'], stats[row['b']]['mean'],
                              tol=tol)
        is_sep = bool(p_primary is not None and p_primary <= alpha
                      and order != 0)
        row_out = dict(row)
        point_better = (row['a'] if order < 0 else
                        row['b'] if order > 0 else None)
        row_out.update({
            'p_holm_pairwise': p_holm,
            'p_maxt_pairwise': p_maxt,
            'p_primary': p_primary,
            'separated': is_sep,
            'order': order,
            # `point_better` is the descriptive point-estimate order;
            # `better` is withheld unless the pair is family-separated, so a
            # reader cannot quote an unresolved point order as a ranking.
            'point_better': point_better,
            'better': point_better if is_sep else None,
            'method': ('maxt' if p_maxt is not None else 'holm'),
        })
        separated[name] = row_out

    # Dominance edges: winner -> loser, only for separated pairs.
    edges = set()
    for row in separated.values():
        if row['separated'] and row['better'] is not None:
            loser = row['b'] if row['better'] == row['a'] else row['a']
            edges.add((row['better'], loser))
    dominated_by = {c: 0 for c in comps}
    dominates = {c: 0 for c in comps}
    for winner, loser in edges:
        dominates[winner] += 1
        dominated_by[loser] += 1
    k = len(comps)
    rank_lower = {c: 1 + dominated_by[c] for c in comps}
    rank_upper = {c: k - dominates[c] for c in comps}
    unique = {c: rank_lower[c] == rank_upper[c] for c in comps}

    def _cyclic():
        # k <= 6; a depth-first cycle check on the dominance graph.
        color = {c: 0 for c in comps}
        adj = {c: [] for c in comps}
        for winner, loser in edges:
            adj[winner].append(loser)

        def visit(node):
            color[node] = 1
            for nxt in adj[node]:
                if color[nxt] == 1:
                    return True
                if color[nxt] == 0 and visit(nxt):
                    return True
            color[node] = 2
            return False

        return any(color[c] == 0 and visit(c) for c in comps)

    cyclic = _cyclic()
    top1 = [c for c in comps
            if all((c, other) in edges for other in comps if other != c)]
    bottom1 = [c for c in comps
               if all((other, c) in edges for other in comps if other != c)]
    unresolved = [row['name'] for row in separated.values()
                  if not row['separated']]
    n_pairs = k * (k - 1) // 2
    full = bool(not cyclic and len(edges) == n_pairs)
    if cyclic:
        consistency = 'cyclic'
    elif full:
        consistency = 'total_order'
    elif edges:
        consistency = 'partial_order'
    else:
        consistency = 'unresolved'
    return {
        'definition': ('ascending Shapley; a more negative value is more '
                       'beneficial; pair order uses the family-adjusted '
                       'two-sided test, not the point estimate alone'),
        'pairs': [separated[row['name']] for row in pair_rows],
        'n_pairs': n_pairs,
        'n_separated_pairs': len(edges),
        'n_unresolved_pairs': len(unresolved),
        'unresolved_pairs': unresolved,
        'edges': [[winner, loser] for winner, loser in sorted(edges)],
        'dominated_by': dominated_by,
        'dominates': dominates,
        'rank_lower': rank_lower,
        'rank_upper': rank_upper,
        'unique_rank': unique,
        'top1': top1,
        'bottom1': bottom1,
        'top1_licensed': bool(len(top1) == 1 and not cyclic),
        'bottom1_licensed': bool(len(bottom1) == 1 and not cyclic),
        'full_ranking_licensed': full,
        'ranking_consistency': consistency,
        'method': joint.get('method'),
        'family_size': n_pairs,
        'note': ('a separated pair is a family-adjusted ordering, not a '
                 'claim that the two components differ in every context; '
                 'context dependence remains in the lattice block'),
    }


def _arb_scale_sensitivity(log_stats, raw_stats, comps, *, tol):
    """Compare the two value-function scales without merging them."""
    log_means = [log_stats[c]['mean'] for c in comps]
    raw_means = [raw_stats[c]['mean'] for c in comps]
    if any(v is None for v in log_means + raw_means):
        return {'status': 'not_comparable',
                'reason': 'a scale has no finite Shapley mean',
                'rank_log': None, 'rank_ppl': None}
    rank_log = _al_rank(log_means, tol=tol)  # noqa: F821
    rank_ppl = _al_rank(raw_means, tol=tol)  # noqa: F821
    sign_flips = []
    for c in comps:
        a, b = log_stats[c]['mean'], raw_stats[c]['mean']
        if (a < -tol and b > tol) or (a > tol and b < -tol):
            sign_flips.append(c)
    order_flips = []
    tie_changes = []
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            o_log = _arb_order(log_stats[a]['mean'], log_stats[b]['mean'],
                              tol=tol)
            o_raw = _arb_order(raw_stats[a]['mean'], raw_stats[b]['mean'],
                              tol=tol)
            if o_log != o_raw:
                if o_log == 0 or o_raw == 0:
                    tie_changes.append([a, b])
                else:
                    order_flips.append([a, b])
    total_log = sum(log_means)
    total_raw = sum(raw_means)

    def _total_parts(values, total):
        """Return (finite_total, common_scale, scaled_total).

        A finite component panel near float64's limit can have an
        unrepresentable total (``1e308 + 1e308``).  The shares are still
        well-defined, so compute them from a common scale rather than
        returning zeros from ``x / inf``; the reported total itself is
        withheld as ``None``.
        """
        finite_total = total if _arb_math.isfinite(total) else None
        scale = max((abs(v) for v in values), default=0.0)
        scaled_total = (sum(v / scale for v in values) if scale > 0.0
                        else 0.0)
        return finite_total, scale, scaled_total
    # Compute one share per component.  The ordinary finite-total branch is
    # byte-identical to the historical x / total; the overflow branch uses
    # the same values divided by their common maximum.
    def _share(value, total, scale, scaled_total):
        if _arb_math.isfinite(total):
            return (value / total) if abs(total) > tol else None
        if (scale <= 0.0 or not _arb_math.isfinite(scaled_total)
                or scaled_total == 0.0):
            return None
        return (value / scale) / scaled_total
    fin_log, scale_log, scaled_log = _total_parts(log_means, total_log)
    fin_raw, scale_raw, scaled_raw = _total_parts(raw_means, total_raw)
    shares = {}
    for c in comps:
        shares[c] = {
            'log_ppl': _share(log_stats[c]['mean'], total_log,
                              scale_log, scaled_log),
            'ppl': _share(raw_stats[c]['mean'], total_raw,
                          scale_raw, scaled_raw),
        }
    dependent = bool(sign_flips or order_flips)
    status = 'scale_dependent' if dependent else 'scale_robust'
    note = ('Shapley is not invariant under monotone transforms; a rank flip '
            'is a property of the chosen value function, not a bug. Log PPL is '
            'the primary relative scale; raw PPL weights absolute changes.')
    return {
        'status': status,
        'rank_log': {c: rank_log[i] for i, c in enumerate(comps)},
        'rank_ppl': {c: rank_ppl[i] for i, c in enumerate(comps)},
        'rank_shift': {c: rank_ppl[i] - rank_log[i]
                       for i, c in enumerate(comps)},
        'rank_spearman': _al_spearman(log_means, raw_means),  # noqa: F821
        'sign_flips': sign_flips,
        'pair_order_flips': order_flips,
        'tie_changes': tie_changes,
        'shares': shares,
        'total_effect': {'log_ppl': fin_log, 'ppl': fin_raw},
        'note': note,
    }


def ablation_ranking(cells, *, components=None, alpha=0.05, min_n=3,
                     primary_scale='log_ppl',
                     scales=('log_ppl', 'ppl'),
                     max_components=AR_MAX_COMPONENTS,
                     tolerance=AR_TOL):
    """Joint inference for Shapley component rankings on a complete lattice.

    ``cells`` maps subset -> ``{seed: positive PPL}``.  The function refuses
    incomplete lattices, unequal seed coverage and fewer than two components;
    it never imputes a cell or ranks on a seed subset that differs per cell.

    The primary scale is ``log_ppl`` (the same relative scale as the lattice's
    §136 attribution); ``ppl`` is always reported as a sensitivity scale when
    requested.  The two scales are not two independent tests and are never
    pooled.
    """
    alpha = _arb_number(alpha, where='alpha')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0, 1), got {_arb_repr(alpha)}')
    min_n = _arb_int(min_n, where='min_n', minimum=0)
    max_components = min(
        _arb_int(max_components, where='max_components', minimum=1),
        AR_MAX_COMPONENTS)
    tolerance = _arb_number(tolerance, where='tolerance')
    if tolerance < 0.0:
        raise ValueError(
            f'tolerance must be non-negative, got {_arb_repr(tolerance)}')
    primary_scale = _arb_scale(primary_scale, where='primary_scale')
    scales = _arb_scales(scales, primary=primary_scale)
    lat = shapley_lattice_attribution(  # noqa: F821 (part_73)
        cells, components=components, scale='ppl', min_n=min_n, alpha=alpha,
        max_components=max_components)
    comps = list(lat.get('components') or [])
    seeds = list((lat.get('seed_coverage') or {}).get('common_seeds') or [])
    base = {
        'kind': AR_RESULT_KIND,
        'schema': AR_SCHEMA,
        'policy': dict(AR_POLICY),
        'components': comps,
        'n_components': len(comps),
        'seeds': seeds,
        'n_seeds': len(seeds),
        'alpha': float(alpha),
        'min_n': int(min_n),
        'max_components': int(max_components),
        'primary_scale': primary_scale,
        'scales': list(scales),
        'p_floor': _arb_p_floor(len(seeds)),
        'resolution_limited': None,
        'status': 'not_estimable',
        'warnings': list(lat.get('warnings') or []),
        'problems': list(lat.get('problems') or []),
        'joint': {},
        'component_stats': {},
        'ranking': None,
        'scale_sensitivity': None,
        'ranking_consistency': None,
        'top1_licensed': False,
        'bottom1_licensed': False,
        'full_ranking_licensed': False,
        'method': None,
        'family_size': None,
        'n_separated_pairs': None,
        'n_unresolved_pairs': None,
        'note': ('exploratory joint attribution; it never changes a frozen '
                 'p-value, Holm family or confirmed verdict'),
    }
    if lat.get('status') in ('not_estimable', 'fail'):
        base['status'] = lat.get('status')
        return base
    if len(comps) < 2:
        base['status'] = 'not_applicable'
        base['problems'].append(
            'ranking needs at least two declared components')
        return base
    if len(seeds) < max(2, int(min_n)):
        base['status'] = 'not_estimable'
        base['problems'].append(
            f'{len(seeds)} common seed(s) are below the ranking floor '
            f'max(2, min_n)={_arb_repr(max(2, int(min_n)))}')
        return base
    # Rebuild the per-seed Shapley vectors from the same normalized cells.
    # Log and raw values are computed from the identical PPL measurements, so
    # the scale comparison is a paired transformation, not two datasets.
    norm = _al_normalise_cells(cells, comps, scale='ppl')  # noqa: F821
    expected = _al_all_subsets(comps)  # noqa: F821
    for subset in expected:
        if subset not in norm:
            base['status'] = 'not_estimable'
            base['problems'].append(f'missing lattice cell {sorted(subset)!r}')
            return base
    log_vec = {c: [] for c in comps}
    raw_vec = {c: [] for c in comps}
    for seed in seeds:
        raw_values = {s: float(norm[s][seed]) for s in expected}
        log_values = {s: _arb_math.log(raw_values[s]) for s in expected}
        phi_raw = _al_shapley(raw_values, comps)  # noqa: F821
        phi_log = _al_shapley(log_values, comps)  # noqa: F821
        for c in comps:
            raw_vec[c].append(float(phi_raw[c]))
            log_vec[c].append(float(phi_log[c]))
    # Internal consistency: the log vectors must reproduce the canonical
    # lattice's per-seed Shapley values.  A divergence is an implementation
    # failure, not a surprising result.
    for c in comps:
        canonical = (lat.get('shapley') or {}).get(c) or {}
        canonical_seed = canonical.get('per_seed') or {}
        for i, seed in enumerate(seeds):
            value = canonical_seed.get(str(seed))
            if value is None or abs(float(value) - log_vec[c][i]) > (
                    tolerance * max(1.0, abs(float(value)),
                                    abs(log_vec[c][i]))):
                base['status'] = 'fail'
                base['problems'].append(
                    f'{c}: log Shapley vector disagrees with the canonical '
                    'lattice replay')
                return base
    p_floor = _arb_p_floor(len(seeds))
    base['p_floor'] = p_floor
    base['resolution_limited'] = bool(p_floor > alpha)
    if base['resolution_limited']:
        base['warnings'].append(
            f'exact two-sided sign-flip floor 2/2**{len(seeds)}={p_floor:.6g} '
            f'exceeds alpha={alpha}; no pair can be separated at this n')
    vectors = {'log_ppl': log_vec, 'ppl': raw_vec}
    # Joint inference is computed once, on the primary scale.  The second
    # scale is a sensitivity readout of the point attribution, not another
    # family of tests; giving both scales their own max-T family would invite
    # exactly the across-scale selection this block is meant to expose.
    primary_joint = _arb_joint(vectors[primary_scale], alpha=alpha)
    no_joint = {'contrasts': {}, 'method': None, 'fallback': False,
                'fallback_reason': None, 'critical_value': None,
                'ci_level_requested': None, 'ci_tail_level': None,
                'ci_note': 'joint inference is reported on the primary scale only',
                'p_floor_single': None, 'n_patterns': None}
    stats = {}
    for scale in scales:
        stats[scale] = _arb_component_stats(
            vectors[scale], seeds, alpha=alpha, min_n=min_n,
            joint=(primary_joint if scale == primary_scale else no_joint))
    for scale in scales:
        _arb_add_ranks(stats[scale], comps, tol=tolerance)
    ranking = _arb_ranking(comps, vectors[primary_scale],
                          stats[primary_scale], alpha=alpha,
                          tol=tolerance)
    if ranking.get('ranking_consistency') == 'cyclic':
        base['warnings'].append(
            'the separated dominance graph is cyclic; a total order is not '
            'licensed even though individual pairs may be separated')
    sensitivity = None
    if len(scales) >= 2:
        sensitivity = _arb_scale_sensitivity(
            stats['log_ppl'], stats['ppl'], comps, tol=tolerance)
        if sensitivity.get('status') == 'scale_dependent':
            base['warnings'].append(
                'the component order is scale-dependent between log PPL and '
                'raw PPL; do not report one ranking as scale-free')
    base.update({
        'status': 'warnings' if base['warnings'] else 'ok',
        'joint': {
            'method': primary_joint.get('method'),
            'fallback': primary_joint.get('fallback'),
            'fallback_reason': primary_joint.get('fallback_reason'),
            'critical_value': primary_joint.get('critical_value'),
            'ci_level_requested': primary_joint.get('ci_level_requested'),
            'ci_tail_level': primary_joint.get('ci_tail_level'),
            'ci_note': primary_joint.get('ci_note'),
            'family_size': len(comps),
            'p_floor_single': p_floor,
        },
        'component_stats': stats,
        'ranking': ranking,
        'scale_sensitivity': sensitivity,
        'ranking_consistency': ranking.get('ranking_consistency'),
        'top1_licensed': bool(ranking.get('top1_licensed')),
        'bottom1_licensed': bool(ranking.get('bottom1_licensed')),
        'full_ranking_licensed': bool(ranking.get('full_ranking_licensed')),
        'method': ranking.get('method'),
        'family_size': ranking.get('family_size'),
        'n_separated_pairs': ranking.get('n_separated_pairs'),
        'n_unresolved_pairs': ranking.get('n_unresolved_pairs'),
    })
    if primary_joint.get('fallback'):
        base['warnings'].append(
            'exact max-T was not computable; the component and pairwise '
            'families use Holm: ' + str(primary_joint.get('fallback_reason')))
        base['status'] = 'warnings'
    return base


def ablation_ranking_from_lattice(lattice, *, primary_scale='log_ppl',
                                  scales=('log_ppl', 'ppl'),
                                  alpha=None, min_n=None, max_components=None):
    """Run :func:`ablation_ranking` on a validated §136 lattice artifact."""
    verdict = validate_ablation_lattice(lattice)  # noqa: F821 (part_73)
    if verdict.get('status') != 'ok':
        raise ValueError(
            'lattice artifact does not replay: '
            + '; '.join(verdict.get('problems') or ['unknown']))
    inputs = lattice.get('inputs') or {}
    comps = list(inputs.get('components') or [])
    raw_scale = inputs.get('scale')
    if raw_scale not in ('ppl', 'log_ppl'):
        raise ValueError(
            f'lattice inputs.scale={_arb_repr(raw_scale)} is not recognised')
    cells = _al_cells_from_records(  # noqa: F821
        inputs.get('cells'), comps, scale=raw_scale)
    if raw_scale == 'log_ppl':
        converted = {}
        for subset, per in cells.items():
            converted[subset] = {}
            for seed, value in per.items():
                try:
                    ppl = _arb_math.exp(value)
                except OverflowError:
                    raise ValueError(
                        'lattice log-PPL cell overflows when converted to PPL')
                if ppl == 0.0:
                    # Underflow is a lost measurement, not a PPL of zero; the
                    # downstream positive-PPL contract would reject it with a
                    # less specific error.  Name the conversion failure.
                    raise ValueError(
                        'lattice log-PPL cell underflows when converted to '
                        'PPL')
                converted[subset][seed] = ppl
        cells = converted
    return ablation_ranking(
        cells, components=comps,
        alpha=inputs.get('alpha') if alpha is None else alpha,
        min_n=inputs.get('min_n') if min_n is None else min_n,
        primary_scale=primary_scale, scales=scales,
        max_components=(inputs.get('max_components')
                        if max_components is None else max_components))


def ablation_ranking_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    return _sha256_json(body)  # noqa: F821 (part_00)


def build_ablation_ranking(cells, *, components=None, alpha=0.05, min_n=3,
                           primary_scale='log_ppl',
                           scales=('log_ppl', 'ppl'),
                           max_components=AR_MAX_COMPONENTS,
                           tolerance=AR_TOL, created_at=None):
    """Run the ranking and stamp a replayable artifact."""
    result = ablation_ranking(
        cells, components=components, alpha=alpha, min_n=min_n,
        primary_scale=primary_scale, scales=scales,
        max_components=max_components, tolerance=tolerance)
    comps = list(result.get('components') or [])
    norm = _al_normalise_cells(  # noqa: F821
        cells, comps if components is not None else None, scale='ppl')
    artifact = {
        'kind': AR_KIND,
        'schema': AR_SCHEMA,
        'policy': dict(AR_POLICY),
        'created_at': created_at or _arb_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'inputs': {
            'cells': _al_cells_to_records(norm, comps),  # noqa: F821
            'components': comps,
            'alpha': float(result.get('alpha') or alpha),
            'min_n': int(result.get('min_n') or min_n),
            'primary_scale': result.get('primary_scale'),
            'scales': list(result.get('scales') or []),
            'max_components': int(result.get('max_components')
                                  or max_components),
        },
        'result': result,
    }
    artifact['content_sha256'] = ablation_ranking_content_sha(artifact)
    return artifact


def validate_ablation_ranking(artifact):
    """Replay a stored ranking artifact from its recorded PPL cells.

    A structurally valid ``not_estimable``/``not_applicable`` result replays
    cleanly and is returned with that status; a changed component, seed,
    Shapley value, ranking, hash or a smuggled frozen-decision field is
    ``fail``.  Missing evidence is never a pass.
    """
    out = {'kind': 'ablation_ranking_validation', 'schema': 1,
           'status': 'fail', 'problems': [], 'warnings': [],
           'result': None, 'result_status': None,
           'content_sha256': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if artifact.get('kind') != AR_KIND:
        out['problems'].append(
            f'artifact.kind must be {AR_KIND!r}, got '
            f'{_arb_repr(artifact.get("kind"))}')
    if artifact.get('schema') != AR_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {AR_SCHEMA}, got '
            f'{_arb_repr(artifact.get("schema"))}')
    if artifact.get('policy') != AR_POLICY:
        out['problems'].append(
            'artifact.policy does not match this implementation')
    stolen = sorted(set(_arb_forbidden_payload(artifact)))
    if stolen:
        out['problems'].append(
            'artifact carries frozen-decision field(s): ' + ', '.join(stolen))
    allowed = {'kind', 'schema', 'policy', 'created_at', 'inputs', 'result',
               'content_sha256'}
    # An untrusted artifact can carry non-string keys.  Sorting mixed key
    # types raises TypeError and repr(huge_int) raises on Python 3.12's
    # digit limit; rank/format by the bounded repr instead.
    extra = sorted(set(artifact) - allowed, key=_arb_repr)
    if extra:
        out['problems'].append(
            'artifact carries field(s) outside the replay schema: '
            + ', '.join(_arb_repr(k) for k in extra))
    recorded = artifact.get('content_sha256')
    out['content_sha256'] = (recorded
                             if isinstance(recorded, str)
                             and len(recorded) == 64 else None)
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append(
            'content_sha256 is missing or not a 64-character digest')
    else:
        try:
            replay_sha = ablation_ranking_content_sha(artifact)
        except (TypeError, ValueError, RecursionError) as exc:
            out['problems'].append(
                f'artifact body is not canonical JSON: {type(exc).__name__}: {exc}')
            replay_sha = None
        if replay_sha is not None and replay_sha != recorded:
            out['problems'].append(
                'content_sha256 does not reproduce from the artifact body')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        return out
    extra_inputs = sorted(set(inputs) - set(_AR_INPUT_KEYS), key=_arb_repr)
    missing_inputs = sorted(set(_AR_INPUT_KEYS) - set(inputs))
    if extra_inputs:
        out['problems'].append(
            'artifact.inputs carries field(s) outside the schema: '
            + ', '.join(_arb_repr(k) for k in extra_inputs))
    if missing_inputs:
        out['problems'].append(
            f'artifact.inputs is missing required key(s): {missing_inputs}')
        return out
    try:
        comps = list(inputs.get('components') or [])
        cells = _al_cells_from_records(  # noqa: F821
            inputs.get('cells'), comps, scale='ppl')
        replay = ablation_ranking(
            cells, components=comps, alpha=inputs.get('alpha'),
            min_n=inputs.get('min_n'),
            primary_scale=inputs.get('primary_scale'),
            scales=inputs.get('scales'),
            max_components=inputs.get('max_components'))
    except Exception as exc:
        out['problems'].append(
            f'artifact.inputs cannot be replayed: {type(exc).__name__}: {exc}')
        return out
    out['result'] = replay
    out['result_status'] = replay.get('status')
    stored = artifact.get('result')
    if not isinstance(stored, dict):
        out['problems'].append('artifact.result must be an object')
        return out
    try:
        same = _al_equiv(stored, replay)  # noqa: F821 (part_73)
    except RecursionError:
        out['problems'].append(
            'artifact.result nesting is too deep to replay')
        return out
    if not same:
        out['problems'].append(
            'artifact.result does not replay from its recorded cells')
        for key in sorted(set(stored) | set(replay), key=_arb_repr):
            try:
                same_key = _al_equiv(stored.get(key), replay.get(key))
            except RecursionError:
                same_key = False
            if not same_key:
                out['problems'].append(
                    f'result.{_arb_repr(key)} differs from the replay')
    if out['problems']:
        out['status'] = 'fail'
    else:
        out['status'] = 'ok'
    return out


def write_ablation_ranking(outdir, artifact, *, overwrite=False):
    """Atomically persist ``ablation_ranking.json``; refuse silent replacement."""
    import os as _arb_os
    import json as _arb_json
    if isinstance(outdir, _arb_os.PathLike):
        outdir = _arb_os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_ablation_ranking(artifact)
    path = _arb_os.path.join(outdir, 'ablation_ranking.json')
    _result = artifact.get('result') if isinstance(artifact, dict) else None
    out = {'kind': 'write_ablation_ranking', 'path': path, 'written': False,
           'unchanged': False, 'conflict': False, 'status': verdict['status'],
           'artifact_status': (_result.get('status')
                               if isinstance(_result, dict) else None),
           'problems': list(verdict['problems'])}
    if verdict['status'] != 'ok':
        return out
    if _arb_os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                old = _arb_json.load(fh)
        except Exception:
            old = None
        if (isinstance(old, dict)
                and old.get('content_sha256') == artifact.get('content_sha256')):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different ablation_ranking.json already exists; pass '
                'overwrite=True only after re-running the analysis')
            return out
    _arb_os.makedirs(outdir, exist_ok=True)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            _arb_json.dump(artifact, fh, ensure_ascii=False, indent=1,
                           allow_nan=False)
            fh.write('\n')
    except Exception as exc:
        # Provenance fields such as ``created_at`` are intentionally not part
        # of the content hash, so a caller can hand in a value that survives
        # validation but cannot be serialised (e.g. a huge integer).  Report
        # that as a failed write instead of leaving a partial tmp file and
        # raising after the validation step already said ``ok``.
        try:
            _arb_os.remove(tmp)
        except OSError:
            pass
        out['status'] = 'fail'
        out['problems'].append(
            f'artifact cannot be serialised: {type(exc).__name__}: {exc}')
        return out
    _arb_os.replace(tmp, path)
    out['written'] = True
    return out
