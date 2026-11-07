# ---- 结果分析的多重宇宙：口径曲线、交集-并集稳健性与族内 max-T --------
#
# §63 的 `analysis_choice_concordance` 把报告实际使用的 8 条聚合口径并排
# 展示，但它是描述性的：不产生 p 值，也不回答「这些口径能否共同支持一个
# 稳健结论」。§124 的跨 panel 合成也把效应尺度、截尾、加权、文档过滤列为
# 未枚举的分析自由度。结果是：分析者仍可以在许多都说得通的口径之间挑选，
# 而报告只呈现一个；「换一个口径也显著」既没有被校准，也没有被拒绝。
#
# 本片段补上这一层。输入是一组**事前声明**的分析规格（specification），
# 每条给出同一主对照的逐种子配对差。核心统计区分两种完全不同的问题：
#
#   1. 稳健性（所有规格都显示同向效应）：这是交集-并集检验
#      （intersection-union test, IUT）。H0 = 「至少一个规格的效应为零」，
#      H1 = 「每个规格的效应都非零」。只要每个规格各自在 alpha 水平拒绝，
#      就可以拒绝 H0；IUT 不需要多重校正，因为「全部拒绝」比任何单个拒绝
#      都更难发生。把 IUT 当成 union test 再乘 m，或者把「m 个规格里 k 个
#      显著」当成证据，都是错的。
#
#   2. 是否存在某个规格显示效应（union / at-least-one）：这是族内 max-T
#      的问题，必须控制 FWER。本片段只在**同一估计量、同一尺度、同一批
#      种子**的 family 内计算精确符号翻转 max-T；跨 family 只做符号与
#      显著性的描述性对账，绝不把不同估计量塞进一个 p 值族。
#
# 另一个常见错误是把「所有规格的 CI 都排除零」当成联合 95% 声明。逐条
# 95% CI 的联合覆盖远低于 95%；本片段不把并排的 CI 叫 simultaneous
# interval，只在 family 内给出 max-T 同时区间，并显式记录其精确尾水平。
#
# SCOPE: 这是**探索性稳健性读数**。它不新增训练、不读取任何 split、不重算
# PPL、不产生确认 p 值、不进入冻结 Holm 族，也不改变 `verified`/
# `confirmed`。规格必须事前声明；本片段把「换了哪些口径」变成可重放的
# 证据，但不能证明规格集合是完备的，也不能修复数据/设计本身的偏差。

import json as _am_json
import math as _am_math
import numbers as _am_numbers
import os as _am_os
import time as _am_time

AM_KIND = 'analysis_multiverse'
AM_ARTIFACT_KIND = 'analysis_multiverse_artifact'
AM_SCHEMA = 1
AM_MIN_SPECS = 3
AM_MIN_SEEDS = 2
AM_MAX_SPECS = 512
AM_MAX_FEATURES = 64

AM_POLICY = {
    'version': 'analysis-multiverse-1',
    'role': ('exploratory robustness readout over a declared set of '
             'defensible analysis specifications for one primary contrast'),
    'primary_rule': ('exactly one predeclared primary specification; it is '
                     'never selected from the observed effects'),
    'family_rule': ('a family groups specifications on the same estimand and '
                    'scale with the exact same seed set; exact max-T is '
                    'computed only inside such a family'),
    'robustness_rule': ('an all-specifications claim is an intersection-union '
                        'test: reject H0 ("some specification has zero '
                        'effect") only when every specification rejects its '
                        'own two-sided null at alpha; no multiplicity '
                        'correction is applied because the all-reject event '
                        'is the intersection'),
    'union_rule': ('a within-family max-T p answers the weaker at-least-one '
                   'question and controls FWER over that declared family; it '
                   'is never evidence that every specification is non-zero'),
    'sign_rule': ('sign agreement is measured against the predeclared primary '
                  'specification, not against the majority of specifications'),
    'resolution_rule': ('each exact two-sided sign-flip p has floor '
                        '2/2**n_nonzero; the IUT p floor is the maximum of '
                        'those floors and can exceed alpha'),
    'interval_rule': ('pointwise 95% intervals are descriptive; only the '
                      'within-family max-T interval is simultaneous, and its '
                      'exact tail level is reported'),
    'claim_scope': ('exploratory; no new confirmatory p-value, no change to '
                    'the frozen Holm family, a panel `verified` flag or a '
                    '`confirmed` verdict'),
}

AM_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'frozen_decision', 'confirmation_verdict', 'significant',
    'significance', 'holm',
})
AM_INPUT_KEYS = ('specs', 'primary', 'alpha', 'min_specs', 'min_seeds')
AM_SPEC_INPUT_KEYS = ('name', 'family', 'deltas', 'features')
AM_SPEC_FIELDS = (
    'family', 'n_seeds', 'seeds', 'mean', 'sd', 'sem', 't_stat',
    'p_exact_signflip', 'p_floor', 'ci95', 'ci_note', 'sign',
    'excludes_zero', 'reject_raw', 'same_sign_as_primary', 'features',
)
AM_FAMILY_FIELDS = (
    'n_specs', 'specs', 'n_seeds', 'seeds', 'joint_status', 'joint_reason',
    'joint', 'effects', 'sign_stable', 'feature_influence',
    'most_influential_feature',
)
AM_RESULT_KEYS = (
    'kind', 'status', 'pattern', 'alpha', 'primary', 'primary_sign',
    'n_specs', 'n_families', 'min_specs', 'min_seeds', 'p_iut',
    'iut_p_floor', 'resolution_limited', 'resolution_limited_specs',
    'specs', 'families', 'summary', 'claim', 'warnings', 'policy',
)


def _am_repr(value):
    """Short, exception-safe repr for diagnostics on untrusted values."""
    return _safe_repr(value, limit=160)


def _am_number(value, *, where, positive=False):
    """Finite JSON number; bool/None/str and non-finite values fail closed."""
    # NumPy integer/floating scalars are numeric measurements even though
    # they are not instances of the builtin int/float (``_finite_scalar``
    # accepts them for the same reason).  ``np.bool_`` is not a Real, so it
    # stays rejected rather than becoming 0.0/1.0.
    if isinstance(value, bool) or not isinstance(value, _am_numbers.Real):
        raise ValueError(f'{where} must be a finite number, got '
                         f'{_am_repr(value)}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where} must be a finite number, got '
                         f'{_am_repr(value)}') from None
    if not _am_math.isfinite(out):
        raise ValueError(f'{where} must be finite, got {_am_repr(value)}')
    if positive and out <= 0.0:
        raise ValueError(f'{where} must be > 0, got {_am_repr(value)}')
    return out


def _am_int(value, *, where, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, _am_numbers.Integral):
        raise ValueError(f'{where} must be an integer, got '
                         f'{_am_repr(value)}')
    out = int(value)
    try:
        # A count that Python 3.12 cannot stringify cannot be written as a
        # JSON number; reject it here rather than failing later while
        # hashing/writing the artifact.  Large but representable values
        # (e.g. 10**400) stay valid.
        str(out)
    except (ValueError, MemoryError):
        raise ValueError(
            f'{where}: integer is too large to represent as JSON') from None
    if minimum is not None and out < minimum:
        raise ValueError(f'{where} must be >= {minimum}, got '
                         f'{_am_repr(value)}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{where} must be <= {maximum}, got '
                         f'{_am_repr(value)}')
    return out


def _am_seed(value, *, where):
    """Exact non-negative integer seed label (``"3"`` and ``3`` agree)."""
    if isinstance(value, str):
        if not value.isdigit():
            raise ValueError(f'{where} must be an integer seed label, got '
                             f'{_am_repr(value)}')
        try:
            value = int(value)
        except (ValueError, MemoryError):
            raise ValueError(
                f'{where}: seed is too large to represent as a JSON key'
            ) from None
    if isinstance(value, bool) or not isinstance(value, _am_numbers.Integral):
        raise ValueError(f'{where} must be an integer seed label, got '
                         f'{_am_repr(value)}')
    if value < 0:
        raise ValueError(f'{where} must be >= 0, got {_am_repr(value)}')
    try:
        str(value)
    except (ValueError, MemoryError):
        raise ValueError(
            f'{where}: seed is too large to represent as a JSON key') from None
    return int(value)


def _am_text(value, *, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where} must be a non-empty string, got '
                         f'{_am_repr(value)}')
    return value.strip()


def _am_feature_value(value, *, where):
    """Scalar feature level; containers/None are not a declared choice."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, _am_numbers.Integral):
        out = int(value)
        try:
            # The feature level is echoed in the strict-JSON result and in
            # the influence table.  Reject an integer that CPython cannot
            # decimalise here, with a named error, instead of failing inside
            # json.dumps while the result is being assembled.
            str(out)
        except (ValueError, MemoryError):
            raise ValueError(
                f'{where}: integer is too large to represent as JSON')
        return out
    if isinstance(value, _am_numbers.Real):
        number = float(value)
        if not _am_math.isfinite(number):
            raise ValueError(f'{where} must be finite, got {_am_repr(value)}')
        return number
    if isinstance(value, str):
        return value
    raise ValueError(f'{where} must be a bool/int/float/string, got '
                     f'{_am_repr(value)}')


def _am_features(value, *, where):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object, got '
                         f'{_am_repr(value)}')
    if len(value) > AM_MAX_FEATURES:
        raise ValueError(f'{where} carries {len(value)} features; the cap is '
                         f'{AM_MAX_FEATURES}')
    out = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'{where} feature names must be non-empty '
                             f'strings, got {_am_repr(key)}')
        name = key.strip()
        if name in AM_FORBIDDEN_KEYS:
            raise ValueError(
                f'{where} feature name {name!r} collides with a frozen-'
                'decision field; free-form choice names must not smuggle it')
        if name in out:
            raise ValueError(f'{where} has duplicate feature {name!r}')
        out[name] = _am_feature_value(raw, where=f'{where}.{name}')
    return out


def _am_delta_map(value, *, where):
    """Normalise ``seed -> paired delta`` into a sorted integer-key mapping."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{where} must be a non-empty seed -> delta object')
    out = {}
    for raw_seed, raw_delta in value.items():
        seed = _am_seed(raw_seed, where=f'{where} seed')
        if seed in out:
            raise ValueError(f'{where} has duplicate seed {seed}')
        out[seed] = _am_number(raw_delta, where=f'{where}[{seed}]')
    return {seed: out[seed] for seed in sorted(out)}


def _am_normalise_specs(specs):
    """Validate and canonicalise the declared specification list."""
    if isinstance(specs, (str, bytes, bytearray)) or not isinstance(
            specs, (list, tuple)) or not specs:
        raise ValueError('specs must be a non-empty list of objects')
    if len(specs) > AM_MAX_SPECS:
        raise ValueError(f'specs carries {len(specs)} entries; the cap is '
                         f'{AM_MAX_SPECS}')
    out, names = [], set()
    for index, raw in enumerate(specs):
        where = f'specs[{index}]'
        if not isinstance(raw, dict):
            raise ValueError(f'{where} must be an object, got '
                             f'{_am_repr(raw)}')
        extra = sorted(set(raw) - set(AM_SPEC_INPUT_KEYS), key=_am_repr)
        if extra:
            raise ValueError(
                f'{where} carries field(s) outside the declaration schema: '
                + ', '.join(_am_repr(key) for key in extra))
        name = _am_text(raw.get('name'), where=f'{where}.name')
        if name in AM_FORBIDDEN_KEYS:
            raise ValueError(
                f'{where}.name {name!r} collides with a frozen-decision '
                'field; a specification name must not smuggle it')
        if name in names:
            raise ValueError(f'duplicate specification name {name!r}')
        names.add(name)
        family = _am_text(raw.get('family', 'primary'),
                          where=f'{where}.family')
        if family in AM_FORBIDDEN_KEYS:
            raise ValueError(
                f'{where}.family {family!r} collides with a frozen-decision '
                'field; a family name must not smuggle it')
        deltas = _am_delta_map(raw.get('deltas'),
                               where=f'{where}.deltas')
        if len(deltas) < AM_MIN_SEEDS:
            raise ValueError(f'{where}.deltas has {len(deltas)} seeds; at '
                             f'least {AM_MIN_SEEDS} paired seeds are required')
        features = _am_features(raw.get('features'),
                                where=f'{where}.features')
        out.append({
            'name': name,
            'family': family,
            'deltas': deltas,
            'features': features,
        })
    return sorted(out, key=lambda rec: rec['name'])


def _am_input_specs(specs):
    """JSON-safe canonical inputs: sorted names and ``[seed, delta]`` pairs."""
    return [{
        'name': rec['name'],
        'family': rec['family'],
        'deltas': [[seed, rec['deltas'][seed]]
                   for seed in sorted(rec['deltas'])],
        'features': {key: rec['features'][key]
                     for key in sorted(rec['features'])},
    } for rec in specs]


def _am_specs_from_inputs(rows):
    """Rebuild the internal spec form from canonical artifact inputs."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError('inputs.specs must be a list')
    out = []
    for index, row in enumerate(rows):
        where = f'inputs.specs[{index}]'
        if not isinstance(row, dict):
            raise ValueError(f'{where} must be an object')
        extra = sorted(set(row) - set(AM_SPEC_INPUT_KEYS), key=_am_repr)
        if extra:
            raise ValueError(
                f'{where} carries field(s) outside the schema: '
                + ', '.join(_am_repr(key) for key in extra))
        pairs = row.get('deltas')
        if not isinstance(pairs, (list, tuple)):
            raise ValueError(f'{where}.deltas must be a list of pairs')
        deltas = {}
        for j, pair in enumerate(pairs):
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2):
                raise ValueError(f'{where}.deltas[{j}] must be [seed, delta]')
            seed = _am_seed(pair[0], where=f'{where}.deltas[{j}].seed')
            if seed in deltas:
                raise ValueError(f'{where}.deltas has duplicate seed '
                                 f'{seed}')
            deltas[seed] = _am_number(
                pair[1], where=f'{where}.deltas[{j}].delta')
        out.append({
            'name': row.get('name'),
            'family': row.get('family', 'primary'),
            'deltas': deltas,
            'features': row.get('features'),
        })
    return _am_normalise_specs(out)


def _am_alpha(value):
    out = _am_number(value, where='alpha')
    if not (0.0 < out < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {_am_repr(value)}')
    return out


def _am_sign(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _am_as_fraction(value):
    """Exact rational for an arbitrary-precision measurement.

    Python/NumPy integers and ``Decimal`` values convert directly; floats use
    their exact binary value.  Used only on the extreme-scale fallback paths,
    so ordinary float panels never pay for the conversion.
    """
    from fractions import Fraction as _Fraction
    try:
        return _Fraction(value)
    except (TypeError, ValueError):
        return _Fraction.from_float(float(value))


def _am_exact_mean(vals):
    try:
        total = sum((_am_as_fraction(v) for v in vals), _am_as_fraction(0))
        out = float(total / len(vals))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return out if _am_math.isfinite(out) else None


def _am_exact_sd(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    try:
        exact = [_am_as_fraction(v) for v in vals]
        mean = sum(exact, _am_as_fraction(0)) / n
        var = sum(((x - mean) ** 2 for x in exact),
                  _am_as_fraction(0)) / (n - 1)
        if var < 0:
            return None
        if var == 0:
            return 0.0
        try:
            fvar = float(var)
        except OverflowError:
            return float('inf')
        sd = _am_math.sqrt(fvar)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return sd if _am_math.isfinite(sd) else float('inf')


def _am_span(values):
    """Representable ``max - min`` span, or ``None`` beyond float64.

    A finite panel can still have a span larger than the float range (for
    example ``[+1e308, -1e308]``).  Reports are strict JSON, so an infinite
    descriptive spread is withheld instead of poisoning the artifact.  Exact
    integer endpoints beyond float64 have an exact difference even though the
    final float conversion is what must be checked.
    """
    lo = min(values)
    hi = max(values)
    try:
        out = hi - lo
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        out = float(out)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _am_math.isfinite(out) else None


def _am_median(values):
    vals = list(values)
    try:
        ordered = sorted(float(x) for x in vals)
    except (TypeError, ValueError, OverflowError):
        # An exact integer beyond float64 cannot enter the historical float
        # median, but its median may still be representable (or explicitly
        # unavailable).  Compute it exactly instead of aborting the readout.
        if not all(isinstance(x, _am_numbers.Integral)
                   and not isinstance(x, bool) for x in vals):
            return None
        ordered = sorted(int(x) for x in vals)
        n = len(ordered)
        if n == 0:
            return None
        mid = n // 2
        try:
            if n % 2:
                out = float(_am_as_fraction(ordered[mid]))
            else:
                out = float(_am_as_fraction(
                    ordered[mid - 1] + ordered[mid]) / 2)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            return None
        return out if _am_math.isfinite(out) else None
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    lo, hi = ordered[mid - 1], ordered[mid]
    # Keep the historical (lo + hi) / 2 bit-for-bit whenever it is finite;
    # only an overflow needs the half-gap form.  For same-sign values the
    # half-gap cannot overflow; opposite-sign values always have a finite
    # sum, so their ordinary midpoint is already safe.
    out = 0.5 * (lo + hi)
    if _am_math.isfinite(out):
        return out
    if (lo > 0.0) == (hi > 0.0):
        return lo + (hi - lo) / 2.0
    return out


def _am_mean(vals):
    """Scale-first arithmetic mean; ``fsum`` alone overflows at 1e308."""
    n = len(vals)
    if n == 0:
        return None
    scale = max(abs(x) for x in vals)
    if scale == 0.0:
        return 0.0
    try:
        out = (_am_math.fsum(x / scale for x in vals) / n) * scale
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        # Multiplying the unitless mean back by an exact integer scale beyond
        # float64 raises even when the true mean (or a zero mean) is
        # representable; form it exactly in that case.
        return _am_exact_mean(vals)
    return out


def _am_sd(vals):
    """Sample SD with a scale-first path that cannot overflow 1e308 panels."""
    n = len(vals)
    if n < 2:
        return 0.0
    scale = max(abs(x) for x in vals)
    if scale == 0.0:
        return 0.0
    scaled = [x / scale for x in vals]
    mean = _am_math.fsum(scaled) / n
    var = _am_math.fsum((x - mean) ** 2 for x in scaled) / (n - 1)
    sd_scaled = _am_math.sqrt(max(0.0, var))
    try:
        out = sd_scaled * scale
    except (TypeError, ValueError, OverflowError):
        # Same conversion hazard as the mean: an exact integer scale beyond
        # float64 (or an all-identical huge-int panel with true SD 0) must be
        # evaluated exactly, not escape.
        return _am_exact_sd(vals)
    return out if _am_math.isfinite(out) else float('inf')


def _am_t_interval(mean, sd, n, alpha):
    """Pointwise two-sided t interval; ``None`` when it is not representable."""
    if n < 2:
        return None, None
    if not _am_math.isfinite(sd):
        return None, None
    if sd == 0.0:
        if _am_math.isfinite(mean):
            return [float(mean), float(mean)], 'zero seed variance'
        return None, None
    try:
        tc = float(_t_critical_alpha(alpha, float(n - 1)))  # noqa: F821
    except Exception:
        return None, None
    if not _am_math.isfinite(tc):
        return None, None
    sem = sd / _am_math.sqrt(n)
    half = tc * sem
    if not _am_math.isfinite(half):
        return None, None
    lo, hi = mean - half, mean + half
    if not (_am_math.isfinite(lo) and _am_math.isfinite(hi)):
        return None, None
    return [float(lo), float(hi)], None


def _am_spec_stats(rec, *, alpha, primary_sign, where):
    vals = [rec['deltas'][seed] for seed in sorted(rec['deltas'])]
    n = len(vals)
    mean = _am_mean(vals)
    sd = _am_sd(vals)
    sem = (sd / _am_math.sqrt(n)) if _am_math.isfinite(sd) else None
    t_stat = (mean / sem) if (sem is not None and sem > 0.0
                              and _am_math.isfinite(mean)) else None
    exact = exact_signflip_p(vals)  # noqa: F821
    p = float(exact['p'])
    floor = exact.get('p_floor')
    floor = float(floor) if floor is not None else None
    ci, ci_note = _am_t_interval(mean, sd, n, alpha)
    sign = _am_sign(mean)
    return {
        'family': rec['family'],
        'n_seeds': int(n),
        'seeds': [int(seed) for seed in sorted(rec['deltas'])],
        'mean': float(mean),
        'sd': (float(sd) if _am_math.isfinite(sd) else None),
        'sem': (float(sem) if (sem is not None and _am_math.isfinite(sem))
                else None),
        't_stat': (float(t_stat) if t_stat is not None else None),
        'p_exact_signflip': p,
        'p_floor': floor,
        'ci95': ci,
        'sign': int(sign),
        'excludes_zero': bool(p <= alpha),
        'reject_raw': bool(p <= alpha),
        'same_sign_as_primary': bool(primary_sign != 0 and sign == primary_sign),
        'features': {key: rec['features'][key] for key in sorted(rec['features'])},
        'ci_note': ci_note,
    }


def _am_feature_level(value):
    return _am_json.dumps(value, ensure_ascii=False, sort_keys=True)


def _am_feature_influence(names, specs, stats, *, primary):
    """Descriptive marginal spread for each declared analytic-choice feature.

    This is deliberately *not* a variance decomposition: features are usually
    crossed and unbalanced, so a level mean still contains the other choices.
    It answers only "which declared choice is associated with the largest
    shift in the point estimate", and reports the raw spread alongside a
    ratio to the primary effect.
    """
    keys = set()
    for name in names:
        keys.update(stats[name]['features'])
    out = {}
    primary_mean = stats[primary]['mean']
    for key in sorted(keys):
        levels = {}
        missing = []
        for name in names:
            feats = specs[name]['features']
            if key not in feats:
                missing.append(name)
                continue
            level = _am_feature_level(feats[key])
            bucket = levels.setdefault(level, {
                'value': feats[key],
                'specs': [],
                'means': [],
                'signs': [],
            })
            bucket['specs'].append(name)
            bucket['means'].append(stats[name]['mean'])
            bucket['signs'].append(stats[name]['sign'])
        level_rows = {}
        for level, bucket in sorted(levels.items()):
            level_rows[level] = {
                'value': bucket['value'],
                'n_specs': len(bucket['specs']),
                'specs': sorted(bucket['specs']),
                'mean': float(_am_mean(bucket['means'])),
                'min': float(min(bucket['means'])),
                'max': float(max(bucket['means'])),
                'signs': sorted(set(int(x) for x in bucket['signs'])),
            }
        level_means = [row['mean'] for row in level_rows.values()]
        spread = _am_span(level_means) if len(level_means) >= 2 else 0.0
        sign_flip = (len({sign for row in level_rows.values()
                          for sign in row['signs']}) > 1)
        relative = None
        if spread is not None and primary_mean != 0.0:
            try:
                relative = spread / abs(primary_mean)
            except (ArithmeticError, OverflowError):
                relative = None
            if relative is not None and not _am_math.isfinite(relative):
                relative = None
        out[key] = {
            'n_levels': len(level_rows),
            'levels': level_rows,
            'specs_missing_feature': sorted(missing),
            'spread': (float(spread) if spread is not None else None),
            'relative_to_primary': (float(relative)
                                    if relative is not None else None),
            'sign_flip': bool(sign_flip),
            'note': ('descriptive marginal spread; levels are unbalanced and '
                     'other declared choices are not held fixed'),
        }
    candidates = [(key, row['spread']) for key, row in out.items()
                  if row['n_levels'] >= 2 and row['spread'] is not None]
    most = None
    if candidates:
        key, spread = sorted(candidates, key=lambda kv: (-kv[1], kv[0]))[0]
        most = {
            'feature': key,
            'spread': float(spread),
            'relative_to_primary': out[key]['relative_to_primary'],
            'sign_flip': bool(out[key]['sign_flip']),
        }
    return out, most


def _am_family_block(family, names, specs, stats, *, alpha, primary):
    rec = {'n_specs': len(names), 'specs': list(names)}
    seeds = [tuple(stats[name]['seeds']) for name in names]
    common = all(seed_tuple == seeds[0] for seed_tuple in seeds)
    rec['n_seeds'] = len(seeds[0]) if common else None
    rec['seeds'] = list(seeds[0]) if common else None
    rec['joint'] = None
    rec['joint_reason'] = None
    if len(names) < 2:
        rec['joint_status'] = 'single_spec'
        rec['joint_reason'] = 'a joint family needs at least two specifications'
    elif not common:
        rec['joint_status'] = 'not_comparable'
        rec['joint_reason'] = ('the declared specifications do not share one '
                               'exact seed set; a joint sign-flip null cannot '
                               'be formed')
    else:
        try:
            joint = maxt_signflip_adjust(  # noqa: F821
                {name: [specs[name]['deltas'][seed]
                        for seed in sorted(specs[name]['deltas'])]
                 for name in names}, alpha=alpha)
        except ValueError as exc:
            rec['joint_status'] = 'not_estimable'
            rec['joint_reason'] = str(exc)
        else:
            rec['joint_status'] = 'ok'
            rec['joint'] = joint
    means = [stats[name]['mean'] for name in names]
    signs = {stats[name]['sign'] for name in names}
    _span = _am_span(means)
    _relative_range = None
    if _span is not None and primary in names and stats[primary]['mean'] != 0.0:
        try:
            _relative_range = _span / abs(stats[primary]['mean'])
        except (ArithmeticError, OverflowError):
            _relative_range = None
        if (_relative_range is not None
                and not _am_math.isfinite(_relative_range)):
            _relative_range = None
    rec['effects'] = {
        'min': float(min(means)),
        'median': float(_am_median(means)),
        'max': float(max(means)),
        'range': (float(_span) if _span is not None else None),
        'primary': (float(stats[primary]['mean'])
                    if primary in names else None),
        'relative_range': (float(_relative_range)
                           if _relative_range is not None else None),
    }
    rec['sign_stable'] = bool(len(signs) == 1 and 0 not in signs)
    influence, most = _am_feature_influence(
        names, specs, stats, primary=primary)
    rec['feature_influence'] = influence
    rec['most_influential_feature'] = most
    return rec


def analysis_multiverse(specs, *, primary, alpha=0.05,
                        min_specs=AM_MIN_SPECS, min_seeds=AM_MIN_SEEDS):
    """Evaluate a declared analysis multiverse for one paired contrast.

    Parameters
    ----------
    specs : list of dicts
        Each entry declares ``name``, ``family`` (default ``"primary"``),
        ``deltas`` (``seed -> paired effect`` on the family's scale; positive
        means the unfavourable direction), and optional ``features``
        (``choice_name -> level``).  All entries must use the same sign
        convention; only the sign/zero status is compared across families.
    primary : str
        The predeclared primary specification.  The data never choose it.
    alpha : float
        Two-sided level for each specification and for the family max-T.
    min_specs : int
        Minimum number of declared specifications for a multiverse claim.
    min_seeds : int
        Minimum paired seeds per specification.

    Returns
    -------
    dict
        ``specs`` (per-specification estimates), ``families`` (within-family
        exact max-T and descriptive choice influence), ``claim`` (IUT result),
        and a ``pattern`` in ``stable`` / ``direction_unstable`` /
        ``precision_unstable`` / ``resolution_limited`` / ``zero_effect`` /
        ``insufficient_specs``.

    Raises
    ------
    ValueError
        Structural declaration errors (duplicate/unknown names, missing
        primary, ragged or non-finite deltas, bad alpha/min_specs/min_seeds).
    """
    a = _am_alpha(alpha)
    min_specs = _am_int(min_specs, where='min_specs', minimum=1,
                        maximum=AM_MAX_SPECS)
    min_seeds = _am_int(min_seeds, where='min_seeds', minimum=2)
    normalised = _am_normalise_specs(specs)
    short = [rec['name'] for rec in normalised
             if len(rec['deltas']) < min_seeds]
    if short:
        raise ValueError(
            f'specification(s) {_am_repr(short)} have fewer than min_seeds='
            f'{_am_repr(min_seeds)} paired seeds; a declared design minimum '
            'must be met, not silently lowered')
    primary_name = _am_text(primary, where='primary')
    lookup = {rec['name']: rec for rec in normalised}
    if primary_name not in lookup:
        raise ValueError(f'primary={primary_name!r} is not a declared '
                         'specification')
    # The primary is evaluated first only so its sign can label the other
    # records; every specification gets exactly the same arithmetic path.
    primary_stats = _am_spec_stats(
        lookup[primary_name], alpha=a, primary_sign=None,
        where=f'specs[{primary_name!r}]')
    primary_sign = int(primary_stats['sign'])
    stats = {}
    for rec in normalised:
        stats[rec['name']] = _am_spec_stats(
            rec, alpha=a, primary_sign=primary_sign,
            where=f'specs[{rec["name"]!r}]')
    names = [rec['name'] for rec in normalised]
    spec_by_name = {rec['name']: rec for rec in normalised}

    families = {}
    for family in sorted({rec['family'] for rec in normalised}):
        members = [name for name in names
                   if spec_by_name[name]['family'] == family]
        families[family] = _am_family_block(
            family, members, spec_by_name, stats, alpha=a,
            primary=primary_name)

    p_iut = max(stats[name]['p_exact_signflip'] for name in names)
    floors = [stats[name]['p_floor'] for name in names
              if stats[name]['p_floor'] is not None]
    iut_p_floor = max(floors) if floors else None
    resolution_limited_specs = [
        name for name in names
        if (stats[name]['p_floor'] is not None
            and stats[name]['p_floor'] > a)]
    resolution_limited = bool(iut_p_floor is not None
                              and iut_p_floor > a)
    sign_flips = [name for name in names
                  if not stats[name]['same_sign_as_primary']]
    not_reject = [name for name in names
                  if not stats[name]['reject_raw']]
    n_same_sign = len(names) - len(sign_flips)
    sign_agreement = n_same_sign / float(len(names))
    all_reject = not not_reject
    n_families = len(families)

    if len(names) < min_specs:
        pattern = 'insufficient_specs'
    elif primary_sign == 0:
        pattern = 'zero_effect'
    elif sign_flips:
        pattern = 'direction_unstable'
    elif resolution_limited:
        pattern = 'resolution_limited'
    elif not all_reject:
        pattern = 'precision_unstable'
    else:
        pattern = 'stable'
    status = ('ok' if pattern == 'stable'
              else 'not_estimable' if pattern == 'insufficient_specs'
              else 'warnings')
    robust = bool(pattern == 'stable')

    warnings = []
    if len(names) < min_specs:
        warnings.append(
            f'{len(names)} declared specifications; the predeclared minimum '
            f'is {min_specs}')
    if n_families > 1:
        warnings.append(
            'specifications span multiple estimand/scale families; effect '
            'magnitudes are reported per family and are never pooled')
    if sign_flips:
        warnings.append('direction differs from the primary specification: '
                        + ', '.join(sorted(sign_flips)))
    if not_reject and not sign_flips:
        warnings.append('sign is stable but not every specification rejects '
                        'its own null: ' + ', '.join(sorted(not_reject)))
    if resolution_limited:
        warnings.append(
            f'the IUT p floor {iut_p_floor!r} exceeds alpha={a!r}; at this '
            'seed count no all-specifications claim is resolvable')
    for family, block in families.items():
        if block['joint_status'] == 'not_comparable':
            warnings.append(f'family {family!r}: {block["joint_reason"]}')
        elif block['joint_status'] == 'not_estimable':
            warnings.append(f'family {family!r}: {block["joint_reason"]}')

    if robust:
        statement = (
            f'all {len(names)} declared specifications reject their own '
            f'two-sided null at alpha={a} in the primary direction '
            f'(IUT p={p_iut}); this is an intersection-union claim and needs '
            'no multiplicity correction, but it is still bounded by the '
            'declared specification set and the paired-seed assumption')
    elif pattern == 'direction_unstable':
        statement = (
            'not supported: at least one declared specification changes the '
            'sign of the primary effect, so "the effect is robust to these '
            'analytic choices" is false')
    elif pattern == 'precision_unstable':
        statement = (
            'not supported: the sign is stable but at least one declared '
            'specification does not reject its own null; a multiverse claim '
            'requires every specification to reject (IUT), not a majority')
    elif pattern == 'resolution_limited':
        statement = (
            'not resolvable: the exact sign-flip resolution floor exceeds '
            'alpha, so "all specifications non-zero" cannot be established '
            'at this seed count (this is not evidence of no effect)')
    elif pattern == 'zero_effect':
        statement = ('not supported: the primary specification has an exact '
                     'zero point estimate, so it supplies no direction')
    else:
        statement = (
            f'not supported: {len(names)} specifications are below the '
            f'predeclared minimum of {min_specs}')

    return {
        'kind': AM_KIND,
        'status': status,
        'pattern': pattern,
        'alpha': float(a),
        'primary': primary_name,
        'primary_sign': primary_sign,
        'n_specs': len(names),
        'n_families': n_families,
        'min_specs': int(min_specs),
        'min_seeds': int(min_seeds),
        'p_iut': float(p_iut),
        'iut_p_floor': (float(iut_p_floor)
                        if iut_p_floor is not None else None),
        'resolution_limited': resolution_limited,
        'resolution_limited_specs': sorted(resolution_limited_specs),
        'specs': {name: stats[name] for name in names},
        'families': families,
        'summary': {
            'n_specs': len(names),
            'n_families': n_families,
            'sign_agreement': float(sign_agreement),
            'n_same_sign': int(n_same_sign),
            'n_sign_flips': len(sign_flips),
            'n_reject_raw': len(names) - len(not_reject),
            'n_not_reject': len(not_reject),
            'n_resolution_limited': len(resolution_limited_specs),
            'all_specs_reject': bool(all_reject),
            'scale_comparable': bool(n_families == 1),
            'note': ('IUT is the all-specifications claim; each family max-T '
                     'is the weaker at-least-one claim; the two are never '
                     'substituted for each other'),
        },
        'claim': {
            'robust_across_specs': robust,
            'direction': int(primary_sign),
            'iut_p': float(p_iut),
            'iut_p_floor': (float(iut_p_floor)
                            if iut_p_floor is not None else None),
            'all_specs_reject': bool(all_reject),
            'sign_agreement': float(sign_agreement),
            'statement': statement,
            'scope': ('exploratory; bounded by the declared specification '
                      'set, the paired-seed symmetry assumption and the '
                      'reported exact resolution floor'),
        },
        'warnings': warnings,
        'policy': dict(AM_POLICY),
    }


def build_analysis_multiverse(specs, *, primary, alpha=0.05,
                              min_specs=AM_MIN_SPECS,
                              min_seeds=AM_MIN_SEEDS, created_at=None):
    """Build a replayable analysis-multiverse artifact.

    The exact declared specifications are stored in ``inputs``; the artifact
    is refused when the declaration is structurally unusable (fewer than
    ``min_specs`` entries).  A measured but non-robust result is still valid
    evidence and is written with ``status="warnings"``.
    """
    a = _am_alpha(alpha)
    min_specs = _am_int(min_specs, where='min_specs', minimum=1,
                        maximum=AM_MAX_SPECS)
    min_seeds = _am_int(min_seeds, where='min_seeds', minimum=2)
    normalised = _am_normalise_specs(specs)
    primary_name = _am_text(primary, where='primary')
    if primary_name not in {rec['name'] for rec in normalised}:
        raise ValueError(f'primary={primary_name!r} is not a declared '
                         'specification')
    declared = _am_input_specs(normalised)
    result = analysis_multiverse(
        normalised, primary=primary_name, alpha=a, min_specs=min_specs,
        min_seeds=min_seeds)
    if result['status'] == 'not_estimable':
        raise ValueError(
            'analysis_multiverse artifact needs at least min_specs declared '
            f'specifications; got {result["n_specs"]} (min_specs='
            f'{min_specs})')
    artifact = {
        'kind': AM_ARTIFACT_KIND,
        'schema': AM_SCHEMA,
        'created_at': (created_at if created_at is not None
                       else _am_time.strftime('%Y-%m-%dT%H:%M:%S')),
        'inputs': {
            'specs': declared,
            'primary': primary_name,
            'alpha': float(a),
            'min_specs': int(min_specs),
            'min_seeds': int(min_seeds),
        },
        'result': result,
    }
    artifact['content_sha256'] = _sha256_json(  # noqa: F821
        {key: value for key, value in artifact.items()
         if key not in ('created_at', 'content_sha256')})
    return artifact


def _am_forbidden_payload(obj):
    """Collect forbidden decision fields anywhere in an untrusted payload.

    Iterative with a seen set: a deeply nested or self-referential artifact
    must be reported as a tamper instead of raising ``RecursionError`` before
    validation can return a failure object.
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
                if isinstance(key, str) and key in AM_FORBIDDEN_KEYS:
                    found.append(key)
                stack.append(value)
        elif isinstance(cur, (list, tuple)):
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            stack.extend(cur)
    return found


def _am_json_equal(a, b):
    """Exact JSON-value equality that distinguishes ``True`` from ``1``.

    Numeric comparison is done directly: converting a finite-but-huge JSON
    integer (``10**400``) to float raises ``OverflowError``, and a tampered
    artifact must produce a failed validation, not an exception.  Python's
    int/float comparison is exact for arbitrarily large integers and never
    overflows.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        if set(a) != set(b):
            return False
        return all(_am_json_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple))
                and isinstance(b, (list, tuple))):
            return False
        return (len(a) == len(b)
                and all(_am_json_equal(x, y) for x, y in zip(a, b)))
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)):
        return a == b
    # A hostile artifact can carry an ndarray (or another non-JSON object)
    # where the replay produced a scalar.  ``a == b`` then returns an array
    # whose truth value is ambiguous; collapse it to a mismatch so the
    # validator reports a named failure instead of raising ValueError.
    try:
        return bool(a == b)
    except (TypeError, ValueError, OverflowError):
        return False


def _am_diff_paths(stored, replay, path='result', out=None, limit=20):
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(stored, dict) and isinstance(replay, dict):
        for key in sorted(set(stored) | set(replay), key=_am_repr):
            if key not in stored:
                out.append(f'{path}.{key}: missing from stored artifact')
            elif key not in replay:
                out.append(f'{path}.{key}: not produced by replay')
            else:
                _am_diff_paths(stored[key], replay[key],
                               f'{path}.{key}', out, limit)
            if len(out) >= limit:
                break
    elif (isinstance(stored, (list, tuple))
          and isinstance(replay, (list, tuple))):
        if len(stored) != len(replay):
            out.append(f'{path}: stored length {len(stored)} != replay '
                       f'length {len(replay)}')
        else:
            for index, (left, right) in enumerate(zip(stored, replay)):
                _am_diff_paths(left, right, f'{path}[{index}]', out, limit)
                if len(out) >= limit:
                    break
    elif not _am_json_equal(stored, replay):
        out.append(f'{path}: stored={_am_repr(stored)}, '
                   f'replay={_am_repr(replay)}')
    return out


def validate_analysis_multiverse(artifact):
    """Replay a stored analysis-multiverse artifact from its declared inputs.

    Missing digest, a non-reproducing digest, a changed estimate/p/family/
    influence field, an extra specification or a smuggled frozen-decision key
    is ``fail``.  A self-consistent but non-robust result stays ``ok`` at the
    validator level and is mapped to a warning by the audit gate.
    """
    out = {'kind': 'analysis_multiverse_validation', 'schema': 1,
           'status': 'fail', 'problems': [], 'warnings': [],
           'content_sha256': None, 'result': None, 'result_status': None,
           'pattern': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if (not isinstance(artifact.get('kind'), str)
            or artifact.get('kind') != AM_ARTIFACT_KIND):
        out['problems'].append(
            f'artifact.kind must be {AM_ARTIFACT_KIND!r}, got '
            f'{_am_repr(artifact.get("kind"))}')
    schema = artifact.get('schema')
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != AM_SCHEMA):
        out['problems'].append(
            f'artifact.schema must be {AM_SCHEMA}, got {_am_repr(schema)}')
    stolen = sorted(set(_am_forbidden_payload(artifact)), key=_am_repr)
    if stolen:
        out['problems'].append(
            'artifact carries frozen-decision field(s) that an exploratory '
            f'multiverse readout must not contain: {stolen}')
    allowed_top = {'kind', 'schema', 'created_at', 'inputs', 'result',
                   'content_sha256'}
    extra_top = sorted((key for key in artifact if key not in allowed_top),
                       key=_am_repr)
    if extra_top:
        out['problems'].append(
            'artifact carries field(s) outside the replay schema: '
            + ', '.join(_am_repr(key) for key in extra_top))
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        inputs = {}
    missing = [key for key in AM_INPUT_KEYS if key not in inputs]
    if missing:
        out['problems'].append(
            f'artifact.inputs is missing required key(s): {missing}')
    extra_inputs = sorted((key for key in inputs
                           if key not in AM_INPUT_KEYS), key=_am_repr)
    if extra_inputs:
        out['problems'].append(
            'artifact.inputs carries field(s) outside the schema: '
            + ', '.join(_am_repr(key) for key in extra_inputs))
    recorded = artifact.get('content_sha256')
    out['content_sha256'] = recorded if isinstance(recorded, str) else None
    if not isinstance(recorded, str) or not recorded:
        out['problems'].append('artifact.content_sha256 must be a non-empty '
                               'string')
    else:
        body = {key: value for key, value in artifact.items()
                if key not in ('created_at', 'content_sha256')}
        try:
            replay_sha = _sha256_json(body)  # noqa: F821
        except Exception as exc:
            out['problems'].append(
                f'artifact cannot be hashed for replay: '
                f'{type(exc).__name__}: {exc}')
        else:
            if replay_sha != recorded:
                out['problems'].append(
                    'artifact.content_sha256 does not match the recorded '
                    'body (the artifact was edited after it was built)')
    if out['problems']:
        return out
    try:
        stored_specs = inputs.get('specs')
        normalised = _am_specs_from_inputs(stored_specs)
        canonical = _am_input_specs(normalised)
        if not _am_json_equal(canonical, stored_specs):
            out['problems'].append(
                'artifact.inputs.specs is not in canonical order/seed form; '
                'the declaration was edited or was never built by '
                'build_analysis_multiverse')
        replay = analysis_multiverse(
            normalised,
            primary=inputs.get('primary'),
            alpha=inputs.get('alpha'),
            min_specs=inputs.get('min_specs'),
            min_seeds=inputs.get('min_seeds'))
    except Exception as exc:
        out['problems'].append(
            f'artifact inputs cannot be replayed: {type(exc).__name__}: {exc}')
        return out
    out['result'] = replay
    out['result_status'] = replay.get('status')
    out['pattern'] = replay.get('pattern')
    stored_result = artifact.get('result')
    if not isinstance(stored_result, dict):
        out['problems'].append('artifact.result must be an object')
        return out
    extra_result = sorted((key for key in stored_result
                           if key not in AM_RESULT_KEYS), key=_am_repr)
    if extra_result:
        out['problems'].append(
            'artifact.result carries field(s) outside the schema: '
            + ', '.join(_am_repr(key) for key in extra_result))
    missing_result = [key for key in AM_RESULT_KEYS if key not in stored_result]
    if missing_result:
        out['problems'].append(
            f'artifact.result is missing field(s): {missing_result}')
    stored_specs_block = stored_result.get('specs')
    if isinstance(stored_specs_block, dict):
        for name, row in sorted(stored_specs_block.items(),
                              key=lambda kv: _am_repr(kv[0])):
            if not isinstance(row, dict):
                out['problems'].append(
                    f'result.specs[{_am_repr(name)}] is not an object')
                continue
            extra = sorted(set(row) - set(AM_SPEC_FIELDS), key=_am_repr)
            if extra:
                out['problems'].append(
                    f'result.specs[{_am_repr(name)}] carries field(s) outside '
                    'the schema: ' + ', '.join(_am_repr(k) for k in extra))
            missing = [key for key in AM_SPEC_FIELDS if key not in row]
            if missing:
                out['problems'].append(
                    f'result.specs[{_am_repr(name)}] is missing field(s): '
                    f'{missing}')
    elif stored_specs_block is not None:
        out['problems'].append('artifact.result.specs must be an object')
    stored_families = stored_result.get('families')
    if isinstance(stored_families, dict):
        for name, row in sorted(stored_families.items(),
                              key=lambda kv: _am_repr(kv[0])):
            if not isinstance(row, dict):
                out['problems'].append(
                    f'result.families[{_am_repr(name)}] is not an object')
                continue
            extra = sorted(set(row) - set(AM_FAMILY_FIELDS), key=_am_repr)
            if extra:
                out['problems'].append(
                    f'result.families[{_am_repr(name)}] carries field(s) '
                    'outside the schema: '
                    + ', '.join(_am_repr(k) for k in extra))
            missing = [key for key in AM_FAMILY_FIELDS if key not in row]
            if missing:
                out['problems'].append(
                    f'result.families[{_am_repr(name)}] is missing field(s): '
                    f'{missing}')
    elif stored_families is not None:
        out['problems'].append('artifact.result.families must be an object')
    try:
        diffs = _am_diff_paths(stored_result, replay)
    except RecursionError:
        out['problems'].append(
            'artifact.result nesting is too deep to replay')
        diffs = []
    if diffs:
        out['problems'].append(
            'artifact.result does not replay from its declared inputs: '
            + '; '.join(diffs))
    if not out['problems']:
        out['status'] = 'ok'
        if replay['pattern'] != 'stable':
            out['warnings'].append(
                f'measured multiverse pattern is {replay["pattern"]!r}, not '
                'a supported robustness claim')
    return out


def write_analysis_multiverse(outdir, artifact, *, overwrite=False):
    """Atomically persist ``analysis_multiverse.json``; refuse silent replacement."""
    if isinstance(outdir, _am_os.PathLike):
        outdir = _am_os.fspath(outdir)
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_analysis_multiverse(artifact)
    path = _am_os.path.join(outdir, 'analysis_multiverse.json')
    result = artifact.get('result') if isinstance(artifact, dict) else None
    out = {'kind': 'write_analysis_multiverse', 'path': path,
           'written': False, 'unchanged': False, 'conflict': False,
           'status': verdict['status'],
           'artifact_status': (result.get('status')
                               if isinstance(result, dict) else None),
           'problems': list(verdict['problems'])}
    if verdict['status'] != 'ok':
        return out
    if _am_os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as handle:
                old = _am_json.load(handle)
        except Exception:
            old = None
        if (isinstance(old, dict)
                and old.get('content_sha256')
                == artifact.get('content_sha256')):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different analysis_multiverse.json already exists; pass '
                'overwrite=True only after re-running the analysis')
            return out
    _am_os.makedirs(outdir, exist_ok=True)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as handle:
            _am_json.dump(artifact, handle, ensure_ascii=False, indent=1,
                          allow_nan=False)
            handle.write('\n')
    except Exception as exc:
        try:
            _am_os.remove(tmp)
        except OSError:
            pass
        out['status'] = 'fail'
        out['problems'].append(
            f'artifact cannot be serialised: {type(exc).__name__}: {exc}')
        return out
    _am_os.replace(tmp, path)
    out['written'] = True
    return out


def analysis_multiverse_markdown(result):
    """Render a compact specification table for a report appendix."""
    if not isinstance(result, dict):
        return '| specification | family | n | mean | p | sign |\n|---|---|---|---|---|---|\n'
    lines = ['| specification | family | n | mean | p | sign |',
             '|---|---|---|---|---|---|']
    for name, row in sorted((result.get('specs') or {}).items()):
        sign = {1: '+', -1: '-', 0: '0'}.get(row.get('sign'), '?')
        mean = row.get('mean')
        p = row.get('p_exact_signflip')
        lines.append(
            f'| {name} | {row.get("family")} | {row.get("n_seeds")} | '
            f'{mean:.6g} | {p:.4g} | {sign} |')
    claim = result.get('claim') or {}
    lines.append('')
    lines.append(f'pattern: `{result.get("pattern")}`; '
                 f'IUT p={claim.get("iut_p")}; '
                 f'robust={claim.get("robust_across_specs")}')
    return '\n'.join(lines)
