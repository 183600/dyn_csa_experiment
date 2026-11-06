# ---- 事先说好「看几次」：精确符号检验的群序列设计、alpha 消耗与发射前门 ----
#
# 这个仓库已经能**事后**纠正重复查看：`look_history.json` 记录每次补种子后的
# 种子集合，`audit_gates.py looks` 用 exact max-over-looks 重算 p。但事后纠正
# 只能把已经发生的 optional stopping 变保守，不能回答一个更早的问题：
# **这个面板本来允许看几次、每次在什么 n 上看、每次的拒绝边界是多少？**
#
# 群序列（group-sequential）设计是实验设计/准备阶段的经典答案。本片段给
# 「按种子补样」的场景一个可重放的精确版本：
#
#   * 声明 look schedule：n_1 < ... < n_K = n_max，每个 look 只看前 n_k 个
#     种子（嵌套，不换种子）；
#   * 统计量用**符号和** W_{n_k} = sum_{i<=n_k} sign(d_i)。在零假设下
#     sign(d_i) 是独立的 Rademacher 变量，所以 (W_{n_1},...,W_{n_K}) 的联合
#     分布可以**整数精确枚举**（动态规划 + 二项系数），不需要正态近似，
#     也不需要 Monte Carlo；
#   * alpha 消耗函数（Pocock / O'Brien-Fleming / linear / custom / none）
#     给出每个 look 的累计名义 alpha；边界取能放进的**最小整数绝对阈值**
#     b_k；可选强制 z_k = b_k/sqrt(n_k) 不高于前一个 look（避免后看更难
#     拒绝），默认关闭以免离散/前载计划不可行，z 上升时显式 warning；
#     实际 FWER 用精确路径计数重算，离散导致的实际值可以低于名义值；
#   * 给定 pilot（validation）符号率 q，精确计算 power、每个 look 的停止
#     概率、期望样本量；给 target power 反解最小 q；若提供 pilot SD，再把 q
#     映回 log-PPL 尺度上的近似 MDE；
#   * 条件 power：在某个 look 观察到 W 后，按假设 q 计算最终跨过边界的概率。
#
# SCOPE: 设计/准备阶段读数。它不训练、不读任何 split、不产生确认 p 值，也不
# 改变冻结的 p/Holm/confirmed。它是**符号检验**的精确群序列计划；仓库的确认
# 检验是均值型 exact sign-flip，因此这里的 power 与 `signflip_power` 一样是
# 计划投影，不是对最终 p 值的承诺。

import datetime as _seq_dt
import hashlib
import json
import math
import numbers
import os
import statistics
import time
from fractions import Fraction

import numpy as np

try:  # Python >=3.8
    from statistics import NormalDist as _SeqNormalDist
except Exception:  # pragma: no cover - the project requires 3.12
    _SeqNormalDist = None


SEQUENTIAL_DESIGN_KIND = 'sequential_design'
SEQUENTIAL_DESIGN_SCHEMA = 1
SEQUENTIAL_MAX_N = 1024
SEQUENTIAL_MAX_LOOKS = 32
SEQUENTIAL_SIDES = ('two', 'greater', 'less')
SEQUENTIAL_SPENDING = ('pocock', 'obrien_fleming', 'obf', 'linear',
                       'custom', 'none')

SEQUENTIAL_DESIGN_POLICY = {
    'version': 1,
    'sides': SEQUENTIAL_SIDES,
    'spending': SEQUENTIAL_SPENDING,
    'role': (
        'prospective group-sequential sign-test design: a pre-registered '
        'look schedule, an alpha-spending function and exact integer '
        'boundaries over nested seed prefixes.  It is a design/preparation '
        'readout, not a confirmatory test.'),
    'null_distribution': (
        'sign(d_i) iid Rademacher under H0, so the joint law of the sign sums '
        'at the declared look sizes is enumerated exactly by dynamic '
        'programming; no normal approximation and no Monte Carlo.'),
    'spending_rule': (
        'the smallest integer absolute boundary whose exact cumulative '
        'crossing probability fits the nominal cumulative spend; the '
        'standardised boundary z_k = b_k/sqrt(n_k) is optionally forced not '
        'to increase (monotone_z), and an increase is otherwise reported.'),
    'power_rule': (
        'power is exact for the sign test under a constant per-seed sign '
        'probability q.  It is a planning projection for the mean-based '
        'exact sign-flip endpoint, just like signflip_power.'),
    'boundary': (
        'this fragment never reads a split, never trains, never produces a '
        'confirmatory p/Holm/confirmed verdict, and cannot make an already '
        'unplanned look planned.  A missing or late plan is reported, not '
        'repaired.'),
}

SEQUENTIAL_DESIGN_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'revision', 'supersedes', 'reason', 'content_sha256',
)
SEQUENTIAL_DESIGN_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'frozen_decision', 'final_decision', 'holm',
})
SEQUENTIAL_DESIGN_INPUTS = (
    'n_max', 'look_sizes', 'alpha', 'sides', 'spending',
    'cumulative_spend', 'family_size', 'pilot_sign_rate', 'pilot_sd',
    'pilot_ppl', 'q_for_power', 'target_power', 'power_curve_q',
    'monotone_z', 'n_pilot', 'pilot_n_positive',
    'pilot_n_negative', 'pilot_n_zero',
)


def _seq_repr(value, limit=160):
    """Bounded, exception-safe repr for diagnostics."""
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _seq_int(value, name, *, minimum=None, maximum=None):
    """Strict non-boolean integer for a design parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    out = int(value)
    if minimum is not None and out < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {out}')
    if maximum is not None and out > int(maximum):
        raise ValueError(f'{name} must be <= {int(maximum)}, got {out}')
    return out


def _seq_probability(value, name, *, lo=0.0, hi=1.0,
                     lo_open=False, hi_open=False):
    """Strict finite probability for a design parameter."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise TypeError(f'{name} must be a real probability, got '
                        f'{type(value).__name__}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f'{name} must be a real probability, got '
                        f'{_seq_repr(value)}') from None
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {_seq_repr(value)}')
    if (out <= lo if lo_open else out < lo) or (out >= hi if hi_open else out > hi):
        raise ValueError(f'{name}={_seq_repr(value)} is outside the allowed range')
    return out


def _seq_positive(value, name):
    """Strict finite positive number (a PPL or a standard deviation)."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise TypeError(f'{name} must be a positive number, got '
                        f'{type(value).__name__}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f'{name} must be a positive number, got '
                        f'{_seq_repr(value)}') from None
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f'{name} must be a finite positive number, got '
                         f'{_seq_repr(value)}')
    return out


def _seq_text(value, name, *, allow_none=False, limit=240):
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string, got {type(value).__name__}')
    text = value.strip()
    if not text:
        raise ValueError(f'{name} must be non-empty')
    return text[:limit]


def _seq_bool(value, name):
    if isinstance(value, bool):
        return bool(value)
    raise TypeError(f'{name} must be a boolean, got {type(value).__name__}')


def _seq_canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, default=str)


def _seq_sha(obj):
    return hashlib.sha256(_seq_canonical(obj).encode('utf-8')).hexdigest()


def _seq_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('content_sha256', 'created_at', 'revision',
                         'supersedes', 'reason')}
    return _seq_sha(body)


def _seq_forbidden_paths(obj, path='artifact'):
    """Return every path whose key is a frozen-decision field."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key)
            child = f'{path}.{key_text}'
            if key_text in SEQUENTIAL_DESIGN_FORBIDDEN_KEYS:
                found.append(child)
            found.extend(_seq_forbidden_paths(value, child))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_seq_forbidden_paths(value, f'{path}[{index}]'))
    return found


def _seq_crosses(w, b, sides):
    if b is None:
        return False
    if sides == 'two':
        return abs(int(w)) >= int(b)
    if sides == 'greater':
        return int(w) >= int(b)
    return int(w) <= -int(b)


def _seq_cross_count(surv, b, sides):
    if b is None:
        return 0
    return int(sum(c for w, c in surv.items() if _seq_crosses(w, b, sides)))


def _seq_noncross(surv, b, sides):
    if b is None:
        return dict(surv)
    return {w: c for w, c in surv.items() if not _seq_crosses(w, b, sides)}


def _seq_step_counts(surv, steps):
    """Exact convolution of surviving path counts with ``steps`` signs.

    ``surv`` maps a current sign sum W to the number of sign sequences that
    reached it without crossing an earlier boundary.  Adding ``m`` signs adds
    ``2*s - m`` with multiplicity C(m, s), so every count stays an exact
    integer.
    """
    steps = int(steps)
    if steps < 0:
        raise ValueError('steps must be non-negative')
    if steps == 0:
        return dict(surv)
    binom = [math.comb(steps, s) for s in range(steps + 1)]
    out = {}
    for w, count in surv.items():
        if not count:
            continue
        for s, ways in enumerate(binom):
            nw = int(w) + 2 * s - steps
            out[nw] = out.get(nw, 0) + int(count) * int(ways)
    return out


def _seq_state_prob(count, w, n, q):
    """Probability of one surviving state at look n under sign probability q."""
    positives = (int(n) + int(w)) // 2
    if positives < 0 or positives > int(n):
        return 0.0
    if (int(n) + int(w)) % 2:
        return 0.0
    try:
        return (float(count) * (float(q) ** positives)
                * ((1.0 - float(q)) ** (int(n) - positives)))
    except (OverflowError, ValueError):
        return 0.0


def _seq_incremental_prob(count, w, w0, n, n0, q):
    """Probability of a future state given a fixed state at an earlier look."""
    steps = int(n) - int(n0)
    positives = (steps + (int(w) - int(w0))) // 2
    if positives < 0 or positives > steps:
        return 0.0
    if (steps + (int(w) - int(w0))) % 2:
        return 0.0
    try:
        return (float(count) * (float(q) ** positives)
                * ((1.0 - float(q)) ** (steps - positives)))
    except (OverflowError, ValueError):
        return 0.0


def _seq_spend_fraction(name, t, alpha, sides):
    """Cumulative alpha spent by fraction ``t`` of the maximum sample size.

    The return value is a **fraction of the comparison alpha** (0..1), so the
    caller can multiply by the Bonferroni-adjusted comparison alpha once.
    """
    t = float(t)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    name = str(name)
    if name == 'pocock':
        return float(math.log1p((math.e - 1.0) * t))
    if name in ('obrien_fleming', 'obf'):
        if _SeqNormalDist is None:  # pragma: no cover
            raise RuntimeError('statistics.NormalDist is unavailable')
        nd = _SeqNormalDist()
        if sides == 'two':
            z = nd.inv_cdf(1.0 - float(alpha) / 2.0)
            return float(2.0 * (1.0 - nd.cdf(z / math.sqrt(t))) / float(alpha))
        z = nd.inv_cdf(1.0 - float(alpha))
        return float((1.0 - nd.cdf(z / math.sqrt(t))) / float(alpha))
    if name == 'linear':
        return float(t)
    raise ValueError(f'unknown spending function {name!r}')


def _seq_nominal_fractions(spending, look_sizes, alpha, sides,
                           cumulative_spend=None):
    if spending == 'custom':
        if not isinstance(cumulative_spend, (list, tuple)):
            raise TypeError('cumulative_spend must be a list for spending="custom"')
        if len(cumulative_spend) != len(look_sizes):
            raise ValueError(
                'cumulative_spend must have one entry per look '
                f'({len(look_sizes)}), got {len(cumulative_spend)}')
        out = []
        for index, raw in enumerate(cumulative_spend):
            value = _seq_probability(
                raw, f'cumulative_spend[{index}]', lo=0.0, hi=1.0)
            out.append(value)
        if any(b < a - 1e-12 for a, b in zip(out, out[1:])):
            raise ValueError('cumulative_spend must be non-decreasing')
        if abs(out[-1] - 1.0) > 1e-12:
            raise ValueError('cumulative_spend[-1] must be 1.0 (the full alpha)')
        return out
    if cumulative_spend is not None:
        raise ValueError('cumulative_spend is only valid with spending="custom"')
    n_max = int(look_sizes[-1])
    if spending == 'none':
        return [0.0] * (len(look_sizes) - 1) + [1.0]
    return [_seq_spend_fraction(spending, int(n) / n_max, alpha, sides)
            for n in look_sizes]


def _seq_tail_prob(n, b, sides):
    """Exact one/two-sided tail probability of |W|>=b or W>=b at look n."""
    n = int(n)
    b = int(b)
    if b <= 0:
        return 1.0
    if b > n:
        return 0.0
    total = 0
    for w in range(b, n + 1, 2):
        total += math.comb(n, (n + w) // 2)
    prob = float(Fraction(total, 1 << n))
    return min(1.0, 2.0 * prob) if sides == 'two' else prob


def sequential_boundaries(look_sizes, *, alpha=0.05, sides='two',
                          spending='obrien_fleming', cumulative_spend=None,
                          family_size=1, monotone_z=False):
    """Exact group-sequential sign-test boundaries over nested seed prefixes.

    Parameters
    ----------
    look_sizes : sequence of int
        Strictly increasing analysis sizes; the last must be the maximum n.
    alpha : float
        Family-wise alpha for the comparison(s).
    sides : {'two', 'greater', 'less'}
        Alternative direction.  ``greater`` rejects W>=b (arm A improves);
        ``less`` rejects W<=-b (arm B improves); ``two`` rejects |W|>=b.
    spending : {'pocock','obrien_fleming','obf','linear','custom','none'}
        Alpha-spending function.  ``none`` spends everything at the final look.
    cumulative_spend : list of float, optional
        For ``custom``: fractions of comparison alpha at each look, ending 1.0.
    family_size : int
        Number of comparisons sharing ``alpha``.  This implementation uses a
        Bonferroni split (``alpha / family_size`` per comparison); it is
        deliberately not called a Holm-sequential procedure.
    monotone_z : bool
        Force ``b_k/sqrt(n_k)`` not to increase across looks.

    Returns
    -------
    dict
        Boundaries, exact FWER, nominal spending and warnings.  No p-value or
        confirmation verdict is produced.
    """
    if isinstance(look_sizes, (str, bytes, bytearray)) or not isinstance(
            look_sizes, (list, tuple)):
        raise TypeError('look_sizes must be a list/tuple of integers')
    looks = []
    for index, raw in enumerate(look_sizes):
        looks.append(_seq_int(raw, f'look_sizes[{index}]', minimum=1,
                              maximum=SEQUENTIAL_MAX_N))
    if not looks:
        raise ValueError('look_sizes must not be empty')
    if len(looks) > SEQUENTIAL_MAX_LOOKS:
        raise ValueError(
            f'look_sizes has {len(looks)} looks; the supported cap is '
            f'{SEQUENTIAL_MAX_LOOKS}')
    if any(b <= a for a, b in zip(looks, looks[1:])):
        raise ValueError(f'look_sizes must be strictly increasing, got {looks!r}')
    alpha = _seq_probability(alpha, 'alpha', lo=0.0, hi=1.0,
                             lo_open=True, hi_open=True)
    sides = _seq_text(sides, 'sides', limit=16)
    if sides not in SEQUENTIAL_SIDES:
        raise ValueError(f'sides must be one of {SEQUENTIAL_SIDES}, got {sides!r}')
    spending = _seq_text(spending, 'spending', limit=32)
    if spending not in SEQUENTIAL_SPENDING:
        raise ValueError(
            f'spending must be one of {SEQUENTIAL_SPENDING}, got {spending!r}')
    family_size = _seq_int(family_size, 'family_size', minimum=1,
                           maximum=100000)
    monotone_z = _seq_bool(monotone_z, 'monotone_z')
    comparison_alpha = float(alpha) / float(family_size)
    fractions = _seq_nominal_fractions(
        spending, looks, comparison_alpha, sides,
        cumulative_spend=cumulative_spend)

    surv = {0: 1}
    prev_n = 0
    prev_z = None
    actual_cum = Fraction(0, 1)
    rows = []
    warnings = []
    for index, (n_k, fraction) in enumerate(zip(looks, fractions)):
        m = int(n_k) - prev_n
        surv = _seq_step_counts(surv, m)
        target = float(comparison_alpha) * float(fraction)
        # A standardised boundary must not become harder to cross at a
        # later look.  ``b/sqrt(n)`` is therefore capped by the previous
        # look's z; within that cap we take the smallest integer boundary
        # that fits the remaining alpha.
        b_max = int(n_k)
        if monotone_z and prev_z is not None:
            b_max = min(b_max, int(math.floor(
                float(prev_z) * math.sqrt(n_k) + 1e-12)))
        chosen = None
        for b in range(1, b_max + 1):
            cross = _seq_cross_count(surv, b, sides)
            cum = actual_cum + Fraction(int(cross), 1 << int(n_k))
            if float(cum) <= target + 1e-12:
                chosen = (int(b), int(cross), cum)
                break
        if chosen is None:
            row = {
                'look': index + 1,
                'n': int(n_k),
                't': float(n_k) / float(looks[-1]),
                'b': None,
                'z_boundary': None,
                'nominal_p_boundary': None,
                'nominal_cumulative_spend': float(target),
                'incremental_nominal_spend': float(
                    target - (float(comparison_alpha) * float(fractions[index - 1])
                              if index else 0.0)),
                'actual_cumulative_fwer': float(actual_cum),
                'incremental_actual': 0.0,
                'crossing_count': 0,
                'attainable': False,
                'note': ('no integer boundary fits the remaining alpha at '
                         'this look; this look cannot reject'),
            }
            warnings.append(
                f'look {index + 1} (n={n_k}) cannot reject: the smallest '
                'attainable boundary already spends more than the remaining '
                'alpha')
        else:
            b, cross, cum = chosen
            prev_cum = actual_cum
            actual_cum = cum
            row = {
                'look': index + 1,
                'n': int(n_k),
                't': float(n_k) / float(looks[-1]),
                'b': int(b),
                'z_boundary': float(b) / math.sqrt(float(n_k)),
                'nominal_p_boundary': _seq_tail_prob(n_k, b, sides),
                'nominal_cumulative_spend': float(target),
                'incremental_nominal_spend': float(
                    target - (float(comparison_alpha) * float(fractions[index - 1])
                              if index else 0.0)),
                'actual_cumulative_fwer': float(actual_cum),
                'incremental_actual': float(actual_cum - prev_cum),
                'crossing_count': int(cross),
                'attainable': True,
                'note': None,
            }
            current_z = float(b) / math.sqrt(float(n_k))
            if prev_z is not None and current_z > float(prev_z) + 1e-12:
                warnings.append(
                    f'look {index + 1} (n={n_k}) has z={current_z:.6g}, '
                    f'higher than the previous look z={float(prev_z):.6g}; '
                    'the plan front-loads alpha, so a later look is harder '
                    'to reject')
            prev_z = current_z
            surv = _seq_noncross(surv, b, sides)
            if float(actual_cum) + 1e-12 < float(target):
                warnings.append(
                    f'look {index + 1} (n={n_k}) is discrete: actual '
                    f'cumulative FWER {float(actual_cum):.6g} is below the '
                    f'nominal {float(target):.6g}')
        rows.append(row)
        prev_n = int(n_k)

    fwer = float(actual_cum)
    final = rows[-1]
    result = {
        'status': 'ok' if final['b'] is not None else 'not_estimable',
        'alpha': float(alpha),
        'family_size': int(family_size),
        'comparison_alpha': float(comparison_alpha),
        'sides': sides,
        'spending': spending,
        'n_max': int(looks[-1]),
        'look_sizes': [int(n) for n in looks],
        'n_looks': len(looks),
        'boundaries': rows,
        'fwer': float(fwer),
        'fwer_nominal': float(comparison_alpha),
        'fwer_ok': bool(fwer <= float(comparison_alpha) + 1e-12),
        'familywise_fwer_bound': float(min(1.0, float(fwer) * family_size)),
        'minimum_attainable_p_final': (
            min(1.0, 2.0 * math.ldexp(1.0, -int(looks[-1])))
            if sides == 'two' else min(1.0, math.ldexp(1.0, -int(looks[-1])))),
        'warnings': warnings,
    }
    if result['status'] == 'not_estimable':
        result['reason'] = (
            'the final look cannot attain the declared alpha with an integer '
            'sign-test boundary; the panel is resolution-limited')
    if family_size > 1:
        result['warnings'].append(
            'family_size>1 uses a Bonferroni split of alpha, not a '
            'Holm-sequential procedure; the reported FWER is per comparison')
    if len(looks) == 1:
        result['warnings'].append(
            'a single look is a fixed design; the spending function has no '
            'interim decision to allocate')
    return result


def _seq_power(rows, q, sides):
    """Exact sign-test power and stopping distribution for one q."""
    q = float(q)
    surv = {0: 1}
    prev_n = 0
    stop = []
    for row in rows:
        n_k = int(row['n'])
        surv = _seq_step_counts(surv, n_k - prev_n)
        b = row.get('b')
        if b is None:
            prob = 0.0
        else:
            prob = sum(_seq_state_prob(count, w, n_k, q)
                       for w, count in surv.items()
                       if _seq_crosses(w, b, sides))
        stop.append(float(prob))
        surv = _seq_noncross(surv, b, sides)
        prev_n = n_k
    no_stop = float(max(0.0, 1.0 - sum(stop)))
    n_max = int(rows[-1]['n'])
    # A path that never crosses a boundary is still observed to n_max.  Omitting
    # that term made the expected sample size collapse toward the early looks.
    expected_n = float(sum(row['n'] * p for row, p in zip(rows, stop))
                       + n_max * no_stop)
    return {
        'power': float(sum(stop)),
        'prob_stop_by_look': stop,
        'expected_n': expected_n,
        'prob_no_stop': no_stop,
    }


def sequential_conditional_power(design, *, look, observed_w, q):
    """Exact sign-test conditional power at a declared interim look.

    ``design`` may be a built artifact or a boundary list.  ``look`` is 1-based.
    Conditioning on the observed sign sum at that look (and on not having
    crossed earlier), the future signs are still iid with the assumed q, so the
    same integer propagation gives an exact conditional probability for the
    sign-test plan.  It is a planning/rehearsal readout, not a confirmatory
    p-value.
    """
    if isinstance(design, dict):
        result = design.get('result') if isinstance(design.get('result'), dict) else design
        rows = result.get('boundaries')
        sides = result.get('sides', 'two')
    else:
        rows = design
        sides = 'two'
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('design has no boundary list')
    look = _seq_int(look, 'look', minimum=1, maximum=len(rows))
    if isinstance(observed_w, (bool, np.bool_)) or not isinstance(observed_w, numbers.Integral):
        raise TypeError('observed_w must be an integer sign sum')
    observed_w = int(observed_w)
    q = _seq_probability(q, 'q')
    index = int(look) - 1
    row = rows[index]
    n_k = int(row['n'])
    if (n_k + observed_w) % 2:
        raise ValueError(
            f'observed_w={observed_w} has the wrong parity for n={n_k}')
    out = {
        'status': 'ok',
        'look': int(look),
        'n': n_k,
        'observed_w': observed_w,
        'q': float(q),
        'sides': sides,
        'already_crossed': bool(_seq_crosses(observed_w, row.get('b'), sides)),
        'conditional_power': 0.0,
        'future_looks': max(0, len(rows) - int(look)),
        'note': ('exact sign-test conditional power under a constant per-seed '
                 'sign probability; not a confirmatory p-value'),
    }
    if out['already_crossed']:
        out['conditional_power'] = 1.0
        return out
    if int(look) == len(rows):
        return out
    surv = {observed_w: 1}
    prev_n = n_k
    total = 0.0
    for future in rows[int(look):]:
        surv = _seq_step_counts(surv, int(future['n']) - prev_n)
        b = future.get('b')
        if b is not None:
            total += sum(_seq_incremental_prob(
                count, w, observed_w, int(future['n']), n_k, q)
                for w, count in surv.items()
                if _seq_crosses(w, b, sides))
        surv = _seq_noncross(surv, b, sides)
        prev_n = int(future['n'])
    out['conditional_power'] = float(total)
    return out


def _seq_min_q_for_power(rows, sides, target_power):
    target_power = float(target_power)
    if _seq_power(rows, 0.5, sides)['power'] >= target_power:
        return 0.5
    lo, hi = 0.5, 1.0 - 1e-12
    if _seq_power(rows, hi, sides)['power'] < target_power:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _seq_power(rows, mid, sides)['power'] >= target_power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _seq_pilot_summary(deltas):
    """Sign rate and SD of a validation pilot (no test split, no imputation)."""
    if isinstance(deltas, (str, bytes, bytearray, dict)):
        raise TypeError('pilot_deltas must be an iterable of numbers')
    try:
        raw = list(deltas)
    except TypeError:
        raise TypeError('pilot_deltas must be an iterable of numbers') from None
    values = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f'pilot_deltas[{index}] is a boolean, not a measurement')
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise TypeError(
                f'pilot_deltas[{index}] is not numeric: {_seq_repr(value)}') from None
        if not math.isfinite(number):
            raise ValueError(f'pilot_deltas[{index}] is not finite')
        values.append(number)
    n = len(values)
    if n == 0:
        raise ValueError('pilot_deltas must not be empty')
    positives = int(sum(1 for x in values if x > 0.0))
    negatives = int(sum(1 for x in values if x < 0.0))
    zeros = int(n - positives - negatives)
    sign_rate = float((positives + 0.5 * zeros) / n)
    sd = float(statistics.stdev(values)) if n >= 2 else None
    return {
        'n_pilot': int(n),
        'n_positive': positives,
        'n_negative': negatives,
        'n_zero': zeros,
        'pilot_sign_rate': sign_rate,
        'pilot_sd': sd,
    }


def _seq_normal_quantile(p):
    if _SeqNormalDist is None:  # pragma: no cover
        return None
    try:
        value = float(_SeqNormalDist().inv_cdf(float(p)))
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def build_sequential_design(
        *, n_max, look_sizes=None, alpha=0.05, sides='two',
        spending='obrien_fleming', cumulative_spend=None, family_size=1,
        pilot_deltas=None, pilot_sign_rate=None, pilot_sd=None,
        pilot_ppl=None, q_for_power=None, target_power=0.80,
        power_curve_q=None, monotone_z=False, n_pilot=None,
        pilot_n_positive=None, pilot_n_negative=None, pilot_n_zero=None,
        source=None, created_at=None,
        revision=0, supersedes=None, reason=None):
    """Build one replayable group-sequential design artifact.

    The returned object is a *pre-run* declaration: it contains the look
    schedule, the alpha-spending rule, exact integer boundaries, exact FWER,
    exact sign-test power and the expected number of seeds.  It is not a
    confirmatory result and carries no frozen-decision field.
    """
    n_max = _seq_int(n_max, 'n_max', minimum=2, maximum=SEQUENTIAL_MAX_N)
    if look_sizes is None:
        look_sizes = [n_max]
    if isinstance(look_sizes, (str, bytes, bytearray)) or not isinstance(
            look_sizes, (list, tuple)):
        raise TypeError('look_sizes must be a list/tuple of integers')
    looks = [_seq_int(v, f'look_sizes[{i}]', minimum=1,
                      maximum=SEQUENTIAL_MAX_N)
             for i, v in enumerate(look_sizes)]
    if not looks:
        raise ValueError('look_sizes must not be empty')
    if looks[-1] != n_max:
        raise ValueError(
            f'the final look must equal n_max ({n_max}), got {looks[-1]}')
    alpha = _seq_probability(alpha, 'alpha', lo=0.0, hi=1.0,
                             lo_open=True, hi_open=True)
    sides = _seq_text(sides, 'sides', limit=16)
    if sides not in SEQUENTIAL_SIDES:
        raise ValueError(f'sides must be one of {SEQUENTIAL_SIDES}, got {sides!r}')
    spending = _seq_text(spending, 'spending', limit=32)
    if spending not in SEQUENTIAL_SPENDING:
        raise ValueError(
            f'spending must be one of {SEQUENTIAL_SPENDING}, got {spending!r}')
    family_size = _seq_int(family_size, 'family_size', minimum=1,
                           maximum=100000)
    target_power = _seq_probability(
        target_power, 'target_power', lo=0.0, hi=1.0, lo_open=True, hi_open=True)
    monotone_z = _seq_bool(monotone_z, 'monotone_z')

    summary = None
    if pilot_deltas is not None:
        summary = _seq_pilot_summary(pilot_deltas)
    if pilot_sign_rate is not None:
        rate = _seq_probability(pilot_sign_rate, 'pilot_sign_rate')
        if summary is not None and abs(summary['pilot_sign_rate'] - rate) > 1e-9:
            raise ValueError(
                'pilot_sign_rate disagrees with the sign rate computed from '
                f'pilot_deltas ({rate} vs {summary["pilot_sign_rate"]})')
    else:
        rate = None
    if summary is not None:
        rate = summary['pilot_sign_rate']
    if n_pilot is not None:
        n_pilot = _seq_int(n_pilot, 'n_pilot', minimum=0,
                           maximum=SEQUENTIAL_MAX_N * 1000)
        if summary is not None and int(n_pilot) != int(summary['n_pilot']):
            raise ValueError(
                f'n_pilot={n_pilot} disagrees with pilot_deltas length '
                f'{summary["n_pilot"]}')
    elif summary is not None:
        n_pilot = int(summary['n_pilot'])
    for _name, _value in (('pilot_n_positive', pilot_n_positive),
                          ('pilot_n_negative', pilot_n_negative),
                          ('pilot_n_zero', pilot_n_zero)):
        if _value is not None:
            _seq_int(_value, _name, minimum=0,
                     maximum=SEQUENTIAL_MAX_N * 1000)
    if summary is not None:
        if (pilot_n_positive is not None
                and int(pilot_n_positive) != int(summary['n_positive'])):
            raise ValueError('pilot_n_positive disagrees with pilot_deltas')
        if (pilot_n_negative is not None
                and int(pilot_n_negative) != int(summary['n_negative'])):
            raise ValueError('pilot_n_negative disagrees with pilot_deltas')
        if (pilot_n_zero is not None
                and int(pilot_n_zero) != int(summary['n_zero'])):
            raise ValueError('pilot_n_zero disagrees with pilot_deltas')
        if pilot_n_positive is None:
            pilot_n_positive = int(summary['n_positive'])
        if pilot_n_negative is None:
            pilot_n_negative = int(summary['n_negative'])
        if pilot_n_zero is None:
            pilot_n_zero = int(summary['n_zero'])
        if pilot_sd is None:
            pilot_sd = summary['pilot_sd']
    if pilot_sd is not None:
        pilot_sd = _seq_positive(pilot_sd, 'pilot_sd')
    if pilot_ppl is not None:
        pilot_ppl = _seq_positive(pilot_ppl, 'pilot_ppl')
        if pilot_ppl > 1e12:
            raise ValueError(f'pilot_ppl is implausibly large: {pilot_ppl!r}')
    if q_for_power is not None:
        q_for_power = _seq_probability(q_for_power, 'q_for_power')
    if q_for_power is None and rate is not None:
        q_for_power = float(rate)
    if q_for_power is None:
        q_for_power = 0.5

    if power_curve_q is None:
        power_curve_q = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
    if isinstance(power_curve_q, (str, bytes, bytearray)) or not isinstance(
            power_curve_q, (list, tuple)):
        raise TypeError('power_curve_q must be a list of probabilities')
    curve_q = []
    for index, raw in enumerate(power_curve_q):
        value = _seq_probability(raw, f'power_curve_q[{index}]')
        if value not in curve_q:
            curve_q.append(value)

    core = sequential_boundaries(
        looks, alpha=alpha, sides=sides, spending=spending,
        cumulative_spend=cumulative_spend, family_size=family_size,
        monotone_z=monotone_z)
    result = dict(core)
    power = _seq_power(core['boundaries'], q_for_power, sides)
    result.update({
        'q_for_power': float(q_for_power),
        'power': float(power['power']),
        'expected_n': float(power['expected_n']),
        'prob_stop_by_look': [float(x) for x in power['prob_stop_by_look']],
        'prob_no_stop': float(power['prob_no_stop']),
        'power_curve': {
            f'{q:.6g}': float(_seq_power(
                core['boundaries'], q, sides)['power']) for q in curve_q
        },
    })
    min_q = _seq_min_q_for_power(core['boundaries'], sides, target_power)
    result['min_q_for_target_power'] = (None if min_q is None else float(min_q))
    result['target_power'] = float(target_power)
    mde = {'q': None, 'log_ppl': None, 'relative_pct': None,
           'pilot_sd': pilot_sd, 'pilot_ppl': pilot_ppl}
    if min_q is not None:
        mde['q'] = float(min_q)
        z = _seq_normal_quantile(min_q)
        if z is not None:
            mde['normal_z'] = float(z)
            if pilot_sd is not None:
                try:
                    log_ppl = float(float(pilot_sd) * float(z))
                except (OverflowError, ValueError):
                    log_ppl = None
                if log_ppl is not None and math.isfinite(log_ppl):
                    mde['log_ppl'] = log_ppl
                    if pilot_ppl is not None:
                        try:
                            relative = float(
                                100.0 * math.expm1(float(pilot_sd) * float(z)))
                        except (OverflowError, ValueError):
                            relative = None
                        if relative is not None and math.isfinite(relative):
                            mde['relative_pct'] = relative
    result['mde'] = mde
    if q_for_power == 0.5:
        result['warnings'].append(
            'no pilot sign rate was supplied: power at q=0.5 is the null '
            'rejection probability, not a planning power')
    if summary is None:
        if rate is None:
            result['pilot'] = {'available': False, 'n_pilot': n_pilot}
        else:
            result['pilot'] = {
                'available': True,
                'n_pilot': n_pilot,
                'n_positive': (None if pilot_n_positive is None
                               else int(pilot_n_positive)),
                'n_negative': (None if pilot_n_negative is None
                               else int(pilot_n_negative)),
                'n_zero': (None if pilot_n_zero is None
                           else int(pilot_n_zero)),
                'pilot_sign_rate': float(rate),
                'pilot_sd': (None if pilot_sd is None else float(pilot_sd)),
            }
    else:
        result['pilot'] = {
            'available': True,
            'n_pilot': int(summary['n_pilot']),
            'n_positive': int(summary['n_positive']),
            'n_negative': int(summary['n_negative']),
            'n_zero': int(summary['n_zero']),
            'pilot_sign_rate': float(summary['pilot_sign_rate']),
            'pilot_sd': (None if summary['pilot_sd'] is None
                         else float(summary['pilot_sd'])),
        }

    inputs = {
        'n_max': int(n_max),
        'look_sizes': [int(n) for n in looks],
        'alpha': float(alpha),
        'sides': sides,
        'spending': spending,
        'cumulative_spend': (None if cumulative_spend is None
                             else [float(x) for x in cumulative_spend]),
        'family_size': int(family_size),
        'pilot_sign_rate': (None if rate is None else float(rate)),
        'pilot_sd': (None if pilot_sd is None else float(pilot_sd)),
        'pilot_ppl': (None if pilot_ppl is None else float(pilot_ppl)),
        'q_for_power': float(q_for_power),
        'target_power': float(target_power),
        'power_curve_q': [float(q) for q in curve_q],
        'monotone_z': bool(monotone_z),
        'n_pilot': n_pilot,
        'pilot_n_positive': (None if pilot_n_positive is None
                             else int(pilot_n_positive)),
        'pilot_n_negative': (None if pilot_n_negative is None
                             else int(pilot_n_negative)),
        'pilot_n_zero': (None if pilot_n_zero is None
                         else int(pilot_n_zero)),
    }
    artifact = {
        'kind': SEQUENTIAL_DESIGN_KIND,
        'schema': SEQUENTIAL_DESIGN_SCHEMA,
        'created_at': (created_at or time.strftime('%Y-%m-%dT%H:%M:%S')),
        'source': (None if source is None else str(source)),
        'inputs': inputs,
        'result': result,
        'revision': _seq_int(revision, 'revision', minimum=0),
        'supersedes': [str(x) for x in (supersedes or [])],
        'reason': (None if reason is None else str(reason)),
    }
    artifact['content_sha256'] = _seq_content_sha(artifact)
    return artifact


def _seq_compare(a, b, path='result', *, rtol=1e-9, atol=1e-12,
                 differences=None):
    """Tolerant structural comparison of two replayable JSON trees."""
    if differences is None:
        differences = []
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            differences.append(f'{path}: {a!r} != {b!r}')
        return differences
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if int(a) != int(b):
                differences.append(f'{path}: {a!r} != {b!r}')
            return differences
        try:
            if not math.isclose(float(a), float(b), rel_tol=rtol,
                                abs_tol=atol):
                differences.append(f'{path}: {a!r} != {b!r}')
        except (TypeError, ValueError, OverflowError):
            if a != b:
                differences.append(f'{path}: {a!r} != {b!r}')
        return differences
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            missing = sorted(set(b) - set(a))
            extra = sorted(set(a) - set(b))
            if missing:
                differences.append(f'{path}: missing keys {missing[:8]}')
            if extra:
                differences.append(f'{path}: extra keys {extra[:8]}')
        for key in sorted(set(a) & set(b), key=str):
            _seq_compare(a[key], b[key], f'{path}.{key}', rtol=rtol,
                         atol=atol, differences=differences)
        return differences
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            differences.append(f'{path}: length {len(a)} != {len(b)}')
        for index, (left, right) in enumerate(zip(a, b)):
            _seq_compare(left, right, f'{path}[{index}]', rtol=rtol,
                         atol=atol, differences=differences)
        return differences
    if a != b:
        differences.append(f'{path}: {a!r} != {b!r}')
    return differences


def validate_sequential_design(artifact, *, replay=True):
    """Validate a sequential-design artifact, replaying it from ``inputs``.

    A missing/malformed artifact is ``fail`` (the caller usually maps a
    genuinely absent file to ``not_verifiable_locally`` before calling).
    """
    out = {'status': 'fail', 'problems': [], 'warnings': [],
           'result_status': None, 'result': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact is not a JSON object')
        return out
    forbidden = _seq_forbidden_paths(artifact)
    if forbidden:
        out['problems'].append(
            'artifact carries frozen-decision field(s): '
            + ', '.join(forbidden[:8]))
        return out
    if artifact.get('kind') != SEQUENTIAL_DESIGN_KIND:
        out['problems'].append(
            f'kind must be {SEQUENTIAL_DESIGN_KIND!r}, got '
            f'{artifact.get("kind")!r}')
        return out
    if artifact.get('schema') != SEQUENTIAL_DESIGN_SCHEMA:
        out['problems'].append(
            f'schema must be {SEQUENTIAL_DESIGN_SCHEMA}, got '
            f'{artifact.get("schema")!r}')
        return out
    extra = sorted(set(artifact) - set(SEQUENTIAL_DESIGN_ARTIFACT_KEYS))
    if extra:
        out['problems'].append(
            f'artifact carries undeclared top-level key(s): {extra[:8]}')
        return out
    stored_sha = artifact.get('content_sha256')
    if not isinstance(stored_sha, str) or len(stored_sha) != 64:
        out['problems'].append('content_sha256 is missing or malformed')
        return out
    if _seq_content_sha(artifact) != stored_sha:
        out['problems'].append(
            'content_sha256 does not match the artifact body')
        return out
    inputs = artifact.get('inputs')
    result = artifact.get('result')
    if not isinstance(inputs, dict):
        out['problems'].append('inputs is not an object')
        return out
    if not isinstance(result, dict):
        out['problems'].append('result is not an object')
        return out
    missing = [key for key in SEQUENTIAL_DESIGN_INPUTS if key not in inputs]
    if missing:
        out['problems'].append(f'inputs is missing required key(s): {missing}')
        return out
    extra = [key for key in inputs if key not in SEQUENTIAL_DESIGN_INPUTS]
    if extra:
        out['problems'].append(f'inputs carries undeclared key(s): {extra}')
        return out
    if not replay:
        out['status'] = 'ok'
        out['result_status'] = result.get('status')
        out['result'] = result
        return out
    try:
        replayed = build_sequential_design(**inputs)
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {exc}')
        return out
    differences = _seq_compare(result, replayed['result'], 'result')
    if differences:
        out['problems'].append(
            'stored result differs from a full replay of the declared inputs')
        out['warnings'].extend(differences[:12])
        return out
    out['result_status'] = replayed['result'].get('status')
    out['result'] = replayed['result']
    out['warnings'].extend(list(replayed['result'].get('warnings') or []))
    out['status'] = ('ok' if replayed['result'].get('status') == 'ok'
                     else 'not_estimable')
    return out


def write_sequential_design(outdir, artifact, *, overwrite=False):
    """Write ``<outdir>/sequential_design.json`` with conflict-safe semantics."""
    verdict = validate_sequential_design(artifact)
    if verdict.get('status') not in ('ok', 'not_estimable'):
        return {'status': 'invalid', 'path': None,
                'problems': list(verdict.get('problems') or [])}
    path = os.path.join(os.fspath(outdir), 'sequential_design.json')
    os.makedirs(os.fspath(outdir), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        try:
            with open(path, encoding='utf-8') as handle:
                existing = json.load(handle)
        except Exception:
            existing = None
        if isinstance(existing, dict) and (
                existing.get('content_sha256') == artifact.get('content_sha256')):
            return {'status': 'unchanged', 'path': path}
        return {'status': 'conflict', 'path': path,
                'note': 'a different sequential_design.json already exists; '
                        'pass overwrite=True only for an explicit revision'}
    atomic_write_json(path, artifact, indent=2)
    return {'status': 'written', 'path': path}


def sequential_design_from_manifest(manifest, *, created_at=None):
    """Read an optional ``statistics.sequential`` declaration, or return None."""
    if not isinstance(manifest, dict):
        return None
    stats = manifest.get('statistics')
    stats = stats if isinstance(stats, dict) else {}
    seq = stats.get('sequential')
    if not isinstance(seq, dict):
        return None
    seeds = manifest.get('seeds')
    n_max = seq.get('n_max')
    if n_max is None and isinstance(seeds, list):
        n_max = len(seeds)
    look_sizes = seq.get('look_sizes')
    if look_sizes is None and isinstance(seeds, list) and isinstance(
            seq.get('n_looks'), int) and seq['n_looks'] >= 1:
        n_looks = int(seq['n_looks'])
        if n_looks == 1:
            look_sizes = [len(seeds)]
        else:
            step = max(1, len(seeds) // n_looks)
            look_sizes = [min(len(seeds), step * (i + 1))
                          for i in range(n_looks)]
            look_sizes[-1] = len(seeds)
            look_sizes = sorted(set(look_sizes))
    comps = manifest.get('primary_comparisons')
    family_size = seq.get('family_size')
    if family_size is None:
        family_size = len(comps) if isinstance(comps, list) and comps else 1
    return build_sequential_design(
        n_max=n_max,
        look_sizes=look_sizes,
        alpha=seq.get('alpha', stats.get('alpha', 0.05)),
        sides=seq.get('sides', 'two'),
        spending=seq.get('spending', 'obrien_fleming'),
        cumulative_spend=seq.get('cumulative_spend'),
        family_size=family_size,
        pilot_sign_rate=seq.get('pilot_sign_rate'),
        pilot_sd=seq.get('pilot_sd'),
        pilot_ppl=seq.get('pilot_ppl'),
        q_for_power=seq.get('q_for_power'),
        target_power=seq.get('target_power',
                             stats.get('min_power_for_test', 0.80)),
        power_curve_q=seq.get('power_curve_q'),
        monotone_z=seq.get('monotone_z', False),
        n_pilot=seq.get('n_pilot'),
        pilot_n_positive=seq.get('pilot_n_positive'),
        pilot_n_negative=seq.get('pilot_n_negative'),
        pilot_n_zero=seq.get('pilot_n_zero'),
        source='manifest.statistics.sequential',
        created_at=created_at)


def sequential_design_from_protocol(protocol, *, created_at=None):
    """Read an optional sequential declaration from a design protocol.

    The canonical place is ``protocol.stopping.sequential``; a legacy/draft
    protocol may put the same object in ``protocol.extras.sequential``.  No
    declaration returns ``None``; this function never invents a look schedule.
    """
    if not isinstance(protocol, dict):
        return None
    stopping = protocol.get('stopping')
    stopping = stopping if isinstance(stopping, dict) else {}
    extras = protocol.get('extras')
    extras = extras if isinstance(extras, dict) else {}
    seq = stopping.get('sequential')
    if not isinstance(seq, dict):
        seq = extras.get('sequential')
    if not isinstance(seq, dict):
        return None
    sample = protocol.get('sample_size')
    sample = sample if isinstance(sample, dict) else {}
    analysis = protocol.get('analysis')
    analysis = analysis if isinstance(analysis, dict) else {}
    question = protocol.get('research_question')
    question = question if isinstance(question, dict) else {}
    family = question.get('primary_family')
    n_max = seq.get('n_max', sample.get('planned_seeds'))
    look_sizes = seq.get('look_sizes')
    if look_sizes is None and isinstance(n_max, int) and isinstance(
            seq.get('n_looks'), int) and seq['n_looks'] >= 1:
        n_looks = int(seq['n_looks'])
        if n_looks == 1:
            look_sizes = [n_max]
        else:
            step = max(1, int(n_max) // n_looks)
            look_sizes = sorted({min(int(n_max), step * (i + 1))
                                 for i in range(n_looks)})
            look_sizes[-1] = int(n_max)
    family_size = seq.get('family_size')
    if family_size is None:
        family_size = len(family) if isinstance(family, list) and family else 1
    return build_sequential_design(
        n_max=n_max,
        look_sizes=look_sizes,
        alpha=seq.get('alpha', analysis.get('alpha', 0.05)),
        sides=seq.get('sides', 'two'),
        spending=seq.get('spending', 'obrien_fleming'),
        cumulative_spend=seq.get('cumulative_spend'),
        family_size=family_size,
        pilot_sign_rate=seq.get('pilot_sign_rate'),
        pilot_sd=seq.get('pilot_sd'),
        pilot_ppl=seq.get('pilot_ppl'),
        q_for_power=seq.get('q_for_power'),
        target_power=seq.get('target_power', sample.get('target_power', 0.80)),
        power_curve_q=seq.get('power_curve_q'),
        monotone_z=seq.get('monotone_z', False),
        n_pilot=seq.get('n_pilot'),
        pilot_n_positive=seq.get('pilot_n_positive'),
        pilot_n_negative=seq.get('pilot_n_negative'),
        pilot_n_zero=seq.get('pilot_n_zero'),
        source='protocol.stopping.sequential',
        created_at=created_at)


def _seq_load_json(path):
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return None


def _seq_timestamp(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return _seq_dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _seq_repo_root():
    """Repository root, whether this fragment is loaded via exp_lib or package."""
    here = os.path.abspath(__file__)
    if os.path.basename(here) == 'exp_lib.py':
        return os.path.dirname(here)
    parent = os.path.dirname(here)
    if os.path.basename(parent) == '_parts':
        return os.path.dirname(os.path.dirname(parent))
    return os.getcwd()


def _seq_is_study_event(event):
    """True only for an execution/look event, not an audit-report event."""
    if not isinstance(event, dict):
        return False
    kind = str(event.get('kind') or '').strip().lower()
    return kind in ('panel_look', 'look', 'cell', 'cell_start', 'cell_end',
                    'training', 'training_run', 'run', 'experiment',
                    'phase_start', 'phase_end')


def _seq_first_event_time(events_path=None):
    """First *study* ledger timestamp, or None.

    ``analysis_audit/experiments.jsonl`` also contains audit-report events;
    an audit entry is not a look at a panel and must not make a design written
    later look post hoc.
    """
    path = events_path or os.path.join(
        _seq_repo_root(), 'analysis_audit', 'experiments.jsonl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if _seq_is_study_event(event):
                    stamp = _seq_timestamp(event.get('created_at'))
                    if stamp is not None:
                        return stamp
    except OSError:
        return None
    return None


def _seq_default_look_history():
    """Preferred look_history.json location (analysis_v12, then results_*)."""
    root = _seq_repo_root()
    preferred = os.path.join(root, 'analysis_v12', 'look_history.json')
    if os.path.exists(preferred):
        return preferred
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return preferred
    for name in names:
        if not name.startswith('results_'):
            continue
        candidate = os.path.join(root, name, 'look_history.json')
        if os.path.exists(candidate):
            return candidate
    return preferred


def _seq_first_look_time(look_history_path=None):
    path = look_history_path or _seq_default_look_history()
    data = _seq_load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get('looks'), list):
        return None, None
    looks = [row for row in data['looks'] if isinstance(row, dict)]
    if not looks:
        return None, data
    stamp = _seq_timestamp(looks[0].get('created_at'))
    return stamp, data


def sequential_design_audit(path=None, *, manifest_path=None,
                            protocol_path=None, look_history_path=None,
                            events_path=None, verbose=False):
    """Design/preparation gate for a prospective group-sequential plan.

    The gate separates five states:

    * ``ok`` -- the artifact replays, the look schedule is prospective, it
      agrees with the manifest's n and the protocol's stopping declaration,
      and no recorded look falls outside the registered schedule;
    * ``warnings`` -- it replays, but provenance is incomplete or a declared
      caveat is present;
    * ``not_verifiable_locally`` -- a sequential design is declared but the
      artifact is missing/unreadable;
    * ``not_applicable`` -- no sequential design is declared (a fixed design);
    * ``fail`` -- the artifact is tampered/inconsistent, or the plan was
      written after the first look, or observed looks are off-schedule.
    """
    root = _seq_repo_root()
    out = {
        'kind': 'sequential_design_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'manifest': None,
        'protocol': None,
        'artifact_status': None,
        'result_status': None,
        'n_max': None,
        'n_looks': None,
        'look_sizes': None,
        'fwer': None,
        'power': None,
        'expected_n': None,
        'declared_by_study': False,
        'timing_ok': None,
        'problems': [],
        'warnings': [],
        'note': ('prospective group-sequential sign-test plan; it is a '
                 'design/preparation gate and never changes a frozen '
                 'p/Holm/confirmed verdict'),
    }
    manifest = None
    if manifest_path is None:
        default_manifest = os.path.join(root, 'confirmation_frozen.json')
        if os.path.exists(default_manifest):
            manifest_path = default_manifest
    if manifest_path is not None:
        manifest = _seq_load_json(manifest_path)
        out['manifest'] = manifest_path
    protocol = None
    if protocol_path is None:
        for candidate in (os.path.join(root, 'analysis_v12',
                                       'design_protocol.json'),
                          os.path.join(root, 'design_protocol.json')):
            if os.path.exists(candidate):
                protocol_path = candidate
                break
    if protocol_path is not None:
        protocol = _seq_load_json(protocol_path)
        out['protocol'] = protocol_path

    declared = False
    if isinstance(manifest, dict):
        stats = manifest.get('statistics')
        stats = stats if isinstance(stats, dict) else {}
        if isinstance(stats.get('sequential'), dict):
            declared = True
    if isinstance(protocol, dict):
        stopping = protocol.get('stopping')
        stopping = stopping if isinstance(stopping, dict) else {}
        extras = protocol.get('extras')
        extras = extras if isinstance(extras, dict) else {}
        if isinstance(stopping.get('sequential'), dict) or isinstance(
                extras.get('sequential'), dict):
            declared = True
    out['declared_by_study'] = bool(declared)

    resolved = path
    if resolved is None:
        candidates = [os.path.join(root, 'analysis_v12',
                                   'sequential_design.json'),
                      os.path.join(root, 'analysis_audit',
                                   'sequential_design.json'),
                      os.path.join(root, 'sequential_design.json')]
        try:
            for name in sorted(os.listdir(root)):
                if name.startswith('results_'):
                    candidates.append(os.path.join(
                        root, name, 'sequential_design.json'))
        except OSError:
            pass
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved = candidate
                break
    if resolved is None:
        out['status'] = ('not_verifiable_locally' if declared
                         else 'not_applicable')
        message = ('no sequential_design.json artifact'
                   + ('; the manifest/protocol declares a sequential design'
                      if declared else '; the study is a fixed single-look design'))
        if declared:
            out['problems'].append(message)
        else:
            out['note'] = message
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
        return out
    out['path'] = os.fspath(resolved)
    if not os.path.exists(resolved):
        out['status'] = 'not_verifiable_locally'
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
        return out
    artifact = _seq_load_json(resolved)
    if not isinstance(artifact, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
        return out
    verdict = validate_sequential_design(artifact)
    out['artifact_status'] = verdict.get('status')
    out['problems'].extend(list(verdict.get('problems') or []))
    out['warnings'].extend(list(verdict.get('warnings') or []))
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    out['result_status'] = result.get('status')
    out['n_max'] = result.get('n_max')
    out['n_looks'] = result.get('n_looks')
    out['look_sizes'] = result.get('look_sizes')
    out['fwer'] = result.get('fwer')
    out['power'] = result.get('power')
    out['expected_n'] = result.get('expected_n')
    if verdict.get('status') == 'fail':
        out['status'] = 'fail'
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
        return out
    if verdict.get('status') == 'not_estimable':
        out['status'] = 'not_verifiable_locally'
        out['problems'].append(
            str(result.get('reason') or 'the design is not estimable'))
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
        return out
    out['status'] = 'ok'

    # Cross-check the declared n against the manifest.
    if isinstance(manifest, dict):
        seeds = manifest.get('seeds')
        if isinstance(seeds, list):
            if out['n_max'] != len(seeds):
                out['status'] = 'fail'
                out['problems'].append(
                    f'design n_max={out["n_max"]} does not match the '
                    f'manifest seed count {len(seeds)}')
        else:
            out['warnings'].append(
                'manifest has no seed list; n_max cannot be cross-checked')
    else:
        out['warnings'].append(
            'no frozen manifest on disk; n_max cannot be cross-checked')

    # Cross-check the protocol's stopping declaration when it names this study.
    if isinstance(protocol, dict):
        stopping = protocol.get('stopping')
        stopping = stopping if isinstance(stopping, dict) else {}
        rule = str(stopping.get('rule') or '')
        if rule == 'fixed_plan_no_optional_stopping' and (
                out['n_looks'] or 1) > 1:
            out['status'] = 'fail'
            out['problems'].append(
                'the protocol declares fixed_plan_no_optional_stopping but '
                f'the plan has {out["n_looks"]} looks')
        if isinstance(stopping.get('sequential'), dict):
            declared = True

    # Timing: the plan must precede the first recorded look/event.
    artifact_time = _seq_timestamp(artifact.get('created_at'))
    first_look, history = _seq_first_look_time(look_history_path)
    first_event = _seq_first_event_time(events_path)
    _seen = [stamp for stamp in (first_look, first_event) if stamp is not None]
    first_seen = min(_seen) if _seen else None
    if artifact_time is None:
        out['warnings'].append(
            'artifact has no parseable created_at; prospective timing cannot '
            'be verified')
        out['timing_ok'] = None
    elif first_seen is None:
        out['warnings'].append(
            'no first-look/ledger timestamp on disk; prospective timing '
            'cannot be verified')
        out['timing_ok'] = None
    else:
        out['timing_ok'] = bool(artifact_time <= first_seen)
        out['first_look_at'] = first_seen.isoformat()
        if artifact_time > first_seen:
            out['status'] = 'fail'
            out['problems'].append(
                'the plan was written after the first recorded look/event; it '
                'is a reconstruction, not a pre-registration')

    # Observed looks must be a subset of the registered schedule.
    if isinstance(history, dict) and isinstance(history.get('looks'), list):
        observed = []
        for row in history['looks']:
            if not isinstance(row, dict):
                continue
            raw = row.get('n')
            if raw is None:
                raw = row.get('n_seeds')
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                    raw, numbers.Integral):
                continue
            observed.append(int(raw))
        if observed:
            planned = set(out['look_sizes'] or [])
            off = sorted(set(observed) - planned)
            if off:
                out['status'] = 'fail'
                out['problems'].append(
                    f'observed look size(s) {off} are not in the registered '
                    f'schedule {sorted(planned)}')
            out['observed_look_sizes'] = observed
    if out['status'] == 'ok' and out['warnings']:
        out['status'] = 'warnings'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])
    return out

