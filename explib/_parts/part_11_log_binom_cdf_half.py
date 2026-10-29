

def _log_binom_cdf_half(n, m):
    """log P(Bin(n, 1/2) <= m), without underflowing to -inf.

    ``_binom_cdf_half`` is float-valued: for a very small lower tail it
    returns 0.0, which loses the information a large family size needs in
    order to decide whether ``m * 2 * cdf`` clears alpha.  This helper keeps
    the log probability for the tail sizes where that distinction matters.
    """
    n = _as_seed_int(n)
    m = _as_seed_int(m)
    if n is None or m is None or n <= 0:
        return 0.0 if (m is not None and m >= 0) else float('-inf')
    if m < 0:
        return float('-inf')
    if m >= n:
        return 0.0
    if m * 2 >= n:
        log_upper = _log_binom_cdf_half(n, n - m - 1)
        if log_upper == float('-inf'):
            return 0.0
        if not math.isfinite(log_upper):
            return log_upper
        if log_upper < -745.0:
            return 0.0
        return float(math.log1p(-math.exp(log_upper)))
    log2n = _int_times_log2(n)
    if m <= 20000 or (n.bit_length() > 1023 and m <= _BINOMIAL_EXACT_INDEX_MAX):
        if not math.isfinite(log2n):
            return float('-inf')
        logs = []
        log_term = -log2n
        for j in range(m + 1):
            if j:
                log_term += math.log(n - j + 1) - math.log(j)
            logs.append(log_term)
        mx = max(logs)
        return float(mx + math.log(math.fsum(math.exp(x - mx) for x in logs)))
    try:
        cdf = _betai(n - m, m + 1, 0.5)
    except (OverflowError, ValueError):
        cdf = float('nan')
    if math.isfinite(cdf) and cdf > 0.0:
        return float(math.log(cdf))
    if not math.isfinite(log2n):
        return _log_binom_normal_cdf_half(n, m)
    # The tail is dominated by its largest term; sum the geometric correction
    # downward instead of iterating over m (which can be huge at scale).
    try:
        log_term = (math.lgamma(n + 1.0) - math.lgamma(m + 1.0)
                    - math.lgamma(n - m + 1.0) - log2n)
        if not math.isfinite(log_term):
            return _log_binom_normal_cdf_half(n, m)
        r = m / (n - m + 1.0)
    except OverflowError:
        return _log_binom_normal_cdf_half(n, m)
    if r >= 1.0:
        return float(log_term)
    return float(log_term + math.log1p(r / (1.0 - r)))


def signflip_min_consistency(n_seeds, m_comparisons, alpha=0.05):
    """规划用：通过 Holm 第一道门所需的最小同向种子数 k_min。

    若 k 个种子同向（n−k 反向），符号检验投影的 p 为
    2·P(Bin(n, 1/2) ≤ n−k)，第一道门要求 m·p < alpha。对**绝对值相等**
    的配对差，这个投影就等于精确双侧符号翻转 p；绝对值不等时，精确
    sign-flip p 只会更大（更不显著），所以本函数给出的是所需同向数的
    下界，真实数据可能仍需更多同向种子。返回使 m·p(k) < alpha 的最小
    k；若全部 n 个同向仍不够，返回 None（方案不可能确认）。二项尾概率
    在 log 空间计算，因此大 n（规模扩展后的补种子计划）也不会整数溢出。"""
    alpha = _validated_alpha(alpha, context='signflip_min_consistency alpha')
    n = _as_seed_int(n_seeds)
    m = _as_seed_int(m_comparisons)
    if n is None or m is None or n <= 0 or m <= 0:
        return None
    def passes(k):
        cdf = _binom_cdf_half(n, n - k)
        if cdf > 0.0:
            try:
                return m * 2.0 * cdf <= alpha + 1e-15
            except OverflowError:
                log_lhs = math.log(m) + math.log(2.0) + math.log(cdf)
        else:
            log_lhs = (math.log(m) + math.log(2.0)
                       + _log_binom_cdf_half(n, n - k))
        return log_lhs <= math.log(alpha) + 1e-15

    if not passes(n):
        return None
    lo, hi = -1, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def signflip_power(n_seeds, m_comparisons, q, alpha=0.05):
    """每种子同向概率为 q 时，通过 Holm 第一道门的功效。

    P(Bin(n, q) ≥ k_min)。q 的含义：在真效应方向上，单个种子的配对 Δ
    落在预注册方向的概率（含种子间噪声）。k_min 来自等幅/符号检验投影，
    因此这里的功效是对真实 sign-flip 功效的上界（绝对值不等时实际功效
    更低）。功效对 q 敏感——预注册时必须报告 q 的假设来源（如 validation
    面板的同向率），不得只报「数学上可能」。n<=0 或方案不可能确认时返回
    0.0。尾部概率直接调用稳定的上尾分发器计算 P(Bin(n,q)>=k_min)，
    既避免 `comb(n,k)*q**k` 在 n 很大时溢出，也避免先算下尾 1-q 再
    相减带来的 `1-(1-q)` 精度损失。"""
    kmin = signflip_min_consistency(n_seeds, m_comparisons, alpha)
    if kmin is None or kmin <= 0:
        return 0.0 if kmin is None else 1.0
    if isinstance(q, (bool, np.bool_)):
        return 0.0
    n = _as_seed_int(n_seeds)
    try:
        q = float(q)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if n is None or not (0.0 <= q <= 1.0) or n <= 0:
        return 0.0
    if q >= 1.0:
        return 1.0
    if q <= 0.0:
        return 0.0
    # Evaluate the upper tail directly.  The old form computed the lower
    # tail at 1-q and subtracted it; for tiny q the beta helper folds the
    # complement back in (1-(1-q)), so q loses ~eps/q relative accuracy and
    # the returned power is quantised/cancelled.  The stable upper-tail
    # dispatcher takes q directly and remains exact down to the tiny-p tail.
    tail = _binomial_sf_stable(kmin, n, q)
    if not math.isfinite(tail):
        # Fallback: log-space binomial tail (overflow-free).
        logq, log1q = math.log(q), math.log1p(-q)
        logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                + k * logq + (n - k) * log1q for k in range(kmin, n + 1)]
        mx = max(logs)
        tail = math.exp(mx) * math.fsum(math.exp(x - mx) for x in logs)
    return float(min(1.0, max(0.0, tail)))


def tost_paired(deltas, margin, conf=0.90):
    """配对 TOST 等价性检验（G0 训练级等价门槛的统计件）。

    「未发现显著差异 ≠ 等价」：等价性必须预设 margin 并检验 |mean| 落在
    (−margin, +margin) 内。这里用真正的配对 t（df=n−1，`_student_t_sf` +
    t 临界 CI），而不是把小样本 se 当作已知的正态 z——n=6 时 90% CI 的
    t 临界值比 z 宽约 20%，用 z 会把「未判等价」误判成等价。返回两侧
    单边 p、conf 水平 CI 与 `equivalent` 判定。margin 必须由计划预先注册
    （如 kernel/eager 面板平均 PPL 相对差 <1%）；本函数不替调用方选择
    margin。n<2 或 margin<=0 时不判定。"""
    d = np.asarray([f for f in (_finite_scalar(x) for x in (deltas if deltas is not None else []))
                    if f is not None], dtype=float)
    n = int(d.size)
    _bad_option = (isinstance(margin, (bool, np.bool_))
                   or isinstance(conf, (bool, np.bool_)))
    try:
        margin = float(margin)
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        margin, conf = float('nan'), float('nan')
        _bad_option = True
    out = {'n': n, 'margin': margin, 'conf': conf, 'equivalent': None,
           'method': 'paired-t (Student t, df=n-1; pre-registered margin required)'}
    if (_bad_option or n < 2 or not math.isfinite(margin) or margin <= 0
            or not math.isfinite(conf) or not (0.0 < conf < 1.0)):
        out['note'] = ('not decidable (n<2, non-positive/non-finite margin, '
                       'or conf outside (0,1))')
        return out
    # A finite but extreme panel can overflow ``mean +/- t*se`` even when the
    # interval is representable, and ``margin - mean`` can overflow before the
    # t statistic is formed.  Both are scale-invariant quantities: compute them
    # on a max-scaled copy when the panel is extreme, then map the mean and the
    # endpoints back.  Ordinary panels keep the historical path bit-for-bit.
    _abs_max = float(np.max(np.abs(d))) if n else 0.0
    _scaled = bool(math.isfinite(_abs_max) and _abs_max > 0.0
                   and (_abs_max < 1e-100 or _abs_max > 1e100))
    if _scaled:
        d_work = d / _abs_max
        margin_work = margin / _abs_max
        mean_work = float(math.fsum(float(x) for x in d_work) / n)
        if n > 1:
            sd_work = math.sqrt(math.fsum(
                (float(x) - mean_work) ** 2 for x in d_work) / (n - 1))
        else:
            sd_work = float('nan')
        mean = float(mean_work * _abs_max)
    else:
        d_work, margin_work = d, margin
        _ms = _scaled_mean_std_if_extreme(d, ddof=1)
        if _ms is not None:
            mean, sd = _ms
        else:
            mean = float(d.mean())
            sd = float(d.std(ddof=1))
        mean_work = mean
        sd_work = sd
    df = n - 1
    # A floating-point mean of k identical values need not be bit-identical to
    # the value (0.1*6 / 6 = 0.09999999999999999), which would turn a sample
    # exactly ON the margin into a tiny-se "significant" rejection.  Use the
    # identical common value when the sample is literally constant.
    identical = bool(n > 0 and np.all(d == d[0]))
    if identical:
        mean = float(d[0])
        mean_work = float(d_work[0])
    out['mean'] = mean
    out['df'] = int(df)
    se_work = 0.0 if identical else sd_work / math.sqrt(n)
    tc = _t_critical(float(conf), float(df))
    if not math.isfinite(tc):
        # _t_critical only rejects invalid conf/df; keep a deterministic
        # fallback rather than silently returning a NaN interval.
        tc = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}.get(round(float(conf), 4), 1.6448536269514722)
        out['note'] = 't critical unavailable; fell back to the normal critical value'
    out['t_critical'] = float(tc)
    ci_lo_work = mean_work - tc * se_work
    ci_hi_work = mean_work + tc * se_work
    if _scaled:
        ci_lo = float(ci_lo_work * _abs_max)
        ci_hi = float(ci_hi_work * _abs_max)
    else:
        ci_lo, ci_hi = float(ci_lo_work), float(ci_hi_work)
    if math.isfinite(ci_lo) and math.isfinite(ci_hi):
        out['ci'] = [ci_lo, ci_hi]
    else:
        # The endpoint is mathematically beyond float64.  An infinite bound is
        # not a measured interval (and strict JSON writers turn it into null),
        # so mark the interval unavailable instead of reporting [-inf, inf].
        out['ci'] = None
        out['ci_unavailable_reason'] = (
            'the confidence interval endpoint exceeds the representable '
            'float64 range')
    if not (se_work > 0.0):
        # Degenerate zero-variance sample.  At the margin boundary the exact
        # t limit is P(T >= 0) = 0.5, not 0: equality cannot be declared
        # equivalent with certainty.
        p_hi = 1.0 if mean > margin else (0.0 if mean < margin else 0.5)
        p_lo = 1.0 if mean < -margin else (0.0 if mean > -margin else 0.5)
    else:
        # H0_upper: mean >= margin  -> T = (margin - mean)/se
        # H0_lower: mean <= -margin -> T = (mean + margin)/se
        p_hi = _student_t_sf((margin_work - mean_work) / se_work, float(df))
        p_lo = _student_t_sf((mean_work + margin_work) / se_work, float(df))
    out['p_hi'] = float(p_hi)
    out['p_lo'] = float(p_lo)
    out['p_tost'] = float(max(p_hi, p_lo))
    # TOST is two one-sided tests at alpha = (1 - conf)/2.  At the default
    # conf=0.90 this is the conventional 0.05; a caller asking for 95% must
    # not get a decision inconsistent with the CI it displays.
    _alpha_one = (1.0 - float(conf)) / 2.0
    out['alpha_one_sided'] = float(_alpha_one)
    out['equivalent'] = bool(p_hi < _alpha_one and p_lo < _alpha_one)
    # Compare on the scaled grid: the interval can be unrepresentable in the
    # input units even though this containment question is still decidable.
    out['ci_within_margin'] = bool(ci_lo_work > -margin_work
                                   and ci_hi_work < margin_work)
    return out


# ---- 消融的实际显著性：预注册 SESOI + 配对 TOST ---------------------------
# 显著性检验回答「能否排除零」，等价性检验回答「能否把效应限定在一个实际
# 可忽略的区间内」。消融的「模块可删/模块无用」主张属于后者，不能由
# p>=alpha 推出。下面的函数只看已经算好的逐种子配对差，不替调用方选 margin：
# margin 必须来自预注册（面板级或 ABLATION_SESOI），单位与 Δ 相同（PPL）。

def _normalise_sesoi_design(spec):
    """Validate a {component: margin spec} mapping; fail closed on ambiguity.

    A spec is either a positive number (interpreted as relative percent of the
    without-module PPL) or a mapping with exactly one of ``relative_pct`` /
    ``absolute_ppl``.  Booleans, zero, negative, NaN/inf and a spec carrying
    both units are rejected instead of being coerced: a malformed SESOI would
    silently change which ablation claims are allowed.
    """
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(f'SESOI must be a mapping component -> margin spec, got {type(spec).__name__}')
    out = {}
    for comp, raw in spec.items():
        if not isinstance(comp, str) or not comp.strip():
            raise ValueError(f'SESOI component names must be non-empty strings, got {comp!r}')
        if isinstance(raw, dict):
            # A normalised spec carries both keys with one of them None; only
            # actual values count as declared units.
            units = [k for k in ('relative_pct', 'absolute_ppl', 'absolute')
                     if k in raw and raw[k] is not None]
            if len(units) != 1:
                raise ValueError(
                    f'SESOI[{comp!r}] must declare exactly one of relative_pct / '
                    f'absolute_ppl, got {sorted(units)}')
            unit = units[0]
            value = raw[unit]
        elif isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, (bool, np.bool_)):
            unit, value = 'relative_pct', raw
        else:
            raise ValueError(
                f'SESOI[{comp!r}] must be a positive number or a mapping, got {raw!r}')
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'SESOI[{comp!r}] must not be a boolean, got {value!r}')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'SESOI[{comp!r}] margin is not a real number: {raw!r}') from None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'SESOI[{comp!r}] margin must be finite and >0, got {value!r}')
        out[comp] = {
            'relative_pct': (value if unit == 'relative_pct' else None),
            'absolute_ppl': (value if unit != 'relative_pct' else None),
            'rationale': (str(raw.get('rationale') or '') if isinstance(raw, dict) else ''),
        }
    return out


def _sesoi_margin_ppl(spec, baseline_ppl):
    """Translate one normalised SESOI spec into an absolute PPL margin.

    Returns ``(margin, reason)``.  A relative margin needs a finite positive
    baseline PPL; without it the margin is unavailable, not silently zero.
    """
    if not isinstance(spec, dict):
        return None, 'no SESOI declared'
    rel = spec.get('relative_pct')
    absolute = spec.get('absolute_ppl')
    if rel is not None:
        base = _finite_scalar(baseline_ppl)
        if base is None or base <= 0.0:
            return None, 'relative SESOI needs a finite positive without-arm PPL'
        margin = float(base) * float(rel) / 100.0
    elif absolute is not None:
        margin = float(absolute)
    else:
        return None, 'SESOI spec carries no margin'
    if not math.isfinite(margin) or margin <= 0.0:
        return None, f'computed margin is not finite/positive ({margin!r})'
    return float(margin), ''


def _practical_decision_from_stats(stats, tost, *, margin, direction, alpha, min_n,
                                   p_holm_tost=None):
    """Combine superiority + equivalence into one practical ablation verdict.

    ``direction`` is the sign of the improvement on the Δ scale: -1 for PPL
    (with-module minus without-module, lower is better), +1 if higher is
    better, 0 when the claim has no direction.  Superiority needs a
    family-corrected significant effect above the margin; equivalence needs
    the TOST interval inside the margin AND the Holm-adjusted TOST p below
    alpha.  A significant effect smaller than the margin is explicitly
    ``sub_sesoi`` — detectable, but too small to support a practical claim.
    """
    if not isinstance(stats, dict):
        raise ValueError(
            f'stats must be a mapping, got {type(stats).__name__}')
    alpha = _validated_alpha(alpha, context='_practical_decision_from_stats alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    _direction = _as_seed_int(direction)
    if _direction is None or _direction not in (-1, 0, 1):
        raise ValueError(
            f'direction must be -1, 0 or +1, got {direction!r}')
    direction = _direction
    if isinstance(margin, (bool, np.bool_)):
        raise ValueError(f'margin must be a positive real number, got {margin!r}')
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'margin must be a positive real number, got {margin!r}') from None
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f'margin must be finite and >0, got {margin!r}')
    _tost = tost if isinstance(tost, dict) else {}
    _sig_raw = stats.get('significant')
    sig = (bool(_sig_raw) if isinstance(_sig_raw, (bool, np.bool_))
           else False)
    _tost_eq_raw = _tost.get('equivalent')
    _tost_equiv = (bool(_tost_eq_raw)
                   if isinstance(_tost_eq_raw, (bool, np.bool_)) else False)
    n = _as_seed_int(stats.get('n'))
    n = 0 if n is None or n < 0 else n
    mean = _finite_scalar(stats.get('mean'))
    out = {
        'n': n, 'mean': mean,
        'complete_pairs': stats.get('complete_pairs'),
        'n_dropped': stats.get('n_dropped'),
        'margin': float(margin),
        'margin_ppl': float(margin),
        'p_floor': stats.get('p_floor'), 'n_nonzero': stats.get('n_nonzero'),
        'p_floor_scope': 'exact sign-flip superiority test (not TOST)',
        'alpha': float(alpha), 'min_n': int(min_n), 'direction': int(direction),
        'tost_p_tost': _tost.get('p_tost'),
        'tost_p_holm': p_holm_tost,
        'tost_ci': _tost.get('ci'),
        'tost_equivalent': _tost_equiv,
        'decision': 'indeterminate', 'claim': 'inconclusive', 'reason': '',
        'statistically_significant': sig,
        'seed_driven': stats.get('seed_driven'),
        'practical_significant': False,
    }
    if n == 0 or mean is None:
        out['decision'] = 'no_data'
        out['claim'] = 'no_data'
        out['reason'] = 'no finite paired difference'
        return out
    # A complete-case subset selected by missing/failed seeds is not the
    # pre-registered paired panel.  `paired_effect_stats` already marks it
    # `complete_pairs=False`, and the reporting gates refuse it; the practical
    # decision path must do the same, otherwise dropping an inconvenient seed
    # can manufacture either a "module useful" or a "module removable" claim
    # from the survivors.  Malformed accounting fails closed.
    _complete = stats.get('complete_pairs')
    _dropped_raw = stats.get('n_dropped')
    _dropped = None if _dropped_raw is None else _as_seed_int(_dropped_raw)
    _incomplete = (_complete is False
                   or (_dropped_raw is not None
                       and (_dropped is None or _dropped > 0)))
    if _incomplete:
        out['decision'] = 'incomplete_pairs'
        out['claim'] = 'inconclusive_incomplete_panel'
        out['reason'] = (
            'the paired panel is incomplete (dropped/missing seeds); the '
            'surviving complete-case subset cannot support a superiority or '
            'equivalence claim')
        out['complete_pairs'] = False
        out['n_dropped'] = _dropped
        return out
    if n < int(min_n):
        out['decision'] = 'underpowered'
        out['claim'] = 'underpowered'
        out['reason'] = f'n={n} < min_n={int(min_n)}; neither a difference nor equivalence is decidable'
        return out
    abs_mean = abs(float(mean))
    if sig:
        if abs_mean <= float(margin):
            out['decision'] = 'sub_sesoi'
            out['claim'] = 'detectable_but_below_sesoi'
            out['reason'] = ('statistically detectable but |Δ| <= pre-registered SESOI; '
                             'not a practically meaningful module contribution')
            out['practical_significant'] = False
        elif int(direction) == 0:
            out['decision'] = 'detectable_difference'
            out['claim'] = 'detectable_difference'
            out['reason'] = 'no directional claim was declared; report the interval only'
            out['practical_significant'] = False
        elif float(mean) * float(direction) > 0.0:
            if stats.get('seed_driven') is True:
                out['decision'] = 'seed_driven_improvement'
                out['claim'] = 'inconclusive_seed_driven'
                out['reason'] = ('the significant improvement is above the SESOI but '
                                 'dropping one seed flips the mean sign; not stable enough '
                                 'for a practical claim')
            else:
                out['decision'] = 'superior'
                out['claim'] = 'module_useful'
                out['reason'] = ('family-corrected significant improvement larger than the '
                                 'pre-registered SESOI')
                out['practical_significant'] = True
        else:
            out['decision'] = 'inferior'
            out['claim'] = 'module_harmful'
            out['reason'] = ('family-corrected significant effect opposite to the declared '
                             'improvement direction and larger than the SESOI')
            out['practical_significant'] = True
        return out
    # No superiority.  Equivalence is the only route to "removable"; a
    # non-significant result alone is NOT enough.  A malformed family p is
    # fail-closed (cannot certify equivalence) instead of crashing the audit.
    _ph = None
    if p_holm_tost is not None:
        try:
            _ph = (None if isinstance(p_holm_tost, (bool, np.bool_))
                   else float(p_holm_tost))
        except (TypeError, ValueError, OverflowError):
            _ph = None
        if _ph is not None and not (math.isfinite(_ph) and 0.0 <= _ph <= 1.0):
            _ph = None
        if _ph is None:
            out['reason'] = ('the supplied family-adjusted TOST p is malformed; '
                             'equivalence cannot be certified')
            return out
    _holm_ok = (_ph is None) or (_ph < float(alpha))
    if _tost_equiv and _holm_ok:
        out['decision'] = 'equivalent'
        out['claim'] = 'module_removable_within_sesoi'
        out['reason'] = ('the TOST interval lies inside ±SESOI and the (Holm-adjusted) '
                         'equivalence p is below alpha; this is evidence of practical '
                         'removability, not a failed significance test')
        return out
    if _ph is not None and _ph >= float(alpha):
        out['reason'] = ('the raw TOST may be inside the margin, but the family-adjusted '
                         'equivalence p does not clear alpha')
    else:
        out['reason'] = ('neither a significant improvement nor an equivalence interval '
                         'inside the SESOI; the data are inconclusive')
    return out


def ablation_practical_decision(deltas, *, margin=None, sesoi=None, baseline_ppl=None,
                                alpha=0.05, min_n=3, direction=-1,
                                p_holm_tost=None, label='contrast'):
    """Adjudicate one ablation contrast as useful / removable / harmful / unclear.

    Parameters
    ----------
    deltas : per-seed paired ``with - without`` effects on the primary metric
        (for PPL, negative means the module helps).
    margin : absolute SESOI in the same units as ``deltas``.  Preferred when
        the caller already fixed the margin.
    sesoi : alternatively a normalised/raw SESOI spec used with
        ``baseline_ppl`` to derive the margin (see ``_normalise_sesoi_design``).
    direction : sign of a practically better effect (-1 for PPL, +1 for
        accuracy, 0 for a non-directional claim).
    p_holm_tost : optional family-adjusted TOST p; when supplied, equivalence
        requires it to be below ``alpha``.

    Returns a machine-readable dict.  ``decision`` is one of
    ``superior`` / ``inferior`` / ``sub_sesoi`` / ``detectable_difference`` /
    ``seed_driven_improvement`` / ``equivalent`` / ``indeterminate`` /
    ``underpowered`` / ``no_data`` / ``not_declared``.
    """
    alpha = _validated_alpha(alpha, context='ablation_practical_decision alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    # Materialise a one-pass iterable exactly once: the same sample feeds the
    # superiority test and the TOST.  Passing a generator straight through
    # would leave the second call with an exhausted iterator (n=0), silently
    # turning a real equivalence result into `not_declared`.
    if deltas is None:
        deltas = []
    elif not isinstance(deltas, (list, tuple, np.ndarray)):
        deltas = list(deltas)
    # Direction is a categorical sign, not a count: int(1.5) == 1 would
    # silently accept a malformed declaration as a one-sided claim.
    _direction = _as_seed_int(direction)
    if _direction is None or _direction not in (-1, 0, 1):
        raise ValueError(f'direction must be -1, 0 or +1, got {direction!r}')
    direction = _direction
    stats = paired_effect_stats(deltas, alpha=alpha, min_n=min_n)
    if margin is None:
        if sesoi is None or (isinstance(sesoi, dict) and not sesoi):
            return {'n': int(stats.get('n') or 0), 'mean': stats.get('mean'),
                    'margin': None, 'margin_ppl': None, 'decision': 'not_declared',
                    'claim': 'not_declared',
                    'reason': 'no pre-registered SESOI was supplied',
                    'p_floor': stats.get('p_floor'),
                    'tost_p_tost': None, 'tost_p_holm': p_holm_tost,
                    'tost_ci': None, 'tost_equivalent': None,
                    'statistically_significant': bool(stats.get('significant')),
                    'seed_driven': stats.get('seed_driven'),
                    'practical_significant': False, 'label': str(label)}
        spec = _normalise_sesoi_design({'__one__': sesoi}).get('__one__')
        margin, why = _sesoi_margin_ppl(spec, baseline_ppl)
        if margin is None:
            return {'n': int(stats.get('n') or 0), 'mean': stats.get('mean'),
                    'margin': None, 'margin_ppl': None, 'decision': 'not_declared',
                    'claim': 'not_declared', 'reason': why,
                    'p_floor': stats.get('p_floor'),
                    'tost_p_tost': None, 'tost_p_holm': p_holm_tost,
                    'tost_ci': None, 'tost_equivalent': None,
                    'statistically_significant': bool(stats.get('significant')),
                    'seed_driven': stats.get('seed_driven'),
                    'practical_significant': False, 'label': str(label)}
    if isinstance(margin, (bool, np.bool_)):
        raise ValueError(f'margin must be a positive real number, got {margin!r}')
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'margin must be a positive real number, got {margin!r}') from None
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f'margin must be finite and >0, got {margin!r}')
    # Use the same alpha for superiority and the two one-sided TOST tests.
    tost = tost_paired(deltas, margin, conf=1.0 - 2.0 * alpha)
    out = _practical_decision_from_stats(
        stats, tost, margin=margin, direction=direction, alpha=alpha,
        min_n=min_n, p_holm_tost=p_holm_tost)
    out.update({'label': str(label), 'baseline_ppl': baseline_ppl,
                'relative_margin_pct': (100.0 * margin / float(baseline_ppl)
                                        if _finite_scalar(baseline_ppl) not in (None, 0.0)
                                        else None)})
    return out


# ---- 2x2 交互的等价性（SESOI + TOST）--------------------------------------

def _normalise_interaction_sesoi_design(spec):
    """Validate a ``{factorial: margin}`` mapping for 2x2 interactions.

    A spec is either a positive number (interpreted as a relative percent of
    the ratio-of-ratios) or a mapping with exactly one of ``relative_pct`` /
    ``absolute_log_ratio``.  ``relative_pct`` is converted to
    ``log1p(pct/100)`` on the natural-log interaction scale; an absolute
    margin is already on that log scale (never PPL).  Booleans, zero,
    negative, NaN/inf, both units at once, and non-string names are rejected
    instead of being coerced: a malformed interaction SESOI would silently
    change which additivity claims are allowed.
    """
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError(
            f'interaction SESOI must be a mapping factorial -> margin spec, '
            f'got {type(spec).__name__}')
    out = {}
    for name, raw in spec.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f'interaction SESOI names must be non-empty strings, got {name!r}')
        if isinstance(raw, dict):
            units = [k for k in ('relative_pct', 'absolute_log_ratio')
                     if k in raw and raw[k] is not None]
            if len(units) != 1:
                raise ValueError(
                    f'interaction SESOI[{name!r}] must declare exactly one of '
                    f'relative_pct / absolute_log_ratio, got {sorted(units)}')
            unit = units[0]
            value = raw[unit]
        elif (isinstance(raw, (int, float, np.integer, np.floating))
              and not isinstance(raw, (bool, np.bool_))):
            unit, value = 'relative_pct', raw
        else:
            raise ValueError(
                f'interaction SESOI[{name!r}] must be a positive number or a '
                f'mapping, got {raw!r}')
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f'interaction SESOI[{name!r}] must not be a boolean, got {value!r}')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'interaction SESOI[{name!r}] margin is not a real number: '
                f'{raw!r}') from None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'interaction SESOI[{name!r}] margin must be finite and >0, '
                f'got {value!r}')
        relative = None
        absolute = None
        if unit == 'relative_pct':
            relative = value
            margin = math.log1p(value / 100.0)
        else:
            absolute = value
            margin = value
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                f'interaction SESOI[{name!r}] converts to a non-finite/non-positive '
                f'log margin ({margin!r})')
        out[name] = {
            'relative_pct': relative,
            'absolute_log_ratio': absolute,
            'margin_log_ratio': float(margin),
            'rationale': (str(raw.get('rationale') or '')
                          if isinstance(raw, dict) else ''),
        }
    return out


def _interaction_decision_from_stats(stats, tost, *, margin, direction, alpha,
                                     min_n, p_holm=None, p_holm_tost=None,
                                     label='interaction'):
    """Combine superiority + TOST into one practical interaction verdict.

    ``direction`` is the pre-registered sign of the interaction (+1 for the
    sub-additive hypothesis, -1 for synergy, 0 for a non-directional claim).
    A family-corrected significant interaction above the margin is
    ``hypothesis_supported`` or ``direction_conflict``; an interaction
    detectable but inside the margin is ``sub_sesoi``; only a TOST whose
    (family-adjusted) interval lies inside the margin is ``additive``.
    A non-significant interaction with no equivalence test stays
    ``indeterminate`` — it is not evidence of additivity.
    """
    if not isinstance(stats, dict):
        raise ValueError(
            f'interaction stats must be a mapping, got {type(stats).__name__}')
    alpha = _validated_alpha(
        alpha, context='_interaction_decision_from_stats alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'interaction min_n must be a non-negative integer, got {min_n!r}')
    min_n = int(_min_n)
    _direction = _as_seed_int(direction)
    if _direction is None or _direction not in (-1, 0, 1):
        raise ValueError(
            f'interaction direction must be -1, 0 or +1, got {direction!r}')
    direction = int(_direction)
    if isinstance(margin, (bool, np.bool_)):
        raise ValueError(
            f'interaction margin must be a positive real number, got {margin!r}')
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'interaction margin must be a positive real number, got {margin!r}') from None
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(
            f'interaction margin must be finite and >0, got {margin!r}')

    def _prob(value):
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            return None
        return v

    _tost = tost if isinstance(tost, dict) else {}
    _sig_raw = stats.get('significant')
    sig = (bool(_sig_raw) if isinstance(_sig_raw, (bool, np.bool_)) else False)
    n = _as_seed_int(stats.get('n'))
    n = 0 if n is None or n < 0 else int(n)
    mean = _finite_scalar(stats.get('mean'))
    p_sup = _prob(p_holm)
    p_tost = _prob(p_holm_tost)
    superiority_family_ok = True
    if p_holm is not None:
        if p_sup is None:
            superiority_family_ok = False
            sig = False
        else:
            sig = bool(sig and p_sup < alpha)
    _tost_eq_raw = _tost.get('equivalent')
    tost_equiv = (bool(_tost_eq_raw)
                  if isinstance(_tost_eq_raw, (bool, np.bool_)) else False)
    tost_family_ok = True
    if p_holm_tost is not None and p_tost is None:
        tost_family_ok = False
    equivalence_ok = bool(
        tost_equiv and tost_family_ok and superiority_family_ok
        and (p_tost is None or p_tost < alpha))
    margin_ratio = (float(math.exp(margin)) if margin < 700.0 else None)
    out = {
        'n': n, 'mean': mean,
        'complete_pairs': stats.get('complete_pairs'),
        'n_dropped': stats.get('n_dropped'),
        'margin': float(margin),
        'margin_log_ratio': float(margin),
        'margin_ratio': margin_ratio,
        'margin_pct_equivalent': (100.0 * (margin_ratio - 1.0)
                                  if margin_ratio is not None else None),
        'p_exact_signflip': stats.get('p_exact_signflip'),
        'p_floor': stats.get('p_floor'), 'n_nonzero': stats.get('n_nonzero'),
        'p_holm': p_sup,
        'family_adjusted_superiority': (p_holm is not None),
        'tost_p_tost': _tost.get('p_tost'),
        'tost_p_holm': p_tost,
        'family_adjusted_tost': (p_holm_tost is not None),
        'tost_ci': _tost.get('ci'),
        'tost_equivalent': tost_equiv,
        'tost_interval_inside_margin': _tost.get('ci_within_margin'),
        'alpha': float(alpha), 'min_n': int(min_n), 'direction': direction,
        'decision': 'indeterminate', 'claim': 'inconclusive', 'reason': '',
        'statistically_significant': bool(sig),
        'practical_significant': False,
        'seed_driven': stats.get('seed_driven'),
        'label': str(label),
    }
    ci = _tost.get('ci')
    if (isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(_finite_scalar(x) is not None for x in ci)
            and float(ci[1]) < 700.0):
        out['tost_ci_ratio'] = [float(math.exp(float(ci[0]))),
                                float(math.exp(float(ci[1])))]
    if n == 0 or mean is None:
        out['decision'] = 'no_data'
        out['claim'] = 'no_data'
        out['reason'] = 'no finite paired interaction difference'
        return out
    # Same complete-panel precondition as the ablation path: a TOST interval
    # computed on survivors of dropped seeds cannot certify additivity.
    _complete = stats.get('complete_pairs')
    _dropped_raw = stats.get('n_dropped')
    _dropped = None if _dropped_raw is None else _as_seed_int(_dropped_raw)
    _incomplete = (_complete is False
                   or (_dropped_raw is not None
                       and (_dropped is None or _dropped > 0)))
    if _incomplete:
        out['decision'] = 'incomplete_pairs'
        out['claim'] = 'inconclusive_incomplete_panel'
        out['reason'] = (
            'the paired interaction panel is incomplete (dropped/missing '
            'seeds); the surviving complete-case subset cannot support a '
            'directional or additivity claim')
        out['complete_pairs'] = False
        out['n_dropped'] = _dropped
        return out
    if n < min_n:
        out['decision'] = 'underpowered'
        out['claim'] = 'underpowered'
        out['reason'] = (
            f'n={n} < min_n={min_n}; neither an interaction nor additivity is '
            'decidable')
        return out
    abs_mean = abs(float(mean))
    if sig:
        if abs_mean <= float(margin):
            out['decision'] = 'sub_sesoi'
            out['claim'] = 'detectable_but_below_sesoi'
            out['reason'] = (
                'family-corrected interaction is detectable but |I| <= the '
                'pre-registered SESOI; this is not a practically relevant '
                'interaction and not proof of exact additivity')
            return out
        if direction == 0:
            out['decision'] = 'detectable_difference'
            out['claim'] = 'detectable_difference'
            out['reason'] = (
                'no directional interaction claim was declared; report the '
                'interval only')
            return out
        if float(mean) * float(direction) > 0.0:
            out['decision'] = 'hypothesis_supported'
            out['claim'] = 'interaction_above_sesoi_in_predicted_direction'
            out['reason'] = (
                'family-corrected interaction is larger than the pre-registered '
                'SESOI in the predicted direction')
        else:
            out['decision'] = 'direction_conflict'
            out['claim'] = 'interaction_above_sesoi_opposite_predicted_direction'
            out['reason'] = (
                'family-corrected interaction is larger than the pre-registered '
                'SESOI and has the opposite sign to the declared hypothesis')
        out['practical_significant'] = True
        return out
    if equivalence_ok:
        out['decision'] = 'additive'
        out['claim'] = 'additive_within_sesoi'
        out['reason'] = (
            'the TOST interval lies inside the pre-registered SESOI and the '
            '(Holm-adjusted) equivalence p is below alpha; this is evidence of '
            'practical additivity, not a failed interaction test')
        return out
    if not superiority_family_ok:
        out['reason'] = (
            'the supplied family-adjusted superiority p is malformed; a '
            'practical interaction cannot be certified')
    elif p_holm_tost is not None and p_tost is None:
        out['reason'] = (
            'the supplied family-adjusted TOST p is malformed; additivity '
            'cannot be certified')
    elif p_holm_tost is not None and p_tost >= alpha:
        out['reason'] = (
            'the raw TOST may be inside the margin, but the family-adjusted '
            'equivalence p does not clear alpha')
    else:
        out['reason'] = (
            'neither a family-corrected interaction above the SESOI nor an '
            'equivalence interval inside it; the data are inconclusive')
    return out


def interaction_practical_decision(deltas, *, margin=None, sesoi=None,
                                   alpha=0.05, min_n=3, direction=1,
                                   p_holm=None, p_holm_tost=None,
                                   label='interaction'):
    """Adjudicate a 2x2 interaction against a pre-registered SESOI.

    Parameters
    ----------
    deltas : per-seed interaction values on the natural-log ratio-of-ratios
        scale (``(log PPL11 - log PPL10) - (log PPL01 - log PPL00)``).
    margin : absolute SESOI in the same log units (preferred when the caller
        already fixed it).  A 1% ratio-of-ratios margin is ``log(1.01)``.
    sesoi : alternatively a raw/normalised interaction SESOI spec
        (``relative_pct`` or ``absolute_log_ratio``).
    direction : pre-registered sign (+1 sub-additive, -1 synergy, 0 none).
    p_holm : optional family-adjusted superiority p.  When supplied, a
        ``hypothesis_supported``/``direction_conflict`` verdict requires it
        to be below alpha; when absent the raw sign-flip verdict is used and
        the result is labelled as not family-adjusted.
    p_holm_tost : optional family-adjusted TOST p; equivalence requires it
        below alpha when supplied.

    Returns ``decision`` in ``hypothesis_supported`` / ``direction_conflict``
    / ``sub_sesoi`` / ``detectable_difference`` / ``additive`` /
    ``indeterminate`` / ``underpowered`` / ``no_data`` / ``not_declared``.
    ``additive`` is a positive equivalence claim about practical additivity
    inside the margin, not a claim that the interaction is exactly zero.
    """
    alpha = _validated_alpha(
        alpha, context='interaction_practical_decision alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(
            f'interaction min_n must be a non-negative integer, got {min_n!r}')
    min_n = int(_min_n)
    if deltas is None:
        deltas = []
    elif not isinstance(deltas, (list, tuple, np.ndarray)):
        deltas = list(deltas)
    _direction = _as_seed_int(direction)
    if _direction is None or _direction not in (-1, 0, 1):
        raise ValueError(
            f'interaction direction must be -1, 0 or +1, got {direction!r}')
    direction = int(_direction)
    stats = paired_effect_stats(deltas, alpha=alpha, min_n=min_n)
    if margin is None:
        if sesoi is None or (isinstance(sesoi, dict) and not sesoi):
            return {
                'n': int(stats.get('n') or 0), 'mean': stats.get('mean'),
                'margin': None, 'margin_log_ratio': None, 'margin_ratio': None,
                'margin_pct_equivalent': None,
                'p_exact_signflip': stats.get('p_exact_signflip'),
                'p_floor': stats.get('p_floor'),
                'p_holm': None, 'family_adjusted_superiority': False,
                'tost_p_tost': None, 'tost_p_holm': p_holm_tost,
                'family_adjusted_tost': (p_holm_tost is not None),
                'tost_ci': None, 'tost_equivalent': None,
                'statistically_significant': bool(stats.get('significant')),
                'seed_driven': stats.get('seed_driven'),
                'practical_significant': False, 'decision': 'not_declared',
                'claim': 'not_declared',
                'reason': 'no pre-registered interaction SESOI was supplied',
                'label': str(label), 'direction': direction,
                'alpha': float(alpha), 'min_n': int(min_n),
            }
        spec = _normalise_interaction_sesoi_design(
            {'__one__': sesoi}).get('__one__')
        if spec is None:
            raise ValueError('interaction SESOI spec normalised to nothing')
        margin = spec['margin_log_ratio']
    if isinstance(margin, (bool, np.bool_)):
        raise ValueError(
            f'interaction margin must be a positive real number, got {margin!r}')
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'interaction margin must be a positive real number, got {margin!r}') from None
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(
            f'interaction margin must be finite and >0, got {margin!r}')
    tost = tost_paired(deltas, margin, conf=1.0 - 2.0 * alpha)
    return _interaction_decision_from_stats(
        stats, tost, margin=margin, direction=direction, alpha=alpha,
        min_n=min_n, p_holm=p_holm, p_holm_tost=p_holm_tost, label=label)


# ---- 跨规模推断（scale ladder）：相对效应、异质性、趋势 ---------------------
# 动机：把 d=128 与 d=512 的「绝对 PPL 差」直接相减，会把基线 PPL 本身的
# 规模差异混进效应里（PPL 不是跨规模可加的）。跨规模的比较必须先转成
# 无量纲效应（同一语料/分词器内的配对 log 比），再谈趋势；跨语料/跨分词器
# 则不能用 PPL 直接比较（需要 bits-per-byte 之类的归一化，见 design_notes
# §22）。下列函数只做数学，不替调用方选择族或预注册方向。

def _gammaincc(a, x):
    """正则化上不完全伽马 Q(a,x)，用于 Cochran Q 的卡方尾概率。"""
    if not (a > 0.0) or math.isnan(x) or x < 0.0:
        return float('nan')
    if not math.isfinite(a):
        return float('nan')
    if x == float('inf'):
        return 0.0
    if not math.isfinite(x):
        return float('nan')
    if x == 0.0:
        return 1.0
    # For a large shape parameter the direct series needs O(sqrt(a)) terms
    # (billions of iterations at a~1e16), loses cancellation accuracy in the
    # tail, and its recurrence uses `a + 1.0`; once `a >= 2**53` that addition
    # rounds back to `a`, leaving the continued fraction with b == 0 and a
    # ZeroDivisionError.  Torch is already a runtime dependency and its
    # regularized incomplete gamma is stable for float64 across this scale.
    if a >= 1e7:
        try:
            _q = torch.special.gammaincc(
                torch.tensor(float(a), dtype=torch.float64),
                torch.tensor(float(x), dtype=torch.float64)).item()
        except Exception:
            _q = float('nan')
        if math.isfinite(_q):
            return float(min(1.0, max(0.0, _q)))
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = float(a)
        s = 1.0 / a
        d = s
        for _ in range(max(1000, int(50.0 * math.sqrt(a)) + 500)):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        if a >= 1000.0:
            r = (x - a) / a
            if abs(r) < 1e-3:
                log_r = (-(r * r) / 2.0
                         + (r ** 3) / 3.0
                         - (r ** 4) / 4.0
                         + (r ** 5) / 5.0
                         - (r ** 6) / 6.0)
            else:
                log_r = math.log1p(r) - r
            inv_a = 1.0 / a
            delta = inv_a * (1.0 / 12.0
                             - inv_a * inv_a * (1.0 / 360.0
                                                - inv_a * inv_a * (1.0 / 1260.0)))
            log_pref = a * log_r + 0.5 * math.log(a / (2.0 * math.pi)) - delta
        else:
            log_pref = -x + a * math.log(x) - gln
        p = s * math.exp(log_pref)
        return float(min(1.0, max(0.0, 1.0 - p)))
    tiny = 1e-300
    # Keep the subtraction from cancelling to zero at the float resolution
    # limit (x == a there); the mathematically equivalent `(x-a) + 1` is
    # exact for this branch and never feeds a zero denominator.
    b = (x - a) + 1.0
    if b <= 0.0:
        return float('nan')
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return float(min(1.0, max(0.0, math.exp(-x + a * math.log(x) - gln) * h)))


def _betacf(a, b, x):
    """不完全贝塔函数的连分式（Numerical Recipes 口径）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    max_iter = max(500, int(min(1000000.0, 10.0 * math.sqrt(max(a, b)) + 200.0)))
    for m in range(1, max_iter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai_log_bt(a, b, x):
    if min(a, b) < 8.0 or a + b < 1000000.0:
        return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    d1 = (b * x - a * (1.0 - x)) / a
    d2 = (a * (1.0 - x) - b * x) / b
    if d1 <= -1.0 or d2 <= -1.0:
        return float('-inf')
    inva, invb, invab = 1.0 / a, 1.0 / b, 1.0 / (a + b)
    return (a * math.log1p(d1) + b * math.log1p(d2)
            - 0.5 * (math.log(a + b) - math.log(a) - math.log(b))
            - 0.5 * math.log(2.0 * math.pi)
            - (inva + invb - invab) / 12.0
            + (inva ** 3 + invb ** 3 - invab ** 3) / 360.0
            - (inva ** 5 + invb ** 5 - invab ** 5) / 1260.0
            + (inva ** 7 + invb ** 7 - invab ** 7) / 1680.0)


def _betai_nr(a, b, x):
    """正则化不完全贝塔 I_x(a,b) 的 Numerical Recipes 路径。"""
    bt = math.exp(_betai_log_bt(a, b, x))
    if x < (a + 1.0) / (a + b + 2.0):
        return float(bt * _betacf(a, b, x) / a)
    return float(1.0 - bt * _betacf(b, a, 1.0 - x) / b)


_BINOMIAL_EXACT_INDEX_MAX = 2000000
_BINOMIAL_TINY_INDEX_MAX = 100
_BINOMIAL_OLD_MAX_N = 100000000.0


def _binom_log_comb(n, k):
    """Stable log C(n,k) when min(k,n-k) is small enough to sum directly.

    The lgamma difference loses all useful digits at the scale of the seed
    planners (n around 1e12-1e16, k around the mean): both lgamma terms are
    ~1e17 and their cancellation error is O(1) in the log.  Summing the log
    ratio (n-i)/(i+1) one factor at a time keeps the same absolute accuracy
    at those scales.  Callers only pass m <= _BINOMIAL_EXACT_INDEX_MAX.
    """
    n = int(n)
    k = int(k)
    m = min(k, n - k)
    if m == 0:
        return 0.0
    return math.fsum(math.log(n - i) - math.log(i + 1) for i in range(m))


def _binom_logpmf(k, n, p):
    return (_binom_log_comb(n, k) + k * math.log(p)
            + (n - k) * math.log1p(-p))


def _binom_cdf_exact_lower(k, n, p):
    """P(Bin(n,p) <= k) for k below the mean, summed away from the mode.

    Starting at k and walking down avoids the O(n) sum from zero; every step
    is a ratio of consecutive pmf values, so no large combinatorial factor is
    ever formed.
    """
    lp = _binom_logpmf(k, n, p)
    if lp < -745.0:
        return 0.0
    q = 1.0 - p
    term = math.exp(lp)
    total = term
    i = int(k)
    while i > 0:
        ratio = (i / (n - i + 1.0)) * (q / p)
        term *= ratio
        i -= 1
        total += term
        if term <= total * 1e-18 or term == 0.0:
            break
    return float(min(1.0, total))


def _binom_sf_exact_upper(k, n, p):
    """P(Bin(n,p) >= k) for k above the mean, summed away from the mode."""
    lp = _binom_logpmf(k, n, p)
    if lp < -745.0:
        return 0.0
    q = 1.0 - p
    term = math.exp(lp)
    total = term
    i = int(k)
    while i < n:
        ratio = ((n - i) / (i + 1.0)) * (p / q)
        term *= ratio
        i += 1
        total += term
        if term <= total * 1e-18 or term == 0.0:
            break
    return float(min(1.0, total))


def _binom_edgeworth_corr(z, var, p):
    g1 = (1.0 - 2.0 * p) / math.sqrt(var)
    g2 = (1.0 - 6.0 * p * (1.0 - p)) / var
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    return (g1 * (1.0 - z2) / 6.0
            + g2 * (3.0 * z - z3) / 24.0
            + g1 * g1 * (z5 - 10.0 * z3 + 15.0 * z) / 72.0)


def _binom_continuity_numerator(k, n, p, correction):
    """Accurate ``k - n*p + correction`` for huge integer counts.

    Once ``k`` exceeds 2**53, converting it to float drops low bits before
    subtracting ``n*p``; near the mean the continuity correction can then
    change sign (e.g. the central CDF at n=1e17 came out on the wrong side of
    0.5).  Split ``p`` into its exact binary rational and subtract the integer
    part exactly before the float conversion.
    """
    k = int(k)
    n = int(n)
    try:
        num, den = float(p).as_integer_ratio()
    except (OverflowError, ValueError):
        return math.fsum((float(k), -float(n) * float(p), float(correction)))
    product = n * num
    mu_int, remainder = divmod(product, den)
    return math.fsum((float(k - mu_int), -float(remainder) / float(den),
                      float(correction)))


def _binom_edgeworth_cdf(k, n, p):
    """Edgeworth CDF for a large binomial count (continuity corrected).

    At this scale the incomplete-beta continued fraction needs O(sqrt(n))
    iterations and its 1e6 cap silently truncates at n~1e14.  For npq large the
    skewness/kurtosis corrections are O(1/n) or smaller, while the exact tail
    sum is still used when min(k,n-k) is small.
    """
    mu = n * p
    var = mu * (1.0 - p)
    sd = math.sqrt(var)
    z = _binom_continuity_numerator(k, n, p, 0.5) / sd
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
    if abs(z) < 1e6:
        cdf += phi * _binom_edgeworth_corr(z, var, p)
    return float(min(1.0, max(0.0, cdf)))


def _binom_edgeworth_sf(k, n, p):
    mu = n * p
    var = mu * (1.0 - p)
    sd = math.sqrt(var)
    z = _binom_continuity_numerator(k, n, p, -0.5) / sd
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    sf = 0.5 * math.erfc(z / math.sqrt(2.0))
    if abs(z) < 1e6:
        sf -= phi * _binom_edgeworth_corr(z, var, p)
    return float(min(1.0, max(0.0, sf)))


def _binomial_cdf_stable(k, n, p):
    k = int(k)
    n = int(n)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    if p > 0.5:
        # X ~ Bin(n,p) is n - Y with Y ~ Bin(n,1-p).  Evaluate the smaller-p
        # side so the legacy beta path never has to take x close to 1.
        return _binomial_sf_stable(n - k, n, 1.0 - p)
    mu = n * p
    if k >= mu:
        return float(1.0 - _binomial_sf_stable(k + 1, n, p))
    # A tiny index is the one case the beta path cannot handle at any n: when
    # min(a,b) is tiny, the lgamma difference in log B(a,b) cancels
    # catastrophically (e.g. sf(1, 1e8, 1e-8) was off by ~1e-7).  Summing the
    # few tail terms is both exact and cheap there.
    if min(k, n - k) <= _BINOMIAL_TINY_INDEX_MAX:
        return _binom_cdf_exact_lower(k, n, p)
    if n <= _BINOMIAL_OLD_MAX_N:
        return float(_betai_nr(float(n - k), float(k + 1), 1.0 - p))
    if min(k, n - k) <= _BINOMIAL_EXACT_INDEX_MAX:
        # p <= 0.5 here.  For tiny p the complement route computes
        # P(X<=k) as 1-P(X>=k+1) and folds x through 1-x; at n=1e16,
        # p=1e-12 that loses ~1e-2 near the mean because the CDF
        # derivative is ~n*pmf.  Close to the mean the direct lower beta
        # avoids the fold; further out the exact lower sum avoids the
        # direct path's loss of precision as the lower tail approaches one.
        if p <= 1e-5:
            _var = mu * (1.0 - p)
            if _var > 0.0 and k >= mu - 0.55 * math.sqrt(_var):
                _a = float(k + 1)
                _b = float(n - k)
                try:
                    _bt = math.exp(_betai_log_bt(_a, _b, p))
                except OverflowError:
                    _bt = float('inf')
                _i = _bt * _betacf(_a, _b, p) / _a
                if math.isfinite(_i) and 0.0 <= _i <= 0.7:
                    return float(1.0 - _i)
            return _binom_cdf_exact_lower(k, n, p)
        return float(1.0 - _betai_nr(float(k + 1), float(n - k), p))
    return _binom_edgeworth_cdf(k, n, p)


def _binomial_sf_stable(k, n, p):
    k = int(k)
    n = int(n)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if p > 0.5:
        # P_p(X>=k) = P_{1-p}(n-X <= n-k).
        return _binomial_cdf_stable(n - k, n, 1.0 - p)
    mu = n * p
    if k <= mu:
        return float(1.0 - _binomial_cdf_stable(k - 1, n, p))
    if min(k, n - k) <= _BINOMIAL_TINY_INDEX_MAX:
        return _binom_sf_exact_upper(k, n, p)
    if n <= _BINOMIAL_OLD_MAX_N:
        return float(_betai_nr(float(k), float(n - k + 1), p))
    if min(k, n - k) <= _BINOMIAL_EXACT_INDEX_MAX:
        return float(_betai_nr(float(k), float(n - k + 1), p))
    return _binom_edgeworth_sf(k, n, p)


def _betai(a, b, x):
    """正则化不完全贝塔 I_x(a,b)。"""
    if (not (a > 0.0 and b > 0.0) or not math.isfinite(a)
            or not math.isfinite(b) or not math.isfinite(x)):
        return float('nan')
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if (a >= 1.0 and b >= 1.0 and a + b >= 1000000.0
            and float(a).is_integer() and float(b).is_integer()):
        n_binom = int(a) + int(b) - 1
        return _binomial_sf_stable(int(a), n_binom, float(x))
    return _betai_nr(a, b, x)


def _student_t_sf_large_df(z, df):
    """Student t 上尾的正态极限展开（z>=0，df 大）。

    `_betai` 在 df 很大时需要在 x=df/(df+t^2) 接近 1 处相减两个巨大的
    lgamma 值，尾概率会丢失约 1e-5（df=1e12）。这里把 t 密度按 1/df 展开
    并在高斯尾上积分；三阶项的绝对误差在 df>=1e4 时已低于 1e-15。
    """
    if z == 0.0:
        return 0.5
    if z > 40.0:
        return 0.0
    inv = 1.0 / df
    inv2 = inv * inv
    inv3 = inv2 * inv
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    z9 = z7 * z2
    z11 = z9 * z2
    q = 0.5 * math.erfc(z / math.sqrt(2.0))
    phi = math.exp(-0.5 * z2) / math.sqrt(2.0 * math.pi)
    i1 = phi * (z + z3) / 4.0
    i2 = (q / 32.0 + phi * (-z / 32.0 - 5.0 * z3 / 96.0
                            - 7.0 * z5 / 96.0 + z7 / 32.0))
    i3 = (-q / 384.0 + phi * (-z / 32.0 + z5 / 64.0 + 7.0 * z7 / 192.0
                              - 11.0 * z9 / 384.0 + z11 / 384.0))
    mass = 1.0 + inv2 / 32.0 - inv3 / 384.0
    return float((q + inv * i1 + inv2 * i2 + inv3 * i3) / mass)


def _student_t_sf(t, df):
    """Student t 的上尾概率 P(T >= t)。"""
    if not (df > 0.0) or math.isnan(t):
        return float('nan')
    if t == float('inf'):
        return 0.0
    if t == float('-inf'):
        return 1.0
    if not math.isfinite(t):
        return float('nan')
    # The 1/df expansion is more accurate than the incomplete-beta route for
    # large df at both moderate and extreme t.  At df=1e10, |t|=11 the latter
    # has ~1e-5 relative error, and by df=1e14 it is ~2e-2.  Keep the
    # incomplete-beta path where it is still the accurate choice: below
    # df~3e5 its far-tail error (~1e-10) beats the expansion's neglected
    # 1/df^4 term (e.g. df=1e4, |t|=10 is ~1e-4 relative), while above that
    # the expansion wins.  |t|>10 between df=1e4 and 1e7 stays on the beta
    # path unless df>=1e7, where the beta tail itself is no longer accurate.
    if df >= 1e7 or (df >= 300000.0 and abs(t) <= 10.0):
        tail = _student_t_sf_large_df(abs(t), df)
    else:
        x = df / (df + t * t)
        tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return float(tail if t >= 0 else 1.0 - tail)


def student_t_two_sided_p(t, df):
    """双侧 Student t 检验 p 值（小样本斜率/交互推断用）。"""
    sf = _student_t_sf(abs(float(t)), float(df))
    if not math.isfinite(sf):
        return float('nan')
    return float(min(1.0, 2.0 * sf))


def _t_critical(conf, df):
    """双侧置信区间的 t 临界值；用二分求解，避免依赖 scipy。

    The upper bracket grows until it actually contains the tail target.  A
    fixed 1000 cap silently returned 1000 for perfectly valid requests with
    high confidence or small/fractional df (e.g. df=0.5, conf=0.99), which
    made the reported CI too narrow and could flip a TOST equivalence or a
    coverage decision.
    """
    if not (0.0 < conf < 1.0) or not (df > 0.0):
        return float('nan')
    target = (1.0 - conf) / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(1100):
        if _student_t_sf(hi, df) <= target:
            break
        lo = hi
        hi *= 2.0
        if not math.isfinite(hi):
            return float('inf')
    else:
        # The tail never reached the target within a representable bracket.
        return float('inf')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def chi2_sf(x, df):
    """卡方分布上尾概率 P(chi2_df >= x)（Cochran Q 用）。"""
    if not (df > 0.0) or math.isnan(x):
        return float('nan')
    if x == float('inf'):
        return 0.0
    if not math.isfinite(x):
        return float('nan')
    if x <= 0.0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def paired_log_ratio(with_values, without_values, seeds=None):
    """把逐种子绝对差转成跨规模可比的 log 效应：log(with / without)。

    正值 = with 臂的 PPL 更高（更差）；零 = 无差异。log 比在乘性尺度上
    对称，且不随基线 PPL 的规模漂移——这是跨 d/token 规模比较的前提。
    只接受有限正数对；非正/非有限对整对丢弃并计数（不会静默变成 NaN）。
    """
    wa = list(with_values)
    wb = list(without_values)
    if len(wa) != len(wb):
        raise ValueError('paired_log_ratio needs equal-length vectors (%d vs %d)' % (len(wa), len(wb)))
    ss = list(range(len(wa))) if seeds is None else list(seeds)
    if len(ss) != len(wa):
        raise ValueError('paired_log_ratio seeds length %d does not match vectors length %d' % (len(ss), len(wa)))
    _norm_ss = [_as_seed_int(s) for s in ss]
    if any(s is None for s in _norm_ss):
        raise ValueError('paired_log_ratio seeds must be integers, got %r' % (ss,))
    ss = _norm_ss
    if len(set(ss)) != len(ss):
        # Two rows for one seed are not independent observations; downstream
        # means/SEs would silently double-count that seed.
        raise ValueError('paired_log_ratio got duplicate seeds %r' % (ss,))
    effects, kept, dropped = [], [], []
    for s, a, b in zip(ss, wa, wb):
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            dropped.append(s)
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError, OverflowError):
            dropped.append(s)
            continue
        if not (math.isfinite(fa) and math.isfinite(fb) and fa > 0.0 and fb > 0.0):
            dropped.append(s)
            continue
        effects.append(float(math.log(fa) - math.log(fb)))
        kept.append(s)
    return {'effects': effects, 'seeds': kept, 'n': len(effects), 'n_dropped': len(dropped), 'dropped_seeds': dropped,
            'convention': 'effect = log(with) - log(without); positive = with arm has higher PPL (worse)'}
