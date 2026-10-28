
def _dump_aggregate(outdir, summary):
    try:
        agg, _excl, records_for_readers = aggregate_with_exclusions(summary)
    except Exception as e:
        # The wrapper itself must not make a finished panel unreadable.  Keep
        # the failure visible (and counted) instead of crashing after training.
        print(f'[aggregate] aggregate({outdir}) failed: {type(e).__name__}: {e}')
        agg, records_for_readers = {}, {}
        _excl = {'kind': 'aggregation_exclusions', 'n_input': len(summary),
                 'n_admitted': 0, 'n_excluded': len(summary),
                 'reason_counts': {f'aggregate wrapper failed: {type(e).__name__}: {e}': len(summary)},
                 'excluded': [{'key': str(k), 'reason': f'aggregate wrapper failed: {type(e).__name__}: {e}'}
                              for k in summary][:200],
                 'excluded_truncated': max(0, len(summary) - 200),
                 'tolerance': 'aggregation failed catastrophically; no number was written as a silent zero'}
    atomic_write_json(os.path.join(outdir, 'aggregation_exclusions.json'), _excl)
    if _excl.get('n_excluded'):
        atomic_write_json(os.path.join(outdir, 'summary.aggregatable.json'),
                          records_for_readers)
        print(f'[aggregate] {_excl["n_excluded"]} record(s) were NOT aggregated '
              f'(of {_excl["n_input"]}); see aggregation_exclusions.json — '
              'the remaining records keep their exact values')
    _pub = {k: {kk: vv for kk, vv in e.items() if kk != '_group_records'} if isinstance(e, dict) and '_group_records' in e else e for k, e in agg.items()}
    atomic_write_json(os.path.join(outdir, 'aggregate.json'), _pub)
    _save_csv(_pub, outdir)
    _print_table(_pub)
    _print_pairs(records_for_readers)
    # Record this aggregation as a look BEFORE the audit reads the history, so
    # the audit's sequential correction sees the same seed set it is auditing.
    try:
        _looks = record_panel_look(outdir, records_for_readers)
        if int(_looks.get('n_looks') or 0) > 1:
            print(f'[look] this panel has {_looks["n_looks"]} analysis look(s) '
                  '(seed top-ups); significance is additionally checked against '
                  'the sequential sign-flip p in methodology_audit.json')
    except Exception as e:
        print(f'[look] could not record the look history ({type(e).__name__}: '
              f'{e}); the sequential correction will be unavailable')
    try:
        methodology_audit(records_for_readers, outdir=outdir, verbose=True)
    except Exception as e:
        print(f'[audit] methodology audit failed ({type(e).__name__}: {e}) — the panel is still aggregated; the audit is advisory and never blocks a run')
    _plot(records_for_readers, agg, outdir)
    return agg

def _finish_panel(outdir, summary):
    return _dump_aggregate(outdir, summary)


# ---- 统计机器的校准：先证明检验本身在零假设下不超发，再谈结果 -------------
# 本仓库的结论依赖一串统计程序：精确符号翻转、meet-in-the-middle、n>40 的
# Lugannani-Rice 鞍点尾、max-over-looks 序列 p、配对 TOST 与 Holm。它们被
# 单元测试覆盖，并不等于它们的**频率学错误率**已被校准。下面的模拟只用
# 局部 RNG、固定种子与合成数据，不触碰训练/评测 RNG、不读取任何结果目录，
# 也不改变任何测量语义；它验证的是「统计机器」，不是「实验假说」。

STATISTICAL_CALIBRATION_POLICY = {
    'version': 'statcal-1',
    'simulation_seed': 20261023,
    'n_sim': 2000,
    'alpha': 0.05,
    'power_floor': 0.60,
    'control_level': 0.001,
    'warning_level': 0.01,
    'exact_n': 12,
    'exact_power_delta': 1.0,
    'mitm_n': 21,
    'mitm_power_delta': 0.8,
    'saddlepoint_exact_n': 21,
    'saddlepoint_exact_draws': 300,
    'saddlepoint_max_abs_error': 0.03,
    'saddlepoint_mean_abs_error': 0.005,
    'saddlepoint_n': 48,
    'saddlepoint_power_delta': 0.5,
    'sequential_union_n': 12,
    'sequential_look_sizes': (6, 9, 12),
    'sequential_power_delta': 0.9,
    'tost_n': 12,
    'tost_margin': 1.0,
    'tost_null_mean': 1.0,
    'tost_alt_mean': 0.0,
    'tost_conf': 0.90,
    'holm_m': 6,
    'holm_alt_p': 1e-06,
    # Interval calibration: the reported seed-t, paired-bootstrap and
    # document/unit-bootstrap intervals are distinct estimators from the
    # hypothesis tests above, so their nominal coverage and non-vacuity are
    # checked separately.  The caps are design limits on the mean interval
    # width in the same units as the simulated data; they reject a trivially
    # wide interval that would pass a coverage-only check.
    'interval_conf': 0.95,
    'interval_n': 12,
    'interval_n_boot': 400,
    'interval_doc_units': 24,
    'interval_doc_between_sd': 1.0,
    'interval_seed_count': 6,
    'interval_seed_effect_sd': 0.2,
    'interval_doc_effect_sd': 0.5,
    'interval_within_sd': 0.1,
    'interval_doc_tokens': 8,
    'interval_combined_mu': 1.0,
    'interval_combined_n_boot': 300,
    'interval_max_width_t': 2.0,
    'interval_max_width_paired_boot': 2.0,
    'interval_max_width_doc_boot': 2.0,
    'interval_max_log_width_combined': 1.2,
    'note': ('deterministic synthetic calibration of the analysis machinery; '
             'it is not evidence about WikiText, the model, or any substantive '
             'hypothesis, and it cannot repair a confounded design'),
}


def _calibration_policy_fp(policy=None):
    payload = dict(STATISTICAL_CALIBRATION_POLICY if policy is None else policy)
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _calibration_fraction(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a real number in (0,1), got {value!r}')
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a real number in (0,1), got {value!r}') from None
    if not (0.0 < v < 1.0):
        raise ValueError(f'{name} must be in (0,1), got {value!r}')
    return v


def _calibration_counts(n_reject, n_sim):
    k = _as_seed_int(n_reject)
    n = _as_seed_int(n_sim)
    if k is None or n is None or n < 1 or k < 0 or k > n:
        raise ValueError(
            'calibration counts require integers 0 <= n_reject <= n_sim and '
            f'n_sim >= 1, got n_reject={n_reject!r}, n_sim={n_sim!r}')
    return int(k), int(n)


def _binomial_sf(k, n, p):
    """P(Bin(n,p) >= k) for an arbitrary p, using the regularized beta."""
    k, n = _calibration_counts(k, n)
    p = _calibration_fraction(p, 'binomial p')
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # Same direct-p contract as the lower-tail wrapper: the stable dispatcher
    # picks the exact small-index sum or the matching beta/Edgeworth branch
    # without ever folding in a complement that could lose precision.
    v = _binomial_sf_stable(k, n, p)
    if not math.isfinite(v):
        raise ValueError('binomial upper tail is not finite')
    return float(min(1.0, max(0.0, v)))


def _binomial_cdf(k, n, p):
    """P(Bin(n,p) <= k) for an arbitrary p, using the regularized beta."""
    k, n = _calibration_counts(k, n)
    p = _calibration_fraction(p, 'binomial p')
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # Route through the stable dispatcher instead of forming 1-p and passing
    # it to _betai: for tiny p that subtraction is exact, but _betai folds the
    # complement back in and ``1-(1-p)`` loses ~eps/p relative accuracy (a
    # 0.2% CDF error at n=1e16, p=1e-12).  The dispatcher takes p directly.
    v = _binomial_cdf_stable(k, n, p)
    if not math.isfinite(v):
        raise ValueError('binomial lower tail is not finite')
    return float(min(1.0, max(0.0, v)))


def calibration_type_i_gate(n_reject, n_sim, *, alpha=0.05,
                            control_level=0.001, warning_level=0.01):
    """One-sided calibration gate for a nominal type-I error rate.

    Under a correctly calibrated procedure the rejection count is Binomial
    (n_sim, alpha).  The gate fails only when the upper-tail probability
    ``P(X >= n_reject | alpha)`` is below ``control_level`` — i.e. when the
    observed false-positive rate is itself significantly above nominal.  A
    rate that is merely above alpha but compatible with Monte-Carlo noise is a
    warning, never a failure, and a conservative procedure (rate below alpha)
    passes.  ``control_level`` is fixed before the simulation; changing it
    after seeing the count would be the same optional-stopping problem the
    pipeline refuses elsewhere.
    """
    k, n = _calibration_counts(n_reject, n_sim)
    a = _validated_alpha(alpha, context='calibration_type_i_gate alpha')
    c = _calibration_fraction(control_level, 'control_level')
    w = _calibration_fraction(warning_level, 'warning_level')
    if w < c:
        raise ValueError('warning_level must be >= control_level')
    rate = k / n
    p_anti = _binomial_sf(k, n, a)
    if p_anti < c:
        status = 'fail'
    elif p_anti < w:
        status = 'warnings'
    else:
        status = 'ok'
    return {'n_sim': n, 'n_reject': k, 'rate': float(rate), 'alpha': a,
            'excess': float(rate - a), 'p_anti_conservative': float(p_anti),
            'control_level': c, 'warning_level': w, 'status': status}


def calibration_power_gate(n_reject, n_sim, *, power_floor=0.60,
                           control_level=0.001):
    """One-sided calibration gate for a sensitivity / positive-control rate.

    A test that never rejects would pass every type-I calibration, so a valid
    statistical assay also needs a positive control.  Under a procedure with
    power at least ``power_floor`` the count is Binomial(n_sim, floor).  The
    gate fails when the lower-tail probability is below ``control_level``
    (observed sensitivity significantly below the declared floor).  The floor
    is a design convention, not an estimate from the experiment.
    """
    k, n = _calibration_counts(n_reject, n_sim)
    f = _calibration_fraction(power_floor, 'power_floor')
    c = _calibration_fraction(control_level, 'control_level')
    rate = k / n
    p_below = _binomial_cdf(k, n, f)
    if p_below < c:
        status = 'fail'
    elif rate < f:
        status = 'warnings'
    else:
        status = 'ok'
    return {'n_sim': n, 'n_reject': k, 'rate': float(rate),
            'power_floor': f, 'p_below_floor': float(p_below),
            'control_level': c, 'status': status}


def calibration_interval_coverage_gate(n_covered, n_sim, *, nominal=0.95,
                                       control_level=0.001,
                                       warning_level=0.01):
    """One-sided calibration gate for a confidence interval's coverage.

    Under a correctly calibrated two-sided interval the number of simulations
    whose interval covers the true target is ``Binomial(n_sim, nominal)``.
    The gate fails only when the lower-tail probability
    ``P(X <= n_covered | nominal)`` is below ``control_level`` — i.e. when the
    observed coverage is significantly *below* nominal beyond Monte-Carlo
    noise.  Over-coverage is conservative and passes; a shortfall that is
    compatible with simulation noise is a warning, not a failure.  The
    thresholds are fixed before looking at the count, exactly as for the
    hypothesis-test calibration gates.
    """
    k, n = _calibration_counts(n_covered, n_sim)
    target = _calibration_fraction(nominal, 'nominal')
    c = _calibration_fraction(control_level, 'control_level')
    w = _calibration_fraction(warning_level, 'warning_level')
    if w < c:
        raise ValueError('warning_level must be >= control_level')
    rate = k / n
    p_under = _binomial_cdf(k, n, target)
    if p_under < c:
        status = 'fail'
    elif rate < target and p_under < w:
        status = 'warnings'
    else:
        status = 'ok'
    return {'n_sim': n, 'n_covered': k, 'rate': float(rate),
            'nominal': target, 'shortfall': float(rate - target),
            'p_under_coverage': float(p_under),
            'control_level': c, 'warning_level': w, 'status': status}


def calibration_interval_width_gate(mean_width, max_width):
    """Reject a vacuous interval (zero, non-finite, or wider than declared).

    Coverage alone cannot distinguish a calibrated interval from an
    arbitrarily wide one: [−inf, +inf] covers the truth with probability one
    and conveys nothing.  Each calibration path therefore declares a maximum
    acceptable mean width in the simulated data's units.  The cap is a design
    limit fixed before the simulation, not a post-hoc description of the
    observed interval.
    """
    try:
        w = float(mean_width)
        cap = float(max_width)
    except (TypeError, ValueError, OverflowError):
        return {'mean_width': None, 'max_width': None, 'status': 'fail',
                'problem': 'non-numeric interval width'}
    if isinstance(mean_width, (bool, np.bool_)) or isinstance(max_width, (bool, np.bool_)):
        return {'mean_width': None, 'max_width': None, 'status': 'fail',
                'problem': 'boolean interval width'}
    if not math.isfinite(w) or not math.isfinite(cap) or w <= 0 or cap <= 0:
        return {'mean_width': w if math.isfinite(w) else None,
                'max_width': cap if math.isfinite(cap) else None,
                'status': 'fail',
                'problem': 'interval width must be finite and > 0'}
    status = 'ok' if w <= cap else 'fail'
    return {'mean_width': w, 'max_width': cap, 'status': status,
            'problem': None if status == 'ok' else
            f'mean interval width {w:.6g} exceeds the declared cap {cap:.6g}'}


def _calibration_add_interval(out, name, coverage_control, width_control, **extra):
    rec = {'coverage_control': coverage_control,
           'width_control': width_control}
    rec.update(extra)
    out['tests'][name] = rec
    for part in ('coverage_control', 'width_control'):
        _st = rec[part].get('status')
        if _st == 'fail':
            out['status'] = 'fail'
            out['violations'].append(
                f'{name}.{part}: interval calibration failed '
                f'({rec[part].get("rate", rec[part].get("mean_width"))!r})')
        elif _st == 'warnings':
            if out['status'] == 'ok':
                out['status'] = 'warnings'
            out['warnings'].append(
                f'{name}.{part}: coverage shortfall compatible with '
                f'Monte-Carlo noise (rate={rec[part].get("rate")!r})')


def _calibration_path_gate(out, name, predicate, expected):
    rec = out['tests'][name]
    methods = rec.get('methods') or []
    ok = any(predicate(m) for m in methods)
    rec['path_ok'] = bool(ok)
    if not ok:
        out['status'] = 'fail'
        out['violations'].append(
            f'{name}: expected {expected} dispatch, observed methods={methods!r}')


def _calibration_add(out, name, null_control, positive_control, **extra):
    rec = {'null_control': null_control, 'positive_control': positive_control}
    rec.update(extra)
    out['tests'][name] = rec
    for part in ('null_control', 'positive_control'):
        _st = rec[part].get('status')
        if _st == 'fail':
            out['status'] = 'fail'
            out['violations'].append(
                f'{name}.{part}: calibration gate failed '
                f'(rate={rec[part].get("rate")!r}, '
                f'n={rec[part].get("n_reject")!r}/{rec[part].get("n_sim")!r})')
        elif _st == 'warnings':
            if out['status'] == 'ok':
                out['status'] = 'warnings'
            out['warnings'].append(
                f'{name}.{part}: rate outside the nominal target but compatible '
                f'with Monte-Carlo noise (rate={rec[part].get("rate")!r})')


def statistical_calibration_suite(*, seed=None, n_sim=None, alpha=None,
                                  power_floor=None, control_level=None,
                                  warning_level=None):
    """Deterministic synthetic calibration of every statistical procedure the
    evidence pipeline relies on.

    The suite exercises a **negative control** (data generated under the null:
    the false-positive rate must not exceed alpha beyond Monte-Carlo noise)
    and a **positive control** (data generated with a known effect: the same
    procedure must retain sensitivity above a declared floor) for the exact
    sign-flip enumeration, the 21-40 seed meet-in-the-middle path, the n>40
    saddlepoint tail (plus a direct saddlepoint-vs-exact agreement check at
    n=21), the max-over-looks sequential p, paired TOST at the margin, and the
    Holm step-down family.  A second group calibrates the interval estimators
    the reports quote (seed-level paired t, paired bootstrap-t, unit/document
    bootstrap-t and the combined seed x document interval) for both coverage
    of the true target and non-vacuity of the reported width.

    All randomness comes from local ``numpy.random.default_rng`` streams
    spawned from ``seed`` through a ``SeedSequence``; no global NumPy/Torch RNG
    is read or written, so the suite cannot perturb a training run or another
    panel's reproducibility.  The output is a pure function of the policy and
    inputs (the simulated p-values, not a clock).  A pass certifies that the
    analysis machinery has the advertised frequentist behaviour on idealized
    independent pairs; it does not certify the exchangeability, independence or
    unconfoundedness of any real panel, and it is not evidence for or against a
    substantive mechanism claim.  Passing is a screen for gross miscalibration,
    not an equivalence test proving the error rate equals alpha exactly.
    """
    policy = dict(STATISTICAL_CALIBRATION_POLICY)
    seed = policy['simulation_seed'] if seed is None else seed
    n_sim = policy['n_sim'] if n_sim is None else n_sim
    alpha = policy['alpha'] if alpha is None else alpha
    power_floor = policy['power_floor'] if power_floor is None else power_floor
    control_level = (policy['control_level'] if control_level is None
                     else control_level)
    warning_level = (policy['warning_level'] if warning_level is None
                     else warning_level)
    _seed = _as_seed_int(seed)
    if _seed is None or _seed < 0:
        raise ValueError(
            f'statistical_calibration_suite: seed must be a non-negative '
            f'integer, got {seed!r}')
    _n_sim = _as_seed_int(n_sim)
    if _n_sim is None or _n_sim < 20:
        raise ValueError(
            f'statistical_calibration_suite: n_sim must be an integer >= 20, '
            f'got {n_sim!r}')
    n_sim = int(_n_sim)
    alpha = _validated_alpha(alpha, context='statistical_calibration alpha')
    power_floor = _calibration_fraction(power_floor, 'power_floor')
    control_level = _calibration_fraction(control_level, 'control_level')
    warning_level = _calibration_fraction(warning_level, 'warning_level')
    if warning_level < control_level:
        raise ValueError('warning_level must be >= control_level')
    policy['simulation_seed'] = int(_seed)
    policy['n_sim'] = n_sim
    policy['alpha'] = alpha
    policy['power_floor'] = power_floor
    policy['control_level'] = control_level
    policy['warning_level'] = warning_level
    children = np.random.SeedSequence(int(_seed)).spawn(11)
    out = {'kind': 'statistical_calibration', 'status': 'ok',
           'seed': int(_seed), 'n_sim': n_sim, 'alpha': alpha,
           'power_floor': power_floor, 'control_level': control_level,
           'warning_level': warning_level, 'tests': {},
           'violations': [], 'warnings': [],
           'policy': policy, 'policy_fp': _calibration_policy_fp(policy),
           'deterministic': True,
           'note': policy['note'],
           'interpretation_boundary': (
               'a pass validates the statistical machinery on synthetic '
               'independent pairs; it does not validate the exchangeability, '
               'randomisation or measurement identity of a real panel, and it '
               'is not evidence for a substantive hypothesis')}

    rng = np.random.default_rng(children[0])
    n_exact = int(policy['exact_n'])
    delta_exact = float(policy['exact_power_delta'])
    k_null = k_alt = 0
    methods = set()
    for _ in range(n_sim):
        d = rng.normal(size=n_exact)
        r = exact_signflip_p(d.tolist())
        methods.add(r.get('method'))
        k_null += bool(r['p'] <= alpha)
    for _ in range(n_sim):
        d = rng.normal(loc=delta_exact, scale=1.0, size=n_exact)
        k_alt += bool(exact_signflip_p(d.tolist())['p'] <= alpha)
    _calibration_add(
        out, 'exact_signflip_enumeration',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        path='exact enumeration', n_seeds=n_exact,
        alternative_mean=delta_exact,
        methods=sorted(m for m in methods if m))
    _calibration_path_gate(out, 'exact_signflip_enumeration',
                           lambda m: m == 'exact', 'exact enumeration')

    rng = np.random.default_rng(children[1])
    n_mitm = int(policy['mitm_n'])
    delta_mitm = float(policy['mitm_power_delta'])
    k_null = k_alt = 0
    methods = set()
    for _ in range(n_sim):
        d = rng.normal(size=n_mitm)
        r = exact_signflip_p(d.tolist())
        methods.add(r.get('method'))
        k_null += bool(r['p'] <= alpha)
    for _ in range(n_sim):
        d = rng.normal(loc=delta_mitm, scale=1.0, size=n_mitm)
        k_alt += bool(exact_signflip_p(d.tolist())['p'] <= alpha)
    _calibration_add(
        out, 'exact_signflip_mitm',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        path='meet-in-the-middle', n_seeds=n_mitm,
        alternative_mean=delta_mitm, methods=sorted(m for m in methods if m))
    _calibration_path_gate(out, 'exact_signflip_mitm',
                           lambda m: m == 'exact-mitm', 'MITM')

    rng = np.random.default_rng(children[2])
    n_sp_exact = int(policy['saddlepoint_exact_n'])
    n_draws = min(int(policy['saddlepoint_exact_draws']), n_sim)
    errors = []
    n_disagree = 0
    for _ in range(n_draws):
        d = rng.normal(size=n_sp_exact)
        exact = exact_signflip_p(d.tolist())
        approx, _method = _signflip_saddlepoint_p(d.tolist())
        errors.append(abs(float(approx) - float(exact['p'])))
        n_disagree += bool((approx <= alpha) != (exact['p'] <= alpha))
    max_err = max(errors) if errors else float('nan')
    mean_err = float(np.mean(errors)) if errors else float('nan')
    tol_max = float(policy['saddlepoint_max_abs_error'])
    tol_mean = float(policy['saddlepoint_mean_abs_error'])
    acc_status = ('fail' if (not math.isfinite(max_err)
                             or max_err > tol_max
                             or not math.isfinite(mean_err)
                             or mean_err > tol_mean) else 'ok')
    out['tests']['saddlepoint_vs_exact'] = {
        'n_seeds': n_sp_exact, 'n_draws': n_draws,
        'max_abs_error': float(max_err), 'mean_abs_error': float(mean_err),
        'max_abs_tolerance': tol_max, 'mean_abs_tolerance': tol_mean,
        'rejection_disagreements': int(n_disagree), 'status': acc_status}
    if acc_status == 'fail':
        out['status'] = 'fail'
        out['violations'].append(
            'saddlepoint_vs_exact: approximation error '
            f'(max={max_err!r} > {tol_max!r} or mean={mean_err!r} > '
            f'{tol_mean!r})')

    rng = np.random.default_rng(children[3])
    n_sp = int(policy['saddlepoint_n'])
    delta_sp = float(policy['saddlepoint_power_delta'])
    k_null = k_alt = 0
    sp_methods = set()
    for _ in range(n_sim):
        d = rng.normal(size=n_sp)
        r = exact_signflip_p(d.tolist())
        sp_methods.add(r.get('method'))
        k_null += bool(r['p'] <= alpha)
    for _ in range(n_sim):
        d = rng.normal(loc=delta_sp, scale=1.0, size=n_sp)
        k_alt += bool(exact_signflip_p(d.tolist())['p'] <= alpha)
    _calibration_add(
        out, 'saddlepoint_tail',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        path='saddlepoint (n>40 dispatch)', n_seeds=n_sp,
        alternative_mean=delta_sp,
        methods=sorted(m for m in sp_methods if m))
    _calibration_path_gate(out, 'saddlepoint_tail',
                           lambda m: m == 'saddlepoint', 'saddlepoint')

    rng = np.random.default_rng(children[4])
    n_union = int(policy['sequential_union_n'])
    look_sizes = [int(x) for x in policy['sequential_look_sizes']]
    if (not look_sizes or look_sizes[-1] != n_union
            or any(x <= 0 or x > n_union for x in look_sizes)
            or any(b <= a for a, b in zip(look_sizes, look_sizes[1:]))):
        raise ValueError('sequential look sizes must be strictly increasing '
                         'and end at sequential_union_n')
    look_defs = [list(range(size)) for size in look_sizes]
    delta_seq = float(policy['sequential_power_delta'])
    k_null = k_alt = 0
    seq_methods = set()
    for _ in range(n_sim):
        d = rng.normal(size=n_union)
        looks = [(s, [float(d[i]) for i in s]) for s in look_defs]
        r = sequential_signflip_p(looks)
        seq_methods.add(r.get('method'))
        k_null += bool(r['p_sequential'] <= alpha)
    for _ in range(n_sim):
        d = rng.normal(loc=delta_seq, scale=1.0, size=n_union)
        looks = [(s, [float(d[i]) for i in s]) for s in look_defs]
        r = sequential_signflip_p(looks)
        k_alt += bool(r['p_sequential'] <= alpha)
    _calibration_add(
        out, 'sequential_looks',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        path='max-over-looks sign-flip', union_seeds=n_union,
        look_sizes=look_sizes, alternative_mean=delta_seq,
        methods=sorted(m for m in seq_methods if m))
    _calibration_path_gate(out, 'sequential_looks',
                           lambda m: 'max-over-looks' in str(m),
                           'max-over-looks sequential')

    rng = np.random.default_rng(children[5])
    n_tost = int(policy['tost_n'])
    margin = float(policy['tost_margin'])
    conf = float(policy['tost_conf'])
    null_mean = float(policy['tost_null_mean'])
    alt_mean = float(policy['tost_alt_mean'])
    k_null = k_alt = 0
    for _ in range(n_sim):
        d = rng.normal(loc=null_mean, scale=1.0, size=n_tost)
        k_null += bool(tost_paired(d.tolist(), margin, conf=conf)['equivalent'])
    for _ in range(n_sim):
        d = rng.normal(loc=alt_mean, scale=1.0, size=n_tost)
        k_alt += bool(tost_paired(d.tolist(), margin, conf=conf)['equivalent'])
    _calibration_add(
        out, 'tost_paired',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        n_seeds=n_tost, margin=margin, conf=conf,
        null_mean=null_mean, alternative_mean=alt_mean)

    rng = np.random.default_rng(children[6])
    m_holm = int(policy['holm_m'])
    alt_p = float(policy['holm_alt_p'])
    k_null = k_alt = 0
    for _ in range(n_sim):
        ps = rng.random(m_holm)
        h = holm_adjust([(f'c{i}', float(p)) for i, p in enumerate(ps)], alpha=alpha)
        k_null += bool(any(v.get('reject') for v in h.values()))
    for _ in range(n_sim):
        ps = np.concatenate(([alt_p], rng.random(m_holm - 1)))
        h = holm_adjust([(f'c{i}', float(p)) for i, p in enumerate(ps)], alpha=alpha)
        k_alt += bool(any(v.get('reject') for v in h.values()))
    _calibration_add(
        out, 'holm_family',
        calibration_type_i_gate(k_null, n_sim, alpha=alpha,
                                control_level=control_level,
                                warning_level=warning_level),
        calibration_power_gate(k_alt, n_sim, power_floor=power_floor,
                               control_level=control_level),
        family_m=m_holm, alternative_p=alt_p)

    # ---- interval calibration: coverage AND non-vacuity -------------------
    # A test that is correctly calibrated can still quote an interval that
    # under-covers or is so wide that it says nothing.  These paths exercise
    # the exact interval constructors the reports use.  The target is the
    # population mean (0 for the centred seed contrasts, 1.0 on the log-PPL
    # scale for the combined interval); the width caps are declared in the
    # policy before the draws.
    conf = float(policy['interval_conf'])
    n_int = int(policy['interval_n'])
    nb_int = int(policy['interval_n_boot'])

    # (1) seed-level paired t interval from paired_effect_stats.ci95_*
    rng = np.random.default_rng(children[7])
    k_cov = 0
    widths = []
    for _ in range(n_sim):
        d = rng.normal(size=n_int)
        st = paired_effect_stats(d.tolist(), min_n=3)
        lo, hi = st.get('ci95_low'), st.get('ci95_high')
        if (lo is not None and hi is not None
                and math.isfinite(float(lo)) and math.isfinite(float(hi))):
            k_cov += bool(float(lo) <= 0.0 <= float(hi))
            widths.append(float(hi) - float(lo))
    _calibration_add_interval(
        out, 'interval_seed_t',
        calibration_interval_coverage_gate(
            k_cov, n_sim, nominal=conf, control_level=control_level,
            warning_level=warning_level),
        calibration_interval_width_gate(
            float(np.mean(widths)) if widths else float('nan'),
            policy['interval_max_width_t']),
        path='paired t interval (paired_effect_stats.ci95)',
        n_units=n_int, target_mean=0.0, conf=conf)

    # (2) paired bootstrap-t interval used by paired_effect_stats
    rng = np.random.default_rng(children[8])
    k_cov = 0
    widths = []
    for _ in range(n_sim):
        d = rng.normal(size=n_int)
        bseed = int(rng.integers(0, 2**31 - 1))
        ci = paired_bootstrap_ci(d.tolist(), n_boot=nb_int, seed=bseed)
        if ci is not None and all(math.isfinite(float(x)) for x in ci):
            k_cov += bool(float(ci[0]) <= 0.0 <= float(ci[1]))
            widths.append(float(ci[1]) - float(ci[0]))
    _calibration_add_interval(
        out, 'interval_paired_bootstrap',
        calibration_interval_coverage_gate(
            k_cov, n_sim, nominal=conf, control_level=control_level,
            warning_level=warning_level),
        calibration_interval_width_gate(
            float(np.mean(widths)) if widths else float('nan'),
            policy['interval_max_width_paired_boot']),
        path='paired bootstrap-t (chunked, deterministic)',
        n_units=n_int, target_mean=0.0, conf=conf, n_boot=nb_int)

    # (3) unit/document bootstrap-t interval
    rng = np.random.default_rng(children[9])
    n_docs = int(policy['interval_doc_units'])
    between = float(policy['interval_doc_between_sd'])
    k_cov = 0
    widths = []
    for _ in range(n_sim):
        vals = rng.normal(0.0, between, size=n_docs)
        bseed = int(rng.integers(0, 2**31 - 1))
        ci = bootstrap_mean_ci(vals.tolist(), n_boot=nb_int,
                               seed=bseed, conf=conf)
        if ci is not None and all(math.isfinite(float(ci[k])) for k in ('ci_low', 'ci_high')):
            k_cov += bool(float(ci['ci_low']) <= 0.0 <= float(ci['ci_high']))
            widths.append(float(ci['ci_high']) - float(ci['ci_low']))
    _calibration_add_interval(
        out, 'interval_unit_bootstrap',
        calibration_interval_coverage_gate(
            k_cov, n_sim, nominal=conf, control_level=control_level,
            warning_level=warning_level),
        calibration_interval_width_gate(
            float(np.mean(widths)) if widths else float('nan'),
            policy['interval_max_width_doc_boot']),
        path='studentized unit/document bootstrap (chunked, deterministic)',
        n_units=n_docs, target_mean=0.0, conf=conf, n_boot=nb_int)

    # (4) combined seed x document interval used for the pooled PPL.  The
    #     synthetic panel has known seed, document and token noise; the
    #     population target is exp(mu), and the interval is checked on the
    #     log-PPL scale so the multiplicative geometry does not hide a
    #     vacuous interval.
    rng = np.random.default_rng(children[10])
    n_seeds_int = int(policy['interval_seed_count'])
    n_tok_int = int(policy['interval_doc_tokens'])
    seed_sd = float(policy['interval_seed_effect_sd'])
    doc_sd = float(policy['interval_doc_effect_sd'])
    within_sd = float(policy['interval_within_sd'])
    mu_int = float(policy['interval_combined_mu'])
    nb_comb = int(policy['interval_combined_n_boot'])
    target_ppl = math.exp(mu_int)
    k_cov = 0
    log_widths = []
    for _ in range(n_sim):
        se_eff = rng.normal(0.0, seed_sd, size=n_seeds_int)
        de_eff = rng.normal(0.0, doc_sd, size=n_docs)
        noise = rng.normal(0.0, within_sd, size=(n_seeds_int, n_docs))
        ce = mu_int + se_eff[:, None] + de_eff[None, :] + noise
        nll = (ce * n_tok_int).tolist()
        ntok = [n_tok_int] * n_docs
        recs = [{'nll_per_row': nll[s], 'n_tokens_per_row': ntok}
                for s in range(n_seeds_int)]
        bseed = int(rng.integers(0, 2**31 - 1))
        ci = _combined_seed_doc_ci(recs, n_boot=nb_comb, seed=bseed, conf=conf)
        if ci.get('status') == 'ok' and ci.get('ci95_ppl'):
            lo, hi = ci['ci95_ppl']
            try:
                k_cov += bool(float(lo) <= target_ppl <= float(hi))
                lw = math.log(float(hi)) - math.log(float(lo))
                if math.isfinite(lw):
                    log_widths.append(lw)
            except (TypeError, ValueError, OverflowError):
                pass
    _calibration_add_interval(
        out, 'interval_combined_seed_doc',
        calibration_interval_coverage_gate(
            k_cov, n_sim, nominal=conf, control_level=control_level,
            warning_level=warning_level),
        calibration_interval_width_gate(
            float(np.mean(log_widths)) if log_widths else float('nan'),
            policy['interval_max_log_width_combined']),
        path='combined seed x document interval (log-PPL scale)',
        n_seeds=n_seeds_int, n_units=n_docs, target_ppl=target_ppl,
        conf=conf, n_boot=nb_comb)

    return out
