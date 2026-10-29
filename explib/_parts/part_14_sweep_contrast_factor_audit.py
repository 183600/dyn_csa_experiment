
def sweep_contrast_factor_audit(declarations, *, factory=None, n_layers=6,
                                verbose=True):
    """Verify declared sweep/sensitivity contrasts against the real arm configs.

    ``ABLATION_DESIGN`` has a definition-site gate, but the v9/v10 seq2k sweep
    declarations (``v9.SEQ2K_CONTRAST_FACTORS``) were prose metadata attached
    to a paired record: nothing checked that the arm named as a "single-factor
    topk dose-response" actually differed in ``index_topk`` alone, or that a
    contrast labeled multi-factor listed all the keys it moves.  A metadata
    label is not evidence.

    Each declaration must name its two arms (``a``/``b``), the exact AttnCfg
    keys it expects to move (``factors``) and whether it is a single-factor
    contrast.  The audit builds both arms through the module factory, diffs
    the configs and refuses:

      * a declared/actual key mismatch in either direction (an extra moved
        key is an undeclared second factor; a missing key means the declared
        intervention disappeared);
      * ``single_factor=True`` on a diff that is not inside ONE declared
        family (including any ``kind`` swap or multi-family bundle);
      * a malformed declaration (missing arms/factors/flag), because an
        unverifiable label cannot be used in a report.

    A declaration may instead name a ``protocol_factor`` plus ``a_level`` /
    ``b_level`` for a factor outside AttnCfg (the v7 warmup cut point).  Such
    a declaration is checked structurally here; the record-level gate
    ``protocol_contrast_reason`` then verifies that only that protocol axis
    differs before a pair is quoted.

    Returns ``{status, pairs, violations}`` where every pair records the
    machine-derived diff, not just the declared label.
    """
    factory = make_layer_cfgs if factory is None else factory
    report = {'pairs': {}, 'violations': [], 'status': 'ok'}
    if not isinstance(declarations, dict) or not declarations:
        report['status'] = 'fail'
        report['violations'].append(
            'sweep factor declarations must be a non-empty name -> declaration object')
        if verbose:
            print('[sweep-factor] REFUSED: no declarations to verify')
        return report
    for name, spec in declarations.items():
        label = str(name)
        if not isinstance(spec, dict):
            report['violations'].append(f'{label}: declaration is not an object')
            continue
        factors = spec.get('factors')
        if (not isinstance(factors, (list, tuple, set, frozenset))
                or not factors
                or any(not isinstance(k, str) or not k.strip() for k in factors)):
            report['violations'].append(
                f'{label}: factors must be a non-empty list of AttnCfg key names')
            continue
        declared_sf = spec.get('single_factor')
        if not isinstance(declared_sf, bool):
            report['violations'].append(
                f'{label}: single_factor must be an explicit boolean')
            continue
        declared_set = {str(k) for k in factors}
        a, b = spec.get('a'), spec.get('b')
        protocol_factor = spec.get('protocol_factor')
        if protocol_factor is not None:
            # A warmup panel's factor is the protocol cut point, which is not
            # an AttnCfg field.  It is still a single-factor contrast, so it
            # gets a declaration rather than being silently unlabelled: the
            # levels must be explicit, distinct integers and the record-level
            # gate (`protocol_contrast_reason`) later refuses any second
            # protocol/run difference.
            a_level = _as_seed_int(spec.get('a_level'))
            b_level = _as_seed_int(spec.get('b_level'))
            ok = True
            if a is not None or b is not None:
                ok = False
                report['violations'].append(
                    f'{label}: a protocol-factor declaration must not also name '
                    'AttnCfg arms (a/b)')
            if protocol_factor != 'warm_steps':
                ok = False
                report['violations'].append(
                    f'{label}: unsupported protocol factor {protocol_factor!r} '
                    '(the only supported axis is warm_steps)')
            if declared_set != {str(protocol_factor)}:
                ok = False
                report['violations'].append(
                    f'{label}: protocol factors {sorted(declared_set)} must be '
                    f'exactly {{protocol_factor!r}}'.format(protocol_factor))
            if declared_sf is not True:
                ok = False
                report['violations'].append(
                    f'{label}: a declared protocol contrast must be '
                    'single_factor=True')
            if a_level is None or b_level is None or a_level == b_level:
                ok = False
                report['violations'].append(
                    f'{label}: a_level/b_level must be distinct integer warmup '
                    'cut points')
            entry = {'mode': 'protocol', 'protocol_factor': protocol_factor,
                     'a_level': a_level, 'b_level': b_level,
                     'declared_factors': sorted(declared_set),
                     'actual_keys': sorted(declared_set),
                     'extra_keys': [], 'missing_keys': [],
                     'declared_single_factor': declared_sf,
                     'actual_single_factor': True,
                     'verified': bool(ok)}
            report['pairs'][label] = entry
            continue
        if not (isinstance(a, str) and a.strip() and isinstance(b, str) and b.strip()):
            report['violations'].append(
                f'{label}: declaration must name both arms (a/b)')
            continue
        actual, per_layer = _design_pair_cfg_keys(a, b, factory, n_layers)
        if isinstance(actual, Exception):
            report['violations'].append(
                f'{label}: arm construction failed ({type(actual).__name__}: {actual})')
            continue
        actual_set = set(actual)
        extra = sorted(actual_set - declared_set)
        missing = sorted(declared_set - actual_set)
        actual_sf = bool(cfg_keys_single_factor(actual_set))
        entry = {'a': a, 'b': b, 'declared_factors': sorted(declared_set),
                 'actual_keys': sorted(actual_set), 'extra_keys': extra,
                 'missing_keys': missing,
                 'declared_single_factor': declared_sf,
                 'actual_single_factor': actual_sf,
                 'per_layer': {str(i): {k: [v0, v1] for k, (v0, v1) in d.items()}
                               for i, d in per_layer.items()}}
        ok = True
        if extra:
            ok = False
            report['violations'].append(
                f'{label}: arms also differ in undeclared key(s) {extra} — the '
                f'declared factors {sorted(declared_set)} do not describe the contrast')
        if missing:
            ok = False
            report['violations'].append(
                f'{label}: declared factor key(s) {missing} do not differ — the '
                f'declared intervention is absent from the arm pair')
        if declared_sf:
            if actual_set != declared_set:
                ok = False
            if not actual_sf:
                ok = False
                report['violations'].append(
                    f'{label}: declared single-factor but the actual diff '
                    f'{sorted(actual_set)} is not one declared intervention family '
                    f'(kind swaps and multi-family bundles are not single factors)')
        entry['verified'] = bool(ok)
        report['pairs'][label] = entry
    report['status'] = 'fail' if report['violations'] else 'ok'
    if verbose:
        if report['violations']:
            print('[sweep-factor] REFUSED — declared contrasts do not match the arms:')
            for _v in report['violations']:
                print(f'  - {_v}')
        else:
            print(f"[sweep-factor] OK: {len(report['pairs'])} declared sweep "
                  f"contrast(s) verified against the actual arm configs")
    return report

def niah_contrast_factor_audit(declarations=None, *, overrides=None,
                               factory=None, n_layers=6, verbose=True):
    declarations = (NIAH_CONTRAST_FACTORS if declarations is None
                    else declarations)
    overrides = (NIAH_INFERENCE_ARM_OVERRIDES if overrides is None
                 else overrides)
    report = {'pairs': {}, 'violations': [], 'status': 'ok'}
    if not isinstance(declarations, dict) or not declarations:
        report['status'] = 'fail'
        report['violations'].append(
            'NIAH factor declarations must be a non-empty name -> declaration object')
        return report
    if not isinstance(overrides, dict):
        report['status'] = 'fail'
        report['violations'].append('NIAH inference overrides must be an object')
        return report
    missing_factory = False
    for name, spec in declarations.items():
        label = str(name)
        if not isinstance(spec, dict):
            report['violations'].append(f'{label}: declaration is not an object')
            continue
        factors = spec.get('factors')
        if (not isinstance(factors, (list, tuple, set, frozenset))
                or not factors
                or any(not isinstance(k, str) or not k.strip() for k in factors)):
            report['violations'].append(
                f'{label}: factors must be a non-empty list of AttnCfg key names')
            continue
        declared_sf = spec.get('single_factor')
        if not isinstance(declared_sf, bool):
            report['violations'].append(
                f'{label}: single_factor must be an explicit boolean')
            continue
        declared_set = {str(k) for k in factors}
        override_tag = spec.get('override')
        a, b = spec.get('a'), spec.get('b')
        if override_tag is not None and (a is not None or b is not None):
            report['violations'].append(
                f'{label}: declaration cannot be both an inference override and an arm pair')
            continue
        if override_tag is not None:
            if not isinstance(override_tag, str) or override_tag not in overrides:
                report['violations'].append(
                    f'{label}: override {override_tag!r} is not in the shared NIAH override map')
                continue
            _override_spec = overrides.get(override_tag)
            if not isinstance(_override_spec, dict):
                report['violations'].append(
                    f'{label}: override {override_tag!r} must be an object')
                continue
            actual_set = {str(k) for k in _override_spec}
            entry = {'mode': 'inference_override', 'override': override_tag,
                     'a': 'csa_fixed_rope', 'b': 'csa_fixed_rope'}
        else:
            if not (isinstance(a, str) and a.strip()
                    and isinstance(b, str) and b.strip()):
                report['violations'].append(
                    f'{label}: declaration must name an override tag or both arms (a/b)')
                continue
            if factory is None:
                missing_factory = True
                continue
            actual, per_layer = _design_pair_cfg_keys(a, b, factory, n_layers)
            if isinstance(actual, Exception):
                report['violations'].append(
                    f'{label}: arm construction failed '
                    f'({type(actual).__name__}: {actual})')
                continue
            actual_set = set(actual)
            entry = {'mode': 'trained_pair', 'a': a, 'b': b,
                     'per_layer': {str(i): {k: [v0, v1] for k, (v0, v1) in d.items()}
                                   for i, d in per_layer.items()}}
        extra = sorted(actual_set - declared_set)
        missing = sorted(declared_set - actual_set)
        actual_sf = bool(cfg_keys_single_factor(actual_set))
        entry.update({'declared_factors': sorted(declared_set),
                      'actual_keys': sorted(actual_set),
                      'extra_keys': extra, 'missing_keys': missing,
                      'declared_single_factor': declared_sf,
                      'actual_single_factor': actual_sf})
        ok = True
        if extra:
            ok = False
            report['violations'].append(
                f'{label}: arms/overrides also differ in undeclared key(s) '
                f'{extra}; the declared factors {sorted(declared_set)} do not '
                'describe the contrast')
        if missing:
            ok = False
            report['violations'].append(
                f'{label}: declared factor key(s) {missing} do not differ; the '
                'declared intervention is absent')
        if declared_sf:
            if actual_set != declared_set:
                ok = False
            if not actual_sf:
                ok = False
                report['violations'].append(
                    f'{label}: declared single-factor but the actual diff '
                    f'{sorted(actual_set)} is not one declared intervention family')
        entry['verified'] = bool(ok)
        report['pairs'][label] = entry
    if report['violations']:
        report['status'] = 'fail'
    elif missing_factory:
        report['status'] = 'not_verifiable'
        report['note'] = ('the trained-arm factory is unavailable; inference '
                          'overrides were checked but trained pairs were not')
    if verbose:
        if report['violations']:
            print('[niah-factor] REFUSED — declarations do not match the arms:')
            for _v in report['violations']:
                print(f'  - {_v}')
        elif report['status'] == 'ok':
            print(f"[niah-factor] OK: {len(report['pairs'])} NIAH contrast(s) "
                  'verified against the shared overrides/arm configs')
    return report

def open_block_matrix_audit(factory=None, *, n_layers=6, verbose=True):
    """Zero-GPU check of the open-block panel's declared 2-factor factorial.

    The open-block panel is deliberately a max_block x reading-rule MATRIX
    (design_notes §24), so its intra-panel contrasts are not one-factor
    leave-one-out cells; what must still hold is that the two declared axes
    are the ONLY config axes that vary: same-cap mode contrasts differ in
    `open_block_mode` alone, same-mode cap contrasts differ in `max_block`
    alone, and scope variants differ in `open_block_scope` alone.  Any other
    key leaking into an intra-matrix diff would make the interaction
    statistics read a third, undeclared factor.

    Every axis cell is checked (not only the closed reading rule), in both
    scopes.  The diff must EQUAL the axis key: an extra key leaks a second
    factor, and a missing key means the declared axis no longer varies (the
    interaction statistic would then be read off a degenerate contrast).
    """
    factory = make_layer_cfgs if factory is None else factory
    report = {'pairs': {}, 'violations': [], 'status': None}
    mbs = list(OPEN_BLOCK_MAX_VALUES)
    modes = list(OPEN_BLOCK_MODES_SHORT)
    scopes = ('full_prefix', 'outside_window')
    checks = []
    seen = set()

    def _add(a, b, allowed, axis):
        if a == b or (a, b) in seen:
            return
        seen.add((a, b))
        checks.append((a, b, frozenset(allowed), axis))

    for scope in scopes:
        # mode axis at every cap
        for mb in mbs:
            for mode in modes:
                if mode != 'closed':
                    _add(open_block_variant_name(mb, mode, scope),
                         open_block_variant_name(mb, 'closed', scope),
                         {'open_block_mode'}, 'mode')
        # cap axis at every reading rule
        for mode in modes:
            for mb in mbs[1:]:
                _add(open_block_variant_name(mb, mode, scope),
                     open_block_variant_name(mbs[0], mode, scope),
                     {'max_block'}, 'cap')
        # scope axis at every (cap, mode) cell
        for mb in mbs:
            for mode in modes:
                _add(open_block_variant_name(mb, mode, 'outside_window'),
                     open_block_variant_name(mb, mode, 'full_prefix'),
                     {'open_block_scope'}, 'scope')
    for a, b, allowed, axis in checks:
        keys, per_layer = _design_pair_cfg_keys(a, b, factory, n_layers)
        if isinstance(keys, Exception):
            report['violations'].append(f'open-block {a} vs {b}: construction failed ({keys})')
            continue
        extra = sorted(set(keys) - allowed)
        missing = sorted(allowed - set(keys))
        label = f'{a} vs {b}'
        report['pairs'][label] = {'axis': axis, 'actual_keys': keys,
                                  'allowed_keys': sorted(allowed),
                                  'extra_keys': extra, 'missing_keys': missing}
        if extra:
            report['violations'].append(
                f'open-block {label}: intra-matrix diff leaks factor key(s) {extra} '
                f'outside the declared axis {sorted(allowed)}')
        if missing:
            report['violations'].append(
                f'open-block {label}: declared {axis} axis key(s) {missing} no longer '
                f'differ — the axis contrast is degenerate')
    report['status'] = 'fail' if report['violations'] else 'ok'
    if verbose:
        if report['violations']:
            print('[open-block-matrix] REFUSED:')
            for _v in report['violations']:
                print(f'  - {_v}')
        else:
            print(f"[open-block-matrix] OK: {len(report['pairs'])} intra-matrix contrast(s) vary only on the declared axes")
    return report

# ---------------------------------------------------------------------------
# 设计前馈（design preflight）：预算台账 / 证据等级 / 动态分块假设
# ---------------------------------------------------------------------------
# 这一层只做**零 GPU 的设计核验**，不读任何 test 数据：
#   * `comparison_budget_ledger` 把「同 token、同参数、同读取预算」从散文
#     声明变成逐臂可复算的台账；它回答的是「这条对照到底控制了哪些预算」，
#     而不是「结果显著吗」。
#   * `evidence_level_audit` 把探索/确认等级写进机器可检查的字段：v6-v11
#     这些在同一 validation split 上选择/比较出来的结果永远是 exploratory；
#     draft 清单永远不能产出 confirmatory 主张。
#   * `dynamic_design_audit` 要求动态分块研究先写清假设（切分质量、检索
#     精度、长程覆盖）和专用对照（随机选择臂、参数代理臂、读取预算匹配
#     臂），再允许冻结。
# 这些函数在 exp_lib 内定义，供 audit_gates/explib.confirmation 共同复用；不依赖任何
# 结果目录，因而可以在训练前、test 访问前运行。
EVIDENCE_LEVELS = {
    'confirmatory': {
        'definition': ('frozen pre-registered comparison evaluated once on a held-out '
                       'endpoint never used for selection'),
        'requirements': ('frozen manifest', 'pre-test power gate passed',
                         'complete frozen seed coverage', 'frozen endpoint',
                         'no post-hoc arm/seed/endpoint change'),
        'may_use_words': ('confirmed', 'confirmatory', '确认性'),
    },
    'exploratory': {
        'definition': ('comparison selected, tuned or observed on the same validation '
                       'split used for analysis; not licensed for confirmation'),
        'requirements': ('declared hypothesis', 'effect size and uncertainty reported',
                         'explicit exploratory label next to every quote'),
        'may_use_words': ('exploratory', 'candidate', '探索性'),
    },
    'descriptive': {
        'definition': ('secondary/diagnostic readout that does not enter any '
                       'inferential claim family'),
        'requirements': ('named as descriptive', 'never folded into Holm family'),
        'may_use_words': ('descriptive', 'diagnostic', '描述性'),
    },
    'draft': {
        'definition': ('design record; not runnable and not a pre-registration'),
        'requirements': ('hypotheses and dedicated controls declared',
                         'disjoint test source reserved', 'power gate defined'),
        'may_use_words': ('draft', '设计稿', '未冻结'),
    },
}
# Explicit registry of panels whose selection/reporting happened on the same
# validation split.  A result directory here can never be re-labelled
# confirmatory by editing a summary: the registry is code, and the audit fails
# when a panel declares `confirmatory` while sitting in this set.
EXPLORATORY_PANEL_REGISTRY = {
    'results_lm_v3_1500': 'v6 core table (9 variants, 1500 steps)',
    'results_lm_v3_long': 'v6 long-run table',
    'results_lm_v4_abl': 'v6 leave-one-out ablation table',
    'results_lm_v5_scale': 'v6 d=384/8L scale probe',
    'results_lm_v7_rope': 'v7 RoPE+QK-norm panel',
    'results_lm_v7_seq2k': 'v7 seq-2048 panel',
    'results_lm_v7_warmup': 'v7 dense-warmup panel',
    'results_lm_v7_long40': 'v7 40k-step long panel',
    'results_lm_v8_rope20k': 'v8 RoPE 20k panel',
    'results_lm_v9_seq2k': 'v9 seq-2048 topk panel',
    'results_v10_mech': 'v10 mechanism/probe panel',
    'results_lm_v10_scale_s': 'v10 scale small arm',
    'results_lm_v10_scale_l': 'v10 scale large arm',
    'results_lm_v10_scale_l8k': 'v10 token-matched large arm',
    'results_lm_v10_scale_lflop': 'v10 compute-proxy large arm',
    'results_open_block': 'open-block panel',
    'results_open_block_aux': 'open-block auxiliary panel',
    'results_open_block_extended': 'open-block extended panel',
}


def _evidence_manifest_path(manifest_path):
    if manifest_path is None:
        return None
    return os.path.abspath(str(manifest_path))


def _design_int(value):
    """Strict integer reader for design-time manifest fields.

    ``int(True) == 1`` and ``int("12") == 12`` are both dangerous in a
    pre-registration: a boolean seed/count can silently become a valid plan.
    ``_as_seed_int`` rejects booleans and non-integral floats while still
    accepting the integer strings used by legacy JSON artifacts.
    """
    return _as_seed_int(value)


def _design_probability(value, *, name, lo=0.0, hi=1.0,
                         lo_open=False, hi_open=False):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    if (f <= lo if lo_open else f < lo) or (f >= hi if hi_open else f > hi):
        return None
    return f


def _design_close(a, b, tol=1e-12):
    """Absolute closeness for design-time scalars, exact for integer counts.

    A declared seed/family count may exceed float64's range.  Converting both
    sides to float then made two equal 401-digit counts fail their own audit
    (OverflowError -> False).  Compare exact integers directly and reserve the
    float path for fractional values.
    """
    try:
        tol = float(tol)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(tol) or tol < 0.0:
        return False
    if (not isinstance(a, (bool, np.bool_))
            and not isinstance(b, (bool, np.bool_))
            and isinstance(a, numbers.Integral)
            and isinstance(b, numbers.Integral)):
        if int(a) == int(b):
            return True
        try:
            from fractions import Fraction
            return abs(int(a) - int(b)) <= Fraction.from_float(tol)
        except (OverflowError, ValueError):
            return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError, OverflowError):
        return False


def planned_power_audit(manifest, *, require_projection=False):
    """Zero-GPU consistency audit of the PRE-TEST power plan.

    A p-value floor is not a power calculation.  This audit checks the
    declared plan against the same exact sign-flip helpers the confirmation
    analysis uses:

      * ``statistics.n_seeds`` equals the unique frozen seed list;
      * the two-sided exact floor ``2/2**n`` and the Holm floor
        ``m*2/2**n`` are the numbers actually declared;
      * the minimum-seed floor still clears alpha;
      * if a projected power is declared (or required), recompute
        ``signflip_power(n, m, q, alpha)`` and refuse a plan whose projection
        is below ``min_power_for_test``;
      * a precondition.power_gate marked ``passed`` carries a numeric power
        at the declared q, not just a status label.

    The projection is the symbol-test projection, not the exact sign-flip
    power for unequal deltas; it is a design aid and is labelled as such.
    """
    out = {'status': 'ok', 'n_seeds': None, 'family_m': None, 'alpha': None,
           'min_seeds_for_confirmation': None, 'p_floor': None,
           'min_attainable_p_holm': None, 'power_q_for_test': None,
           'min_power_for_test': None, 'projected_power': None,
           'minimum_same_sign_seeds': None, 'declared': {},
           'violations': [], 'warnings': [],
           'method': ('exact two-sided sign-flip floor; power projection is the '
                      'equal-magnitude symbol-test aid, not the exact power for '
                      'unequal deltas')}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['violations'].append(
            f'manifest must be an object, got {type(manifest).__name__}')
        return out
    stat = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    comps = manifest.get('primary_comparisons')
    if not isinstance(comps, list):
        comps = []
    raw_seeds = manifest.get('seeds')
    seeds = []
    if isinstance(raw_seeds, list):
        for i, s in enumerate(raw_seeds):
            v = _design_int(s)
            if v is None or v < 0:
                out['violations'].append(
                    f'seeds[{i}]={s!r} is not a non-negative integer seed')
            else:
                seeds.append(v)
        if len(set(seeds)) != len(seeds):
            out['violations'].append('seeds contain duplicates')
    else:
        out['warnings'].append('manifest.seeds is not a list; the frozen seed count is unverifiable')
    n_stat = _design_int(stat.get('n_seeds'))
    if 'n_seeds' in stat and n_stat is None:
        out['violations'].append(
            f'statistics.n_seeds={stat.get("n_seeds")!r} is not an integer')
    n = len(seeds) if seeds else n_stat
    m = len(comps)
    out['n_seeds'], out['family_m'] = n, m
    if n_stat is not None and seeds and n_stat != len(seeds):
        out['violations'].append(
            f'statistics.n_seeds={n_stat} does not equal len(seeds)={len(seeds)}')
    if not n or n <= 0:
        out['violations'].append('no positive seed count; the power floor is unverifiable')
        n = None
    if m <= 0:
        out['warnings'].append('primary_comparisons is empty; the family size is zero')
    alpha_raw = stat.get('alpha', 0.05)
    alpha = _design_probability(alpha_raw, name='alpha', lo_open=True, hi_open=True)
    if alpha is None:
        out['violations'].append(
            f'statistics.alpha={alpha_raw!r} is not a finite number in (0,1)')
        alpha = 0.05
    out['alpha'] = alpha
    min_n = _design_int(stat.get('min_seeds_for_confirmation'))
    out['min_seeds_for_confirmation'] = min_n
    if min_n is None or min_n < 2 or (n and min_n > n):
        out['violations'].append(
            f'statistics.min_seeds_for_confirmation={min_n!r} is outside '
            f'[2, {n if n else "n"}]')
        min_n = None
    if n and m:
        floor = signflip_p_floor(n)
        out['p_floor'] = float(floor)
        declared_floor = stat.get('p_floor')
        if declared_floor is not None:
            out['declared']['p_floor'] = declared_floor
            if not _design_close(declared_floor, floor):
                out['violations'].append(
                    f'statistics.p_floor={declared_floor!r} is not the exact '
                    f'two-sided floor 2/2**n={floor!r} for n={n}')
        family_floor = holm_min_attainable_p(n, m, alpha)
        out['min_attainable_p_holm'] = float(family_floor)
        declared_family = stat.get('min_attainable_p_holm')
        if declared_family is not None:
            out['declared']['min_attainable_p_holm'] = declared_family
            if not _design_close(declared_family, family_floor):
                out['violations'].append(
                    f'statistics.min_attainable_p_holm={declared_family!r} is '
                    f'not m*2/2**n={family_floor!r} for n={n}, m={m}')
        if min_n:
            min_floor = holm_min_attainable_p(min_n, m, alpha)
            out['min_seed_attainable_p_holm'] = float(min_floor)
            if not math.isfinite(min_floor) or min_floor > alpha:
                out['violations'].append(
                    f'the minimum seed floor n={min_n} cannot attain Holm '
                    f'p<={alpha} (floor={min_floor!r})')
    q_raw = stat.get('power_q_for_test')
    mp_raw = stat.get('min_power_for_test')
    q = _design_probability(q_raw, name='power_q_for_test',
                            lo_open=True, hi_open=True) if q_raw is not None else None
    mp = _design_probability(mp_raw, name='min_power_for_test',
                             lo=0.0, hi=1.0) if mp_raw is not None else None
    out['power_q_for_test'], out['min_power_for_test'] = q, mp
    need_projection = bool(require_projection or q_raw is not None or mp_raw is not None)
    if need_projection:
        if q is None:
            out['violations'].append(
                f'statistics.power_q_for_test={q_raw!r} is not a finite '
                'probability in (0,1)')
        if mp is None:
            out['violations'].append(
                f'statistics.min_power_for_test={mp_raw!r} is not a finite '
                'probability in [0,1]')
        if n and m and q is not None:
            projected = float(signflip_power(n, m, q, alpha))
            out['projected_power'] = projected
            try:
                out['minimum_same_sign_seeds'] = signflip_min_consistency(n, m, alpha)
            except Exception:
                out['minimum_same_sign_seeds'] = None
            if mp is not None and projected + 1e-12 < mp:
                out['violations'].append(
                    f'the projected power at q={q} is {projected:.6f} < '
                    f'min_power_for_test={mp}; the planned contrast is '
                    'underpowered and cannot confirm even if it runs')
    proj = stat.get('power_projection')
    if isinstance(proj, dict):
        out['declared']['power_projection'] = proj
        pq = _design_probability(proj.get('q'), name='power_projection.q',
                                 lo_open=True, hi_open=True)
        pn = _design_int(proj.get('n_seeds'))
        pm = _design_int(proj.get('family_m'))
        pp = _design_probability(proj.get('power'), name='power_projection.power')
        if pq is None or pn is None or pm is None or pp is None:
            out['violations'].append(
                'statistics.power_projection must carry finite q/power and '
                'integer n_seeds/family_m')
        else:
            if n and pn != n:
                out['violations'].append(
                    f'power_projection.n_seeds={pn} != frozen seed count {n}')
            if m and pm != m:
                out['violations'].append(
                    f'power_projection.family_m={pm} != primary comparison count {m}')
            if q is not None and not _design_close(pq, q):
                out['violations'].append(
                    f'power_projection.q={pq} != statistics.power_q_for_test={q}')
            if n and m:
                want = float(signflip_power(pn, pm, pq, alpha))
                if not _design_close(pp, want, tol=1e-9):
                    out['violations'].append(
                        f'power_projection.power={pp!r} != recomputed '
                        f'signflip_power({pn},{pm},{pq})={want!r}')
    # The frozen v1/v2 power table is a design record too.  Each declared
    # q -> power cell must still equal the current sign-flip projection;
    # a stale table after changing n/m/alpha would misstate the plan.
    _table = stat.get('power_first_holm_gate')
    if isinstance(_table, dict) and n and m:
        for _key, _val in _table.items():
            _txt = str(_key).strip()
            if '=' not in _txt:
                out['warnings'].append(
                    f'power_first_holm_gate key {_key!r} has no q=... form; skipped')
                continue
            _q = _design_probability(_txt.split('=', 1)[1], name='power q',
                                     lo_open=True, hi_open=True)
            if _q is None:
                out['violations'].append(
                    f'power_first_holm_gate key {_key!r} has an invalid q')
                continue
            _want = float(signflip_power(n, m, _q, alpha))
            if not _design_close(_val, _want, tol=5e-4):
                out['violations'].append(
                    f'power_first_holm_gate[{_key!r}]={_val!r} != recomputed '
                    f'signflip_power({n},{m},{_q})={_want!r}')
    pre = manifest.get('precondition') if isinstance(manifest.get('precondition'), dict) else {}
    gate = pre.get('power_gate') if isinstance(pre.get('power_gate'), dict) else {}
    if gate.get('status') == 'passed':
        gp = _design_probability(gate.get('power'), name='power_gate.power')
        if gp is None:
            out['violations'].append(
                'precondition.power_gate.status=passed but power is missing/not '
                'a finite probability; a status label is not a power gate')
        else:
            if mp is not None and gp + 1e-12 < mp:
                out['violations'].append(
                    f'precondition.power_gate.power={gp} < min_power_for_test={mp}')
            if q is not None and gate.get('q') is not None and not _design_close(gate.get('q'), q):
                out['violations'].append(
                    f'precondition.power_gate.q={gate.get("q")!r} != '
                    f'statistics.power_q_for_test={q}')
    if out['violations']:
        out['status'] = 'fail'
    return out


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _normal_quantile(p):
    p = float(p)
    if not (0.0 < p < 1.0):
        return float('nan')
    lo, hi = -9.0, 9.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _design_direction(value):
    if value is None:
        return None, None
    if isinstance(value, (bool, np.bool_)):
        return None, f'expected_sign={value!r} must not be a boolean'
    if isinstance(value, numbers.Real) and not isinstance(value, complex):
        f = _finite_scalar(value)
        if f in (-1.0, 1.0):
            return int(f), None
        return None, f'expected_sign={value!r} must be -1 or +1'
    text = str(value).strip().lower()
    if text in ('positive', '+', '+1', '1'):
        return 1, None
    if text in ('negative', '-', '-1'):
        return -1, None
    return None, f'expected_sign={value!r} must be positive, negative, -1, +1, or null'


def _q_for_target_power(n_seeds, m_comparisons, target_power, alpha):
    if signflip_power(n_seeds, m_comparisons, 1.0, alpha) + 1e-15 < target_power:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if signflip_power(n_seeds, m_comparisons, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _n_for_target_power(n_seeds, m_comparisons, q, target_power, alpha, n_max):
    if q is None or q <= 0.5:
        return None
    for n in range(2, int(n_max) + 1):
        if signflip_power(n, m_comparisons, q, alpha) + 1e-15 >= target_power:
            return int(n)
    return None


def _n_for_target_halfwidth(sd, target, *, conf=0.95, n_max=512):
    """Smallest n in ``[2, n_max]`` whose paired-t CI half-width meets target.

    ``t_crit(conf, n-1) * sd / sqrt(n)`` is monotone decreasing in n, so a
    binary search returns the minimal integer without enumerating every n.
    The previous linear scan cost thousands of incomplete-beta evaluations at
    the upper end of ``n_max``.
    """
    try:
        sd = float(sd)
        target = float(target)
        hi = int(n_max)
    except (TypeError, ValueError, OverflowError):
        return None
    if (hi < 2 or not (math.isfinite(sd) and sd > 0.0)
            or not (math.isfinite(target) and target > 0.0)):
        return None

    def _ok(n):
        tc = _t_critical(conf, float(n) - 1.0)
        return (math.isfinite(tc)
                and tc * sd / math.sqrt(float(n)) <= target)

    if _ok(2):
        return 2
    if not _ok(hi):
        return None
    lo = 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _ok(mid):
            hi = mid
        else:
            lo = mid
    return int(hi)


def paired_design_sensitivity(n_seeds, m_comparisons, *, alpha=0.05,
                              target_power=0.80, pilot_deltas=None,
                              pilot_sd=None, sesoi=None, expected_sign=None,
                              target_halfwidth=None,
                              require_sesoi_power=False, n_max=512):
    """Design-stage power, MDE and precision for a paired seed panel.

    This audit answers a question the p-value floor and the declared-q power
    projection cannot answer: *what effect size can this panel actually
    resolve, and is that enough for the pre-registered practical margin?*

    It is deliberately separate from ``planned_power_audit``.  That audit
    checks that the frozen n/m/alpha/floors are internally consistent.  This
    audit takes a pilot or validation vector of paired deltas (or only their
    SD), converts a pre-registered SESOI into the sign probability
    ``q = Phi(|SESOI| / sigma)``, and evaluates the SAME Holm/sign-flip
    planning projection used by ``signflip_power``.  It also reports the
    minimum detectable mean under a normal sign model and the expected paired
    t CI half-width at the planned n.

    The normal sign conversion and the MDE are approximations.  The
    sign-flip projection is itself a symbol-test upper bound when the |d_i|
    are unequal, so this output is a planning aid, not a promise about the
    one-shot test.  No result file or test-split value is read here.
    """
    out = {'kind': 'paired_design_sensitivity', 'status': 'ok',
           'n_seeds': None, 'm_family': None, 'alpha': None,
           'target_power': None, 'p_floor': None,
           'min_attainable_p_holm': None, 'power_curve': {},
           'min_q_for_target_power': None, 'pilot': {}, 'mde': {},
           'precision': {}, 'sesoi': {}, 'violations': [],
           'warnings': [],
           'policy': {
               'alpha': 'two-sided exact sign-flip planning projection',
               'q_definition': ('probability that one seed\'s paired delta '
                                'falls in the pre-registered direction'),
               'sesoi_rule': ('q_sesoi = Phi(|SESOI|/pilot_sd); underpowered '
                              'at the margin is a design failure only when '
                              'require_sesoi_power is true'),
               'mde_rule': ('normal sign-model approximation: '
                            'sd * Phi^-1(q_target); not an exact sign-flip '
                            'effect-size calculation'),
               'precision_rule': ('paired t 95% CI half-width at the planned n; '
                                  'document-level precision requires the '
                                  'document-cluster SD, not the seed SD'),
               'n_rule': ('n is the planned seed count; required-n searches '
                          'assume the pilot q and SD stay fixed and do not '
                          'model attrition or incomplete pairs'),
           }}
    n = _design_int(n_seeds)
    m = _design_int(m_comparisons)
    n_max_i = _design_int(n_max)
    out['n_seeds'], out['m_family'] = n, m
    if n is None or n < 2:
        out['violations'].append(
            f'n_seeds={n_seeds!r} must be an integer >= 2')
        n = None
    if m is None or m < 1:
        out['violations'].append(
            f'm_comparisons={m_comparisons!r} must be a positive integer')
        m = None
    if n_max_i is None or n_max_i < 2 or n_max_i > 10000:
        out['violations'].append(
            f'n_max={n_max!r} must be an integer in [2, 10000]')
        n_max_i = 512
    a = _design_probability(alpha, name='alpha', lo_open=True, hi_open=True)
    if a is None:
        out['violations'].append(
            f'alpha={alpha!r} must be a finite probability in (0,1)')
    out['alpha'] = a
    tp = _design_probability(target_power, name='target_power',
                             lo_open=True, hi_open=True)
    if tp is None:
        out['violations'].append(
            f'target_power={target_power!r} must be a finite probability in (0,1)')
    out['target_power'] = tp
    if not isinstance(require_sesoi_power, (bool, np.bool_)):
        out['violations'].append(
            f'require_sesoi_power={require_sesoi_power!r} must be a boolean')
        require_sesoi_power = False
    require_sesoi_power = bool(require_sesoi_power)
    out['require_sesoi_power'] = require_sesoi_power
    target_halfwidth_value = None
    if target_halfwidth is not None:
        target_halfwidth_value = _finite_scalar(target_halfwidth)
        if target_halfwidth_value is None or target_halfwidth_value <= 0.0:
            out['violations'].append(
                f'target_halfwidth={target_halfwidth!r} must be a finite positive number')
            target_halfwidth_value = None
    out['target_halfwidth'] = target_halfwidth_value
    direction, direction_error = _design_direction(expected_sign)
    if direction_error:
        out['violations'].append(direction_error)
    out['expected_sign'] = direction
    raw_deltas = []
    dropped = 0
    if isinstance(pilot_deltas, (str, bytes)):
        out['violations'].append(
            'pilot_deltas must be a sequence of numbers, not a string')
        pilot_deltas = None
    if pilot_deltas is not None:
        try:
            for value in pilot_deltas:
                f = _finite_scalar(value)
                if f is None:
                    dropped += 1
                else:
                    raw_deltas.append(f)
        except TypeError:
            out['violations'].append(
                f'pilot_deltas must be an iterable of finite numbers, got {type(pilot_deltas).__name__}')
            raw_deltas = []
    d = np.asarray(raw_deltas, dtype=float)
    # The seed-delta scale is arbitrary.  numpy.mean/std square deviations, so
    # a finite pilot panel below sqrt(tiny) (or above sqrt(max_float)) reports
    # sd=0/inf and the whole planning output would lose its MDE and SESOI
    # fields for a units-only reason.  Rescale once and use the same mean/SD
    # for the consistency check and the reported pilot.
    mean = float(d.mean()) if d.size else None
    _pilot_sd_calc = None
    if d.size >= 2:
        _ms = _scaled_mean_std_if_extreme(d, ddof=1)
        if _ms is not None:
            mean, _pilot_sd_calc = _ms
        else:
            _pilot_sd_calc = float(d.std(ddof=1))
    explicit_sd = None
    if pilot_sd is not None:
        explicit_sd = _finite_scalar(pilot_sd)
        if explicit_sd is None or explicit_sd <= 0.0:
            out['violations'].append(
                f'pilot_sd={pilot_sd!r} must be a finite positive number')
            explicit_sd = None
    sd = None
    if explicit_sd is not None:
        sd = float(explicit_sd)
        if _pilot_sd_calc is not None:
            # Relative agreement: an absolute 1e-12 margin calls any two tiny
            # pilot SDs equal and any two 1e150 SDs different.
            if not math.isclose(_pilot_sd_calc, sd, rel_tol=1e-12, abs_tol=0.0):
                out['warnings'].append(
                    'pilot_sd overrides the SD recomputed from pilot_deltas; '
                    'report which sample each value came from')
    elif d.size >= 2:
        sd = _pilot_sd_calc
        if not (math.isfinite(sd) and sd > 0.0):
            out['warnings'].append(
                'pilot deltas have zero/non-finite SD; MDE and SESOI power '
                'are not computable')
            sd = None
    elif d.size == 1:
        out['warnings'].append(
            'only one finite pilot delta; a paired SD needs at least two')
    if dropped:
        out['warnings'].append(
            f'{dropped} non-finite/non-numeric pilot delta(s) were dropped')
    q_sign_rate = None
    direction_ok = None
    q_normal = None
    if d.size and direction is not None:
        q_sign_rate = float(np.count_nonzero(d * direction > 0.0) / d.size)
        if mean is not None and mean != 0.0:
            direction_ok = bool(mean * direction > 0.0)
        if sd is not None and sd > 0.0:
            q_normal = float(min(1.0, max(0.0, _normal_cdf(abs(mean) / sd))))
        if direction_ok is False:
            out['warnings'].append(
                'the pilot mean points AGAINST the pre-registered direction; '
                'the confirmation premise, not just its power, is in question')
    out['pilot'] = {
        'status': ('not_provided' if (not d.size and explicit_sd is None)
                   else 'available'),
        'n': int(d.size), 'n_dropped': int(dropped),
        'mean': mean, 'sd': sd, 'expected_sign': direction,
        'direction_ok': direction_ok, 'q_sign_rate': q_sign_rate,
        'q_normal_implied': q_normal,
        'source': ('deltas' if d.size else ('pilot_sd' if sd is not None else None)),
    }
    if n is not None and m is not None and a is not None:
        out['p_floor'] = float(signflip_p_floor(n))
        out['min_attainable_p_holm'] = float(holm_min_attainable_p(n, m, a))
        if out['min_attainable_p_holm'] > a + 1e-15:
            out['violations'].append(
                f'with n={n} seeds and m={m} comparisons the minimum attainable '
                f'Holm p is {out["min_attainable_p_holm"]:.6g} > alpha={a}; '
                'the panel is resolution-limited even if every seed agrees')
        for q in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99):
            out['power_curve'][f'q={q:.3f}'] = float(
                signflip_power(n, m, q, a))
        if tp is not None:
            out['min_q_for_target_power'] = _q_for_target_power(n, m, tp, a)
            if out['min_q_for_target_power'] is None:
                out['warnings'].append(
                    f'target_power={tp} is unreachable at n={n}, m={m}, alpha={a}')
    if sd is not None and n is not None:
        tc = _t_critical(0.95, float(n - 1))
        half = (float(tc) * sd / math.sqrt(n)) if math.isfinite(tc) else None
        out['precision'] = {
            'confidence': 0.95, 't_critical': float(tc) if math.isfinite(tc) else None,
            'ci95_halfwidth': half, 'target_halfwidth': None,
            'n_required_for_target_halfwidth': None, 'meets_target': None,
        }
        if target_halfwidth_value is not None:
            th = target_halfwidth_value
            out['precision']['target_halfwidth'] = float(th)
            out['precision']['meets_target'] = bool(half is not None and half <= th)
            out['precision']['n_required_for_target_halfwidth'] = (
                _n_for_target_halfwidth(sd, th, n_max=n_max_i))
        if half is not None and out['precision'].get('meets_target') is False:
            out['warnings'].append(
                'the expected paired 95% CI half-width at the planned n '
                f'({half:.6g}) exceeds target_halfwidth={target_halfwidth!r}')
    if q_sign_rate is not None and tp is not None and n is not None and m is not None and a is not None:
        needed = _n_for_target_power(n, m, q_sign_rate, tp, a, n_max_i)
        out['pilot']['required_n_for_target_power'] = needed
        out['pilot']['n_shortfall'] = (None if needed is None
                                       else max(0, needed - n))
        observed_power = float(signflip_power(n, m, q_sign_rate, a))
        out['pilot']['power_at_q_sign_rate'] = observed_power
        if observed_power + 1e-15 < tp:
            out['warnings'].append(
                f'the observed pilot sign rate q={q_sign_rate:.6g} gives power '
                f'{observed_power:.6g} < target_power={tp}')
    if sd is not None and tp is not None and out.get('min_q_for_target_power'):
        q_target = out['min_q_for_target_power']
        if q_target > 0.5 + 1e-12:
            mde = sd * _normal_quantile(q_target)
            out['mde'] = {
                'absolute': float(mde), 'q_for_target_power': float(q_target),
                'relative_to_pilot_sd': (float(mde) / sd if sd else None),
                'method': ('normal sign-probability approximation '
                           '(sd * Phi^-1(q_target)); sign-flip power remains '
                           'a symbol-test planning projection'),
            }
            if sesoi is not None:
                sesoi_value = _finite_scalar(sesoi)
                if sesoi_value is not None and sesoi_value > 0.0:
                    out['mde']['multiple_of_sesoi'] = float(mde / sesoi_value)
    sesoi_value = None
    if sesoi is not None:
        sesoi_value = _finite_scalar(sesoi)
        if sesoi_value is None or sesoi_value <= 0.0:
            out['violations'].append(
                f'sesoi={sesoi!r} must be a finite positive margin in the '
                'same units as the paired deltas')
            sesoi_value = None
    out['sesoi'] = {
        'margin': sesoi_value, 'status': 'not_declared',
        'q_implied': None, 'power_at_sesoi': None,
        'required_n_for_target_power': None, 'power_ok': None,
        'note': ('no pre-registered SESOI supplied; statistical detectability '
                 'does not establish practical relevance'),
    }
    if require_sesoi_power and sesoi is None:
        out['violations'].append(
            'require_sesoi_power=true but no SESOI margin was declared')
    if sesoi_value is not None:
        if sd is None:
            out['sesoi'].update({'status': 'not_verifiable',
                                 'note': 'SESOI declared but no pilot SD is available'})
            if require_sesoi_power:
                out['violations'].append(
                    'require_sesoi_power=true but the pilot SD is missing')
        elif direction is None:
            out['sesoi'].update({'status': 'no_direction',
                                 'note': ('a non-directional comparison has no '
                                          'sign probability; report the margin '
                                          'and CI width instead')})
            if require_sesoi_power:
                out['violations'].append(
                    'require_sesoi_power=true but the comparison is non-directional')
        elif tp is not None and n is not None and m is not None and a is not None:
            q_sesoi = float(min(1.0, max(0.0, _normal_cdf(abs(sesoi_value) / sd))))
            power_sesoi = float(signflip_power(n, m, q_sesoi, a))
            needed = _n_for_target_power(n, m, q_sesoi, tp, a, n_max_i)
            ok = bool(power_sesoi + 1e-15 >= tp)
            out['sesoi'].update({
                'status': 'ok' if ok else 'underpowered',
                'q_implied': q_sesoi, 'power_at_sesoi': power_sesoi,
                'required_n_for_target_power': needed, 'power_ok': ok,
                'note': ('the planned n meets the target power at the '
                         'pre-registered margin' if ok else
                         'the planned n cannot reach target power at the '
                         'pre-registered margin'),
            })
            if not ok:
                out['warnings'].append(
                    f'the pre-registered SESOI has power {power_sesoi:.6g} < '
                    f'target_power={tp}')
                if require_sesoi_power:
                    out['violations'].append(
                        'require_sesoi_power=true and the planned n is '
                        'underpowered at the pre-registered SESOI')
    if not out['pilot'].get('n') and pilot_sd is None:
        out['warnings'].append(
            'no pilot deltas or SD supplied; the power curve is informational '
            'and MDE/SESOI power cannot be computed')
    if out['violations']:
        out['status'] = 'fail'
    return out


def _attrition_probability_runnable(n_seeds, pair_retention, threshold):
    """P(K >= threshold) for K ~ Binomial(n_seeds, pair_retention)."""
    if threshold is None or threshold > n_seeds:
        return 0.0
    if pair_retention >= 1.0:
        return 1.0
    if pair_retention <= 0.0:
        return 0.0
    return float(_binomial_sf_stable(int(threshold), int(n_seeds),
                                     float(pair_retention)))


def _attrition_joint_power(n_seeds, pair_retention, threshold,
                           power_by_k, log_factorials):
    """P(reject AND analysis runnable) under independent pair retention.

    ``power_by_k`` is the conditional sign-flip power at k complete pairs.
    The joint event requires K >= threshold and, conditional on K, a
    rejection in the pre-registered direction.
    """
    if threshold is None or threshold > n_seeds:
        return 0.0
    if pair_retention <= 0.0:
        return 0.0
    if pair_retention >= 1.0:
        return float(power_by_k[n_seeds])
    logp = math.log(pair_retention)
    log1p = math.log1p(-pair_retention)
    lf = log_factorials
    terms = []
    for k in range(max(0, int(threshold)), n_seeds + 1):
        if power_by_k[k] <= 0.0:
            continue
        log_pmf = (lf[n_seeds] - lf[k] - lf[n_seeds - k]
                   + k * logp + (n_seeds - k) * log1p)
        if log_pmf > -745.0:
            terms.append(math.exp(log_pmf) * power_by_k[k])
    return float(min(1.0, max(0.0, math.fsum(terms))))


def attrition_power_audit(n_seeds, m_comparisons, *, q,
                          pair_retention=None, alpha=0.05,
                          target_power=0.80,
                          require_complete_coverage=True,
                          min_complete_pairs=None,
                          availability_target=None,
                          require_attrition_power=False,
                          n_max=512,
                          retention_scenarios=(1.0, 0.99, 0.97, 0.95, 0.90)):
    """Design-stage power after pair loss, not power on a perfect panel.

    ``paired_design_sensitivity`` and the pre-test power gate condition on
    every planned pair entering the analysis.  A confirmation rule with
    ``require_complete_coverage=true`` is all-or-nothing: one missing pair
    makes the comparison unconfirmed even if the remaining pairs would reject.
    The probability of a confirmable result is therefore

        P(reject AND runnable) = E[ 1(K >= threshold) * power(K) ],

    where K ~ Binomial(n_seeds, pair_retention).  This audit reports that
    joint probability, the chance the analysis is runnable at all, the
    retention needed to reach the target, and a retention stress-test table.
    It does not infer attrition from the test split and never changes a
    frozen decision rule.

    ``pair_retention`` is a planning assumption, not a measured quantity.
    Independent losses are the optimistic case; failures driven by the same
    machine/week/health gate are correlated, and losses associated with the
    effect (for example a crashing arm) violate the exchangeability the
    conditional power projection already assumes.  A complete-coverage plan
    cannot replace a missing seed with a smaller-n confirmation; treating
    ``p**n`` as the runnable probability is the best case.
    """
    out = {
        'kind': 'attrition_power_audit', 'status': 'ok',
        'n_seeds': None, 'm_family': None, 'alpha': None, 'q': None,
        'target_power': None, 'pair_retention': None,
        'require_complete_coverage': None, 'min_complete_pairs': None,
        'runnable_at': None, 'family_floor': None,
        'expected_complete_pairs': None,
        'conditional_power': None, 'probability_runnable': None,
        'probability_at_least_family_floor': None,
        'unconditional_power': None, 'power_loss': None,
        'power_loss_relative': None, 'target_met': None,
        'coverage_bottleneck': None,
        'required_nominal_seeds': None,
        'max_unconditional_power': None,
        'n_at_max_unconditional_power': None,
        'min_pair_retention_for_target': None,
        'failure_reason': None, 'scenarios': [],
        'violations': [], 'warnings': [],
        'policy': {
            'loss_model': ('independent Bernoulli pair retention; correlated '
                           'or effect-related loss is worse than reported'),
            'conditional_power': ('equal-magnitude symbol-test projection '
                                  'signflip_power(k, m, q, alpha); a design '
                                  'aid, not the exact power for unequal |d_i|'),
            'complete_coverage': ('runnable iff all planned pairs complete; '
                                  'unconditional power = P(K=n) * power(n)'),
            'min_complete_pairs': ('when complete coverage is off, runnable '
                                   'iff K >= max(Holm floor, min_complete_pairs)'),
            'retention_rule': ('pair_retention must be declared or supplied; '
                               'the audit does not estimate it from outcome data'),
        },
    }
    n = _design_int(n_seeds)
    m = _design_int(m_comparisons)
    n_max_i = _design_int(n_max)
    if n is None or n < 2:
        out['violations'].append(f'n_seeds={n_seeds!r} must be an integer >= 2')
        n = None
    if m is None or m < 1:
        out['violations'].append(
            f'm_comparisons={m_comparisons!r} must be a positive integer')
        m = None
    if n_max_i is None or n_max_i < 2 or n_max_i > 10000:
        out['violations'].append(
            f'n_max={n_max!r} must be an integer in [2, 10000]')
        n_max_i = 512
    a = _design_probability(alpha, name='alpha', lo_open=True, hi_open=True)
    if a is None:
        out['violations'].append(
            f'alpha={alpha!r} must be a finite probability in (0,1)')
    tp = _design_probability(target_power, name='target_power',
                             lo_open=True, hi_open=True)
    if tp is None:
        out['violations'].append(
            f'target_power={target_power!r} must be a finite probability in (0,1)')
    qv = _design_probability(q, name='q')
    if qv is None:
        out['violations'].append(
            f'q={q!r} must be a finite probability in [0,1]')
    if not isinstance(require_complete_coverage, (bool, np.bool_)):
        out['violations'].append(
            f'require_complete_coverage={require_complete_coverage!r} must be a boolean')
        require_complete_coverage = True
    require_cov = bool(require_complete_coverage)
    if not isinstance(require_attrition_power, (bool, np.bool_)):
        out['violations'].append(
            f'require_attrition_power={require_attrition_power!r} must be a boolean')
        require_attrition_power = False
    require_attr = bool(require_attrition_power)
    min_pairs = None
    if min_complete_pairs is not None:
        min_pairs = _design_int(min_complete_pairs)
        if min_pairs is None or min_pairs < 0:
            out['violations'].append(
                f'min_complete_pairs={min_complete_pairs!r} must be a '
                'non-negative integer')
            min_pairs = None
    p = None
    if pair_retention is not None:
        p = _design_probability(pair_retention, name='pair_retention')
        if p is None:
            out['violations'].append(
                f'pair_retention={pair_retention!r} must be a finite '
                'probability in [0,1]')
    avail_t = None
    if availability_target is not None:
        avail_t = _design_probability(availability_target,
                                      name='availability_target',
                                      lo_open=True)
        if avail_t is None:
            out['violations'].append(
                f'availability_target={availability_target!r} must be a '
                'finite probability in (0,1]')
    out['n_seeds'], out['m_family'] = n, m
    out['alpha'], out['target_power'] = a, tp
    out['q'], out['pair_retention'] = qv, p
    out['require_complete_coverage'] = require_cov
    out['min_complete_pairs'] = min_pairs
    out['availability_target'] = avail_t
    out['require_attrition_power'] = require_attr
    search_max = n_max_i
    out['n_max'] = search_max
    if not require_cov and n is not None and n > 10000:
        out['violations'].append(
            f'planned n_seeds={n} exceeds the partial-coverage scan cap '
            '10000; the attrition audit cannot verify a joint power for '
            'this design')
        out['failure_reason'] = 'scan_cap'
        out['status'] = 'fail'
        return out
    array_max = max(search_max, n) if (not require_cov and n is not None) else search_max
    if (not require_cov and min_pairs is not None and n is not None
            and min_pairs > n):
        out['warnings'].append(
            f'min_complete_pairs={min_pairs} exceeds the planned n={n}; '
            'the current panel cannot satisfy its own floor')
    if not require_cov and min_pairs is not None and min_pairs > search_max:
        out['warnings'].append(
            f'min_complete_pairs={min_pairs} exceeds the search cap '
            f'n_max={search_max}; runnable is impossible within this scan')
    if n is None or m is None or a is None or qv is None or tp is None:
        if out['violations']:
            out['status'] = 'fail'
        return out
    floor_max = max(search_max, min(n or 0, 10000))
    family_floor = signflip_min_n_for_holm(m, a, n_max=floor_max)
    out['family_floor'] = family_floor
    if family_floor is None:
        out['violations'].append(
            f'no n <= {floor_max} reaches the Holm floor for m={m} at '
            f'alpha={a}; the family is resolution-limited')

    def _runnable_threshold(nn):
        if require_cov:
            return nn
        if family_floor is None:
            return None
        return max(family_floor, min_pairs or 0)

    runnable_at = _runnable_threshold(n)
    out['runnable_at'] = runnable_at
    if (not require_cov and family_floor is not None and min_pairs is not None
            and min_pairs < family_floor):
        out['warnings'].append(
            f'min_complete_pairs={min_pairs} is below the Holm resolution '
            f'floor {family_floor}; the floor governs runnability')
    power_by_k = [0.0] * (array_max + 1)
    for k in range(2, array_max + 1):
        power_by_k[k] = float(signflip_power(k, m, qv, a))
    log_factorials = [math.lgamma(i + 1.0) for i in range(array_max + 1)]

    def _power_at(k):
        k = int(k)
        if 0 <= k <= array_max:
            return float(power_by_k[k])
        return float(signflip_power(k, m, qv, a))

    def _joint(nn, pp, threshold):
        nn = int(nn)
        if nn <= array_max:
            return _attrition_joint_power(
                nn, float(pp), threshold, power_by_k, log_factorials)
        if threshold is None or threshold > nn:
            return 0.0
        if pp <= 0.0:
            return 0.0
        if pp >= 1.0:
            return _power_at(nn)
        logp = math.log(pp)
        log1p = math.log1p(-pp)
        lf_n = math.lgamma(nn + 1.0)
        terms = []
        for k in range(max(0, int(threshold)), nn + 1):
            pk = _power_at(k)
            if pk <= 0.0:
                continue
            log_pmf = (lf_n - math.lgamma(k + 1.0)
                       - math.lgamma(nn - k + 1.0)
                       + k * logp + (nn - k) * log1p)
            if log_pmf > -745.0:
                terms.append(math.exp(log_pmf) * pk)
        return float(min(1.0, max(0.0, math.fsum(terms))))

    def _avail(nn, pp, threshold):
        return _attrition_probability_runnable(int(nn), float(pp), threshold)

    conditional = _power_at(n)
    out['conditional_power'] = conditional
    if p is not None:
        out['expected_complete_pairs'] = float(n * p)
        out['probability_runnable'] = _avail(n, p, runnable_at)
        if family_floor is not None:
            out['probability_at_least_family_floor'] = _avail(
                n, p, family_floor)
        out['unconditional_power'] = _joint(n, p, runnable_at)
        out['power_loss'] = float(conditional - out['unconditional_power'])
        out['power_loss_relative'] = (
            float(out['power_loss'] / conditional) if conditional > 0.0 else None)
        out['target_met'] = bool(
            out['unconditional_power'] + 1e-15 >= tp
            and (avail_t is None
                 or out['probability_runnable'] + 1e-15 >= avail_t))
        out['coverage_bottleneck'] = bool(
            conditional + 1e-15 >= tp
            and out['unconditional_power'] + 1e-15 < tp)
    if conditional + 1e-15 >= tp:
        def _retention_ok(pp):
            if _joint(n, pp, runnable_at) + 1e-15 < tp:
                return False
            if (avail_t is not None
                    and _avail(n, pp, runnable_at) + 1e-15 < avail_t):
                return False
            return True
        if _retention_ok(1.0):
            lo, hi = 0.0, 1.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _retention_ok(mid):
                    hi = mid
                else:
                    lo = mid
            out['min_pair_retention_for_target'] = float(hi)
    scenarios = []
    if retention_scenarios is not None:
        if isinstance(retention_scenarios, (str, bytes)):
            out['violations'].append(
                'retention_scenarios must be an iterable of probabilities')
        else:
            try:
                scenarios = list(retention_scenarios)
            except TypeError:
                out['violations'].append(
                    'retention_scenarios must be an iterable of probabilities')
                scenarios = []
    for idx, sp in enumerate(scenarios):
        spv = _design_probability(sp, name=f'retention_scenarios[{idx}]')
        if spv is None:
            out['violations'].append(
                f'retention_scenarios[{idx}]={sp!r} must be a probability in [0,1]')
            continue
        if any(abs(spv - x) <= 1e-15 for x in
               [s['pair_retention'] for s in out['scenarios']]):
            continue
        joint = _joint(n, spv, runnable_at)
        avail = _avail(n, spv, runnable_at)
        out['scenarios'].append({
            'pair_retention': float(spv),
            'expected_complete_pairs': float(n * spv),
            'probability_runnable': avail,
            'conditional_power': conditional,
            'unconditional_power': joint,
            'power_loss': float(conditional - joint),
            'availability_ok': (None if avail_t is None
                                else bool(avail + 1e-15 >= avail_t)),
            'target_met': bool(joint + 1e-15 >= tp
                               and (avail_t is None
                                    or avail + 1e-15 >= avail_t)),
        })
    if p is not None and not any(
            abs(p - s['pair_retention']) <= 1e-15 for s in out['scenarios']):
        joint = _joint(n, p, runnable_at)
        avail = _avail(n, p, runnable_at)
        out['scenarios'].append({
            'pair_retention': float(p),
            'expected_complete_pairs': float(n * p),
            'probability_runnable': avail,
            'conditional_power': conditional,
            'unconditional_power': joint,
            'power_loss': float(conditional - joint),
            'availability_ok': (None if avail_t is None
                                else bool(avail + 1e-15 >= avail_t)),
            'target_met': bool(joint + 1e-15 >= tp
                               and (avail_t is None
                                    or avail + 1e-15 >= avail_t)),
        })
    out['scenarios'].sort(key=lambda s: s['pair_retention'], reverse=True)
    if p is not None:
        max_joint = 0.0
        max_n = None
        required = None
        for nn in range(2, search_max + 1):
            threshold = _runnable_threshold(nn)
            joint = _joint(nn, p, threshold)
            if joint > max_joint + 1e-15:
                max_joint, max_n = joint, nn
            if required is not None:
                continue
            avail = _avail(nn, p, threshold)
            if (joint + 1e-15 >= tp
                    and (avail_t is None or avail + 1e-15 >= avail_t)):
                required = nn
        out['max_unconditional_power'] = float(max_joint)
        out['n_at_max_unconditional_power'] = max_n
        out['required_nominal_seeds'] = required
        if out['target_met']:
            out['status'] = 'ok'
        elif required is not None:
            out['status'] = 'underpowered'
            out['warnings'].append(
                f'the planned n={n} has attrition-adjusted power '
                f'{out["unconditional_power"]:.6g} < target_power={tp}; '
                f'the smallest nominal n meeting the target at '
                f'pair_retention={p} is {required}')
        elif family_floor is None:
            out['status'] = 'impossible'
            out['failure_reason'] = 'resolution_floor'
        elif max_joint + 1e-15 < tp:
            out['status'] = 'impossible'
            out['failure_reason'] = 'power'
            out['warnings'].append(
                f'no n <= {search_max} reaches attrition-adjusted power '
                f'{tp} at pair_retention={p}; the maximum is {max_joint:.6g}')
        else:
            out['status'] = 'impossible'
            out['failure_reason'] = 'availability'
            out['warnings'].append(
                f'no n <= {search_max} meets both target power={tp} and '
                f'availability_target={avail_t} at pair_retention={p}')
        if require_attr and not out['target_met']:
            out['violations'].append(
                'require_attrition_power=true but the planned panel has '
                f'attrition-adjusted power {out["unconditional_power"]:.6g} '
                f'< target_power={tp}'
                + ('' if avail_t is None else
                   f' or runnable probability {out["probability_runnable"]:.6g} '
                   f'< availability_target={avail_t}'))
    else:
        if require_attr:
            out['violations'].append(
                'require_attrition_power=true but no pair_retention was declared')
        else:
            out['warnings'].append(
                'no pair_retention was declared; the retention table is a '
                'stress test, not a pass, and no attrition-adjusted verdict '
                'is issued')
        out['status'] = 'fail' if require_attr else 'not_declared'
    if out['violations']:
        out['status'] = 'fail'
    return out


_UNIT_GRANULARITIES = (
    ('article', 'ppl_per_article', 'article_ids'),
    ('document', 'ppl_per_doc', 'doc_ids'),
    ('row', 'ppl_per_seq', None),
)


def _unit_sequence(value):
    """Non-empty one-dimensional unit vector (list, tuple or NumPy array)."""
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size > 0
    return isinstance(value, (list, tuple)) and len(value) > 0
