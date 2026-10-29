

# ---------------------------------------------------------------------------
# 7d) methodology：受控对比 / 消融完整性 / 统计可靠性
# ---------------------------------------------------------------------------

def audit_methodology(outdirs=None, design=None, alpha=0.05, min_n=3, write=True):
    """对比实验与消融审计（零 GPU，读落盘 summary）。

    对每个结果目录复用 `exp_lib.methodology_audit`，并额外给出三项面板级
    读数：
      * controlled_variables：逐臂全部种子记录（不是只看第一条）的
        run_cfg / train_knobs / structured hyperparams 一致性；
      * ablation_completeness：本面板实际进入族的创新点是否每个都具备全部
        声明的留一对照（缺臂会点名，部分组合不能支撑「各自有用」）；
      * arm_summary：每条臂的 n、mean±std 与是否达到 min_n（关键实验至少
        3–5 个种子的可复核读数）。

    `ablation_coverage` 进一步汇总跨 panel 的 complete/verified 覆盖、
    replicated / single_panel 复现状态，以及任意 panel 的显著反方向
    （direction_conflict）或实际有害（practically_harmful）证据；后者
    关闭 `all_components_useful_allowed`，不会因为另一个 panel 通过而被
    平均掉。`component_draws` 记录每个 component 的 seeds / evaluation
    fingerprints / code / knobs 身份，`replication_independence` 把
    `replicated` 细分为独立 draws、仅新 seeds、共享 seeds 和未核验四档；
    这些标签只影响复现措辞，不参与 p 值或 Holm 校正。

    没有 `results_*/summary.json` 时返回 not_verifiable_locally，而不是
    伪造一个通过。"""
    L = _import_exp_lib()
    # These are gate parameters, not measurements.  Validate them even when
    # there are no panels to audit: float(True)=1.0 / int(3.5)=3 would
    # otherwise be echoed as if the audit had run under a valid threshold.
    alpha = L._validated_alpha(alpha, context='audit_methodology alpha')
    _min_n = L._as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    dirs = list(outdirs) if outdirs is not None else list(_iter_result_dirs())
    out = {'status': 'ok', 'alpha': float(alpha), 'min_n': int(min_n), 'panels': {},
           'ablation_coverage': {}}
    _reports = {}
    for d in dirs:
        _dir = d if os.path.isabs(d) else os.path.join(REPO, d)
        summ = _load(os.path.join(_dir, 'summary.json'))
        if not summ:
            continue
        rep = L.methodology_audit(summ, design=design, alpha=alpha, min_n=min_n,
                                  outdir=_dir, write=write, verbose=False)
        _reports[d] = rep
        _panel = {
            'ablation_complete': rep.get('ablation_complete'),
            'ablation_completeness': rep.get('ablation_completeness'),
            'component_draws': rep.get('component_draws'),
            'controlled_variables': rep.get('controlled_variables'),
            'arm_summary': rep.get('arm_summary'),
            'arms_below_min_seeds': rep.get('arms_below_min_seeds'),
            'verified_components': rep.get('verified_components'),
            'practically_useful_components': rep.get('practically_useful_components'),
            'practical_equivalent_components': (
                (rep.get('sesoi') or {}).get('equivalent_components')),
            'sub_sesoi_components': rep.get('sub_sesoi_components'),
            'practically_harmful_components': rep.get('practically_harmful_components'),
            'practically_inconclusive_components': rep.get('practically_inconclusive_components'),
            'sesoi_family': (rep.get('sesoi') or {}).get('family'),
            'unverified_components': rep.get('unverified_components'),
            'system_level_only_components': rep.get('system_level_only_components'),
            'incomplete_components': rep.get('incomplete_components'),
            # Seed robustness / capacity confounding are per-contrast reads;
            # surface them at panel level so a report reader does not have to
            # know the nested entry layout to notice them.
            'seed_driven_components': sorted(
                c for c, es in (rep.get('components') or {}).items()
                if any(e.get('seed_driven') for e in es)),
            'capacity_confounded_components': sorted(
                c for c, es in (rep.get('components') or {}).items()
                if any(e.get('capacity_confounded') for e in es)),
            'interaction_family': (rep.get('interactions') or {}).get('family'),
            'interaction_factorials': (rep.get('interactions') or {}).get('factorials'),
            'significant_interactions': (
                (rep.get('interactions') or {}).get('significant_interactions')),
            'interaction_direction_conflicts': (
                (rep.get('interactions') or {}).get('direction_conflict_interactions')),
            'warnings': rep.get('warnings'),
        }
        out['panels'][d] = _panel
        if _panel['arms_below_min_seeds'] or _panel['warnings']:
            out['status'] = 'warn'
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no results_*/summary.json found — run the experiment phases first; the audit is zero-GPU over saved records'
    else:
        out['ablation_coverage'] = L.ablation_coverage_across_panels(_reports, design=design)
        # Project-level claim gate: "each proposed innovation is useful" needs
        # a complete ablation AND corrected verification for every component.
        out['all_components_useful_allowed'] = bool(
            out['ablation_coverage'].get('all_components_useful_allowed'))
        out['all_components_useful_reasons'] = list(
            out['ablation_coverage'].get('all_components_useful_reasons') or [])
        out['unverified_components'] = list(
            out['ablation_coverage'].get('unverified_components') or [])
        out['verified_only_incomplete'] = list(
            out['ablation_coverage'].get('verified_only_incomplete') or [])
        out['replicated_components'] = list(
            out['ablation_coverage'].get('replicated_anywhere') or [])
        out['independently_replicated_components'] = list(
            out['ablation_coverage'].get('independently_replicated_anywhere') or [])
        out['new_seed_replicated_components'] = list(
            out['ablation_coverage'].get('new_seed_replicated_anywhere') or [])
        out['shared_seed_replicated_components'] = list(
            out['ablation_coverage'].get('shared_seed_replicated_anywhere') or [])
        out['unattested_replicated_components'] = list(
            out['ablation_coverage'].get('unattested_replicated_anywhere') or [])
        out['single_panel_verified_components'] = list(
            out['ablation_coverage'].get('single_panel_verified') or [])
        out['direction_conflict_across_panels'] = list(
            out['ablation_coverage'].get('direction_conflict_anywhere') or [])
        out['practically_harmful_across_panels'] = list(
            out['ablation_coverage'].get('practically_harmful_anywhere') or [])
        out['opposite_mean_across_panels'] = list(
            out['ablation_coverage'].get('opposite_mean_anywhere') or [])
        if out['ablation_coverage'].get('status') != 'ok':
            out['status'] = 'warn'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e) single-factor：单因素设计契约（定义点审计，不读任何 results_*）
# ---------------------------------------------------------------------------

def audit_single_factor(manifest_path=None):
    """单因素设计审计（零 GPU、零落盘依赖）。

    与 `methodology`（读 results_*/summary.json 的记录级门）互补，本审计在
    **定义点**核验每条声明对照的臂配置 diff：
      * ABLATION_DESIGN / BASELINE_DESIGN 的每条对照，其实际 AttnCfg diff
        必须恰好等于声明的干预键集（`ABLATION_INTERVENTION_KEYS`）：多一个键
        = 第二个因素混进「单因素」对照；少一个键 = 干预消失；缺声明 =
        fail-closed，加对照不声明键集本身就是错误；
      * open-block 2x2 矩阵：同一 cap 的 mode 对照只差 `open_block_mode`，
        同一 mode 的 cap 对照只差 `max_block`，scope 对照只差
        `open_block_scope`——交互统计只能读这两根轴；每个轴单元（含两种
        scope）都要核验，多一个键=漏进第三因素，少一个键=轴退化；
      * v9/v10 seq2k 扫描声明（`v9.SEQ2K_CONTRAST_FACTORS`）：把元数据里的
        `single_factor`/`factors` 与真实臂配置 diff 对账，单因素标签不能只
        靠手写；
      * v7 RoPE / 选择器 / sink 对照声明（`v7.ROPE_CONTRAST_FACTORS`）：把
        位置编码、选择器、全块读取与 sink 对照的 factors 与 v7 工厂的真实
        arm diff 对账；
      * v10 NIAH 对照声明（`exp_lib.NIAH_CONTRAST_FACTORS`）：推理期 randidx
        /allblocks override 必须只改 `indexer_mode`/`index_topk`，from-scratch
        对照必须与 v7 工厂的真实 arm diff 一致；dense 对照明确为架构级；
      * 冻结清单（confirmation_frozen.json）：每条 primary comparison 的
        attention-cfg diff 是否单因素（`cfg_keys_single_factor` 的一族判据），
        与 `mechanism_claim_allowed` 标志的一致性。`True` 标志但 cfg 非单因素
        => fail（完整的含目标/参数量核验在 `python -m explib.confirmation plan`，本审计给出
        同一口径的 cfg 级读数）。
    """
    L = _import_exp_lib()
    out = {'status': 'ok', 'design': {}, 'open_block_matrix': {},
           'sweep_contrasts': {}, 'v7_contrasts': {}, 'niah_contrasts': {},
           'frozen_manifest': {}, 'factorial_design': {},
           'content_controls': {}}
    design_rep = L.single_factor_design_audit(verbose=False)
    out['design'] = design_rep
    ob_rep = L.open_block_matrix_audit(verbose=False)
    out['open_block_matrix'] = ob_rep
    fact_rep = L.factorial_design_audit()
    out['factorial_design'] = fact_rep
    ctrl_rep = L.content_control_audit()
    out['content_controls'] = ctrl_rep
    if (design_rep.get('status') != 'ok' or ob_rep.get('status') != 'ok'
            or fact_rep.get('status') != 'ok'
            or ctrl_rep.get('status') != 'ok'):
        out['status'] = 'fail'
    # v9/v10 attach a `single_factor`/`factors` label to each seq2k paired
    # record.  Verify that label against the actual arm construction instead of
    # trusting the hand-written metadata.
    try:
        import explib.resume as V9
        sw_rep = L.sweep_contrast_factor_audit(V9.SEQ2K_CONTRAST_FACTORS,
                                               verbose=False)
    except Exception as _e:
        sw_rep = {'status': 'not_verifiable_locally',
                  'note': f'cannot import the v9 sweep declarations '
                          f'({type(_e).__name__}: {_e})'}
    out['sweep_contrasts'] = sw_rep
    if sw_rep.get('status') == 'fail':
        out['status'] = 'fail'
    # The frozen family resolves through the v7 factory (rope arms); the base
    # factory is its delegate for non-rope arms, so one factory covers both.
    try:
        import explib.rope as V7
        _factory = V7.make_layer_cfgs_v7
    except Exception as _e:  # pragma: no cover - import failure is an environment defect
        out['frozen_manifest'] = {'status': 'not_verifiable_locally',
                                   'note': f'cannot import the v7 arm factory ({type(_e).__name__}: {_e})'}
        _factory = None
    if _factory is not None:
        _v7_rep = L.sweep_contrast_factor_audit(
            V7.ROPE_CONTRAST_FACTORS, factory=_factory, verbose=False)
        out['v7_contrasts'] = _v7_rep
        if _v7_rep.get('status') == 'fail':
            out['status'] = 'fail'
        _niah_rep = L.niah_contrast_factor_audit(
            factory=_factory, verbose=False)
        out['niah_contrasts'] = _niah_rep
        if _niah_rep.get('status') == 'fail':
            out['status'] = 'fail'
    _mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    if _factory is not None:
        if not os.path.exists(_mp):
            out['frozen_manifest'] = {'status': 'not_verifiable_locally',
                                       'note': f'no freeze manifest at {_mp}'}
        else:
            try:
                with open(_mp, encoding='utf-8') as _f:
                    man = json.load(_f)
            except Exception as _e:
                out['frozen_manifest'] = {'status': 'fail',
                                          'note': f'manifest is not readable JSON ({type(_e).__name__}: {_e})'}
                man = None
            if isinstance(man, dict):
                _arms = [str(a) for a in (man.get('arms') or [])]
                _cmp = {}
                _bad = []
                _comps = man.get('primary_comparisons')
                if not isinstance(_comps, list) or not _comps:
                    # An empty/malformed family audits zero comparisons; an
                    # 'ok' verdict would claim the gate passed when nothing was
                    # checked.  Fail closed.
                    out['frozen_manifest'] = {
                        'status': 'fail', 'manifest': _mp,
                        'note': ('primary_comparisons must be a non-empty list, '
                                 f'got {type(_comps).__name__}'
                                 if not isinstance(_comps, list) else
                                 'primary_comparisons is empty; no declared '
                                 'comparison was audited')}
                    out['status'] = 'fail'
                    comps_for_audit = []
                else:
                    comps_for_audit = _comps
                for comp in comps_for_audit:
                    if not isinstance(comp, dict):
                        _bad.append(f'non-object primary comparison entry {comp!r}')
                        continue
                    _name = str(comp.get('name'))
                    _a, _b = str(comp.get('a')), str(comp.get('b'))
                    _keys, _per_layer = L._design_pair_cfg_keys(_a, _b, _factory, 6)
                    if isinstance(_keys, Exception):
                        _bad.append(f'{_name}: arm construction failed ({_keys})')
                        _cmp[_name] = {'a': _a, 'b': _b, 'cfg_diff_keys': None,
                                       'cfg_single_factor': False,
                                       'mechanism_claim_allowed': comp.get('mechanism_claim_allowed'),
                                       'claim_flag_cfg_consistent': False}
                        continue
                    _sf_cfg = L.cfg_keys_single_factor(_keys)
                    _flag = comp.get('mechanism_claim_allowed')
                    _consistent = (_flag is not True) or _sf_cfg
                    if not _consistent:
                        _bad.append(f'{_name}: mechanism_claim_allowed=true but the cfg diff '
                                    f'keys {sorted(_keys)} are not one declared single-factor family '
                                    f'(kind swaps are architecture-level by definition)')
                    _cmp[_name] = {'a': _a, 'b': _b, 'cfg_diff_keys': sorted(_keys),
                                   'cfg_single_factor': _sf_cfg,
                                   'mechanism_claim_allowed': _flag,
                                   'claim_flag_cfg_consistent': _consistent,
                                   'note': ('full verdict (objectives + parameter totals) is '
                                            'enforced by `python -m explib.confirmation plan` at preflight')}
                if comps_for_audit or out['frozen_manifest'].get('status') != 'fail':
                    out['frozen_manifest'] = {'status': ('fail' if _bad else 'ok'),
                                              'manifest': _mp, 'comparisons': _cmp,
                                              'violations': _bad,
                                              'arms_checked': _arms}
                if _bad:
                    out['status'] = 'fail'
            else:
                # A readable but non-object manifest (or an unreadable one,
                # handled above) cannot be audited.  Leaving frozen_manifest
                # empty previously let the overall status stay 'ok', so a
                # malformed gate artifact silently passed.
                out['frozen_manifest'] = {
                    'status': 'fail', 'manifest': _mp,
                    'note': ('manifest must be a JSON object, got '
                             f'{type(man).__name__}' if man is not None
                             else 'manifest is not readable JSON')}
                out['status'] = 'fail'
            if (isinstance(out.get('frozen_manifest'), dict)
                    and out['frozen_manifest'].get('status') == 'fail'):
                out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e-2) factorial：2x2 交互设计的定义点审计（零 GPU）
# ---------------------------------------------------------------------------

def audit_content_controls():
    """Ablation control-ladder audit (zero GPU, design point).

    Checks that the content ablation has BOTH a presence control (zero
    content) and a matched alignment placebo (the same pooled table shifted
    one block back), that both are scheduled, that each control differs from
    its base in exactly ``content_mode``, and that the two leave-one-out
    families are declared.  A missing/unconstructed control is a failure, not
    a pass: "content matters" cannot be read from the zero-content contrast
    alone.
    """
    L = _import_exp_lib()
    out = L.content_control_audit()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_factorial_design(manifest_path=None, *, n_layers=6):
    """2x2 因子交互设计审计（零 GPU、零落盘依赖）。

    复用 `exp_lib.factorial_design_audit`，在定义点核验三组交互：
      * 四个 cell 完整、臂唯一且已排程；
      * 每个因子在另一因子的两个水平上做**同一组** AttnCfg 改动
        （并行操纵；否则交互比较的是两种不同干预）；
      * 改动键属于该组件的声明干预集，且不交换 `kind`；
      * 每组交互都有完整、指标对齐的预注册假设。

    这是 `single_factor` 的补集：单因素审计回答「每个臂只动了一个组件吗」，
    factorial 审计回答「两个组件的 2x2 是否可识别、交互是否可解释」。

    返回还包含：
      * `interaction_coverage`：声明范围内哪些组件对已有 2x2、哪些缺失；
      * `component_anchor`：每个 factorial 因子是否锚定到已声明的留一法对照。
    未测交互只记为设计缺口，不会把已有 2x2 的结论作废；但也不能据此把
    留一法效应相加后声称模块可加。"""
    L = _import_exp_lib()
    out = L.factorial_design_audit(n_layers=n_layers)
    if manifest_path:
        try:
            with open(manifest_path, encoding='utf-8') as _f:
                _man = json.load(_f)
            out['manifest_arms'] = list(_man.get('arms') or [])
            _missing = sorted(set(
                a for rec in (out.get('factorials') or {}).values()
                for a in (rec.get('cells') or {}).values()
                if a not in set(out['manifest_arms'])))
            out['manifest_missing_arms'] = _missing
            if _missing:
                out['status'] = 'fail'
                out['violations'].append(
                    'factorial arm(s) absent from the frozen manifest: '
                    + ', '.join(_missing))
        except Exception as _e:
            out['status'] = 'fail'
            out['violations'].append(
                f'cannot read manifest {manifest_path}: '
                f'{type(_e).__name__}: {_e}')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e-2b) interaction_coverage：核心组件两两交互覆盖审计（零 GPU）
# ---------------------------------------------------------------------------

def audit_interaction_coverage(scope=None, waivers=None, *, verbose=True):
    """核心组件两两交互覆盖审计（零 GPU）。

    留一法完整不等于交互已刻画。该审计枚举 `INTERACTION_SCOPE` 中的组件对，
    区分 `tested` / `waived` / `missing`；缺失的交互只能报告为未测，不能
    写成可加，也不能把单独验证的留一法效应直接相加。`waivers` 是可选的
    `{(a, b): reason}` 显式排除，不是零交互证据。"""
    L = _import_exp_lib()
    out = L.interaction_coverage_audit(scope=scope, waivers=waivers)
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_factorial_anchor(design=None, factorials=None, *, verbose=True):
    """检查每个 2x2 因子的 cell 对照是否锚定到声明的留一法对照（零 GPU）。"""
    L = _import_exp_lib()
    out = L.ablation_factorial_anchor_audit(design=design, factorials=factorials)
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e-3) dose_response：有序剂量/机制梯度的定义点审计（零 GPU）
# ---------------------------------------------------------------------------

def audit_dose_response(manifest_path=None, design=None, *, direction=None,
                        dose_key=None, factory=None, verbose=True):
    """有序剂量设计的定义点审计（零 GPU）。

    从冻结清单的可选 ``dose_response_design`` 字段读取声明，或由调用方直接
    传入 ``design``（``{levels, direction, dose_key, ...}`` 或 levels 本身）。
    核验“至少三个水平、剂量排序、只改变一个声明键、同一种子集、同一测量
    协议、方向/形状预注册”；清单没有该字段时明确返回 ``not_applicable``，
    不把“没有声明”误报成通过。记录级趋势检验在
    ``exp_lib.dose_response_trend``，本入口只核验设计。
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    spec = design
    source = 'argument'
    if spec is None:
        man = _load_manifest_file(mp)
        if not isinstance(man, dict):
            out = {'status': 'not_verifiable_locally',
                   'manifest': mp,
                   'note': 'no readable frozen manifest; no ordered-dose '
                           'declaration can be audited'}
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        spec = man.get('dose_response_design') or man.get('dose_response')
        source = mp
        if spec is None:
            out = {'status': 'not_applicable',
                   'manifest': mp,
                   'note': ('the manifest declares no dose_response_design; '
                            'an ordered ladder is not used by this plan')}
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
    if isinstance(spec, dict) and isinstance(spec.get('levels'), (dict, list, tuple)):
        levels = spec.get('levels')
        direction = spec.get('direction') if direction is None else direction
        dose_key = spec.get('dose_key') if dose_key is None else dose_key
        kwargs = {
            'n_layers': spec.get('n_layers', 6),
            'min_levels': spec.get('min_levels', 3),
            'min_seeds': spec.get('min_seeds', 3),
            'require_seeds': spec.get('require_seeds', True),
            'require_protocol': spec.get('require_protocol', False),
            'shape': spec.get('shape', 'linear'),
        }
    else:
        levels = spec
        kwargs = {}
    try:
        rep = L.dose_response_design_audit(
            levels, direction=direction, dose_key=dose_key,
            factory=factory, **kwargs)
    except Exception as e:
        rep = {'kind': 'dose_response_design', 'status': 'fail',
               'violations': [f'{type(e).__name__}: {e}'],
               'note': 'dose-response design audit could not be completed'}
    rep['source'] = source
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


# ---------------------------------------------------------------------------
# 7f) design preflight：预算台账 / 证据等级 / 动态分块假设与对照
# ---------------------------------------------------------------------------

def _load_manifest_file(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def audit_design_preflight(manifest_path=None, dynamic_path=None):
    """零 GPU 设计前馈审计：在训练/test 访问前核对实验设计。

    这是本轮科研方法改进的统一入口，回答四个在跑 GPU 之前就必须回答的问题：

      1. **预算控制**：主清单与动态草稿里每条对照的 token 预算、参数总量、
         读取预算是否真的匹配？机制主张是否只挂在专用对照臂上，而不是只跟
         dense 基线比？（`exp_lib.comparison_budget_ledger`）
      2. **证据等级**：v6-v11（同一 validation split 上选择/比较）是否仍被
         标成 exploratory？draft 清单是否被当成确认性计划？
         （`exp_lib.evidence_level_audit`）
      3. **假设驱动**：动态分块是否把「切分质量 / 检索精度 / 长程覆盖」写成
         可证伪的独立假设，并安排了 random-vs-random、read-matched、参数
         代理、sink 匹配等对照？（`exp_lib.dynamic_design_audit`）
      4. **功效与冻结**：种子数/族大小是否至少数学上可达 Holm 显著，是否声明
         了 test 前功效门槛？（两个 manifest 的统计块）

    返回结构化报告；任一硬性设计缺陷 => `status=fail`。它不读任何
    results_*/ 数值，也不调用 test loader。
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    dp = dynamic_path or os.path.join(REPO, 'confirmation_frozen_dynamic.json')
    out = {'status': 'ok', 'frozen_manifest': mp, 'dynamic_manifest': dp,
           'frozen': {}, 'dynamic': {}, 'violations': [], 'warnings': []}
    main = _load_manifest_file(mp)
    if not isinstance(main, dict):
        out['violations'].append(f'frozen manifest is missing/unreadable: {mp}')
        out['frozen'] = {'status': 'not_verifiable_locally'}
    else:
        ledger = L.comparison_budget_ledger(main)
        ev = L.evidence_level_audit(main, manifest_path=mp,
                                    panel_paths=list(L.EXPLORATORY_PANEL_REGISTRY))
        out['frozen'] = {
            'status': ('fail' if (ledger.get('status') == 'fail'
                                  or ev.get('status') == 'fail') else 'ok'),
            'evidence_level': ev.get('manifest_level'),
            'token_budget': ledger.get('token_budget'),
            'design_cost': ledger.get('design_cost'),
            'comparisons': ledger.get('comparisons'),
            'dedicated_controls': ledger.get('dedicated_controls'),
            'power_plan': ev.get('power_plan'),
            'ledger_violations': ledger.get('violations'),
            'evidence_violations': ev.get('violations'),
            'evidence_warnings': ev.get('warnings'),
        }
        out['violations'].extend(f'frozen: {v}' for v in
                                 (ledger.get('violations') or []) +
                                 (ev.get('violations') or []))
        out['warnings'].extend(f'frozen: {w}' for w in
                               (ledger.get('warnings') or []) +
                               (ev.get('warnings') or []))
        if out['frozen']['status'] == 'fail':
            out['status'] = 'fail'
    dyn = _load_manifest_file(dp)
    if not isinstance(dyn, dict):
        out['warnings'].append(
            f'no readable dynamic draft at {dp}; the dynamic/HCA study has no '
            'design record yet')
        out['dynamic'] = {'status': 'not_verifiable_locally'}
    else:
        dd = L.dynamic_design_audit(dyn, manifest_path=dp)
        dl = L.comparison_budget_ledger(dyn)
        de = L.evidence_level_audit(dyn, manifest_path=dp, panel_paths=[])
        out['dynamic'] = {
            'status': ('fail' if (dd.get('status') == 'fail'
                                  or dl.get('status') == 'fail'
                                  or de.get('status') == 'fail') else 'ok'),
            'declared_status': dyn.get('status'),
            'evidence_level': de.get('manifest_level'),
            'runnable': dd.get('runnable'),
            'promotion_ready': dd.get('promotion_ready'),
            'hypotheses': sorted(dd.get('hypotheses') or {}),
            'power': dd.get('power'),
            'calibration': dd.get('calibration'),
            'design_cost': dl.get('design_cost'),
            'dedicated_controls': dl.get('dedicated_controls'),
            'comparisons': dl.get('comparisons'),
            'design_violations': dd.get('violations'),
            'ledger_violations': dl.get('violations'),
            'evidence_violations': de.get('violations'),
            'warnings': (dd.get('warnings') or []) + (dl.get('warnings') or [])
                        + (de.get('warnings') or []),
        }
        out['violations'].extend(f'dynamic: {v}' for v in
                                 (dd.get('violations') or []) +
                                 (dl.get('violations') or []) +
                                 (de.get('violations') or []))
        out['warnings'].extend(f'dynamic: {w}' for w in out['dynamic']['warnings'])
        if out['dynamic']['status'] == 'fail':
            out['status'] = 'fail'
        # A draft being non-runnable is expected, not a violation; make the
        # distinction explicit so a green design audit is not mistaken for a
        # green confirmation.
        if str(dyn.get('status') or '').lower() == 'draft':
            out['warnings'].append(
                'dynamic draft is intentionally non-runnable: `promotion_ready` '
                'must become true (disjoint source + measured read-budget '
                'calibration + passed power gate) before any dynamic confirmation')
    # A recorded violation is a hard design defect, including the case where
    # the frozen manifest is missing/unreadable.  Previously that branch only
    # appended to `violations` while `status` stayed 'ok', so the CLI exited 0
    # on a checkout with no frozen plan.
    if out['violations']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_design_sensitivity(manifest_path=None,
                            pilot_path='analysis_v12/power_validation.json'):
    """Zero-GPU design sensitivity: can the planned n resolve the SESOI?

    The frozen design preflight verifies that the declared q has enough power.
    It cannot say whether that q corresponds to an effect the study would
    care about.  This audit reads the same manifest and, when available, the
    validation power artifact written by ``python -m explib.confirmation power``.  For every
    primary comparison it reports the exact Holm floor, the sign-flip power
    curve, the minimum q needed for target power, the approximate MDE, the
    expected paired 95% CI half-width, and (if the manifest declares one) the
    power at the pre-registered SESOI.

    Missing pilot data or a missing SESOI is reported as a warning, not a
    fabricated pass: the audit never invents an effect size.  A declared SESOI
    that the planned n cannot resolve fails only when the manifest sets
    ``require_sesoi_power``; otherwise the underpowered margin is named and
    kept visible.  Each directional cell also carries
    ``attrition_power_audit``: with ``require_complete_coverage`` a missing
    pair blocks confirmation, so the conditional power is multiplied by the
    probability that the full panel actually completes.  A declared
    ``statistics.attrition.pair_retention`` gates that block only when
    ``require_attrition_power`` is true; otherwise it is a named warning and
    stress table.  This reads no test split and changes no frozen decision.
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    pp = pilot_path
    if pp and not os.path.isabs(str(pp)):
        pp = os.path.join(REPO, str(pp))
    out = {'kind': 'design_sensitivity', 'status': 'ok', 'manifest': mp,
           'pilot': pp, 'pilot_status': 'not_available', 'n_seeds': None,
           'm_family': None, 'alpha': None, 'target_power': None,
           'comparisons': {}, 'violations': [], 'warnings': []}
    manifest = _load_manifest_file(mp)
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['violations'].append(
            f'frozen manifest is missing/unreadable: {mp}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    stat = manifest.get('statistics') if isinstance(
        manifest.get('statistics'), dict) else {}
    _attr_cfg = (stat.get('attrition')
                 if isinstance(stat.get('attrition'), dict) else {})
    _attr_require_raw = _attr_cfg.get('require_attrition_power', False)
    if isinstance(_attr_require_raw, (bool, np.bool_)):
        _attr_require = bool(_attr_require_raw)
    else:
        _attr_require = False
        out['violations'].append(
            'statistics.attrition.require_attrition_power='
            f'{_attr_require_raw!r} must be a boolean')
    _req_cov_raw = stat.get('require_complete_coverage', False)
    if isinstance(_req_cov_raw, (bool, np.bool_)):
        _req_cov = bool(_req_cov_raw)
    else:
        _req_cov = False
        out['violations'].append(
            'statistics.require_complete_coverage='
            f'{_req_cov_raw!r} must be a boolean')
    out['attrition'] = {
        'declared': bool(_attr_cfg),
        'pair_retention': _attr_cfg.get('pair_retention'),
        'require_attrition_power': _attr_require,
        'availability_target': _attr_cfg.get('availability_target'),
        'require_complete_coverage': _req_cov,
        'note': ('pair_retention is a planning stress test, not an observed '
                 'attrition rate; missing the declaration is a warning unless '
                 'require_attrition_power is true'),
    }
    if _attr_cfg.get('pair_retention') is None:
        out['warnings'].append(
            'the frozen manifest declares no statistics.attrition.pair_retention; '
            'the attrition table below is a stress test, not a pass')
    seeds = manifest.get('seeds')
    comps = manifest.get('primary_comparisons')
    n = len(seeds) if isinstance(seeds, list) else None
    m = len(comps) if isinstance(comps, list) else None
    out['n_seeds'], out['m_family'] = n, m
    out['alpha'] = stat.get('alpha', 0.05)
    out['target_power'] = stat.get('min_power_for_test', 0.80)
    if n is None or n < 2:
        out['violations'].append(
            f'manifest.seeds must be a list with at least two seeds, got {seeds!r}')
    elif isinstance(seeds, list):
        _seed_vals = []
        for _i, _s in enumerate(seeds):
            _sv = L._as_seed_int(_s)
            if _sv is None or _sv < 0:
                out['violations'].append(
                    f'manifest.seeds[{_i}]={_s!r} is not a non-negative integer seed')
            else:
                _seed_vals.append(_sv)
        if len(set(_seed_vals)) != len(_seed_vals):
            out['violations'].append(
                'manifest.seeds contains duplicates; the power floor would be '
                'computed from a seed count that the panel cannot realise')
        _n_stat = L._as_seed_int(stat.get('n_seeds'))
        if 'n_seeds' in stat and _n_stat is None:
            out['violations'].append(
                f'statistics.n_seeds={stat.get("n_seeds")!r} is not an integer')
        elif _n_stat is not None and _n_stat != n:
            out['violations'].append(
                f'statistics.n_seeds={_n_stat} does not equal len(seeds)={n}')
    if m is None or m < 1:
        out['violations'].append(
            f'manifest.primary_comparisons must be a non-empty list, got {comps!r}')
    if not isinstance(comps, list):
        comps = []
    pilot = _load_manifest_file(pp) if pp else None
    pilot_comps = (pilot.get('comparisons')
                   if isinstance(pilot, dict) and isinstance(
                       pilot.get('comparisons'), dict) else {})
    if pilot_comps:
        out['pilot_status'] = 'available'
    else:
        out['warnings'].append(
            f'no readable power-validation artifact at {pp}; pass a pilot SD '
            'or run `python -m explib.confirmation power` before relying on MDE/SESOI '
            'readouts')
    pilot_cells = 0
    _seen_names = set()
    for comp in comps:
        if not isinstance(comp, dict):
            out['violations'].append(
                f'primary comparison is not an object: {comp!r}')
            continue
        name = str(comp.get('name') or f'comparison_{len(out["comparisons"])}')
        if name in _seen_names:
            out['violations'].append(
                f'duplicate primary-comparison name {name!r}; the family size '
                'counts it twice but a name-keyed report can only hold one, so '
                'the Holm family and the per-comparison readout would disagree')
            continue
        _seen_names.add(name)
        pcell = pilot_comps.get(name) if isinstance(pilot_comps, dict) else None
        deltas = pcell.get('deltas') if isinstance(pcell, dict) else None
        if not isinstance(deltas, list):
            deltas = None
        else:
            pilot_cells += 1
        sesoi = comp.get('sesoi')
        if sesoi is None:
            sesoi = stat.get('sesoi')
        require_sesoi = (comp['require_sesoi_power']
                         if 'require_sesoi_power' in comp
                         else stat.get('require_sesoi_power', False))
        rep = L.paired_design_sensitivity(
            n, m or 0, alpha=out['alpha'],
            target_power=out['target_power'], pilot_deltas=deltas,
            expected_sign=comp.get('expected_sign'), sesoi=sesoi,
            require_sesoi_power=require_sesoi)
        _dir = rep.get('expected_sign')
        _q_attr = (rep.get('pilot') or {}).get('q_sign_rate')
        _q_source = 'pilot_sign_rate'
        if _q_attr is None:
            _q_attr = stat.get('power_q_for_test')
            _q_source = 'statistics.power_q_for_test'
        if _dir in (-1, 1) and _q_attr is not None:
            rep['attrition'] = L.attrition_power_audit(
                n, m or 0, q=_q_attr,
                pair_retention=_attr_cfg.get('pair_retention'),
                alpha=out['alpha'], target_power=out['target_power'],
                require_complete_coverage=stat.get(
                    'require_complete_coverage', False),
                min_complete_pairs=stat.get('min_seeds_for_confirmation'),
                availability_target=_attr_cfg.get('availability_target'),
                require_attrition_power=_attr_cfg.get(
                    'require_attrition_power', False))
            rep['attrition']['q_source'] = _q_source
            out['violations'].extend(
                f'{name}: {v}'
                for v in (rep['attrition'].get('violations') or []))
            out['warnings'].extend(
                f'{name}: {w}'
                for w in (rep['attrition'].get('warnings') or []))
            if rep['attrition'].get('status') == 'fail':
                out['status'] = 'fail'
        else:
            rep['attrition'] = {
                'status': 'not_applicable',
                'reason': ('a directional q is required for a sign-flip '
                           'attrition projection'),
            }
        out['comparisons'][name] = rep
        out['violations'].extend(
            f'{name}: {v}' for v in (rep.get('violations') or []))
        out['warnings'].extend(
            f'{name}: {w}' for w in (rep.get('warnings') or []))
        if rep.get('status') == 'fail':
            out['status'] = 'fail'
    if out['pilot_status'] == 'available' and pilot_cells == 0:
        out['pilot_status'] = 'no_matching_comparisons'
        out['warnings'].append(
            'the power artifact has comparisons but none carry a deltas list '
            'matching a frozen primary comparison name')
    if (out['pilot_status'] != 'available' or pilot_cells == 0) and \
            out['status'] == 'ok':
        out['status'] = 'not_verifiable'
    if out['violations']:
        out['status'] = 'fail'
    if (not any(isinstance(c, dict) and c.get('sesoi') for c in comps)
            and not stat.get('sesoi')):
        out['warnings'].append(
            'the frozen manifest declares no SESOI for its primary '
            'comparisons; statistical detectability is not practical relevance')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_evaluation_design(manifest_path=None,
                           pilot_path='analysis_v12/power_validation.json'):
    """Zero-GPU audit of the test-time evaluation-unit design.

    The sensitivity audit plans the seed panel.  This audit plans the other
    sampling axis: the frozen number of independent evaluation units
    (documents/articles) and whether that count can support a population
    claim.  When ``python -m explib.confirmation power`` has written per-unit deltas for each
    comparison, it also reports the approximate unit-level MDE, CI half-width
    and SESOI power.

    A missing unit pilot is ``not_verifiable`` with a warning, not a pass.
    The frozen prefix slice is reported as such; population inference is
    refused below the repository's independent-unit floor.  No test data is
    read and no frozen decision is changed.
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    pp = pilot_path
    if pp and not os.path.isabs(str(pp)):
        pp = os.path.join(REPO, str(pp))
    manifest = _load_manifest_file(mp)
    pilot = _load_manifest_file(pp) if pp else None
    out = L.evaluation_design_audit(manifest, pilot)
    out['manifest'] = mp
    out['pilot'] = pp
    out['pilot_status'] = 'available' if isinstance(pilot, dict) else 'not_available'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 8) 规模核算：6ND 估算 + TimeGuard 记录对账
# ---------------------------------------------------------------------------

def train_flops_estimate(params, tokens, attn_overhead=0.35):
    """6ND 估算 + 注意力项系数。

    6·N·D 是前向+反向的常用近似；稀疏选择/池化/激活重算与长序列注意力
    不与参数量成比例，因此用一个**显式声明的**注意力项系数。"""
    def _positive(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}')
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}') from None
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}')
        return f
    n = _positive(params, 'params')
    d_tok = _positive(tokens, 'tokens')
    if isinstance(attn_overhead, (bool, np.bool_)):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}')
    try:
        overhead = float(attn_overhead)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}') from None
    if not (math.isfinite(overhead) and overhead >= 0.0):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}')
    base = 6.0 * n * d_tok
    return {'flops_6nd': base, 'flops_with_attn_proxy': base * (1.0 + overhead), 'attn_overhead_assumed': overhead, 'note': 'analytic PROXY, not a measurement; report alongside TimeGuard seconds'}


def audit_budget(state_files=('run_time_state_v10.json', 'run_time_state_v12.json'), panels=(('results_lm_v10_scale_s', 'csa_fixed', 8000), ('results_lm_v10_scale_l8k', 'csa_fixed', 8000), ('results_lm_v10_scale_lflop', 'csa_fixed', None))):
    """规模核算（零 GPU）：每面板 6ND 估算 + 已记账实测秒数。"""
    out = {'status': 'ok', 'panels': {}}
    for d, v, steps in panels:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if summ is None:
            out['panels'][d] = 'not present locally'
            continue
        def _positive_int(value):
            if value is None or isinstance(value, (bool, np.bool_)):
                return None
            if isinstance(value, numbers.Integral):
                return int(value) if int(value) > 0 else None
            if isinstance(value, numbers.Real):
                f = float(value)
                return int(f) if math.isfinite(f) and f.is_integer() and f > 0 else None
            return None

        def _positive_real(value):
            if value is None or isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                return None
            f = float(value)
            return f if math.isfinite(f) and f > 0.0 else None

        recs = []
        for r in summ.values():
            if not isinstance(r, dict) or r.get('variant') != v:
                continue
            _p = _positive_real(r.get('params'))
            if _p is not None:
                recs.append(r)
        if not recs:
            out['panels'][d] = 'no param records'
            continue
        r0 = recs[0]
        _steps = _positive_int(steps) if steps is not None else _positive_int(r0.get('steps'))
        tok = _positive_int(r0.get('tokens_seen'))
        if tok is None and _steps is not None:
            _batch = _positive_int(r0.get('batch_size', 12))
            _seq = _positive_int(r0.get('seq_len', 512))
            if _batch is not None and _seq is not None:
                tok = _steps * _batch * _seq
        est = (train_flops_estimate(_positive_real(r0.get('params')), tok)
               if tok is not None else None)
        out['panels'][d] = {'params': r0.get('params'), 'steps': _steps,
                            'tokens': tok, 'estimate': est}
    for sf in state_files:
        st = _load(_audit_artifact(sf))
        if st is not None:
            # TimeGuard stores `runs` as an integer counter; older/foreign
            # state files may store a list.  len(integer) would crash the whole
            # audit on any machine that has actually run a cell.
            _runs = st.get('runs')
            if isinstance(_runs, (list, tuple, dict, set)):
                _run_count = len(_runs)
            elif isinstance(_runs, (int, float)) and not isinstance(_runs, bool):
                _run_count = int(_runs)
            else:
                _run_count = st.get('run_count')
            out.setdefault('measured', {})[sf] = {'booked_seconds': st.get('booked_seconds'), 'run_count': _run_count}
    if all(not isinstance(x, dict) for x in out['panels'].values()):
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'scale panels not present locally; estimates + TimeGuard bookkeeping run on the experiment machine'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 9) 测量工具：prefill/decode 与容量读数（含 CPU 微型自检）
# ---------------------------------------------------------------------------

def measure_prefill_decode(model, ids, device, warmup=1, iters=3):
    """端到端 prefill / 前缀推进计时（测量代码，非实验）。

    prefill = 整段前向一次的墙钟；所谓 decode 当前每次从 token 0 重跑完整
    前缀，因此它是 prefix-prefill 均摊，不是有 KV cache 的增量 decode。
    实现在 exp_lib.measure_efficiency（p50/p95、tokens/s 等读数）。"""
    try:
        L = _import_exp_lib()
        return L.measure_efficiency(model, ids, device, warmup=warmup, iters=iters)
    except Exception as _e:
        print(f'[audit] NOTE: exp_lib.measure_efficiency unavailable ({type(_e).__name__}: {_e}); using the local fallback')
    import torch
    import time as _t
    res = {'n_tokens': int(ids.shape[1])}
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(iters):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        res['prefill_ms_per_iter'] = (_t.perf_counter() - t0) * 1000.0 / iters
        t0 = _t.perf_counter()
        n_dec = min(32, int(ids.shape[1]))
        for t in range(1, n_dec + 1):
            model(ids[:, :t])
        _per_tok = (_t.perf_counter() - t0) * 1000.0 / n_dec
        res['prefix_prefill_ms_per_token'] = _per_tok
        # Backwards-compatible alias, but the name must not be read as an
        # incremental decode measurement: there is no KV cache in this path.
        res['decode_ms_per_token_avg'] = _per_tok
        res['decode_has_kv_cache'] = False
        res['decode_metric'] = 'prefix_prefill (no KV cache; re-runs the whole prefix)'
        if device.type == 'cuda':
            res['peak_mem_mb'] = torch.cuda.max_memory_allocated() / (1 << 20)
    return res


def audit_training_health(outdirs=None):
    """Zero-GPU audit of recorded training-curve health and NaN triage.

    A run that overfit, stalled, or recorded a non-finite event may still have a
    PPL number.  This audit makes that diagnostic history explicit and refuses
    a record whose own triage contradicts its events.  It does not select
    models and does not change any panel.
    """
    dirs = outdirs or list(_iter_result_dirs())
    out = {'status': 'ok', 'panels': {}, 'warnings': [], 'errors': [],
           'note': 'diagnostic only: a healthy record is not thereby a valid treatment claim, and an unhealthy record is not silently deleted'}
    _Lmod = None
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not isinstance(summ, dict) or not summ:
            continue
        cells = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            w = []
            e = []
            h = r.get('curve_health')
            status = str(h.get('status') or 'unknown') if isinstance(h, dict) else 'missing'
            _known_health = {'healthy', 'overfitting', 'underfitting_or_stalled',
                             'validation_degrading', 'validation_improving_train_flat',
                             'nonfinite', 'insufficient_data', 'unmonitored'}
            if not isinstance(h, dict):
                w.append('no curve_health record')
            elif status in ('overfitting', 'underfitting_or_stalled',
                            'validation_degrading', 'validation_improving_train_flat',
                            'nonfinite'):
                w.append(f'curve_health={status}')
            elif status not in _known_health:
                w.append(f'unknown curve_health status {status!r}; '
                         'the run is not verified healthy')
            if (isinstance(h, dict) and status == 'healthy'
                    and r.get('health_history') == []):
                w.append('curve_health=healthy with an EMPTY health_history; '
                         'the claim cannot be recomputed from the curve')
            _diag_raw = r.get('opt_diag_summary')
            if _diag_raw is not None and not isinstance(_diag_raw, dict):
                e.append(f'opt_diag_summary is not an object ({type(_diag_raw).__name__})')
            diag = _diag_raw if isinstance(_diag_raw, dict) else {}
            _guard_raw = diag.get('nan_guard')
            if _guard_raw is not None and not isinstance(_guard_raw, dict):
                e.append(f'nan_guard is not an object ({type(_guard_raw).__name__})')
            guard = _guard_raw if isinstance(_guard_raw, dict) else {}
            def _nonneg(v):
                if (isinstance(v, (bool, np.bool_))
                        or not isinstance(v, numbers.Real)):
                    return None
                f = float(v)
                if not (math.isfinite(f) and f >= 0.0 and f == math.floor(f)):
                    return None
                return int(f)
            def _count_field(source, key):
                # Missing means zero events; a PRESENT but malformed count is
                # unknown evidence and must fail closed rather than silently
                # becoming zero (which would let a corrupt record look clean).
                raw = source.get(key) if isinstance(source, dict) else None
                if raw is None:
                    return 0
                value = _nonneg(raw)
                if value is None:
                    e.append(f'{key} is not a non-negative integer ({raw!r}); '
                             'the non-finite event count is unverifiable, not zero')
                    return 0
                return value
            n_loss = _count_field(diag, 'nonfinite_loss_steps')
            n_grad = _count_field(diag, 'nonfinite_grad_steps')
            n_guard = _count_field(guard, 'elements_cleaned')
            has_events = bool(n_loss or n_grad or n_guard)
            # A recorded data audit is the first branch of the NaN triage.
            # Only validate it when present: legacy records without one stay
            # on their existing warning path (missing evidence is not evidence
            # of corruption), but a present-and-failed audit must fail closed.
            _data_audit = r.get('data_audit')
            if _data_audit is not None:
                if not isinstance(_data_audit, dict):
                    e.append(f'data_audit is not an object ({type(_data_audit).__name__})')
                else:
                    if str(_data_audit.get('status') or '') != 'ok':
                        e.append(f'data_audit status={_data_audit.get("status")!r}; '
                                 'the data contract did not pass')
                    _secs = _data_audit.get('sections') if isinstance(_data_audit.get('sections'), dict) else {}
                    for _name in ('train', 'validation'):
                        _sec = _secs.get(_name) if isinstance(_secs.get(_name), dict) else {}
                        if not _sec:
                            continue
                        _nf = _count_field(_sec, 'n_nonfinite')
                        _oor = _count_field(_sec, 'n_out_of_range')
                        if _nf or _oor:
                            e.append(f'data_audit.{_name}: {_nf} non-finite and '
                                     f'{_oor} out-of-range token value(s)')
                    _ls = _data_audit.get('label_shift')
                    if isinstance(_ls, dict) and _ls.get('checked') and not _ls.get('ok'):
                        e.append('data_audit.label_shift failed: inputs/targets are '
                                 'not a one-token causal shift')
            # Event counters and their event log must not contradict each
            # other: an event log with loss/grad entries while the matching
            # counter is zero is a partially written record, not a clean run.
            _ev_raw = diag.get('events')
            if _ev_raw is not None and not isinstance(_ev_raw, list):
                e.append(f'opt_diag_summary.events is not a list ({type(_ev_raw).__name__})')
                _evs = []
            else:
                _evs = _ev_raw if isinstance(_ev_raw, list) else []
            _ev_loss = sum(1 for _x in _evs if isinstance(_x, dict) and _x.get('kind') == 'loss')
            _ev_grad = sum(1 for _x in _evs if isinstance(_x, dict) and _x.get('kind') == 'grad')
            if _ev_loss > n_loss:
                e.append(f'events contain {_ev_loss} non-finite loss step(s) but '
                         f'nonfinite_loss_steps={n_loss}')
            if _ev_grad > n_grad:
                e.append(f'events contain {_ev_grad} non-finite gradient step(s) but '
                         f'nonfinite_grad_steps={n_grad}')
            _fbs = diag.get('first_bad_step')
            if _fbs is not None:
                _fbsv = _nonneg(_fbs)
                if _fbsv is None or _fbsv < 1:
                    e.append(f'first_bad_step is not a positive integer ({_fbs!r})')
                elif not has_events and not _evs:
                    w.append(f'first_bad_step={_fbsv} is recorded but every event '
                             'counter/log is empty; the record is inconsistent')
            # The stored `curve_health` is a cache; the recorded curve is the
            # evidence.  Recompute it when present so a record cannot claim
            # `healthy` while its own history is overfitting/non-finite.
            _hist = r.get('health_history')
            if _hist is not None and not isinstance(_hist, list):
                e.append(f'health_history is not a list ({type(_hist).__name__})')
            elif isinstance(_hist, list) and _hist:
                if _Lmod is None:
                    try:
                        _Lmod = _import_exp_lib()
                    except Exception as _le:
                        # No classifier in this environment: the curve health
                        # stays unverified rather than being trusted blindly.
                        _Lmod = False
                        w.append(f'health_history present but the classifier is '
                                 f'unavailable ({type(_le).__name__}: {_le}); '
                                 'the recorded curve_health is unverified')
                if _Lmod is False:
                    pass
                else:
                    try:
                        _rh = _Lmod.classify_training_health(_hist)
                    except Exception as _he:
                        e.append(f'health_history could not be classified '
                                 f'({type(_he).__name__}: {_he}); the curve health is unverifiable')
                        _rh = None
                if _Lmod is not False and _rh is not None:
                    _rstat = str(_rh.get('status') or 'unknown')
                    if isinstance(h, dict):
                        if status != _rstat:
                            _msg = (f'recorded curve_health={status} contradicts the '
                                    f'health_history recomputation ({_rstat}: '
                                    f'{_rh.get("reason")})')
                            # A missing non-finite curve is a masked numerical
                            # failure; a trend-label difference is a (possibly
                            # version-dependent) diagnostic mismatch.
                            if status in ('healthy', 'unknown') and _rstat == 'nonfinite':
                                e.append(_msg)
                            else:
                                w.append(_msg)
                        _bs, _bp = h.get('best_step'), h.get('best_val_ppl')
                        if _bs is not None:
                            _bsv = _nonneg(_bs)
                            if _bsv is None:
                                w.append(f'curve_health.best_step is not a non-negative '
                                         f'integer ({_bs!r}); cannot verify it against the curve')
                            elif _rh.get('best_step') is not None and _bsv != int(_rh['best_step']):
                                w.append(f'curve_health.best_step={_bsv} does not match the '
                                         f'curve best step {int(_rh["best_step"])}')
                        if _bp is not None:
                            # PPL is a positive real, not an integer count.
                            _bpv = None
                            if (not isinstance(_bp, (bool, np.bool_))
                                    and isinstance(_bp, numbers.Real)):
                                _f = float(_bp)
                                _bpv = _f if (math.isfinite(_f) and _f > 0.0) else None
                            _rbp = _rh.get('best_val_ppl')
                            if _bpv is None:
                                w.append(f'curve_health.best_val_ppl is not a finite '
                                         f'positive number ({_bp!r})')
                            elif (isinstance(_rbp, (int, float)) and not isinstance(_rbp, bool)
                                  and math.isfinite(float(_rbp))
                                  and abs(_bpv - float(_rbp)) > 1e-6 * max(1.0, abs(float(_rbp)))):
                                w.append(f'curve_health.best_val_ppl={_bpv:g} does not match the '
                                         f'curve best {float(_rbp):g}')
                        _fm = h.get('final_minus_best')
                        if _fm is not None:
                            _fmv = None
                            if (not isinstance(_fm, (bool, np.bool_))
                                    and isinstance(_fm, numbers.Real)):
                                _f = float(_fm)
                                _fmv = _f if math.isfinite(_f) else None
                            _rfm = _rh.get('final_minus_best')
                            if _fmv is None:
                                w.append(f'curve_health.final_minus_best is not finite ({_fm!r})')
                            elif (isinstance(_rfm, (int, float)) and not isinstance(_rfm, bool)
                                  and math.isfinite(float(_rfm))
                                  and abs(_fmv - float(_rfm)) > 1e-6 * max(1.0, abs(float(_rfm)))):
                                w.append(f'curve_health.final_minus_best={_fmv:g} does not match '
                                         f'the curve {float(_rfm):g}')
                        _sb = h.get('steps_since_best')
                        if _sb is not None:
                            _sbv = _nonneg(_sb)
                            _rsb = _rh.get('steps_since_best')
                            if _sbv is None:
                                w.append(f'curve_health.steps_since_best is not a non-negative '
                                         f'integer ({_sb!r})')
                            elif _rsb is not None and _sbv != int(_rsb):
                                w.append(f'curve_health.steps_since_best={_sbv} does not match '
                                         f'the curve {int(_rsb)}')
                    else:
                        w.append(f'no curve_health record; health_history recomputes to '
                                 f'{_rstat} ({_rh.get("reason")})')
            triage = r.get('instability_diagnosis')
            if isinstance(triage, dict):
                tstat = str(triage.get('status') or 'unknown')
                if has_events and tstat == 'clean':
                    e.append('instability_diagnosis says clean but non-finite events are recorded')
                if status == 'nonfinite' and tstat == 'clean':
                    e.append('validation curve is non-finite but instability_diagnosis says clean')
                if tstat == 'unverifiable':
                    w.append('instability_diagnosis is unverifiable (malformed diagnostic record); '
                             'non-finite counts/health cannot be treated as clean')
            elif has_events:
                w.append('non-finite events without instability_diagnosis (legacy/unverified)')
            if status == 'nonfinite' and not isinstance(triage, dict):
                w.append('non-finite validation curve without triage (legacy/unverified)')
            es = r.get('early_stop')
            if isinstance(es, dict) and bool(es.get('stopped')):
                _budget = {}
                for _key in ('steps_completed', 'steps'):
                    _raw = r.get(_key)
                    if _raw is None:
                        continue
                    _val = _nonneg(_raw)
                    if _val is None:
                        e.append(f'{_key} is not a non-negative integer ({_raw!r}); '
                                 'the early-stop budget is unverifiable')
                    else:
                        _budget[_key] = _val
                if len(_budget) < 2:
                    w.append('early_stop.stopped without both steps_completed and '
                             'planned steps; the completed budget is unverifiable')
                elif _budget['steps_completed'] >= _budget['steps']:
                    w.append('early_stop.stopped is true but steps_completed >= planned steps')
            if w or e:
                cells[key] = {'health': status, 'warnings': w, 'errors': e}
        if cells:
            out['panels'][d] = cells
            for key, c in cells.items():
                out['errors'] += [f'{d}/{key}: {x}' for x in c['errors']]
                out['warnings'] += [f'{d}/{key}: {x}' for x in c['warnings']]
    if out['errors']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    return out


def audit_efficiency_demo():
    """CPU 微型自检：证明测量代码可运行（不是实验，秒级）。"""
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f'[audit efficiency] SKIPPED: torch not importable ({e})')
        return {'status': 'skipped_no_torch'}
    L = _import_exp_lib()
    import explib.rope as _V7  # noqa: F401 — installs the rope-variant patches
    import copy as _copy
    L.set_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 128, cfgs).to(dev)
    ids = torch.randint(0, 256, (1, 128), dtype=torch.long)
    res = measure_prefill_decode(model, ids, dev, warmup=0, iters=1)
    res['status'] = 'ok'
    res['note'] = 'CPU demo of the measurement harness; run measure_prefill_decode on the real checkpoints (per arm, per length) for the efficiency panel'
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ---------------------------------------------------------------------------
# report：汇总本地可执行的审计
# ---------------------------------------------------------------------------

def audit_efficiency_pairs(outdirs=None, pairs=None, *, tolerance=0.05):
    """Zero-GPU audit of MEASURED prefill time between arms.

    The efficiency question is not "is there a timing field" but "is a
    speed/cheapness claim supported".  This audit pairs records only when both
    arms have `efficiency.status == 'ok'` at the SAME token count, on the same
    device and torch version, with the same KV-cache semantics; it reports the
    p50 ratio and refuses the words "faster/cheaper" unless the ratio is
    outside `tolerance` in the claimed direction.  It never converts analytic
    FLOPs into a latency claim.

    `pairs` may be a list of (arm_a, arm_b); by default every arm pair sharing
    a panel is reported (descriptive)."""
    import numpy as np
    # The band controls whether a speed claim is allowed at all.  A boolean,
    # NaN, negative or >=1 tolerance would make the claimed direction
    # meaningless (negative even labels every ratio as faster), so refuse it
    # instead of silently changing the gate.
    if isinstance(tolerance, (bool, np.bool_)):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}')
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}') from None
    if not (math.isfinite(tolerance) and 0.0 <= tolerance < 1.0):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}')
    dirs = outdirs or list(_iter_result_dirs())
    out = {'status': 'ok', 'panels': {}, 'claims_allowed': [], 'refused': [],
           'tolerance': tolerance,
           'note': 'measured prefill only; no KV cache in this model; analytic FLOPs are never substituted for timing'}
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not summ:
            continue
        eff = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            v = r.get('variant')
            if v is None:
                v = str(key).split('::')[0]
            elif not isinstance(v, str) or not v:
                # A non-string arm id is not a usable comparison key.
                continue
            e = r.get('efficiency')
            if isinstance(e, dict) and e.get('status') == 'ok':
                eff.setdefault(v, []).append(e)
        if not eff:
            continue
        panel = {}
        _pairs = pairs if pairs else [(a, b) for i, a in enumerate(sorted(eff)) for b in sorted(eff)[i + 1:]]
        for a, b in _pairs:
            if a not in eff or b not in eff:
                continue
            def _num(v):
                # Timing/token fields are measurements: booleans and
                # non-finite values are not numbers, whatever float() accepts.
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                    return None
                f = float(v)
                return f if math.isfinite(f) else None

            def _med(rows, key):
                _v = [f for f in (_num(r.get(key)) for r in rows) if f is not None]
                return (float(np.median(_v)) if _v else None)
            # Environment/KV identity must hold for EVERY seed in both arms,
            # not only for the first record: a mixed arm would otherwise have
            # its median compared across different hardware/semantics.
            def _env_of(r):
                _dev, _tor = r.get('device'), r.get('torch')
                if (isinstance(_dev, str) and _dev.strip()
                        and isinstance(_tor, str) and _tor.strip()):
                    return (_dev, _tor)
                # Missing/blank provenance is not a match: two records that
                # both omit the device must not permit a speed claim.
                return None
            _env_a = {_env_of(r) for r in eff[a]}
            _env_b = {_env_of(r) for r in eff[b]}
            _kv_a = {r.get('decode_has_kv_cache') for r in eff[a]
                     if isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))}
            _kv_b = {r.get('decode_has_kv_cache') for r in eff[b]
                     if isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))}
            _kv_complete = (all(isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))
                                for r in eff[a] + eff[b]))
            same_env = bool(None not in _env_a and None not in _env_b
                            and len(_env_a) == 1 and len(_env_b) == 1
                            and _env_a == _env_b)
            same_kv = bool(_kv_complete and len(_kv_a) == 1 and len(_kv_b) == 1
                           and _kv_a == _kv_b)
            # New measurement records carry the batch size.  The same token
            # count at batch 1 vs batch 8 is a different execution shape; if
            # either arm recorded batch sizes, they must agree.  Legacy
            # records without the field stay comparable (but `batch_status`
            # says the shape was not recorded).
            def _batch_of(value):
                if isinstance(value, (bool, np.bool_)) or value is None:
                    return None
                if isinstance(value, numbers.Integral):
                    return int(value) if int(value) >= 1 else None
                if isinstance(value, numbers.Real):
                    f = float(value)
                    return int(f) if (math.isfinite(f) and f == math.floor(f)
                                      and f >= 1.0) else None
                return None
            _batch_a = {_batch_of(r.get('batch')) for r in eff[a]
                        if r.get('batch') is not None}
            _batch_b = {_batch_of(r.get('batch')) for r in eff[b]
                        if r.get('batch') is not None}
            # A partly recorded arm is not a legacy all-unrecorded arm: one
            # run with batch=8 and another without the field could be a
            # different execution shape, so the pair is unverifiable.
            _batch_a_complete = all(
                r.get('batch') is not None and _batch_of(r.get('batch')) is not None
                for r in eff[a])
            _batch_b_complete = all(
                r.get('batch') is not None and _batch_of(r.get('batch')) is not None
                for r in eff[b])
            if not (_batch_a or _batch_b):
                same_batch = True
                batch_status = 'unrecorded'
            elif (not _batch_a_complete or not _batch_b_complete
                  or None in _batch_a or None in _batch_b
                  or len(_batch_a) != 1 or len(_batch_b) != 1
                  or _batch_a != _batch_b):
                same_batch = False
                batch_status = 'different_or_incomplete'
            else:
                same_batch = True
                batch_status = 'same'
            # Median over seeds, not the first record: a single slow seed must
            # not decide a "faster" claim.
            _ta, _tb = _med(eff[a], 'prefill_ms_p50'), _med(eff[b], 'prefill_ms_p50')
            def _token_count(value):
                # Token counts are exact integers.  `int(1.5)` would turn a
                # fractional/malformed shape into a different row count and
                # let two runs be compared as if they used the same input.
                if isinstance(value, (bool, np.bool_)):
                    return None
                if isinstance(value, numbers.Integral):
                    return int(value) if int(value) > 0 else None
                if isinstance(value, numbers.Real):
                    f = float(value)
                    return (int(f) if (math.isfinite(f) and f == math.floor(f)
                                      and f > 0) else None)
                return None

            _na = sorted({_token_count(r.get('n_tokens')) for r in eff[a]},
                         key=lambda x: (x is None, -1 if x is None else x))
            _nb = sorted({_token_count(r.get('n_tokens')) for r in eff[b]},
                         key=lambda x: (x is None, -1 if x is None else x))
            rec = {'a': a, 'b': b, 'same_device_torch': same_env, 'same_kv_semantics': same_kv,
                   'same_batch': same_batch, 'batch_status': batch_status,
                   'a_batch': sorted(_batch_a, key=lambda x: (x is None, -1 if x is None else x)),
                   'b_batch': sorted(_batch_b, key=lambda x: (x is None, -1 if x is None else x)),
                   'a_ms_p50': _ta, 'b_ms_p50': _tb, 'a_tokens': _na, 'b_tokens': _nb,
                   'ratio_b_over_a': None, 'speed_claim_allowed': False, 'faster_arm': None, 'reason': None}
            if not (isinstance(_ta, (int, float)) and isinstance(_tb, (int, float)) and _ta > 0 and _tb > 0):
                rec['reason'] = 'missing or non-positive measured prefill time'
            elif (None in _na or None in _nb or _na != _nb
                  or len(_na) != 1 or len(_nb) != 1):
                rec['reason'] = (f'different, ambiguous or non-integer measured '
                                 f'token counts ({_na} vs {_nb}); a latency ratio '
                                 'across shapes is not a speed comparison')
            elif not same_env:
                rec['reason'] = 'different or inconsistent device/torch version recorded (within or across arms)'
            elif not same_kv:
                rec['reason'] = 'different or inconsistent KV-cache semantics recorded (within or across arms)'
            elif not same_batch:
                rec['reason'] = 'different, ambiguous or incomplete batch sizes recorded (within or across arms)'
            else:
                ratio = float(_tb) / float(_ta)
                rec['ratio_b_over_a'] = ratio
                rec['faster_arm'] = (b if ratio < 1.0 - tolerance else (a if ratio > 1.0 + tolerance else None))
                rec['speed_claim_allowed'] = rec['faster_arm'] is not None
                if not rec['speed_claim_allowed']:
                    rec['reason'] = f'ratio {ratio:.3f} is within +/-{tolerance:.0%} of 1: no measured speed difference'
            panel[f'{a} vs {b}'] = rec
            (out['claims_allowed'] if rec['speed_claim_allowed'] else out['refused']).append(f'{d}: {a} vs {b}')
        if panel:
            out['panels'][d] = panel
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('no records carry a measured `efficiency` block with status ok — the efficiency panel needs '
                       'checkpoints measured with exp_lib.measure_efficiency / measure_efficiency_sweep; analytic FLOPs do not qualify')
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'}, ensure_ascii=False, indent=2)[:4000])
    return out


def audit_release_readiness(path='RELEASES.md'):
    """Zero-GPU check that a numeric claim has an auditable evidence release.

    "Reproducible code" is not "auditable results".
    This audit parses the tracked release index and refuses to call the
    repository auditable unless at least one row names a commit, an archive
    URL, both SHA-256 digests and a verifier.  It also reports whether the
    CURRENT commit is covered.  It cannot verify a remote archive without
    downloading it; `archive_release.py verify` does that locally.
    """
    import re
    import subprocess
    out = {'status': 'no_release', 'rows': 0, 'complete_rows': 0,
           'current_commit': None, 'current_commit_released': False,
           'problems': [], 'path': path,
           'note': ('a repository with no evidence release is source-only: its generated reports are not '
                    'independently auditable. Publish release_<tag>.tar.gz plus a manifest and record it here.')}
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        out['current_commit'] = head or None
    except Exception:
        pass
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        out['problems'].append(f'{path} is missing')
        return out
    rows = []
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 7:
            continue
        if cells[0].lower() in ('tag', '---') or set(cells[0]) <= {'-', ':'}:
            continue
        if cells[0].startswith('_none'):
            continue
        rows.append(cells)
    out['rows'] = len(rows)
    for cells in rows:
        tag, _date, commit, url, asha, msha, verifier = cells[:7]
        missing = []
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}', commit or ''):
            missing.append('commit')
        if not re.match(r'https?://', url or ''):
            missing.append('archive URL')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', asha or ''):
            missing.append('archive SHA-256')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', msha or ''):
            missing.append('manifest SHA-256')
        if not verifier or verifier in ('—', '-', 'n/a', 'none'):
            missing.append('verifier')
        if missing:
            out['problems'].append(f'{tag}: missing {missing}')
            continue
        out['complete_rows'] += 1
        if out['current_commit'] and out['current_commit'].startswith(commit):
            out['current_commit_released'] = True
    if out['complete_rows'] == 0:
        out['status'] = 'no_release' if not rows else 'incomplete'
        out['note'] = ('no complete evidence release row: every numeric claim in any generated report remains '
                       'source-side only and is NOT independently auditable')
    else:
        out['status'] = 'ok'
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    return out
