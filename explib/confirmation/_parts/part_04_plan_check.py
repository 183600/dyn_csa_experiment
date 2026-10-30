

def plan_check(manifest_path=FREEZE_MANIFEST):
    """Zero-GPU pre-flight for the frozen confirmation plan.

    A stamp-only fix would turn P5T from "fails fast" into "burns the
    one-shot test access, then admits 0/48 records".  This check closes that
    hole WITHOUT reading test data: it derives the implementation descriptor
    from the same
    `make_layer_cfgs` + `_uses_batched_eager` code the trainer uses, builds
    one synthetic record per (arm, seed) with the real recipe/admission
    schema, and runs `_admit_confirmation_records` + the power gate.

    It fails if:
      * the manifest stamp does not match this code;
      * any frozen arm's predicted train_knobs violates the implementation
        binding (e.g. a false eager_batch certification);
      * a per-arm `mr` in the manifest disagrees with `variant_mlp_ratio`;
      * any primary comparison's two arms cannot be admitted;
      * the pre-registered minimum n cannot attain Holm significance.

    Returns a report dict.  Raises `SystemExit` with all errors on failure.
    """
    m = _load_manifest(manifest_path)
    errs = []
    notes = []
    impl = _frozen_implementation(m)
    if not impl:
        errs.append('manifest declares no implementation binding (eager_batch/amp_dtype); manifest v3 must bind the implementation')
    sig = _recipe_signature(m) or {}
    arms = list(m['arms'])
    seeds = [int(s) for s in m['seeds']]
    _stat = m.get('statistics') or {}
    _alpha = float(_stat.get('alpha', 0.05))
    _n_min = int(_stat.get('min_seeds_for_confirmation') or len(seeds))
    _eval = m.get('eval') or {}
    _lens = [int(x) for x in (_eval.get('by_length_ppl') or [])]
    _nseq = int(_eval.get('n_seqs_clean') or 0)
    if len(set(arms)) != len(arms):
        errs.append(f'frozen arms are not unique: {arms}')
    if not _lens:
        errs.append('manifest eval.by_length_ppl is empty; P5T has no endpoint grid')
    if _nseq <= 0:
        errs.append('manifest eval.n_seqs_clean is missing/zero; the frozen row count is part of the slice')
    if _n_min < 2 or _n_min > len(seeds):
        errs.append(f'statistics.min_seeds_for_confirmation={_n_min} is outside [2, {len(seeds)}]')
    _min_holm = L.holm_min_attainable_p(_n_min, len(m['primary_comparisons']), _alpha) if _n_min >= 2 else float('inf')
    if not math.isfinite(_min_holm) or _min_holm > _alpha:
        errs.append(f'the pre-registered minimum n={_n_min} cannot attain Holm significance (min attainable Holm p={_min_holm:.4g} > alpha={_alpha})')
    # Predict the implementation descriptor exactly as the trainer will.
    _dev = type('_PlanCuda', (), {'type': 'cuda'})()
    _bs = int(sig.get('bs', 12))
    _n_layers = int(sig.get('L', 6))
    _amp_norm = L._normalise_amp_mode(impl.get('amp_dtype')) if impl and impl.get('amp_dtype') else None
    _per_arm = ((m.get('binding') or {}).get('per_arm_recipe_signature') or {})
    if _per_arm and set(_per_arm) != set(arms):
        errs.append(f'binding.per_arm_recipe_signature must name exactly the frozen arms {sorted(arms)} (got {sorted(_per_arm)})')
    _records = {}
    _pred = {}
    _arm_cfgs = {}
    for arm in arms:
        try:
            cfgs = L.make_layer_cfgs(_n_layers, arm)
        except Exception as _e:
            errs.append(f'{arm}: make_layer_cfgs failed ({type(_e).__name__}: {_e})')
            continue
        try:
            _eb = L._uses_batched_eager(cfgs, device=_dev, batch_size=_bs)
        except Exception as _e:
            errs.append(f'{arm}: _uses_batched_eager failed ({type(_e).__name__}: {_e})')
            _eb = None
        _tk = L.training_knobs(amp_dtype=_amp_norm, eager_batch=bool(_eb))
        _why = _implementation_reason({'train_knobs': _tk}, impl)
        if _why:
            errs.append(f'{arm}: implementation binding cannot be satisfied by the real dispatch ({_why}; predicted train_knobs={_tk})')
        if impl and 'eager_batch' in impl and _eb is not None and bool(_eb) != bool(impl['eager_batch']):
            errs.append(f'{arm}: predicted eager_batch={bool(_eb)} != frozen {bool(impl["eager_batch"])!r}')
        _expected_mr = None
        if isinstance(_per_arm.get(arm), dict):
            _expected_mr = _per_arm[arm].get('mr')
        if _expected_mr is not None:
            try:
                _actual_mr = L.variant_mlp_ratio(arm, int(sig.get('v', 8192)), d=int(sig.get('d', 256)),
                                                 n_layers=_n_layers, n_heads=int(sig.get('H', 8)),
                                                 d_head=int(sig.get('Dh', 32)), seq_len=int(sig.get('sl', 512)),
                                                 matched={arm}, ref_variant='csa_fixed_rope') if arm in ('full_rope', 'full_sw128_matched_rope', 'full_rope_sink') else 4.0
            except Exception as _e:
                errs.append(f'{arm}: variant_mlp_ratio failed ({type(_e).__name__}: {_e})')
                _actual_mr = None
            if _actual_mr is not None and not math.isclose(float(_actual_mr), float(_expected_mr), rel_tol=1e-6, abs_tol=0.0):
                errs.append(f'{arm}: frozen mr={_expected_mr!r} != computed mr={_actual_mr!r}; the parameter-matching knob is not pinned to this code')
        else:
            _actual_mr = 4.0
            notes.append(f'{arm}: no per-arm mr frozen; the recipe signature will not catch a mr mutation')
        _recipe = _plan_recipe(m, arm, _expected_mr if _expected_mr is not None else _actual_mr)
        _sig_arm = _recipe_signature(m, arm)
        if _sig_arm is None:
            errs.append(f'{arm}: no recipe signature (shared or per-arm) is declared')
        elif not _recipe_matches_signature(_recipe, _sig_arm):
            errs.append(f'{arm}: synthetic recipe does not satisfy the frozen signature {_sig_arm!r} (decoded {_recipe_knobs(_recipe)!r})')
        _pred[arm] = {'eager_batch': (bool(_eb) if _eb is not None else None),
                      'amp_dtype': _tk.get('amp_dtype'),
                      'train_knobs': _tk, 'recipe': _recipe}
        _arm_cfgs[arm] = cfgs
        # Parameter matching is part of the frozen comparison contract: the
        # analysis refuses a pair whose totals differ by >1%, but discovering
        # that only after P5T would waste the access.  Build the real shapes
        # here (CPU) and check it up front.
        try:
            _mr_for_shape = float(_expected_mr) if _expected_mr is not None else float(_actual_mr)
            _shape_model = L.SmallGPT(int(sig.get('v', 8192)), int(sig.get('d', 256)),
                                      _n_layers, int(sig.get('H', 8)), int(sig.get('Dh', 32)),
                                      int(sig.get('sl', 512)), cfgs, mlp_ratio=_mr_for_shape)
            _pred[arm]['params'] = int(L.count_params(_shape_model))
            # The ACTUAL optimisation objective of this arm (comp_reg / delta-LR
            # / boundary policy, read off the built model, not the arm name) —
            # the single-factor verdict below compares the two arms' objectives
            # through the same descriptor the record-level gate uses.
            _pred[arm]['train_objective'] = L.train_objective(
                _shape_model, comp_lambda=float(sig.get('cl', 0.05)),
                delta_lr_mult=float(sig.get('dlm', 10.0)))
            del _shape_model
        except Exception as _e:
            errs.append(f'{arm}: could not compute the parameter count for the frozen shape ({type(_e).__name__}: {_e})')
            _pred[arm]['params'] = None
        for s in seeds:
            _records[f'{arm}::s{s}'] = {
                'variant': arm, 'seed': int(s), 'split': 'test',
                'by_len': {str(Ln): {'ppl': 100.0, 'n_tok': int(_nseq) * int(Ln - 1),
                                     'n_seq': int(_nseq), 'ppl_per_seq': [100.0] * int(_nseq),
                                     'truncated': False}
                           for Ln in _lens},
                'data_fp': 'plan-test-slice-fp', 'train_fp': 'plan-train-fp',
                '_code': L.code_stamp(), 'manifest_sha256': m.get('manifest_sha256'),
                'recipe': _recipe, 'train_knobs': _tk, 'n_seq': int(_nseq),
                'params': 11089904,
            }
    try:
        admitted, _fp = _admit_confirmation_records(_records, m)
    except SystemExit as _e:
        errs.append(f'admission run raised: {_e}')
        admitted = {}
    if len(admitted) != len(arms) * len(seeds):
        errs.append(f'admission accepted {len(admitted)}/{len(arms) * len(seeds)} synthetic frozen cells — P5T would burn the test access and then admit only this many real records')
    _comp_report = {}
    for comp in m['primary_comparisons']:
        _a, _b = comp.get('a'), comp.get('b')
        try:
            _Ln = int(str(comp.get('endpoint', 'by_len.ppl@4096')).rsplit('@', 1)[-1])
        except (TypeError, ValueError, OverflowError):
            _Ln = None
        if _Ln is None:
            errs.append(f'comparison {comp.get("name")!r} has an unparseable endpoint {comp.get("endpoint")!r}; the frozen by-length endpoint must be `...@<integer>`')
        _missing = [x for x in (_a, _b) if x not in arms]
        if _missing:
            errs.append(f'comparison {comp.get("name")!r} names arms outside the frozen list: {_missing}')
        if _Ln is not None and _lens and _Ln not in _lens:
            errs.append(f'comparison {comp.get("name")!r} endpoint L{_Ln} is not in eval.by_length_ppl {_lens}')
        _low = [s for s in seeds if (_a, s) not in admitted or (_b, s) not in admitted]
        if _low:
            errs.append(f'comparison {comp.get("name")!r} cannot form all {len(seeds)} pairs; missing seeds {_low}')
        _pa = (_pred.get(_a) or {}).get('params')
        _pb = (_pred.get(_b) or {}).get('params')
        _rel = None
        if _pa and _pb:
            _rel = abs(_pa - _pb) / max(_pa, _pb, 1)
            if _rel > 0.01:
                errs.append(f'comparison {comp.get("name")!r} is NOT parameter-matched at the frozen shapes ({_pa} vs {_pb}, rel diff {_rel:.4%} > 1%) — the analysis would refuse it after the test access')
        # ---- single-factor verdict vs the mechanism-claim flag (review-7) ----
        # A frozen comparison may carry `mechanism_claim_allowed=true` only if
        # it is verifiably single-factor: the attention-cfg diff isolates ONE
        # declared intervention family (no `kind` swap), the two arms' ACTUAL
        # objectives match, and the parameter totals are matched.  A True flag
        # on a multi-factor contrast is a design defect and refuses the plan;
        # a False flag on a config-single-factor contrast stays allowed (the
        # caveat may rely on read-budget/interpretation constraints outside
        # the attention cfg — conservative direction is never an error).
        _diff_keys = None
        if _a in _arm_cfgs and _b in _arm_cfgs:
            _diff_keys = set()
            for _ca, _cb in zip(_arm_cfgs[_a], _arm_cfgs[_b]):
                _diff_keys.update(L.layer_cfg_diff(_ca, _cb).keys())
        _obj_reason = L._objective_diff({'train_objective': (_pred.get(_a) or {}).get('train_objective')},
                                        {'train_objective': (_pred.get(_b) or {}).get('train_objective')})[0] \
            if ((_pred.get(_a) or {}).get('train_objective') is not None
                and (_pred.get(_b) or {}).get('train_objective') is not None) else 'unverifiable train_objective (arm model not built)'
        _sf = bool((_diff_keys is not None) and L.cfg_keys_single_factor(_diff_keys)
                   and (_obj_reason is None)
                   and (_rel is None or _rel <= 0.01))
        _flag = comp.get('mechanism_claim_allowed')
        if _flag is True and not _sf:
            _why = (f'cfg diff keys {sorted(_diff_keys)}'
                    if _diff_keys is not None else 'cfg diff not computable')
            if _obj_reason is not None:
                _why += f'; objective mismatch ({_obj_reason})'
            if _rel is not None and _rel > 0.01:
                _why += f'; parameter totals differ by {_rel:.4%}'
            errs.append(f'comparison {comp.get("name")!r} declares mechanism_claim_allowed=true but is NOT single-factor ({_why}) — a mechanism claim requires the attention-cfg diff to isolate one declared intervention family, matching objectives and matched parameters; re-derive the flag or re-design the arms BEFORE any test access')
        if _flag is not True and _sf:
            notes.append(f'comparison {comp.get("name")!r} is config-single-factor ({sorted(_diff_keys)}) but carries a conservative mechanism_claim_allowed={_flag!r}; keeping the conservative flag (the caveat may rest on constraints outside the attention cfg, e.g. read budget or interpretation)')
        _comp_report[comp.get('name')] = {'a': _a, 'b': _b, 'endpoint_L': _Ln, 'pairs_planned': len(seeds) - len(_low), 'params_a': _pa, 'params_b': _pb, 'param_rel_diff': _rel,
                                         'cfg_diff_keys': (sorted(_diff_keys) if _diff_keys is not None else None),
                                         'objective_matched': (_obj_reason is None),
                                         'single_factor': _sf,
                                         'mechanism_claim_allowed': _flag,
                                         'claim_flag_consistent': (_flag is not True) or _sf}
    # ---- design preflight: budgets + evidence level (zero GPU) ----------
    # Single-factor / parameter / objective checks above answer "is the
    # attention cfg the only declared factor?".  They do NOT answer "do the
    # arms share a token and read/compute budget?" or "is this manifest allowed
    # to call the result confirmatory?".  Both are design questions and must
    # refuse the plan before any GPU spend or test access.
    _design = {}
    try:
        _ledger = L.comparison_budget_ledger(m)
        _ev = L.evidence_level_audit(m, manifest_path=manifest_path, panel_paths=[])
        _evd = L.evaluation_design_audit(m)
        _design = {'budget_ledger': _ledger, 'evidence_level': _ev,
                   'evaluation_design': _evd}
        if _ledger.get('status') == 'fail':
            errs.extend(f'budget ledger: {v}' for v in (_ledger.get('violations') or []))
        if _ev.get('status') == 'fail':
            errs.extend(f'evidence level: {v}' for v in (_ev.get('violations') or []))
        if _evd.get('status') == 'fail':
            errs.extend(f'evaluation design: {v}'
                        for v in (_evd.get('violations') or []))
        notes.extend(f'budget ledger: {w}' for w in (_ledger.get('warnings') or []))
        notes.extend(f'evidence level: {w}' for w in (_ev.get('warnings') or []))
    except Exception as _e:
        errs.append(
            f'the design preflight could not run ({type(_e).__name__}: {_e}); a '
            'confirmation needs a verifiable token/parameter/read-budget ledger '
            'and evidence-level declaration')
    # Practical-significance readiness: a future plan that requires a SESOI
    # must declare one for every primary comparison.  The shipped v3 manifest
    # predates the block, so this is only a note today (never a retroactive
    # failure of the already-frozen plan).
    try:
        _psd = L.confirmation_sesoi_design_audit(m)
        if _psd.get('status') == 'fail':
            errs.extend(f'practical significance: {p}' for p in (_psd.get('problems') or []))
        else:
            notes.extend(f'practical significance: {w}' for w in (_psd.get('warnings') or []))
    except Exception as _pse:
        _psd = {'kind': 'confirmation_sesoi_design_audit', 'status': 'not_estimable',
                'problems': [f'{type(_pse).__name__}: {_pse}']}
        notes.append(f'practical significance: design audit unavailable ({_pse})')
    report = {'ok': not errs, 'manifest': manifest_path, 'manifest_sha256': m.get('manifest_sha256'),
              'code_semantics': L.CODE_SEMANTICS, 'n_arms': len(arms), 'n_seeds': len(seeds),
              'n_min_for_confirmation': _n_min, 'min_attainable_p_holm': _min_holm,
              'predicted': _pred, 'admitted': len(admitted), 'comparisons': _comp_report,
              'design_preflight': _design,
              'errors': errs, 'notes': notes,
              'practical_significance_design': _psd}
    if errs:
        print('[v12 plan] REFUSED — the frozen plan is not runnable end-to-end:')
        for _e in errs:
            print(f'  - {_e}')
        raise SystemExit('[v12 plan] FAILED: fix the plan/records BEFORE any P5C training or P5T test access')
    print(f'[v12 plan] OK: {len(admitted)}/{len(arms) * len(seeds)} synthetic records admitted; {len(_comp_report)} comparisons form all pairs; min n={_n_min} (min Holm p={_min_holm:.4g} <= alpha={_alpha})')
    for _a, _p in _pred.items():
        print(f'  [plan] {_a:32s} eager_batch={_p["eager_batch"]} amp={_p["amp_dtype"]} mr={(_per_arm.get(_a) or {}).get("mr", 4.0)} params={_p.get("params")}')
    for _n in notes:
        print(f'  [plan][note] {_n}')
    _archive_plan_snapshot(report, m, manifest_path)
    globals()['_PLAN_DONE'] = True
    return report


def run_plan(manifest_path=FREEZE_MANIFEST):
    return plan_check(manifest_path)


DYNAMIC_MANIFEST = 'confirmation_frozen_dynamic.json'

def dynamic_plan_check(manifest_path=DYNAMIC_MANIFEST):
    """Zero-GPU design preflight for the DYNAMIC/HCA draft.

    Deliberately does NOT call `_load_manifest`: that loader refuses
    `status=draft`, which is exactly the state a design record must be in
    before its disjoint test source and power gate exist.  This check audits
    the draft as a DESIGN (hypotheses + dedicated controls + budget ledger +
    evidence level) and prints why it is not runnable yet.  Passing it never
    upgrades the draft to a confirmation.
    """
    if not os.path.exists(manifest_path):
        raise SystemExit(
            f'[v12 dynamic-plan] REFUSE: no dynamic design record at '
            f'{manifest_path!r}; write the hypotheses and dedicated controls '
            'before adding dynamic/HCA arms')
    try:
        with open(manifest_path, encoding='utf-8') as _f:
            m = json.load(_f)
    except Exception as _e:
        raise SystemExit(
            f'[v12 dynamic-plan] REFUSE: {manifest_path} is not readable JSON '
            f'({type(_e).__name__}: {_e})') from None
    d = L.dynamic_design_audit(m, manifest_path=manifest_path)
    led = L.comparison_budget_ledger(m)
    ev = L.evidence_level_audit(m, manifest_path=manifest_path, panel_paths=[])
    errs = list(d.get('violations') or []) + list(led.get('violations') or []) \
        + list(ev.get('violations') or [])
    report = {'ok': not errs, 'manifest': manifest_path,
              'declared_status': m.get('status'),
              'evidence_level': ev.get('manifest_level'),
              'runnable': d.get('runnable'),
              'promotion_ready': d.get('promotion_ready'),
              'power': d.get('power'),
              'calibration': d.get('calibration'),
              'design_cost': led.get('design_cost'),
              'design': d, 'budget_ledger': led, 'evidence_level_audit': ev,
              'errors': errs}
    if errs:
        print('[v12 dynamic-plan] REFUSED — the dynamic design is not coherent:')
        for _e in errs:
            print(f'  - {_e}')
        raise SystemExit(
            '[v12 dynamic-plan] FAILED: fix the dynamic design BEFORE any '
            'dynamic/HCA training or any disjoint-test access')
    print(f"[v12 dynamic-plan] OK: design coherent; evidence_level={ev.get('manifest_level')}; "
          f"hypotheses={sorted(d.get('hypotheses') or {})}; "
          f"runnable={d.get('runnable')}; promotion_ready={d.get('promotion_ready')}")
    _pw = d.get('power') or {}
    _cal = d.get('calibration') or {}
    _cost = led.get('design_cost') or {}
    if _pw:
        print(f"  [dynamic-plan] power: n={_pw.get('n_seeds')} m={_pw.get('family_m')} "
              f"projected@{_pw.get('power_q_for_test')}={_pw.get('projected_power')} "
              f"min={_pw.get('min_power_for_test')}")
    if _cal:
        print(f"  [dynamic-plan] read calibration: status={_cal.get('status')} "
              f"expected_topk={_cal.get('expected_topk')} arm={_cal.get('arm')}")
    if _cost:
        print(f"  [dynamic-plan] design cost: {_cost.get('n_runs')} runs, "
              f"{_cost.get('consumed_tokens_total')} consumed tokens (scheduling aid)")
    for _w in (d.get('warnings') or []) + (led.get('warnings') or []) + (ev.get('warnings') or []):
        print(f'  [dynamic-plan][note] {_w}')
    if not d.get('promotion_ready'):
        print('  [dynamic-plan] DRAFT: this is a design record, NOT a confirmation. '
              'Reserve a disjoint test source, measure the realized dynamic block '
              'length to calibrate the read-matched fixed top-k, and pass the '
              'pre-test power gate before freezing.')
    return report


def run_smoke():
    import copy
    import shutil
    # Smoke must never write into the frozen wt103_cache: a differently
    # tokenised tok_v5_*.json could make a later cold P5T build fail AFTER the
    # test read.  An isolated cache under the smoke outdir keeps the frozen
    # cache clean.
    os.environ.setdefault('WT103_CACHE_DIR', os.path.join('results_smoke_v12', 'wt103_cache'))
    print('[smoke] 1) tiny frozen-confirmation roundtrip on a synthetic local test split')
    L.set_seed(0)
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    payload = dict(V10.P3MT_PAYLOAD, outdir='results_smoke_v12', seeds=[0], steps=40, eval_lens=[512, 1024])
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope', 'csa_fixed_rope_allblocks']), guard=None, label='v12 smoke', par=False)
    # a smoke freeze manifest: same schema, tiny arms/seeds, current code stamp.
    # statistics.descriptive_only=1 opts the smoke manifest out of the load-time
    # power gate (n=1 can never confirm — that is fine for a plumbing smoke);
    # binding.training_recipe_signature exercises the structured-recipe admission gate.
    _smoke_manifest = {'frozen_at': time.strftime('%Y-%m-%d'), 'code_semantics_expected': L.CODE_SEMANTICS, 'arms': ['csa_fixed_rope', 'csa_fixed_rope_allblocks'], 'seeds': [0], 'primary_comparisons': [{'name': 'smoke: allblocks_scratch - learned @ L1024', 'a': 'csa_fixed_rope_allblocks', 'b': 'csa_fixed_rope', 'endpoint': 'by_len.ppl@1024'}], 'statistics': {'test': 'exact two-sided sign-flip', 'n_seeds': 1, 'p_floor': 1.0, 'family': 'primary_comparisons', 'correction': 'holm step-down', 'decision_rule': 'smoke only', 'descriptive_only': True}, 'binding': {'training_recipe': {'steps': 40, 'sl': 512, 'bs': 12}}, 'eval': {'by_length_ppl': [512, 1024], 'n_seqs_clean': 4, 'distractor_probe': {'mode': 'uniform', 'rhos': [0.0, 0.5], 'eval_lens': [1024], 'target': 512, 'n_seq': 4}}}
    _mpath = 'results_smoke_v12/frozen.json'
    L.atomic_write_json(_mpath, _smoke_manifest, indent=2)
    # smoke test data: rows from the (tiny local dump or HF) validation split,
    # standing in for the test split so the smoke never depends on wiki.test.raw
    _, val_ids, _, _, _ = L.load_wikitext(1024, 4000000)
    cfg = dict(FINALTEST_CFG, manifest=_mpath, outdir='results_smoke_v12', ckpt_dir='results_smoke_v12/ckpt', mech_summary='results_smoke_v12/summary.json', bylen_summary='results_smoke_v12/bylen.json', probe_summary='results_smoke_v12/distractor.json', n_seq=4, chunk=2, stats_out='results_smoke_v12/test_stats.json', test_ids=np.asarray(val_ids[:4, :1025], dtype=np.int64), smoke=True)
    stats = run_finaltest(cfg)
    got = {r['variant'] for r in json.load(open('results_smoke_v12/bylen.json', encoding='utf-8')).values() if isinstance(r, dict)}
    assert {'csa_fixed_rope', 'csa_fixed_rope_allblocks'} <= got, f'finaltest bylen cells missing: {got}'
    _probe = {}
    if os.path.exists('results_smoke_v12/distractor.json'):
        _probe = json.load(open('results_smoke_v12/distractor.json', encoding='utf-8'))
    _pgot = {(r.get('mode', 'uniform'), r['arm'], float(r['rho'])) for r in _probe.values() if isinstance(r, dict)}
    _pwant = {('uniform', 'learned', 0.0), ('uniform', 'learned', 0.5), ('uniform', 'dense', 0.0), ('uniform', 'dense', 0.5)}
    _pwant = {w for w in _pwant if w[1] != 'dense'}  # full_rope arm is not in the smoke manifest
    assert _pwant <= _pgot, f'finaltest probe cells missing: {_pwant - _pgot}'
    assert 'comparisons' in stats and stats['comparisons'], 'finaltest stats produced no comparison record'
    print(f'  finaltest roundtrip OK ({len(got)} bylen arms, {len(_probe)} probe cells, {len(stats["comparisons"])} frozen comparison(s))')
    print('[smoke] 2) report builder on the smoke artifacts')
    build_report(out='results_smoke_v12/REPORT_v12.md', stats_p='results_smoke_v12/stats.json')
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    print('\n[smoke] PASSED')

def _run_cli():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'plan', 'dynamic-plan', 'analysis', 'report', 'phase', 'finaltest', 'power', 'auto')
    if mode in ('-h', '--help', 'help'):
        print(__doc__)
        raise SystemExit(0)
    if mode not in _MODES:
        print(f'[v12] unknown mode {mode!r}; expected one of {_MODES} (no argument means "full").  Refusing to start a run.')
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f'[v12] mode "phase" needs a phase name, e.g. `python -m explib.confirmation phase P5C`.  Available: {[p[0] for p in PHASES]}')
        raise SystemExit(2)
    print(f'[v12] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'plan':
            run_plan(sys.argv[2] if len(sys.argv) > 2 else FREEZE_MANIFEST)
        elif mode == 'dynamic-plan':
            dynamic_plan_check(sys.argv[2] if len(sys.argv) > 2 else DYNAMIC_MANIFEST)
        elif mode == 'analysis':
            v12_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'power':
            # zero-GPU: exact conditional power on the P5C validation deltas,
            # before the one-shot test access (never calls load_wikitest)
            run_power(sys.argv[2] if len(sys.argv) > 2 else FREEZE_MANIFEST)
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v12: phase {sys.argv[2]} results')
        elif mode == 'finaltest':
            run_finaltest(dict(FINALTEST_CFG))
        elif mode == 'auto':
            # Unattended entry: run_full already runs the zero-GPU plan gate
            # (fail-closed) before any GPU time and archives the approval
            # snapshot; `auto` is the explicit no-human-pause name for it.
            run_full()
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v12: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode in ('full', 'auto'):
            schedule_shutdown(2400)
        sys.exit(1)
