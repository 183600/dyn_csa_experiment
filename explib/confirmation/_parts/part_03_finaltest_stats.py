

def finaltest_stats(manifest=None, bylen_path=None, probe_path=None, out='analysis_v12/test_stats.json'):
    """Zero-GPU: paired confirmation statistics under the frozen plan.

    统计入口现在是**严格准入制**——种子白名单、唯一
    记录、非空数据指纹、单一冻结 test 切片、_code/recipe 与冻结计划的机器
    校验、manifest 哈希绑定，缺一项即不入统计（点名剔除）或拒绝运行
    （歧义/混切片）。by_length 汇总同样只由可采纳记录构成。"""
    if manifest is None:
        manifest = _load_manifest(FREEZE_MANIFEST)
    bylen_path = bylen_path or os.path.join(TEST_OUTDIR, 'bylen.json')
    probe_path = probe_path or os.path.join(TEST_OUTDIR, 'distractor.json')
    raw = {}
    if os.path.exists(bylen_path):
        raw = json.load(open(bylen_path, encoding='utf-8'))
    admitted, unified_fp = _admit_confirmation_records(raw, manifest)
    # The analysis entry point is as much part of the frozen round as P5T:
    # reuse the same persistent lock so a manifest or test-slice change after
    # the first access is refused here too, not only during evaluation.  A
    # non-empty confirmation file without its first-access lock is refused:
    # a copied bylen.json alone cannot prove which plan/slice produced it.
    if unified_fp is not None:
        _lock_dir = os.path.dirname(os.path.abspath(bylen_path))
        if not os.path.exists(_freeze_lock_path(_lock_dir)):
            raise SystemExit(f'[v12] REFUSE: {bylen_path} has admissible confirmation records but no freeze lock in {_lock_dir} — refuse to analyse a copied result that cannot be tied to the first test-split access')
        _adm_train_fps = {r.get('train_fp') for r in admitted.values() if isinstance(r.get('train_fp'), str) and r.get('train_fp')}
        _check_or_write_freeze_lock(_lock_dir, manifest, unified_fp, stage='analysis',
                                    train_fp=(next(iter(_adm_train_fps)) if len(_adm_train_fps) == 1 else None))
    _stat = manifest.get('statistics') or {}
    _alpha = float(_stat.get('alpha', 0.05))
    _require_cov = bool(_stat.get('require_complete_coverage'))
    # The declared family is part of the artifact, not only of the manifest in
    # the repository.  Without it a tampered artifact could delete one
    # comparison and its Holm slot, shrinking the correction family while
    # every surviving number stayed internally consistent.  The p-value replay
    # audit compares the recorded names against the comparisons it can see.
    _family_names = [str(c.get('name')) for c in (manifest.get('primary_comparisons') or [])]
    stats = {'alpha': float(_alpha), 'manifest': {'frozen_at': manifest.get('frozen_at'), 'manifest_sha256': manifest.get('manifest_sha256'), 'arms': manifest['arms'], 'seeds': manifest['seeds'], 'code_semantics_expected': manifest.get('code_semantics_expected'), 'statistics': manifest.get('statistics'), 'scope': manifest.get('scope'), 'revision': manifest.get('revision'), 'primary_comparison_names': _family_names, 'family_size': len(_family_names)}, 'admission': {'n_admitted': len(admitted), 'test_slice_fp': unified_fp, 'n_effective_seeds': len({int(s) for (_v, s) in admitted}), 'n_declared_seeds': len(manifest['seeds'])}, 'comparisons': {}, 'holm': {}, 'by_length': {}, 'probe': _test_probe_stats(probe_path, manifest, unified_fp)}
    # The exact sign-flip p is valid only under within-seed exchangeability.
    # admission binds the corpus/slice/row count/recipe/code, but the RNG stream
    # and batch order are not independently re-derivable from a record; report
    # them instead of silently assuming them.
    _rngs = sorted({str(r.get('rng_fp')) for r in admitted.values() if r.get('rng_fp')})
    # init_core_fp is a function of the SEED: a healthy panel has a different
    # digest for every seed and the SAME digest for the two arms within a
    # seed.  The old global "all values equal" test was both conceptually
    # wrong (it would fail a healthy panel) and vacuously true when no record
    # carried the field.  Summarise the per-seed two-arm contract instead.
    _init_by_seed = {}
    for (_v, _s), _r in admitted.items():
        _iv = _r.get('init_core_fp')
        # Same rule as the per-comparison audit: a malformed non-string value
        # is unverified, never stringified into a match.
        if isinstance(_iv, str) and _iv.strip():
            _init_by_seed.setdefault(int(_s), {}).setdefault(str(_v), set()).add(_iv.strip())
    _init_mismatch, _init_complete = [], True
    if set(_init_by_seed) != {int(s) for s in manifest['seeds']}:
        # A panel-level "matched initialization" statement needs evidence for
        # every declared seed, not only the ones that happened to be admitted.
        _init_complete = False
    for _s, _byarm in _init_by_seed.items():
        _vals = set().union(*_byarm.values())
        # More than one digest for the same seed is a definite inconsistency
        # whether it appears across arms or (via a duplicate record) within
        # one arm.  Admission refuses duplicate (arm, seed) records, so the
        # second case is defensive.
        if len(_vals) > 1 or any(len(_x) > 1 for _x in _byarm.values()):
            _init_mismatch.append(int(_s))
        if set(_byarm) != set(manifest['arms']):
            _init_complete = False
    _inits = sorted({x for _byarm in _init_by_seed.values() for _v in _byarm.values() for x in _v})
    _same_init = (None if not _init_by_seed
                  else (False if _init_mismatch
                        else (True if _init_complete else None)))
    stats['validity'] = {
        'rng_fp_values': _rngs,
        'same_rng_fp': len(_rngs) <= 1,
        'init_core_fp_values': _inits,
        'init_core_fp_recorded': bool(_init_by_seed),
        'init_core_fp_seeds': sorted(_init_by_seed),
        'init_core_fp_complete': bool(_init_complete),
        'init_core_fp_mismatch_seeds': _init_mismatch,
        'same_init_core_fp': _same_init,
        'n_seq_values': sorted({n for n in (_record_int(r.get('n_seq')) for r in admitted.values()) if n is not None}),
        'note': ('exact sign-flip requires within-seed exchangeability (same data order / eval rows / corruption grid). data_fp, train_fp, n_seq and recipe are machine-checked; the shared-trunk initialization digest (init_core_fp) is checked PER SEED between the two arms (a definite mismatch drops that pair; a one-sided/absent digest is UNVERIFIED, never read as a match), and rng_fp is reported but NOT enforced (architecture-induced RNG consumption is part of the arm). The exchangeability claim therefore remains a declared assumption, but the matched-initialization part of it is now measured and enforced rather than vacuously true.'),
    }
    # The frozen comparison is parameter-matched, not compute-matched.
    # Refuse a non-descriptive plan that does not disclose
    # both facts, so the caveat cannot be silently dropped from the report.
    _cbd = manifest.get('compute_budget_disclosure')
    _descriptive = (manifest.get('statistics') or {}).get('descriptive_only') is True
    if not _descriptive:
        _missing = [k for k in ('parameter_matching', 'flops', 'read_fraction')
                    if not (isinstance(_cbd, dict) and _cbd.get(k))]
        if _missing:
            raise SystemExit(f'[p5t stats] REFUSE: the frozen plan lacks compute-budget disclosure {_missing}. Parameter matching is not compute matching; the report must state the FLOPs/read-budget mismatch next to every primary comparison .')
    stats['compute_budget_disclosure'] = _cbd
    stats['parameter_binding'] = {
        'required_by_manifest': bool(_manifest_requires_params(manifest)),
        'confirmation_requires_complete': bool(_manifest_requires_params(manifest)),
        'n_admitted_records': len(admitted),
        'n_admitted_without_params': sum(
            1 for r in admitted.values()
            if (_record_int(r.get('params')) or 0) <= 0),
        'note': ('a disclosed parameter-matching contract makes params part of '
                 'the admission identity; a missing count is excluded rather '
                 'than read as param_matched=None'),
    }
    for v in manifest['arms']:
        rows = [r for (rv, s), r in admitted.items() if rv == v]
        _lens = sorted({n for n in (_record_int(k) for r in rows for k in (r.get('by_len') or {}))
                        if n is not None})
        for Ln in _lens:
            _by_seed = {}
            _nll_sum = 0.0
            _n_tok = 0
            _pool_ok = True
            for r in rows:
                bl = r.get('by_len') or {}
                c = bl.get(Ln, bl.get(str(Ln)))
                if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')):
                    _by_seed[r.get('seed')] = float(c['ppl'])
                    _nll_f = L._finite_scalar(c.get('nll_sum'))
                    _nt_i = _record_int(c.get('n_tok'))
                    if _nll_f is not None and _nt_i is not None and _nt_i > 0:
                        _nll_sum += _nll_f
                        _n_tok += _nt_i
                    else:
                        _pool_ok = False
            if _by_seed:
                _dl = list(_by_seed.values())
                _pooled = L._safe_exp_ppl(_nll_sum / _n_tok) if (_pool_ok and _n_tok > 0) else None
                stats['by_length'].setdefault(v, {})[str(Ln)] = {'mean': float(_pooled if _pooled is not None else np.mean(_dl)),
                                                                  'mean_arithmetic': float(np.mean(_dl)),
                                                                  'ppl_pooled': _pooled, 'n_tokens': _n_tok,
                                                                  'n': len(_dl), 'ppl_by_seed': {str(s): p for s, p in _by_seed.items()}, 'params': next((r.get('params') for r in rows if isinstance(r.get('params'), (int, float))), None), 'n_seqs': next((c.get('n_seq') for _r in rows for c in [((_r.get('by_len') or {}).get(Ln) or (_r.get('by_len') or {}).get(str(Ln)) or {})] if isinstance(c, dict) and c.get('n_seq') is not None), None)}
    for comp in manifest['primary_comparisons']:
        name = comp['name']
        a, b = (comp['a'], comp['b'])
        ep = comp.get('endpoint', 'by_len.ppl@4096')
        Ln = int(str(ep).rsplit('@', 1)[-1])
        exp_sign = comp.get('expected_sign')
        if exp_sign is not None:
            exp_sign = str(exp_sign).lower()
        recs_a = {s: r for (rv, s), r in admitted.items() if rv == a}
        recs_b = {s: r for (rv, s), r in admitted.items() if rv == b}
        # a paired delta is only valid on ONE held-out slice; the admission
        # gate has already enforced that EVERY admissible record shares the
        # same frozen fingerprint, so pairing cannot mix slices (and a missing
        # fingerprint could never enter here — None==None is rejected).
        common = sorted(set(recs_a) & set(recs_b))
        # Drop only the incomplete PAIR, not the whole comparison: one failed
        # L4096 cell would otherwise omit the entire comparison.  Use every
        # complete pair and report the dropped seeds.  Whether n<12 may still
        # confirm is a PRE-REGISTERED choice (statistics.min_seeds_for_confirmation,
        # manifest v3: 8, the mathematical minimum for Holm/4 with alpha=0.05).
        def _cell_ppl(rec):
            bl = rec.get('by_len') or {}
            c = bl.get(Ln, bl.get(str(Ln)))
            return float(c['ppl']) if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')) else None
        # A paired sign-flip test is only valid on pairs that started from the
        # same shared trunk.  Check it PER SEED (the digest is seed-dependent):
        # a definite mismatch is not a pair and is dropped by name; a legacy
        # one-sided/absent digest is reported as unverified, never as a match.
        _init_aud = L.paired_init_core_audit(recs_a, recs_b, common)
        _init_status = {e['seed']: e['status'] for e in _init_aud['per_seed']}
        dl = []
        usable = []
        dropped = []
        for s in common:
            if _init_status.get(int(s)) == 'mismatch':
                dropped.append({'seed': int(s), 'reason': 'init_core_fp mismatch: the two arms did not start from the same shared trunk, so the seed is not a matched pair'})
                continue
            pa, pb = (_cell_ppl(recs_a[s]), _cell_ppl(recs_b[s]))
            if pa is None or pb is None:
                dropped.append({'seed': int(s), 'reason': f'missing usable untruncated cell at L{Ln}'})
                continue
            dl.append(pa - pb)
            usable.append(s)
        if len(usable) < 2:
            stats['comparisons'][name] = {'omitted': f'<2 complete seed pairs at the frozen endpoint (common records: {common}; usable pairs: {usable}; whitelist {manifest["seeds"]})', 'n': 0, 'expected_sign': exp_sign, 'dropped_pairs': dropped, 'init_core_fp': _init_aud}
            continue
        res = V.exact_sign_permutation(dl)
        _n_eff = len(usable)
        _n_min = int((manifest.get('statistics') or {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))
        res['seeds'] = usable
        res['n_effective'] = int(_n_eff)
        res['coverage_ok'] = bool(_n_eff == len(manifest['seeds']))
        res['missing_seeds'] = sorted(set(int(s) for s in manifest['seeds']) - set(int(s) for s in usable))
        res['dropped_pairs'] = dropped
        res['init_core_fp'] = _init_aud
        res['same_init_core_fp'] = _init_aud['pairing_ok']
        res['init_core_fp_recorded'] = any(('a' in e) or ('b' in e)
                                          for e in _init_aud['per_seed'])
        res['min_seeds_for_confirmation'] = _n_min
        # Diagnostics must use the n that actually entered THIS comparison,
        # so p_floor / min_attainable_p_holm / power describe the test that
        # really ran.
        # Effective resolution uses the number of NON-ZERO paired deltas;
        # exact ties cannot resolve anything.  This must describe the test
        # that actually ran, not the theoretical all-nonzero floor.
        _floor_n = int(res.get('n_nonzero', _n_eff))
        _fl_eff = (1.0 if _floor_n == 0 else L.signflip_p_floor(_floor_n))
        if math.isfinite(_fl_eff):
            res['p_floor_effective'] = _fl_eff
            res['min_attainable_p_holm_effective'] = L.holm_min_attainable_p(_floor_n, len(manifest['primary_comparisons']), _alpha)
            res['power_first_holm_gate_effective'] = {f'q={q:.2f}': L.signflip_power(_floor_n, len(manifest['primary_comparisons']), q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        _param = _param_match_detail(recs_a, recs_b, usable, a, b)
        res.update(_param)
        _act = []
        for _recs in (recs_a, recs_b):
            for s in usable:
                _pbd = _recs[s].get('param_breakdown')
                if isinstance(_pbd, dict):
                    _act_f = L._finite_scalar(_pbd.get('active'))
                    if _act_f is not None:
                        _act.append(_act_f)
        res['active_params_mean'] = (float(np.mean(_act)) if _act else None)
        res['data_fp'] = unified_fp
        res['manifest_sha256'] = manifest.get('manifest_sha256')
        res['delta_definition'] = f'{a} - {b} (paired per seed, test split)'
        res['expected_sign'] = exp_sign
        res['mechanism_claim_allowed'] = comp.get('mechanism_claim_allowed')
        res['claim_note'] = comp.get('claim_note')
        res['confirmation_coverage_required'] = bool(_require_cov)
        # The full by-length curve on the SAME admitted records.  Only the
        # frozen endpoint enters the Holm family / the confirmation decision;
        # the other lengths (notably L512, the training length) are
        # descriptive and are reported so an in-distribution reversal cannot be
        # hidden behind the L4096 extrapolation endpoint.
        res['by_length_curve'] = {}
        for _Ln2 in sorted({int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or [])}):
            _dl2, _used2 = _paired_len_deltas(recs_a, recs_b, usable, _Ln2)
            if len(_dl2) < 2:
                continue
            _o2 = L.exact_signflip_p(_dl2)
            res['by_length_curve'][str(_Ln2)] = {
                'n': len(_dl2), 'seeds': _used2, 'mean': float(np.mean(_dl2)),
                # Persist the exact paired deltas behind EVERY curve p, not
                # only behind the primary endpoint.  A reader (or the replay
                # audit) can then recompute the descriptive p instead of
                # trusting the stored float; it is the same vector the test
                # above consumed.
                'per_seed': [float(x) for x in _dl2],
                'n_nonzero': int(_o2.get('n_nonzero') or 0),
                'p_floor': _o2.get('p_floor'),
                'method': _o2.get('method'),
                'p_exact_signflip': float(_o2['p']),
                'role': ('primary' if _Ln2 == Ln else 'descriptive'),
                'descriptive_only': bool(_Ln2 != Ln),
                'note': ('primary frozen endpoint' if _Ln2 == Ln else
                         'same records, different length; not in the frozen Holm family and never confirmatory'),
            }
        # Does the arm contrast CHANGE with length?  Separate per-length
        # sign-flip p-values cannot answer that (a difference in significance
        # is not a significant difference).  The interaction uses the same
        # seeds at every length, conditions on those seeds, and is post-hoc:
        # it is never added to the frozen Holm family or the decision rule.
        _li_lens = sorted(
            {int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or [])}
            | {int(Ln)})
        _li = _length_interaction_analysis(
            recs_a, recs_b, usable, _li_lens, primary_len=Ln,
            alpha=_alpha, min_n=6)
        if _li is not None and _li.get('n_complete', 0) >= 2:
            res['length_interaction'] = _li
        res['in_distribution_L512'] = res['by_length_curve'].get('512')
        if exp_sign in ('positive', 'negative') and math.isfinite(res.get('mean', float('nan'))):
            res['direction_ok'] = bool((res['mean'] > 0) == (exp_sign == 'positive'))
        else:
            res['direction_ok'] = None
        _doc = _doc_level_analysis(recs_a, recs_b, usable, Ln,
                                   primary_deltas=dl)
        if _doc is not None:
            res['doc_sampling'] = _doc
        stats['comparisons'][name] = res
    # Holm is corrected over the DECLARED frozen family, not over whichever
    # comparisons happened to produce a finite p: an omitted comparison enters
    # with p=1.0 so a missing arm can never shrink m and relax the correction.
    _fam = []
    for k, v in stats['comparisons'].items():
        p = v.get('p_exact_signflip')
        _fam.append((k, float(p) if isinstance(p, (int, float)) and math.isfinite(p) else 1.0))
    if _fam:
        stats['holm'] = L.holm_adjust(_fam, alpha=_alpha)
    _rej = [k for k, v in stats['holm'].items() if v.get('p_holm') is not None and float(v['p_holm']) < _alpha]
    # The pre-registered decision rule (manifest v4) requires p_holm < alpha
    # AND the observed direction AND parameter matching AND, when
    # statistics.require_complete_coverage is true, all frozen seeds.  A
    # complete-case subset can be selected by wall-clock skips, missing
    # checkpoints or the health gate; those are not random attrition, so the
    # shipped plan refuses to call such a subset "confirmed".  The 8-pair floor
    # remains the mathematical resolution floor and is still reported.
    # A disclosed parameter-matching contract is only satisfied by a COMPLETE
    # per-seed parameter record.  `param_matched is not false` alone would also
    # accept None (missing/inconsistent evidence), which is exactly the
    # unknown-as-pass path the admission gate was added to close.  Legacy
    # manifests without the disclosure keep the historical wording.
    _param_required = bool(_manifest_requires_params(manifest))

    # Matched initialization is a VALIDITY precondition of the paired test,
    # not an extra effect-size threshold: a pair whose two arms carry
    # different shared-trunk digests is dropped above, and a comparison that
    # still shows a definite mismatch (e.g. a legacy manifest without
    # complete coverage) cannot be called confirmed.  `None` (unverified
    # legacy evidence) keeps the historical permissive path, but it is never
    # printed as equality.
    def _init_gate(k):
        return stats['comparisons'].get(k, {}).get('same_init_core_fp') is not False

    def _param_gate(k):
        if not _param_required:
            return True
        return stats['comparisons'].get(k, {}).get('param_match_complete') is True

    _conf = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is True
             and stats['comparisons'].get(k, {}).get('param_matched') is not False
             and _param_gate(k)
             and _init_gate(k)
             and int(stats['comparisons'].get(k, {}).get('n_effective') or 0) >= int(stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))
             and (not _require_cov or stats['comparisons'].get(k, {}).get('coverage_ok') is True)]
    stats['coverage_requirement'] = {
        'require_complete_coverage': bool(_require_cov),
        'n_declared_seeds': len(manifest['seeds']),
        'note': ('complete coverage (every frozen seed) is required for a CONFIRMED verdict'
                 if _require_cov else
                 'legacy manifest: a complete-case subset with n_effective >= the declared floor may be confirmed'),
    }
    # A two-sided test with a pre-registered direction has THREE admissible
    # outcomes, not two: support, refutation (a Holm-significant result in the
    # opposite direction), and inconclusive.  "Otherwise exploratory" is the
    # wrong reading of a significant reversal.  The verdict helper partitions
    # the SAME Holm rejections under the SAME validity gates; it adds no test,
    # no alpha and no multiplicity correction.  `confirmed` is kept exactly as
    # the frozen rule computed it; the helper's `supported` set is cross-checked
    # against it rather than allowed to silently redefine the decision.
    _verdict_in = {}
    for _c in manifest['primary_comparisons']:
        _k = str(_c.get('name'))
        _cv = dict(stats['comparisons'].get(_k) or {})
        _cv['p_holm'] = (stats['holm'].get(_k) or {}).get('p_holm')
        if 'expected_sign' not in _cv:
            _cv['expected_sign'] = _c.get('expected_sign')
        if 'mechanism_claim_allowed' not in _cv:
            _cv['mechanism_claim_allowed'] = _c.get('mechanism_claim_allowed')
        _verdict_in[_k] = _cv
    _verdict_block = L.directional_claim_verdicts(
        _verdict_in,
        names=[str(_c.get('name')) for _c in manifest['primary_comparisons']],
        alpha=_alpha, require_complete_coverage=bool(_require_cov),
        require_param_match=_param_required,
        default_min_seeds=len(manifest['seeds']))
    stats['verdicts'] = _verdict_block['verdicts']
    stats['verdict_summary'] = {
        k: _verdict_block[k] for k in
        ('alpha', 'supported', 'refuted', 'inconclusive',
         'significant_nondirectional', 'not_testable',
         'opposite_nonsignificant', 'reverse_significant_unverified',
         'counts', 'rule', 'role')}
    stats['confirmed'] = _conf
    stats['refuted'] = list(_verdict_block['refuted'])
    stats['inconclusive'] = list(_verdict_block['inconclusive'])
    stats['not_testable'] = list(_verdict_block['not_testable'])
    stats['opposite_nonsignificant'] = list(_verdict_block['opposite_nonsignificant'])
    stats['reverse_significant_unverified'] = list(_verdict_block['reverse_significant_unverified'])
    if set(_conf) != set(_verdict_block['supported']):
        print('[p5t stats] WARNING: the frozen confirmation set and the three-way '
              f'supported set disagree (confirmed={sorted(_conf)}, '
              f'supported={sorted(_verdict_block["supported"])}); the frozen '
              'decision is left untouched and the mismatch is recorded.')
    stats['verdict_consistency'] = {
        'frozen_confirmed': sorted(_conf),
        'supported': sorted(_verdict_block['supported']),
        'match': set(_conf) == set(_verdict_block['supported']),
        'note': ('the frozen decision rule remains authoritative; a mismatch is '
                 'recorded here rather than silently reconciled'),
    }
    # ---- post-hoc robustness: never part of the frozen decision -----------
    # A frozen verdict can be statistically valid and still hinge on one seed.
    # The diagnostic takes the SAME per-seed deltas that produced the primary
    # p-values and asks how many seeds the verdict can lose before the raw
    # test / the whole Holm family changes.  It is written next to the
    # decision, not into it: `confirmed` above is untouched.
    _rob_pairs = {}
    for _k, _v in stats['comparisons'].items():
        _dl = _v.get('per_seed')
        _sd = _v.get('seeds')
        if (isinstance(_dl, list) and isinstance(_sd, list)
                and len(_dl) == len(_sd) and len(_dl) >= 1
                and 'omitted' not in _v):
            _rob_pairs[_k] = {'deltas': _dl, 'seeds': _sd}
    _confirmation_robustness_block(
        stats, _rob_pairs, alpha=_alpha,
        family_size=len(manifest['primary_comparisons']),
        param_required=_param_required,
        family_names=[str(c.get('name')) for c in manifest['primary_comparisons']])
    stats['significant_nondirectional'] = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is None]
    stats['not_confirmed_reasons'] = {k: {'param_matched': stats['comparisons'].get(k, {}).get('param_matched'),
                                          'param_match_complete': stats['comparisons'].get(k, {}).get('param_match_complete'),
                                          'param_match_reason': stats['comparisons'].get(k, {}).get('param_match_reason'),
                                          'coverage_ok': stats['comparisons'].get(k, {}).get('coverage_ok'),
                                          'coverage_required': bool(_require_cov),
                                          'n_effective': stats['comparisons'].get(k, {}).get('n_effective'),
                                          'min_seeds_for_confirmation': stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation'),
                                          'same_init_core_fp': stats['comparisons'].get(k, {}).get('same_init_core_fp'),
                                          'init_core_fp_mismatch_seeds': (stats['comparisons'].get(k, {}).get('init_core_fp') or {}).get('mismatch_seeds'),
                                          'direction_ok': stats['comparisons'].get(k, {}).get('direction_ok'),
                                          'verdict': (stats['verdicts'].get(k) or {}).get('verdict'),
                                          'verdict_reason': (stats['verdicts'].get(k) or {}).get('reason'),
                                          'gate_failures': (stats['verdicts'].get(k) or {}).get('gate_failures')}
                                      for k in _rej if k not in _conf}
    # ---- held-out access accounting ---------------------------------------
    # The freeze lock has always counted test reads and recorded re-freezes,
    # but this analysis never consumed the counter.  A panel evaluated twice
    # (or re-frozen after its first read) could therefore keep a non-empty
    # `confirmed` set at nominal alpha.  The access audit exports
    # `confirmed_effective`; it never changes the frozen p/Holm or the
    # historical `confirmed` field.  A pre-declared multi-look budget applies
    # its Bonferroni spending against the PLANNED look count, not the observed
    # one, so an early stop cannot buy back alpha.
    if unified_fp is not None and admitted:
        _ta = L.confirmatory_access_audit(
            _freeze_lock_path(_lock_dir), stats=stats, manifest=manifest,
            alpha=_alpha,
            lock_label=os.path.join(TEST_OUTDIR, '.freeze_lock.json'))
        stats['test_access'] = _ta
        _ta_eff = _ta.get('confirmed_effective')
        stats['confirmed_effective'] = (
            list(_ta_eff) if _ta_eff is not None else None)
        if _ta.get('confirmatory_allowed') is False:
            _why = (_ta.get('downgrade_reasons')
                    or _ta.get('problems') or ['unknown'])
            print('[p5t stats] WARNING: the held-out access ledger cannot '
                  f'support a confirmatory claim ({", ".join(str(x) for x in _why)}); '
                  '`confirmed` remains the frozen arithmetic set, but '
                  '`confirmed_effective` is empty. See stats["test_access"].')
        elif (_ta.get('policy') or {}).get('declared'):
            print('[p5t stats] held-out access: '
                  f'{_ta.get("n_test_reads")} read(s) under a pre-declared '
                  f'{(_ta.get("policy") or {}).get("method")} budget; '
                  f'alpha_effective={_ta.get("alpha_effective")!r}, '
                  f'confirmed_effective={stats["confirmed_effective"]}')
    _m = len(manifest['primary_comparisons'])
    _n = len(manifest['seeds'])
    _fl = L.signflip_p_floor(_n)
    if math.isfinite(_fl):
        _min_holm = L.holm_min_attainable_p(_n, _m, _alpha)
        stats['min_attainable_p_holm'] = _min_holm
        stats['power_first_holm_gate'] = {f'q={q:.2f}': L.signflip_power(_n, _m, q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        if _min_holm > _alpha:
            # unreachable in practice: _load_manifest refuses such plans; keep
            # the belt-and-braces diagnostic in case a plan bypasses the gate.
            print(f'[p5t stats] WARNING: with n={_n} seeds and a Holm family of m={_m}, the smallest attainable Holm p is {_min_holm:.4g} > alpha={_alpha} — NO comparison can be confirmed regardless of the data. The load-time power gate should have refused this manifest; re-check which manifest/records produced this state.')
    if unified_fp is not None and admitted:
        # Read the recorded evaluation units from whichever frozen endpoint
        # carries them; smoking/alternate manifests need not contain L=4096.
        # A real article count is preferred when every admissible cell carries
        # it (the independent corpus unit); otherwise the line-level document
        # count; otherwise the row count.  Mixed article coverage fails closed
        # for a population claim rather than borrowing the larger line count.
        _n_art, _n_doc, _n_row = 0, 0, 0
        _art_cells = _doc_cells = _row_cells = 0
        _doc_len = None
        for _r in admitted.values():
            for _k, _c in (_r.get('by_len') or {}).items():
                if not isinstance(_c, dict):
                    continue
                _has_a = (isinstance(_c.get('ppl_per_article'), list)
                          and bool(_c['ppl_per_article']))
                _has_d = (isinstance(_c.get('ppl_per_doc'), list)
                          and bool(_c['ppl_per_doc']))
                if _has_a:
                    _art_cells += 1
                    if len(_c['ppl_per_article']) > _n_art:
                        _n_art = len(_c['ppl_per_article'])
                        _doc_len = _k
                if _has_d:
                    _doc_cells += 1
                    if len(_c['ppl_per_doc']) > _n_doc:
                        _n_doc = len(_c['ppl_per_doc'])
                        if not _has_a:
                            _doc_len = _k
                elif (isinstance(_c.get('ppl_per_seq'), list)
                      and _c['ppl_per_seq']):
                    _row_cells += 1
                    _n_row = max(_n_row, len(_c['ppl_per_seq']))
                    if not _has_a and not _has_d:
                        _doc_len = _k
        _mixed_article = bool(_art_cells and _doc_cells
                              and _art_cells != _doc_cells)
        if _art_cells:
            _gran = 'article'
            _n_units = _n_art
        elif _doc_cells:
            _gran = 'document'
            _n_units = _n_doc
        elif _row_cells:
            _gran = 'row'
            _n_units = _n_row
        else:
            _n_units = 0
            _gran = 'row'
        if _n_units:
            stats['doc_sampling_granularity'] = _gran
            stats['doc_sampling_unit'] = _gran
            stats['doc_sampling_endpoint'] = _doc_len
            stats['doc_sampling_n_units'] = int(_n_units)
            stats['doc_sampling_mixed_article_coverage'] = bool(_mixed_article)
            stats['population_claim_allowed'] = bool(
                _n_units >= L.MIN_DOCS_FOR_POPULATION_CLAIM
                and not _mixed_article)
            if (not stats['population_claim_allowed']):
                _unit = {'article': 'articles',
                         'document': 'lines (the cache calls them documents)',
                         'row': 'fixed eval rows'}[_gran]
                if _mixed_article:
                    stats['doc_sampling_note'] = (
                        'population-level claims are refused because article '
                        'coverage is mixed across admissible cells; the '
                        'larger line-level count is not substituted for the '
                        'independent unit (design_notes sections 20/43/54)')
                else:
                    stats['doc_sampling_note'] = (
                        f'{_unit} recorded (n={_n_units}); between-unit '
                        'variance is reported per comparison in '
                        f'doc_sampling — n<{L.MIN_DOCS_FOR_POPULATION_CLAIM} '
                        'does not cover the language-population distribution, '
                        'so population-level claims are refused; the '
                        'confirmed claims are about the frozen evaluation '
                        'slice (design_notes sections 20/43/54)'
                        + ('' if _gran == 'article' else
                           '; the article unit is the independent corpus '
                           'unit — report it when the cache/tokenizer can '
                           'supply it'))
    print(f'[p5t stats] admitted {len(admitted)}/{len(manifest["arms"]) * _n} frozen (arm, seed) cells on slice {str(unified_fp)[:12]}…; {len(_fam)} frozen primary comparison(s); {len(_rej)} survive Holm; {len(_conf)} also match the pre-registered direction')
    _rob_s = ((stats.get('robustness') or {}).get('summary') or {})
    if _rob_s:
        print(f'[p5t stats] robustness (post-hoc, does NOT change the verdict): '
              f'raw-significant={_rob_s.get("n_raw_significant")}/{_rob_s.get("n_comparisons")}; '
              f'Holm rejections stable under every one-seed drop='
              f'{_rob_s.get("all_holm_rejections_stable_under_one_seed_drop")}; '
              f'min drop/flip fragility={_rob_s.get("min_drop_fragility")}/'
              f'{_rob_s.get("min_flip_fragility")}; unstable seeds='
              f'{_rob_s.get("unstable_seeds")}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    _analysis_inputs = {
        'manifest': manifest.get('_path') or FREEZE_MANIFEST,
        'bylen': bylen_path,
    }
    if unified_fp is not None and admitted:
        # The access ledger is an input to the access verdict: bind its bytes
        # so editing test_reads/superseded after the analysis invalidates the
        # stamped stats artifact as well as the replay gate.
        _analysis_inputs['freeze_lock'] = _freeze_lock_path(_lock_dir)
    if os.path.exists(probe_path):
        _analysis_inputs['probe'] = probe_path
    stats = L.stamped_analysis(
        'finaltest_stats', stats, outdir=(os.path.dirname(out) or '.'),
        name=os.path.basename(out), inputs=_analysis_inputs)
    print(f'[p5t stats] wrote {out}')
    return stats

def _by_len_cell(rec, Ln):
    """The by-length cell for one record/length, else None."""
    bl = (rec or {}).get('by_len') or {}
    c = bl.get(Ln, bl.get(str(Ln)))
    return c if isinstance(c, dict) else None


def _ppl_at_len(rec, Ln):
    """Usable, untruncated PPL for one (record, length), else None."""
    c = _by_len_cell(rec, Ln)
    if (c is not None and c.get('truncated') is False
            and L.ppl_is_usable(c.get('ppl'))):
        return float(c['ppl'])
    return None


def _paired_len_deltas(recs_a, recs_b, common, Ln):
    dl, used = [], []
    for s in common:
        pa, pb = _ppl_at_len(recs_a.get(s), Ln), _ppl_at_len(recs_b.get(s), Ln)
        if pa is None or pb is None:
            continue
        dl.append(pa - pb)
        used.append(int(s))
    return dl, used


def _signflip_power_table(abs_d, m_family, alpha, qs, expected_sign):
    """Exact conditional power of the sign-flip test for fixed |d_i|.

    Enumerates all 2**n sign patterns of the observed validation deltas.  The
    two-sided p of a candidate pattern is its rank in that same sign-flip null;
    Holm is screened conservatively as min(1, m*p).  Power at q is the
    q**k (1-q)**(n-k) mass of patterns that reject *and* point in the
    pre-registered direction.  This is the simulation the manifest asks for
    before the one-shot test.

    The rank of every pattern is obtained by sorting the 2**n null sums once
    (O(2**n log 2**n)) instead of building the all-pairs comparison matrix
    (O(4**n) time and memory), so the exact table remains usable for the
    16-seed dynamic panel and remains bounded at n<=20.  Larger panels refuse
    explicitly rather than silently exhausting memory; the exact sign-flip
    analysis itself switches to meet-in-the-middle/saddlepoint above that
    point, but conditional power still needs every sign pattern.
    """
    abs_d = np.asarray([abs(float(x)) for x in abs_d], dtype=float)
    n = int(abs_d.size)
    if n < 2:
        raise ValueError('power needs at least two paired deltas')
    if n > 20:
        raise ValueError(f'power enumeration is capped at n<=20 (got {n}); split the panel or add a Monte-Carlo path consciously')
    # The conditional power table is a property of the sign-flip ranks, so a
    # units-only rescaling of the observed |deltas| must not change it.  The
    # null sums below are compared with an absolute 1e-12 tie tolerance; left
    # unscaled, a small-but-representable validation panel (e.g. 1e-13
    # deltas) makes every pattern look tied while a large one can overflow the
    # sums.  Work on a unit-scaled copy, exactly as `exact_signflip_p` does;
    # the tolerance is then relative to the panel.
    _scale = float(np.max(abs_d)) if n else 0.0
    if math.isfinite(_scale) and _scale > 0.0:
        abs_d = abs_d / _scale
    total = 1 << n
    bits = (1 << np.arange(n, dtype=np.int64))[None, :]
    _idx = np.arange(total, dtype=np.int64)
    try:
        k_plus = np.bitwise_count(_idx)
    except AttributeError:
        k_plus = np.fromiter((int(i).bit_count() for i in _idx),
                             dtype=np.int16, count=total)
    abs_sums = np.empty(total, dtype=float)
    k_power = np.empty(total, dtype=np.int16)
    k_same = np.empty(total, dtype=np.int16)
    sign_ok = np.empty(total, dtype=bool)
    step = max(1, min(total, (1 << 18) // max(n, 1)))
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & bits) != 0, 1.0, -1.0)
        sums = signs @ abs_d
        abs_sums[off:off + bs] = np.abs(sums)
        _kp = k_plus[off:off + bs]
        if expected_sign == 'negative':
            sign_ok[off:off + bs] = sums < 0
            k_power[off:off + bs] = n - _kp
            k_same[off:off + bs] = n - _kp
        elif expected_sign == 'positive':
            sign_ok[off:off + bs] = sums > 0
            k_power[off:off + bs] = _kp
            k_same[off:off + bs] = _kp
        else:
            sign_ok[off:off + bs] = True
            k_power[off:off + bs] = _kp
            k_same[off:off + bs] = np.maximum(_kp, n - _kp)
    sorted_abs = np.sort(abs_sums)
    p = (total - np.searchsorted(sorted_abs, abs_sums - 1e-12,
                                 side='left')) / float(total)
    p_holm = np.minimum(1.0, p * float(m_family))
    power = {}
    for q in qs:
        q = float(q)
        prob = (q ** k_power) * ((1.0 - q) ** (n - k_power))
        power[f'q={q:.2f}'] = float((prob * ((p_holm < alpha) & sign_ok)).sum())
    min_same = None
    for s in range(n + 1):
        sel = (k_same == s) & sign_ok
        if sel.any() and float(p_holm[sel].min()) < alpha:
            min_same = int(s)
            break
    return {'n': n, 'enumerated': total, 'm_family': int(m_family),
            'alpha': float(alpha), 'expected_sign': expected_sign,
            'power': power, 'min_same_sign_for_alpha': min_same,
            'min_attainable_p_holm': float(p_holm.min()),
            'note': ('exact enumeration over the 2**n sign patterns of the observed |d_i|; '
                     'Holm screened as min(1, m*p); power conditions on these magnitudes, '
                     'not on an assumed effect size')}


def run_power(manifest_path=FREEZE_MANIFEST, summary='results_v10_mech/summary.json',
              out='analysis_v12/power_validation.json'):
    """Zero-GPU pre-test power calibration on the P5C validation deltas.

    The manifest requires this BEFORE the one-shot test access: with the
    observed per-seed |delta| vector, how much probability mass actually lies
    in the reject region at the pre-registered direction probability q?  A
    plan whose power at `statistics.power_q_for_test` is below
    `statistics.min_power_for_test` must add seeds and record a revision
    *before* P5T (V12_ALLOW_LOW_POWER=1 is the explicit override).  Each
    directional cell also carries an attrition stress table: under the frozen
    coverage rule, a missing pair can make the comparison unconfirmed even
    when the conditional power is adequate.  Reads only the validation
    summary; never calls load_wikitest.
    """
    manifest = _load_manifest(manifest_path)
    if not os.path.exists(summary):
        raise SystemExit(f'[p5pw] REFUSE: {summary} does not exist — run P5C/P3MT first so the power '
                         'preflight has validation deltas; the test split must not be read to size the panel')
    raw = json.load(open(summary, encoding='utf-8'))
    if not isinstance(raw, dict):
        raise SystemExit(f'[p5pw] REFUSE: {summary} is not a JSON object')
    stat = manifest.get('statistics') or {}
    alpha = float(stat.get('alpha', 0.05))
    comps = list(manifest['primary_comparisons'])
    m_family = len(comps)
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    power_q = float(stat.get('power_q_for_test', 0.90))
    if f'q={power_q:.2f}' not in {f'q={q:.2f}' for q in qs}:
        qs.append(power_q)
    min_power = float(stat.get('min_power_for_test', 0.80))
    require = bool(stat.get('require_power_gate'))
    _eval_design = manifest.get('evaluation_design')
    if not isinstance(_eval_design, dict):
        _eval_design = manifest.get('evaluation')
    if not isinstance(_eval_design, dict):
        _eval_design = {}
    _min_units = _eval_design.get('min_units_for_population')
    if _min_units is None:
        _min_units = stat.get(
            'min_units_for_population', L.MIN_DOCS_FOR_POPULATION_CLAIM)
    _attr_cfg = (stat.get('attrition')
                 if isinstance(stat.get('attrition'), dict) else {})
    report = {'kind': 'power_validation', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': manifest_path, 'manifest_sha256': manifest.get('manifest_sha256'),
              'summary': summary, 'alpha': alpha, 'm_family': m_family,
              'power_q_for_test': power_q, 'min_power_for_test': min_power,
              'require_power_gate': require, 'comparisons': {}}
    _seed_whitelist = {int(s) for s in manifest['seeds']}
    all_pass = True
    for comp in comps:
        name = comp.get('name')
        a, b = comp.get('a'), comp.get('b')
        Ln = int(str(comp.get('endpoint', 'by_len.ppl@4096')).rsplit('@', 1)[-1])
        exp, _exp_err = L._design_direction(comp.get('expected_sign'))
        exp = {1: 'positive', -1: 'negative', None: None}[exp]
        recs_a, recs_b = {}, {}
        skipped = 0
        for key, rec in raw.items():
            if not isinstance(rec, dict) or not isinstance(key, str) or '::' not in key:
                continue
            v, _sep, s = key.partition('::')
            if v not in (a, b):
                continue
            if not L.result_is_current(rec, V.CKPT_CODE, 'by_len'):
                skipped += 1
                continue
            try:
                seed = int(str(s).replace('seed', '').replace('s', ''))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if seed not in _seed_whitelist:
                # The frozen sample size/seed set is part of the
                # pre-registration; an extra legacy seed must not enlarge the
                # power panel (and a duplicate key spelling must not silently
                # replace an existing record).
                skipped += 1
                continue
            _dst = recs_a if v == a else recs_b
            if seed in _dst:
                skipped += 1
                continue
            _dst[seed] = rec
        common = sorted(set(recs_a) & set(recs_b))
        dl, used = _paired_len_deltas(recs_a, recs_b, common, Ln)
        cell = {'endpoint': f'by_len.ppl@{Ln}', 'n': len(dl), 'seeds': used,
                'skipped_stale_records': skipped}
        _sesoi = comp.get('sesoi')
        if _sesoi is None:
            _sesoi = stat.get('sesoi')
        _require_sesoi = (comp['require_sesoi_power']
                          if 'require_sesoi_power' in comp
                          else stat.get('require_sesoi_power', False))
        _sens = L.paired_design_sensitivity(
            len(manifest['seeds']), m_family, alpha=alpha,
            target_power=min_power,
            pilot_deltas=(dl if len(dl) >= 2 else None),
            expected_sign=exp, sesoi=_sesoi,
            require_sesoi_power=_require_sesoi)
        cell['sensitivity'] = _sens
        if _sens.get('status') == 'fail':
            all_pass = False
        if exp in ('positive', 'negative'):
            _q_attr = (_sens.get('pilot') or {}).get('q_sign_rate')
            _q_source = 'pilot_sign_rate'
            if _q_attr is None:
                _q_attr = power_q
                _q_source = 'statistics.power_q_for_test'
            cell['attrition'] = L.attrition_power_audit(
                len(manifest['seeds']), m_family, q=_q_attr,
                pair_retention=_attr_cfg.get('pair_retention'),
                alpha=alpha, target_power=min_power,
                require_complete_coverage=stat.get(
                    'require_complete_coverage', False),
                min_complete_pairs=stat.get('min_seeds_for_confirmation'),
                availability_target=_attr_cfg.get('availability_target'),
                require_attrition_power=_attr_cfg.get(
                    'require_attrition_power', False))
            cell['attrition']['q_source'] = _q_source
            if cell['attrition'].get('status') == 'fail':
                all_pass = False
        else:
            cell['attrition'] = {
                'status': 'not_applicable',
                'reason': ('a directional q is required for a sign-flip '
                           'attrition projection'),
            }
        _unit_pairs = [(s, _by_len_cell(recs_a.get(s), Ln),
                        _by_len_cell(recs_b.get(s), Ln)) for s in used]
        _unit_pairs = [(c_a, c_b) for _s, c_a, c_b in _unit_pairs
                       if c_a is not None and c_b is not None]
        _unit = L.paired_unit_deltas([c_a for c_a, _ in _unit_pairs],
                                     [c_b for _, c_b in _unit_pairs])
        cell['evaluation_unit'] = _unit
        if _unit.get('status') == 'ok':
            # The pilot units come from validation, but the precision design
            # must use the FROZEN test-slice unit count.  Passing the pilot
            # count here silently sized the test panel from the validation
            # panel (and could allow a population claim the test slice cannot
            # support).
            _frozen_units = L.evaluation_design_units(manifest)
            _planned_n = _frozen_units['raw_n_units']
            _planned_label = _frozen_units['unit_label']
            if _planned_n is not None and not _frozen_units['label_declared']:
                _planned_label = _unit['unit']
            _up = L.unit_precision_design(
                _planned_n, pilot_unit_deltas=_unit['deltas'],
                unit_label=_planned_label, sesoi=_sesoi,
                target_power=min_power, alpha=alpha,
                require_sesoi_power=_require_sesoi,
                min_units_for_population=_min_units)
            _up['pilot_n_units'] = _unit['n_units']
            _up['pilot_unit'] = _unit['unit']
            if _planned_label != _unit['unit']:
                _up.setdefault('warnings', []).append(
                    f'frozen evaluation unit {_planned_label!r} differs from '
                    f'the pilot unit {_unit["unit"]!r}; the pilot SD is not a '
                    'validated estimate at the frozen granularity')
            cell['unit_deltas'] = list(_unit['deltas'])
            if _unit.get('ids') is not None:
                cell['unit_ids'] = list(_unit['ids'])
            cell['unit_precision'] = _up
            if _up.get('status') == 'fail':
                all_pass = False
        if len(dl) >= 2 and exp in ('positive', 'negative'):
            obs = L.exact_signflip_p(dl)
            mean = float(np.mean(dl))
            tab = _signflip_power_table(np.abs(dl), m_family, alpha, qs, exp)
            sign_ok = (mean < 0) if exp == 'negative' else (mean > 0)
            key = f'q={power_q:.2f}'
            cell.update({'deltas': [float(x) for x in dl], 'mean': mean,
                         'p_exact_signflip': float(obs['p']),
                         'p_holm_screen': float(min(1.0, obs['p'] * m_family)),
                         'observed_sign_ok': sign_ok, 'power': tab})
            cell['power_ok'] = bool(cell['power']['power'].get(key, 0.0) >= min_power
                                    and sign_ok)
            all_pass &= cell['power_ok']
        elif len(dl) >= 2:
            # A non-directional mechanism readout has no "probability of the
            # pre-registered direction"; it is reported descriptively and must
            # not block the directional panel's power gate.
            obs = L.exact_signflip_p(dl)
            cell.update({'deltas': [float(x) for x in dl], 'mean': float(np.mean(dl)),
                         'p_exact_signflip': float(obs['p']),
                         'p_holm_screen': float(min(1.0, obs['p'] * m_family)),
                         'observed_sign_ok': None, 'power': None, 'power_ok': True,
                         'note': 'non-directional readout; excluded from the directional power gate'})
        else:
            cell['power_ok'] = False
            cell['error'] = f'fewer than two complete validation pairs at L{Ln}'
            all_pass = False
        # In-distribution readout on the SAME validation records.  This is the
        # 512-token contrast the frozen test never makes primary; it is
        # descriptive (m=1) and makes a reversal visible before test.
        dl512, used512 = _paired_len_deltas(recs_a, recs_b, common, 512)
        if len(dl512) >= 2:
            o512 = L.exact_signflip_p(dl512)
            cell['in_distribution_L512'] = {
                'n': len(dl512), 'seeds': used512, 'mean': float(np.mean(dl512)),
                'p_exact_signflip': float(o512['p']),
                'expected_sign_ok': (None if exp not in ('positive', 'negative')
                                     else bool((np.mean(dl512) < 0) if exp == 'negative' else (np.mean(dl512) > 0))),
                'descriptive_only': True, 'm_family': 1,
                'note': 'validation, in-distribution descriptive readout; not part of the frozen test family'}
        report['comparisons'][name] = cell
    report['all_pass'] = bool(all_pass)
    report['status'] = 'pass' if all_pass else 'fail'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    L.stamped_analysis('power_validation', report, outdir=(os.path.dirname(out) or '.'),
                       name=os.path.basename(out),
                       inputs={'manifest': manifest.get('_path') or manifest_path,
                               'summary': summary})
    print(f'[p5pw] power preflight: {report["status"]} (q={power_q:.2f}, min={min_power:.2f}) -> {out}')
    for _k, _v in report['comparisons'].items():
        _pw = ((_v.get('power') or {}).get('power') or {}).get(f'q={power_q:.2f}')
        print(f'  [p5pw] {_k}: n={_v.get("n")} power={_pw if _pw is not None else "n/a"} '
              f'ok={_v.get("power_ok")}')
        _attr = _v.get('attrition') or {}
        if _attr.get('status') not in (None, 'not_applicable'):
            _min_p = _attr.get('min_pair_retention_for_target')
            print(f'  [p5pw] {_k}: attrition={_attr.get("status")} '
                  f'runnable={_attr.get("probability_runnable")} '
                  f'joint={_attr.get("unconditional_power")} '
                  f'min_pair_retention={_min_p if _min_p is not None else "n/a"}')
    allow = os.environ.get('V12_ALLOW_LOW_POWER', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if require and not all_pass and not allow:
        raise SystemExit(f'[p5pw] REFUSE: the pre-test power calibration is below {min_power:.2f} at q={power_q:.2f}; '
                         'add seeds and record the revision in confirmation_frozen.json BEFORE touching test '
                         '(V12_ALLOW_LOW_POWER=1 is the explicit override)')
    if require and not all_pass:
        print('[p5pw] WARNING: V12_ALLOW_LOW_POWER=1 — running an under-powered frozen panel; '
              'the report must say so')
    return report


def v12_analysis(out='analysis_v12/stats.json'):
    """Zero-GPU aggregate: the shared v10 analysis over the extended probe +
    the frozen final-test statistics (when they exist on disk)."""
    V10.v10_analysis(out=out)
    st = json.load(open(out, encoding='utf-8'))
    if os.path.exists(os.path.join(TEST_OUTDIR, 'bylen.json')):
        st['finaltest'] = finaltest_stats()
        # this file's LAST writer is v12_analysis: re-stamp so the trace names
        # the code that produced the final content, not the earlier v10 pass
        st = L.stamped_analysis('v12_analysis', st, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    return st

def _fmt_pm(cell):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        return f'{cell["mean"]:.2f} ±{cell.get("std", 0.0):.2f}'
    return f'{cell:.2f}'

def _refutation_markdown(ft, alpha=0.05):
    """Render the three-way directional verdicts that are NOT support.

    Pure function of the final-test stats dict so the wording and the
    "refutation is not equivalence / not the opposite mechanism" boundary can
    be tested without a GPU panel or a full report build.
    """
    if not isinstance(ft, dict):
        return []
    _alpha = alpha
    try:
        _alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        pass

    def _names(key):
        raw = ft.get(key)
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (list, tuple)):
            return []
        return [str(x) for x in raw]

    ref = _names('refuted')
    unverified = _names('reverse_significant_unverified')
    opposite = _names('opposite_nonsignificant')
    if not (ref or unverified or opposite):
        return []
    verdicts = ft.get('verdicts')
    if not isinstance(verdicts, dict):
        verdicts = {}

    def _detail(name):
        v = verdicts.get(name)
        if not isinstance(v, dict):
            return f'`{name}`'
        p = v.get('p_holm')
        ptxt = (f'{float(p):.4g}' if isinstance(p, (int, float))
                and not isinstance(p, (bool, np.bool_))
                and math.isfinite(float(p)) else '—')
        scope = v.get('claim_scope') or 'unspecified'
        return f'`{name}`（p_holm={ptxt}, claim_scope={scope}）'

    out = []
    A = out.append
    if ref:
        A(f'**反证结果（方向冲突）**：{"; ".join(_detail(k) for k in ref)} 在 Holm 校正后 '
          f'p<{_alpha:g}，但方向与预注册相反。这是对该**方向性主张**的反证/方向冲突，'
          '不是「未确认」，也不是「无效应」。')
        A('')
        A('> 边界：反证仍继承同一 Holm 族的 FWER 控制，但没有预先注册 SESOI 时，'
          '它只说明该方向性优势不成立，**不证明相反机制**，也不能写成「显著更差/实际等价」；'
          'system-level 比较（`mechanism_claim_allowed=false`）的反证只否定所测系统差异，'
          '不否定机制本身。冻结 p/Holm/alpha 与 `confirmed` 均未被本段改变。')
        A('')
    if unverified:
        A(f'**显著反向但有效性门未过**：{"; ".join(_detail(k) for k in unverified)} 的 '
          'Holm p 低于 alpha 且点估计方向相反，但覆盖/参数/初始化/样本下限至少一项未满足；'
          '因此只能记为 `inconclusive`，不能据此宣布反证。')
        A('')
    if opposite:
        A(f'**方向相反但未达显著**：{"; ".join(_detail(k) for k in opposite)} 的点估计与预注册方向相反，'
          '但 Holm 校正后不显著；这只是不确定（inconclusive），既不是支持也不是反证。')
        A('')
    return out


def build_report(out='REPORT_v12.md', stats_p='analysis_v12/stats.json'):
    v12_analysis(out=stats_p)
    st = json.load(open(stats_p, encoding='utf-8'))
    ft = st.get('finaltest') or {}
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v12 确认报告（冻结清单 + 独立 test split）')
    A('')
    A(f'> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    A('> 参考论文：arXiv:2606.19348。**范围声明**：小规模 WikiText-103 上的受控研究，不是 DeepSeek-V4 的全量复现；结论不外推到其规模、语料、长上下文训练或生产系统。')
    A('> 说明：本报告全部数字由 `python -m explib.confirmation report` 从 `results_lm_test/`、`results_v10_mech/`、`analysis_v12/` 的落盘产物计算得到，无手填数值。')
    A('')
    A('本驱动的确认性协议：v7→v11 的全部面板与 v10/v12 的机制扩展属于**探索性**证据；本报告只报告**冻结后**在 held-out test split（wiki.test.raw，其它 phase 不读）上的一次性确认评估。')
    A('')
    A('---')
    A('')
    A('## 冻结清单（预注册）')
    A('')
    m = ft.get('manifest') or {}
    if not m and os.path.exists(FREEZE_MANIFEST):
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            m = {'frozen_at': _mc.get('frozen_at'), 'manifest_sha256': None, 'arms': _mc.get('arms'), 'seeds': _mc.get('seeds'), 'code_semantics_expected': _mc.get('code_semantics_expected'), 'statistics': _mc.get('statistics'), 'scope': _mc.get('scope'), 'revision': _mc.get('revision')}
        except Exception:
            pass
    if m:
        _sha = m.get('manifest_sha256') or (_file_sha256(FREEZE_MANIFEST) if os.path.exists(FREEZE_MANIFEST) else None)
        A(f'- 冻结时间：{m.get("frozen_at", "?")}；清单 sha256：`{str(_sha)[:16]}…`（结果记录与冻结锁均绑定该哈希）；代码语义戳：`{m.get("code_semantics_expected", "?")}`（运行期校验一致才放行）。')
        A(f"- 冻结臂：{', '.join((f'`{v}`' for v in m.get('arms', [])))}；种子：{m.get('seeds')}。")
        _stat = m.get('statistics') or {}
        A(f"- 检验：{_stat.get('test', 'exact two-sided sign-flip')}；n={_stat.get('n_seeds', '?')}（p 值分辨率下限 {_stat.get('p_floor', float('nan'))}）；族 = primary_comparisons；校正 = {_stat.get('correction', 'holm step-down')}。")
        A(f"- 决策规则：{_stat.get('decision_rule', '—')}")
        _rev = m.get('revision') or {}
        if isinstance(_rev, dict) and _rev.get('history'):
            A(f"- 修订史：{'；'.join((f"{_r.get('date', '?')} {_r.get('change', '?')}" for _r in _rev['history']))}")
        _pw = ft.get('power_first_holm_gate') or {}
        if _pw:
            A(f"- 功效注记（第一道 Holm 门，逐种子同向概率 q）：{'；'.join((f"{k}→{v:.2f}" for k, v in _pw.items()))}——功效对 q 敏感，q 假设来源见清单 statistics.power_note；数学上可拒绝 ≠ 已足功效。")
        _scope = m.get('scope') or {}
        if isinstance(_scope, dict) and _scope:
            A(f"- 范围声明：{_scope.get('confirmatory_claims', '')} {_scope.get('not_covered', '')} {_scope.get('required_for', '')}")
        A('')
        A('| 冻结的主比较 |')
        A('|---|')
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            for c in _mc.get('primary_comparisons', []):
                A(f"| {c.get('name', '?')} — {c.get('a')} vs {c.get('b')} @ {c.get('endpoint', '?')} |")
        except Exception:
            A('| （冻结清单不可读——见运行日志） |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python -m explib.confirmation phase P5C`（覆盖全部六个冻结臂、种子取自清单），再跑 `python -m explib.confirmation phase P5T`（首次评估会写入冻结锁 `.freeze_lock.json`，此后清单不可再改）。')
        A('')
    A('---')
    A('')
    A('## P5T 独立最终测试（test split，冻结终点）')
    A('')
    # The report must use the SAME frozen alpha as finaltest_stats, not a
    # hard-coded 0.05 that can disagree with the decision rule.
    try:
        _report_alpha = float(ft.get('alpha', (m.get('statistics') or {}).get('alpha', 0.05)))
    except (TypeError, ValueError, OverflowError):
        _report_alpha = 0.05
    bl = ft.get('by_length') or {}
    if bl:
        _lens = sorted({int(k) for v in bl.values() for k in v})
        A('| 臂 | params | n_seq | ' + ' | '.join((f'PPL@{Ln}' for Ln in _lens)) + ' |')
        A('|---|---|---|' + '---|' * len(_lens))
        for v in sorted(bl):
            row = []
            _p0 = None
            _n0 = None
            for Ln in _lens:
                e = bl[v].get(str(Ln))
                if isinstance(e, dict):
                    if _p0 is None:
                        _p0 = e.get('params')
                    if _n0 is None:
                        _n0 = e.get('n_seqs')
                row.append(f'{e["mean"]:.2f} (n={e["n"]})' if e else '—')
            A('| `%s` | %s | %s | ' % (v, _p0 if _p0 is not None else '—', _n0 if _n0 is not None else '—') + ' | '.join(row) + ' |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python -m explib.confirmation phase P5C`（现覆盖全部六个冻结臂、种子数取自清单；主臂 0-5 已由 P3MT/P4MT 训练，resume 自动跳过），然后 `python -m explib.confirmation phase P5T`。冻结清单已入库（见上），随时可以执行；执行前请勿据任何 validation 读数调整冻结内容。')
        A('')
    comps = ft.get('comparisons') or {}
    if comps:
        A('| 冻结主比较（test split，配对） | n | 覆盖完整 | 参数匹配 | 初始化配对 | Δ mean±std | Δ 95% CI（文章/文档 token 池化 cluster；seed×unit 两层 bootstrap / 行 bootstrap） | d_z | p (exact) | p 下限 | Holm p | Holm 拒绝 | 方向符合 |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        holm = ft.get('holm') or {}
        for k, v in comps.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | — | — | — | —（{v["omitted"]}） | — | — | — | — | — | — | — |')
                continue
            _h = holm.get(k) or {}
            _hp = f'{_h.get("p_holm", float("nan")):.3f}' if _h.get('p_holm') is not None else '—'
            _hr = '✔' if (_h.get('p_holm') is not None and float(_h['p_holm']) < _report_alpha) else '✘'
            _dir = v.get('direction_ok')
            _ds = '—（非方向性）' if _dir is None else ('✔' if _dir else '✘')
            _ipa = v.get('same_init_core_fp')
            _ipa_txt = '—（未验证）' if _ipa is None else ('✔' if _ipa else '✘')
            _dsm = v.get('doc_sampling') or {}
            _dcitxt = _doc_ci_markdown_cell(_dsm)
            _dwin = _dsm.get('doc_a_better_fraction')
            _du = _dsm.get('unit') or _dsm.get('granularity')
            _dunit = {'article': '文章', 'document': '文档'}.get(_du, '行')
            if isinstance(_dwin, (int, float)) and _dcitxt != '—':
                _dcitxt += f'；{100.0 * _dwin:.0f}% {_dunit} Δ<0'
            A(f'| {k} | {v["n"]} | {v.get("coverage_ok")} | {v.get("param_matched")} | {_ipa_txt} | {v["mean"]:+.2f} ± {v.get("std", 0):.2f} | {_dcitxt} | {v.get("cohens_dz", float("nan")):+.2f} | {v["p_exact_signflip"]:.3f} | {v.get("p_floor", float("nan")):.3f} | {_hp} | {_hr} | {_ds} |')
        A('')
        _init_dropped = {k: [d for d in (v.get('dropped_pairs') or [])
                             if 'init_core_fp' in str(d.get('reason', ''))]
                         for k, v in comps.items()}
        _init_dropped = {k: ds for k, ds in _init_dropped.items() if ds}
        if _init_dropped:
            A('（初始化配对：' + '；'.join(
                f'{k} 丢弃种子 {[d["seed"] for d in ds]}'
                for k, ds in sorted(_init_dropped.items()))
              + '。这些种子的两臂 shared-trunk 初始化摘要不同，已按「不是配对」剔除，'
                '不进入精确符号翻转检验；覆盖门因此会拒绝把它们当作完整面板。）')
            A('')
        if any(isinstance((_v.get('doc_sampling') or {}).get('pooled_delta_ppl_ci95'),
                          (list, tuple))
               for _v in comps.values()):
            A('（CI 口径：优先按**文章**（顶层 ` = Title = ` 分组）聚类的 token 池化 Δ '
              'cluster-robust 区间；没有文章映射时退回按行级「文档」聚类。WikiText 的 '
              'raw row 是一行，不是一篇文章，行级 bootstrap 会把同一文章的段落当作独立样本'
              '（伪重复），因此文章级区间是主口径，行列字段仅作冻结契约的兼容证据。'
              '点估计与冻结比较的 token 池化 Δ 相同；同表 `Δ mean±std` 的离散度是种子间'
              '训练随机性，两者不确定性来源不同。cluster 区间**条件在已训练种子上**，'
              '只覆盖评测单元抽样；紧跟其后的 seed×unit 两层 bootstrap 同时重采样种子与'
              '单元，是总不确定性的描述性读数（不是新的检验，不进入 p/Holm/confirmed，'
              '种子数很少时百分位区间可能偏窄）。等权逐单元 bootstrap 仍保存在 '
              '`analysis_v12/test_stats.json` 的 `per_doc_delta_ppl_ci95`，只作描述，'
              '不再占据主表 CI 列。）')
            A('')
        _ta = (ft.get('test_access')
               if isinstance(ft.get('test_access'), dict) else {})
        _ta_blocked = _ta.get('confirmatory_allowed') is False
        if _ta_blocked:
            _ta_policy = (_ta.get('policy')
                          if isinstance(_ta.get('policy'), dict) else {})
            _ta_reasons = ', '.join(
                str(x) for x in (_ta.get('downgrade_reasons') or []))
            _ta_probs = '; '.join(str(x) for x in (_ta.get('problems') or []))
            A('**测试访问降级（held-out split）**：冻结锁记录 '
              f'`test_reads={_ta.get("n_test_reads")}`，预注册访问策略为 '
              f'`{_ta_policy.get("method", "unknown")}`（max_looks='
              f'`{_ta_policy.get("max_looks", "?")}`），降级原因：'
              f'{_ta_reasons or "冻结锁结构/工件不一致"}。'
              + (f' 结构问题：{_ta_probs}。' if _ta_probs else '')
              + ' 冻结 p/Holm 与 `confirmed` 字段没有被改写，但它只是按冻结公式'
              '算出的算术集合；`confirmed_effective` 为空。重复读取同一 test 切片'
              '（换 checkpoint、代码或分析口径后再看）是选择机会，不能按名义 α '
              '当作确认性证据；该读数只能标为探索性，或在**首次访问前**预注册'
              '多次 look 的 α 分配后重跑。')
            A('')
        _conf = (ft.get('confirmed_effective')
                 if 'confirmed_effective' in ft else ft.get('confirmed'))
        if _conf is None and _ta.get('descriptive_only'):
            # A descriptive-only plan has no confirmatory claim to fall back
            # to; do not re-derive one from the frozen arithmetic table.
            _conf = []
        if _conf is None:
            _conf = [k for k, v in holm.items() if v.get('reject') and (comps.get(k, {}).get('direction_ok') is True) and (comps.get(k, {}).get('param_matched') is not False) and (comps.get(k, {}).get('same_init_core_fp') is not False) and (comps.get(k, {}).get('coverage_ok') is True)]
        if _conf:
            A(f'**确认结果**：{len(_conf)}/{len(comps)} 个冻结主比较在 Holm 校正后成立**且方向符合预注册**（{sorted(_conf)}）——这是预注册规则下的确认性证据。')
            _frag_conf = []
            for _k in _conf:
                _dsm = comps.get(_k, {}).get('doc_sampling') or {}
                _ui = _dsm.get('unit_influence') or {}
                if _ui.get('fragile_to_one_unit') is True:
                    _u = {'article': '文章', 'document': '文档'}.get(
                        _dsm.get('unit') or _dsm.get('granularity'), '单元')
                    _frag_conf.append(f'{_k}（最敏感{_u} `{_ui.get("most_influential_unit")}`）')
            if _frag_conf:
                A('')
                A('**单单元敏感度提示**：' + '；'.join(_frag_conf) +
                  ' 的确认方向依赖删除单个评测单位；确认判定不变，但引用时必须同时报告'
                  '下方「单元影响诊断」，不能把结论描述为对评测文档稳健。')
            _choice_conf = []
            for _k in _conf:
                _cmp = comps.get(_k)
                _ds = _cmp.get('doc_sampling') if isinstance(_cmp, dict) else None
                _cc = (_ds.get('analysis_choice_concordance')
                       if isinstance(_ds, dict) else None)
                if isinstance(_cc, dict) and _cc.get('fragile') is True:
                    _choice_conf.append(
                        f'{_k}（{_cc.get("status")}）')
            if _choice_conf:
                A('')
                A('**分析口径敏感度提示**：' + '；'.join(_choice_conf) +
                  ' 的确认方向或精度结论依赖分析口径；确认判定不变，但引用时必须'
                  '同时报告下方「分析口径一致性」表，不能只报单一聚合。')
        else:
            _cov_req = bool((ft.get('coverage_requirement') or {}).get('require_complete_coverage'))
            _sig = [k for k, _h in holm.items()
                    if _h.get('p_holm') is not None and float(_h['p_holm']) < _report_alpha
                    and comps.get(k, {}).get('direction_ok') is True
                    and comps.get(k, {}).get('param_matched') is not False
                    and comps.get(k, {}).get('same_init_core_fp') is not False]
            _cov_blocked = [k for k in _sig if comps.get(k, {}).get('coverage_ok') is False]
            if _cov_req and _cov_blocked:
                A(f'**确认结果**：{sorted(_cov_blocked)} 达到「Holm p < {_report_alpha:g}」且方向符合，'
                  '但至少一个冻结种子没有进入该比较。按 v4 的 `require_complete_coverage`，缺失种子不是随机缺失，'
                  '不确认；该面板只能按实际 n 报告为 descriptive/unconfirmed。')
            else:
                A(f'**确认结果**：没有冻结主比较同时满足「Holm 校正后 p < {_report_alpha:g}」与「方向符合预注册」。按预注册的决策规则，相应主张**不获确认**；探索性面板的正向读数保持「探索性」标签，不得写成结论。若某比较的 Holm 显著方向与预注册相反，下面单列为反证/方向冲突，不能改写成「未确认」「无效应」或「探索性」。')
        _rob_lines = _robustness_markdown(ft.get('robustness') or {})
        if _rob_lines:
            lines.extend(_rob_lines)
        _doc_inf_lines = _doc_influence_markdown(ft)
        if _doc_inf_lines:
            lines.extend(_doc_inf_lines)
        _var_lines = _variance_components_markdown(ft)
        if _var_lines:
            lines.extend(_var_lines)
        _choice_lines = _analysis_choice_markdown(ft)
        if _choice_lines:
            lines.extend(_choice_lines)
        if _ta_blocked:
            # A refutation is a directional confirmatory statement too.  When
            # the access ledger is downgraded, the frozen `refuted` set cannot
            # be presented as a confirmatory reversal either.
            _ref_lines = []
        else:
            _ref_lines = _refutation_markdown(ft, _report_alpha)
        if _ref_lines:
            lines.extend(_ref_lines)
        _nd = ft.get('significant_nondirectional') or []
        if _nd:
            A('')
            A(f'**非方向性机制读数**：{sorted(_nd)} 在 Holm 校正后达到 p<{_report_alpha:g}，但冻结清单未给方向（expected_sign=null），只能作为机制读数，不能计入「方向符合预注册」的确认集合。')
        if ft.get('min_attainable_p_holm') is not None:
            A('')
            A(f"（诊断：n 个种子与 m 元 Holm 族下，最小可达 Holm p = {ft['min_attainable_p_holm']:.4f}。加载期功效门槛已拒绝「数学上不可能确认」的清单；若此处仍 > {_report_alpha:g}，说明有绕过门槛的旧状态，需回查清单与记录的来源。）")
        _dsn = ft.get('doc_sampling_note')
        if _dsn:
            A('')
            A(f"（文档抽样：{_dsn}）")
        A('')
        # Full by-length curve on the same admitted records.  Only the frozen
        # endpoint is confirmatory; L512 is the training length and makes an
        # in-distribution reversal visible instead of hiding it behind L4096.
        _lens_all = sorted({int(x) for _v in comps.values()
                            for x in ((_v.get('by_length_curve') or {}).keys())})
        if _lens_all:
            A('### 全长度曲线（同一批 test 记录；只有冻结终点属于确认族）')
            A('')
            A('| 比较 | ' + ' | '.join(f'L{x}' for x in _lens_all) + ' |')
            A('|---|' + '---|' * len(_lens_all))
            for _k, _v in comps.items():
                _curve = _v.get('by_length_curve') or {}
                _row = []
                for _x in _lens_all:
                    _c = _curve.get(str(_x))
                    if not _c:
                        _row.append('—')
                    else:
                        _row.append(f'{_c["mean"]:+.2f} (p={_c["p_exact_signflip"]:.3f})')
                A(f'| {_k} | ' + ' | '.join(_row) + ' |')
            A('')
            A('> L512 是训练长度，其余长度是外推；L512 与冻结终点方向相反时，冻结终点的“优势”只能读作长度外推鲁棒性，不能读作“CSA 更好”。')
            A('')
            _li_lines = []
            for _k, _v in comps.items():
                _one = _length_interaction_markdown(_k, _v.get('length_interaction'))
                if _one:
                    _li_lines.extend(_one)
            if _li_lines:
                A('### 长度效应修饰（post-hoc；不改变冻结判定）')
                A('')
                lines.extend(_li_lines)
        _claim_no = [k for k, v in comps.items() if v.get('mechanism_claim_allowed') is False]
        if _claim_no:
            A('**主张边界（v4）**：以下比较不自动构成机制证据（`mechanism_claim_allowed=false`）：')
            for _k in _claim_no:
                A(f"- `{_k}`：{comps[_k].get('claim_note') or 'multi-factor / not budget-matched'}")
            A('这些比较即使 Holm 显著，也只能作为 system-level 差异；不得写成“CSA 的 block selection 机制被证实”。')
            A('')
        _cov = ft.get('coverage_requirement') or {}
        if _cov:
            _cov_missing = [k for k, v in comps.items() if v.get('coverage_ok') is False]
            A(f"**覆盖要求**：require_complete_coverage={_cov.get('require_complete_coverage')}；"
              f"缺失完整种子的比较：{sorted(_cov_missing) if _cov_missing else '无'}。"
              + ('覆盖不足的比较不进入 CONFIRMED 集合（缺失不是随机缺失，不能当作较小的 n 使用）。'
                 if _cov.get('require_complete_coverage') else ''))
            A('')
    _pr = ft.get('probe') or {}
    if _pr.get('available') and _pr.get('cells'):
        A('### P5T test split 干扰探针（描述性二次证据，不进入确认性检验）')
        A('')
        A(f"> 与主表同一 held-out 切片（fp `{str(_pr.get('slice_fp'))[:12]}…`），按冻结臂/种子/配方/代码语义准入（{_pr.get('n_admitted')} 条记录，剔除 {_pr.get('n_excluded')} 条）。本表只作描述性二次证据，**不参与 Holm 校正，也不影响确认结论**；validation 探针的探索性读数见 `analysis_v12/stats.json` 的 v10 部分。")
        A('')
        A('| 臂 | arm | L | rho | mode | n | PPL mean |')
        A('|---|---|---|---|---|---|---|')
        for _c in _pr['cells']:
            _amb = '（歧义：多个探针参数化，未合并）' if _c.get('ambiguous_cell') else ''
            A(f"| `{_c['variant']}` | {_c['arm']} | {_c['eval_len']} | {_c['rho']} | {_c['mode']} | {_c['n']} | {_c['mean']:.2f}{_amb} |")
        A('')
        _pctr = _pr.get('contrasts') or {}
        if _pctr:
            A('| 描述性配对差（test split） | n | Δ mean | p (exact) |')
            A('|---|---|---|---|')
            for _k, _v in _pctr.items():
                A(f"| {_k} | {_v.get('n')} | {_v.get('mean', float('nan')):+.2f} | {_v.get('p_exact_signflip', float('nan')):.3f} |")
            A('')
    elif _pr:
        A('**P5T test split 干扰探针**：本次运行未找到可采纳的 test 探针记录，统计已显式标注为不可用。')
        A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A(f'| `{TEST_OUTDIR}/training_health.json` | P5H：冻结臂的训练曲线健康门禁（仅诊断，不参与选择） |')
    A(f'| `{TEST_OUTDIR}/bylen.json` | P5T：冻结臂在 test split 的干净 by-length 评估 |')
    A(f'| `{TEST_OUTDIR}/distractor.json` | P5T：test split 上的 uniform 腐坏探针（描述性） |')
    A('| `analysis_v12/stats.json` | v10 扩展探针统计 + 冻结确认性统计 |')
    A('| `analysis_v12/test_stats.json` | 冻结主比较 + Holm 校正（含准入记录、逐文档 doc_sampling 与功效注记） |')
    A('| `confirmation_frozen.json` | 预注册冻结清单（含修订史、范围声明与机器可校验的绑定块） |')
    A('| `results_lm_test/.freeze_lock.json` | 首次测试访问锁（清单哈希 + test 切片指纹；gitignore 的结果目录内，不入库） |')
    A('| `python -m explib.confirmation` | 本驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v12 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V12_NO_PUSH'):
        print(f'[git] push skipped (V12_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V12_NO_SHUTDOWN'):
        print('[v12] shutdown suppressed (V12_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v12] instance shuts down in {delay_s}s.')

def run_full():
    # Fail before the first GPU-second / before P5T can touch test.  This is
    # the same zero-GPU plan check CI runs; on a real box it also verifies that
    # the frozen manifest and this code agree.
    plan_check()
    guard = make_guard()
    guard.report()
    print(f'[v12] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state["booked_seconds"] / 3600:.2f} h)')
    git_push('v12: confirmation driver (frozen-manifest v3/revision 4: plan gate + coverage + pre-test power gate + freeze-lock binding)')
    import par
    _ctx = par.resolve_ctx(label='v12')
    all_ok = True
    completed = []
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v12] INCOMPLETE: stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            all_ok = False
            break
        if not V.cuda_healthy():
            print(f'[v12] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v12 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except SystemExit as _se:
            # a REFUSE (e.g. freeze-manifest gate) is a controlled stop, not a crash
            print(f'[v12] phase {pname} refused: {_se}')
            all_ok = False
            break
        except Exception:
            traceback.print_exc()
            all_ok = False
            break
        completed.append(pname)
        all_ok &= git_push(f'v12: phase {pname} results')
    incomplete = [p[0] for p in PHASES if p[0] not in completed]
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v12: stats + REPORT_v12.md (analysis_v12)')
    guard.report()
    if incomplete:
        print(f'\n[v12] INCOMPLETE — phases not completed: {incomplete}.')
        if 'P5T' in incomplete:
            print('[v12] The confirmatory test-split result is NOT available; do not describe this run as complete.')
    else:
        print('\n[v12] ALL PHASES DONE.')
    schedule_shutdown(120 if (all_ok and not incomplete) else 2400)

def _plan_recipe(manifest, arm, mr):
    """Build a P1L-format recipe string carrying the frozen signature.

    This is a SYNTHETIC descriptor used only by the zero-GPU plan check; it
    never enters a result file.  It mirrors the fields `run_lenphase_cell`
    writes so `_recipe_matches_signature` (including the per-arm `mr`) is
    exercised on the same code path as a real checkpoint.
    """
    _sig = _recipe_signature(manifest) or {}
    _nseq = int(((manifest.get('eval') or {}).get('n_seqs_clean')) or 0)
    return (f"p1l_steps{_sig.get('steps', 3000)}_tl{_sig.get('sl', 512)}"
            f"_v{_sig.get('v', 8192)}_bs{_sig.get('bs', 12)}"
            f"_lr{_sig.get('lr', 0.0003)}_nt{_sig.get('nt', 8000000)}"
            f"_wu{_sig.get('wu', 50)}_wd{_sig.get('wd', 0.1)}"
            f"_cl{_sig.get('cl', 0.05)}_dlm{_sig.get('dlm', 10.0)}"
            f"_mr{float(mr):.8g}-csa_fixed_rope_ns{_nseq}"
            f"_dfplan_gc1.0_b0.9,0.999_eps1e-08_tkplan_detplan_trfpplan")


def _archive_plan_snapshot(report, manifest, manifest_path=FREEZE_MANIFEST, outdir='analysis_v12'):
    """Persist what the plan gate approved, for post-hoc audit of an unattended run.

    The old protocol had a human pause between `plan` and the confirmation
    chain.  The automated gate replaces it with machine checks; this snapshot
    is the durable record of exactly those checks at approval time — the four
    fields a human would have eyeballed (arms / seeds / primary comparisons /
    alpha), the manifest sha256, the code stamp and the git tree — so the
    review can happen asynchronously (even after the one-shot P5T access)
    instead of blocking the run.  Local artifact by design (analysis_*/ is
    gitignored; the tamper-evident binding lives in the freeze lock and the
    per-record manifest_sha256).  A write failure must not fail an otherwise
    passed plan — the warning keeps the gap visible.
    """
    if os.path.abspath(str(manifest_path)) != os.path.abspath(FREEZE_MANIFEST):
        # A smoke/test manifest is a plumbing roundtrip, not a pre-registration;
        # only the real freeze gets the durable audit snapshot (otherwise a
        # smoke plan_check would silently overwrite the confirmation record).
        return
    try:
        _git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        _head = _git.stdout.strip() if _git.returncode == 0 else None
        _stat = (manifest.get('statistics') or {})
        _comps = [{'name': c.get('name'), 'a': c.get('a'), 'b': c.get('b'), 'endpoint': c.get('endpoint')}
                  for c in (manifest.get('primary_comparisons') or []) if isinstance(c, dict)]
        snap = {
            'archived_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'archived_at_epoch': time.time(),
            'purpose': ('audit record of the automated plan gate: what the machine approved '
                        'before P5C training and the one-shot P5T test access'),
            'git_head': _head,
            'code_semantics': L.CODE_SEMANTICS,
            # The git commit alone cannot identify a dirty work tree, a source
            # tarball, or an edit that leaves CODE_SEMANTICS unchanged.  Bind
            # the exact source bytes and the dependency-lock contract that the
            # plan gate approved, so a later P5T/analysis artifact can be
            # compared against this record instead of trusting the stamp.
            'source_tree': L.source_tree_manifest(),
            'dependency_contract': L.dependency_contract(),
            'manifest': manifest_path,
            'manifest_sha256': report.get('manifest_sha256'),
            'human_review': {
                'arms': list(manifest.get('arms') or []),
                'seeds': list(manifest.get('seeds') or []),
                'primary_comparisons': _comps,
                'alpha': _stat.get('alpha', 0.05),
                'min_seeds_for_confirmation': _stat.get('min_seeds_for_confirmation'),
                'decision_rule': _stat.get('decision_rule'),
            },
            'plan': report,
        }
        os.makedirs(outdir, exist_ok=True)
        L.atomic_write_json(os.path.join(outdir, 'plan_snapshot.json'), snap, indent=2)
        print(f'[v12 plan] snapshot archived -> {outdir}/plan_snapshot.json (unattended-run audit record; the human review is now asynchronous and non-blocking)')
    except Exception as _e:
        print(f'[v12 plan] WARNING: plan snapshot not archived ({type(_e).__name__}: {_e}) — the checks passed, but keep the console log as the audit record')
