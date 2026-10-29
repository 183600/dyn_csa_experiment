

def _print_methodology_audit(report):
    print('\n========== methodology audit: controlled comparison / ablation / statistics ==========')
    print('  variants with measurable records: ' + (', '.join(report['variants_present']) or '(none)'))
    print('  Δ = with-module − without-module (perplexity; negative = the module helps).')
    _fam = report.get('family') or {}
    print('  family: %d declared contrasts, m_eff=%s; Holm p is the family-corrected p.' % (int(_fam.get('declared_contrasts') or 0), _fam.get('effective_m')))
    if _fam.get('min_seeds_for_family_significance'):
        print('  minimum seeds for a family-corrected rejection at alpha=%g: n>=%d (below this every p is resolution-limited).' % (float(report.get('alpha', 0.05)), int(_fam['min_seeds_for_family_significance'])))
    _lh = report.get('look_history') or {}
    if int(_lh.get('panel_looks') or 0) > 1:
        print('  analysis looks: %d (seed top-ups); %d contrast(s) use the exact '
              'max-over-looks p%s'
              % (int(_lh['panel_looks']), int(_lh.get('n_contrasts_look_adjusted') or 0),
                 (': DOWNGRADED ' + ', '.join(_lh['downgraded_naive_significant'][:3]))
                 if _lh.get('downgraded_naive_significant') else ''))
    _abl = report.get('ablation_completeness') or {}
    if _abl:
        _missing = '; '.join('%s=%s' % (c, ','.join(a))
                             for c, a in sorted((_abl.get('missing_arms') or {}).items()))
        _omitted = _abl.get('omitted_components') or []
        if _abl.get('complete') is None:
            _state, _tail = 'not applicable', ' — no declared ablation contrast is measured in this panel'
        elif report.get('ablation_complete'):
            _state, _tail = 'complete', ''
        else:
            _detail = _missing or '(unspecified)'
            if _omitted:
                _detail += '; omitted components: ' + ', '.join(_omitted)
            _state, _tail = 'INCOMPLETE', ' — missing ' + _detail
        print('  ablation completeness (%s): %s%s'
              % (_abl.get('mode'), _state, _tail))
    _ix = report.get('interactions') or {}
    if _ix.get('factorials'):
        _ixf = _ix.get('family') or {}
        print('  --- 2x2 factorial interactions (log-PPL; exploratory, separate Holm family) ---')
        print('  interaction family: m=%s, alpha=%g, min attainable p_Holm=%s'
              % (_ixf.get('m'), float(_ix.get('alpha', 0.05)),
                 _ixf.get('min_attainable_p_holm')))
        _ixl = _ix.get('look_history') or {}
        if int(_ixl.get('n_look_adjusted') or 0):
            print('  interaction looks: %d contrast(s) use the exact '
                  'max-over-looks p%s'
                  % (int(_ixl['n_look_adjusted']),
                     ('; DOWNGRADED ' + ', '.join(
                         _ixl.get('downgraded_naive_significant') or []))
                     if _ixl.get('downgraded_naive_significant') else ''))
        for _name, _irec in _ix['factorials'].items():
            _ic = _irec.get('interaction') or {}
            if not _ic.get('available'):
                print('    %-42s %s (%s)'
                      % (_name, _irec.get('status'),
                         _ic.get('reason') or 'no complete 2x2 panel'))
                continue
            _ph = _ic.get('p_holm')
            print('    %-42s n=%-2d  interaction=%+.4f  ratio=%.4f  p=%s  p_Holm=%s%s'
                  % (_name, int(_ic.get('n') or 0), float(_ic.get('mean') or 0.0),
                     float(_ic.get('ratio_of_ratios') or float('nan')),
                     ('%.4f' % float(_ic['p_exact_signflip']))
                     if isinstance(_ic.get('p_exact_signflip'), (int, float)) else 'n/a',
                     ('%.4f' % float(_ph)) if isinstance(_ph, (int, float)) else 'n/a',
                     '  [SIGN CONFLICT vs pre-registered direction]'
                     if _name in (_ix.get('direction_conflict_interactions') or []) else ''))
            print('        predicted sign=%s, observed=%s; %s; mechanism claim allowed=%s'
                  % (_ic.get('predicted_sign'), _ic.get('observed_sign'),
                     _ic.get('effect_vs_noise') or 'n<min_n',
                     _ic.get('mechanism_claim_allowed')))
    if report.get('arm_summary'):
        print('  arm means (mean ± sample std over seeds):')
        for _k, _e in report['arm_summary'].items():
            print('    %-30s n=%-2d  %.4f ± %.4f%s' % (
                _k, _e['n_seeds'], _e.get('ppl_primary', _e['ppl_mean']), _e['ppl_std'],
                '' if _e['meets_min_seeds'] else '  (BELOW min_n)'))
    _md = report.get('missingness') or {}
    _mrows = [r for r in (_md.get('contrasts') or [])
              if (r.get('n_unobserved') or r.get('n_missing') or r.get('n_excluded'))]
    if _mrows:
        _cov = _md.get('coverage') or {}
        print('  --- missing-data sensitivity (planned frame: %s) ---'
              % (_md.get('frame_basis') or 'observed union'))
        print('      planned cells: %s; missing: %s; present-but-unusable: %s; '
              'coverage=%s%s'
              % (_cov.get('n_planned_cells'), _cov.get('n_missing_cells'),
                 _cov.get('n_unusable_cells'),
                 ('%.3f' % float(_cov['coverage_rate']))
                 if isinstance(_cov.get('coverage_rate'), (int, float)) else 'n/a',
                 '; UNEQUAL ARM COVERAGE' if _cov.get('unequal_arm_coverage') else ''))
        for _r in _mrows[:12]:
            _b = _r.get('overall_mean_bounds')
            _bs = ('[%+.4f, %+.4f]' % (float(_b[0]), float(_b[1]))
                   if isinstance(_b, (list, tuple)) and len(_b) == 2 else 'n/a')
            print('      %-42s missing=%-2s excluded=%-2s Δ-bounds=%s  %s'
                  % (_r.get('name'), _r.get('n_missing'), _r.get('n_excluded'),
                     _bs, _r.get('status')))
        if _md.get('sign_not_robust_contrasts'):
            print('      SIGN-NOT-ROBUST: ' + ', '.join(
                _md['sign_not_robust_contrasts'][:5]))
        if _md.get('margin_not_robust_contrasts'):
            print('      MARGIN-NOT-ROBUST: ' + ', '.join(
                _md['margin_not_robust_contrasts'][:5]))
    for comp, entries in report['components'].items():
        live = [e for e in entries if e['n_shared'] > 0]
        if not live:
            continue
        _cv = report.get('controlled_variables', {}).get(comp, {})
        _kn = _cv.get('train_knobs_equal')
        _kn_s = {True: 'optimizer/schedule knobs shared', False: 'OPTIMIZER/SCHEDULE KNOBS DIFFER', None: 'optimizer/schedule knobs not recorded (legacy or absent)'}[_kn]
        print('  [%s]  %s; arms=%s' % (comp, _kn_s, ','.join(_cv.get('arms_present') or []) or 'none'))
        for e in live:
            _c = 'controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons']))
            _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
            _ph = e.get('p_holm_global')
            _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
            _tag = '' if e.get('tag_with', '') == e.get('tag_without', '') == '' else '   [%s vs %s]' % (e.get('tag_with') or '(base)', e.get('tag_without') or '(base)')
            print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, _tag))
            print('        with %.4f±%.4f vs without %.4f±%.4f; relative Δ=%+.2f%% of baseline, seed noise=%.2f%% (SNR=%.2f)' % (
                e.get('with_mean', float('nan')), e.get('with_std', float('nan')),
                e.get('without_mean', float('nan')), e.get('without_std', float('nan')),
                e.get('relative_effect_pct', float('nan')), e.get('relative_noise_pct', float('nan')),
                e.get('signal_to_noise', float('nan'))))
            _lo, _hi = e.get('ci95_low'), e.get('ci95_high')
            if isinstance(_lo, (int, float)) and isinstance(_hi, (int, float)) and math.isfinite(_lo) and math.isfinite(_hi):
                print('        95%% CI for Δ: [%+.4f, %+.4f]%s' % (
                    _lo, _hi, '; SEED-DRIVEN (dropping one seed flips the sign)' if e.get('seed_driven') else ''))
            elif e.get('seed_driven'):
                print('        SEED-DRIVEN: dropping one seed flips the sign of the mean')
            _pi = e.get('prediction_interval_95')
            _ps = e.get('prob_new_seed_same_sign')
            if (isinstance(_pi, (list, tuple)) and len(_pi) == 2
                    and all(isinstance(v, (int, float)) and math.isfinite(v)
                            for v in _pi)):
                _tail = ''
                if isinstance(_ps, (int, float)) and math.isfinite(float(_ps)):
                    _tail = ('; P(new seed has the observed sign)=%.3f' % float(_ps))
                print('        predictive 95%% interval for ONE new seed: [%+.4f, %+.4f]%s '
                      '(model-based; not a CI for the mean and not a posterior)'
                      % (float(_pi[0]), float(_pi[1]), _tail))
            _rb = e.get('paired_row_bootstrap_delta') or {}
            if _rb.get('status') == 'ok' and _rb.get('ci95_delta_ppl'):
                print('        evaluation-row bootstrap 95%% CI for Δ: [%+.4f, %+.4f] (%d rows; row uncertainty only)' % (
                    float(_rb['ci95_delta_ppl'][0]), float(_rb['ci95_delta_ppl'][1]),
                    int(_rb.get('n_rows') or 0)))
            if e.get('capacity_confounded'):
                _pb = e.get('param_breakdown') or {}
                _extra = []
                for _k, _label in (('non_embedding_rel_diff', 'non-embedding'),
                                   ('active_rel_diff', 'active')):
                    if isinstance(_pb.get(_k), (int, float)):
                        _extra.append('%s %.2f%%' % (_label, 100.0 * float(_pb[_k])))
                print('        CAPACITY CONFOUND: total parameter difference %.2f%%%s — report the parameter delta or add a parameter-matched control' % (
                    100.0 * float(e.get('param_rel_diff') or 0.0),
                    ('; ' + ', '.join(_extra)) if _extra else ''))
            print('        %s | %s' % (_c, e['effect_vs_noise']))
            _pe = e.get('sesoi') or {}
            if _pe.get('margin_ppl') is not None:
                # `0.0 or nan` would turn an exact zero TOST p into "nan";
                # test for None instead of truthiness.
                _tost_p = _pe.get('tost_p_tost')
                _tost_s = ('%.4g' % float(_tost_p)
                           if _tost_p is not None else 'n/a')
                print('        practical: %s (|Δ|=%.4f, margin=%.4f PPL; TOST p=%s, Holm p=%s) — %s'
                      % (e.get('practical_decision'), abs(float(_pe.get('mean') or 0.0)),
                         float(_pe['margin_ppl']), _tost_s,
                         ('%.4g' % float(_pe['tost_p_holm']))
                         if _pe.get('tost_p_holm') is not None else 'n/a',
                         _pe.get('reason') or ''))
    _sesoi = report.get('sesoi') or {}
    if _sesoi.get('declared_components') or _sesoi.get('declared_baselines'):
        _fam = _sesoi.get('family') or {}
        print('  SESOI / practical equivalence (pre-registered; m=%s, TOST+Holm, alpha=%g):'
              % (_fam.get('m'), float(report.get('alpha', 0.05))))
        if _sesoi.get('useful_components'):
            print('    practically useful (>SESOI): ' + ', '.join(_sesoi['useful_components']))
        if _sesoi.get('equivalent_components'):
            print('    practically removable (TOST inside SESOI; NOT a failed significance test): '
                  + ', '.join(_sesoi['equivalent_components']))
        if _sesoi.get('sub_sesoi_components'):
            print('    detectable but below SESOI: ' + ', '.join(_sesoi['sub_sesoi_components']))
        if _sesoi.get('harmful_components'):
            print('    practically HARMFUL (opposite direction, >SESOI): '
                  + ', '.join(_sesoi['harmful_components']))
        if _sesoi.get('inconclusive_components'):
            print('    inconclusive (neither >SESOI nor equivalent): '
                  + ', '.join(_sesoi['inconclusive_components']))
    if report['verified_components']:
        print('  verified innovations (controlled AND family-Holm p<alpha AND |Δ|>seed noise AND, when declared, >SESOI): ' + ', '.join(report['verified_components']))
    if report.get('equivalent_components'):
        print('  equivalent innovations (TOST inside the pre-registered SESOI after family adjustment): ' + ', '.join(report['equivalent_components']))
    if report.get('sub_sesoi_components'):
        print('  sub-SESOI innovations (significant but too small to be practically useful): ' + ', '.join(report['sub_sesoi_components']))
    if report.get('practically_indeterminate_components'):
        print('  practically indeterminate (significant/seed-driven but not clearly >SESOI): ' + ', '.join(report['practically_indeterminate_components']))
    if report.get('measured_components'):
        print('  measured innovations (controlled contrast with n>=%d; descriptive unless also verified): ' % int(report.get('min_n', 3)) + ', '.join(report['measured_components']))
    if report.get('nominally_significant_components'):
        print('  nominally significant (raw p<alpha, before family correction): ' + ', '.join(report['nominally_significant_components']))
    if report.get('direction_conflict_components'):
        print('  DIRECTION CONFLICT (significant but opposite to the pre-registered hypothesis): ' + ', '.join(report['direction_conflict_components']))
    if report.get('direction_unspecified_components'):
        print('  direction-unspecified (significant, but the hypothesis declared predicted_sign=0): ' + ', '.join(report['direction_unspecified_components']))
    if report.get('resolution_limited_components'):
        print('  resolution-limited (non-noise effect, but the family-corrected p floor > alpha at this n; need n>=%s — add seeds, do NOT read as a null): ' % str((_fam.get('min_seeds_for_family_significance') or '?')) + ', '.join(report['resolution_limited_components']))
    if report.get('system_level_only_components'):
        print('  SYSTEM-LEVEL ONLY (significant but the objective/compute budget favours the claimed winner): ' + ', '.join(report['system_level_only_components']))
    if report.get('noise_level_components'):
        print('  noise-level (effect within seed noise across all contrasts): ' + ', '.join(report['noise_level_components']))
    if report.get('underpowered_components'):
        print('  underpowered innovations (contrasts exist but n<%d — add seeds before claiming an effect): ' % int(report.get('min_n', 3)) + ', '.join(report['underpowered_components']))
    if report['unverified_components']:
        print('  unverified innovations (no controlled leave-one-out contrast): ' + ', '.join(report['unverified_components']))
    if report.get('baselines'):
        print('  --- primary baseline claims (parameter-matched dense control) ---')
        for comp, entries in report['baselines'].items():
            live = [e for e in entries if e['n_shared'] > 0]
            if not live:
                continue
            _pc = report.get('controlled_variables', {}).get(comp, {}).get('param_matched')
            print('  [%s]  parameter-controlled=%s' % (comp, _pc))
            for e in live:
                _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
                _ph = e.get('p_holm_global')
                _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
                print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s  params=%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, ('yes' if e.get('param_controlled') else ('no' if e.get('param_controlled') is False else 'n/a'))))
                print('        %s | %s' % ('controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons'])), e['effect_vs_noise']))
        _mt = report.get('metric_audit') or {}
        print('  metric policy: %s (warnings: %d)' % (_mt.get('status') or 'n/a', len(_mt.get('warnings') or [])))
        _bt = report.get('baseline_tuning') or {}
        print('  baseline tuning: status=%s tuning_verified=%s' % (_bt.get('status'), _bt.get('tuning_verified')))
        _sb = report.get('simple_baseline') or {}
        if _sb:
            print('  count-model floor: status=%s baseline_verified=%s best=%s PPL=%s' % (
                _sb.get('status'), _sb.get('baseline_verified'), _sb.get('best_baseline'), _sb.get('best_ppl')))
            if _sb.get('panel_beats_simple_floor') is not None:
                print('    panel reference %s PPL=%.4f; comparable=%s; beats simple floor=%s' % (
                    _sb.get('panel_reference_arm'), float(_sb.get('panel_reference_ppl') or float('nan')),
                    _sb.get('comparable_to_panel'), _sb.get('panel_beats_simple_floor')))
            if _sb.get('method_arm_floor_status'):
                print('    declared method-arm floor gate: %s (arms: %s)' % (
                    _sb.get('method_arm_floor_status'),
                    ', '.join(sorted((_sb.get('method_arm_floor_contrasts') or {}))) or 'none'))
        if report.get('verified_baselines'):
            print('  statistically verified baseline claims (NOT yet evidence against a tuned baseline): ' + ', '.join(report['verified_baselines']))
        if report.get('tuned_verified_baselines'):
            print('  tuned verified baseline claims (equal-search tuning artifact present): ' + ', '.join(report['tuned_verified_baselines']))
        if report.get('untuned_verified_baselines'):
            print('  statistically verified but UNTUNED baseline claims (not evidence against a tuned baseline): ' + ', '.join(report['untuned_verified_baselines']))
        if report.get('direction_conflict_baselines'):
            print('  DIRECTION CONFLICT baseline claims: ' + ', '.join(report['direction_conflict_baselines']))
        if report.get('nominally_significant_baselines'):
            print('  nominally significant baseline claims (raw p<alpha): ' + ', '.join(report['nominally_significant_baselines']))
        if report.get('resolution_limited_baselines'):
            print('  resolution-limited baseline claims: ' + ', '.join(report['resolution_limited_baselines']))
        if report.get('unverified_baselines'):
            print('  unverified baseline claims (missing/incomplete/param-mismatched): ' + ', '.join(report['unverified_baselines']))
        _bcg = report.get('baseline_claim_gate') or {}
        if _bcg.get('applicable'):
            print('  baseline claim gate: allowed=%s status=%s' % (
                _bcg.get('claim_allowed'), _bcg.get('status')))
            for _name, _rung in (_bcg.get('rungs') or {}).items():
                print('    %-24s %-10s %s' % (_name, _rung.get('status'), _rung.get('detail')))
    for w in report['warnings']:
        print('  [warn] ' + w)
    print('======================================================================================')

def add_paired(*_a, **_kw):
    raise NotImplementedError('use explib.resume.add_paired (exp_lib cannot import explib.rope)')

def normalize_seed_list(seeds, *, default=(0,)):
    """Validate and de-duplicate a panel's seed list, preserving order.

    A duplicated seed would enqueue the SAME ``variant::seed`` cell twice
    (the second run silently overwriting the first record), and an explicitly
    empty/absent list would train the fallback seed 0 while the manifest
    planned no cells at all.  Both make the on-disk trace disagree with the
    work that actually ran, so the axes are normalised once, before
    probing/tracking.
    Accepts a scalar, an iterable of integers, or a comma/space separated
    string; booleans and non-integral floats are rejected rather than silently
    truncated to a different seed.
    """
    if seeds is None:
        seeds = list(default)
    elif isinstance(seeds, (bool, np.bool_)):
        raise ValueError(f'seed must be an integer, got boolean {seeds!r}')
    elif isinstance(seeds, numbers.Real):
        seeds = [seeds]
    elif isinstance(seeds, np.ndarray) and seeds.ndim == 0:
        seeds = [seeds.item()]
    elif isinstance(seeds, (str, bytes)):
        seeds = seeds.decode().split() if isinstance(seeds, bytes) else seeds.replace(',', ' ').split()
    out = []
    seen = set()
    for raw in seeds:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f'seed must be an integer, got boolean {raw!r}')
        # Use the shared scalar reader: it handles NumPy scalars and rejects
        # non-integral numeric objects (e.g. Decimal('1.5')) instead of
        # silently truncating them with int(), and enforces the RNG domain.
        try:
            s = _seed_value(raw)
        except ValueError as _se:
            raise ValueError(f'seed must be an integer in [0, {MAX_SEED}], got {raw!r}') from _se
        if s in seen:
            print(f'[seeds] WARNING: duplicate seed {s} removed from the panel plan (it would otherwise be trained twice under the same key)')
            continue
        seen.add(s)
        out.append(s)
    if not out:
        # The fallback is still a declared seed plan, not a raw int() cast:
        # default=(True,) or default=(1.5,) must not silently become seed 1.
        try:
            _fallback = list(default)
        except TypeError:
            _fallback = [default]
        for raw in _fallback:
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f'default seed must be an integer, got boolean {raw!r}')
            try:
                s = _seed_value(raw)
            except ValueError as _se:
                raise ValueError(f'default seed must be an integer in [0, {MAX_SEED}], got {raw!r}') from _se
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def normalize_variant_list(variants):
    """Validate and de-duplicate a panel's variant list, preserving order."""
    if variants is None:
        return []
    if isinstance(variants, str):
        variants = [variants]
    out = []
    seen = set()
    for raw in variants:
        v = str(raw)
        if v in seen:
            print(f'[variants] WARNING: duplicate variant {v!r} removed from the panel plan (it would otherwise be trained twice under the same key)')
            continue
        seen.add(v)
        out.append(v)
    return out


_LAST_SEED = None


def set_seed(seed):
    """Seed every RNG that can affect a run: Python ``random``, NumPy's
    legacy global RNG, PyTorch's CPU RNG and every CUDA device, plus the
    Python hash seed (best effort for child processes).  Returns the integer
    seed so callers can record exactly what was used.  torch's deterministic
    algorithm / cuDNN switches are pinned at the same time, so two runs with
    the same seed on the same machine follow the same kernels."""
    global _LAST_SEED
    seed = _seed_value(seed)
    _LAST_SEED = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _attb_bump_epoch()
    torch.manual_seed(seed)
    _pin_cpu_threads()
    # Pin the backend determinism switches on CPU too: the cuDNN flags are
    # inert without CUDA, while CSA_DETERMINISTIC must still enable torch's
    # deterministic-algorithm mode on a CPU box (a CPU run is a first-class
    # reproduction target).  CUDA RNGs are seeded explicitly below.
    _pin_cuda_determinism()
    if torch.cuda.is_available():
        # manual_seed already covers all devices; the explicit current-device
        # call documents the contract and is harmless.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # TensorFlow is not a dependency of this repo, but a future/embedded
    # TensorFlow path must not become an unseeded RNG source.  Only touch it
    # when the user has already imported it (never import TensorFlow here:
    # that would make every CPU-only run import it needlessly).
    _tf = sys.modules.get('tensorflow')
    if _tf is not None:
        try:
            _tf.random.set_seed(seed)
            try:
                _tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception as _tfe:
            print(f'[determinism] could not seed the already-imported TensorFlow RNG ({type(_tfe).__name__}: {_tfe})')
    return seed
_CPU_THREADS_PINNED = None

def rng_fingerprint():
    """Digest of the *current* Python/NumPy/Torch/CUDA RNG state.

    Recorded immediately after `set_seed(seed)` in every training record.  Two
    runs with the same seed on the same stack must produce the same digest;
    a mismatch means some import/construction consumed randomness before the
    run started (and therefore that the seed is not fully controlled).  The
    digest is a diagnostic only: it never enters the resume fingerprint."""
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode('utf-8', 'replace'))
    _ns = np.random.get_state()
    h.update(str(_ns[0]).encode('ascii', 'replace'))
    h.update(np.asarray(_ns[1], dtype=np.uint64).tobytes())
    h.update(repr((_ns[2], _ns[3], _ns[4])).encode('ascii', 'replace'))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        try:
            for _st in torch.cuda.get_rng_state_all():
                h.update(_st.cpu().numpy().tobytes())
        except Exception:
            # a broken CUDA context is reported elsewhere; the CPU part of
            # the fingerprint is still a useful diagnostic
            pass
    return h.hexdigest()[:16]

def init_core_fingerprint(model):
    """Hash the initialization of the parameter tensors every arm shares.

    A paired-seed design only blocks the *seed*, not the initialization:
    different architectures consume different amounts of RNG, so "seed=3" does
    not by itself prove that the two arms started from the same embedding /
    position / LM-head / final-norm tensors.  This digest covers exactly the
    trunk pieces whose names, shapes and construction order do not depend on
    the attention variant (they are built before the blocks), so equality is a
    real matched-initialization check rather than a tautology.

    It is recorded per training record and `pair_reason` refuses a pair whose
    two records carry different digests.  Records from before this field
    existed carry no digest; a one-sided digest is unverifiable, exactly like
    the other provenance gates.
    """
    h = hashlib.sha256()
    sd = model.state_dict() if hasattr(model, 'state_dict') else {}
    for name in ('tok.weight', 'pos.weight', 'head.weight', 'norm.weight'):
        t = sd.get(name)
        if t is None:
            continue
        arr = t.detach().to('cpu').float().contiguous().numpy()
        h.update(name.encode('ascii', 'replace'))
        h.update(repr(tuple(arr.shape)).encode('ascii', 'replace'))
        h.update(arr.tobytes())
    return h.hexdigest()[:16]

def paired_init_core_audit(records_a, records_b, seeds):
    """Audit the matched-initialization contract for PAIRED seeds.

    ``init_core_fp`` hashes the shared trunk initialization.  It depends on
    the seed, so the valid test is equality between the TWO ARMS at the SAME
    seed; equality across different seeds is neither expected nor required.
    Treating the whole panel as one value set would mark a healthy panel as
    mismatched (every seed has a different init) or, when the field is
    absent, vacuously pass.  This helper performs the per-seed check once so
    the confirmation analysis and its tests share one definition.

    A pair is ``verified`` only when both records carry a non-empty digest
    and the digests are equal.  ``mismatch`` is definite evidence that the
    two arms did not start from the same shared trunk, so the caller must
    drop that pair.  A one-sided or absent digest is ``unverified`` (legacy
    records predate the field): it is reported, never read as equality, and
    does not by itself drop the pair.

    ``pairing_ok`` is ``True`` only when every requested seed is verified,
    ``False`` when at least one pair mismatches, and ``None`` when the
    evidence is merely incomplete.  It is deliberately tri-state: absence of
    evidence is not a pass.
    """
    if not isinstance(records_a, dict) or not isinstance(records_b, dict):
        raise ValueError('paired_init_core_audit: records_a/records_b must be mappings')
    if isinstance(seeds, (str, bytes)):
        raise ValueError('paired_init_core_audit: seeds must be an iterable of integers')
    try:
        seed_list = list(seeds)
    except TypeError:
        raise ValueError('paired_init_core_audit: seeds must be an iterable') from None
    _seen = set()
    norm = []
    for s in seed_list:
        si = _as_seed_int(s)
        if si is None:
            raise ValueError(f'paired_init_core_audit: seed {s!r} is not an integer')
        if si in _seen:
            raise ValueError(f'paired_init_core_audit: duplicate seed {si}')
        _seen.add(si)
        norm.append(si)

    def _val(rec):
        if not isinstance(rec, dict):
            return None
        v = rec.get('init_core_fp')
        # Malformed evidence is unverified, not a value: str(123) == '123'
        # must not let a non-digest become a "match".
        if not isinstance(v, str):
            return None
        s = v.strip()
        return s if s else None

    per_seed = []
    counts = {'verified': 0, 'mismatch': 0, 'unverified': 0, 'missing': 0}
    for s in norm:
        ra, rb = records_a.get(s), records_b.get(s)
        va, vb = _val(ra), _val(rb)
        if ra is None or rb is None:
            status = 'missing'
        elif va is not None and vb is not None:
            status = 'verified' if va == vb else 'mismatch'
        else:
            status = 'unverified'
        counts[status] += 1
        entry = {'seed': int(s), 'status': status}
        if va is not None:
            entry['a'] = va
        if vb is not None:
            entry['b'] = vb
        per_seed.append(entry)
    n_pairs = len(norm)
    mismatch = [e['seed'] for e in per_seed if e['status'] == 'mismatch']
    unverified = [e['seed'] for e in per_seed if e['status'] == 'unverified']
    missing = [e['seed'] for e in per_seed if e['status'] == 'missing']
    if mismatch:
        pairing_ok = False
    elif n_pairs and counts['verified'] == n_pairs:
        pairing_ok = True
    else:
        pairing_ok = None
    return {'n_pairs': int(n_pairs), 'n_verified': int(counts['verified']),
            'n_mismatch': int(counts['mismatch']),
            'n_unverified': int(counts['unverified']),
            'n_missing': int(counts['missing']),
            'verified_seeds': [e['seed'] for e in per_seed if e['status'] == 'verified'],
            'mismatch_seeds': mismatch, 'unverified_seeds': unverified,
            'missing_seeds': missing,
            'pairing_ok': pairing_ok,
            'keep_seeds': [e['seed'] for e in per_seed if e['status'] != 'mismatch'],
            'per_seed': per_seed,
            'definition': ('verified = both arms at the same seed carry an equal non-empty '
                           'init_core_fp; mismatch = both present but different (drop the pair); '
                           'unverified = one-sided/absent digest (legacy, not a match); '
                           'pairing_ok is True only if every requested pair is verified, '
                           'False if any mismatches, None if evidence is incomplete.'),
            'note': ('matched initialization is a pairing validity precondition, not a '
                     'measured outcome; no training or evaluation value is changed')}

def capture_rng_state():
    """Snapshot every RNG stream a short diagnostic may consume.

    Diagnostics such as the small-batch overfit probe and the duplicate-run
    reproducibility probe call ``set_seed`` themselves.  Without a snapshot
    they would advance the process-wide RNG streams and silently couple a
    later training cell to how many probes ran before it.  The returned object
    is opaque: restore it with ``restore_rng_state``.
    """
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        try:
            state['cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            state['cuda'] = None
    else:
        state['cuda'] = None
    return state

def restore_rng_state(state):
    """Restore a snapshot from ``capture_rng_state`` (best effort on CUDA).

    Restoring is what makes the pre-flight checks side-effect free: the
    training cell that follows sees exactly the RNG state it would have seen
    if the checks had never run, so adding/diagnosing a probe cannot change a
    recorded loss curve.  A broken CUDA context is reported by the caller's
    training loop; the CPU streams are still restored here."""
    if not isinstance(state, dict):
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['cuda'])
        except Exception:
            pass

class _preserve_rng_state:
    """Context manager: run a diagnostic without perturbing the RNG streams.

    Also restores ``PYTHONHASHSEED`` because ``set_seed`` updates it for child
    processes; a probe must not leave that environment mutation behind."""

    def __enter__(self):
        self._env = os.environ.get('PYTHONHASHSEED')
        self._state = capture_rng_state()
        return self

    def __exit__(self, exc_type, exc, tb):
        restore_rng_state(self._state)
        if self._env is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = self._env
        return False

def dataloader_generator(seed):
    """A torch.Generator for a future torch DataLoader (scale path).

    The current loader is `batch_iter`, which is seeded directly by the run
    seed.  This helper is for DataLoader-based code: pass it as
    ``DataLoader(..., generator=L.dataloader_generator(seed))`` and pass
    ``worker_init_fn=L.seed_worker`` so worker RNGs are derived from the same
    seed instead of the OS entropy pool."""
    g = torch.Generator()
    g.manual_seed(_seed_value(seed))
    return g

def seed_worker(worker_id):
    """DataLoader worker_init_fn: seed Python/NumPy/Torch from torch's
    per-worker base seed.  PyTorch already seeds its own worker RNG before
    calling this hook; re-seeding it explicitly documents the contract and
    also covers older versions where a custom generator changes the base."""
    _seed = int(torch.initial_seed()) % (2 ** 32)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    _pin_cpu_threads()
    _pin_cuda_determinism()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

def dataloader_kwargs(seed):
    """DataLoader kwargs that pin BOTH the sampler order and worker RNGs.

    Scale path: `DataLoader(ds, **L.dataloader_kwargs(seed), num_workers=n)`.
    The generator controls shuffling/sampling; worker_init_fn derives each
    worker's Python/NumPy RNG from the same seed tree instead of OS entropy.
    """
    return {'generator': dataloader_generator(seed), 'worker_init_fn': seed_worker}

def deterministic_dataloader(dataset, *, seed=0, **kwargs):
    """Thin `torch.utils.data.DataLoader` wrapper with reproducible seeding.

    Existing caller-supplied `generator` / `worker_init_fn` are respected."""
    from torch.utils.data import DataLoader
    # An explicit None means "not supplied", not "please use OS entropy".
    if kwargs.get('generator') is None:
        kwargs['generator'] = dataloader_generator(seed)
    if kwargs.get('worker_init_fn') is None:
        kwargs['worker_init_fn'] = seed_worker
    return DataLoader(dataset, **kwargs)

def dataloader_contract_check(seed=0, *, num_workers=2, batch_size=4, batches=2, device=None):
    """Exercise a REAL seeded DataLoader, including worker processes.

    `seed_contract_check` draws the Python/NumPy/Torch streams directly and
    simulates the worker hook.  That cannot catch a wiring mistake such as a
    DataLoader built without `generator=`/`worker_init_fn=`, or a worker
    dataset that draws from the OS entropy pool.  This check therefore builds
    the actual `deterministic_dataloader` with `num_workers>0`, iterates the
    same seed twice and a different seed once, and compares the full batch
    stream (sampler order + every per-item Python/NumPy/Torch draw).

    The probe is side-effect free on the process RNG streams and is safe to
    call before a panel; it is skipped into the report rather than raised so a
    caller can decide whether worker processes are required on its platform.
    """
    seed = _seed_value(seed)
    _nw = _exact_count(num_workers, minimum=0)
    _bs = _exact_count(batch_size, minimum=1)
    _nb = _exact_count(batches, minimum=1)
    if _nw is None:
        raise ValueError(f'dataloader_contract_check: num_workers must be a non-negative integer, got {num_workers!r}')
    if _bs is None:
        raise ValueError(f'dataloader_contract_check: batch_size must be a positive integer, got {batch_size!r}')
    if _nb is None:
        raise ValueError(f'dataloader_contract_check: batches must be a positive integer, got {batches!r}')
    num_workers, batch_size, batches = int(_nw), int(_bs), int(_nb)
    if num_workers > 0:
        # Probe the OS primitive before constructing torch's worker iterator.
        # When POSIX semaphores are unavailable (a restricted sandbox), the
        # iterator's __init__ can fail after allocating a partially
        # initialised object whose __del__ raises an unraisable
        # AttributeError.  A clean contract report is both more reproducible
        # and easier for a caller to skip than a late GC failure.
        try:
            import multiprocessing as _mp
            _sem_probe = _mp.Semaphore(1)
        except (PermissionError, OSError) as e:
            return {
                'kind': 'dataloader_contract', 'seed': int(seed),
                'num_workers': num_workers, 'batch_size': batch_size,
                'batches': batches, 'device': str(DEVICE if device is None else device),
                'checks': {'stream_nonempty': False,
                           'same_seed_reproducible': False,
                           'different_seed_changes_stream': False,
                           'rng_state_preserved': None,
                           'worker_semaphores': False},
                'error': f'{type(e).__name__}: {e}',
                'passed': False,
                'diagnosis': ('DataLoader worker processes require POSIX '
                              f'semaphores, which this sandbox forbids '
                              f'({type(e).__name__}: {e})'),
            }
        else:
            try:
                _sem_probe.close()
            except Exception:
                pass
    try:
        from torch.utils.data import Dataset
    except Exception as e:  # pragma: no cover - torch is a hard dependency
        return {'kind': 'dataloader_contract', 'seed': int(seed),
                'num_workers': num_workers, 'passed': False,
                'diagnosis': f'could not import torch DataLoader: {type(e).__name__}: {e}'}

    class _RNGDataset(Dataset):
        def __init__(self, n):
            self.n = int(n)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return torch.tensor([int(i),
                                 int(random.random() * 1e9),
                                 int(np.random.random() * 1e9),
                                 int(torch.randint(0, 2 ** 31 - 1, (1,)).item())],
                                dtype=torch.int64)

    _n_items = max(batch_size * batches * max(1, num_workers), 16)

    def _stream(s):
        if num_workers == 0:
            # Without worker processes the dataset's own Python/NumPy/Torch
            # draws run in the parent; seed them explicitly so the same-seed
            # comparison still tests the loader wiring rather than leftover
            # process-state drift.
            set_seed(s)
        dl = deterministic_dataloader(_RNGDataset(_n_items), batch_size=batch_size,
                                      shuffle=True, seed=s, num_workers=num_workers)
        out = []
        for _b in dl:
            out.append(_b.tolist())
            if len(out) >= batches:
                break
        del dl
        return out

    _fp_before = rng_fingerprint()
    try:
        with _preserve_rng_state():
            a = _stream(seed)
            b = _stream(seed)
            c = _stream(_other_seed(seed))
        error = None
    except Exception as e:
        a = b = c = []
        error = f'{type(e).__name__}: {e}'
    _fp_after = rng_fingerprint()
    checks = {
        'stream_nonempty': bool(a and all(a)),
        'same_seed_reproducible': bool(a and a == b),
        'different_seed_changes_stream': bool(a and a != c),
        'rng_state_preserved': bool(_fp_before == _fp_after),
    }
    failed = [k for k, ok in checks.items() if not ok]
    report = {
        'kind': 'dataloader_contract',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'code': code_stamp(),
        'seed': int(seed),
        'num_workers': num_workers,
        'batch_size': batch_size,
        'batches': batches,
        'device': str(DEVICE if device is None else device),
        'checks': checks,
        'same_seed_stream': a,
        'other_seed_stream': c,
        'rng_fp_before': _fp_before,
        'rng_fp_after': _fp_after,
        'error': error,
        'passed': bool(not failed and error is None),
        'diagnosis': ('ok' if (not failed and error is None)
                      else ('error: ' + error if error else 'failed checks: ' + ', '.join(failed))),
    }
    return report

def seed_contract_check(seed=0, *, device=None, check_subprocess=False, check_dataloader_workers=False, dataloader_workers=2):
    """Explicit audit of every RNG source a reproducible run depends on.

    `reproducibility_check` proves that a short TRAINING stream repeats; this
    check is cheaper and more direct: it enumerates the seed contract itself
    (Python, NumPy, PyTorch CPU, every CUDA device, an already-imported
    TensorFlow, the DataLoader generator and the DataLoader worker hook), runs
    it twice with the same seed and once with a different seed, and verifies
    that same-seed draws are identical while every stream actually changes.
    The probe is side-effect free on the RNG streams (same
    `_preserve_rng_state` contract as the other pre-flights), so it can run
    before a panel without shifting a recorded loss curve.

    ``check_subprocess=True`` additionally re-executes a tiny draw script in
    two fresh interpreters with the same seed/PYTHONHASHSEED, which catches a
    child-process seeding mismatch that an in-process check cannot see.
    """
    seed = _seed_value(seed)
    _alt_seed = _other_seed(seed)
    keys = ['python', 'numpy', 'torch', 'dataloader_generator']
    if torch.cuda.is_available():
        keys.extend(['cuda', 'cuda_devices'])
    if sys.modules.get('tensorflow') is not None:
        keys.append('tensorflow')

    def _draws(s):
        set_seed(s)
        d = {
            'python': float(random.random()),
            'numpy': float(np.random.random()),
            'torch': float(torch.rand(1).item()),
        }
        if torch.cuda.is_available():
            d['cuda'] = [float(x) for x in torch.rand(2, device='cuda')]
            d['cuda_devices'] = {}
            for _di in range(int(torch.cuda.device_count())):
                with torch.cuda.device(_di):
                    d['cuda_devices'][str(_di)] = [float(x) for x in torch.rand(2, device='cuda')]
        g = dataloader_generator(s)
        d['dataloader_generator'] = [int(x) for x in torch.randint(0, 2 ** 31 - 1, (4,), generator=g)]
        _tf = sys.modules.get('tensorflow')
        if _tf is not None:
            try:
                d['tensorflow'] = float(np.asarray(_tf.random.uniform((1,))).reshape(-1)[0])
            except Exception as _tfe:
                d['tensorflow_error'] = f'{type(_tfe).__name__}: {_tfe}'
        return d

    def _worker_draws(s):
        # Reproduce torch's worker base seed explicitly, then let the hook
        # derive the worker's Python/NumPy/Torch streams from it.
        torch.manual_seed(s)
        seed_worker(0)
        g = dataloader_generator(s)
        return {
            'python': float(random.random()),
            'numpy': float(np.random.random()),
            'torch': float(torch.rand(1).item()),
            'dataloader_generator': [int(x) for x in torch.randint(0, 2 ** 31 - 1, (4,), generator=g)],
        }

    checks = {}
    errors = {}
    _fp_before = rng_fingerprint()
    with _preserve_rng_state():
        try:
            a = _draws(seed)
            b = _draws(seed)
            c = _draws(_alt_seed)
            checks['same_seed_reproducible'] = bool(a == b)
            changed = {}
            for k in keys:
                if k not in a or k not in c:
                    continue
                if k == 'tensorflow' and 'tensorflow_error' in a:
                    changed[k] = None
                    continue
                changed[k] = bool(a[k] != c[k])
            checks['different_seed_changes_streams'] = bool(all(v is not False for v in changed.values()) and changed)
        except Exception as e:
            errors['rng_draws'] = f'{type(e).__name__}: {e}'
            a = b = c = {}
            changed = {}
            checks['same_seed_reproducible'] = False
            checks['different_seed_changes_streams'] = False
        try:
            w1 = _worker_draws(seed)
            w2 = _worker_draws(seed)
            w3 = _worker_draws(_alt_seed)
            checks['worker_seed_reproducible'] = bool(w1 == w2)
            checks['worker_different_seed_changes'] = bool(w1 != w3)
        except Exception as e:
            errors['worker_seed'] = f'{type(e).__name__}: {e}'
            w1 = w2 = w3 = {}
            checks['worker_seed_reproducible'] = False
            checks['worker_different_seed_changes'] = False
        checks['cuda_seeded'] = bool('cuda' in a) if torch.cuda.is_available() else None
        checks['cuda_devices_seeded'] = (bool('cuda_devices' in a and len(a.get('cuda_devices') or {}) == int(torch.cuda.device_count())) if torch.cuda.is_available() else None)
        checks['tensorflow_seeded'] = bool('tensorflow' in a and 'tensorflow_error' not in a) if sys.modules.get('tensorflow') is not None else None
        sub = None
        if check_subprocess:
            code = (
                'import json,random,numpy as np,torch;'
                'import exp_lib as L;'
                'L.set_seed(%d);'
                'print(json.dumps([random.random(),float(np.random.random()),float(torch.rand(1).item())]))' % seed
            )
            env = dict(os.environ)
            env['PYTHONHASHSEED'] = str(seed)
            try:
                outs = [subprocess.check_output([sys.executable, '-c', code], cwd=os.path.dirname(os.path.abspath(__file__)), env=env, text=True, timeout=180).strip().splitlines()[-1] for _ in range(2)]
                sub = {'same_seed_reproducible': bool(outs[0] == outs[1]), 'draws': outs}
                checks['subprocess_seed_reproducible'] = sub['same_seed_reproducible']
            except Exception as e:
                errors['subprocess_seed'] = f'{type(e).__name__}: {e}'
                checks['subprocess_seed_reproducible'] = False
        dl_contract = None
        if check_dataloader_workers:
            # Actual worker PROCESSES, not just the hook's draws: this catches
            # a DataLoader created without the seeded generator/worker_init_fn.
            dl_contract = dataloader_contract_check(
                seed, num_workers=dataloader_workers)
            checks['dataloader_worker_processes'] = bool(dl_contract.get('passed'))
            if not dl_contract.get('passed'):
                errors['dataloader_worker_processes'] = str(dl_contract.get('diagnosis'))
    _fp_after = rng_fingerprint()
    _rng_preserved = bool(_fp_before == _fp_after)
    checks['rng_state_preserved'] = _rng_preserved
    _hash_contract = hashseed_contract()
    # Not a pass/fail check: a plain `python driver.py` launch is
    # hash-randomised before exp_lib can import.  Record the residual gap so a
    # reviewer can see whether this run was launched reproducibly.
    warnings = []
    if not _hash_contract['pinned_at_startup']:
        warnings.append(_hash_contract['note'])
    failed = [k for k, ok in checks.items() if ok is False]
    passed = not failed and not errors
    report = {
        'kind': 'rng_seed_contract',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'code': code_stamp(),
        'seed': seed,
        'device': str(DEVICE if device is None else device),
        'determinism': determinism_label(),
        'streams': keys,
        'checks': checks,
        'different_seed_changed': changed,
        'draws_same_seed': a,
        'draws_other_seed': c,
        'worker_draws_a': w1,
        'worker_draws_b': w2,
        'worker_draws_other_seed': w3,
        'subprocess': sub,
        'dataloader_contract': dl_contract,
        'hashseed': _hash_contract,
        'warnings': warnings,
        'rng_fp_before': _fp_before,
        'rng_fp_after': _fp_after,
        'rng_state_preserved': _rng_preserved,
        'errors': errors,
        'passed': bool(passed),
        'diagnosis': ('ok' if passed else 'failed/errored RNG streams: ' + ', '.join(failed + sorted(errors))),
    }
    return report

def _pin_cpu_threads():
    global _CPU_THREADS_PINNED
    if _CPU_THREADS_PINNED is None:
        want = os.environ.get('CSA_CPU_THREADS', '').strip()
        try:
            n = max(1, int(want)) if want else 1
        except ValueError:
            print(f'[determinism] CSA_CPU_THREADS={want!r} is not an integer — using 1')
            n = 1
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        os.environ.setdefault('MKL_NUM_THREADS', str(n))
        _CPU_THREADS_PINNED = n
    else:
        n = _CPU_THREADS_PINNED
    try:
        if int(torch.get_num_threads()) != int(n):
            torch.set_num_threads(n)
    except Exception as e:
        print(f'[determinism] could not pin the CPU thread count to {n} ({type(e).__name__}: {e}) — a CPU run may not be reproducible')
    return n

def pin_tf32(enabled=None):
    """Pin TF32 explicitly instead of inheriting a framework default.

    Reproducibility depends on the actual matmul precision mode; two boxes (or
    two torch versions) with different defaults can otherwise disagree while
    looking identically configured.  ``CSA_TF32=1`` opts in; the default is
    off (highest float32 precision)."""
    if enabled is None:
        enabled = os.environ.get('CSA_TF32', '').strip().lower() in ('1', 'true', 'yes', 'on')
    enabled = bool(enabled)
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
    except Exception as e:
        print(f'[determinism] could not pin TF32 ({type(e).__name__}: {e})')
    try:
        torch.set_float32_matmul_precision('high' if enabled else 'highest')
    except Exception as e:
        print(f'[determinism] could not set float32_matmul_precision ({type(e).__name__}: {e})')
    return enabled

def _pin_cuda_determinism():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pin_tf32()
    mode = os.environ.get('CSA_DETERMINISTIC', '').strip().lower()
    if mode in ('1', 'true', 'yes', 'strict'):
        try:
            torch.use_deterministic_algorithms(True)
            return 'strict'
        except Exception as e:
            print(f"[determinism] CSA_DETERMINISTIC=1 requested but the strict mode could not be enabled ({type(e).__name__}: {e}); falling back to cudnn-deterministic only.  The active level is recorded in each record's `run_cfg`.")
            return 'cudnn'
    if mode in ('warn', 'warn_only'):
        torch.use_deterministic_algorithms(True, warn_only=True)
        return 'warn'
    return 'cudnn'

def determinism_label():
    if torch.are_deterministic_algorithms_enabled():
        return 'strict' if not torch.is_deterministic_algorithms_warn_only_enabled() else 'warn'
    if not torch.cuda.is_available():
        return 'cpu'
    return 'cudnn' if torch.backends.cudnn.deterministic else 'off'

def _hashseed_pinned_value(value):
    """Whether a PYTHONHASHSEED value actually fixes hash randomisation.

    CPython accepts only a decimal uint32 or the literal ``random``; the latter
    deliberately re-randomises the process, so it is not a pin even though the
    environment variable is present.
    """
    if value is None:
        return False
    text = str(value).strip()
    if not text or text.lower() == 'random':
        return False
    try:
        seed = int(text, 10)
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 <= seed <= 2 ** 32 - 1


def hashseed_contract():
    """Report whether the CURRENT interpreter was started hash-seeded.

    ``PYTHONHASHSEED`` is consumed by CPython before any repository module is
    imported, so ``set_seed`` cannot repair an interpreter that was launched
    without it.  The experiment code is written to avoid set/dict iteration
    order, but a future data path must not silently rely on that promise.
    This report separates:

    * ``pinned_at_startup`` — a valid numeric seed was present before
      ``exp_lib`` imported (a re-launched/CI/par.py child has this true);
      ``PYTHONHASHSEED=random`` is not a pin;
    * ``child_seed`` — what every child process we spawn will inherit;
    * ``effective_current_process`` — ``fixed`` only when the startup pin was
      present, else ``randomised``.

    It is evidence, not a hard gate: a normal ``python driver.py`` invocation
    has always been hash-randomised, and failing the whole panel on that would
    be a false alarm.  ``seed_contract_check`` records it alongside the other
    RNG streams and a confirmation driver can require it explicitly.
    """
    env_at_startup = _PYTHONHASHSEED_AT_STARTUP
    # A valid numeric startup value is the actual pin; `random` is explicitly
    # not.  `sys.flags.hash_randomization` is recorded separately as evidence
    # rather than folded into the pin test, so a unit test can exercise this
    # branch by setting the captured startup value in the current process.
    pinned = _hashseed_pinned_value(env_at_startup)
    return {
        'pinned_at_startup': bool(pinned),
        'value_at_startup': env_at_startup,
        'child_seed': os.environ.get('PYTHONHASHSEED'),
        'sys_hash_randomization': bool(getattr(sys.flags, 'hash_randomization', 0)),
        'effective_current_process': ('fixed' if pinned else 'randomised'),
        'note': ('the current interpreter was launched with a fixed PYTHONHASHSEED'
                 if pinned else
                 'the current interpreter was launched WITHOUT PYTHONHASHSEED; '
                 'CPython hash randomisation cannot be changed after startup. '
                 'Child processes are pinned, and this experiment does not depend '
                 'on hash iteration order, but launch with PYTHONHASHSEED=0 for '
                 'bit-identical Python-level set/dict ordering across runs.'),
    }

def _jsonable(v):
    """Recursively convert a config/provenance value to JSON-safe data.
    Sets are sorted so two equal configs always serialise identically."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted((_jsonable(x) for x in v), key=str)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def _git_provenance():
    """HEAD / branch / dirty state, with a bounded dirty-file list.  Never
    raises: a source tarball without .git is still a valid experiment."""
    def _run(*args):
        try:
            p = subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=20)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    head = _run('rev-parse', 'HEAD')
    branch = _run('rev-parse', '--abbrev-ref', 'HEAD')
    dirty = _run('status', '--porcelain')
    files = dirty.splitlines() if dirty else []
    # A commit id alone does NOT identify a dirty work tree: two runs from the
    # same HEAD with different uncommitted edits would otherwise claim the
    # same code version.  Hash the diff and the status instead of storing the
    # (possibly secret-bearing) contents.  Untracked files are included via
    # the status hash; their contents are left to the operator to archive.
    def _sha(text):
        return hashlib.sha256((text or '').encode('utf-8', 'replace')).hexdigest()[:16]
    diff_text = _run('diff', '--no-color', '--no-ext-diff', 'HEAD')
    return {'git_commit': head, 'git_branch': branch, 'git_dirty': bool(files), 'git_dirty_count': len(files), 'git_dirty_files': files[:20], 'git_status_sha256': _sha(dirty), 'git_diff_sha256': _sha(diff_text)}

def _gpu_names():
    try:
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        return []

def _gpu_identity():
    """Per-visible-GPU identity used to explain (not eliminate) cross-machine
    drift: name, UUID, device capacity and compute capability.  Never raises."""
    out = []
    if not torch.cuda.is_available():
        return out
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        return out
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            out.append({
                'index': i,
                'name': getattr(p, 'name', None),
                'uuid': str(getattr(p, 'uuid', '')) or None,
                'total_memory_bytes': int(getattr(p, 'total_memory', 0)) or None,
                'major': int(getattr(p, 'major', 0)),
                'minor': int(getattr(p, 'minor', 0)),
            })
        except Exception as e:
            out.append({'index': i, 'error': f'{type(e).__name__}: {e}'})
    return out

def _dependency_fingerprint():
    """Content hashes of the dependency declarations plus installed versions.

    The git commit identifies the source but not the environment.  Two runs
    from the same commit can silently differ because a floating dependency
    range resolved to a newer wheel; recording the lock/manifest hashes and
    the installed package versions makes that visible in every provenance
    block without changing the content-based resume fingerprint."""
    root = os.path.dirname(os.path.abspath(__file__))
    files = {}
    for name in ('requirements.txt', 'requirements.lock', 'requirements-ci.txt',
                 'pyproject.toml'):
        path = os.path.join(root, name)
        try:
            with open(path, 'rb') as fh:
                data = fh.read()
            files[name] = {'sha256': hashlib.sha256(data).hexdigest()[:16], 'bytes': len(data)}
        except OSError:
            files[name] = None
    packages = {}
    try:
        from importlib import metadata as _md
        for pkg in ('torch', 'numpy', 'tokenizers', 'datasets', 'matplotlib', 'tensorflow'):
            try:
                packages[pkg] = _md.version(pkg)
            except Exception:
                packages[pkg] = None
    except Exception as e:
        files['_error'] = f'{type(e).__name__}: {e}'
    return {'files': files, 'packages': packages}

# ---------------------------------------------------------------------------
# Source-tree identity
#
# A git commit identifies the source only while the work tree is clean AND the
# reader still has .git.  The frozen-manifest contract deliberately binds
# CODE_SEMANTICS, not the bytes of exp_lib.py (see the continuous-loop header),
# and a source tarball / dirty work tree has no commit identity at all.  The
# source-tree digest is the content-level complement: it hashes exactly the
# files that can change a measurement or its analysis, so an artifact can be
# tied to the code that produced it without trusting git.  It is evidence, not
# a semantic stamp; it never enters run_cfg / the resume fingerprint.
# ---------------------------------------------------------------------------
_SOURCE_TREE_HASH_CACHE = {}
_SOURCE_TREE_CACHE_CAP = 512
_SOURCE_TREE_CONFIG_FILES = (
    # requirements.lock is deliberately NOT here: it is a machine-specific
    # environment record (README tells every machine to regenerate it), so
    # folding it into the source digest would make the same source look
    # different on a CPU test box and a CUDA experiment box.  The lock's hash
    # and contract live in environment_snapshot() instead.
    'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
    'confirmation_frozen.json', 'confirmation_frozen_dynamic.json',
)
_SOURCE_TREE_SKIP_FILES = ('conftest.py',)
_SOURCE_TREE_SKIP_DIRS = {
    '.git', '.github', '.venv', 'venv', 'env', 'ENV', '__pycache__',
    '.ruff_cache', '.pytest_cache', '.mypy_cache', '.cache', 'node_modules',
    'build', 'dist', 'site-packages', 'archive', 'par_jobs',
    'wt103_cache', 'fineweb_cache', 'wt103_raw', 'datasets', 'data', 't',
    # `make install-check` populates .build/install-check with a full copy of
    # the source, and `make build` writes .build/dist.  Hashing that scratch
    # tree made the "measurement source" depend on whether a packaging check
    # had run: the freeze lock would refuse an unchanged tree, reproduce-
    # compare would report code_changed, and archives would pack duplicate
    # copies.  These are generated build/venv trees, never measurement source.
    '.build', '.tox', '.nox', '.eggs', '.hypothesis',
}
_SOURCE_TREE_SKIP_SUFFIXES = ('.egg-info',)
_SOURCE_TREE_SKIP_PREFIXES = ('results_', 'analysis_')


def _source_tree_rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, '/')


def _source_file_sha256(path, st=None):
    """sha256 of one source file, memoised on the full stat identity."""
    global _SOURCE_TREE_HASH_CACHE
    if st is None:
        st = os.stat(path)
    key = (os.path.abspath(path), st.st_dev, st.st_ino, st.st_size,
           st.st_mtime_ns, st.st_ctime_ns)
    ent = _SOURCE_TREE_HASH_CACHE.get(key)
    if ent is not None:
        return ent
    digest = file_sha256(path)
    if len(_SOURCE_TREE_HASH_CACHE) >= _SOURCE_TREE_CACHE_CAP:
        _SOURCE_TREE_HASH_CACHE.clear()
    _SOURCE_TREE_HASH_CACHE[key] = digest
    return digest


def _source_tree_paths(root):
    """Deterministic list of source/config files under ``root``.

    Generated artifact directories are pruned, not walked: a 1M-file results
    tree must not turn a provenance call into an I/O sweep.  README files are
    intentionally outside the scope (they document the project but cannot
    change a measurement; the operational tree is what this digest binds).

    Symlinked files are included only when their target stays inside ``root``.
    A link to an unrelated external file must not make the source digest (and
    every artifact that records it) depend on bytes outside the repository.
    """
    root = os.path.abspath(root)
    root_real = os.path.realpath(root)
    out = []
    seen = set()

    def _inside_root(path):
        try:
            return os.path.commonpath([root_real, os.path.realpath(path)]) == root_real
        except (OSError, ValueError):
            return False

    for rel in _SOURCE_TREE_CONFIG_FILES:
        p = os.path.join(root, rel)
        if os.path.isfile(p) and _inside_root(p):
            out.append(p)
            seen.add(os.path.abspath(p))
    for cur, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs
            if d not in _SOURCE_TREE_SKIP_DIRS
            and not any(d.endswith(sfx) for sfx in _SOURCE_TREE_SKIP_SUFFIXES)
            and not any(d.startswith(pref) for pref in _SOURCE_TREE_SKIP_PREFIXES)
        )
        for n in sorted(names):
            if not n.endswith('.py'):
                continue
            if n.startswith('test_') or n in _SOURCE_TREE_SKIP_FILES:
                # Tests exercise the measurement code; they do not define it.
                # Keeping them out of the digest means a new CPU test cannot
                # masquerade as a semantic change to a frozen panel.
                continue
            p = os.path.join(cur, n)
            ap = os.path.abspath(p)
            if ap in seen or not os.path.isfile(p) or not _inside_root(p):
                continue
            seen.add(ap)
            out.append(p)
    return sorted(out, key=lambda p: _source_tree_rel(p, root))


def source_tree_manifest(root=None, include=None):
    """Content manifest of the measurement-relevant source tree.

    ``include`` lets a test/audit restrict the manifest to explicit relative
    paths.  The full manifest is the durable form written into a plan snapshot;
    per-record provenance uses ``source_tree_summary`` so it stays small.
    """
    root = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    root_real = os.path.realpath(root)
    if include is None:
        paths = _source_tree_paths(root)
    else:
        # A bare string is one path, not an iterable of characters.  Repeating
        # the same file must not count its bytes twice (the per-file map keeps
        # one entry), and a symlink may not smuggle in an out-of-tree file.
        if isinstance(include, (str, bytes)):
            include = [include]
        paths = []
        _seen_rel = set()
        for rel in include:
            rel = os.fsdecode(rel) if isinstance(rel, bytes) else str(rel)
            rel = rel.replace('\\', '/')
            p = os.path.join(root, rel)
            ap = os.path.abspath(p)
            if os.path.commonpath([root, ap]) != root:
                raise ValueError(f'source_tree_manifest: {rel!r} escapes {root!r}')
            if not os.path.isfile(ap):
                continue
            try:
                _inside = os.path.commonpath([root_real, os.path.realpath(ap)]) == root_real
            except (OSError, ValueError):
                _inside = False
            if not _inside:
                raise ValueError(
                    f'source_tree_manifest: {rel!r} resolves outside {root!r}')
            rel_norm = _source_tree_rel(ap, root)
            if rel_norm in _seen_rel:
                continue
            _seen_rel.add(rel_norm)
            paths.append(ap)
        paths = sorted(paths, key=lambda p: _source_tree_rel(p, root))
    files = {}
    unreadable = []
    for p in paths:
        rel = _source_tree_rel(p, root)
        try:
            st = os.stat(p)
            digest = _source_file_sha256(p, st)
        except OSError as e:
            unreadable.append({'path': rel, 'error': f'{type(e).__name__}: {e}'})
            continue
        files[rel] = {'sha256': digest, 'bytes': int(st.st_size)}
    # Derive the total from the deduplicated per-file map: duplicate include
    # entries (or two spellings of one path) must not inflate `bytes`.
    total = sum(int(entry['bytes']) for entry in files.values())
    # Hash paths, sizes and per-file digests in sorted order.  Including the
    # path makes a rename visible even when the bytes are unchanged; including
    # the size makes truncation visible even before a digest is compared.
    blob = ''.join(f'{rel}\0{files[rel]["bytes"]}\0{files[rel]["sha256"]}\n'
                   for rel in sorted(files))
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()
    return {
        'schema': 1,
        'algorithm': 'sha256(sorted relpath NUL bytes NUL sha256 LF)',
        'sha256': digest,
        'n_files': len(files),
        'bytes': int(total),
        'files': files,
        'unreadable': unreadable,
        'scope': ('all non-generated, non-test *.py under the repo root plus '
                  'pyproject/setup/requirements.txt and both frozen manifests; '
                  'requirements.lock (machine-specific environment), README*.md, '
                  'docs, tests and CI configuration are excluded'),
    }


def source_tree_summary(root=None):
    """Compact source-tree identity for per-record provenance."""
    m = source_tree_manifest(root=root)
    out = {k: m[k] for k in ('schema', 'algorithm', 'sha256', 'n_files',
                            'bytes', 'scope')}
    if m.get('unreadable'):
        out['unreadable'] = m['unreadable']
    return out


_LOCK_ENV_KEYS = ('python-version', 'python-implementation', 'platform')


def parse_requirements_lock(path=None):
    """Parse a ``name==version`` lock without importing packaging.

    Options (e.g. ``--extra-index-url``) are recorded but ignored for the
    version comparison; ambiguous/duplicate pins are an error rather than
    "last line wins", because that would silently change the environment a
    release was verified against.  Machine-readable interpreter clauses
    (``# python-version:``, ``# python-implementation:``, ``# platform:``)
    are collected in ``environment``; every other comment is ignored.  A
    legacy lock without the clauses is accepted but reports the interpreter
    as unverified rather than inventing a match.
    """
    path = os.path.abspath(path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'requirements.lock'))
    out = {'path': path, 'exists': os.path.isfile(path), 'pins': {},
           'options': [], 'invalid': [], 'sha256': None, 'environment': {}}
    if not out['exists']:
        return out
    with open(path, 'rb') as fh:
        raw = fh.read()
    out['sha256'] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8', 'replace')
    _pin_re = re.compile(
        r'^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;]+)$')
    _env_re = re.compile(
        r'^#\s*(' + '|'.join(re.escape(k) for k in _LOCK_ENV_KEYS)
        + r')\s*:\s*(.*?)\s*$', re.IGNORECASE)
    for line in text.splitlines():
        _comment = line.strip()
        if _comment.startswith('#'):
            _em = _env_re.match(_comment)
            if _em:
                _ek = _em.group(1).lower()
                _ev = _em.group(2).strip()
                if not _ev:
                    out['invalid'].append(
                        f'empty environment declaration: {_ek}')
                elif _ek in out['environment'] and out['environment'][_ek] != _ev:
                    out['invalid'].append(
                        f'duplicate/conflicting environment declaration: {_ek}')
                else:
                    out['environment'][_ek] = _ev
            continue
        s = line.split('#', 1)[0].strip()
        if not s:
            continue
        if s.startswith('-'):
            out['options'].append(s)
            continue
        # `pip-compile --generate-hashes` puts the hashes on continuation
        # lines (which start with `--hash`, handled above) and may leave a
        # trailing backslash on the pin itself.  Accept both forms so a
        # hash-pinned lock is compared, not silently rejected as malformed.
        s = s.rstrip('\\').strip().split(' --hash', 1)[0].strip()
        m = _pin_re.match(s.split(';', 1)[0].strip())
        if not m:
            out['invalid'].append(s)
            continue
        name = re.sub(r'[-_.]+', '-', m.group(1)).lower()
        ver = m.group(2).strip()
        if name in out['pins']:
            # Even an identical duplicate makes the lock ambiguous (the two
            # lines could acquire different markers/hashes later); the
            # contract is one unambiguous pin per distribution.
            if out['pins'][name]['version'] != ver:
                out['invalid'].append(f'duplicate/conflicting pin: {s}')
            else:
                out['invalid'].append(f'duplicate pin: {s}')
            continue
        out['pins'][name] = {'name': m.group(1), 'version': ver}
    return out


def _interpreter_contract(declared=None):
    """Compare the lock's interpreter clauses with the running interpreter.

    Package pins alone do not identify a numerical stack: the same wheels can
    run on a different Python patch or platform.  A legacy lock without the
    clauses is ``unverified`` (never a silent match).  Only clauses that are
    actually declared are compared; the caller decides how to report it.
    """
    import platform as _platform
    declared = declared if isinstance(declared, dict) else {}
    expected = {}
    for key in _LOCK_ENV_KEYS:
        value = declared.get(key)
        if isinstance(value, str) and value.strip():
            expected[key] = value.strip()
    actual = {
        'python-version': _platform.python_version(),
        'python-implementation': _platform.python_implementation(),
        'platform': f'{_platform.system()} {_platform.machine()}'.strip(),
    }
    problems = []
    for key, want in expected.items():
        got = actual.get(key)
        if str(got) != str(want):
            problems.append(f'{key}: lock declares {want!r}, running {got!r}')
    missing = [key for key in _LOCK_ENV_KEYS if key not in expected]
    if not expected:
        status = 'unverified'
        note = ('the lock does not declare python-version / '
                'python-implementation / platform; the interpreter identity '
                'cannot be verified from the lock')
    elif problems:
        status = 'mismatch'
        note = 'the running interpreter/platform does not match the lock'
    else:
        status = 'match'
        if missing:
            note = ('the declared interpreter/platform clauses match, but the '
                    'lock does not declare ' + ', '.join(missing)
                    + '; the environment is only partially verified')
        else:
            note = 'the running interpreter/platform matches the lock clauses'
    return {'declared': expected, 'actual': actual, 'status': status,
            'problems': problems, 'missing': missing, 'note': note}
