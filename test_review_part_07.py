"""CPU review checks (split group 07; t_methodology_gate_fail_closed_edges .. t_encoder_fit_audit_detects_holdout_vocabulary)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_methodology_gate_fail_closed_edges():
    # The report's local Holm twin must use the same scalar contract as the
    # library: a boolean is not a probability, and an invalid alpha is not a
    # threshold.
    for _p in (False, True, np.bool_(False), np.bool_(True), -0.1, 1.5, 'x'):
        assert BR.holm_adjust([('a', _p), ('b', 0.02)])['a'] == 1.0, _p
    for _bad in (True, np.bool_(True), 0, 1, -0.1, 1.5, float('nan')):
        for _fn in (lambda a: BR.holm_adjust([('a', 0.01)], alpha=a),
                    lambda a: BR.global_multiplicity_adjust([('a', 0.01)], alpha=a),
                    lambda a: L.holm_min_attainable_p(6, 4, a)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid alpha {_bad!r} accepted by a report twin')

    # A descriptor present on only SOME seeds of an arm is not equality: the
    # arm is unverifiable, and the fairness audit must fail instead of
    # trusting the populated subset.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=49.0, params=100,
                                train_knobs=(dict(knobs) if i else None))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=50.0, params=100,
                                train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={},
                            verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['train_knobs_equal'] is False, cv
    assert cv['partially_recorded'] == ['a'] and cv['controlled'] is False, cv
    assert any('only on SOME seed records' in w for w in r['warnings']), r['warnings']
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(s, f)
        with contextlib.redirect_stdout(io.StringIO()):
            fair = A.audit_fairness(path, arms=['a', 'b'])
    assert fair['status'] == 'fail', fair
    assert any('omit train_knobs' in p for p in fair['problems']), fair['problems']

    # A malformed (unhashable) params field must not crash the fairness audit.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'x': dict(s['a::seed0'], params={'bad': 1})}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            r_bad = A.audit_fairness(path, arms=['a'])
    assert r_bad['arms']['a']['n_records'] == 1, r_bad

    # Full-factorial completeness fails closed on malformed or duplicated
    # cells instead of dropping them and reporting a complete table.
    base = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC',
            ('A', 'B', 'C'): 'ABC'}
    assert L.ablation_completeness_audit(arms_by_combo=base)['complete'] is True
    dup = dict(base, **{'A+B': 'dup'})
    r = L.ablation_completeness_audit(arms_by_combo=dup)
    assert r['complete'] is False and r['duplicate_combos'], r
    bad = dict(base)
    bad[1] = 'x'
    r = L.ablation_completeness_audit(arms_by_combo=bad)
    assert r['complete'] is False and r['structural_errors'], r
    bad = dict(base)
    bad[('A', 'B', 'C')] = None
    r = L.ablation_completeness_audit(arms_by_combo=bad)
    assert r['complete'] is False and r['structural_errors'], r
    # Project-level coverage must understand full-factorial reports too: they
    # name missing CELLS (a list), not a per-component arm mapping.
    _fa = {(): 'none', ('A',): 'A', ('B',): 'B', ('A', 'B'): 'AB'}
    _fr = L.ablation_completeness_audit(arms_by_combo=_fa)
    assert _fr['complete'] is True, _fr
    assert set(_fr['complete_components']) == {'A', 'B'}, _fr
    _cov = L.ablation_coverage_across_panels(
        {'p': {'ablation_completeness': _fr,
               'measured_components': ['A', 'B'],
               'verified_components': ['A', 'B'],
               'unverified_components': []}},
        design={'A': [('A', 'none', '')], 'B': [('B', 'none', '')]})
    assert set(_cov['complete_anywhere']) == {'A', 'B'}, _cov
    assert set(_cov['usable_anywhere']) == {'A', 'B'}, _cov
    _fb = dict(_fa)
    _fb.pop(('A', 'B'))
    _fr2 = L.ablation_completeness_audit(arms_by_combo=_fb)
    _cov2 = L.ablation_coverage_across_panels(
        {'p': {'ablation_completeness': _fr2,
               'measured_components': [], 'verified_components': [],
               'unverified_components': ['A', 'B']}},
        design={'A': [('A', 'none', '')], 'B': [('B', 'none', '')]})
    assert _cov2['complete_anywhere'] == [], _cov2
    assert _cov2['components']['A']['missing_arms'].get('p'), _cov2

    # Leave-one-out mode must also reject a self-contrast, malformed arm ids
    # and a non-mapping design instead of treating them as a complete ablation.
    for _design in ({'A': [('full', 'full', '')]},
                    {'A': [('full', [])]},
                    {'A': None}):
        r = L.ablation_completeness_audit(_design, ['full'])
        assert r['complete'] is False and r['structural_errors'], (_design, r)
    r = L.ablation_completeness_audit([], ['full'])
    assert r['complete'] is False and r['structural_errors'], r

    # Wrapper gate parameters are validated even with no panels to audit.
    for _bad in (True, 0, 1.5, float('nan')):
        try:
            A.audit_methodology(outdirs=[], alpha=_bad, write=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_methodology accepted alpha={_bad!r}')
    for _bad in (True, 3.5, -1, float('nan')):
        try:
            A.audit_methodology(outdirs=[], min_n=_bad, write=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_methodology accepted min_n={_bad!r}')

    # A non-mapping summary is invalid evidence, not an AttributeError.
    assert L._variant_seed_map(1) == ({}, set())
    assert L._variant_seed_map([('a', {})]) == ({}, set())
    for _bad_summary in (1, True, 'x', [None], [1]):
        _mr = L.methodology_audit(_bad_summary, design={}, baselines={},
                                  write=False, verbose=False)
        assert any('summary is' in w for w in _mr['warnings']), (_bad_summary, _mr)

    # The report's own significance gate validates its minimum-n parameter.
    _st = {'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02,
           'mean': 1.0, 'std': 0.5}
    assert BR._sig_gate(_st) is True
    for _bad in (True, 3.5, 0, -1, float('nan')):
        assert BR._sig_gate(_st, min_n=_bad) is False, _bad

    # Capacity and compute tolerances are gate parameters: a bool/negative/
    # non-finite value must not silently disable the control.
    _ra = {'params': 1000}
    _rb = {'params': 1000}
    _recs = {('a', '', 0): {'params': 1000, 'train_time_s': 1.0, 'tokens_seen': 10},
             ('b', '', 0): {'params': 1000, 'train_time_s': 1.0, 'tokens_seen': 10}}
    for _bad in (True, -0.1, float('nan'), 'x'):
        for _fn in (lambda t: L.param_pair_reason(_ra, _rb, tolerance=t),
                    lambda t: L._compute_control_detail(_recs, 'a', 'b', tolerance=t)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid control tolerance {_bad!r} was accepted')

    # The final PPL is measured on `eval_subset`; two runs can share the full
    # val fingerprint yet score different subset rows.  Both pairing twins
    # must compare its content fingerprint.
    _kn = L.training_knobs()
    _a = {'run_cfg': 'r', 'steps': 1, 'tokens_seen': 1, 'train_knobs': _kn,
          'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e', 'eval_subset_fp': 's'}
    _b = dict(_a)
    assert L.pair_reason(_a, _b) is None and BR._pair_reason(_a, _b) is None
    _b['eval_subset_fp'] = 's2'
    assert 'eval_subset_fp' in (L.pair_reason(_a, _b) or ''), L.pair_reason(_a, _b)
    assert 'eval_subset_fp' in (BR._pair_reason(_a, _b) or ''), BR._pair_reason(_a, _b)
    _b.pop('eval_subset_fp')
    assert 'unverifiable eval_subset_fp' in (L.pair_reason(_a, _b) or '')
    assert 'unverifiable eval_subset_fp' in (BR._pair_reason(_a, _b) or '')

    # The report's displayed resolution floor is 2/2**n for EVERY n, not a
    # Monte-Carlo-like 1/200001 beyond n=20 (the sign-flip test is exact/MITM
    # up to n=40 and deterministic saddlepoint above).
    for _n in (6, 20, 21, 30, 40, 100):
        assert abs(BR._signflip_floor(_n) - math.ldexp(2.0, -_n)) < 1e-300
    for _bad in (0, -1, True, np.bool_(True), 'x'):
        assert not math.isfinite(BR._signflip_floor(_bad)), _bad

    # Scale pre-flight gates are not coerced either.
    def _tier(name, steps, params):
        cfg = {'steps': steps, 'batch_size': 2, 'seq_len': 8,
               'n_train_tokens': steps * 16, 'lr': 3e-4, 'warmup': 10,
               'seeds': [0, 1, 2, 3]}
        return {'name': name, 'cfg': cfg, 'params': params}
    for _bad in (True, -1.0, float('nan'), 'x'):
        try:
            L.scale_ladder_audit([_tier('a', 1000, 1000)], tolerance=_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'scale_ladder_audit accepted tolerance={_bad!r}')
    out = L.scale_ladder_audit(
        [_tier('a', 1000, 1000), _tier('b', 2000, 2000)],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=True))
    assert out['status'] == 'fail' and any('family_m' in p for p in out['problems']), out
    pts = [{'label': f'd{d}', 'd': d, 'crossover_tokens': 1e6,
            'train_tokens': 5e7} for d in (128, 256, 384, 512)]
    for _kw in ({'min_points': True}, {'min_points': 1}, {'min_points': 3.5},
                {'volume_tol': -0.1}, {'volume_tol': float('nan')},
                {'d_spread_min': 1.0}, {'d_spread_min': True}):
        try:
            L.crossover_scaling_fit(pts, **_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'crossover_scaling_fit accepted {_kw!r}')

    # Baseline tuning evidence uses the same exact-count / finite-spread
    # contract.  abs(NaN - s) > tol was False, so a corrupted ppl_std used to
    # PASS the per-seed std check; a boolean step budget compared equal to 1.
    def _tune_art(trial):
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better', 'search_seeds': [2, 3],
                'arms': {'a': {'trials': [trial],
                               'selected': {'lr': 1e-3},
                               'selected_metric_value': trial.get('ppl')}}}
    _trial = {'config': {'lr': 1e-3}, 'ppl': 1.5, 'seeds': [0, 1],
              'seed_ppls': {'0': 1.0, '1': 2.0}, 'ppl_std': float('nan'),
              'steps': 10, 'tokens_seen': 100}
    out = L.baseline_tuning_audit(_tune_art(_trial))
    assert any('ppl_std' in p for p in out['problems']), out['problems']
    _trial = dict(_trial, ppl_std=float(np.std([1.0, 2.0], ddof=1)), steps=True)
    out = L.baseline_tuning_audit(_tune_art(_trial))
    assert any('exact positive counts' in p for p in out['problems']), out['problems']


def test_baseline_seed_plan_rejects_duplicate_draws():
    # A duplicated seed identity is a planning error, not an extra replicate.
    # It used to be silently de-duplicated here, so the power floor could be
    # checked against a larger n than the records that actually ran.
    for kw in ({'search_seeds': [0, 0, 1], 'final_seeds': [2, 3]},
               {'search_seeds': [0, 1], 'final_seeds': [2, 2]}):
        proto = dict(L.BASELINE_TUNING_PROTOCOL, **kw)
        try:
            BT._search_final_seeds(proto)
        except ValueError as e:
            assert 'duplicate seed' in str(e), e
        else:
            raise AssertionError(f'duplicate declared seeds were accepted: {kw}')
    # Unique identities are normalised once (integer strings accepted) so the
    # driver and the audit consume the same plan.
    search, final, overlap = BT._search_final_seeds(
        {'search_seeds': [3, '1', 2], 'final_seeds': ['5', 4]})
    assert search == [1, 2, 3] and final == [4, 5] and overlap == [], (search, final, overlap)


def test_baseline_plan_preflight_and_tuned_hypothesis():
    # The shipped design must be launchable AND powerful enough for its
    # declared Holm family (four tuned contrasts need n>=8 final seeds).
    plan = L.baseline_plan()
    assert plan['status'] == 'ok' and plan['may_launch'], plan
    assert plan['confirmatory_capable'] is True, plan
    tuned = plan['tuned']
    assert tuned['family_size'] == 4 and len(tuned['declared_contrasts']) == 4, tuned
    assert tuned['search_final_seed_disjoint'] is True, tuned
    assert tuned['final_seeds_sufficient'] is True, tuned
    assert tuned['min_final_seeds_for_family_significance'] == 8, tuned
    # The design records an explicit baseline ladder per tuning group: each
    # method arm must have a comparison partner, and the matched subset must
    # actually sit inside the group.
    assert len(tuned['tuning_groups']) == 2, tuned
    for _g in tuned['tuning_groups']:
        assert _g['has_baseline'] is True, _g
        assert _g['has_matched_baseline'] is True, _g
        assert _g['method_arm'] not in _g['matched'], _g
        for _b in _g['matched']:
            assert _b in _g['arms'], _g
    no_partner = dict(L.BASELINE_TUNING_PROTOCOL, arms=['method'],
                      arm_groups=[{'arms': ['method'], 'method_arm': 'method',
                                   'matched': []}])
    pn = L.baseline_tuning_plan(no_partner)
    assert pn['status'] == 'fail' and any(
        'no baseline/control arm' in p for p in pn['problems']), pn
    outside = dict(L.BASELINE_TUNING_PROTOCOL,
                   arm_groups=[{'arms': list(L.BASELINE_TUNING_PROTOCOL['arms']),
                                'method_arm': 'csa_dynamic',
                                'matched': ['not_in_group']}])
    pout = L.baseline_tuning_plan(outside)
    assert pout['status'] == 'fail' and any(
        'not in this group' in p for p in pout['problems']), pout
    simple = plan['simple']
    assert simple['status'] == 'ok' and simple['expected_cells'] == {
        'unigram': 6, 'bigram': 36}, simple
    # Equal-search tuning is not itself the claim: the tuned-baseline
    # comparison is now a pre-registered, metric-bound hypothesis too.
    ha = L.hypothesis_audit(L.TUNED_BASELINE_HYPOTHESIS_DESIGN, 'tuned_baseline')
    assert ha['status'] == 'ok' and not ha['invalid'], ha
    for table in (L.HYPOTHESIS_DESIGN, L.BASELINE_HYPOTHESIS_DESIGN,
                  L.OPEN_BLOCK_HYPOTHESIS_DESIGN, L.FLOOR_HYPOTHESIS_DESIGN):
        for name, spec in table.items():
            assert spec.get('metric_policy') == 'lm_perplexity', (name, spec)
    # Structural defects fail before any GPU spend.
    overlap = dict(L.BASELINE_TUNING_PROTOCOL, search_seeds=[0],
                   final_seeds=[0, 1, 2, 3, 4, 5, 6, 7])
    po = L.baseline_tuning_plan(overlap)
    assert po['status'] == 'fail' and po['search_final_seed_disjoint'] is False, po
    dup = dict(L.BASELINE_TUNING_PROTOCOL, search_seeds=[100, 100])
    pd = L.baseline_tuning_plan(dup)
    assert pd['status'] == 'fail' and any('duplicate seed' in p for p in pd['problems']), pd
    same_fold = dict(L.BASELINE_TUNING_PROTOCOL,
                     selection_row_split={'enabled': True, 'seed': 1, 'n_folds': 2,
                                          'selection_fold': 1, 'evaluation_fold': 1})
    pf = L.baseline_tuning_plan(same_fold)
    assert pf['status'] == 'fail' and any('identical' in p for p in pf['problems']), pf
    no_group = dict(L.BASELINE_TUNING_PROTOCOL,
                    arm_groups=[{'arms': ['csa_dynamic'], 'method_arm': 'csa_dynamic'}])
    pg = L.baseline_tuning_plan(no_group)
    assert pg['status'] == 'fail' and any('not covered' in p for p in pg['problems']), pg
    # A resolvable-but-underpowered plan is descriptive, not structurally
    # invalid: it must be visible as such rather than silently launched as
    # confirmatory evidence.
    weak = dict(L.BASELINE_TUNING_PROTOCOL, search_seeds=[100, 101, 102],
                final_seeds=[0, 1, 2])
    pw = L.baseline_tuning_plan(weak)
    assert pw['status'] == 'warn' and pw['may_launch'] is True, pw
    assert pw['final_seeds_sufficient'] is False and pw['confirmatory_capable'] is False, pw
    # The count-model grid must be complete rather than a cherry-picked cell.
    bad_simple = dict(L.SIMPLE_BASELINE_POLICY,
                      grids={'unigram': {'alpha': [0.1]},
                             'bigram': {'alpha': [0.1]}})
    ps = L.simple_baseline_plan(bad_simple)
    assert ps['status'] == 'fail', ps
    assert any('missing' in p for p in ps['problems']), ps
    assert any('fewer than 2 levels' in w for w in ps['warnings']), ps


def test_bootstrap_and_zero_ce_scaling_contract():
    # Chunking must bound the per-chunk working set when the unit count grows;
    # ordinary panels keep the historical 256-draw chunk exactly.
    assert L._bootstrap_draw_chunk(100) == 256
    assert L._bootstrap_draw_chunk(4096) == 256
    assert L._bootstrap_draw_chunk(1 << 20) == 1
    assert L._bootstrap_draw_chunk(1 << 30) == 1
    # Zero cross-entropy is a valid measurement (PPL=1), not a missing
    # component.  Filtering `> 0` made a perfectly predicted panel
    # unverifiable and dropped the document bootstrap entirely.
    recs = [{'variant': 'csa', 'seed': s, 'ppl': 1.0,
             'nll_per_row': [0.0] * 4, 'n_tokens_per_row': [10] * 4}
            for s in (0, 1, 2)]
    r = L._combined_seed_doc_ci(recs)
    assert r['status'] == 'ok', r
    assert r['se_seed_logppl'] == 0.0 and r['se_doc_logppl'] == 0.0, r
    assert r['se_combined_logppl'] == 0.0 and r['ci95_ppl'] == [1.0, 1.0], r


def test_error_analysis_excludes_nonfinite_targets():
    # A divergent token (non-finite CE) is an unmeasured target.  Counting it
    # as CE=0 diluted every bucket and the pooled PPL, making a broken eval
    # look better than a measured one.
    vocab = 5

    class _StubModel:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2), dtype=torch.float32)

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab), dtype=torch.float32)

    vb = (np.arange(8, dtype=np.int64) % vocab).reshape(2, 4)
    orig = L.F.cross_entropy

    def _fake_ce(input, target, reduction='mean', **kw):
        out = torch.ones(tuple(target.shape), dtype=torch.float32)
        out[0, 0] = float('inf')
        return out

    L.F.cross_entropy = _fake_ce
    try:
        r = L.analyze_errors(_StubModel(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=2, chunk=2)
    finally:
        L.F.cross_entropy = orig
    assert r['n_nonfinite_tokens'] == 1, r
    assert r['n_targets'] == 5, r
    assert abs(r['mean_ce'] - 1.0) < 1e-12, r
    # Every bucket denominator excludes the unmeasured token.
    assert sum(b['n'] for b in r['buckets'].values() if b['n']) > 0, r['buckets']
    assert r['buckets']['position:first10%']['n'] == 1, r['buckets']
    assert all(seq['n_tokens'] in (2, 3) for seq in r['worst_sequences']), r['worst_sequences']
    # The whole-split cause tally covers exactly the finite targets.
    _all = r.get('cause_counts_all') or {}
    assert sum(v['n'] for v in _all.values()) == 5, _all
    assert _all['ambiguous_context']['n'] == 5, _all
    assert abs(sum(v['share'] for v in _all.values()) - 1.0) < 1e-12, _all
    # A real parameterised model (the only kind error_analysis.py loads) must
    # not hit "can't call numpy() on a Tensor that requires grad"; analysis
    # runs under inference mode and restores the model's training flag.
    torch.manual_seed(0)
    real = L.SmallGPT(17, 16, 1, 2, 8, 12, L.make_layer_cfgs(1, 'full')).train()
    vb_real = np.random.default_rng(0).integers(0, 17, (3, 12))
    r_real = L.analyze_errors(real, vb_real, 17, torch.device('cpu'),
                              top_n=5, top_tokens=100, chunk=2, topk=5)
    assert r_real['n_targets'] == 33 and real.training is True, r_real
    try:
        L.analyze_errors(real, vb_real, 17, torch.device('cpu'), top_tokens=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('a negative top_tokens budget must be refused')
    r_zero = L.analyze_errors(real, vb_real, 17, torch.device('cpu'), top_tokens=0)
    assert r_zero['worst_tokens'] == [], r_zero['worst_tokens']


def test_gapfill_numeric_reader_rejects_booleans():
    import run_gapfill as RG
    for _bad in (True, False, np.bool_(True), np.bool_(False)):
        assert RG._num_or_none(_bad) is None, _bad
    assert RG._num_or_none(3.0) == 3 and RG._num_or_none('4') == 4
    assert RG._num_or_none(1.5) is None


def test_gapfill_announces_family_seed_plan():
    import run_gapfill as RG
    assert set(RG.L.RUN_ABL['variants']) >= {'csa_dynamic', 'csa_dyn_nocomp'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        short = RG._announce_seed_plan('P-ABL', RG.L.RUN_ABL, [0, 1, 2])
    txt = buf.getvalue()
    # Three seeds cannot carry a 14-contrast Holm family; the phase must say so
    # before it runs rather than after its p-values are read.
    assert short['n_seeds'] == 3 and short['declared_contrasts'] >= 10, short
    assert short['sufficient_for_family_rejection'] is False, short
    assert 'WARNING' in txt and 'exploratory/descriptive' in txt, txt
    need = short['min_seeds_for_family_significance']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        enough = RG._announce_seed_plan('P-ABL', RG.L.RUN_ABL, list(range(need)))
    assert enough['sufficient_for_family_rejection'] is True, enough
    assert 'WARNING' not in buf.getvalue(), buf.getvalue()


def test_release_tag_cannot_escape_output_dir():
    import archive_release as AR
    assert AR._release_tag('v13') == 'v13'
    assert AR._release_tag('v13.2-rc_1') == 'v13.2-rc_1'
    for _bad in ('../v13', 'a/b', '', '.', '..', 'a b', 'a\nb', '-v13', 'x' * 65, None):
        try:
            AR._release_tag(_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'release tag {_bad!r} was accepted as a filename component')


def test_audit_leakage_missing_id_cache_is_not_ok():
    # Metas and persisted tokenizers can be present while the actual id
    # caches/sidecars are missing.  That split is NOT independently verifiable;
    # leaving the verdict at 'ok' would let a confirmation gate treat it as
    # clean.
    from tokenizers import Tokenizer, models
    cache = tempfile.mkdtemp(prefix='leak_missing_ids_')
    try:
        tok = Tokenizer(models.WordLevel({'[UNK]': 0, 'a': 1, 'b': 2},
                                         unk_token='[UNK]'))
        tok.save(os.path.join(cache, 'tok_v5_x.json'))
        tok.save(os.path.join(cache, 'test_tok_v2_x.json'))
        common = {'vocab': 3, 'fit_fp': 'FITXYZ'}
        with open(os.path.join(cache, 'meta_v5_x.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(common, src_fp='train'), f)
        with open(os.path.join(cache, 'test_meta_v2_x.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(common, src_fp='test'), f)
        out = A.audit_leakage(cache_dir=cache, shingle=1, max_rows_shingled=2)
        assert out['status'] != 'ok', out
        assert 'test id cache is absent' in out.get('note', ''), out
        assert 'document-id sidecar is missing' in out.get('note', ''), out
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def test_v12_binds_current_train_cache_to_frozen_stream():
    # P5T must not pair a newly rebuilt train/test cache with checkpoints
    # trained on a different token stream.
    d = tempfile.mkdtemp(prefix='v12_cache_bind_')
    try:
        for name, fp in (('meta_v5_a.json', 'FROZEN'), ('meta_v5_b.json', 'OTHER'),
                         ('meta_v5_c.json', None)):
            with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
                json.dump({} if fp is None else {'src_fp': fp}, f)
        assert V12._current_train_cache_fps(d) == {'FROZEN', 'OTHER'}
        assert V12._require_current_train_cache_fp('FROZEN', d) is True
        try:
            V12._require_current_train_cache_fp('MISSING', d)
        except SystemExit as e:
            assert 'different token stream' in str(e), e
        else:
            raise AssertionError('a mismatched frozen train fingerprint was accepted')
        empty = tempfile.mkdtemp(prefix='v12_cache_empty_')
        try:
            try:
                V12._require_current_train_cache_fp('FROZEN', empty)
            except SystemExit as e:
                assert 'no train-cache meta' in str(e), e
            else:
                raise AssertionError('a missing train cache was accepted')
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_release_archive_packs_tagged_cache_provenance():
    # The loaders write meta_v5_*.json / test_meta_v2_*.json / tok_v5_*.json;
    # a fixed set of legacy filenames packed none of them.
    import archive_release as AR
    d = tempfile.mkdtemp(prefix='release_cache_meta_')
    try:
        for name in ('meta_v5_sl512_cap8000000_v8192_vs512.json',
                     'test_meta_v2_sl4097_v8192_ns16.json',
                     'tok_v5_sl512_cap8000000_v8192_vs512.json',
                     'test_tok_v2_sl4097_v8192_ns16.json',
                     'train_ids_v5_x.npy.idsfp.json',
                     'train_ids_v5_x.npy', 'test_ids_v2_x.npy',
                     'unrelated.txt'):
            with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
                f.write('x')
        got = {os.path.basename(p) for p in AR._cache_provenance_files(d)}
        assert any(n.startswith('meta_v5_') for n in got), got
        assert any(n.startswith('test_meta_v2_') for n in got), got
        assert any(n.startswith('tok_v5_') for n in got), got
        assert any(n.startswith('test_tok_v2_') for n in got), got
        assert any(n.endswith('.idsfp.json') for n in got), got
        assert not any(n.endswith('.npy') for n in got), got
        assert 'unrelated.txt' not in got, got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_scale_corpus_fp_is_budget_independent():
    # The corpus/tokenizer identity must not change when a scale tier simply
    # trains longer.  `train_fp`/`data_version` intentionally DO change (they
    # identify the exact served bytes); `corpus_fp` must not.
    tr = (np.arange(64, dtype=np.int64) * 7 + 3) % 97
    cfg = {'dataset': 'unit', 'seq_len': 16, 'cache_dir': './unit_cache'}
    _old_cap = L.TOKENIZER_FIT_TOKENS
    try:
        L.TOKENIZER_FIT_TOKENS = 8
        a = L.experiment_provenance(dict(cfg, n_train_tokens=16),
                                    train_ids=tr[:16], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
        b = L.experiment_provenance(dict(cfg, n_train_tokens=48),
                                    train_ids=tr[:48], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
    finally:
        L.TOKENIZER_FIT_TOKENS = _old_cap
    assert a['corpus_fp'] == b['corpus_fp'], (a, b)
    assert a['data_version'] != b['data_version'], 'the served bytes/budget still differ'
    # A changed token inside the tokenizer-fit prefix is a different corpus.
    tr2 = tr.copy()
    tr2[0] = (int(tr2[0]) + 1) % 97
    try:
        L.TOKENIZER_FIT_TOKENS = 8
        c = L.experiment_provenance(dict(cfg, n_train_tokens=16),
                                    train_ids=tr2[:16], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
    finally:
        L.TOKENIZER_FIT_TOKENS = _old_cap
    assert c['corpus_fp'] != a['corpus_fp']
    # M0 prefers the stable fingerprint; a mixed provenance generation is
    # refused rather than letting the new subset speak for the whole panel.
    def _rec(cfp, dv):
        return {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
                'corpus_fp': cfp, 'data_version': dv}
    s1 = {'a': _rec(a['corpus_fp'], a['data_version'])}
    s2 = {'a': _rec(b['corpus_fp'], b['data_version'])}
    assert M0._panel_corpus_fp(s1) == a['corpus_fp']
    assert M0._panel_corpus_fp(s1) == M0._panel_corpus_fp(s2)
    legacy = {'a': {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
                    'data_version': a['data_version']}}
    assert M0._panel_corpus_fp(legacy) == a['data_version']
    try:
        M0._panel_corpus_fp({**s1, 'b': {'variant': 'full', 'seed': 0, 'ppl': 11.0,
                                         'data_version': a['data_version']}})
    except ValueError:
        pass
    else:
        raise AssertionError('mixed corpus_fp/legacy provenance must be refused')
    # The cross-scale consumers accept a shared stable fingerprint even when
    # the training budgets (data_version) differ.
    pts = [{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2], 'seeds': [0, 1],
            'corpus_fp': a['corpus_fp'], 'vocab': 97},
           {'label': 'S2', 'scale': 2, 'effects': [0.2, 0.3], 'seeds': [0, 1],
            'corpus_fp': b['corpus_fp'], 'vocab': 97}]
    assert L.meta_analyze_effects(pts)['status'] == 'ok', L.meta_analyze_effects(pts)


def test_signflip_zero_ties_lower_resolution_floor():
    # Zero deltas are sign-neutral: p is unchanged, but the attainable floor
    # is 2/2**(#nonzero), not 2/2**n.
    sf = L.exact_signflip_p([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert sf['n'] == 6 and sf['n_nonzero'] == 2, sf
    assert abs(sf['p_floor'] - 0.5) < 1e-15, sf
    assert abs(sf['p'] - 0.5) < 1e-15, sf
    st = L.paired_effect_stats([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], min_n=3)
    assert st['n'] == 6 and st['n_nonzero'] == 2, st
    assert abs(st['p_floor'] - 0.5) < 1e-15, st
    assert st['resolution_limited'] is True, st
    claim = L.reportable_claim(st, family_size=1)
    assert claim['p_floor'] == 0.5 and claim['n_nonzero'] == 2, claim
    assert claim['level'] == 'resolution_limited', claim
    all_zero = L.exact_signflip_p([0.0, 0.0, 0.0])
    assert all_zero['p'] == 1.0 and all_zero['p_floor'] == 1.0, all_zero
    assert L.paired_effect_stats([0.0] * 6)['p_floor'] == 1.0
    # The report twin must obey the same floor.
    z = BR.exact_signflip([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert z['n_nonzero'] == 2 and abs(z['p_floor'] - 0.5) < 1e-15, z
    # The shared v7-v12 helper used by the probe/final-test reports too.
    v = V7.exact_sign_permutation([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert v['n_nonzero'] == 2 and abs(v['p_floor'] - 0.5) < 1e-15, v


def test_zero_aware_family_resolution_floor():
    # Exact ties reduce the attainable family floor too: with n=8 but only 6
    # non-zero paired deltas, 4 * 2/2**6 = 0.125 > alpha, not 4 * 2/2**8.
    comps = {c: {'p_exact_signflip': 1.0, 'n': 8, 'n_nonzero': 6}
             for c in 'ABCD'}
    fam = L.attach_holm(comps, family='zero ties')
    assert abs(fam['family_min_attainable_p_holm'] - 0.125) < 1e-15, fam
    assert fam['family_resolution_limited'] is True, fam
    # Without the tie count the theoretical floor is still reported.
    none = {c: {'p_exact_signflip': 1.0, 'n': 8} for c in 'ABCD'}
    assert abs(L.attach_holm(none)['family_min_attainable_p_holm']
               - 0.03125) < 1e-15
    st = {'n': 8, 'n_nonzero': 6, 'p': 0.01, 'p_holm': 0.02,
          'mean': 1.0, 'std': 0.1}
    claim = L.reportable_claim(st, family_size=4)
    assert claim['allowed'] is False, claim
    assert abs(claim['min_attainable_p_holm_family'] - 0.125) < 1e-15, claim
    # A fabricated non-zero count larger than the panel cannot buy resolution.
    bad = L.reportable_claim(dict(st, n_nonzero=9), family_size=1)
    assert bad['allowed'] is False and any('n_nonzero=9 exceeds n=8' in r
                                           for r in bad['reasons']), bad
    # A malformed/negative tie count must not silently revert to the finer
    # all-nonzero floor; the family resolution is conservatively coarsened.
    fam_bad = L.attach_holm({'x': {'n': 8, 'n_nonzero': -1,
                                   'p_exact_signflip': 0.01}})
    assert fam_bad['family_resolution_limited'] is True, fam_bad
    assert fam_bad['family_min_attainable_p_holm'] == 1.0, fam_bad
    # Per-contrast floors: the family floor is m * min(floor), so a single
    # tie-heavy contrast blocks the whole tuned family.
    assert abs(L.holm_min_attainable_p_floors([0.03125, 0.5]) - 0.0625) < 1e-15
    assert L.holm_min_attainable_p_floors([1.0, 0.5]) == 1.0
    assert math.isnan(L.holm_min_attainable_p_floors([]))
    # Malformed floors are treated as 1 (the least resolvable value), so they
    # cannot shrink the family floor below the valid comparisons' minimum.
    assert abs(L.holm_min_attainable_p_floors([0.03125, None, 'x']) - 0.09375) < 1e-15
    assert L.holm_min_attainable_p_floors([None, 'x']) == 1.0


def test_scale_meta_refuses_visible_attrition():
    # A common effect (same mean at every scale) pools only on complete panels.
    def _common(planned=None, n_seeds=6):
        out = []
        for i in range(3):
            pt = {'label': 'S%d' % i, 'scale': 10.0 ** i,
                  'effects': [0.1 + 0.1 * j for j in range(n_seeds)],
                  'seeds': list(range(n_seeds))}
            if planned is not None:
                pt['n_planned'] = planned
            out.append(pt)
        return out
    complete = _common()
    m = L.meta_analyze_effects(complete)
    assert m['status'] == 'ok' and m['poolable'] is True, m
    assert m['incomplete_points'] == [], m
    # One planned seed absent from every point: the complete-case estimates
    # are numerically identical, but the missing draw must block the pooled
    # "common effect".
    missing = _common(planned=7)
    m2 = L.meta_analyze_effects(missing)
    assert m2['status'] == 'ok' and m2['poolable'] is False, m2
    assert m2['incomplete_points'] == ['S0', 'S1', 'S2'], m2
    assert m2['attrition']['S0']['n_missing'] == 1, m2['attrition']
    # A non-finite effect is a dropped pair too, not a smaller honest n.
    dropped = [dict(p, effects=list(p['effects'])) for p in complete]
    dropped[1]['effects'][0] = float('nan')
    m3 = L.meta_analyze_effects(dropped)
    assert m3['poolable'] is False and m3['incomplete_points'] == ['S1'], m3
    assert m3['attrition']['S1']['n_dropped'] == 1, m3
    # Declared exact seed plans also catch an absent seed.
    planned = [dict(p, planned_seeds=list(range(7))) for p in complete]
    m4 = L.meta_analyze_effects(planned)
    assert m4['poolable'] is False and m4['attrition']['S0']['n_missing'] == 1, m4

    # A perfectly linear ladder would normally be a significant trend.
    def _trend(planned=None):
        out = []
        for i in range(3):
            pt = {'label': 'S%d' % i, 'scale': 10.0 ** i,
                  'effects': [float(i) + 0.1 * j for j in range(6)],
                  'seeds': list(range(6))}
            if planned is not None:
                pt['n_planned'] = planned
            out.append(pt)
        return out
    r = L.scaling_trend(_trend())
    assert r['significant'] is True, r
    r2 = L.scaling_trend(_trend(planned=7))
    assert r2['significant'] is False and r2['verdict'] == 'incomplete_scale_point', r2
    assert 'incomplete paired panel' in r2['note'], r2
    # A perfectly consistent scale (zero within-scale variance) is reported,
    # not silently dropped from the family.  It blocks the pooled claim
    # because 1/se^2 is undefined without a variance model.
    zero = [{'label': 'S%d' % i, 'scale': 10.0 ** i,
             'effects': [0.9, 1.0, 1.1, 1.0], 'seeds': [0, 1, 2, 3]}
            for i in range(4)]
    zero[2] = {'label': 'S2', 'scale': 100.0,
               'effects': [1.0, 1.0, 1.0, 1.0], 'seeds': [0, 1, 2, 3]}
    mz = L.meta_analyze_effects(zero)
    assert mz['zero_variance_points'] == ['S2'], mz
    assert mz['n_usable'] == 3, 'the consistent point is not part of the DL weights'
    assert mz['poolable'] is False and 'zero within-scale variance' in mz['note'], mz
    assert len(mz['per_point']) == 4, 'the row stays visible'
    # A scale point whose every effect is non-finite is not deleted from the
    # family: it is named as empty and blocks both pooling and a trend.
    empty = [{'label': 'S%d' % i, 'scale': 10.0 ** i,
              'effects': [float(i) + 0.1 * j for j in range(6)],
              'seeds': list(range(6))} for i in range(3)]
    empty[1] = {'label': 'S1', 'scale': 10.0,
                'effects': [float('nan')], 'seeds': [0], 'n_planned': 6}
    me = L.meta_analyze_effects(empty)
    assert me['empty_points'] == ['S1'] and me['poolable'] is False, me
    # effects=None is also a declared empty tier, not a reason to delete it.
    none_eff = [dict(p) for p in complete]
    none_eff.append({'label': 'Snone', 'scale': 1000.0, 'effects': None,
                     'seeds': [0, 1, 2, 3]})
    mn = L.meta_analyze_effects(none_eff)
    assert mn['empty_points'] == ['Snone'] and mn['poolable'] is False, mn
    assert len(L._scale_points([{'label': 'Snone', 'scale': 1000.0,
                                 'effects': None}])) == 1
    # An n=1 tier has no within-scale SE and cannot enter the DL weights; it
    # must still block the pooled claim instead of silently disappearing.
    one_seed = [dict(p, n_planned=6) for p in complete]
    one_seed.append({'label': 'S3', 'scale': 1000.0, 'effects': [0.1],
                     'seeds': [0], 'n_planned': 6})
    mo = L.meta_analyze_effects(one_seed)
    assert mo['poolable'] is False and mo['underpowered_points'] == ['S3'], mo
    re = L.scaling_trend(empty, min_points=2)
    assert re['empty_points'] == ['S1'] and re['significant'] is False, re
    assert re['verdict'] == 'incomplete_scale_point', re
    # Invalid plan metadata fails closed instead of being silently truncated.
    for bad in (True, 3.5, -1, 'six'):
        try:
            L._scale_points([{'label': 'S', 'scale': 1, 'effects': [0.1],
                              'seeds': [0], 'n_planned': bad}])
        except ValueError:
            pass
        else:
            raise AssertionError(f'n_planned={bad!r} was accepted')
    # A declared tier whose scale is missing/non-positive cannot enter the
    # regression, but it must stay visible and block a confirmable trend
    # instead of silently disappearing from the ladder.
    ladder = [dict(p) for p in complete]
    ladder.append({'label': 'Sbad', 'scale': float('nan'),
                   'effects': [0.1] * 6, 'seeds': list(range(6))})
    mi = L.meta_analyze_effects(ladder)
    assert mi['invalid_scale_points'] == ['Sbad'] and mi['poolable'] is False, mi
    assert 'invalid/non-positive scale' in mi['note'], mi
    ri = L.scaling_trend(ladder)
    assert ri['invalid_scale_points'] == ['Sbad'], ri
    assert ri['significant'] is False and ri['verdict'] == 'incomplete_scale_point', ri
    assert 'Sbad' in ri['declared_labels'], ri
    assert 'invalid/non-positive scale' in ri['note'], ri
    # An exact planned seed list is not attestation by itself: without declared
    # seed identities, equal counts could be a different set of draws.
    unattested = [{'label': p['label'], 'scale': p['scale'],
                   'effects': list(p['effects']),
                   'planned_seeds': list(range(len(p['effects'])))}
                  for p in complete]
    raw = L._scale_points(unattested)
    assert all(r['complete_pairs'] is False
               and r['seed_plan_attested'] is False for r in raw), raw
    mu = L.meta_analyze_effects(unattested)
    assert mu['unattested_plan_points'] == ['S0', 'S1', 'S2'], mu
    assert mu['poolable'] is False, mu
    assert 'exact planned seed list' in mu['note'], mu
    ru = L.scaling_trend(unattested)
    assert ru['significant'] is False and ru['verdict'] == 'incomplete_scale_point', ru


def test_m0_cross_scale_uses_declared_seed_plans():
    fp = 'steps10_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _panel(seeds, base):
        out = {}
        for v in ('csa_dynamic', 'csa_fixed', 'hybrid_dynamic', 'full_matched'):
            for seed in seeds:
                out[f'{v}::seed{seed}'] = dict(
                    variant=v, seed=seed, run_cfg=fp, steps=10, tokens_seen=10,
                    ppl=base + 0.1 * seed + (0.5 if v == 'csa_fixed' else 0.0),
                    params=100, train_knobs=knobs, data_fp='d', val_fp='v',
                    corpus_fp='corpus')
        return out

    tmp = tempfile.mkdtemp()
    try:
        r = M0.scale_analysis({'S1': _panel([0, 1], 10.0),
                               'S2': _panel([0, 1], 20.0)}, outdir=tmp)
        meta = r['cross_scale']['dynamic_vs_fixed']['meta']
        # S1 plans 4 seeds and S2 plans 3; the truncated panels must be visible
        # as attrition, not certified complete from the two surviving records.
        got = {p['label']: p['n_missing'] for p in meta['per_point']}
        assert got == {'S1': 2, 'S2': 1}, got
        assert meta['poolable'] is False, meta
        assert set(meta['incomplete_points']) == {'S1', 'S2'}, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_m0_records_exclude_synthesized_reconstructions():
    base = {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
            'run_cfg': 'fp', 'steps': 1, 'tokens_seen': 1,
            'corpus_fp': 'corpus'}
    summary = {
        'real': dict(base),
        'synth': dict(base, seed=1, synthesized=True, ppl=9.0),
    }
    recs = M0._records_by_seed(summary, 'csa_dynamic')
    assert set(recs) == {0}, recs
    assert M0._panel_corpus_fp(summary) == 'corpus'
    assert M0._panel_corpus_fp({'synth': dict(base, synthesized=True,
                                             corpus_fp='other')}) is None


def test_m0_cross_scale_pairs_are_control_gated():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    fp2 = fp.replace('_lr0.0003_', '_lr0.003_')
    knobs = L.training_knobs()

    def _rec(v, seed, cfg, ppl):
        return {'variant': v, 'seed': seed, 'run_cfg': cfg, 'steps': 1,
                'tokens_seen': 1, 'ppl': ppl, 'train_knobs': knobs}
    summary = {
        'a::seed0': _rec('a', 0, fp, 10.0),
        'a::seed1': _rec('a', 1, fp, 10.5),
        'a::seed2': _rec('a', 2, fp, 11.0),
        'b::seed0': _rec('b', 0, fp, 11.0),
        'b::seed1': _rec('b', 1, fp2, 11.5),
        'b::seed2': _rec('b', 2, fp, 12.0),
    }
    paired, gated = M0._paired_ppl_by_seed(summary, 'a', 'b')
    assert sorted(paired) == [0, 2], paired
    assert len(gated) == 1 and gated[0][0] == 1, gated
    assert 'run_cfg' in gated[0][1], gated
    assert paired[0] == (10.0, 11.0), paired
    # A malformed nested provenance block has no corpus identity; it must not
    # crash the panel-fingerprint reader on a list .get().
    _bad_prov = _rec('a', 0, fp, 10.0)
    _bad_prov['provenance'] = {'data': ['not', 'a', 'mapping']}
    assert M0._panel_corpus_fp({'a::seed0': _bad_prov}, 'a') is None
    # Nested tuning records: cross-scale pairing must use the held-out primary
    # estimand, not the full-validation `ppl` that still contains the rows
    # which selected the hyper-parameters.  A nested record without holdout
    # accounting is dropped (visible as gated), never silently replaced.
    def _nested(v, seed, cfg, full, hold):
        r = _rec(v, seed, cfg, full)
        r.update({'ppl_estimation_rows': 'nested_validation_holdout',
                  'ppl_holdout': hold})
        return r
    nested = {
        'a::seed0': _nested('a', 0, fp, 99.0, 10.0),
        'a::seed1': _nested('a', 1, fp, 99.0, 10.5),
        'b::seed0': _nested('b', 0, fp, 1.0, 11.0),
        'b::seed1': {k: v for k, v in _nested('b', 1, fp, 1.0, 11.5).items()
                     if k not in ('ppl_holdout',)},
    }
    np_paired, np_gated = M0._paired_ppl_by_seed(nested, 'a', 'b')
    assert np_paired == {0: (10.0, 11.0)}, (np_paired, np_gated)
    # Seed 1's b record declares the nested estimand but has no holdout
    # accounting, so it is not a measurable pair; the raw full-validation
    # values (99 vs 1) must not re-enter the analysis.
    assert np_gated == [], np_gated
    # Escalating tier labels must not make S10 look smaller than S2.
    pts = [{'label': 'S10', 'scale': 1000.0}, {'label': 'S2', 'scale': 10.0},
           {'label': 'S1', 'scale': 1.0}]
    assert M0._extreme_scale_labels(pts) == ('S1', 'S10')
    # NumPy numeric scalars are valid declared scales too.
    assert M0._extreme_scale_labels([{'label': 'S10', 'scale': np.int64(1000)},
                                     {'label': 'S2', 'scale': np.float32(10)},
                                     {'label': 'S1', 'scale': 1}]) == ('S1', 'S10')
    assert M0._extreme_scale_labels([{'label': 'S9', 'scale': float('nan')},
                                     {'label': 'S1', 'scale': 1.0}]) is None


def test_scale_trend_degenerate_fit_and_empty_tiers():
    # A perfectly flat WLS line has zero residual variance.  That is not a
    # maximally significant increasing trend; it is no detectable trend.
    pts = [{'label': f'S{i}', 'scale': 10 ** i, 'effects': [0.1, 0.2, 0.3]}
           for i in range(3)]
    r = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert r['wls']['se'] == 0.0 and r['wls']['p'] == 1.0, r['wls']
    assert r['significant'] is False and r['verdict'] == 'no_detectable_trend', r
    # A declared-but-empty tier must not disable paired inference over the
    # scales that DO carry measured effects.
    pts = [{'label': 'A', 'scale': 1, 'effects': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'seeds': list(range(6))},
           {'label': 'B', 'scale': 10, 'effects': [], 'seeds': []},
           {'label': 'C', 'scale': 100, 'effects': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'seeds': list(range(6))}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert r['paired']['labels'] == ['A', 'C'], r['paired']
    assert r['paired']['n_complete_seeds'] == 6, r['paired']
    assert 'B' in r['empty_points'] and r['significant'] is False, r
    # Six same-sign per-seed slopes can reject the exact test while still
    # being within their own seed spread; that is not a usable trend claim.
    pts = [{'label': 'A', 'scale': 1, 'effects': [0.0] * 6, 'seeds': list(range(6))},
           {'label': 'B', 'scale': 10, 'effects': [0.01] * 3 + [2.0] * 3, 'seeds': list(range(6))},
           {'label': 'C', 'scale': 100, 'effects': [0.02] * 3 + [4.0] * 3, 'seeds': list(range(6))}]
    rn = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert rn['p_primary'] < 0.05, rn
    assert rn['paired']['within_seed_noise'] is True, rn['paired']
    assert rn['significant'] is False and rn['verdict'] == 'noise_level', rn


def test_scale_non_estimable_variance_blocks_pooling():
    big = 1e308
    pts = [{'label': 'BIG', 'scale': 1, 'effects': [big, -big], 'seeds': [0, 1]},
           {'label': 'S1', 'scale': 2, 'effects': [0.10, 0.11], 'seeds': [0, 1]},
           {'label': 'S2', 'scale': 3, 'effects': [0.1005, 0.1105], 'seeds': [0, 1]},
           {'label': 'S3', 'scale': 4, 'effects': [0.101, 0.111], 'seeds': [0, 1]}]
    m = L.meta_analyze_effects(pts, min_seeds=2)
    assert 'BIG' in m['non_estimable_points'], m
    assert m['poolable'] is False and m['n_usable'] == 3, m
    assert any('non-estimable' in m['note'] for _ in [0]), m['note']


def test_tuning_margin_does_not_invent_precision():
    # Missing marginal SD is unavailable uncertainty, not zero variance.
    m = L.tuning_selection_margin({'ppl': 10.0}, {'ppl': 10.001})
    assert m['se'] is None and m['decisive'] is None, m
    assert m['method'] != 'independent_marginal_se', m
    # When the paired path is used, the numerator must come from the SAME
    # common seeds as the paired SE, not from the full trial means.
    best = {'ppl': -333.3, 'ppl_std': 1.0, 'n_seeds': 3,
            'seed_ppls': {'0': 0.0, '1': 0.0, '2': -1000.0}}
    runner = {'ppl': 0.0, 'ppl_std': 1.0, 'n_seeds': 3,
              'seed_ppls': {'0': 100.0, '1': -100.0, '3': 0.0}}
    m = L.tuning_selection_margin(best, runner)
    assert m['method'] == 'paired_search_seeds' and m['n_paired'] == 2, m
    assert m['margin'] == 0.0 and m['decisive'] is False, m
    # Both rows expose per-seed evidence but share fewer than two draws: the
    # independent marginal SE is a different (unpaired) sample and must not
    # rescue a decisive verdict here.
    _lonely_b = {'ppl': 10.0, 'ppl_std': 0.1, 'n_seeds': 2,
                 'seed_ppls': {'0': 10.0, '1': 10.0}}
    _lonely_r = {'ppl': 9.0, 'ppl_std': 0.1, 'n_seeds': 2,
                 'seed_ppls': {'2': 9.0, '3': 9.0}}
    _lm = L.tuning_selection_margin(_lonely_b, _lonely_r)
    assert _lm['method'] == 'insufficient_common_seeds', _lm
    assert _lm['decisive'] is None and _lm['se'] is None, _lm


def test_scale_ladder_missing_seed_plan_is_not_common():
    def tier(name, seeds):
        return {'name': name, 'params': 1000,
                'cfg': {'steps': 1000, 'batch_size': 2, 'seq_len': 8,
                        'n_train_tokens': 16000, 'lr': 3e-4,
                        'warmup': 100, 'seeds': seeds}}
    out = L.scale_ladder_audit([tier('S1', list(range(8))), tier('S2', [])],
                               policy=dict(L.SCALE_TRANSFER_POLICY, family_m=4))
    assert out['n_common_seeds'] == 0, out
    assert out['seed_set_intersection'] == [], out
    assert out['common_seeds_sufficient_for_family'] is False, out


def test_small_n_split_remainder_and_string_order():
    # Guaranteeing one row for val/test must not then hand them the whole
    # remainder: 80/10/10 at n=8 is (6,1,1), not (4,1,3).
    for n, want in ((4, [2, 1, 1]), (5, [3, 1, 1]), (8, [6, 1, 1])):
        idx = L.chronological_split_indices(n)
        got = [len(idx[k]) for k in ('train', 'val', 'test')]
        assert got == want, (n, got)
    idx = L.chronological_split_indices(3, 0.5, 0.5, 0.0)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [1, 2, 0], idx
    idx = L.chronological_split_indices(4, 0.1, 0.8, 0.1)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [1, 2, 1], idx
    # Lexicographic string event ids remain valid ordering keys, even when
    # the literal text is 'nan' or 'inf'.
    idx = L.split_indices(3, time=np.array(['a', 'nan', 'b'], dtype=object),
                          train_frac=1 / 3, val_frac=1 / 3, test_frac=1 / 3)
    assert sorted(sum((list(idx[k]) for k in ('train', 'val', 'test')), [])) == [0, 1, 2], idx


def test_split_discipline_recomputes_fingerprints_and_doc_sidecars():
    with tempfile.TemporaryDirectory() as d:
        tag, ttag = 'tiny', 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 100).reshape(2, 8)
        te = (np.arange(16, dtype=np.int64) + 200).reshape(2, 8)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        meta = {'vocab': 256, 'seq_len': 8, 'split_strategy': 'ordered',
                'shuffle_at_split': False, 'tokenizer_fit_split': 'train',
                'src_fp': L._ids_fp(tr),
                'val_fp': L._ids_fp(np.ascontiguousarray(va).reshape(-1))}
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 256, 'seq_len': 8, 'split_strategy': 'official_test',
                       'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('train_fingerprint_verified') is True, rec
        assert rec.get('val_fingerprint_verified') is True, rec
        assert rec['status'] != 'fail', rec
        # A replaced validation file with stale metadata must fail closed.
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va + 1)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('validation id-cache fingerprint' in w for w in out['warnings']), out
        # Restore and add a test doc sidecar without the train/val sidecars:
        # substituting empty arrays would have reported a clean zero overlap.
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_doc_ids_{ttag}.npy'),
                np.array([9, 10], dtype=np.int64))
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('train/validation doc-id sidecars are missing' in w
                   for w in out['warnings']), out


def test_archive_refuses_links_and_repo_escape():
    import archive_release as AR
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as d:
        outside = tempfile.mkdtemp(prefix='archive_outside_')
        try:
            with open(os.path.join(outside, 'secret.txt'), 'w', encoding='utf-8') as f:
                f.write('secret')
            try:
                AR._collect([outside])
            except ValueError as e:
                assert 'outside the repository' in str(e), e
            else:
                raise AssertionError('an absolute directory outside the repo was packed')
            try:
                AR._collect(['../' + os.path.basename(outside)])
            except ValueError:
                pass
            else:
                raise AssertionError('a parent-relative directory was packed')
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        # A symlink inside the repository is still a link member: make would
        # hash its target but tar would store the link, so its own archive
        # could never verify.  Refuse it at collect time.
        _linkdir = tempfile.mkdtemp(dir=AR.REPO, prefix='archive_link_test_')
        try:
            with open(os.path.join(_linkdir, 'target.txt'), 'w',
                      encoding='utf-8') as f:
                f.write('target')
            os.symlink(os.path.join(_linkdir, 'target.txt'),
                       os.path.join(_linkdir, 'link.txt'))
            try:
                AR._collect([os.path.relpath(_linkdir, AR.REPO)])
            except ValueError as e:
                assert 'symbolic link' in str(e), e
            else:
                raise AssertionError('a symlink member was packed')
            # The checkpoint-body filter must not run before the symlink
            # check: a linked `model.pt` is skipped from packing (no bodies)
            # but still enters `_ckpt_hash_table`, which used to follow the
            # link and hash bytes outside the repository.
            os.symlink(os.path.join(_linkdir, 'target.txt'),
                       os.path.join(_linkdir, 'model.pt'))
            try:
                AR._collect([os.path.relpath(_linkdir, AR.REPO)],
                            include_ckpt=False)
            except ValueError as e:
                assert 'symbolic link' in str(e), e
            else:
                raise AssertionError('a symlinked checkpoint was packed/skipped')
            try:
                AR._ckpt_hash_table([os.path.relpath(_linkdir, AR.REPO)])
            except ValueError as e:
                assert 'symbolic link' in str(e), e
            else:
                raise AssertionError('a symlinked checkpoint was hashed')
        finally:
            shutil.rmtree(_linkdir, ignore_errors=True)
        # The release output directory is confined just like the tag and the
        # packed inputs: --out must not write outside the repository.
        for _bad_out in ('../escape', outside, os.path.join(outside, 'nested')):
            try:
                AR._safe_outdir(_bad_out)
            except ValueError:
                pass
            else:
                raise AssertionError(f'outdir {_bad_out!r} escaped the repository')
        assert AR._safe_outdir('archive') == os.path.join(
            os.path.realpath(AR.REPO), 'archive')
        # A symlink member with the expected name must not satisfy the
        # manifest; it was never hashed as a regular file.
        path = os.path.join(d, 'release_x.tar.gz')
        payload = b'hello'
        man = {'tag': 'x', 'files': {'data.txt': {
            'sha256': __import__('hashlib').sha256(payload).hexdigest(),
            'bytes': len(payload)}}}
        man_bytes = json.dumps(man).encode('utf-8')
        with tarfile.open(path, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            link = tarfile.TarInfo('link.txt')
            link.type = tarfile.SYMTYPE
            link.linkname = 'data.txt'
            tf.addfile(link)
            mt = tarfile.TarInfo('release_x.MANIFEST.sha256')
            mt.size = len(man_bytes)
            tf.addfile(mt, io.BytesIO(man_bytes))
        assert AR.cmd_verify(path) == 1
        # The same archive without the unhashed link still verifies.
        path2 = os.path.join(d, 'release_y.tar.gz')
        with tarfile.open(path2, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_y.MANIFEST.sha256')
            mt.size = len(man_bytes)
            tf.addfile(mt, io.BytesIO(man_bytes))
        assert AR.cmd_verify(path2) == 0
        # verify() must also refuse an archive whose member (or manifest path)
        # would escape the extraction directory, even if every hash matches.
        evil = os.path.join(d, 'release_z.tar.gz')
        evil_man = {'tag': 'z', 'files': {'../evil.txt': {
            'sha256': __import__('hashlib').sha256(payload).hexdigest(),
            'bytes': len(payload)}}}
        evil_bytes = json.dumps(evil_man).encode('utf-8')
        with tarfile.open(evil, 'w:gz') as tf:
            ti = tarfile.TarInfo('../evil.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_z.MANIFEST.sha256')
            mt.size = len(evil_bytes)
            tf.addfile(mt, io.BytesIO(evil_bytes))
        assert AR.cmd_verify(evil) == 1
        # A manifest that is not a files mapping is refused instead of
        # crashing the verifier with a KeyError.
        broken = os.path.join(d, 'release_broken.tar.gz')
        broken_bytes = b'[]'
        with tarfile.open(broken, 'w:gz') as tf:
            mt = tarfile.TarInfo('release_broken.MANIFEST.sha256')
            mt.size = len(broken_bytes)
            tf.addfile(mt, io.BytesIO(broken_bytes))
        assert AR.cmd_verify(broken) == 1
        # A manifest whose per-file entries are not objects (e.g. a bare
        # string instead of {"sha256": ...}) must be refused, not crash the
        # verifier with an AttributeError.
        malformed = os.path.join(d, 'release_malformed.tar.gz')
        malformed_man = {'tag': 'm', 'files': {'data.txt': 'not-an-object'}}
        malformed_bytes = json.dumps(malformed_man).encode('utf-8')
        with tarfile.open(malformed, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_malformed.MANIFEST.sha256')
            mt.size = len(malformed_bytes)
            tf.addfile(mt, io.BytesIO(malformed_bytes))
        assert AR.cmd_verify(malformed) == 1
        # A present-but-malformed sha256/byte count, or a non-object
        # checkpoint_hashes table, must be refused instead of crashing on
        # slicing/len().
        for _name, _bad_files, _bad_ck in (
                ('badsha', {'data.txt': {'sha256': {'x': 1}, 'bytes': 3}}, None),
                ('badbytes', {'data.txt': {'sha256': __import__('hashlib').sha256(payload).hexdigest(),
                                           'bytes': True}}, None),
                ('badck', {'data.txt': {'sha256': __import__('hashlib').sha256(payload).hexdigest(),
                                        'bytes': len(payload)}}, True)):
            _p = os.path.join(d, f'release_{_name}.tar.gz')
            _man = {'tag': _name, 'files': _bad_files}
            if _bad_ck is not None:
                _man['checkpoint_hashes'] = _bad_ck
            _mb = json.dumps(_man).encode('utf-8')
            with tarfile.open(_p, 'w:gz') as tf:
                ti = tarfile.TarInfo('data.txt')
                ti.size = len(payload)
                tf.addfile(ti, io.BytesIO(payload))
                mt = tarfile.TarInfo(f'release_{_name}.MANIFEST.sha256')
                mt.size = len(_mb)
                tf.addfile(mt, io.BytesIO(_mb))
            assert AR.cmd_verify(_p) == 1, _name
        # A corrupt/non-tar archive is a verification failure, not a crash.
        corrupt = os.path.join(d, 'release_corrupt.tar.gz')
        with open(corrupt, 'wb') as f:
            f.write(b'not a tar archive')
        assert AR.cmd_verify(corrupt) == 1


def test_build_report_warmup_uses_primary_estimand():
    def rec(full, hold, warm, seed=0):
        return {'variant': 'csa_fixed', 'seed': seed, 'warm_steps': warm,
                'ppl': full, 'ppl_holdout': hold,
                'ppl_estimation_rows': 'nested_validation_holdout',
                'nll_sum_holdout': math.log(hold) * 40, 'n_tokens_holdout': 40,
                'run_cfg': 'cfg', 'steps': 1, 'tokens_seen': 40,
                'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e'}
    s = {'a::w0::seed0': rec(99.0, 10.0, 0),
         'a::w5000::seed0': rec(1.0, 20.0, 5000)}
    old_s, old_l = BR.summary, BR._legacy_warm_tags_cached
    BR.summary = lambda *a, **k: s
    BR._legacy_warm_tags_cached = lambda *a, **k: True
    try:
        out = BR.sec_p0w()
    finally:
        BR.summary, BR._legacy_warm_tags_cached = old_s, old_l
    # The table and the paired conclusion must both use the held-out PPL.
    assert '| `csa_fixed` | 0 | 10.00 ' in out, out
    assert '| `csa_fixed` | 5000 | 20.00 ' in out, out
    assert '99.00' not in out, out
    assert '+10.00 vs 基线' in out, out


def test_pool_kernel_empty_sequence_matches_eager():
    import csa_kernels as C
    g = C.PoolGeometry([[]], overlap=4, T=0, device=torch.device('cpu'), max_block=128)
    X = torch.randn(1, 0, 3)
    Z = torch.randn(1, 0, 3)
    Bp = torch.randn(129, 3)
    bid = torch.zeros(1, 0, dtype=torch.int32)
    y, last, Bm = C.pool_variable_blocks_kernel(X, None, Z, None, bid, g, Bp, Bp)
    assert y.shape == (1, 0, 3) and last.shape == (1, 0) and Bm == 0, (y.shape, last.shape, Bm)
    y2, last2, Bm2 = C.pool_blocks_single_kernel(X, Z, bid, g, Bp)
    assert y2.shape == (1, 0, 3) and last2.shape == (1, 0) and Bm2 == 0, (y2.shape, last2.shape, Bm2)


def test_pool_geometry_accepts_no_cap():
    # `max_block=None` (dynamic boundaries only) is a supported scale
    # configuration.  The pool geometry is fully determined by the counts
    # table, so it must not crash on int(None) — neither when built directly
    # nor through the cache used by the fused-kernel path.
    import csa_kernels as C
    dev = torch.device('cpu')
    g = C.PoolGeometry([[2, 1, 3]], overlap=2, T=6, device=dev, max_block=None)
    assert g.max_block is None and g.R == 1 and g.Bm == 3, g.__dict__
    g2 = C._pool_geometry_cached([[2, 1, 3]], 2, 6, dev, None)
    assert g2.max_block is None and g2.R == 1 and g2.Bm == 3
    g3 = C._pool_geometry_cached([[2, 1, 3]], 2, 6, dev, 128)
    assert g3.max_block == 128


def test_kernel_attention_grad_matches_eager_at_boundaries():
    # The torch-reference twin of the fused attention must match autograd on
    # the two boundary cases eager's graph defines: clamp_min(soft, 1e-12)
    # has zero derivative at/outside the boundary (so soft=0 must not emit a
    # uniform-softmax S-path gradient), and nan_to_num(sink) blocks the
    # gradient at a non-finite sink entry.
    import csa_kernels as C
    dev = torch.device('cpu')

    def _grads(mode, q, kb, vb, topk, last, soft=None, sink=None, w=0):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, kb, vb))
        sf = soft.clone().requires_grad_(True) if soft is not None else None
        ss = sink.clone().requires_grad_(True) if sink is not None else None
        ksw = torch.zeros(q.shape[0], w, q.shape[2], q.shape[3], device=dev)
        vsw = torch.zeros_like(ksw)
        if mode == 'eager':
            out = L.gathered_attention(qq[0], kk[0], vv[0], topk[0], last[0],
                                       ksw[0] if w else None, vsw[0] if w else None,
                                       w, 1.0, soft=None if sf is None else sf[0],
                                       sink_logits=ss)
        else:
            out = C.gathered_attention_kernel(qq, kk, vv, topk, last, ksw, vsw,
                                              w, 1.0, soft=sf, sink_logits=ss,
                                              impl='ref')
        out.sum().backward()
        return (out.detach(), qq.grad, kk.grad, vv.grad,
                None if sf is None else sf.grad,
                None if ss is None else ss.grad)

    def _same(case, **kw):
        ref = _grads('ref', **kw)
        eager = _grads('eager', **kw)
        for name, a, b in zip(('out', 'gq', 'gk', 'gv', 'gsoft', 'gsink'), ref, eager):
            if a is None:
                assert b is None, (case, name)
                continue
            assert torch.allclose(a, b, atol=2e-5, rtol=2e-5), (case, name, a, b)

    q = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    kb = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    vb = torch.tensor([[[[1., 2.]], [[3., 4.]]]])
    topk = torch.tensor([[[0, 1], [0, 1]]])
    last = torch.tensor([[-1, 0]])
    _same('soft=0', q=q, kb=kb, vb=vb, topk=topk, last=last,
          soft=torch.zeros(1, 2, 2))
    _same('soft=1e-12', q=q, kb=kb, vb=vb, topk=topk, last=last,
          soft=torch.full((1, 2, 2), 1e-12))

    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 4)
    kb = torch.randn(1, 3, 2, 4)
    vb = torch.randn(1, 3, 2, 4)
    topk = torch.tensor([[[0], [1], [0], [2]]])
    last = torch.tensor([[0, 1, 0, 2]])
    _same('nan sink', q=q, kb=kb, vb=vb, topk=topk, last=last,
          sink=torch.tensor([float('nan'), 0.3]))


def test_kernel_attention_block_only_window_contract():
    # The public eager entry documents w=0 as block-only attention with
    # k_sw=v_sw=None.  The kernel/ref twin must accept the same form, and a
    # caller-supplied zero-width window tensor must receive a zero gradient
    # of its own shape (not the hard-coded [R, T, nh, hd] table the backward
    # used before).  A non-zero window without its K/V tables is a clear
    # caller error, not an opaque NoneType failure.
    import csa_kernels as C
    T, nh, hd, B, nb = 6, 1, 3, 3, 2
    torch.manual_seed(0)
    q0, kb0, vb0 = (torch.randn(T, nh, hd), torch.randn(B, nh, hd),
                    torch.randn(B, nh, hd))
    topk = torch.stack([torch.randperm(B)[:nb] for _ in range(T)]).to(torch.int32)
    last = torch.tensor([0, 1, 2])
    dO = torch.randn(T, nh, hd)

    def _eager(empty_window):
        q, kb, vb = (x.clone().requires_grad_(True) for x in (q0, kb0, vb0))
        if empty_window:
            ks = torch.zeros(0, nh, hd, requires_grad=True)
            vs = torch.zeros(0, nh, hd, requires_grad=True)
        else:
            ks = vs = None
        out = L.gathered_attention(q, kb, vb, topk, last, None, None, 0, 0.7)
        (out * dO).sum().backward()
        return out, q.grad, kb.grad, ks, vs

    def _kernel(empty_window):
        q, kb, vb = (x.clone().requires_grad_(True) for x in (q0, kb0, vb0))
        if empty_window:
            ks = torch.zeros(1, 0, nh, hd, requires_grad=True)
            vs = torch.zeros(1, 0, nh, hd, requires_grad=True)
        else:
            ks = vs = None
        out = C.gathered_attention_kernel(
            q[None], kb[None], vb[None], topk[None], last[None], ks, vs, 0, 0.7,
            impl='ref')
        (out * dO[None]).sum().backward()
        return out[0], q.grad, kb.grad, ks, vs

    for empty_window in (False, True):
        e_out, e_gq, e_gkb, e_ks, e_vs = _eager(empty_window)
        k_out, k_gq, k_gkb, k_ks, k_vs = _kernel(empty_window)
        assert torch.allclose(k_out, e_out, atol=1e-6, rtol=1e-6), empty_window
        assert torch.allclose(k_gq, e_gq, atol=1e-6, rtol=1e-6), empty_window
        assert torch.allclose(k_gkb, e_gkb, atol=1e-6, rtol=1e-6), empty_window
        if empty_window:
            for k_t, e_t in ((k_ks, e_ks), (k_vs, e_vs)):
                assert tuple(k_t.grad.shape) == tuple(k_t.shape) == (1, 0, nh, hd)
                assert torch.count_nonzero(k_t.grad).item() == 0

    # A non-zero window still requires its K/V tables: refuse it explicitly.
    try:
        C.gathered_attention_kernel(
            torch.randn(1, 4, 1, 2), torch.randn(1, 2, 1, 2),
            torch.randn(1, 2, 1, 2), torch.zeros(1, 4, 1, dtype=torch.int32),
            torch.zeros(1, 2, dtype=torch.long), None, None, 2, 1.0, impl='ref')
    except ValueError as exc:
        assert 'requires k_sw/v_sw' in str(exc)
    else:
        raise AssertionError('a non-zero window without K/V must be refused')


def test_kernel_ref_length_one_dynamic_arm_matches_eager():
    # At T=1 there is no consecutive-pair similarity, so the eager gate hoist
    # returns no row and the segmenter supplies the single one-token block.
    # The kernel/ref path must accept the same input instead of demanding a
    # gate row that cannot exist.
    import csa_kernels as C
    for variant in ('csa_dynamic', 'hybrid_dynamic'):
        cfgs = L.make_layer_cfgs(2, variant)
        model = L.SmallGPT(97, 32, 2, 4, 8, 1, cfgs, mlp_ratio=2.0)
        ids = torch.tensor([[3], [5]], dtype=torch.long)
        with C.kernel_env(None):
            eager = model(ids)
        with C.kernel_env('ref'):
            kernel = model(ids)
        assert torch.allclose(eager, kernel, atol=1e-5, rtol=1e-5), (
            variant, eager, kernel)


def test_kernel_reinforce_boundary_logp_survives_batching():
    # `csa_dynamic_task` trains its cut policy through REINFORCE.  The batched
    # kernel/ref path used to return None (dropping the only boundary-loss
    # term), and the eager row loop used to keep only the LAST row's logp.
    # Both must now expose the summed log-probability and decision count.
    import csa_kernels as C
    got = {}
    for mode in (None, 'ref'):
        with C.kernel_env(mode):
            torch.manual_seed(0)
            cfgs = L.make_layer_cfgs(1, 'csa_dynamic_task')
            model = L.SmallGPT(128, 64, 1, 4, 16, 32, cfgs).to('cpu')
            ids = torch.randint(0, 128, (2, 32),
                                generator=torch.Generator().manual_seed(1))
            model(ids)
            assert model.boundary_logp is not None, (mode, 'logp missing')
            got[mode] = (float(model.boundary_logp.detach()),
                         int(model.boundary_decisions))
    assert got[None][1] == got['ref'][1], got
    assert abs(got[None][0] - got['ref'][0]) <= 1e-4, got


def test_probe_contrasts_require_same_validation_slice():
    # The distractor-probe cell index drops val_fp from its lookup key; two
    # arms measured on different validation slices each look unambiguous, so
    # the pairing gate itself must compare the slice fingerprints.  Same for
    # the derived delta-NLL entries.
    import explib.corruption as V10
    import explib.length as V11
    code = L.code_stamp()
    pp = {'n_seq': 4, 'seq_len': 8, 'target': 2, 'chunk': 2}

    def _rec(v, arm, seed, vfp, ppl, rho=0.5, mode='uniform'):
        return {'variant': v, 'arm': arm, 'seed': seed, 'eval_len': 8,
                'rho': rho, 'mode': mode, 'ppl_mean': ppl,
                'probe_params': dict(pp), '_code': code, 'recipe': 'r',
                'val_fp': vfp}

    raw = {}
    for s in (0, 1, 2):
        raw[f'l{s}'] = _rec('csa_fixed_rope', 'learned', s, 'A', 10.0 + s)
        raw[f'd{s}'] = _rec('full_rope', 'dense', s, 'B', 11.0 + s)
    td = tempfile.mkdtemp(prefix='probe_', dir=REPO)
    old = dict(V10.PROBE)
    try:
        sp = os.path.join(td, 'distractor.json')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        V10.PROBE.update(summary=sp, eval_lens=[8], rhos=[0.0, 0.5],
                         modes={'uniform': [0.5]}, niah={})
        r10 = V10.v10_analysis(os.path.join(td, 'v10.json'))
        assert not [k for k in r10['probe']['contrasts'] if k.startswith('L8 r0.5')], \
            r10['probe']['contrasts']
        r11 = V11.v11_analysis(os.path.join(td, 'v11.json'))
        assert not [k for k in r11['probe']['contrasts'] if k.startswith('L8 r0.5')], \
            r11['probe']['contrasts']
    finally:
        V10.PROBE.clear()
        V10.PROBE.update(old)
        shutil.rmtree(td, ignore_errors=True)


def test_greedy_match_batch_is_order_independent():
    # `_greedy_match_counts` sorts its predicate before the rank tie-break.
    # The batched helper must canonicalise each subset the same way, otherwise
    # its match count depends on the caller's tuple order.
    gt = np.array([3, 7, 10, 14, 15, 16, 19, 22])
    sub = (9, 8, 6)
    gs, order = np.sort(gt), np.argsort(gt, kind='stable')
    got = L._greedy_match_count_batch([sub], gs, order, tol=3)
    assert got == [L._greedy_match_counts(np.array(sub), gt, tol=3)[0]], got


def test_pool_variable_blocks_accepts_no_overlap():
    # `overlap=None` is the no-overlap/HCA contract.  `_pool_parts` then
    # returns a four-tuple while `_pool_ordered` unpacks eight, so this public
    # entry point used to raise a bare ValueError; it must return the exact
    # single-lane pooling instead.
    bid = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    n, F, cap = 6, 3, 8
    Xa, Za = torch.randn(n, F), torch.randn(n, F)
    Xb, Zb = torch.randn(n, F), torch.randn(n, F)
    Bpa, Bpb = torch.randn(cap, F), torch.randn(cap, F)
    got = L.pool_variable_blocks(Xa, Xb, Za, Zb, bid, Bpa, Bpb, None, n_blocks=3)
    ref = L.pool_blocks_single(Xa, Za, Bpa, bid, n_blocks=3)
    assert torch.allclose(got[0], ref[0]) and torch.equal(got[1], ref[1]) and got[2] == ref[2], got


def test_efficiency_median_uses_both_middle_seeds():
    # Four seeds: the median of 10, 20, 30, 100 is (20+30)/2, not the upper
    # middle value 30.
    rows = [{'variant': 'a', 'ppl': 10.0, 'prefill_ms_p50': l,
             'params': p, 'tokens_seen': 100}
            for l, p in ((10.0, 1000), (20.0, 2000), (30.0, 3000), (100.0, 4000))]
    s = BR.efficiency_accuracy_summary(rows)[0]
    assert s['latency_median'] == 25.0 and s['params_median'] == 2500.0, s
    assert s['tokens_seen_median'] == 100.0, s


def test_eager_batch_rejects_typos_and_nonfinite():
    assert L._normalise_eager_batch(None) is None
    assert L._normalise_eager_batch(True) is True
    assert L._normalise_eager_batch('auto') is None
    assert L._normalise_eager_batch('false') is False
    assert L._normalise_eager_batch(0) is False and L._normalise_eager_batch(1.0) is True
    for bad in ('flase', 'maybe', float('nan'), 1.5, -1, 2):
        try:
            L._normalise_eager_batch(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eager_batch={bad!r} was silently coerced')


def test_gapfill_merge_keeps_secondary_and_invalidates_stale_fields():
    import run_gapfill as G
    with tempfile.TemporaryDirectory() as d:
        snap = {'full': {'seeds': [0, 1], 'ppls': [10.0, 12.0], 'n_seeds': 2,
                         'ppl_mean': 11.0, 'ppl_std': math.sqrt(2.0),
                         'avg_dyn_block_len': 4.0,
                         'ppl_curve': [[1, 10.0], [2, 11.0]]}}
        fresh = {'full': {'seeds': [0, 1], 'ppls': [20.0, 22.0], 'n_seeds': 2,
                          'ppl_mean': 21.0, 'ppl_std': math.sqrt(2.0),
                          'avg_dyn_block_len': 8.0,
                          'ppl_curve': [[1, 20.0], [2, 21.0]]}}
        json.dump(fresh, open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, snap)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        # Re-measuring the same seeds must refresh the secondary values, not
        # leave the snapshot's old ones next to the fresh PPLs.
        assert out['ppls'] == [20.0, 22.0] and out['avg_dyn_block_len'] == 8.0, out
        assert out['ppl_curve'] == [[1, 20.0], [2, 21.0]], out
        # Recovering a lost seed must not leave fields computed on the old
        # subset looking current.
        json.dump({'full': {'seeds': [0, 1], 'ppls': [10.0, 20.0]}},
                  open(os.path.join(d, 'aggregate.json.v6bak'), 'w'))
        live = {'full': {'seeds': [0], 'ppls': [10.0], 'n_seeds': 1,
                         'ppl_mean': 10.0, 'ppl_std': 0.0, 'ppl_primary': 10.0,
                         'nll_sum': 10.0, 'n_tokens': 1, 'ppl_curve': [[1, 10.0]]}}
        json.dump(live, open(os.path.join(d, 'aggregate.json'), 'w'))
        G._fold_backup_into_aggregate(d)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        assert out['seeds'] == [0, 1] and out['ppl_mean'] == 15.0, out
        for _f in ('ppl_primary', 'nll_sum', 'n_tokens', 'ppl_curve'):
            assert out[_f] is None, (out, _f)
        # Valid JSON with the wrong top-level type is skipped, not crashed on.
        json.dump([], open(os.path.join(d, 'aggregate.json.v6bak'), 'w'))
        G._fold_backup_into_aggregate(d)
    # Partial seed overlap cannot be merged from aggregate means: the old mean
    # includes the overlapping seeds, so neither the old-only/fresh weights nor
    # the old curve can be split without per-seed values.  The per-seed PPLs
    # still merge exactly; the affected seed-averaged fields are invalidated.
    with tempfile.TemporaryDirectory() as d:
        snap = {'full': {'seeds': [0, 1, 2], 'ppls': [10.0, 12.0, 14.0],
                         'n_seeds': 3, 'ppl_mean': 12.0, 'ppl_std': 2.0,
                         'avg_dyn_block_len': 3.0,
                         'ppl_curve': [[1, 10.0], [2, 12.0]]}}
        fresh = {'full': {'seeds': [1, 2, 3], 'ppls': [20.0, 22.0, 24.0],
                          'n_seeds': 3, 'avg_dyn_block_len': 9.0,
                          'ppl_curve': [[1, 20.0], [2, 22.0]]}}
        json.dump(fresh, open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, snap)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        assert out['seeds'] == [0, 1, 2, 3] and out['ppls'] == [10.0, 20.0, 22.0, 24.0], out
        assert out['ppl_mean'] == 19.0, out
        assert out['avg_dyn_block_len'] is None and out['ppl_curve'] is None, out
        assert out['merge_stale_fields'] == 'partial_seed_overlap', out
    # A valid-JSON non-object fresh aggregate (or snapshot) must not crash the
    # merge on `.items()` / `.get()`.
    with tempfile.TemporaryDirectory() as d:
        json.dump([], open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, {'full': {'seeds': [0], 'ppls': [1.0]}})
        assert json.load(open(os.path.join(d, 'aggregate.json'))) == []
        json.dump({'full': {'seeds': [0], 'ppls': [2.0]}},
                  open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, ['not', 'a', 'mapping'])
        out = json.load(open(os.path.join(d, 'aggregate.json')))
        assert out['full']['ppls'] == [2.0], out
    # A malformed aggregate ENTRY (valid JSON, non-object value) must be
    # skipped by the synthesizer instead of crashing on `.get`.
    with tempfile.TemporaryDirectory() as d:
        _old_repo = G.REPO
        G.REPO = d
        try:
            _panel = os.path.join(d, 'results_lm_v3_long')
            os.makedirs(_panel)
            _tok = (G._LONG_STEPS * int(G.L.RUN_LONG['batch_size'])
                    * int(G.L.RUN_LONG['seq_len']))
            json.dump({'bad': [1, 2],
                       'full': {'seeds': [0], 'ppls': [2.0],
                                'tokens_seen': _tok}},
                      open(os.path.join(_panel, 'aggregate.json'), 'w'))
            _sum, _agg = G.synthesize_long_summary()
            _out = json.load(open(os.path.join(_panel, 'summary.json')))
            assert 'full::seed0' in _out and 'bad' not in ' '.join(_out), _out
        finally:
            G.REPO = _old_repo


def test_open_block_resume_binds_data_fingerprints():
    import explib.open_block as OB
    tr = np.arange(16, dtype=np.int64)
    va = np.arange(8, dtype=np.int64).reshape(1, 8)
    rec = {'run_fingerprint': 'fp', 'code': L.code_stamp(), 'ppl': 10.0,
           'nll_sum': 10.0, 'n_tokens': 10,
           'data_fp': L._ids_fp(tr), 'eval_fp': L._ids_fp(va.reshape(-1))}
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is True
    # A record without exact token accounting is not usable by `analyse`, so
    # it must not be skipped as a current cell on resume.
    no_acc = {k: v for k, v in rec.items() if k not in ('nll_sum', 'n_tokens')}
    assert OB._is_current(no_acc, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    frac = dict(rec, n_tokens=10.5)
    assert OB._is_current(frac, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr + 1),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1) + 1)) is False
    # A legacy record without the fingerprints cannot be certified current.
    legacy = {k: v for k, v in rec.items() if k not in ('data_fp', 'eval_fp')}
    assert OB._is_current(legacy, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    # The panel manifest's "measured" set must use the same current-record
    # contract: a stale run/data/eval fingerprint (or missing accounting) is
    # not a completed cell for this panel.
    planned = [f'{v}::seed0' for v in ('current', 'stale_fp', 'stale_data',
                                       'no_acc', 'legacy')]
    summary = {
        'current::seed0': rec,
        'stale_fp::seed0': dict(rec, run_fingerprint='other'),
        'stale_data::seed0': dict(rec, data_fp=L._ids_fp(tr + 1)),
        'no_acc::seed0': no_acc,
        'legacy::seed0': legacy,
    }
    assert OB._current_cells(summary, planned, 'fp',
                             data_fp=L._ids_fp(tr),
                             eval_fp=L._ids_fp(va.reshape(-1))) == {
                                 'current::seed0'}, summary


def test_analysis_checkpoint_carries_model_geometry():
    import error_analysis as E
    ids = np.arange(2048, dtype=np.int64) % 97
    val = (np.arange(64, dtype=np.int64) % 97).reshape(2, 32)
    with tempfile.TemporaryDirectory() as d:
        r = L.train_variant('full', ids, val, 97, seed=0, d=32, n_layers=1,
                            n_heads=2, d_head=16, seq_len=32, batch_size=1,
                            steps=1, lr=1e-3, weight_decay=0.0, warmup=0,
                            comp_lambda=0.0, eval_every=0, mlp_ratio=4.0,
                            device=torch.device('cpu'), log_every=0,
                            save_ckpt='final', ckpt_dir=d)
        path = r['checkpoints']['final']['path']
        # The standard analysis checkpoint must be loadable by error_analysis;
        # it used to omit cfg/vocab/geometry entirely.
        model, meta = E._load_model(path)
        assert meta.get('cfg') is not None and meta.get('vocab') == 97, meta
        assert meta.get('d') == 32 and meta.get('train_len') == 32, meta


def test_v12_admission_rejects_synthesized_records():
    m = _manifest_for_admission(seeds=(0, 1))
    good = {}
    for v in ('a_arm', 'b_arm'):
        for sd in (0, 1):
            good[f'{v}::s{sd}'] = _rec(v, sd, 'FP')
    adm, _fp = V12._admit_confirmation_records(good, m)
    assert len(adm) == 4, adm
    synth = dict(good)
    synth['a_arm::s0'] = dict(synth['a_arm::s0'], synthesized=True)
    adm, _fp = V12._admit_confirmation_records(synth, m)
    assert ('a_arm', 0) not in adm and len(adm) == 3, adm


def test_seed_and_nested_input_guards():
    # Reproduction probes must not truncate 0.2/0.8 to one seed or treat a
    # boolean as a draw identity.
    for bad in ([0.2, 0.8], [True, 1], ['1.5']):
        try:
            L.reproducibility_guard(['full'], seeds=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_guard accepted seeds={bad!r}')
    # Both nested folds are required; a partial request must not crash on None.
    rng = np.random.default_rng(3)
    train = rng.integers(0, 16, size=200)
    val = rng.integers(0, 16, size=(4, 8))
    for kw in ({'selection_rows': [0]}, {'evaluation_rows': [1]}):
        try:
            L.simple_baseline_fit(train, val, 16, protocol=dict(
                L.SIMPLE_BASELINE_POLICY,
                arms={'u': {'order': 1}}, grids={'u': {'alpha': [0.1]}}), **kw)
        except ValueError as e:
            assert 'BOTH' in str(e), e
        else:
            raise AssertionError(f'partial nested folds {kw!r} were accepted')


def test_malformed_panel_summary_is_excluded():
    # A JSON file that parses but is not a cell -> record object is not a
    # panel.  Readers must exclude it instead of crashing inside .items(), and
    # the resume writer must refuse to overwrite it with an empty summary.
    td = tempfile.mkdtemp(prefix='panel_shape_')
    try:
        sp = os.path.join(td, 'summary.json')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a', 'seed': 0, 'ppl': 1.0}], f)
        assert L.read_panel(td) == {}
        assert L.ppl_by_seed(td) == {}
        assert L.variant_presence(td) == {}
        try:
            L._read_summary_json(sp)
        except ValueError:
            pass
        else:
            raise AssertionError('a non-object summary must not be resumed from')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump('not a panel', f)
        assert L.read_panel(td) == {}
        assert L.variant_presence(td) == {}
        # Audit readers share the same object contract: a list/scalar is
        # unverifiable, not something to call .items() on.
        import audit_gates as A
        assert A._load(sp) is None
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a'}], f)
        assert A._load(sp) is None
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 1.0}}, f)
        assert isinstance(A._load(sp), dict)
        # A malformed arm id inside an otherwise valid panel record must not
        # crash the presence reader on an unhashable dict/list key.
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': ['not', 'a', 'string'],
                                    'seed': 0, 'ppl': 1.0}}, f)
        assert L.variant_presence(td) == {}
        # A malformed nested provenance block is missing evidence, not an
        # AttributeError in the shared-validation fingerprint check.
        assert L.by_len_cells([{'provenance': ['bad'],
                                'by_len': {4: {'ppl': 10.0}}}], 4) == [
            {'ppl': 10.0}]
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_meta_seed_overlap_controls_pooling():
    # The DL pooled CI treats scale points as independent studies.  Declaring
    # seed identities is not enough: disjoint sets are independent, a complete
    # common set has a paired sensitivity analysis, but a partially overlapping
    # ladder is neither and must not be pooled as a common effect.
    def _pt(label, scale, seeds, effects):
        return {'label': label, 'scale': scale, 'seeds': list(seeds),
                'effects': list(effects), 'corpus_fp': 'fp', 'vocab': 8000}
    eff = [0.10, 0.20, 0.30, 0.40]
    partial = [_pt('S0', 1, [0, 1, 2, 3], eff),
               _pt('S1', 10, [1, 3, 4, 5], eff),
               _pt('S2', 100, [0, 2, 4, 5], eff)]
    m = L.meta_analyze_effects(partial)
    assert m['status'] == 'ok' and m['seed_dependence'] == 'shared_seeds_unpaired', m
    assert m['seed_overlap']['n_common_seeds_all_points'] == 0, m['seed_overlap']
    assert all(v > 0 for v in m['seed_overlap']['per_pair_common_seeds'].values()), m['seed_overlap']
    assert m['poolable'] is False, m
    assert 'independence' in m['note'], m['note']
    # Disjoint draws are genuinely independent replications.
    disjoint = [_pt('S0', 1, [0, 1, 2, 3], eff),
                _pt('S1', 10, [4, 5, 6, 7], eff),
                _pt('S2', 100, [8, 9, 10, 11], eff)]
    md = L.meta_analyze_effects(disjoint)
    assert md['seed_dependence'] == 'disjoint_seed_sets' and md['poolable'] is True, md
    assert md['seed_overlap']['n_common_seeds_all_points'] == 0, md['seed_overlap']
    # Undeclared seed identities cannot be assumed independent.
    unattested = [{k: v for k, v in p.items() if k != 'seeds'} for p in disjoint]
    mu = L.meta_analyze_effects(unattested)
    assert mu['seed_dependence'] == 'unattested' and mu['poolable'] is False, mu
    # The WLS trend fallback uses the same independence assumption.  A
    # partially overlapping ladder with fewer than two complete common seeds
    # must not turn a significant unpaired slope into a confirmed trend.
    trend = [_pt('S0', 1, [0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
             _pt('S1', 10, [1, 3, 4, 5], [1.0, 1.1, 1.2, 1.3]),
             _pt('S2', 100, [0, 2, 4, 5], [2.0, 2.1, 2.2, 2.3])]
    rt = L.scaling_trend(trend, min_seeds=3, min_points=3)
    assert rt['seed_dependence'] == 'shared_seeds_unpaired', rt
    assert rt['seed_overlap']['n_common_seeds_all_points'] == 0, rt['seed_overlap']
    assert rt['paired']['n_complete_seeds'] < 2, rt['paired']
    assert rt['primary_inference'].startswith('WLS'), rt['primary_inference']
    assert rt['significant'] is False and rt['verdict'] == 'seed_dependence_unresolved', rt
    # Disjoint seed sets keep the unpaired slope valid.
    trend_disjoint = [_pt('S0', 1, [0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
                      _pt('S1', 10, [4, 5, 6, 7], [1.0, 1.1, 1.2, 1.3]),
                      _pt('S2', 100, [8, 9, 10, 11], [2.0, 2.1, 2.2, 2.3])]
    rd = L.scaling_trend(trend_disjoint, min_seeds=3, min_points=3)
    assert rd['seed_dependence'] == 'disjoint_seed_sets', rd
    assert rd['significant'] is True, rd
    # A within-scale SE that is only numerically zero also makes 1/se^2
    # unbounded; it must be named and excluded from the inverse-variance pool.
    nearzero = [_pt('S0', 1, [0, 1, 2, 3], [0.1, 0.1, 0.1, 0.1 + 1e-15]),
                _pt('S1', 10, [4, 5, 6, 7], [0.2, 0.3, 0.4, 0.5]),
                _pt('S2', 100, [8, 9, 10, 11], [0.3, 0.4, 0.5, 0.6])]
    mn = L.meta_analyze_effects(nearzero)
    assert 'S0' in mn['zero_variance_points'], mn
    assert mn['n_usable'] == 2 and mn['poolable'] is False, mn
    # Every scale may have n>=min_seeds while only two seeds are measured at
    # EVERY scale; the paired dependence then has less resolution than the
    # per-scale n suggests and must block the pooled claim.
    shared2 = [_pt('S0', 1, [0, 1, 2, 3, 4, 5], eff + [0.5, 0.6]),
               _pt('S1', 10, [0, 1, 6, 7, 8, 9], eff + [0.5, 0.6]),
               _pt('S2', 100, [0, 1, 10, 11, 12, 13], eff + [0.5, 0.6])]
    ms = L.meta_analyze_effects(shared2, min_seeds=3)
    assert ms['seed_dependence'] == 'shared_seeds_paired', ms
    assert ms['paired_common_effect']['n'] == 2, ms['paired_common_effect']
    assert ms['paired_common_effect']['underpowered'] is True, ms['paired_common_effect']
    assert ms['poolable'] is False, ms
    assert 'complete seed' in ms['note'], ms['note']


def test_preprocessing_fit_audit_detects_full_table_leak():
    import data_guard as DG
    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(40, 3))
    Xho = rng.normal(loc=4.0, scale=2.0, size=(20, 3))
    train_only = DG.LeakageSafePreprocessor().fit(Xtr)
    a = DG.audit_preprocessor_fit(train_only, Xtr, Xho)
    assert a['status'] == 'train_only_verified' and a['leak_detected'] is False, a
    full_fit = DG.LeakageSafePreprocessor().fit(np.vstack([Xtr, Xho]))
    b = DG.audit_preprocessor_fit(full_fit, Xtr, Xho)
    assert b['status'] == 'leak_detected' and b['leak_detected'] is True, b
    assert b['stats_match_train'] is False and b['stats_match_train_plus_holdout'] is True, b
    # A different fitting table is not a proven leak, but it is not verifiable.
    c = DG.audit_preprocessor_fit(
        DG.LeakageSafePreprocessor().fit(rng.normal(size=(40, 3))), Xtr, Xho)
    assert c['status'] == 'unverifiable' and c['leak_detected'] is False, c
    # The train-only audit works without holdouts when stats already match.
    d = DG.audit_preprocessor_fit(train_only, Xtr)
    assert d['status'] == 'train_only_verified', d
    # split_and_prepare must expose a preprocessor that passes the audit.
    out = DG.split_and_prepare(Xtr, preprocess=True)
    e = DG.audit_preprocessor_fit(out['preprocessor'], Xtr,
                                  [out['X']['val'], out['X']['test']])
    assert e['status'] in ('train_only_verified', 'unverifiable'), e
    # A plain list-of-rows holdout is ONE table, not one part per row.  The
    # old list-is-parts rule transposed it into a (n_rows*n_features, 1)
    # column and rejected the audit on a shape mismatch.
    _ll = [[float(v) for v in row] for row in Xho.tolist()]
    f = DG.audit_preprocessor_fit(full_fit, [[float(v) for v in row]
                                             for row in Xtr.tolist()], _ll)
    assert f['status'] == 'leak_detected' and f['n_holdout'] == len(_ll), f


def test_split_targets_use_exact_fraction_arithmetic():
    import data_guard as DG
    # The intended 4/7 split must survive the float representation of 4/7.
    assert DG._floor_fraction_product(7, 4 / 7) == 4
    assert DG._floor_fraction_product(7, 2 / 7) == 2
    # At row counts beyond float64's exact integer range the target sizes must
    # still sum to n and follow the exact rational fraction, not a rounded
    # product.
    n = 2 ** 53 + 1
    sizes = DG._resolve_sizes(n, 0.8, 0.1, 0.1)
    assert sum(sizes) == n, sizes
    assert sizes == (7205759403792795, 900719925474099, 900719925474099), sizes
    # A genuinely positive sub-1e-15 fraction must not collapse to exactly
    # zero: at a large row count it still selects rows.
    assert DG._fraction_value(1e-16) != 0
    assert DG._floor_fraction_product(10 ** 18, 1e-16) > 0
    assert DG._floor_fraction_product(10 ** 18, 1e-15) == 1000
    # The tolerated 1e-6 fraction slack is not idle at scale: 0.01 + 1e-9 +
    # 0.99 is over 1 by 1e-9, and at n > 1e9 the exact floors add up to n+741.
    # A valid split must still reconcile to n (removing the excess from the
    # split where one row moves it least), not raise "inconsistent sizes".
    n_over = 742878783538
    sizes_over = DG._resolve_sizes(n_over, 0.01, 1e-9, 0.99)
    assert sum(sizes_over) == n_over, sizes_over
    assert all(v >= 0 for v in sizes_over), sizes_over
    assert sizes_over[1] > 0, sizes_over
    # The reconciliation must be O(1) in the excess, not one row at a time:
    # the tolerated 1e-6 slack allows ~1e12 excess rows at n=1e18.
    sizes_scale = DG._resolve_sizes(10 ** 18, 0.01, 1e-9, 0.99)
    assert sum(sizes_scale) == 10 ** 18, sizes_scale
    assert all(v >= 0 for v in sizes_scale), sizes_scale
    # The same reconciliation must preserve the requested one-row-per-split
    # floor and the exact 80/10/10 result when the fractions sum to one.
    assert DG._resolve_sizes(3, 0.01, 1e-9, 0.99) == (1, 1, 1)
    assert DG._resolve_sizes(n, 0.8, 0.1, 0.1) == (
        7205759403792795, 900719925474099, 900719925474099)
    try:
        DG._resolve_sizes(10, 0.8, 0.1, 0.1, train_size=11)
    except ValueError:
        pass
    else:
        raise AssertionError('oversized explicit split must still be refused')


def test_encoder_fit_audit_detects_holdout_vocabulary():
    import data_guard as DG
    Xtr = np.array([['a'], ['b'], ['a'], ['b']], dtype=object)
    Xho = np.array([['c'], ['a']], dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(Xtr)
    a = DG.audit_encoder_fit(cat, Xtr, Xho)
    assert a['status'] == 'train_only_verified' and a['leak_detected'] is False, a
    cat_leak = DG.LeakageSafeCategoricalEncoder().fit(np.vstack([Xtr, Xho]))
    b = DG.audit_encoder_fit(cat_leak, Xtr, Xho)
    assert b['status'] == 'leak_detected' and b['leak_detected'] is True, b
    freq = DG.LeakageSafeFrequencyEncoder().fit(Xtr)
    c = DG.audit_encoder_fit(freq, Xtr, Xho)
    assert c['status'] == 'train_only_verified', c
    freq_leak = DG.LeakageSafeFrequencyEncoder().fit(np.vstack([Xtr, Xho]))
    d = DG.audit_encoder_fit(freq_leak, Xtr, Xho)
    assert d['status'] == 'leak_detected' and d['leak_detected'] is True, d
    # A list-of-rows holdout must be one table here too.
    _cat_ll = DG.LeakageSafeCategoricalEncoder().fit(
        np.vstack([Xtr, Xho]))
    _ll_audit = DG.audit_encoder_fit(_cat_ll, Xtr.tolist(), Xho.tolist())
    assert _ll_audit['leak_detected'] is True and _ll_audit['n_holdout'] == len(Xho), _ll_audit
    try:
        DG.audit_encoder_fit(DG.LeakageSafeTargetEncoder().fit(
            Xtr, np.array([0.0, 1.0, 0.0, 1.0])), Xtr)
    except TypeError:
        pass
    else:
        raise AssertionError('target encoder is not a single-table fit and must be refused')

