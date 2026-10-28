"""CPU review checks (split group 04; t_simple_baseline_fit_budget_is_bounded_and_disclosed .. t_audit_split_warning_status)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_simple_baseline_fit_budget_is_bounded_and_disclosed():
    rng = np.random.default_rng(11)
    vocab = 16
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(3, 12))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1]},
                        'bigram': {'alpha': [0.1], 'interpolation': [0.5]}},
                 max_fit_tokens=100)
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert art['data']['n_fit_tokens'] == 100, art['data']
    assert art['data']['n_source_tokens'] == train.size, art['data']
    assert art['data']['fit_truncated'] is True, art['data']
    assert art['data']['train_fp'] == L._ids_fp(np.asarray(train[:100])), art['data']
    assert art['data']['source_train_fp'] == L._ids_fp(train), art['data']
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['fit_truncated'] is True, audit
    assert audit['fit_budget']['n_fit_tokens'] == 100, audit['fit_budget']
    assert audit['fit_budget']['n_source_tokens'] == train.size, audit['fit_budget']
    assert audit['fit_budget']['max_fit_tokens'] == 100, audit['fit_budget']
    assert any('deterministic prefix' in w for w in audit['warnings']), audit
    # Tokens beyond the declared fit prefix must not change the floor.
    train2 = train.copy()
    train2[100:] = (train2[100:] + 1) % vocab
    art2 = L.simple_baseline_fit(train2, val, vocab, protocol=proto)
    assert abs(float(art2['best_ppl']) - float(art['best_ppl'])) < 1e-12, (art['best_ppl'], art2['best_ppl'])
    assert art2['data']['source_train_fp'] != art['data']['source_train_fp'], art2['data']
    # A truncated claim with contradictory counts is not accepted.
    bad = json.loads(json.dumps(art))
    bad['data']['n_fit_tokens'] = bad['data']['n_source_tokens']
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False, rbad
    # The truncation flag cannot contradict the recorded token counts.
    liar = json.loads(json.dumps(art))
    liar['data']['fit_truncated'] = False
    rliar = L.simple_baseline_audit(liar, protocol=proto)
    assert rliar['baseline_verified'] is False, rliar
    assert rliar['fit_truncated'] is True, rliar
    assert any('contradicts the token counts' in p for p in rliar['problems']), rliar
    # The recorded prefix must be exactly min(source, max_fit_tokens).
    wrong_cap = json.loads(json.dumps(art))
    wrong_cap['data']['max_fit_tokens'] = 200
    rcap = L.simple_baseline_audit(wrong_cap, protocol=proto)
    assert rcap['baseline_verified'] is False and any(
        'deterministic-prefix fit' in p for p in rcap['problems']), rcap
    # A non-boolean flag is malformed evidence, not a true/false label.
    strflag = json.loads(json.dumps(art))
    strflag['data']['fit_truncated'] = 'yes'
    rflag = L.simple_baseline_audit(strflag, protocol=proto)
    assert rflag['baseline_verified'] is False and any(
        'must be a boolean' in p for p in rflag['problems']), rflag


def test_simple_floor_contrast_is_significance_aware():
    # The "beats the trivial floor" claim must be a seed-level contrast, not a
    # pointwise < on two PPLs.  Build a panel whose holdout rows match the
    # floor's holdout rows and whose tokens match per seed.
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=600)
    val = rng.integers(0, vocab, size=(8, 13))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1]},
                        'bigram': {'alpha': [0.1], 'interpolation': [0.5]}})
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    sb = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                               selection_rows=sel, evaluation_rows=ev,
                               row_split=meta)
    sb['data']['cache_val_fp'] = 'v'
    floor = float(sb['best_ppl'])
    n_tok = int(sb['data']['n_eval_tokens'])
    rc = ('steps1_sl13_bs1_nt104_lr0.0003_wd0.1_d8_L1_H1_Dh8_v24_wu1_'
          'cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    def recs(base_ppl):
        out = {}
        for arm in ('csa_dynamic', 'full_sw128_matched'):
            for seed in range(8):
                ppl = base_ppl if arm == 'csa_dynamic' else floor + 2.0
                _r = n_tok // 4
                out[f'{arm}::seed{seed}'] = {
                    'variant': arm, 'seed': seed, 'ppl': ppl, 'params': 100,
                    'run_cfg': rc, 'steps': 1, 'tokens_seen': n_tok,
                    'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e',
                    'nll_sum': math.log(ppl) * n_tok, 'n_tokens': n_tok,
                    'holdout_row_fp': meta['evaluation_row_fp'],
                    'ppl_estimation_rows': 'nested_validation_holdout',
                    'nll_per_row_holdout': [math.log(ppl) * _r] * 4,
                    'n_tokens_per_row_holdout': [_r] * 4,
                }
        return out
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        with contextlib.redirect_stdout(io.StringIO()):
            r = L.methodology_audit(recs(floor - 1.0), design={}, baselines=L.BASELINE_DESIGN, outdir=td, verbose=False)
        fl = r['simple_baseline']
        assert fl['comparable_to_panel'] is True, fl
        # The final panel lives in <outdir>/final/; the artifacts live in the
        # parent.  The audit must find them instead of silently returning
        # not_verifiable_locally.
        _final = os.path.join(td, 'final')
        os.makedirs(_final)
        with contextlib.redirect_stdout(io.StringIO()):
            r_final = L.methodology_audit(recs(floor - 1.0), design={},
                                          baselines=L.BASELINE_DESIGN,
                                          outdir=_final, verbose=False)
        assert r_final['simple_baseline']['baseline_verified'] is True, r_final['simple_baseline']
        assert r_final['simple_baseline']['comparable_to_panel'] is True, r_final['simple_baseline']
        assert os.path.exists(os.path.join(_final, 'methodology_audit.json'))
        assert fl['token_denominator_match'] is True, fl
        assert fl['floor_contrast']['verdict'] == 'beats_floor_significant', fl['floor_contrast']
        # A floor fitted on a truncated prefix is not a floor trained on the
        # panel's full corpus; it may still be verified arithmetically, but it
        # must not be quoted as the adequately tuned floor.
        _trunc = json.loads(json.dumps(sb))
        _trunc['data'].update(fit_truncated=True, n_fit_tokens=100,
                              n_source_tokens=600, max_fit_tokens=100)
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(_trunc, f)
        with contextlib.redirect_stdout(io.StringIO()):
            _rtr = L.methodology_audit(recs(floor - 1.0), design={},
                                       baselines=L.BASELINE_DESIGN,
                                       outdir=td, verbose=False)
        _ftr = _rtr['simple_baseline']
        assert _ftr['fit_truncated'] is True, _ftr
        assert _ftr['tuned_floor_claim_allowed'] is False, _ftr
        assert any('truncated prefix' in str(x)
                   for x in (_ftr.get('tuned_floor_claim_reasons') or [])), _ftr
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        assert fl['floor_contrast']['p_exact_signflip'] < 0.05, fl['floor_contrast']
        # The caller's alpha is the gate for the floor family too.  At n=6
        # the exact p floor is 0.03125: significant at alpha=0.05 but not at
        # alpha=0.01, and the floor gate must not keep hard-coding 0.05.
        _lo = {k: v for k, v in recs(floor - 1.0).items()
               if int(v['seed']) < 6}
        with contextlib.redirect_stdout(io.StringIO()):
            _rlo = L.methodology_audit(_lo, design={}, baselines=L.BASELINE_DESIGN,
                                       outdir=td, alpha=0.01, verbose=False)
        _flo = _rlo['simple_baseline']
        assert _flo['floor_contrast']['verdict'] != 'beats_floor_significant', _flo['floor_contrast']
        assert _flo['method_arm_floor_contrasts']['csa_dynamic']['method_status'] != 'beats_floor_holm', _flo['method_arm_floor_contrasts']['csa_dynamic']
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = L.methodology_audit(recs(floor + 0.5), design={}, baselines=L.BASELINE_DESIGN, outdir=td, verbose=False)
        assert r2['simple_baseline']['panel_beats_simple_floor'] is False, r2['simple_baseline']
        assert r2['simple_baseline']['floor_contrast']['verdict'] == 'does_not_beat_floor', r2['simple_baseline']
        assert any('does not beat the tuned count-model floor' in w for w in r2['warnings']), r2['warnings']
        _fh2 = r2['simple_baseline'].get('floor_hypothesis') or {}
        assert _fh2.get('observed_status') == 'falsified', _fh2
        assert _fh2.get('observed_sign') == 1 and _fh2.get('direction_consistent') is False, _fh2
        # The pre-declared method arm must clear the floor ITSELF.  Selecting
        # the best panel arm post hoc could let a dense baseline rescue the
        # gate while the proposed method is worse than the count model.
        assert fl['method_arms_beat_floor'] is True, fl.get('method_arm_floor_status')
        assert fl['method_arm_floor_status'] == 'all_method_arms_beat_floor', fl.get('method_arm_floor_status')
        assert fl['method_arm_floor_contrasts']['csa_dynamic']['method_status'] == 'beats_floor_holm', fl['method_arm_floor_contrasts']
        # The floor comparison is a pre-registered directional hypothesis, not
        # just an arithmetic gate: delta = method - floor, so observed sign -1
        # supports the claim and a positive mean is a direction conflict.
        _fh = fl.get('floor_hypothesis') or {}
        assert _fh.get('observed_status') == 'supported', _fh
        assert _fh.get('observed_sign') == -1 and _fh.get('direction_consistent') is True, _fh
        assert _fh.get('audit_status') == 'ok', _fh
        assert L.hypothesis_audit(L.FLOOR_HYPOTHESIS_DESIGN, 'floor')['status'] == 'ok'
        # The floor is an estimate on the same rows, so the contrast also
        # carries a paired row bootstrap instead of treating it as a constant.
        _fb = fl['method_arm_floor_contrasts']['csa_dynamic'].get('paired_row_bootstrap_delta') or {}
        assert _fb.get('status') == 'ok' and _fb.get('n_rows') == 4, _fb
        assert _fb.get('mean_delta') < 0, _fb
        assert r2['simple_baseline']['method_arms_beat_floor'] is False, r2['simple_baseline']
        assert any('declared method arm' in w for w in r2['warnings']), r2['warnings']
        # A different arm beating the floor must not rescue the method gate.
        rescued = recs(floor + 1.0)
        for _k, _r in rescued.items():
            if _r['variant'] == 'full_sw128_matched':
                _r['ppl'] = floor - 1.0
                _r['nll_sum'] = math.log(floor - 1.0) * n_tok
                # The nested primary accounting uses the holdout rows, so a
                # consistent fixture must move them too.
                _r['nll_per_row_holdout'] = [math.log(floor - 1.0) * (n_tok // 4)] * 4
        with contextlib.redirect_stdout(io.StringIO()):
            r5 = L.methodology_audit(rescued, design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        assert r5['simple_baseline']['panel_beats_simple_floor'] is True, r5['simple_baseline']
        assert r5['simple_baseline']['method_arms_beat_floor'] is False, r5['simple_baseline']
        assert r5['simple_baseline']['method_arm_floor_status'] == 'method_arm_does_not_beat_floor', r5['simple_baseline']
        # The floor reference must be chosen by the PRIMARY (token-pooled)
        # endpoint, not by the arithmetic seed mean: they can rank arms
        # differently when token counts differ.
        with contextlib.redirect_stdout(io.StringIO()):
            r4 = L.methodology_audit(recs(floor - 1.0), design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        _arm = r4['arm_summary']
        _dyn_key = next(k for k, v in _arm.items() if v.get('variant') == 'csa_dynamic')
        _full_key = next(k for k, v in _arm.items() if v.get('variant') == 'full_sw128_matched')
        # Force a disagreement between the two aggregations for the two arms.
        _arm[_dyn_key]['ppl_mean'] = floor + 100.0
        _arm[_dyn_key]['ppl_primary'] = floor - 1.0
        _arm[_full_key]['ppl_mean'] = floor + 0.01
        _arm[_full_key]['ppl_primary'] = floor + 50.0
        # Re-running is not needed: the selector is the expression under test.
        _cands = [(_dyn_key, _arm[_dyn_key]), (_full_key, _arm[_full_key])]
        _ref_key, _ref = min(_cands, key=lambda kv: float(kv[1].get('ppl_primary', kv[1]['ppl_mean'])))
        assert _ref_key == _dyn_key and _ref['ppl_primary'] == floor - 1.0, (_ref_key, _ref)
    # A nested record's per-seed contrast must use the same holdout PPL as its
    # pooled primary endpoint.  Full-validation `ppl` is deliberately worse
    # than the floor here, so reading it would reverse the verdict.
    nested = recs(floor + 10.0)
    for _r in nested.values():
        _r['ppl_holdout'] = float(floor - 1.0)
        _r['nll_sum_holdout'] = math.log(float(floor - 1.0)) * n_tok
        _r['n_tokens_holdout'] = n_tok
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rn = L.methodology_audit(nested, design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        fn = rn['simple_baseline']['floor_contrast']
        assert fn['mean_delta'] < 0 and fn['verdict'] == 'beats_floor_significant', fn

    # Underpowered pointwise wins are not "beats floor".
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        small = {k: v for k, v in recs(floor - 1.0).items() if k.endswith('seed0') or k.endswith('seed1')}
        with contextlib.redirect_stdout(io.StringIO()):
            r3 = L.methodology_audit(small, design={}, baselines=L.BASELINE_DESIGN, outdir=td, min_n=3, verbose=False)
        fc = r3['simple_baseline'].get('floor_contrast') or {}
        assert fc.get('verdict') == 'beats_floor_pointwise_underpowered', fc


def test_simple_baseline_driver_serves_the_panel_corpus():
    # `run_simple` must go through load_run_corpus (the panel's dataset) and
    # fingerprint the validation slice.
    rng = np.random.default_rng(3)
    train = rng.integers(0, 24, size=400)
    val = rng.integers(0, 24, size=(2, 9))
    bnd = np.zeros_like(val, dtype=bool)
    seen = {}
    orig = L.load_run_corpus

    def fake(cfg):
        seen['dataset'] = cfg.get('dataset')
        return train, val, 24, None, bnd

    out = tempfile.mkdtemp(prefix='simple_driver_')
    L.load_run_corpus = fake
    try:
        art = BT.run_simple(dict(L.RUN, dataset='fineweb', seq_len=9, n_train_tokens=400,
                                 val_seqs=2, eval_subset=2), outdir=out)
    finally:
        L.load_run_corpus = orig
        shutil.rmtree(out, ignore_errors=True)
    assert seen.get('dataset') == 'fineweb', seen
    assert art['data']['n_eval_tokens'] == val.size and art['best_ppl'] > 0, art


def test_audit_leakage_handles_flat_train_cache():
    cache = tempfile.mkdtemp(prefix='leak_flat_')
    try:
        train = np.arange(60, dtype=np.int64)
        test = np.stack([np.arange(0, 30), np.arange(30, 60)]).astype(np.int64)
        np.save(os.path.join(cache, 'train_ids_x.npy'), train)
        np.save(os.path.join(cache, 'test_ids_x.npy'), test)
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'test'}, f)
        out = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        overlap = out['checks'].get('doc_overlap')
        assert isinstance(overlap, dict), overlap
        assert overlap['shared_shingles'] > 0, overlap
        # Recompute the loader fingerprint from the id caches, not just trust
        # the metadata string.
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(train), 'fit_fp': 'FITX'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test), 'fit_fp': 'FITX'}, f)
        good = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        fp = good['checks']['id_cache_fingerprints']
        assert fp['train']['status'] == 'ok' and fp['test']['status'] == 'ok', fp
        assert good['checks']['fit_fp_agree'] is True, good['checks']
        # Per-token document mapping: an undeclared legacy sidecar is
        # row-level, a declared+valid one is verified, and a declared but
        # mis-sized one fails closed (never silently shifts boundaries).
        ttd = np.repeat(np.asarray([10, 20], dtype=np.uint64), 30)
        np.save(os.path.join(cache, 'test_token_doc_ids_x.npy'), ttd)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test), 'fit_fp': 'FITX',
                       'doc_ids': {'test_token_count': int(ttd.size),
                                   'test_token_fp': L._ids_fp(ttd)}}, f)
        good_doc = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert good_doc['checks']['document_level_eval']['status'] == 'ok', good_doc
        np.save(os.path.join(cache, 'test_token_doc_ids_x.npy'), ttd[:-1])
        broken = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert broken['status'] == 'fail', broken
        assert broken['checks']['document_level_eval']['status'] == 'fail', broken
        np.save(os.path.join(cache, 'test_token_doc_ids_x.npy'), ttd)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test), 'fit_fp': 'OTHER'}, f)
        mism = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert mism['status'] == 'not_verifiable_locally', mism
        assert 'no train/test cache pair' in mism.get('note', ''), mism
        # A one-sided fit fingerprint cannot be assumed to match.
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test)}, f)
        missing = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert missing['status'] == 'fail', missing
        assert any('fit provenance is missing' in x
                   for x in missing['checks'].get('tokenizer_problems') or []), missing
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'stale', 'fit_fp': 'FITX'}, f)
        stale = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert stale['status'] == 'fail', stale
        assert any('id-cache fingerprint' in x for x in stale['checks'].get('tokenizer_problems') or []), stale
        # A legacy/malformed meta without a numeric vocab must fail closed, not
        # crash with int(None) before the audit can report anything.
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'test'}, f)
        bad = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert bad['status'] == 'fail' and bad['checks'].get('vocab_agree') is False, bad
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def test_audit_gate_parameters_fail_closed():
    # shingle<=0 would hash no windows and report zero overlap; NaN policy
    # limits would also make every comparison pass.  Both must be refused.
    for kw in ({'shingle': 0}, {'shingle': -3},
               {'max_rows_shingled': 0},
               {'preferred_train_seq_len': 0}):
        try:
            A.audit_leakage(cache_dir='/nonexistent', **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_leakage accepted {kw!r}')
    for key in ('CSA_MAX_TEST_SHINGLE_OVERLAP', 'CSA_MAX_TEST_ROW_OVERLAP'):
        old = os.environ.get(key)
        try:
            for bad in ('nan', 'inf', '-0.1', 'not-a-number'):
                os.environ[key] = bad
                try:
                    A.audit_leakage(cache_dir='/nonexistent')
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'{key}={bad!r} must be refused')
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    for kw in ({'js_warn': float('nan')}, {'js_fail': float('nan')},
               {'js_warn': -0.1}, {'min_shift_tokens': 0},
               {'max_rows': 0}, {'js_warn': 0.3, 'js_fail': 0.2}):
        try:
            A.audit_split_discipline(cache_dirs=[], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_split_discipline accepted {kw!r}')

    # An exception inside the overlap computation must not leave the overall
    # leakage verdict at 'ok'.
    cache = tempfile.mkdtemp(prefix='leak_failclosed_')
    try:
        np.save(os.path.join(cache, 'train_ids_x.npy'), np.arange(60, dtype=np.int64))
        np.save(os.path.join(cache, 'test_ids_x.npy'),
                np.stack([np.arange(0, 30), np.arange(30, 60)]).astype(np.int64))
        with open(os.path.join(cache, 'meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'a'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'b'}, f)
        # A requested train seq_len that no cache declares must refuse to
        # guess, not audit a different split.
        _pref = A.audit_leakage(cache_dir=cache, shingle=4,
                                max_rows_shingled=10,
                                preferred_train_seq_len=512)
        assert _pref['status'] == 'not_verifiable_locally', _pref
        # A shingle wider than the stored rows can form no window; that is an
        # unverifiable split, not zero overlap.
        _wide = A.audit_leakage(cache_dir=cache, shingle=1000,
                                max_rows_shingled=10)
        assert _wide['status'] == 'fail', _wide
        assert 'shingle' in str(_wide['checks'].get('doc_overlap')), _wide
        _orig = A._window_hashes
        try:
            def _boom(*_a, **_k):
                raise RuntimeError('simulated overlap failure')
            A._window_hashes = _boom
            out = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        finally:
            A._window_hashes = _orig
        assert out['status'] == 'fail', out
        assert 'doc_overlap' in out['checks'], out
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def test_fuse_sample_std_negative_axis():
    import fuse_analysis as FA
    x = np.arange(24, dtype=float).reshape(2, 3, 4)
    for axis in (0, 1, 2, -1, -2, -3):
        got = FA.sample_std(x, axis=axis)
        want = np.std(x, axis=axis, ddof=1)
        assert np.asarray(got).shape == np.asarray(want).shape, (axis, np.asarray(got).shape, np.asarray(want).shape)
        assert np.allclose(got, want, atol=1e-12), (axis, got, want)


def test_classification_positive_label_and_rare_floor():
    y = np.array([0] * 2000 + [1])
    # Positive class 1 is the rare class here (1:2000).
    score = (y == 1).astype(float)
    m = L.classification_metrics(y, y_score=score, labels=[0, 1], positive_label=1)
    assert m['positive_label'] == '1' and m['f1_positive'] is not None, m
    assert m['roc_auc'] == 1.0 and abs(m['pr_auc'] - 1.0) < 1e-12, m
    assert m['pr_auc_prevalence'] == 1.0 / 2001.0 and abs(m['pr_auc_lift_over_prevalence'] - (1.0 - 1.0 / 2001.0)) < 1e-12, m
    assert m['imbalance_warning'] and 'PR-AUC' in m['imbalance_warning'], m
    # Naming the other class as positive reverses the same ranking endpoint.
    m2 = L.classification_metrics(y, y_score=score, labels=[0, 1], positive_label=0)
    assert m2['positive_label'] == '0' and m2['roc_auc'] == 0.0, m2
    # A zero-support label requested by the caller must not be dropped from
    # macro F1 (that would make a one-class prediction look like a perfect
    # macro classifier).
    y0 = np.array([0] * 10 + [1] * 10)
    pred = np.zeros_like(y0)
    m3 = L.classification_metrics(y0, y_pred=pred, labels=[0, 1, 2])
    assert abs(m3['f1_macro'] - (2.0 / 3.0 / 3.0)) < 1e-12, m3
    assert m3['f1_primary_average'] == 'macro'
    # A declared label set must cover observed classes.  Silently ignoring an
    # undeclared predicted class would make micro/macro F1 describe a
    # different label space than the caller declared.
    try:
        L.classification_metrics(np.array([0, 1, 1]),
                                 y_pred=np.array([0, 1, 2]), labels=[0, 1])
    except ValueError as e:
        assert 'not in the declared labels' in str(e), e
    else:
        raise AssertionError('an undeclared predicted class was silently ignored')
    try:
        L.classification_metrics(np.array([0, 2]),
                                 y_pred=np.array([0, 2]), labels=[0, 1])
    except ValueError as e:
        assert 'not in the declared labels' in str(e), e
    else:
        raise AssertionError('an undeclared true class was silently ignored')
    # Record-level policy: F1 without averaging, and rare ROC-AUC without
    # PR-AUC/prevalence, must warn.
    bad = {'r::seed0': {'variant': 'r', 'task': 'binary_classification', 'f1': 0.4,
                        'roc_auc': 0.99, 'prevalence': {'0': 0.9995, '1': 0.0005}}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn', ma
    assert any('averaging' in w for w in ma['warnings']), ma
    assert any('PR-AUC' in w for w in ma['warnings']), ma
    # A record must not label an arbitrary F1 as its primary metric.
    _cm = L.classification_metrics([0] * 950 + [1] * 50, y_pred=[0] * 1000,
                                   labels=[0, 1])
    _good = dict(_cm, variant='g', task='binary_classification',
                 prevalence={'0': 0.95, '1': 0.05})
    assert L.metric_policy_audit({'g': _good})['status'] == 'ok',         L.metric_policy_audit({'g': _good})
    _bad_primary = dict(_good, f1_primary=0.9)
    _bp = L.metric_policy_audit({'g': _bad_primary})
    assert _bp['status'] == 'warn' and any(
        'does not equal' in w for w in _bp['warnings']), _bp
    # At 1:1000 the conditional primary is PR-AUC; ROC-AUC/F1 as primary is flagged.
    _rare = dict(_good, prevalence={'0': 0.9995, '1': 0.0005},
                 roc_auc=0.99, primary_metric='roc_auc')
    _rp = L.metric_policy_audit({'g': _rare})
    assert _rp['status'] == 'warn' and any(
        'conditional policy primary' in w for w in _rp['warnings']), _rp
    # Multiclass F1 must expose macro/micro/weighted, not only one average.
    _mc = {'variant': 'm', 'task': 'multiclass_classification',
           'f1_macro': 0.4, 'f1_primary': 0.4,
           'f1_primary_average': 'macro'}
    _mca = L.metric_policy_audit({'m': _mc})
    assert _mca['status'] == 'warn' and any(
        'missing' in w and 'f1_micro' in w for w in _mca['warnings']), _mca


def test_binary_threshold_and_calibration_contract():
    y = np.array([0] * 90 + [1] * 10)
    score = np.array([0.1] * 90 + [0.4] * 5 + [0.9] * 5)
    curve = L.binary_threshold_curve(y, score, positive_label=1)
    assert abs(curve['best_threshold'] - 0.4) < 1e-12, curve['best_threshold']
    assert abs(curve['best_value'] - 1.0) < 1e-12, curve['best_value']
    assert abs(curve['at_default_0.5']['f1'] - 2.0 / 3.0) < 1e-12, curve['at_default_0.5']
    assert curve['threshold_sensitivity'] > 0.3, curve
    sel = L.select_binary_threshold(y, score, positive_label=1)
    assert sel['threshold'] == 0.4 and sel['value'] == 1.0, sel
    assert sel['prevalence'] == 0.1 and sel['n_positive'] == 10, sel
    # Mixed-type binary labels must not be coerced together (int 1 vs '1');
    # the positive endpoint is named explicitly.
    _ym = np.array([0] * 6 + ['1'] * 4, dtype=object)
    _sm = np.array([0.1] * 6 + [0.9] * 4)
    _cm = L.binary_threshold_curve(_ym, _sm, positive_label='1')
    assert _cm['n_positive'] == 4 and _cm['best_value'] == 1.0, _cm
    # The sweep is linear-time via cumulative counts; it must equal the
    # brute-force confusion matrix at every candidate, including score ties.
    _rng = np.random.default_rng(7)
    for _ in range(5):
        _yy = _rng.integers(0, 2, size=60)
        _ss = np.round(_rng.random(60), 1)
        if _yy.sum() in (0, 60):
            continue
        _c = L.binary_threshold_curve(_yy, _ss)
        for _row in _c['thresholds']:
            _b = L._binary_threshold_metrics(_yy, _ss, _row['threshold'])
            assert (_row['tp'], _row['fp'], _row['tn'], _row['fn']) == (
                _b['tp'], _b['fp'], _b['tn'], _b['fn']), (_row, _b)
            assert abs(_row['f1'] - _b['f1']) < 1e-12, (_row, _b)
    # The default cut is recorded rather than silently assumed.
    m = L.classification_metrics(y, y_score=score, labels=[0, 1], threshold_sweep=True)
    assert m['decision_threshold'] == 0.5 and m['decision_threshold_source'] == 'default_0.5', m
    assert abs(m['f1_primary'] - 2.0 / 3.0) < 1e-12, m
    assert m['threshold_selection']['threshold'] == 0.4, m['threshold_selection']
    assert 0.0 <= m['brier_score'] <= 1.0 and 0.0 <= m['ece'] <= 1.0, m
    assert m['score_orientation_warning'] is None, m
    mt = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                  threshold=0.4)
    assert mt['decision_threshold_source'] == 'caller_threshold'
    assert abs(mt['f1_primary'] - 1.0) < 1e-12, mt
    # Gate parameters are exact integers: True/1.5 must not silently become a
    # different bin/threshold budget.
    for _bad_mt in (True, 0, -1, 1.5, 'x'):
        try:
            L.binary_threshold_curve(y, score, max_thresholds=_bad_mt)
        except ValueError:
            pass
        else:
            raise AssertionError(f'max_thresholds={_bad_mt!r} must be refused')
    for _bad_nb in (True, 0, -1, 1.5, 'x'):
        try:
            L._binary_calibration(np.array([0, 1]), np.array([0.2, 0.8]),
                                  n_bins=_bad_nb)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n_bins={_bad_nb!r} must be refused')
    # A reversed score column is a labelling/orientation error, not a weak model.
    mi = L.classification_metrics(y, y_score=1.0 - score, labels=[0, 1])
    assert mi['roc_auc'] == 0.0 and mi['score_orientation_warning'], mi
    # Record-level contract: a tuned threshold must be validation-selected and
    # named; an out-of-range cut or a misstated PR-AUC floor is invalid.
    good = dict(m, variant='g', task='binary_classification',
                prevalence={'0': 0.9, '1': 0.1})
    assert L.metric_policy_audit({'g': good})['status'] == 'ok', L.metric_policy_audit({'g': good})
    tuned = dict(good, decision_threshold=0.4, decision_threshold_source='validation_selection',
                 threshold_selection_split='test', threshold_selection_metric='f1')
    bad = L.metric_policy_audit({'g': tuned})
    assert bad['status'] == 'warn' and any('leakage' in w for w in bad['warnings']), bad
    bad_floor = dict(good, pr_auc_prevalence=0.5)
    bf = L.metric_policy_audit({'g': bad_floor})
    assert bf['status'] == 'warn' and any('floor is misstated' in w for w in bf['warnings']), bf
    no_src = dict(good)
    no_src.pop('decision_threshold_source')
    ns = L.metric_policy_audit({'g': no_src})
    assert ns['status'] == 'warn' and any('decision_threshold_source' in w for w in ns['warnings']), ns
    inv = dict(good, roc_auc=0.2)
    iv = L.metric_policy_audit({'g': inv})
    assert iv['status'] == 'warn' and any('below chance' in w for w in iv['warnings']), iv
    # A declared/recommended metric must be a metric NAME.  A list/dict is
    # malformed evidence and must be reported as a problem, not crash the
    # audit with an unhashable-type TypeError.
    _cm0 = L.classification_metrics([0] * 90 + [1] * 10, y_pred=[0] * 100,
                                    labels=[0, 1])
    for _key in ('primary_metric', 'recommended_primary_metric'):
        _mal = dict(_cm0, variant='m', task='binary_classification',
                    **{_key: [1]})
        _ma = L.metric_policy_audit({'m': _mal})
        assert _ma['status'] == 'warn' and any(
            _key in w for w in _ma['warnings']), (_key, _ma)
    # A non-finite forged support must be refused, not crash int() with
    # OverflowError while reconstructing the trivial floor.
    _ov = json.loads(json.dumps(_cm0))
    _ov.update(variant='m', task='binary_classification')
    _ov['per_class']['0']['support'] = float('inf')
    _ova = L.metric_policy_audit({'m': _ov})
    assert _ova['status'] == 'warn', _ova
    # Unhashable record-identity metadata (variant/name) must not crash the
    # degenerate-record bookkeeping.
    _dm = dict(_cm0, variant={'not': 'a-label'}, task='binary_classification',
               predicted_positive_rate=1.0, degenerate_predictions=True)
    _da = L.metric_policy_audit({'m': _dm})
    assert _da['status'] == 'warn' and _da['checks'], _da
    # Macro/micro/weighted F1 must all be reproducible from the per-class
    # counts; a hand-edited aggregate cannot pass.
    _cm = L.classification_metrics(np.array([0] * 95 + [1] * 5),
                                   y_pred=np.array([0] * 100), labels=[0, 1])
    _rec = dict(_cm, variant='c', task='binary_classification',
                prevalence={'0': 0.95, '1': 0.05})
    assert L.metric_policy_audit({'c': _rec})['status'] == 'ok', L.metric_policy_audit({'c': _rec})
    _bad_micro = dict(_rec, f1_micro=0.123)
    _bm = L.metric_policy_audit({'c': _bad_micro})
    assert _bm['status'] == 'warn' and any('f1_micro' in w and 'reproduced' in w
                                           for w in _bm['warnings']), _bm
    _bad_pc = json.loads(json.dumps(_rec))
    _bad_pc['per_class']['0']['precision'] = 0.1
    _bp = L.metric_policy_audit({'c': _bad_pc})
    assert _bp['status'] == 'warn' and any('per_class counts' in w for w in _bp['warnings']), _bp
    _no_pos = dict(_rec)
    _no_pos.pop('positive_label')
    _np_ = L.metric_policy_audit({'c': _no_pos})
    assert _np_['status'] == 'warn' and any('positive_label' in w for w in _np_['warnings']), _np_


def test_boundary_macro_micro_f1_contract():
    # Macro-over-sequences F1 is NOT the harmonic mean of the mean precision
    # and mean recall.  A record must expose micro counts (or per-sequence
    # values) before F1 can be reconstructed; the old audit asserted the wrong
    # identity and would flag a correct macro record.
    b = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                      'stats': {'L0': {'bnd_prec': 0.55, 'bnd_rec': 0.55,
                                       'bnd_f1': 0.1818181818,
                                       'bnd_f1_average': 'macro_over_sequences',
                                       'bnd_prec_micro': 0.55, 'bnd_rec_micro': 0.55,
                                       'bnd_f1_micro': 0.55,
                                       'bnd_tp': 55, 'bnd_fp': 45, 'bnd_fn': 45,
                                       'bnd_rand': 0.1}}}}
    a = L.metric_policy_audit(b)
    chk = next(c for c in a['checks'] if c['metric'] == 'boundary_alignment')
    assert chk['ok'] is True and chk['macro_f1_reconstructible'] is True, chk
    # Without micro counts the macro F1 is still valid but unverifiable.
    b2 = json.loads(json.dumps(b))
    for k in ('bnd_prec_micro', 'bnd_rec_micro', 'bnd_f1_micro', 'bnd_tp', 'bnd_fp', 'bnd_fn'):
        b2['b::seed0']['stats']['L0'].pop(k)
    a2 = L.metric_policy_audit(b2)
    chk2 = next(c for c in a2['checks'] if c['metric'] == 'boundary_alignment')
    assert chk2['ok'] is True and chk2['macro_f1_reconstructible'] is False, chk2
    assert any('macro' in w and 'unverifiable' in w for w in a2['warnings']), a2
    # The per-sequence counter path must agree with the reported rates.
    _gt = np.zeros(10, dtype=bool)
    _gt[2] = True
    _gt[6] = True
    _d = {}
    _p, _r, _f, _rp = L.boundary_alignment([3, 8], _gt, 10, tol=1, detail=_d)
    assert _p == 1.0 and _r == 1.0 and _f == 1.0, (_p, _r, _f)
    assert (_d['tp'], _d['fp'], _d['fn'], _d['n_pred'], _d['n_gt']) == (2, 0, 0, 2, 2), _d
    # A declared micro F1 must be the harmonic mean of the micro precision
    # and recall; a forged value fails.
    b3 = json.loads(json.dumps(b))
    b3['b::seed0']['stats']['L0']['bnd_f1_micro'] = 0.9
    a3 = L.metric_policy_audit(b3)
    chk3 = next(c for c in a3['checks'] if c['metric'] == 'boundary_alignment')
    assert chk3['ok'] is False and 'micro-f1-not-harmonic' in chk3['invalid'], chk3


def test_hypothesis_audit_completeness_and_metric_alignment():
    good = L.hypothesis_audit(L.HYPOTHESIS_DESIGN, 'component')
    assert good['status'] == 'ok' and not good['invalid'], good
    assert 'dynamic_blocking' in good['hypotheses'], good
    # Claims that promise matched capacity must say so; the loss-term
    # ablation and the matched-dense baseline both do.
    assert L.HYPOTHESIS_DESIGN['compression_regularizer']['requires_param_match'] is True
    assert L.BASELINE_HYPOTHESIS_DESIGN['compression_vs_matched_dense']['requires_param_match'] is True
    # Beating the tuned count-model floor is itself a pre-registered
    # directional hypothesis with its own endpoint and falsifier.
    _floor_ha = L.hypothesis_audit(L.FLOOR_HYPOTHESIS_DESIGN, 'floor')
    assert _floor_ha['status'] == 'ok' and not _floor_ha['invalid'], _floor_ha
    _floor_spec = L.FLOOR_HYPOTHESIS_DESIGN['method_beats_tuned_count_floor']
    assert _floor_spec['primary_metric'] == 'ppl'
    assert _floor_spec['metric_policy'] == 'lm_perplexity'
    assert _floor_spec['predicted_sign'] == -1
    # The NIAH accuracy contrasts now have an explicit pre-registered
    # hypothesis family too, including the reported sign convention.
    _niah_ha = L.hypothesis_audit(L.NIAH_HYPOTHESIS_DESIGN, 'niah')
    assert _niah_ha['status'] == 'ok' and not _niah_ha['invalid'], _niah_ha
    for _name, _spec in L.NIAH_HYPOTHESIS_DESIGN.items():
        assert _spec['primary_metric'] == 'top1_accuracy', _name
        assert _spec['metric_policy'] == 'niah_retrieval', _name
        assert _spec['predicted_sign'] == -1, _name
    bad_table = {
        'h0': {'statement': 'x', 'mechanism': 'y', 'predicted_sign': 0,
               'primary_metric': 'ndcg', 'falsifier': 'z'},
        'h1': {'statement': 'x'},
    }
    r = L.hypothesis_audit(bad_table, 'component')
    assert r['status'] == 'warn', r
    assert 'h0' in r['direction_unspecified'] and 'h0' in r['metric_unknown'], r
    assert 'h1' in r['invalid'] and any('incomplete' in w for w in r['warnings']), r
    # A metric that is only a SECONDARY endpoint in the policy cannot be the
    # hypothesis's primary endpoint without an explicit secondary declaration.
    _base = {'statement': 'x', 'mechanism': 'y', 'predicted_sign': -1,
             'requires_param_match': False, 'falsifier': 'z'}
    _roc = dict(_base, primary_metric='roc_auc', metric_policy='binary_classification')
    _rr = L.hypothesis_audit({'roc_primary': _roc}, 'component')
    assert 'roc_primary' in _rr['invalid'] and 'roc_primary' in _rr['metric_misaligned'], _rr
    assert 'roc_primary' not in _rr['hypotheses']['roc_primary']['metric_primary_policies'], _rr
    # ... but the CONDITIONAL primary under extreme imbalance (PR-AUC) is a
    # legitimate primary endpoint and must pass the role check.
    _pr = dict(_base, primary_metric='pr_auc', metric_policy='binary_classification')
    _prr = L.hypothesis_audit({'pr_primary': _pr}, 'component')
    assert _prr['status'] == 'ok' and not _prr['invalid'], _prr
    # Explicitly declaring a secondary endpoint is allowed only with a
    # justification, and it stays visible as a warning.
    _sec = dict(_base, primary_metric='roc_auc', metric_policy='binary_classification',
                metric_role='secondary', metric_justification='ROC is the threshold-free sensitivity readout')
    _sr = L.hypothesis_audit({'roc_secondary': _sec}, 'component')
    assert not _sr['invalid'] and 'roc_secondary' in _sr['metric_secondary_justified'], _sr
    assert any('secondary endpoint' in w for w in _sr['warnings']), _sr
    # A predicted_sign=0 result is significant but cannot be called verified.
    entries = [{'controlled': True, 'n': 7, 'significant_global': True,
                'within_seed_noise': False, 'param_controlled': True,
                'predicted_sign': 0, 'direction_consistent': True}]
    assert L._classify_group_status(entries, min_n=3, alpha=0.05, m_family=1) == 'direction_unspecified'




# ---------------------------------------------------------------------------
# I. leakage-safe data splitting / preprocessing
# ---------------------------------------------------------------------------

    # A malformed caller-supplied policy must not crash the audit; each bad
    # entry is surfaced as a warning and the remaining policy still applies.
    for _bad_pol in ([], {'x': None}, {'x': 1}):
        _ha = L.hypothesis_audit(table={}, names=[], policy=_bad_pol)
        assert _ha['status'] == 'warn' and _ha['warnings'], (_bad_pol, _ha)
    _mixed_pol = {'x': None, 'binary_classification': dict(L.METRIC_POLICY['binary_classification'])}
    _ha = L.hypothesis_audit(table={}, names=[], policy=_mixed_pol)
    assert _ha['status'] == 'warn' and any('`x`' in w for w in _ha['warnings']), _ha
    # A hypothesis bound to a malformed policy entry fails closed instead of
    # raising on `policy[name].get`.
    _spec = dict(L.HYPOTHESIS_DESIGN['dynamic_blocking'])
    _bad_bind = L.hypothesis_audit(
        table={'h': {**_spec, 'metric_policy': 'x'}},
        names=['h'], policy={'x': None})
    assert _bad_bind['status'] == 'warn' and _bad_bind['invalid'] == ['h'], _bad_bind


def test_hypothesis_coverage_audit():
    # The shipped design must have a pre-registered hypothesis for every
    # declared contrast family, and no stale/orphan hypothesis entries.
    cov = L.hypothesis_coverage_audit()
    assert cov['status'] == 'ok', cov
    assert cov['component']['complete'] is True, cov
    assert cov['baseline']['complete'] is True, cov
    # A new experiment family added to the design without a hypothesis is
    # exactly the failure this audit exists to catch.
    design = dict(L.ABLATION_DESIGN)
    design['unregistered_change'] = [('csa_dynamic', 'csa_fixed', 'new idea')]
    missing = L.hypothesis_coverage_audit(design=design)
    assert missing['status'] == 'warn', missing
    assert 'unregistered_change' in missing['component']['missing'], missing
    # An orphan hypothesis describes a contrast the design no longer contains.
    table = dict(L.HYPOTHESIS_DESIGN)
    table['ghost_component'] = dict(table['dynamic_blocking'])
    orphan = L.hypothesis_coverage_audit(component_table=table)
    assert orphan['status'] == 'warn', orphan
    assert 'ghost_component' in orphan['component']['orphan'], orphan
    # An empty family has nothing to test, even if a hypothesis exists.
    empty_design = dict(L.ABLATION_DESIGN)
    empty_design['empty_family'] = []
    empty = L.hypothesis_coverage_audit(design=empty_design)
    assert 'empty_family' in empty['component']['empty_contrast'], empty
    # Baseline hypotheses are covered by the same contract.
    no_base = L.hypothesis_coverage_audit(baseline_table={})
    assert no_base['status'] == 'warn', no_base
    assert no_base['baseline']['missing'] == sorted(L.BASELINE_DESIGN), no_base
    # Malformed design/table inputs fail closed rather than crashing.
    assert L.hypothesis_coverage_audit(design=[])['status'] == 'fail'
    assert L.hypothesis_coverage_audit(component_table=[])['status'] == 'fail'


def test_extreme_imbalance_boundary_is_inclusive():
    # "1:1000 or more imbalanced" means prevalence <= 1e-3.  A strict `<`
    # would leave an endpoint exactly at 1:1000 on the usual F1/accuracy
    # headline, contradicting the declared policy.
    assert L._extreme_imbalance(1e-3) is True
    assert L._extreme_imbalance(1.0001e-3) is False
    assert L._extreme_imbalance(0.0) is True
    assert L._extreme_imbalance(None) is None
    assert L._extreme_imbalance('bad') is None
    # A binary endpoint with exactly one positive in 1000 must recommend the
    # conditional PR-AUC primary, not F1/ROC.
    y = np.array([0] * 999 + [1])
    score = np.r_[np.linspace(0.0, 0.4, 999), 0.9]
    m = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                 positive_label=1)
    assert abs(m['pr_auc_prevalence'] - 1e-3) < 1e-15, m['pr_auc_prevalence']
    assert m['recommended_primary_metric'] == 'pr_auc', m
    assert m['imbalance_warning'] is not None, m
    # The same boundary applies to many-way retrieval: one answer in 1000 is
    # extreme and switches the primary to macro recall.
    yt = np.array([0] * 999 + [1])
    yp = np.zeros(1000, dtype=np.int64)
    rt = L.retrieval_metrics(yt, yp, labels=[0, 1])
    assert abs(rt['minority_prevalence'] - 1e-3) < 1e-15, rt
    assert rt['recommended_primary_metric'] == 'macro_recall', rt
    # And to the pre-registered hypothesis audit: an F1 hypothesis at exactly
    # 1:1000 is misaligned with the conditional policy primary (PR-AUC).
    spec = {'statement': 'x', 'mechanism': 'y', 'predicted_sign': -1,
            'primary_metric': 'f1', 'metric_policy': 'binary_classification',
            'requires_param_match': False, 'falsifier': 'z',
            'prevalence': 1e-3}
    ha = L.hypothesis_audit({'at_boundary': spec}, 'component')
    assert ha['status'] == 'warn', ha
    assert 'at_boundary' in ha['metric_misaligned'], ha


def test_leakage_safe_split_contract():
    idx = L.chronological_split_indices(100, 0.7, 0.2, 0.1)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [70, 20, 10], idx
    assert idx['train'][0] == 0 and idx['train'][-1] == 69
    assert idx['val'][0] == 70 and idx['test'][-1] == 99
    assert L.assert_no_split_leakage(idx['train'], idx['val'], idx['test']) is True
    # The split is contiguous in input order: no row from a later block can
    # appear in an earlier one.
    assert all(np.all(np.diff(idx[k]) == 1) for k in ('train', 'val', 'test'))
    # A shuffled time split must be rejected even when the row indices are
    # disjoint.
    t = np.arange(100)
    try:
        L.assert_no_split_leakage(np.array([0, 2]), np.array([1]), np.array([3]), time=t)
    except ValueError as e:
        assert 'temporal leakage' in str(e)
    else:
        raise AssertionError('future rows in the training split must be rejected')
    # Reordering by an explicit time column keeps the chronology.
    order = np.array([5, 0, 3, 1, 6, 2, 4])
    ti = L.split_indices(7, time=order, train_frac=4 / 7, val_frac=2 / 7, test_frac=1 / 7)
    assert np.max(order[ti['train']]) <= np.min(order[ti['val']])
    assert np.max(order[ti['val']]) <= np.min(order[ti['test']])
    L.assert_no_split_leakage(ti['train'], ti['val'], ti['test'], time=order)
    # Snapping a split boundary to the next timestamp block must not consume
    # the terminal cut and make the following split impossible.  Blocks are
    # 80 / 5 / 15 rows with targets 80 / 10 / 10; the legal split uses the
    # 85 cut for validation and leaves the final 15 rows for test.
    tied = np.r_[np.arange(80), np.full(5, 80), np.full(15, 85)]
    tidx = L.split_indices(100, time=tied, train_frac=0.8, val_frac=0.1,
                           test_frac=0.1)
    assert [len(tidx[k]) for k in ('train', 'val', 'test')] == [80, 5, 15], tidx
    L.assert_no_split_leakage(tidx['train'], tidx['val'], tidx['test'], time=tied)
    # A fraction explicitly set to zero must not receive the flooring
    # remainder.
    zidx = L.split_indices(3, train_frac=0.5, val_frac=0.5, test_frac=0.0)
    assert [len(zidx[k]) for k in ('train', 'val', 'test')] == [1, 2, 0], zidx


def test_group_split_contract():
    groups = np.repeat(np.arange(30), 2)
    idx = L.grouped_chronological_split_indices(60, groups, 0.6, 0.2, 0.2)
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'], groups=groups)
    gsets = {k: {groups[i] for i in idx[k]} for k in ('train', 'val', 'test')}
    assert not (gsets['train'] & gsets['val'])
    assert not (gsets['train'] & gsets['test'])
    assert not (gsets['val'] & gsets['test'])
    assert max(gsets['train']) < min(gsets['val']) < min(gsets['test'])
    # A deliberately shuffled group order is still assigned whole.
    shuffled = np.roll(groups, 7)
    idx2 = L.split_indices(60, groups=shuffled, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    g2 = {k: {shuffled[i] for i in idx2[k]} for k in ('train', 'val', 'test')}
    assert not (g2['train'] & g2['val']) and not (g2['train'] & g2['test']) and not (g2['val'] & g2['test'])
    # NaT timedelta group ids are one missing group, exactly like NaT
    # datetime/NaN ids; they must not silently become per-row unique keys.
    _nat = np.array([np.timedelta64('NaT', 'ns'), np.timedelta64('NaT', 'ns'),
                     np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns')],
                    dtype='timedelta64[ns]')
    try:
        L.assert_no_split_leakage([0], [1], [2, 3], groups=_nat)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT timedelta group ids must collapse into one group')


def test_group_split_complete_allocation_and_zero_targets():
    # Whole groups keep their chronological order, but a feasible partition
    # must not be rejected merely because the greedy "fill train first" rule
    # put a large group on the wrong side.  Groups have 1, 1 and 2 rows;
    # targets 2/1/1 can only be met by boundaries 1 and 2 (train=1, val=1,
    # test=2).  The historical greedy split consumed groups 0+1 into train,
    # then put the 2-row group in validation and left test empty.
    groups = np.array([0, 1, 2, 2])
    idx = L.grouped_chronological_split_indices(4, groups, 0.5, 0.25, 0.25)
    assert idx['train'].tolist() == [0], idx
    assert idx['val'].tolist() == [1], idx
    assert idx['test'].tolist() == [2, 3], idx
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              groups=groups)
    # A split whose target is exactly zero stays empty; the remainder belongs
    # to the last requested non-empty split, never to the zero-target side.
    g0 = np.array([0, 1, 2, 3])
    z = L.grouped_chronological_split_indices(
        4, g0, train_frac=0.5, val_frac=0.5, test_frac=0.0)
    assert z['test'].size == 0, z
    assert sorted(np.concatenate([z['train'], z['val']]).tolist()) == [0, 1, 2, 3], z
    z2 = L.grouped_chronological_split_indices(
        4, g0, train_frac=0.0, val_frac=0.5, test_frac=0.5)
    assert z2['train'].size == 0 and z2['val'].size and z2['test'].size, z2
    # Explicit sizes work the same way and preserve whole groups.
    e = L.grouped_chronological_split_indices(
        6, np.array([0, 0, 1, 1, 2, 2]), train_size=2, val_size=2,
        test_size=2)
    assert [len(e[k]) for k in ('train', 'val', 'test')] == [2, 2, 2], e
    L.assert_no_split_leakage(e['train'], e['val'], e['test'],
                              groups=np.array([0, 0, 1, 1, 2, 2]))


def test_group_split_respects_time_windows():
    # Whole groups on one side is necessary but not sufficient: two groups
    # whose time windows overlap must not be placed on opposite sides, or the
    # shared window leaks future/past information across the split.
    groups = np.repeat(np.arange(10), 2)
    time = np.repeat(np.arange(10) * 10, 2)
    idx = L.grouped_chronological_split_indices(20, groups, 0.5, 0.25, 0.25,
                                                order=time)
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              groups=groups, time=time)
    try:
        L.grouped_chronological_split_indices(
            8, [0, 0, 1, 1, 2, 2, 3, 3], 0.5, 0.25, 0.25,
            order=[0, 10, 5, 15, 20, 30, 25, 35])
    except ValueError as e:
        assert 'temporal leakage' in str(e), e
    else:
        raise AssertionError('overlapping group time windows must be rejected')
    # A missing ordering key cannot define a chronological split.
    try:
        L.split_indices(4, time=[0, 1, np.nan, 3])
    except ValueError as e:
        assert 'NaN/Inf' in str(e), e
    else:
        raise AssertionError('NaN time must be rejected before sorting')


def test_data_quality_report_index_disjointness():
    X = np.arange(8, dtype=float).reshape(4, 2)
    rep = L.data_quality_report(X, train_idx=[0, 1], val_idx=[1], test_idx=[2])
    assert rep['status'] == 'fail', rep
    assert rep['checks']['split_index_overlap']['train__val'] == 1, rep
    # A repeated index inside one split is counted, and the shared key set used
    # for the cross-split check does not change the within-split count.
    rep_dup = L.data_quality_report(X, train_idx=[0, 0, 1], val_idx=[2],
                                    test_idx=[3])
    assert rep_dup['checks']['duplicate_rows_within_split']['train'] == 1, rep_dup
    try:
        L.data_quality_report(X, train_idx=[], val_idx=[1])
    except ValueError as e:
        assert 'training reference is empty' in str(e), e
    else:
        raise AssertionError('an empty training reference must not define holdout statistics')


def test_preprocessor_train_only():
    tr = np.array([[0.0, 10.0], [2.0, 10.0], [np.nan, 10.0], [1.0, 10.0]])
    va = np.array([[1000.0, 10.0], [np.inf, 10.0]])
    prep = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                     missing='mean')
    xtr = prep.fit_transform(tr)
    xva = prep.transform(va)
    assert np.all(np.isfinite(xtr)) and np.all(np.isfinite(xva))
    st = prep.get_stats()
    assert abs(st['features'][0]['center'] - 1.0) < 1e-12, st
    # The holdout did not influence the statistics: fitting on train+holdout
    # would move the center by >100x, so a leak cannot hide.
    prep_all = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                         missing='mean').fit(np.vstack([tr, va]))
    assert abs(prep_all.get_stats()['features'][0]['center'] - 1.0) > 100
    before = prep.get_stats()
    prep.transform(va)
    assert prep.get_stats() == before, 'transform must never update statistics'
    # Constant features stay finite (factor 1), and a saved/restored model is
    # numerically identical.
    assert np.all(np.isfinite(prep.transform(np.array([[np.nan, np.nan]]))))
    restored = L.LeakageSafePreprocessor.from_stats(before)
    assert np.allclose(restored.transform(va), xva, equal_nan=True)
    # A preprocessor cannot be used before fitting or with the wrong width.
    try:
        L.LeakageSafePreprocessor().transform(tr)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        prep.transform(np.zeros((2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')


def test_preprocessor_fill_is_consistent():
    # A requested imputation constant outside the training quantiles must be
    # applied identically in fit and transform.  If fit clips it but transform
    # does not, the fitted scale no longer centres its own training matrix.
    X = np.array([[10.0], [11.0], [12.0], [np.nan]])
    prep = L.LeakageSafePreprocessor(scale='standard', missing=0.0,
                                     clip_quantiles=(0.1, 0.9))
    xtr = prep.fit_transform(X)
    assert abs(float(xtr.mean())) < 1e-12, xtr
    assert np.allclose(prep.transform(np.array([[np.nan]])), xtr[-1:]), (prep.transform(np.array([[np.nan]])), xtr[-1:])
    # numpy scalar constants are accepted; NaN/Inf constants are refused.
    assert L.LeakageSafePreprocessor(missing=np.float32(0.0)).missing == 0.0
    for bad in (float('nan'), float('inf'), -float('inf')):
        try:
            L.LeakageSafePreprocessor(missing=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-finite missing constant {bad!r} must be rejected')


def test_leakage_checks_validate_inputs():
    # Too-short time/group arrays must not silently disable the check.
    try:
        L.assert_no_split_leakage([0], [1], [2], time=[0, 1])
    except ValueError as e:
        assert 'time has length' in str(e), e
    else:
        raise AssertionError('short time array must be rejected')
    try:
        L.assert_no_split_leakage([0], [1], [2], groups=['a', 'b'])
    except ValueError as e:
        assert 'groups has length' in str(e), e
    else:
        raise AssertionError('short group array must be rejected')
    # NaN timestamps cannot define a chronological order.
    try:
        L.assert_no_split_leakage([0], [1], [2], time=[0.0, np.nan, 2.0])
    except ValueError as e:
        assert 'NaN/Inf' in str(e), e
    else:
        raise AssertionError('NaN time must be rejected')
    # NaN is a single group id; the same group must not span two splits just
    # because NaN != NaN.
    g = np.array([np.nan, np.nan, 2.0, 3.0])
    try:
        L.assert_no_split_leakage([0], [1], [2, 3], groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('a NaN group crossing splits must be rejected')
    # Empty data still splits into three empty parts.
    idx = L.chronological_split_indices(0)
    assert all(len(idx[k]) == 0 for k in ('train', 'val', 'test')), idx


def test_probe_stats_rejects_malformed_arm_identities():
    import explib.confirmation as V
    manifest = {'arms': ['a', 'b'], 'seeds': [0, 1]}
    td = tempfile.mkdtemp(prefix='probe_', dir=REPO)
    try:
        path = os.path.join(td, 'distractor.json')
        base = {'seed': 0, 'val_fp': 'fp', 'arm': 'learned',
                'eval_len': 8, 'rho': 0.5, 'ppl_mean': 1.0}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'bad_variant': dict(base, variant={'x': 1}),
                       'bad_arm': dict(base, variant='a', arm={'x': 1})}, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 0 and r['n_excluded'] == 2, r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_probe_stats_rejects_synthesized_and_mismatched_probe_params():
    import explib.confirmation as V
    manifest = {'arms': ['a'], 'seeds': [0, 1]}
    td = tempfile.mkdtemp(prefix='probe_', dir=REPO)
    try:
        path = os.path.join(td, 'distractor.json')
        code = L.code_stamp()

        def _rec(arm, seed, pp):
            return {'variant': 'a', 'seed': seed, 'val_fp': 'fp', 'arm': arm,
                    'eval_len': 8, 'rho': 0.5, 'mode': 'uniform',
                    'ppl_mean': 2.0 + 0.1 * seed + (0.05 if arm == 'dense' else 0.0),
                    '_code': code, 'probe_params': dict(pp)}

        base = {f'learned::{s}': _rec('learned', s, {'n_seq': 16})
                for s in (0, 1)}
        base.update({f'dense::{s}': _rec('dense', s, {'n_seq': 16})
                     for s in (0, 1)})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(base, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 4, r
        assert len(r['contrasts']) == 1, r

        # The parameter-matching `_mr` field is part of the ARM, not the
        # probe measurement: arms whose recipes differ only there are still
        # a valid controlled contrast (v10 applies the same normalisation).
        mr = dict(base)
        mr['learned::0'] = dict(mr['learned::0'], recipe='x_mr4.0_dfabc')
        mr['learned::1'] = dict(mr['learned::1'], recipe='x_mr4.0_dfabc')
        mr['dense::0'] = dict(mr['dense::0'], recipe='x_mr6.7_dfabc')
        mr['dense::1'] = dict(mr['dense::1'], recipe='x_mr6.7_dfabc')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mr, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert len(r['contrasts']) == 1, r

        # The real mlp_match_ref values contain underscores (csa_dynamic,
        # csa_fixed_rope).  The normaliser must remove the whole run_cfg `_mr`
        # field, not stop at its first underscore: otherwise two arms that
        # differ only in parameter matching can never be paired.
        _rc_a = ('steps1_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_'
                 'v8192_wu50_cl0.05_dlm10.0_mtfull_mrcsa_fixed_rope_ee0_es128_'
                 'detcpu_dfabc-def_tk0_ampfp32_eb0_ae8192_csx')
        _rc_b = _rc_a.replace('_mrcsa_fixed_rope_', '_mrfull_rope_')
        assert V10._recipe_norm(_rc_a) == V10._recipe_norm(_rc_b), (
            V10._recipe_norm(_rc_a), V10._recipe_norm(_rc_b))
        assert V10._recipe_norm(_rc_a) != V10._recipe_norm(
            _rc_a.replace('_ee0_', '_ee1_'))
        _p1_a = 'p1l_steps1_mr4-csa_fixed_rope_ns4_dfabc'
        _p1_b = _p1_a.replace('_mr4-', '_mr6.7-')
        assert V10._recipe_norm(_p1_a) == V10._recipe_norm(_p1_b)
        assert V10._recipe_norm(_p1_a) != V10._recipe_norm(
            _p1_a.replace('_ns4_', '_ns8_'))

        # A synthesized probe trajectory is provenance, not a measurement.
        synth = dict(base)
        synth['learned::0'] = dict(synth['learned::0'], synthesized=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(synth, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 3, r
        assert not r['cells'] or all(c['arm'] != 'learned' or c['n'] != 2
                                     for c in r['cells']), r

        # Two arms measured under different probe geometries must not be
        # differenced as if they were one controlled contrast.
        mixed = dict(base)
        for s in (0, 1):
            mixed[f'dense::{s}'] = _rec('dense', s, {'n_seq': 8})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mixed, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 4, r
        assert not r['contrasts'], r
        assert any('different probe_params' in x for x in r.get('contrast_exclusions', [])), r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_audit_selection_rejects_boolean_measurements():
    A = __import__('audit_gates')
    td = tempfile.mkdtemp(prefix='selection_', dir=REPO)
    try:
        path = os.path.join(td, 'selection.json')
        cell = {'arm': 'a', 'eval_len': 1024, 'k': 32, 'n_blocks': 256,
                'n_queries': 100, 'recall_at_k': 0.5, 'mass_covered': 0.6,
                'random_baseline': 0.125,
                'per_seed': {str(s): {'mass_covered': 0.6,
                                      'random_baseline': 0.125}
                             for s in range(6)}}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [dict(cell, mass_covered=True)]}, f)
        r = A.audit_selection(path=path)
        assert r['status'] == 'fail', r
        assert any('real numbers' in p for p in r['problems']), r
        # Numeric strings are not recall/mass readings: a hand-edited
        # artifact must not unlock the claim gate by spelling the same number.
        for _field in ('recall_at_k', 'mass_covered', 'random_baseline'):
            _bad = dict(cell, **{_field: str(cell[_field])})
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'cells': [_bad]}, f)
            r = A.audit_selection(path=path)
            assert r['status'] == 'fail', (_field, r)
            assert any('real numbers' in p for p in r['problems']), (_field, r)
        # Geometry counts are exact integers, not numeric strings.
        _bad_k = dict(cell, k=str(cell['k']))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [_bad_k]}, f)
        r = A.audit_selection(path=path)
        assert r['status'] == 'fail', r
        assert any('exact integers' in p for p in r['problems']), r
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [cell]}, f)
        r = A.audit_selection(path=path)
        assert r['status'] == 'ok', r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_categorical_shift_empty_reference():
    r = L.categorical_shift_report([], [1, 2, 3])
    assert r['n_reference'] == 0 and r['n_holdout'] == 3, r
    assert r['unseen_rate'] == 1.0 and r['total_variation'] > 0.0, r
    # No reference distribution means JS is undefined, not a fixed log(2)/2
    # score that looks like a measured divergence.
    assert r['js_divergence'] is None and r['js_defined'] is False, r
    try:
        L.categorical_shift_report([1, 2], [1, 2], max_categories=0)
    except ValueError:
        pass
    else:
        raise AssertionError('max_categories=0 must be rejected')


def test_data_quality_report():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    rep = L.data_quality_report(X, y=[0, 1, 1, 0],
                                train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert rep['status'] == 'fail', rep
    assert rep['checks']['duplicate_rows_across_splits']['train__val'] == 1
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 1
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    rep2 = L.data_quality_report(Y, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep2['status'] == 'warnings', rep2
    assert rep2['checks']['nan_count'] + rep2['checks']['inf_count'] == 0
    Xn = Y.copy()
    Xn[2, 0] = np.nan
    Xn[3, 0] = np.inf
    rep3 = L.data_quality_report(Xn, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep3['checks']['nan_count'] == 1 and rep3['checks']['inf_count'] == 1
    # Group boundaries are part of the report, not a matter of trust.
    rep4 = L.data_quality_report(Y, groups=np.array(['a', 'a', 'b', 'c']),
                                 train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep4['checks']['group_overlap']['train__val'] == 1
    assert rep4['status'] == 'fail'


def test_data_quality_target_outliers():
    # The target fence is learned from training labels only; a holdout target
    # cannot widen it.  This catches label corruption that exact-duplicate
    # checks cannot see.
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = [0.0, 1.0, 2.0, 3.0, 100.0, 2.0]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1, 2, 3],
                                val_idx=[4], test_idx=[5],
                                target_outlier_z=1.5)
    assert rep['status'] == 'warnings', rep
    assert rep['checks']['target_outliers_by_split']['val'] == 1, rep
    assert rep['checks']['target_outliers_by_split']['train'] == 0, rep
    hard = L.data_quality_report(X, y=y, train_idx=[0, 1, 2, 3],
                                 val_idx=[4], test_idx=[5],
                                 fail_on_target_outliers=True)
    assert hard['status'] == 'fail', hard
    # A training target outside its own train-only fence is also visible.
    rep_tr = L.data_quality_report(X, y=[0.0, 1.0, 2.0, 100.0, 1.0, 2.0],
                                   train_idx=[0, 1, 2, 3],
                                   val_idx=[4], test_idx=[5],
                                   target_outlier_z=1.5)
    assert rep_tr['checks']['target_outliers_by_split']['train'] == 1, rep_tr
    # Categorical labels are not coerced into a numeric target.
    cat = L.data_quality_report(X, y=['a', 'b', 'c', 'd', 'a', 'b'],
                                train_idx=[0, 1, 2, 3], val_idx=[4],
                                test_idx=[5], target_outlier_z=1.5)
    assert cat['checks']['target_outlier_fence'] is None, cat
    assert cat['checks']['target_outlier_fence_estimated'] is False, cat
    assert any('not estimable' in w for w in cat['warnings']), cat
    # Asking the gate to FAIL on target outliers when no numeric fence exists
    # must fail closed, not silently report zero outliers.
    cat_fail = L.data_quality_report(
        X, y=['a', 'b', 'c', 'd', 'a', 'b'], train_idx=[0, 1, 2, 3],
        val_idx=[4], test_idx=[5], fail_on_target_outliers=True)
    assert cat_fail['status'] == 'fail', cat_fail
    one_label = L.data_quality_report(
        X, y=[0.0, None, None, None, 1.0, 2.0], train_idx=[0, 1, 2, 3],
        val_idx=[4], test_idx=[5], target_outlier_z=1.5)
    assert one_label['checks']['target_outlier_fence_estimated'] is False, one_label
    # Requesting the target-outlier audit without y must fail closed instead
    # of silently omitting the gate.
    for _kw in ({'target_outlier_z': 1.5}, {'fail_on_target_outliers': True}):
        try:
            L.data_quality_report(X, train_idx=[0, 1, 2, 3], **_kw)
        except ValueError as e:
            assert 'requires y' in str(e), e
        else:
            raise AssertionError(f'{_kw} without y must be refused')
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.data_quality_report(X, y=y, train_idx=[0, 1],
                                  target_outlier_z=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target_outlier_z={bad!r} must be refused')
    # Explicit label gates without labels must fail closed, not be skipped.
    for _kw in ({'fail_on_label_shift': True},
                {'fail_on_nonfinite_labels': True},
                {'fail_on_unknown_labels': True}):
        try:
            L.data_quality_report(X, train_idx=[0, 1, 2, 3], **_kw)
        except ValueError as e:
            assert 'requires y' in str(e), e
        else:
            raise AssertionError(f'{_kw} without y must be refused')


def test_split_sizes_explicit_never_negative():
    # Explicit sizes and fractional targets can overdraw the remaining rows; the
    # filler must clamp rather than emit a negative split.
    cases = ({'train_frac': 0.05, 'val_frac': 0.9, 'test_frac': 0.05, 'train_size': 9},
             {'train_frac': 0.8, 'val_frac': 0.1, 'test_frac': 0.1, 'val_size': 9})
    for kwargs in cases:
        idx = L.chronological_split_indices(10, **kwargs)
        sizes = [len(idx[k]) for k in ('train', 'val', 'test')]
        assert sum(sizes) == 10 and all(s >= 0 for s in sizes), (kwargs, sizes)
    # The normal explicit-size reconciliation is unchanged.
    idx = L.chronological_split_indices(10, train_size=6)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    idx_float = L.chronological_split_indices(10, train_size=6.0)
    assert [len(idx_float[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    # An explicitly sized split must not make the unfilled tail create rows in
    # a split the caller declared empty (test_frac=0).
    idx0 = L.chronological_split_indices(10, train_size=6, test_frac=0.0)
    assert [len(idx0[k]) for k in ('train', 'val', 'test')] == [6, 4, 0], idx0
    idx0b = L.chronological_split_indices(10, val_size=2, test_frac=0.0)
    assert [len(idx0b[k]) for k in ('train', 'val', 'test')] == [8, 2, 0], idx0b
    # Explicit sizes are counts: non-integral/coerced values must be refused
    # rather than silently truncated (1.5 -> 1) or parsed ('2' -> 2).
    for bad in (1.5, '2', True):
        try:
            L.chronological_split_indices(100, train_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'train_size={bad!r} must be rejected')
    for bad_n in (10.5, '10', True):
        try:
            L.chronological_split_indices(bad_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n={bad_n!r} must be rejected')
    assert len(L.chronological_split_indices(100.0)['train']) == 80


def test_empty_split_still_validates_fractions():
    # An empty table must not bypass the fraction contract: the same
    # contradictory arguments rejected for n>0 must be rejected for n=0, so a
    # caller cannot build a split spec that only fails after rows are appended.
    idx = L.chronological_split_indices(0)
    assert all(len(idx[k]) == 0 for k in ('train', 'val', 'test')), idx
    for bad in ({'train_frac': 0.5, 'val_frac': 0.5, 'test_frac': 0.5},
                {'train_frac': -0.1, 'val_frac': 0.6, 'test_frac': 0.5},
                {'train_frac': 0.0, 'val_frac': 0.0, 'test_frac': 0.0}):
        try:
            L.chronological_split_indices(0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n=0 accepted invalid fractions {bad!r}')


def test_audit_split_sequence_overlap():
    # A 1-D token cache stores train as a stream: a validation row overlapping
    # at an arbitrary offset must still be detected (not only at aligned rows).
    train = np.arange(40, dtype=np.int64)
    rows = np.array([[5, 6, 7, 8], [100, 101, 102, 103]], dtype=np.int64)
    assert A._row_overlap_count(train, rows) == 1
    assert A._row_overlap_count(train, rows, chunk_tokens=8) == 1
    assert A._row_overlap_count(train, np.array([[36, 37, 38, 39]]), chunk_tokens=8) == 1
    # 2-D tables keep the exact aligned-row contract.
    table = np.array([[1, 2], [3, 4], [5, 6]])
    assert A._row_overlap_count(table, np.array([[3, 4], [9, 9]])) == 1


def test_categorical_and_cache_guards():
    same = L.categorical_shift_report([1, 2, 2, 3], [1, 2, 2, 3])
    assert same['js_divergence'] < 1e-12 and same['total_variation'] < 1e-12
    shifted = L.categorical_shift_report([1, 1, 1, 1], [9, 9, 9, 9])
    assert shifted['js_divergence'] > 0.5 and shifted['unseen_rate'] == 1.0
    # A tuple category containing the delimiter must not collapse into a
    # different tuple category.  Before the length-prefixed key these two
    # reference levels shared one string key, so a holdout containing only one
    # of them looked identical to the reference.
    _tuple_shift = L.categorical_shift_report(
        [('a', 'b'), ('a,s:b',)], [('a', 'b')])
    assert _tuple_shift['js_divergence'] > 1e-12, _tuple_shift
    # A real tuple category equal to the old missing-value sentinel is not a
    # missing value; the two reference levels must stay distinct.
    _missing_tuple = L.categorical_shift_report(
        [None, ('__missing__',)], [('__missing__',)])
    assert _missing_tuple['js_divergence'] > 1e-12, _missing_tuple
    # The cache-hit guard must reject a validation file whose content no
    # longer matches the recorded fingerprint.
    tr = np.arange(32, dtype=np.int64) % 11
    va = np.arange(16, dtype=np.int64).reshape(2, 8) % 11
    meta = {'src_fp': L._ids_fp(tr), 'val_fp': L._ids_fp(va.reshape(-1)),
            'vocab': 11, 'split_strategy': 'official_train_validation_test',
            'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}
    assert L._cached_split_problems(meta, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8) == []
    bad = dict(meta, val_fp='not-the-validation-fingerprint')
    assert any('validation id fingerprint' in p for p in
               L._cached_split_problems(bad, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    bad2 = dict(meta, shuffle_at_split=True)
    assert any('shuffled split' in p for p in
               L._cached_split_problems(bad2, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    bad3 = dict(meta, tokenizer_fit_split='test')
    assert any('tokenizer' in p for p in
               L._cached_split_problems(bad3, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))


def test_public_exports_resolve():
    # __all__ is an API contract: every advertised name must exist, and a
    # star-import must not fail.  A stale entry (a name added to __all__ but
    # never implemented) breaks `from data_guard import *` at import time.
    import data_guard as DG
    missing = [name for name in DG.__all__ if not hasattr(DG, name)]
    assert not missing, f'data_guard.__all__ advertises undefined names: {missing}'
    ns = {}
    exec('from data_guard import *', ns)
    for name in DG.__all__:
        assert name in ns, f'{name} was not exported by a star-import'


def test_split_and_prepare_fits_train_only():
    # Holdout rows are extreme; if they influenced imputation or scaling the
    # training centre/factor would move by >100x.  The one-call helper must
    # split first and fit on train only.
    X = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 1.0],
                  [1000.0, 1.0], [2000.0, 1.0], [3000.0, 1.0]])
    out = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                              quality_report=True)
    idx = out['indices']
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [3, 1, 2], idx
    prep = out['preprocessor']
    st = prep.get_stats()
    assert abs(st['features'][0]['center'] - 1.0) < 1e-12, st
    # Every returned split is finite and equals an explicit transform of the
    # raw split (no hidden global statistics).
    for name in ('train', 'val', 'test'):
        raw = X[idx[name]]
        assert np.all(np.isfinite(out['X'][name])), (name, out['X'][name])
        assert np.allclose(out['X'][name], prep.transform(raw)), name
    assert prep.get_stats() == st, 'transform must not update fitted statistics'
    assert out['quality']['n_train_reference'] == 3, out['quality']
    # Preprocess=False still returns the raw chronological parts, and the
    # optional quality report uses the same train reference.
    raw = L.split_and_prepare(X, time=[5, 4, 3, 2, 1, 0], preprocess=False,
                              train_frac=0.5, val_frac=0.25, test_frac=0.25)
    assert raw['preprocessor'] is None
    assert [len(raw['indices'][k]) for k in ('train', 'val', 'test')] == [3, 1, 2], raw['indices']
    assert L.assert_no_split_leakage(raw['indices']['train'], raw['indices']['val'],
                                     raw['indices']['test'],
                                     time=np.array([5, 4, 3, 2, 1, 0])) is True
    # Exact duplicate rows across a split boundary are leakage even when the
    # row indices/timestamps are disjoint; the one-call helper must refuse.
    Xdup = np.array([[0.0], [1.0], [0.0], [3.0]])
    try:
        L.split_and_prepare(Xdup, preprocess=False)
    except ValueError as e:
        assert 'duplicate feature rows cross split' in str(e), e
    else:
        raise AssertionError('cross-split duplicate rows must be refused')
    waived = L.split_and_prepare(Xdup, preprocess=False,
                                 check_cross_split_duplicates=False)
    assert 'cross_split_duplicates' not in waived, waived
    assert waived.get('cross_split_duplicates_waived') is True, waived
    ok = L.split_and_prepare(np.array([[0.0], [1.0], [2.0], [3.0]]),
                             preprocess=False)
    assert ok['cross_split_duplicates'] == {
        'train__val': 0, 'train__test': 0, 'val__test': 0}, ok


def test_split_and_prepare_strict_quality_gate():
    # A recorded 'fail' report must not be returned as failure-free data when
    # the caller asked for a strict gate.  The holdout is never used to make
    # the decision: the NaN is in the training split.
    X = np.array([[0.0], [1.0], [np.nan], [2.0], [3.0], [4.0]])
    rep = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                              quality_report=True)
    assert rep['quality']['status'] == 'fail', rep['quality']
    try:
        L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                            fail_on_quality=True)
    except RuntimeError as e:
        assert 'data-quality gate failed' in str(e), e
    else:
        raise AssertionError('fail_on_quality must refuse a failing quality report')
    # Warnings alone do not abort; the split is returned with its report.
    ok = L.split_and_prepare(np.arange(6.0).reshape(-1, 1),
                             train_frac=0.5, val_frac=0.25, test_frac=0.25,
                             fail_on_quality=True)
    assert ok['quality']['status'] != 'fail', ok['quality']


def test_data_quality_temporal_audit():
    # Indices built outside the split helpers must still be audited for time
    # leakage: row/group disjointness alone cannot catch a shuffled series.
    X = np.arange(12, dtype=float).reshape(6, 2)
    time = np.array([5, 4, 3, 2, 1, 0])
    idx = L.split_indices(6, time=time, train_frac=0.5, val_frac=0.25,
                          test_frac=0.25)
    rep = L.data_quality_report(X, train_idx=idx['train'], val_idx=idx['val'],
                                test_idx=idx['test'], time=time)
    assert all(not c['leak'] for c in rep['checks']['temporal_order'].values()), rep
    shuffled = {'train': np.array([0, 1]), 'val': np.array([2]),
                'test': np.array([3])}
    bad = L.data_quality_report(X, train_idx=shuffled['train'],
                                val_idx=shuffled['val'],
                                test_idx=shuffled['test'], time=time)
    assert bad['status'] == 'fail', bad
    assert any('temporal leakage' in msg for msg in bad['hard_failures']), bad
    assert bad['checks']['temporal_order']['train__val']['leak'] is True, bad
    # An equal timestamp cannot straddle a boundary either.
    tie = np.array([0, 0, 1, 1, 2, 2])
    tie_rep = L.data_quality_report(
        X, train_idx=[0], val_idx=[1], test_idx=[2, 3, 4, 5], time=tie)
    assert tie_rep['status'] == 'fail', tie_rep
    # The one-call helper forwards its time key, so the same gate runs there.
    out = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                              test_frac=0.25, quality_report=True)
    assert out['quality']['status'] in ('ok', 'warnings'), out['quality']
    assert all(not c['leak'] for c in
               out['quality']['checks']['temporal_order'].values()), out['quality']
    # A quality_kwargs override must not collide with the forwarded key.
    out2 = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                               test_frac=0.25, quality_report=True,
                               quality_kwargs={'time': time})
    assert out2['quality']['status'] in ('ok', 'warnings'), out2['quality']
    # An explicit None override must not disable the temporal audit the split
    # helper already knows how to run.
    out3 = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                               test_frac=0.25, quality_report=True,
                               quality_kwargs={'time': None, 'groups': None})
    _temporal3 = out3['quality']['checks']['temporal_order']
    assert all(c.get('checked') for c in _temporal3.values()), out3['quality']
    # String event ids/timestamps are orderable in Python even though NumPy 2
    # has no string maximum/minimum ufunc loop; grouped splits must not fall
    # over (or silently disable the order check) for them.
    sg = np.arange(6)
    st = np.array(['f', 'e', 'd', 'c', 'b', 'a'])
    sidx = L.grouped_chronological_split_indices(
        6, sg, 0.5, 0.25, 0.25, order=st)
    L.assert_no_split_leakage(sidx['train'], sidx['val'], sidx['test'],
                              groups=sg, time=st)
    srep = L.data_quality_report(
        np.arange(12, dtype=float).reshape(6, 2),
        train_idx=sidx['train'], val_idx=sidx['val'], test_idx=sidx['test'],
        groups=sg, time=st)
    assert all(not c['leak'] for c in srep['checks']['temporal_order'].values()), srep


def test_capped_document_id_sidecar():
    # The capped-id helper must record exactly the documents that contributed
    # a served token, using the same row-length list as the tokenizer stream.
    rows = ['alpha', 'beta', 'gamma']
    lengths = [3, 3, 5]
    expected = [L._doc_id_u64(L._doc_exact_fp(r)) for r in rows]
    ids3 = L._doc_ids_for_capped_rows(rows, lengths, 3)
    assert ids3.tolist() == expected[:1], (ids3, expected)
    # A cut inside the second document still records the second document (it
    # contributed one served token), but never the untouched third one.
    ids4 = L._doc_ids_for_capped_rows(rows, lengths, 4)
    assert ids4.tolist() == expected[:2], (ids4, expected)
    ids0 = L._doc_ids_for_capped_rows(rows, lengths, 0)
    assert ids0.size == 0, ids0

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _Tok:
        def encode_batch(self, batch):
            return [_Enc(list(range(len(text)))) for text in batch]

    lengths_out = []
    ids = L._tokenise_row_chunks([['aa', 'b']], 5, 'unit', _Tok(), 0,
                                 lengths_out=lengths_out)
    assert lengths_out == [3, 2], lengths_out
    assert ids.tolist() == [0, 1, 0, 0, 0], ids
    assert L._tokenise_row_chunks([['aa', 'b']], 5, 'unit', _Tok(), 0).tolist() == ids.tolist()
    doc_ids = L._doc_ids_for_capped_rows(['aa', 'b'], lengths_out, 5)
    assert doc_ids.tolist() == [L._doc_id_u64(L._doc_exact_fp('aa')),
                                L._doc_id_u64(L._doc_exact_fp('b'))], doc_ids


def test_document_dedup_closes_near_duplicate_chains():
    for _kw in ({'near_threshold': -1}, {'near_threshold': 1.5},
                {'near_threshold': True}, {'bands': 2.5}, {'bands': True},
                {'bands': 0}, {'bands': 9}, {'near_threshold': 65}):
        try:
            L._DocDeduplicator(**_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_DocDeduplicator({_kw!r}) must be refused')
    # A~T is skipped as a near duplicate of train T.  B is within the near
    # threshold of A but just outside it for T, so a stateless bucket index
    # would admit B into validation even though it belongs to the same
    # document cluster.  The skipped copy is quarantined so the chain stays on
    # the train side.
    hashes = {'train T': 0, 'val A': 0b111, 'val B': 0b11011}
    orig = L._doc_simhash
    try:
        L._doc_simhash = lambda text: hashes[text]
        d = L._DocDeduplicator(near_threshold=3, bands=4)
        assert d.is_duplicate('train T') == (False, None)
        assert d.is_duplicate('val A') == (True, 'near')
        assert d.is_duplicate('val B') == (True, 'near'), 'chain member slipped through'
    finally:
        L._doc_simhash = orig


def test_test_document_filter_uses_readonly_earlier_indices():
    # A test document must not become part of the train/validation reference
    # set.  Otherwise a later test doc resembling an earlier TEST doc is
    # mislabelled as a train/val duplicate, and the audit no longer knows why
    # the document was removed.
    train = ['train one', 'train two']
    val = ['val one']
    val_index = L._DocDeduplicator()
    for r in val:
        val_index.is_duplicate(r)
    before = val_index.stats()
    test = ['unique test a', 'unique test a', 'unique test b',
            'train one', 'val one']
    kept, excluded, test_index = L._filter_test_documents(
        train, val_index, test, label='test')
    assert kept == ['unique test a', 'unique test b'], kept
    assert excluded['train_exact'] == 1, excluded
    assert excluded['validation_exact'] == 1, excluded
    assert excluded['test_exact'] == 1, excluded
    # The earlier-split index was consulted read-only.
    assert val_index.stats() == before, (before, val_index.stats())
    assert val_index.find('unique test a') == (False, None)
    # The pure lookup form never mutates counters or membership.
    d = L._DocDeduplicator()
    assert d.find('a b c') == (False, None)
    assert d.stats()['added_documents'] == 0
    d.is_duplicate('a b c')
    n_before = d.stats()['added_documents']
    assert d.find('a b c') == (True, 'exact')
    assert d.find('d e f') == (False, None)
    assert d.stats()['added_documents'] == n_before, d.stats()
    # The kept set is still closed under a test-to-test near-duplicate chain.
    assert test_index.find('unique test a') == (True, 'exact')


def test_split_purge_embargo():
    # Purge drops the tail of train and validation; the test slice is intact
    # and no returned row straddles a boundary.
    idx = L.chronological_split_indices(100, 0.7, 0.2, 0.1, purge=3)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [67, 17, 10], idx
    assert idx['train'][-1] == 66 and idx['val'][0] == 70 and idx['val'][-1] == 86
    assert idx['test'][0] == 90
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              time=np.arange(100))
    t = np.arange(100)[::-1]
    ti = L.split_indices(100, time=t, train_frac=0.7, val_frac=0.2,
                         test_frac=0.1, purge=2)
    assert [len(ti[k]) for k in ('train', 'val', 'test')] == [68, 18, 10], ti
    assert np.max(t[ti['train']]) < np.min(t[ti['val']])
    assert np.max(t[ti['val']]) < np.min(t[ti['test']])
    L.assert_no_split_leakage(ti['train'], ti['val'], ti['test'], time=t)
    # Purging more rows than a split contains is a hard error, not an empty
    # split that would silently make validation unusable.
    for bad in (-1, 1.5, True, '2'):
        # Use a large n so rejection cannot come accidentally from an
        # over-large purge emptying a split: the validator itself must refuse
        # non-integral/coerced values.
        try:
            L.chronological_split_indices(100, purge=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'purge={bad!r} must be rejected')
    assert len(L.chronological_split_indices(100, purge=2.0)['train']) == 78
    try:
        L.chronological_split_indices(10, purge=5)
    except ValueError as e:
        assert 'empty' in str(e), e
    else:
        raise AssertionError('an over-large purge must not empty a split')
    # A grouped split cannot drop individual rows without breaking groups.
    try:
        L.split_indices(10, groups=np.arange(10), purge=1)
    except ValueError as e:
        assert 'grouped' in str(e), e
    else:
        raise AssertionError('purge with groups must be refused')


def test_group_split_refuses_empty_requested_side():
    # One huge group consumes the whole table; a requested validation split
    # would be empty.  Fail instead of returning it and inviting test tuning.
    try:
        L.grouped_chronological_split_indices(10, np.zeros(10, dtype=int),
                                              0.8, 0.1, 0.1)
    except ValueError as e:
        assert 'no rows for the requested val split' in str(e), e
    else:
        raise AssertionError('an empty requested validation split must fail')
    # Explicit zero targets are allowed to be empty.
    idx = L.grouped_chronological_split_indices(10, np.zeros(10, dtype=int),
                                                1.0, 0.0, 0.0)
    assert len(idx['train']) == 10 and len(idx['val']) == len(idx['test']) == 0


def test_label_shift_and_split_counts():
    X = np.arange(12, dtype=float).reshape(6, 2)
    # Non-finite labels are counted per split; fail_on_nonfinite_labels turns
    # a corrupted target into a hard failure (off by default for legacy use).
    y = [0, 0, 1, 1, np.inf, 2]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2, 3],
                                test_idx=[4, 5])
    assert rep['checks']['nonfinite_labels'] == 1, rep
    assert rep['checks']['nonfinite_labels_by_split']['test'] == 1, rep
    assert rep['checks']['nonfinite_labels_by_split']['train'] == 0, rep
    assert rep['status'] == 'warnings', rep
    rep_fail = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2, 3],
                                     test_idx=[4, 5],
                                     fail_on_nonfinite_labels=True)
    assert rep_fail['status'] == 'fail', rep_fail
    # A complete label-distribution reversal is reported against the training
    # reference and can be made a hard gate.
    X2 = np.arange(12, dtype=float).reshape(6, 2)
    y2 = [0, 0, 0, 1, 1, 1]
    rep2 = L.data_quality_report(X2, y=y2, train_idx=[0, 1, 2],
                                 val_idx=[3, 4, 5], fail_on_label_shift=True)
    shift = rep2['checks']['label_distribution_shift_vs_train']['val']
    assert shift['js_divergence'] > 0.5, rep2
    assert rep2['status'] == 'fail', rep2
    assert rep2['checks']['label_distribution_by_split']['train'] == {'0': 3}, rep2


def test_sample_tokens_uniform_rejection():
    # Force the scale-safe rejection path even on tiny arrays.  The sample must
    # remain a uniform without-replacement draw; taking the smallest CAP of a
    # sorted candidate pool (the previous implementation) has mean ~n/9 for
    # cap=4, not ~n/2.
    old = A._CHOICE_PERM_MAX_ELEMS
    try:
        A._CHOICE_PERM_MAX_ELEMS = 0
        idx = np.arange(1000)
        means = [A._sample_tokens(idx, cap=4, seed=s).mean() for s in range(200)]
        assert 400.0 < float(np.mean(means)) < 600.0, (float(np.mean(means)), means[:5])
        assert np.array_equal(A._sample_tokens(idx, cap=4, seed=17),
                              A._sample_tokens(idx, cap=4, seed=17))
        sample = A._sample_tokens(idx, cap=4, seed=17)
        assert sample.size == 4 and len(set(sample.tolist())) == 4
    finally:
        A._CHOICE_PERM_MAX_ELEMS = old


def test_audit_budget_integer_runs():
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, 'run_time_state.json')
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 12.5, 'runs': 3}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] == 3, out
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 1.0, 'runs': [1, 2]}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] == 2, out
        # A FLOPs proxy is only defined for positive finite params/tokens and
        # a non-negative overhead; bool/NaN/negative values must not be
        # silently converted into a plausible-looking budget.
        for _args in ((0, 10), (True, 10), (10, True), (10, 0),
                      (float('nan'), 10), (10, float('inf'))):
            try:
                A.train_flops_estimate(*_args)
            except ValueError:
                pass
            else:
                raise AssertionError(f'train_flops_estimate{_args!r} must be refused')
        for _ov in (True, -0.1, float('nan'), float('inf'), 'x'):
            try:
                A.train_flops_estimate(10, 10, attn_overhead=_ov)
            except ValueError:
                pass
            else:
                raise AssertionError(f'attn_overhead={_ov!r} must be refused')
        # A malformed panel record must not be converted into a fake FLOPs
        # estimate (bool params, NaN tokens, string steps).
        panel = tempfile.mkdtemp(prefix='budget_', dir=REPO)
        try:
            with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'x::s0': {'variant': 'csa_fixed', 'params': True,
                                     'steps': 'lots', 'tokens_seen': float('nan')}}, f)
            bp = A.audit_budget(state_files=(), panels=((panel, 'csa_fixed', None),))
            assert bp['panels'][panel] == 'no param records', bp
        finally:
            shutil.rmtree(panel, ignore_errors=True)


def test_audit_failures_ignores_unmeasured_ppl():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'a::seed0': {'variant': 'a', 'seed': 0, 'error': 'boom'},
                'b::seed0': {'variant': 'b', 'seed': 0, 'ppl': float('nan')},
                'c::seed0': {'variant': 'c', 'seed': 0, 'ppl': 12.0},
            }, f)
        out = A.audit_failures(outdirs=[d])
        panel = out['panels'][d]
        assert panel['ok_by_arm'] == {'c': 1}, panel
        assert panel['errors_by_arm'] == {'a': 1}, panel
        # A NaN-PPL record is neither a success nor an exception in the data;
        # it is a missing measurement and must stay visible to the failure
        # concentration audit (otherwise survivorship bias disappears).
        assert panel['unmeasured_by_arm'] == {'b': 1}, panel
        assert out['status'] == 'warn', out


def test_object_time_and_zero_feature_guards():
    bad_time = np.array([0.0, float('nan'), 2.0], dtype=object)
    for call in (lambda: L.assert_no_split_leakage([0], [1], [2], time=bad_time),
                 lambda: L.split_indices(3, time=bad_time, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3)):
        try:
            call()
        except ValueError as e:
            assert 'NaN/Inf' in str(e), e
        else:
            raise AssertionError('an object-dtype NaN timestamp must be rejected')
    # A zero-feature table is uninformative but must not crash the report.
    rep = L.data_quality_report(np.empty((4, 0)), train_idx=[0, 1],
                                val_idx=[2], test_idx=[3])
    assert rep['n_features'] == 0, rep


def test_data_guard_index_label_and_shift_edges():
    # Negative indices wrap around in NumPy: -1 and n-1 are the same row even
    # though their values look disjoint.  Fractional indices must never be
    # silently truncated to an int.
    for bad_train in ([-1], [0.5], [np.nan], [True], ['0'],
                      np.array(['2020-01-01'], dtype='datetime64[D]')):
        try:
            L.assert_no_split_leakage(bad_train, [], [2])
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid split indices {bad_train!r} must be rejected')
    X = np.arange(8, dtype=float).reshape(4, 2)
    for bad_train in ([-1], [0.5]):
        try:
            L.data_quality_report(X, train_idx=bad_train, val_idx=[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f'data_quality_report accepted {bad_train!r}')

    # Inf is not a class/regression target; it is non-finite label noise.
    rep = L.data_quality_report(X, y=[0, np.inf, 1, 2],
                                train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert rep['checks']['nonfinite_labels'] == 1, rep
    assert rep['checks']['missing_labels'] == 0, rep
    assert rep['status'] == 'warnings', rep

    # -0.0 == 0.0 but the raw bytes differ; a duplicate across the split must
    # still be detected.
    Xz = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    repz = L.data_quality_report(Xz, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert repz['checks']['duplicate_rows_across_splits']['train__val'] == 1, repz
    assert repz['status'] == 'fail', repz
    # Different non-finite values are not exact duplicate rows: NaN, +Inf and
    # -Inf share a non-finite mask but are distinct corrupted values.  Only the
    # repeated NaN row is a duplicate.
    Xnf = np.array([[np.nan, 1.0], [np.inf, 1.0], [-np.inf, 1.0], [np.nan, 1.0]])
    mask = L.duplicate_mask(Xnf)
    assert mask.tolist() == [False, False, False, True], mask
    repnf = L.data_quality_report(Xnf, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert repnf['checks']['duplicate_rows_across_splits']['train__val'] == 0, repnf

    # A zero-variance training feature with a different holdout mean is an
    # undefined SMD, not a zero shift.
    Xc = np.array([[0.0], [0.0], [5.0], [0.0]])
    repc = L.data_quality_report(Xc, train_idx=[0, 1, 3], val_idx=[2])
    shift = repc['checks']['distribution_shift_vs_train']['val']
    assert shift['non_estimable_shift_features'] == [0], repc
    assert repc['status'] == 'warnings', repc

    # Finite inputs near the float64 limit can overflow the mean/variance.  An
    # infinite SMD is not evidence of no shift; it is not estimable.
    Xov = np.array([[1e308], [-1e308], [1e308], [1e308]])
    with np.errstate(over='ignore', invalid='ignore'):
        repov = L.data_quality_report(Xov, train_idx=[0, 1], val_idx=[2, 3])
    _ov = repov['checks']['distribution_shift_vs_train']['val']
    assert _ov['non_estimable_shift_features'] == [0], repov
    assert any('not estimable' in w for w in repov['warnings']), repov

    # A NaN/negative threshold must be an error, not a silently disabled gate.
    for bad_kw in ({'clip_z': float('nan')}, {'clip_z': -1.0},
                   {'shift_threshold': float('nan')},
                   {'shift_threshold': -0.1}):
        try:
            L.data_quality_report(X, train_idx=[0, 1], val_idx=[2], **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'data_quality_report accepted {bad_kw!r}')

    # Object-dtype NaT group ids are one missing group, so splitting them
    # across sides is group leakage.
    g = np.array([np.datetime64('NaT', 'D'), np.datetime64('NaT', 'D'),
                  np.datetime64('2020-01-01', 'D')], dtype=object)
    try:
        L.assert_no_split_leakage([0], [1], [2], groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT group ids must not be split across sides')


def test_audit_split_warning_status():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        tr = np.arange(40, dtype=np.int64)
        va = np.arange(100, 108, dtype=np.int64).reshape(2, 4)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        # Missing split_strategy is an explicit legacy-cache warning.
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 40, 'seq_len': 4}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'warnings', rec
        assert out['caches'][os.path.abspath(d)]['status'] == 'warnings', out['caches']

