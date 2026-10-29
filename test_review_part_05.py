"""CPU review checks (split group 05; t_audit_split_preferred_seq_len .. t_scale_point_provenance_roundtrip)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_audit_split_preferred_seq_len():
    # A cache directory can hold several protocol variants.  Naming the train
    # sequence length must restrict the audit to that stream instead of
    # silently certifying/failing a probe cache the panel never used.
    with tempfile.TemporaryDirectory() as d:
        for tag, sl in (('s4', 4), ('s8', 8)):
            np.save(os.path.join(d, f'train_ids_{tag}.npy'),
                    np.arange(40, dtype=np.int64))
            np.save(os.path.join(d, f'val_batch_{tag}.npy'),
                    (np.arange(16, dtype=np.int64) + 40).reshape(2, 8))
            with open(os.path.join(d, f'meta_{tag}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump({'vocab': 80, 'seq_len': sl,
                           'split_strategy': 'ordered',
                           'shuffle_at_split': False,
                           'tokenizer_fit_split': 'train'}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                       preferred_train_seq_len=8)
        assert set(out['caches'][os.path.abspath(d)]['corpora']) == {'s8'}, out
        out4 = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                        preferred_train_seq_len=4)
        assert set(out4['caches'][os.path.abspath(d)]['corpora']) == {'s4'}, out4
        missing = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                           preferred_train_seq_len=512)
        assert missing['status'] == 'not_verifiable_locally', missing
        for bad in (0, -1, True, '8'):
            try:
                A.audit_split_discipline(cache_dirs=[d],
                                         preferred_train_seq_len=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'preferred_train_seq_len={bad!r} must be refused')


def test_audit_split_test_metadata_discipline():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        ttag = 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 40).reshape(2, 8)
        te = (np.arange(16, dtype=np.int64) + 56).reshape(2, 8)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 80, 'seq_len': 8, 'split_strategy': 'ordered',
                       'shuffle_at_split': False,
                       'tokenizer_fit_split': 'train'}, f)
        good_test = {'vocab': 80, 'seq_len': 8, 'split_strategy': 'official_test',
                     'shuffle_at_split': False,
                     'tokenizer_fit_split': 'train',
                     'near_duplicate_cleanup': {
                         'policy': 'test_documents_deduplicated_against_train_and_validation'}}
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(good_test, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_shuffle_at_split') is False, rec
        assert rec['status'] != 'fail', rec
        # P5T uses a longer test sequence than the training cache.  The test
        # cache must still be paired for the leakage audit.
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(good_test, seq_len=16), f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_pairing') != 'not_found', rec
        assert rec.get('test_tag') == ttag and rec.get('test_shape') == [2, 8], rec
        assert rec.get('exact_row_overlap_train_test') == 0, rec
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(good_test, f)
        bad_test = dict(good_test, shuffle_at_split=True,
                        tokenizer_fit_split='test',
                        near_duplicate_cleanup={'policy': 'none'})
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(bad_test, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert rec.get('test_shuffle_at_split') is True, rec
        assert any('test metadata declares a shuffled split' in w for w in out['warnings']), out
        assert any('test tokenizer was fitted on' in w for w in out['warnings']), out
        assert any('duplicate cleanup' in w for w in out['warnings']), out


def test_corpus_audit_overlap_is_unique_across_windows():
    # The same validation row repeated in two disjoint training windows is ONE
    # overlapping validation row, not two.
    tr = np.arange(200, dtype=np.int64) % 10
    tr[0:3] = [7, 8, 9]
    tr[13:16] = [7, 8, 9]
    va = np.array([[7, 8, 9]], dtype=np.int64)
    rep = L.audit_corpus_split(tr, va, 20, dataset='wikitext',
                               max_scan_tokens=64, sample_rows=1, strict=False)
    assert rep['exact_row_overlap_in_sample'] == 1, rep
    assert rep['sampled_val_rows'] == 1, rep


def test_corpus_audit_memo_and_thresholds():
    tr = np.arange(40, dtype=np.int64)
    va = (np.arange(8, dtype=np.int64) + 100).reshape(2, 4)
    # The memo key must include the split attestation: a clean read for one
    # meta must not answer for a different (failing) one.
    clean = L.audit_corpus_split(tr, va, 200, dataset='fineweb',
                                 max_scan_tokens=64, sample_rows=1, strict=False)
    assert clean['status'] == 'warnings', clean
    bad_meta = {'doc_ids': {'disjoint': False, 'overlap': 1,
                            'train_count': 1, 'validation_count': 1}}
    bad = L.audit_corpus_split(tr, va, 200, dataset='fineweb', meta=bad_meta,
                               max_scan_tokens=64, sample_rows=1, strict=False)
    assert bad['status'] == 'fail', bad
    assert any('document-disjoint' in p for p in bad['hard_failures']), bad
    # A non-strict read that cached a failing report must never satisfy a
    # later strict call.
    try:
        L.audit_corpus_split(tr, va, 200, dataset='fineweb', meta=bad_meta,
                             max_scan_tokens=64, sample_rows=1, strict=True)
    except RuntimeError as e:
        assert 'DATA GATE REFUSED' in str(e), e
    else:
        raise AssertionError('strict=True must refuse a cached failing split')
    # NaN/negative gates silently disable comparisons; refuse them.
    for kw in ({'shift_warn': float('nan')}, {'shift_fail': float('nan')},
               {'shift_warn': -0.1}, {'shift_fail': -1.0},
               {'shift_warn': 0.3, 'shift_fail': 0.2},
               {'sample_rows': 0}, {'max_scan_tokens': True}):
        try:
            L.audit_corpus_split(tr, va, 200, dataset='fineweb', strict=False, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_corpus_split accepted {kw!r}')


def test_audit_length_requires_eval_span():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bylen.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'by_len': {'512': {
                'n_tok': 511, 'eval_span': None, 'n_seq': 1}}}}, f)
        out = A.audit_length(path)
        assert out['status'] == 'fail', out
        assert out['by_length']['512']['same_scored_targets_across_records'] is False, out


def test_audit_fairness_nan_seed_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': float('nan'),
                                    'ppl': 10.0, 'run_cfg': 'x'}}, f)
        out = A.audit_fairness(path, arms=['a'])
        assert out['arms']['a']['seeds'] == [], out
        # Malformed run_cfg/params values are reported, not a TypeError: a
        # dict run_cfg is unhashable and mixed param types are unorderable.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 10.0,
                                    'run_cfg': {'bad': 1}, 'params': 10},
                       'a::seed1': {'variant': 'a', 'seed': 1, 'ppl': 11.0,
                                    'run_cfg': 'x', 'params': 'bad'}}, f)
        out_bad = A.audit_fairness(path, arms=['a'])
        assert isinstance(out_bad, dict) and out_bad['status'] == 'fail', out_bad
        assert out_bad['arms']['a']['params'] == [10, 'bad'], out_bad['arms']['a']
        # A summary that does not contain any requested arm is not auditable.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        out_empty = A.audit_fairness(path, arms=['a'])
        assert out_empty['status'] == 'not_verifiable_locally', out_empty
        # The metric audit must not report 'ok' when there is no panel to
        # check; policy declarations alone are not record-level verification.
        old_repo = A.REPO
        try:
            A.REPO = d
            os.makedirs(os.path.join(d, 'results_empty'), exist_ok=True)
            with open(os.path.join(d, 'results_empty', 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({}, f)
            m = A.audit_metrics()
            assert m['status'] != 'ok', m
        finally:
            A.REPO = old_repo


def test_audit_boundary_rejects_invalid_readings():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a': {'boundary_alignment': {'precision': 0.8,
                                                     'random_precision': 0.2}}}, f)
        ok = A.audit_boundary(outdirs=[d])
        assert ok['status'] == 'ok' and ok['panels'][d][0]['excess'] == 0.8 - 0.2, ok
        # Strings/booleans/non-finite readings cannot support a claim, and
        # must not crash the audit either.
        for bad in ({'precision': 'x', 'random_precision': 0.2},
                    {'precision': True, 'random_precision': 0.2},
                    {'precision': float('nan'), 'random_precision': 0.2},
                    {'precision': 1.4, 'random_precision': 0.2},
                    {'precision': 0.8, 'random_precision': -0.1}):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'a': {'boundary_alignment': bad}}, f)
            r = A.audit_boundary(outdirs=[d])
            assert r['status'] == 'fail' and r['problems'], r


def test_audit_malformed_arm_id_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'k': {'variant': {'bad': 1}, 'seed': 0, 'ppl': 10.0,
                             'stats': {}, 'efficiency': {'status': 'ok'}}}, f)
        for fn in (A.audit_attribution, A.audit_failures, A.audit_efficiency_pairs):
            out = fn([d])
            assert isinstance(out, dict) and out['status'] != 'fail', (fn.__name__, out)


def test_audit_split_empty_cache_is_not_ok():
    with tempfile.TemporaryDirectory() as d:
        out = A.audit_split_discipline(cache_dirs=[d])
        assert out['status'] == 'not_verifiable_locally', out
        assert out['caches'][os.path.abspath(d)]['status'] == 'not_verifiable_locally', out


def test_baseline_audit_malformed_seed_keys():
    art = {
        'arms': {'a': {'selected': {'lr': 1e-3}, 'selected_metric_value': 1.0,
                       'trials': [{'config': {'lr': 1e-3}, 'ppl': 1.0,
                                   'seed_ppls': {'not-a-seed': 1.0},
                                   'ppl_std': 'not-numeric'}]}},
        'protocol': {'selection_metric': 'ppl', 'selection_split': 'validation',
                     'direction': 'lower_is_better', 'trials': [{'lr': 1e-3}],
                     'arms': ['a'], 'search_seeds': [0], 'final_seeds': [0]},
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better',
        'final_records': {'a::seed0': {'seed': 0, 'variant': 'a',
                                       'hyperparams': {'lr': 'not-numeric'}}},
    }
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail', out
    assert any('seed_ppls keys' in p for p in out['problems']), out

    # A numeric seed set with a non-numeric ppl_std / selected hyperparameter
    # must be reported, not raise ValueError inside the audit.
    art['arms']['a']['trials'] = [{'config': {'lr': 1e-3}, 'ppl': 1.5,
                                   'seeds': [0, 1],
                                   'seed_ppls': {'0': 1.0, '1': 2.0},
                                   'ppl_std': 'not-numeric'}]
    out = L.baseline_tuning_audit(art)
    assert any('ppl_std is not numeric' in p for p in out['problems']), out
    assert any('is not numeric' in p for p in out['problems']), out

    # A declared final seed must be parsed exactly; silently dropping 1.5 (or
    # iterating a scalar) would let the audit certify a different seed panel.
    art['protocol']['final_seeds'] = [0, 1.5]
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final seed' in p for p in out['problems']), out
    art['protocol']['final_seeds'] = 3
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final_seeds' in p for p in out['problems']), out
    # final_records must be a mapping; a scalar must not reach `.items()` or
    # be silently treated as "no final records".
    art['protocol']['final_seeds'] = [0]
    art['final_records'] = 1.5
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final_records must be a mapping' in p for p in out['problems']), out
    # A non-scalar metric name must be reported, not used as a dict key.
    art['final_records'] = {}
    art['selection_metric'] = {'not': 'a metric'}
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('not a metric name' in p for p in out['problems']), out
    # Scalar protocol/arm trial fields and non-object selected configs are
    # malformed evidence: report them instead of raising TypeError.
    _base = json.loads(json.dumps(art))
    _base['selection_metric'] = 'ppl'
    _base['protocol']['selection_metric'] = 'ppl'
    _mutations = [
        lambda a: a['protocol'].__setitem__('trials', 1.5),
        lambda a: a['protocol'].__setitem__('arms', 1.5),
        lambda a: a['protocol'].__setitem__('inherits_tuning', ['not-a-map']),
        lambda a: a['arms']['a'].__setitem__('trials', True),
        lambda a: a['arms']['a'].__setitem__('selected', 1.5),
        lambda a: a['arms']['a']['trials'].__setitem__(
            0, {'config': ['not-a-map'], 'ppl': 1.0}),
    ]
    for _mut in _mutations:
        _bad = json.loads(json.dumps(_base))
        _mut(_bad)
        _out = L.baseline_tuning_audit(_bad)
        assert _out['status'] == 'fail', (_out,)


def test_simple_baseline_audit_malformed_records():
    art = {
        'baselines': {'a': {
            'selected': {'alpha': 0.1}, 'selected_metric_value': 1.0,
            'trials': [{'config': {'alpha': 0.1}, 'ppl': 1.0,
                        'n_tokens': 'not-an-int'},
                       {'config': {'alpha': 0.2}, 'ppl': 2.0,
                        'n_tokens': 10}]}},
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better', 'data': ['not', 'a', 'dict'],
        'protocol': {'selection_metric': 'ppl', 'selection_split': 'validation',
                     'direction': 'lower_is_better', 'arms': ['not-a-dict']},
    }
    out = L.simple_baseline_audit(art)
    assert out['status'] == 'fail', out
    assert any('n_tokens is not an integer' in p for p in out['problems']), out
    # Fractional token counts must not be truncated to a different budget.
    art2 = json.loads(json.dumps(art))
    for _t in art2['baselines']['a']['trials']:
        _t['n_tokens'] = 10.5
    out2 = L.simple_baseline_audit(art2)
    assert out2['status'] == 'fail', out2
    assert any('n_tokens is not an integer' in p for p in out2['problems']), out2
    # A mapping/list is not a metric name.  The audit must report it instead
    # of passing it to dict.get() as an unhashable key.
    art3 = json.loads(json.dumps(art))
    art3['selection_metric'] = {'not': 'a metric'}
    out3 = L.simple_baseline_audit(art3)
    assert out3['status'] == 'fail' and any('not a metric name' in p for p in out3['problems']), out3
    # Malformed declared grids/arm metadata are evidence problems, not
    # AttributeError/TypeError crashes in the audit.
    for _bad_proto in (
            {'arms': {'a': {'order': 1}}, 'grids': ['not-a-mapping']},
            {'arms': {'a': ['not-a-mapping']},
             'grids': {'a': {'alpha': [0.1]}}},
            {'arms': {'a': {'order': 1}},
             'grids': {'a': {'alpha': 0.1}}},
            {'arms': {'a': {'order': 1}},
             'grids': {'a': {'alpha': ['not-numeric']}}}):
        _bad_art = json.loads(json.dumps(art3))
        _bad_art['protocol'] = dict(_bad_proto,
                                    selection_metric='ppl',
                                    selection_split='validation',
                                    direction='lower_is_better')
        _bad_out = L.simple_baseline_audit(_bad_art)
        assert _bad_out['status'] == 'fail', (_bad_proto, _bad_out)


def test_reportable_claim_gate():
    base = {'n': 6, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.01, 'p_holm': 0.02}
    r = L.reportable_claim(base)
    assert r['allowed'] is True and r['level'] == 'exploratory_significant', r
    # n<6 is underpowered even with a tiny raw p.
    r = L.reportable_claim(dict(base, n=3, p_holm=0.02))
    assert r['allowed'] is False and r['level'] == 'underpowered', r
    # Missing Holm is uncorrected -> not reportable.
    r = L.reportable_claim({k: v for k, v in base.items() if k != 'p_holm'})
    assert r['allowed'] is False and any('Holm' in x for x in r['reasons']), r
    # Effect inside seed noise is not reportable.
    r = L.reportable_claim(dict(base, mean=-0.1, std=0.2))
    assert r['allowed'] is False and any('seed noise' in x for x in r['reasons']), r
    # A family too large for n=6 cannot reach significance at all.
    r = L.reportable_claim(base, family_size=100)
    assert r['allowed'] is False and r['min_attainable_p_holm_family'] > 0.05, r
    # Confirmatory is only honoured when the caller says so.
    r = L.reportable_claim(base, evidence_level='confirmatory')
    assert r['level'] == 'confirmatory', r
    # A mean whose sign flips when one seed is dropped is not a stable
    # directional claim, even if the aggregate p happens to clear Holm.
    r = L.reportable_claim(dict(base, seed_driven=True))
    assert r['allowed'] is False and any('leave-one-seed-out' in x for x in r['reasons']), r
    # A complete-case p-value computed after dropping missing pairs is not a
    # complete-panel significance claim.
    r = L.reportable_claim(dict(base, n_input=7, n_dropped=1,
                                dropped_seeds=[6]))
    assert r['allowed'] is False and any('complete-case' in x for x in r['reasons']), r
    assert r['n_input'] == 7 and r['n_dropped'] == 1, r
    # Seeds skipped by the same-recipe control gate are also an incomplete
    # paired panel: the surviving subset was selected by that mismatch.
    for _key in ('n_skipped_config_mismatch', 'n_skipped'):
        _rs = L.reportable_claim(dict(base, **{_key: 1}))
        assert _rs['allowed'] is False, (_key, _rs)
        assert any('controlled-comparison' in x for x in _rs['reasons']), (_key, _rs)
        assert _rs.get('n_skipped_config_mismatch') == 1, (_key, _rs)
    _rsk = L.reportable_claim(dict(base, n_skipped_config_mismatch=0))
    assert _rsk['allowed'] is True, _rsk
    assert L.reportable_claim(dict(base, n_skipped_config_mismatch=-1))['allowed'] is False
    assert L.reportable_claim(dict(base, n_skipped_config_mismatch=True))['allowed'] is False
    # Negative or internally inconsistent dropout accounting is not a clean
    # complete panel; it must not bypass the complete-case refusal.
    for _bad in (dict(base, n_input=-1, n_dropped=0),
                 dict(base, n_input=6, n_dropped=-1),
                 dict(base, n_input=3, n_dropped=6),
                 dict(base, n_input=3, n_dropped=0),
                 dict(base, n_input=7, n_dropped=0),
                 dict(base, n_input=7),
                 dict(base, n_input=6.5),
                 dict(base, n_dropped=0.5),
                 dict(base, n_nonzero=-1),
                 dict(base, n_nonzero=1.5)):
        _rb = L.reportable_claim(_bad)
        assert _rb['allowed'] is False, (_bad, _rb)
        assert any('n_input' in x or 'n_dropped' in x or 'panel accounting' in x
                   or 'n_nonzero' in x for x in _rb['reasons']), (_bad, _rb)
    # A threshold outside (0,1) must be rejected, not used to relabel every
    # p-value as significant (alpha=1.5) or unreachable (alpha=0).
    _alpha_pts = [{'label': f'S{i}', 'scale': i + 1,
                   'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2]}
                  for i in range(3)]
    for _bad in (True, 0, 1, -0.1, 1.5, float('nan'), 'not-a-number'):
        for _fn in (lambda a: L.reportable_claim(base, alpha=a),
                    lambda a: L.paired_effect_stats([-1.0, -1.1, -0.9], alpha=a),
                    lambda a: L.holm_adjust([('x', 0.01)], alpha=a),
                    lambda a: L.attach_holm({'x': {'n': 6, 'p_exact_signflip': 0.01}}, alpha=a),
                    lambda a: L.meta_analyze_effects(_alpha_pts, alpha=a),
                    lambda a: L.scaling_trend(_alpha_pts, alpha=a),
                    lambda a: L.loglog_regression([(1, 1), (2, 4), (3, 9)], alpha=a),
                    lambda a: L.methodology_seed_plan(6, alpha=a)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid alpha {_bad!r} was accepted')
    # Minimum-n floors are gate parameters too: bool/fractional values must
    # not be truncated into a weaker floor.
    for _bad_min in (True, 3.5, float('nan'), -1):
        for _fn in (lambda m: L.paired_effect_stats([-1.0, -1.1, -0.9], min_n=m),
                    lambda m: L.meta_analyze_effects([], min_seeds=m),
                    lambda m: L.scaling_trend([], min_seeds=m),
                    lambda m: L.methodology_audit({}, design={}, baselines={},
                                                  min_n=m, write=False, verbose=False)):
            try:
                _fn(_bad_min)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid min_n={_bad_min!r} was accepted')
    # Holm cannot be smaller than the raw p it adjusts.
    r = L.reportable_claim(dict(base, p_holm=0.001))
    assert r['allowed'] is False and any('smaller than the raw p' in x for x in r['reasons']), r
    # Booleans are flags, not p-values.  In particular False must never enter
    # the family as p=0 and become the most significant result.
    for _p in (False, True):
        h = L.holm_adjust([('x', _p)])
        assert h['x']['p_holm'] is None, h
        r = L.reportable_claim(dict(base, p_exact_signflip=_p,
                                    p_holm=0.02))
        assert r['allowed'] is False, r
        r = L.reportable_claim(dict(base, p_holm=_p))
        assert r['allowed'] is False, r
    fam = L.attach_holm({'x': {'n': 6, 'p_exact_signflip': False}})
    assert fam['tested'] == 0, fam
    # Booleans are also not effect measurements.
    assert L.reportable_claim(dict(base, mean=True, std=0.2))['allowed'] is False
    assert L.reportable_claim(dict(base, mean=-1.0, std=False))['allowed'] is False
    # The gate's own parameters must not be silently coerced.  int(5.9)->5
    # would loosen the n floor, and int(True)->1 for family_size would turn a
    # flag into a one-comparison family; both must fail closed.
    for _bad_min in (5.9, True, float('nan'), 0):
        r = L.reportable_claim(base, min_n=_bad_min)
        assert r['allowed'] is False, (r, _bad_min)
        assert any('min_n' in x for x in r['reasons']), r
    for _bad_fam in (0, -3, True, 3.5, float('nan')):
        r = L.reportable_claim(base, family_size=_bad_fam)
        assert r['allowed'] is False, (r, _bad_fam)
        assert any('family_size' in x for x in r['reasons']), r


    # Probabilities outside [0,1] cannot support a claim, even when every
    # strict "< alpha" comparison would accept a negative value.
    bad_neg = dict(base, p_exact_signflip=-0.1, p_holm=-0.05)
    assert L.reportable_claim(bad_neg)['allowed'] is False, L.reportable_claim(bad_neg)
    assert BR._sig_gate(bad_neg, label='negative p demo') is False
    # A Holm p below its raw p is a mis-attached family correction.
    bad_holm = dict(base, p_exact_signflip=0.04, p_holm=0.02)
    assert L.reportable_claim(bad_holm)['allowed'] is False, L.reportable_claim(bad_holm)
    assert BR._sig_gate(bad_holm, label='holm order demo') is False


def test_holm_family_resolution_ceiling():
    comps = {f'c{i}': {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25} for i in range(100)}
    fam = L.attach_holm(comps, family='100 exploratory contrasts')
    assert fam['family_resolution_limited'] is True, fam
    assert fam['family_min_attainable_p_holm'] > 0.05, fam
    assert fam['n_effective_min'] == 3 and fam['n_effective_max'] == 3, fam
    # A small, powered family is not flagged.
    comps2 = {f'c{i}': {'n': 12, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.001} for i in range(4)}
    fam2 = L.attach_holm(comps2, family='4 confirmatory contrasts', exploratory=False)
    assert fam2['family_resolution_limited'] is False, fam2
    assert fam2['evidence_level'] == 'confirmatory', fam2


def test_crossover_scaling_refusals():
    def pts(n, d=(128, 256, 384, 512), tok=(50e6,), censored=False, vol=(50e6,)):
        return [{'label': f'd{dd}', 'd': dd, 'crossover_tokens': tok[0], 'train_tokens': vol[i % len(vol)],
                 'censored': censored, 'reconstructed': False} for i, dd in enumerate(d[:n])]
    # <4 distinct widths -> refused.
    f = L.crossover_scaling_fit(pts(3))
    assert f['status'] == 'insufficient_points', f
    # Any censored point refuses the fit (informative censoring).
    f = L.crossover_scaling_fit(pts(4, censored=True))
    assert f['status'] == 'censored' and f['reasons'], f
    # Training-volume spread > 25% refuses the d slope.
    f = L.crossover_scaling_fit(pts(4, vol=(25e6, 50e6)))
    assert f['status'] == 'training_volume_confounded', f
    # Duplicate widths keep the token-matched panel and are named.
    dup = pts(4) + [{'label': 'd512_half', 'd': 512, 'crossover_tokens': 40e6, 'train_tokens': 25e6}]
    f = L.crossover_scaling_fit(dup)
    assert f['status'] == 'ok' and 'd512_half' in (f.get('deduped_by_d') or []), f
    # A crossover beyond the panel's own training budget is impossible and
    # must not enter a scaling fit.
    bad = pts(4)
    bad[1] = dict(bad[1], crossover_tokens=60e6, train_tokens=50e6)
    fb = L.crossover_scaling_fit(bad)
    assert fb['status'] == 'inconsistent_points', fb
    assert fb['inconsistent_panels'] == ['d256'], fb
    # A clean 4-point fit carries a slope CI and is labelled descriptive.
    f = L.crossover_scaling_fit(pts(4))
    assert f['status'] == 'ok' and f.get('slope') is not None and f.get('ci') is not None, f
    assert f['claim_level'] == 'descriptive', f
    assert f['volume_ratio'] <= 1.25, f
    # A perfectly recovered power law has zero residual variance: the slope's
    # t statistic is infinite, so p must be 0, not NaN.
    perfect = [{'x': d, 'y': 2.0 * d ** 1.5, 'label': f'd{d}'} for d in (16, 32, 64, 128)]
    lr = L.loglog_regression(perfect)
    assert lr['status'] == 'ok' and abs(lr['slope'] - 1.5) < 1e-12, lr
    assert math.isfinite(lr['p']) and lr['p'] < 1e-12, lr


def test_scale_ladder_budget_and_transfer_audit():
    def tier(name, steps, params, seq=8, batch=2, tokens=None, lr=3e-4):
        cfg = {'steps': steps, 'batch_size': batch, 'seq_len': seq,
               'n_train_tokens': tokens if tokens is not None else steps * batch * seq,
               'lr': lr, 'warmup': max(1, steps // 10), 'seeds': [0, 1, 2, 3]}
        return {'name': name, 'cfg': cfg, 'params': params}
    a = tier('S1', 1000, 1000)
    b = tier('S2', 2000, 2000)  # same tokens/param, same lr
    out = L.scale_ladder_audit([a, b])
    assert out['budget_matched'] is True, out
    assert out['lr_constant'] is True, out
    # A constant LR is disclosed as an unvalidated transfer assumption.
    assert any('transfer' in w for w in out['warnings']), out
    # A different tokens-per-parameter budget is a visible confound.
    b2 = tier('S2', 3000, 2000)
    out2 = L.scale_ladder_audit([a, b2])
    assert out2['budget_matched'] is False, out2
    assert any('tokens-per-parameter' in w for w in out2['warnings']), out2
    # A declared token budget that disagrees with steps*batch*seq is a hard
    # fail, not a warning.
    bad = tier('S1', 1000, 1000, tokens=999999)
    out3 = L.scale_ladder_audit([bad, b])
    assert out3['status'] == 'fail', out3
    assert any('n_train_tokens' in p for p in out3['problems']), out3
    # Missing parameter counts keep the budget claim unverifiable.
    out4 = L.scale_ladder_audit([tier('S1', 1000, None), b])
    assert out4['budget_matched'] is None, out4
    assert any('params missing' in w for w in out4['warnings']), out4
    # The deployed plan must carry an explicit transfer policy and expose the
    # S1/S2 budget ratio rather than silently calling it matched.
    pol = L.SCALE_TRANSFER_POLICY
    assert pol.get('lr_transfer') and pol.get('tokens_per_param_tolerance') is not None, pol
    # A tokenizer width is an exact identity.  int() coercion would truncate a
    # fractional/boolean value into a DIFFERENT vocab that looks shared.
    va = tier('S1', 1000, 1000); va['cfg']['vocab'] = 32768
    vb = tier('S2', 2000, 2000); vb['cfg']['vocab'] = 32768.5
    out5 = L.scale_ladder_audit([va, vb])
    assert out5['status'] == 'fail', out5
    assert any('vocab' in p for p in out5['problems']), out5
    assert out5['corpus_binding'].get('vocab') != 'shared', out5
    vc = tier('S2', 2000, 2000); vc['cfg']['vocab'] = True
    out6 = L.scale_ladder_audit([va, vc])
    assert out6['status'] == 'fail', out6
    # A boolean budget/seed plan is a flag, not an integer; `or 0`/`or []`
    # must not turn it into a valid zero.
    for _field, _val in (('warmup', False), ('n_train_tokens', False),
                         ('seeds', False), ('seeds', '0,1')):
        _bt = tier('S1', 1000, 1000); _bt['cfg'][_field] = _val
        _bo = L.scale_ladder_audit([_bt, b])
        assert _bo['status'] == 'fail', (_bo, _field, _val)
        assert any(_field in p for p in _bo['problems']), (_bo, _field, _val)
    # Boolean/non-finite learning rates must not be coerced to 1.0/NaN.
    for _bad_lr in (True, False, float('nan'), float('inf')):
        _bad = tier('S1', 1000, 1000); _bad['cfg']['lr'] = _bad_lr
        _o = L.scale_ladder_audit([_bad, b])
        assert _o['status'] == 'fail', (_o, _bad_lr)
        assert any('lr=' in p for p in _o['problems']), _o
    out7 = L.scale_ladder_audit([va, tier('S2', 2000, 2000) | {'vocab': 32768}])
    assert out7['corpus_binding'].get('vocab') == 'shared', out7
    # The optimizer/schedule/augmentation descriptor must be shared across
    # tiers.  A mismatch is a hard fail; missing/partial is unverifiable.
    kn = L.training_knobs()
    ka = tier('S1', 1000, 1000); ka['cfg']['train_knobs'] = kn
    kb = tier('S2', 2000, 2000); kb['cfg']['train_knobs'] = kn
    okn = L.scale_ladder_audit([ka, kb])
    assert okn['train_knobs_binding'] == 'shared', okn
    badk = tier('S2', 2000, 2000)
    badk['cfg']['train_knobs'] = dict(kn, grad_clip_norm=2.0)
    obk = L.scale_ladder_audit([ka, badk])
    assert obk['status'] == 'fail', obk
    assert any('training knobs differ' in p for p in obk['problems']), obk
    opk = L.scale_ladder_audit([ka, b])
    assert opk['train_knobs_binding'] == 'partial', opk
    assert any('train_knobs descriptor' in w for w in opk['warnings']), opk
    onk = L.scale_ladder_audit([tier('S1', 1000, 1000), tier('S2', 2000, 2000)])
    assert onk['train_knobs_binding'] == 'unverifiable', onk
    assert any('train_knobs descriptor' in w for w in onk['warnings']), onk
    badtype = tier('S1', 1000, 1000); badtype['cfg']['train_knobs'] = 'adamw'
    obt = L.scale_ladder_audit([badtype, kb])
    assert obt['status'] == 'fail', obt
    assert any('train_knobs is not an object' in p for p in obt['problems']), obt
    # A non-constant LR is a declared transfer choice, never a matched
    # optimizer control: it must produce a confound warning, not silence.
    lrv = tier('S2', 2000, 2000, lr=1e-3)
    olr = L.scale_ladder_audit([tier('S1', 1000, 1000), lrv])
    assert olr['lr_constant'] is False and olr['lr_values'] == [0.0003, 0.001], olr
    assert any('learning rate DIFFERS' in w for w in olr['warnings']), olr
    # Seed identities are the basis of a cross-tier paired contrast.  A plan
    # with duplicate/negative seeds is malformed, and differing seed sets must
    # be disclosed instead of treating each tier's extra seeds as replication.
    shared = L.scale_ladder_audit([tier('S1', 1000, 1000), tier('S2', 2000, 2000)])
    assert shared['seed_set_shared'] is True and shared['n_common_seeds'] == 4, shared
    diff = tier('S2', 2000, 2000); diff['cfg']['seeds'] = [2, 3, 4, 5]
    od = L.scale_ladder_audit([tier('S1', 1000, 1000), diff])
    assert od['seed_set_shared'] is False and od['seed_set_intersection'] == [2, 3], od
    assert any('seed set' in w for w in od['warnings']), od
    dup = tier('S1', 1000, 1000); dup['cfg']['seeds'] = [0, 0, 1, 1]
    odup = L.scale_ladder_audit([dup, b])
    assert odup['status'] == 'fail' and any('duplicate seed' in p for p in odup['problems']), odup
    neg = tier('S1', 1000, 1000); neg['cfg']['seeds'] = [-1, 0, 1, 2]
    oneg = L.scale_ladder_audit([neg, b])
    assert oneg['status'] == 'fail' and any('negative' in p for p in oneg['problems']), oneg

    # A tier's power is a property of the DECLARED Holm family, not of the raw
    # sign-flip test.  n=6 can reach raw p=2/64=0.03125 but not Holm p for a
    # 3-comparison family (3*2/64=0.09375), so it must stay descriptive.
    def _with_seeds(name, seeds):
        t = tier(name, 1000, 1000)
        t['cfg']['seeds'] = list(seeds)
        return t
    fam = L.scale_ladder_audit(
        [_with_seeds('S1', range(6)), _with_seeds('S2', range(4))],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=3))
    _s1, _s2 = fam['tiers']
    assert abs(_s1['p_floor'] - 2.0 / 64.0) < 1e-15, _s1
    assert abs(_s1['p_floor_family_holm'] - 3.0 * 2.0 / 64.0) < 1e-15, _s1
    assert _s1['descriptive_only'] is True, _s1
    assert _s1['descriptive_only_basis'] == 'holm_family', _s1
    assert _s2['descriptive_only'] is True, _s2
    # The paired cross-tier contrast can use only the 4 common seeds, not the
    # 6 seeds one tier happens to have.
    assert fam['n_common_seeds'] == 4, fam
    assert abs(fam['common_seed_paired_p_floor'] - 2.0 / 16.0) < 1e-15, fam
    assert fam['common_seeds_sufficient_for_family'] is False, fam
    assert any('common seed' in w for w in fam['warnings']), fam
    # Seven common seeds clear a 3-comparison family at alpha=0.05
    # (3*2/128=0.046875), so the same ladder becomes planned-sufficient.
    fam_ok = L.scale_ladder_audit(
        [_with_seeds('S1', range(7)), _with_seeds('S2', range(7))],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=3))
    assert fam_ok['family_min_seeds'] == 7, fam_ok
    assert fam_ok['common_seeds_sufficient_for_family'] is True, fam_ok
    assert all(t['descriptive_only'] is False for t in fam_ok['tiers']), fam_ok
    # Malformed declarations fail closed instead of crashing the pre-flight
    # audit (a truthy scalar `cfg` used to reach `.get`).
    mal = L.scale_ladder_audit([dict(a, cfg='not-a-recipe'), b])
    assert mal['status'] == 'fail' and any('cfg is not an object' in p
                                          for p in mal['problems']), mal
    mal2 = L.scale_ladder_audit([123, b])
    assert mal2['status'] == 'fail' and any('not a dict' in p
                                           for p in mal2['problems']), mal2
    for _bad_policy in (123, 'x', [1], True):
        try:
            L.scale_ladder_audit([a, b], policy=_bad_policy)
        except ValueError as e:
            assert 'policy' in str(e), e
        else:
            raise AssertionError(f'policy={_bad_policy!r} must be refused')


def test_scale_ladder_huge_token_budget_is_reportable():
    # A syntactically valid scale plan is allowed to declare a token budget
    # far beyond any training hardware.  The audit must compare it with exact
    # integer arithmetic and report an unrepresentable tokens/parameter ratio
    # as unverifiable, not crash while converting the budget to float.
    H = 10 ** 400

    def tier(name, steps, params):
        return {'name': name,
                'cfg': {'steps': steps, 'batch_size': 1, 'seq_len': 1,
                        'n_train_tokens': steps, 'lr': 3e-4,
                        'warmup': steps // 10, 'seeds': [0, 1]},
                'params': params}

    out = L.scale_ladder_audit([tier('S1', H, H), tier('S2', 2 * H, H)])
    assert out['status'] in ('ok', 'warn'), out
    assert out['budget_matched'] is False, out  # 1 vs 2 tokens/param
    assert out['tiers'][0]['tokens_per_param'] == 1.0, out['tiers'][0]

    out2 = L.scale_ladder_audit([tier('S1', H, 1), tier('S2', 2 * H, 1)])
    assert out2['budget_matched'] is None, out2
    assert all(t['tokens_per_param'] is None for t in out2['tiers']), out2
    assert sum('not representable' in w for w in out2['warnings']) == 2, out2


def test_init_core_fingerprint_and_pair_gate():
    dev = torch.device('cpu')
    L.set_seed(3)
    m1 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'full')).to(dev)
    L.set_seed(3)
    m2 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'csa_fixed')).to(dev)
    L.set_seed(4)
    m3 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'full')).to(dev)
    fa, fb, fc = (L.init_core_fingerprint(m) for m in (m1, m2, m3))
    assert fa == fb, 'same seed + shared trunk must hash identically across variants'
    assert fa != fc, 'a different seed must produce a different init digest'
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    common = dict(run_cfg=fp, steps=1, tokens_seen=1, train_knobs=L.training_knobs(),
                  data_fp='d', val_fp='v', eval_fp='e')
    a = dict(common, variant='a', seed=0, ppl=10.0, init_core_fp=fa)
    b = dict(common, variant='b', seed=0, ppl=11.0, init_core_fp=fa)
    assert L.pair_reason(a, b) is None, L.pair_reason(a, b)
    why = L.pair_reason(a, dict(b, init_core_fp=fc))
    assert why is not None and 'init_core_fp' in why, why
    why = L.pair_reason(a, {k: v for k, v in b.items() if k != 'init_core_fp'})
    assert why is not None and 'unverifiable init_core_fp' in why, why
    # The zero-GPU report builder's local pairing gate must mirror this field
    # too; otherwise it can print a contrast the statistics layer refuses.
    assert BR._pair_reason(a, b) is None, BR._pair_reason(a, b)
    why = BR._pair_reason(a, dict(b, init_core_fp=fc))
    assert why is not None and 'init_core_fp' in why, why
    why = BR._pair_reason(a, {k: v for k, v in b.items() if k != 'init_core_fp'})
    assert why is not None and 'unverifiable init_core_fp' in why, why


def test_param_and_compute_control_detail():
    bd = {'total': 1000, 'non_embedding': 800, 'active': 700, 'embedding': 200}
    recs = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd, 'train_time_s': 10.0, 'tokens_seen': 100.0},
            ('b', 'p', 0): {'params': 1000, 'param_breakdown': {'total': 1000, 'non_embedding': 800, 'active': 600, 'embedding': 200},
                            'train_time_s': 20.0, 'tokens_seen': 100.0}}
    _rel, ok = L._param_control(recs, 'a', 'b')
    assert ok is True, 'total parameters are matched'
    d = L._param_control_detail(recs, 'a', 'b')
    assert d['total_controlled'] is True and d['active_controlled'] is False, d
    assert d['active_rel_diff'] > 0.01, d
    assert L._param_capacity_controlled(d) is False, d
    assert 'breakdown' in L.param_pair_reason(recs[('a', 'p', 0)], recs[('b', 'p', 0)]), 'breakdown must fail closed'
    # One-sided breakdown is unverifiable, not a total-only pass.
    one = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
           ('b', 'p', 0): {'params': 1000}}
    od = L._param_control_detail(one, 'a', 'b')
    assert od['breakdown_one_sided'] is True and L._param_capacity_controlled(od) is None, od
    assert 'one record carries it' in L.param_pair_reason(one[('a', 'p', 0)], one[('b', 'p', 0)]), 'one-sided breakdown must fail closed'
    # A single consistent seed must not hide a different/corrupted record of
    # the same arm: the arm-level capacity verdict is unverifiable, not the
    # first record's pass.
    mixed = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
             ('a', 'p', 1): {'params': 1200, 'param_breakdown': bd},
             ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._params_of_arm(mixed, 'a') is None
    assert L._param_control(mixed, 'a', 'b') == (None, None)
    md = L._param_control_detail(mixed, 'a', 'b')
    assert md['total_controlled'] is None, md
    assert L._param_capacity_controlled(md) is None, md
    mixed_bd = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
                ('a', 'p', 1): {'params': 1000, 'param_breakdown': dict(bd, active=650)},
                ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._param_breakdown_of_arm(mixed_bd, 'a') is None
    assert L._param_capacity_controlled(L._param_control_detail(mixed_bd, 'a', 'b')) is None
    partial = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
               ('a', 'p', 1): {'params': 1000},
               ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._param_breakdown_of_arm(partial, 'a') is None
    assert L._param_capacity_controlled(L._param_control_detail(partial, 'a', 'b')) is None
    c = L._compute_control_detail(recs, 'a', 'b')
    assert c['compute_matched'] is False and c['more_compute_arm'] == 'b', c
    # Compute control consumes exact token counts: a fractional/bool count is
    # not a valid denominator, so the comparison is unverifiable rather than
    # silently using 100.5 (or True->1) tokens.
    for _bad_tok in (100.5, True, float('nan')):
        frac = {('a', 'p', 0): {'params': 1000, 'train_time_s': 10.0, 'tokens_seen': _bad_tok},
                ('b', 'p', 0): {'params': 1000, 'train_time_s': 20.0, 'tokens_seen': _bad_tok}}
        _c = L._compute_control_detail(frac, 'a', 'b')
        assert _c['compute_matched'] is None, (_c, _bad_tok)
        assert _c['time_ratio_a_over_b'] is None and _c['proxy_ratio_a_over_b'] is None, _c
    assert abs(c['time_ratio_a_over_b'] - 0.5) < 1e-12, c
    # A significant contrast whose extra compute favours the winner is
    # system-level, not a verified mechanism.
    entry = {'controlled': True, 'param_controlled': True, 'n': 6, 'significant_global': True,
             'predicted_sign': -1, 'direction_consistent': True, 'within_seed_noise': False,
             'mechanism_claim_allowed': False}
    assert L._classify_group_status([entry], min_n=6, alpha=0.05, m_family=1) == 'system_level_only'
    # End-to-end: b wins while consuming 2x the measured compute -> the
    # contrast lands in system_level_only, never in verified_components.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=100, ppl=50.0, params=1000,
                             train_knobs=kn, train_time_s=10.0, data_fp='d', val_fp='v', eval_fp='e')
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=100, ppl=49.0, params=1000,
                             train_knobs=kn, train_time_s=20.0, data_fp='d', val_fp='v', eval_fp='e')
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    assert 'c' in r['system_level_only_components'] and 'c' not in r['verified_components'], r
    assert r['components']['c'][0]['compute_confound_direction'] == 'favors_claimed_winner', r['components']['c'][0]


def test_capacity_breakdown_gates_baseline_claim():
    # Equal TOTAL parameters but a reallocated active count (dense MLP width /
    # dead indexer projections) is not a capacity-matched control.  The
    # breakdown must reach the verdict, not only the console.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=49.0, params=1000,
                             param_breakdown={'total': 1000, 'non_embedding': 800,
                                              'active': 700, 'embedding': 200},
                             train_knobs=kn)
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=50.0, params=1000,
                             param_breakdown={'total': 1000, 'non_embedding': 800,
                                              'active': 600, 'embedding': 200},
                             train_knobs=kn)
    r = L.methodology_audit(s, design={}, baselines={'comp': [('a', 'b', 'x')]},
                            verbose=False)
    e = r['baselines']['comp'][0]
    assert e['param_controlled'] is True, e
    assert e['param_capacity_controlled'] is False, e
    assert e['capacity_confounded'] is True, e
    assert 'comp' not in r['verified_baselines'], r['verified_baselines']
    assert 'comp' in r['unverified_baselines'], r['unverified_baselines']
    assert any('match on TOTAL parameters' in w for w in r['warnings']), r['warnings']


def test_hypothesis_param_match_requirement_is_enforced():
    # Every hypothesis must declare whether equal capacity is part of the
    # claim; an omitted field is a design defect, not a silent default.
    hp = {'statement': 'x', 'mechanism': 'y', 'predicted_sign': -1,
          'primary_metric': 'ppl', 'falsifier': 'z'}
    a = L.hypothesis_audit({'h': hp}, 'component')
    assert 'h' in a['invalid'] and any('requires_param_match' in w for w in a['warnings']), a
    hp2 = dict(hp, requires_param_match=True)
    assert L.hypothesis_audit({'h': hp2}, 'component')['status'] == 'ok'
    # A significant contrast whose hypothesis promises matched parameters is
    # system-level (capacity-confounded), never 'verified'.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=49.0, params=1200, train_knobs=kn)
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=50.0, params=1000, train_knobs=kn)
    spec = {'statement': 'a beats b at matched capacity', 'mechanism': 'm',
            'predicted_sign': -1, 'primary_metric': 'ppl',
            'requires_param_match': True, 'falsifier': 'f'}
    saved = L.HYPOTHESIS_DESIGN.get('custom_capacity')
    L.HYPOTHESIS_DESIGN['custom_capacity'] = spec
    try:
        r = L.methodology_audit(s, design={'custom_capacity': [('a', 'b', 'x')]},
                                baselines={}, verbose=False)
    finally:
        if saved is None:
            L.HYPOTHESIS_DESIGN.pop('custom_capacity', None)
        else:
            L.HYPOTHESIS_DESIGN['custom_capacity'] = saved
    e = r['components']['custom_capacity'][0]
    assert e['requires_param_match'] is True and e['param_match_unmet'] is True, e
    assert e['mechanism_claim_allowed'] is False, e
    assert 'custom_capacity' in r['system_level_only_components'], r
    assert 'custom_capacity' not in r['verified_components'], r


def test_stale_aggregate_refused():
    cur = BR._CUR_CS
    d = {'a@cfg1': {'variant': 'a', 'run_cfg': f'steps1_d8_L1_csOLD1', 'n_seeds': 4, 'ppl_mean': 10.0},
         'a@cfg2': {'variant': 'a', 'run_cfg': f'steps1_d8_L1_csOLD2', 'n_seeds': 2, 'ppl_mean': 11.0}}
    assert BR._agg_entry_scan(d, 'a') is None, 'no current stamp among multiple groups must be refused'
    d['a@cfg3'] = {'variant': 'a', 'run_cfg': f'steps1_d8_L1_cs{cur}', 'n_seeds': 1, 'ppl_mean': 12.0}
    assert BR._agg_entry_scan(d, 'a') is d['a@cfg3'], 'the current-stamped group must be selected'
    single = {'a': {'variant': 'a', 'run_cfg': 'steps1_d8_L1_csOLD', 'n_seeds': 4, 'ppl_mean': 10.0}}
    assert BR._agg_entry_scan(single, 'a') is None, 'a single explicit stale run_cfg must be refused'


def test_sec_p1l_excludes_stale_records():
    stale = {'v::s0': {'variant': 'v', 'seed': 0, '_code': 'v0.0', 'by_len': {'512': {'ppl': 10.0}, '4096': {'ppl': 40.0}}}}
    BR._LOADED[('summary', 'results_len')] = stale
    try:
        txt = BR.sec_p1l()
    finally:
        BR._LOADED.pop(('summary', 'results_len'), None)
    assert '陈旧记录已排除' in txt, txt[:400]
    assert '10.0' not in txt and '40.0' not in txt, 'stale PPL must not be quoted in the section'
    assert 'ratio' not in txt or '不计算 ratio' in txt, txt[:400]


def test_efficiency_pairs_audit_and_sweep():
    td = tempfile.mkdtemp(prefix='eff_', dir=REPO)
    try:
        eff_a = {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
                 'prefill_ms_p50': 10.0, 'decode_has_kv_cache': False}
        eff_b = dict(eff_a, prefill_ms_p50=20.0)
        summ = {'a::s0': {'variant': 'a', 'efficiency': eff_a}, 'b::s0': {'variant': 'b', 'efficiency': eff_b}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summ, f)
        r = A.audit_efficiency_pairs(outdirs=[td])
        rec = r['panels'][td]['a vs b']
        assert rec['speed_claim_allowed'] is True and rec['faster_arm'] == 'a', rec
        assert abs(rec['ratio_b_over_a'] - 2.0) < 1e-12, rec
        # A different token count is not a speed comparison.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=4096)}}, f)
        r2 = A.audit_efficiency_pairs(outdirs=[td])
        rec2 = r2['panels'][td]['a vs b']
        assert rec2['speed_claim_allowed'] is False and 'token counts' in rec2['reason'], rec2
        # A fractional token count is not the same shape as its truncated int:
        # int(1.5) == 1 must not certify a latency comparison.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, n_tokens=1.5)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=1)}}, f)
        _rf = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rf['speed_claim_allowed'] is False and 'token counts' in _rf['reason'], _rf
        # A zero/negative token count is not a measured shape either, even
        # when both arms happen to agree on it.
        for _bad_n in (0, -1):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, n_tokens=_bad_n)},
                           'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=_bad_n)}}, f)
            _rn = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
            assert _rn['speed_claim_allowed'] is False and 'token counts' in _rn['reason'], _rn
        # A mixed device/KV record inside one arm makes the medians
        # incomparable even when the first record of each arm matches.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'a::s1': {'variant': 'a', 'efficiency': dict(eff_a, device='cuda:1')},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b},
                       'b::s1': {'variant': 'b', 'efficiency': dict(eff_b, prefill_ms_p50=20.0)}}, f)
        r3 = A.audit_efficiency_pairs(outdirs=[td])
        rec3 = r3['panels'][td]['a vs b']
        assert rec3['speed_claim_allowed'] is False and 'device' in rec3['reason'], rec3
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, prefill_ms_p50=True)},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        r4 = A.audit_efficiency_pairs(outdirs=[td])
        rec4 = r4['panels'][td]['a vs b']
        assert rec4['speed_claim_allowed'] is False, rec4
        # Missing device/torch or KV provenance is unverifiable, not equal.
        _bare = {'status': 'ok', 'n_tokens': 512, 'prefill_ms_p50': 10.0}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(_bare)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(_bare, prefill_ms_p50=20.0)}}, f)
        _rm = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rm['same_device_torch'] is False and _rm['speed_claim_allowed'] is False, _rm
        _no_kv = dict(eff_a)
        _no_kv.pop('decode_has_kv_cache')
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': _no_kv},
                       'b::s0': {'variant': 'b', 'efficiency': dict(_no_kv, prefill_ms_p50=20.0)}}, f)
        _rk = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rk['same_kv_semantics'] is False and _rk['speed_claim_allowed'] is False, _rk
        # The same token count at a different batch size is a different
        # execution shape, so it cannot support a speed claim.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, batch=1)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, batch=8)}}, f)
        _rb = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rb['same_batch'] is False and _rb['speed_claim_allowed'] is False, _rb
        assert 'batch' in _rb['reason'], _rb
        # Missing batch is legacy-compatible but explicitly not recorded.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        _rleg = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rleg['same_batch'] is True and _rleg['batch_status'] == 'unrecorded', _rleg
        assert _rleg['speed_claim_allowed'] is True, _rleg
        # An invalid tolerance cannot widen (negative) or disable the speed gate.
        for _bad_tol in (True, -0.05, float('nan'), float('inf'), 1.0, 'x'):
            try:
                A.audit_efficiency_pairs(outdirs=[td], tolerance=_bad_tol)
            except ValueError:
                pass
            else:
                raise AssertionError(f'tolerance={_bad_tol!r} must be refused')
    finally:
        shutil.rmtree(td, ignore_errors=True)
    L.set_seed(0)
    model = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'full')).eval()
    ids = torch.randint(0, 64, (1, 20), dtype=torch.long)
    sw = L.measure_efficiency_sweep(model, ids, torch.device('cpu'), lengths=(8, 16), batches=(1, 2), warmup=0, iters=1)
    assert sw['status'] == 'ok' and len(sw['rows']) == 4, sw
    assert sw['requested_seq_lens'] == [8, 16], sw
    assert sw['measured_seq_lens'] == [8, 16] and sw['skipped_seq_lens'] == [], sw
    assert sw['decode_has_kv_cache'] is False and sw['measurement'] == 'cpu perf_counter', sw
    _partial = L.measure_efficiency_sweep(
        model, ids, torch.device('cpu'), lengths=(8, 100), batches=(1,),
        warmup=0, iters=1)
    assert _partial['status'] == 'ok', _partial
    assert _partial['measured_seq_lens'] == [8], _partial
    assert _partial['skipped_seq_lens'] == [100], _partial
    assert 'not measured' in _partial.get('skipped_seq_lens_reason', ''), _partial
    # A malformed/empty shape grid must not produce an empty row list that
    # still claims status=ok: that looks like a completed sweep.
    for _kw in ({'lengths': ()}, {'lengths': (8.5,)}, {'lengths': (True,)},
                {'lengths': (100,)}, {'batches': ()}, {'batches': (0,)},
                {'batches': (True,)}, {'warmup': -1}, {'iters': 0}):
        _args = dict(lengths=(8,), batches=(1,), warmup=0, iters=1)
        _args.update(_kw)
        try:
            L.measure_efficiency_sweep(model, ids, torch.device('cpu'), **_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency_sweep accepted {_kw!r}')
    for _kw in ({'warmup': -1}, {'warmup': 1.5}, {'iters': 0},
                {'iters': True}, {'decode_tokens': 0}):
        _args = dict(warmup=0, iters=1, decode_tokens=1)
        _args.update(_kw)
        try:
            L.measure_efficiency(model, ids, torch.device('cpu'), **_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency accepted {_kw!r}')
    for _bad_ids in (torch.empty((1, 0), dtype=torch.long),
                     torch.empty((0, 8), dtype=torch.long),
                     torch.empty((1, 1, 1), dtype=torch.long)):
        try:
            L.measure_efficiency(model, _bad_ids, torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency accepted ids shape {tuple(_bad_ids.shape)}')
        try:
            L.measure_efficiency_sweep(model, _bad_ids, torch.device('cpu'),
                                       lengths=(1,), batches=(1,),
                                       warmup=0, iters=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency_sweep accepted ids shape {tuple(_bad_ids.shape)}')


def test_efficiency_pareto_is_grouped():
    a = {'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
         'decode_has_kv_cache': False, 'prefill_ms_p50': 10.0}
    assert BR.efficiency_group_key(a) == ('cuda:0', '2.8.0', 512, False)
    assert BR.efficiency_group_key(dict(a, n_tokens=4096)) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, device='cuda:1')) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, torch='2.9.0')) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, decode_has_kv_cache=True)) != BR.efficiency_group_key(a)
    for bad in (dict(a, device=''), dict(a, torch=None), dict(a, n_tokens=True),
                dict(a, n_tokens=1.5), dict(a, n_tokens=0),
                {k: v for k, v in a.items() if k != 'decode_has_kv_cache'}):
        assert BR.efficiency_group_key(bad) is None, bad
    # Lower latency at higher PPL is dominated; only the true staircase stays.
    fr = BR.efficiency_pareto([(10.0, 100.0, 'a'), (20.0, 90.0, 'b'),
                               (30.0, 95.0, 'c'), (5.0, 110.0, 'd')])
    assert [x[2] for x in fr] == ['d', 'a', 'b'], fr
    assert BR.efficiency_pareto([(float('nan'), 90.0, 'x'), (10.0, 100.0, 'a')]) == [(10.0, 100.0, 'a')]
    # Zero time / non-positive PPL are not measurements and must not become a
    # frontier point that dominates every real measurement.
    assert BR.efficiency_pareto([(0.0, 50.0, 'z'), (-1.0, 40.0, 'n'),
                                 (1.0, 0.0, 'p'), (10.0, 100.0, 'a')]) == [(10.0, 100.0, 'a')]


def test_efficiency_accuracy_tradeoff_summary():
    rows = [{'variant': 'best', 'ppl': p, 'prefill_ms_p50': 20.0,
             'params': 2000, 'peak_mem_allocated_mb': 200.0}
            for p in (48.0, 48.1, 47.9)]
    rows += [{'variant': 'cheap', 'ppl': p, 'prefill_ms_p50': 10.0,
              'params': 1000, 'peak_mem_allocated_mb': 100.0}
             for p in (50.0, 50.1, 49.9)]
    rows += [{'variant': 'tiny', 'ppl': p, 'prefill_ms_p50': 21.0,
              'params': 2000, 'peak_mem_allocated_mb': 200.0}
             for p in (48.05, 48.05, 48.05)]
    s = BR.efficiency_accuracy_summary(rows)
    assert [r['variant'] for r in s] == ['best', 'tiny', 'cheap'], s
    assert s[0]['ppl_sem'] is not None and s[0]['latency_ratio_vs_best'] == 1.0, s[0]
    cheap = [r for r in s if r['variant'] == 'cheap'][0]
    assert cheap['distinguishable'] is True, cheap
    assert cheap['ppl_delta_pct_vs_best'] > 3.0, cheap
    assert 'latency_2x_or_more' not in cheap['flags'], cheap
    assert cheap['params_ratio_vs_best'] == 0.5, cheap
    tiny = [r for r in s if r['variant'] == 'tiny'][0]
    assert 'ppl_gain_below_0.5pct' in tiny['flags'], tiny
    assert 'ppl_gain_within_seed_noise' in tiny['flags'], tiny
    # Unverifiable uncertainty is named instead of being treated as significant.
    unk = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    b = [r for r in unk if r['variant'] == 'b'][0]
    assert b['distinguishable'] is None and 'ppl_uncertainty_unverifiable' in b['flags'], b
    zero = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    zb = [r for r in zero if r['variant'] == 'b'][0]
    assert zb['distinguishable'] is None, zb
    assert 'zero_within_seed_variance_unverifiable' in zb['flags'], zb
    budget = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 1000},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 500}])
    assert all('train_budget_differs' in r['flags'] for r in budget), budget
    assert all(r['budget_matched'] is False for r in budget), budget
    # Missing budget is unverifiable, not equal to another missing budget.
    missing = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    assert all(r['budget_matched'] is False for r in missing), missing
    assert all('train_budget_unverifiable' in r['flags'] for r in missing), missing
    # Two medians can agree while an arm mixes budgets; that is still not a
    # controlled comparison.
    mixed = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 100},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 300},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 200},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 200}])
    assert all(r['budget_matched'] is False for r in mixed), mixed
    assert all('train_budget_inconsistent_within_arm' in r['flags'] for r in mixed), mixed
    matched = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 1000},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 1000}])
    assert all(r['budget_matched'] is True for r in matched), matched
    # Zero is not a trained budget, and two invalid records must not match.
    zero_tok = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 0}])
    assert all(r['budget_matched'] is False for r in zero_tok), zero_tok
    assert all('train_budget_unverifiable' in r['flags'] for r in zero_tok), zero_tok
    # Invalid latency/parameter/memory readings are unverifiable, not numbers
    # that can support a ratio.
    invalid = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'z', 'ppl': 40.0, 'prefill_ms_p50': 0.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0,
         'params': 0, 'peak_mem_allocated_mb': -1.0}])
    assert [r['variant'] for r in invalid] == ['a', 'b'], invalid
    bb = [r for r in invalid if r['variant'] == 'b'][0]
    assert bb['params_median'] is None and bb['peak_mem_max'] is None, bb


def test_report_error_analysis_section():
    rec = {'checkpoint': '/tmp/x/ckpt.pt', 'split': 'validation',
           'mean_ce': 4.2, 'pooled_ppl': 66.7,
           'rare_target_basis': 'unavailable: no train_ids',
           'rare_target_problem': 'empty train_ids: token frequencies are unmeasured',
           'buckets': {'position:last10%': {'n': 3, 'mean_ce': 5.0, 'perplexity': 148.4}},
           'cause_counts': {'ambiguous_context': 2, 'annotation_or_rare_target': 1},
           'cause_counts_all': {'ambiguous_context': {'n': 9, 'share': 0.9, 'mean_ce': 4.1},
                                'annotation_or_rare_target': {'n': 1, 'share': 0.1, 'mean_ce': 6.0}},
           'cause_guide': {'ambiguous_context': 'read the context'},
           'worst_tokens': [{'row': 0, 'position': 3, 'ce': 9.5,
                             'target_text': 'x', 'pred_text': 'y',
                             'context_text': 'the surrounding words',
                             'cause_primary': 'ambiguous_context'}],
           'warning': 'validation only'}
    orig = BR.load
    BR.load = lambda path, default=None: rec if path.endswith(os.path.join('analysis_errors', 'summary.json')) else default
    try:
        txt = BR.sec_error_analysis()
    finally:
        BR.load = orig
    assert 'Bad-case analysis' in txt and 'ambiguous_context' in txt, txt
    assert 'rare-target basis' in txt and 'empty train_ids' in txt, txt
    assert 'ckpt.pt' in txt and 'validation only' in txt, txt
    assert 'read the context' in txt and 'Whole-split' in txt, txt
    assert 'the surrounding words' in txt, txt
    rec2 = dict(rec, checkpoint='/tmp/y/ckpt2.pt')
    orig_paths = BR._error_analysis_paths
    BR._error_analysis_paths = lambda: ['/a/summary.json', '/b/summary.json']
    BR.load = lambda path, default=None: {'/a/summary.json': rec, '/b/summary.json': rec2}.get(path, default)
    try:
        txt2 = BR.sec_error_analysis()
    finally:
        BR._error_analysis_paths = orig_paths
    assert 'ckpt.pt' in txt2 and 'ckpt2.pt' in txt2 and '###' in txt2, txt2
    # A valid-JSON summary with malformed collection fields must not crash the
    # report builder (the old code sliced/`.get`ed them unconditionally).
    bad = dict(rec, worst_tokens={'not': 'a list'}, cause_guide=1,
               cause_counts={'ambiguous_context': 2})
    BR.load = lambda path, default=None: bad
    try:
        txt3 = BR.sec_error_analysis()
    finally:
        BR.load = orig
    assert 'Bad-case analysis' in txt3 and 'ckpt.pt' in txt3, txt3


def test_report_training_health_actions():
    rec = {'ppl': 51.0, 'eval_subset_rows': 16,
           'curve_health': {'status': 'overfitting', 'best_val_ppl': 48.0,
                            'best_step': 500, 'final_minus_best': 3.0,
                            'steps_since_best': 500, 'train_ce_slope_per_step': -1e-4,
                            'val_ce_slope_per_step': 2e-5, 'gap_last': 1.2,
                            'gap_slope_per_step': 3e-5, 'n_eval_points': 7,
                            'trend_reliable': True,
                            'actions': ['add regularization', 'report the best checkpoint']},
           'health_history': [{'val_ci95_low': 47.0, 'val_ci95_high': 49.0}]}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec}}
    try:
        txt = BR.sec_training_health()
    finally:
        BR._result_summaries = orig
    assert 'steps since best' in txt and 'add regularization' in txt, txt
    assert '500' in txt and 'gap slope' in txt, txt
    # The number of curve points is distinct from the number of documents per
    # eval; a 16-doc subset at 7 eval points must not be reported as "16 rows".
    assert 'eval points' in txt and 'eval docs' in txt, txt
    assert '| 7 | 16 | True |' in txt, txt
    # Malformed history/actions collections must not crash the section.
    bad = dict(rec, health_history=1, curve_health=dict(rec['curve_health'], actions='one action'))
    BR._result_summaries = lambda: {'results_unit': {'b::s0': bad}}
    try:
        txt2 = BR.sec_training_health()
    finally:
        BR._result_summaries = orig
    assert 'one action' in txt2 and 'n/a' in txt2, txt2


def test_report_early_stop_section():
    rec = {'steps': 100, 'steps_completed': 40, 'tokens_seen': 400,
           'early_stop': {'stopped': True, 'best_val_ppl': 48.0,
                          'best_step': 30, 'reason': 'patience reached'}}
    normal = {'steps': 100, 'steps_completed': 100,
              'early_stop': {'stopped': False, 'reason': 'patience not reached'}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec, 'b::s0': normal}}
    try:
        txt = BR.sec_early_stop()
    finally:
        BR._result_summaries = orig
    assert 'Early stop' in txt and '40' in txt and '100' in txt, txt
    assert 'patience reached' in txt and 'True' in txt, txt
    assert 'b::s0' not in txt, txt
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'b::s0': normal}}
    try:
        assert BR.sec_early_stop() == ''
    finally:
        BR._result_summaries = orig


def test_report_efficiency_groups_are_separate():
    def _eff(n, lat):
        return {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': n,
                'decode_has_kv_cache': False, 'prefill_ms_p50': lat,
                'peak_mem_allocated_mb': 10.0}
    summ = {'results_unit': {
        'a::s0': {'variant': 'a', 'ppl': 50.0, 'params': 1000, 'tokens_seen': 100,
                  'train_tokens_per_s': 123.4, 'efficiency': _eff(512, 10.0)},
        'b::s0': {'variant': 'b', 'ppl': 40.0, 'params': 1000, 'tokens_seen': 100,
                  'efficiency': _eff(4096, 5.0)},
        'c::s0': {'variant': 'c', 'ppl': 45.0, 'params': 1000, 'tokens_seen': 100,
                  'efficiency': {'status': 'ok', 'prefill_ms_p50': 1.0}}}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: summ
    try:
        txt = BR.sec_efficiency()
    finally:
        BR._result_summaries = orig
    assert '512 tokens' in txt and '4096 tokens' in txt, txt
    assert 'excluded from every frontier' in txt and '`c::s0`' in txt, txt
    assert '123.4' in txt, txt
    # The two token counts are separate group headings, not one merged frontier.
    assert txt.count('device `cuda:0`') == 2, txt


def test_report_efficiency_cost_sentence_uses_correct_direction():
    def _eff(ms):
        return {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
                'decode_has_kv_cache': False, 'prefill_ms_p50': ms,
                'peak_mem_allocated_mb': 10.0}
    summ = {'results_unit': {}}
    for i, (p, ms) in enumerate(((48.0, 30.0), (48.1, 30.0), (47.9, 30.0))):
        summ['results_unit'][f'best::s{i}'] = {
            'variant': 'best', 'ppl': p, 'params': 1000, 'tokens_seen': 100,
            'efficiency': _eff(ms)}
    for i, (p, ms) in enumerate(((50.0, 10.0), (50.1, 10.0), (49.9, 10.0))):
        summ['results_unit'][f'cheap::s{i}'] = {
            'variant': 'cheap', 'ppl': p, 'params': 1000, 'tokens_seen': 100,
            'efficiency': _eff(ms)}
    orig = BR._result_summaries
    BR._result_summaries = lambda: summ
    try:
        txt = BR.sec_efficiency()
    finally:
        BR._result_summaries = orig
    assert 'improves PPL by' in txt, txt
    assert 'gains -' not in txt, txt
    assert 'but is 3.0× slower' in txt, txt
    assert '4.17%' in txt, txt


def test_report_nan_section_uses_triage():
    rec = {'instability_diagnosis': {
        'status': 'instability_observed', 'primary_suspect': 'fp16_overflow',
        'checks': [{'check': 'fp16_overflow', 'status': 'observed',
                    'evidence': 'overflows=2', 'action': 'switch to bf16'}]},
        'opt_diag_summary': {'first_bad_step': 17, 'nonfinite_loss_steps': 0,
                             'nonfinite_grad_steps': 2, 'amp_scaler_overflows': 2,
                             'nan_guard': {'elements_cleaned': 0}}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec}}
    try:
        txt = BR.sec_nan_diagnostics()
    finally:
        BR._result_summaries = orig
    assert 'Non-finite' in txt and 'fp16_overflow' in txt, txt
    assert 'switch to bf16' in txt and '17' in txt, txt
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': {'opt_diag_summary': {
        'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
        'nan_guard': {'elements_cleaned': 0}}}}}
    try:
        assert BR.sec_nan_diagnostics() == ''
    finally:
        BR._result_summaries = orig


def test_audit_training_health_records():
    td = tempfile.mkdtemp(prefix='health_', dir=REPO)
    try:
        clean = {'curve_health': {'status': 'healthy'},
                 'opt_diag_summary': {'nonfinite_loss_steps': 0,
                                      'nonfinite_grad_steps': 0,
                                      'nan_guard': {'elements_cleaned': 0}},
                 'instability_diagnosis': {'status': 'clean'}}
        over = {'curve_health': {'status': 'overfitting', 'reason': 'val rising'}}
        contra = {'curve_health': {'status': 'healthy'},
                  'opt_diag_summary': {'nonfinite_loss_steps': 1,
                                       'nonfinite_grad_steps': 0,
                                       'nan_guard': {'elements_cleaned': 0}},
                  'instability_diagnosis': {'status': 'clean'}}
        legacy = {'curve_health': {'status': 'nonfinite'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': clean, 'b::s0': over}, f)
        r = A.audit_training_health(outdirs=[td])
        assert r['status'] == 'warnings' and 'b::s0' in str(r['warnings']), r
        assert 'a::s0' not in str(r['warnings']), r
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': clean, 'c::s0': contra}, f)
        r2 = A.audit_training_health(outdirs=[td])
        assert r2['status'] == 'fail' and any('clean but non-finite' in x for x in r2['errors']), r2
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'d::s0': legacy}, f)
        r3 = A.audit_training_health(outdirs=[td])
        assert r3['status'] == 'warnings' and any('without triage' in x for x in r3['warnings']), r3
        # A malformed event count is unknown, not zero: treating it as zero
        # would let a corrupt record pass as clean.
        malformed = {'curve_health': {'status': 'healthy'},
                     'opt_diag_summary': {'nonfinite_loss_steps': 1.5,
                                          'nonfinite_grad_steps': 0,
                                          'nan_guard': {'elements_cleaned': 0}},
                     'instability_diagnosis': {'status': 'clean'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'m::s0': malformed}, f)
        r4 = A.audit_training_health(outdirs=[td])
        assert r4['status'] == 'fail' and any('unverifiable, not zero' in x
                                              for x in r4['errors']), r4
        # A numeric string is not an event count either, even when it spells 0.
        string_count = json.loads(json.dumps(clean))
        string_count['opt_diag_summary']['nonfinite_loss_steps'] = '0'
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'sc::s0': string_count}, f)
        rsc = A.audit_training_health(outdirs=[td])
        assert rsc['status'] == 'fail' and any(
            'unverifiable, not zero' in x for x in rsc['errors']), rsc
        # The cached status is not evidence on its own: a record claiming
        # healthy while its own curve is overfitting must be surfaced, and a
        # masked non-finite curve point must fail closed.
        over_hist = [{'step': i * 100, 'train_ce': 6.0 - 0.3 * i,
                      'val_ce': 4.0 + 0.05 * i, 'val_ppl': math.exp(4.0 + 0.05 * i)}
                     for i in range(5)]
        fake_clean = {'curve_health': {'status': 'healthy', 'best_step': 999,
                                       'best_val_ppl': 1.0,
                                       'final_minus_best': -5.0,
                                       'steps_since_best': 777},
                      'health_history': over_hist,
                      'opt_diag_summary': {'nonfinite_loss_steps': 0,
                                           'nonfinite_grad_steps': 0,
                                           'nan_guard': {'elements_cleaned': 0}},
                      'instability_diagnosis': {'status': 'clean'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'h::s0': fake_clean}, f)
        r5 = A.audit_training_health(outdirs=[td])
        assert r5['status'] == 'warnings', r5
        assert any('contradicts the health_history' in x for x in r5['warnings']), r5
        assert any('best_step=999' in x for x in r5['warnings']), r5
        assert any('best_val_ppl=1' in x for x in r5['warnings']), r5
        assert any('final_minus_best=-5' in x for x in r5['warnings']), r5
        assert any('steps_since_best=777' in x for x in r5['warnings']), r5
        # Numeric strings are not cached curve readings.
        string_best = json.loads(json.dumps(fake_clean))
        string_best['curve_health']['best_val_ppl'] = '1.0'
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'sb::s0': string_best}, f)
        rsb = A.audit_training_health(outdirs=[td])
        assert rsb['status'] == 'warnings', rsb
        assert any('best_val_ppl is not a finite' in x
                   for x in rsb['warnings']), rsb
        nan_hist = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                    {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}]
        masked_nan = dict(fake_clean, health_history=nan_hist)
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'n::s0': masked_nan}, f)
        r6 = A.audit_training_health(outdirs=[td])
        assert r6['status'] == 'fail', r6
        assert any('contradicts the health_history' in x for x in r6['errors']), r6
        # Counter/log disagreement and a first_bad_step without any event
        # are record-integrity failures, not clean runs.
        for label, rec2 in (
                ('v::s0', {'curve_health': {'status': 'healthy'},
                           'opt_diag_summary': {
                               'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                               'nan_guard': {'elements_cleaned': 0},
                               'events': [{'kind': 'loss', 'loss': float('nan')}]}}),
                ('w::s0', {'curve_health': {'status': 'healthy'},
                           'opt_diag_summary': {
                               'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                               'nan_guard': {'elements_cleaned': 0},
                               'first_bad_step': 12}})):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({label: rec2}, f)
            ry = A.audit_training_health(outdirs=[td])
            if label == 'v::s0':
                assert ry['status'] == 'fail' and any(
                    'nonfinite_loss_steps=0' in x for x in ry['errors']), ry
            else:
                assert ry['status'] == 'warnings' and any(
                    'first_bad_step=12' in x for x in ry['warnings']), ry
        # Unknown status names and a healthy claim with an explicitly empty
        # curve are not verified health.
        for label, rec2 in (('u::s0', {'curve_health': {'status': 'fine'},
                                       'opt_diag_summary': {}}),
                            ('e::s0', {'curve_health': {'status': 'healthy'},
                                       'health_history': [],
                                       'opt_diag_summary': {}})):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({label: rec2}, f)
            rx = A.audit_training_health(outdirs=[td])
            assert rx['status'] == 'warnings', (label, rx)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_release_readiness_audit():
    td = tempfile.mkdtemp(prefix='rel_', dir=REPO)
    old_repo = A.REPO
    try:
        A.REPO = td
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n| _none yet_ | — | — | — | — | — | — |\n')
        r = A.audit_release_readiness()
        assert r['status'] == 'no_release' and r['complete_rows'] == 0, r
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n'
                    '| v13 | 2026-10-20 | abcdef1 | https://example.org/release_v13.tar.gz | ' + 'a' * 64 + ' | ' + 'b' * 64 + ' | https://example.org/verify.txt |\n')
        r2 = A.audit_release_readiness()
        assert r2['status'] == 'ok' and r2['complete_rows'] == 1 and not r2['problems'], r2
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n'
                    '| v13 | 2026-10-20 | — | — | — | — | — |\n')
        r3 = A.audit_release_readiness()
        assert r3['status'] == 'incomplete' and r3['problems'], r3
    finally:
        A.REPO = old_repo
        shutil.rmtree(td, ignore_errors=True)


def test_selection_attribution_contract():
    td = tempfile.mkdtemp(prefix='sel_', dir=REPO)
    old_repo = A.REPO
    try:
        A.REPO = td
        r = A.audit_selection()
        assert r['status'] == 'not_verifiable_locally', r
        os.makedirs(os.path.join(td, 'analysis_mechanism'))
        cell = {'arm': 'csa_fixed', 'eval_len': 4096, 'k': 32, 'n_blocks': 1024, 'n_queries': 64,
                'recall_at_k': 0.6, 'mass_covered': 0.8, 'random_baseline': 0.03125,
                'per_seed': {str(i): {'mass_covered': 0.8 - 0.01 * i, 'random_baseline': 0.03125} for i in range(6)}}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [cell]}, f)
        r2 = A.audit_selection()
        assert r2['status'] == 'ok' and r2['cells'][0]['mass_excess'] > 0, r2
        assert r2['cells'][0]['claim_allowed'] is True, r2['cells'][0]
        bad = dict(cell)
        bad.pop('recall_at_k')
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad]}, f)
        r3 = A.audit_selection()
        assert r3['status'] == 'fail' and r3['problems'], r3
        # Block/query counts are exact geometry.  int(1.5) would silently
        # certify a different k/n_blocks than the artifact declares.
        frac = dict(cell, k=1.5)
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [frac]}, f)
        rf = A.audit_selection()
        assert rf['status'] == 'fail' and any('exact integers' in x for x in rf['problems']), rf
        # per_seed is part of the declared schema: without it the seed-level
        # claim cannot be checked and must be refused structurally.
        bad2 = dict(cell)
        bad2.pop('per_seed')
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad2]}, f)
        r4 = A.audit_selection()
        assert r4['status'] == 'fail' and any('per_seed' in x for x in r4['problems']), r4
        # Malformed seed rows must not inflate the n>=6 claim gate.
        bad3 = dict(cell)
        bad3['per_seed'] = {str(i): ({'mass_covered': 0.8, 'random_baseline': 0.03} if i < 2 else {})
                            for i in range(6)}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad3]}, f)
        r5 = A.audit_selection()
        assert r5['cells'][0]['claim_allowed'] is False and r5['cells'][0]['n_seeds_valid'] == 2, r5
        # Non-integer and duplicate-normalised seed keys must not count as
        # independent seeds for the n>=6 claim gate.
        bad4 = dict(cell)
        good = {'mass_covered': 0.8, 'random_baseline': 0.03}
        bad4['per_seed'] = {'0': good, 'bad': good, '00': good}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad4]}, f)
        r6 = A.audit_selection()
        assert r6['cells'][0]['n_seeds_valid'] == 1, r6
        assert any('per_seed' in x for x in r6['problems']), r6
        # A present-but-NaN/out-of-range per-seed reading is malformed, not a
        # silent drop that leaves the reported n larger than the tested sample.
        nan_cell = dict(cell)
        nan_cell['per_seed'] = {str(i): {'mass_covered': 0.8, 'random_baseline': 0.03}
                                for i in range(6)}
        nan_cell['per_seed']['2'] = {'mass_covered': float('nan'), 'random_baseline': 0.03}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [nan_cell]}, f)
        r7 = A.audit_selection()
        assert r7['status'] == 'fail' and any('per_seed' in x for x in r7['problems']), r7
        assert r7['cells'][0]['n_seeds_valid'] == 5, r7['cells'][0]
        # Three displayed cells with the same p=2/64 are a family, not three
        # independent claims: Holm/3 pushes p_holm above alpha and none may
        # unlock the selection-mechanism claim.
        multi = [dict(cell, arm=f'csa_arm{i}') for i in range(3)]
        with open(os.path.join(td, 'analysis_mechanism',
                               'selection_attribution.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'cells': multi}, f)
        r8 = A.audit_selection()
        assert r8['status'] == 'ok', r8
        assert r8['claim_family']['m'] == 3, r8['claim_family']
        for c in r8['cells']:
            assert abs(c['seed_test']['p_exact_signflip'] - 0.03125) < 1e-12, c
            assert c['p_holm'] >= 0.09, c
            assert c['claim_allowed'] is False, c
    finally:
        A.REPO = old_repo
        shutil.rmtree(td, ignore_errors=True)


def test_audit_length_missing_truncated_flag():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095}}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail' and any('truncated' in x for x in r['problems']), r
        # A direct caller may pass np.bool_(False); the identity check used to
        # skip the n_tok == n_seq*(L-1) contract for that valid spelling.
        _old_load = A._load
        try:
            A._load = lambda _p: {'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 17, 'eval_span': 4095,
                'truncated': np.bool_(False), 'n_seq': 16}}}}
            rb = A.audit_length(p)
        finally:
            A._load = _old_load
        assert rb['status'] == 'fail' and any('n_seq*(L-1)' in x for x in rb['problems']), rb
        # A full-length cell must report the SCORED span L-1, not the input
        # width L: otherwise every arm can agree on an eval_span that its own
        # token denominator contradicts.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4096,
                'truncated': False, 'n_seq': 16}}}}, f)
        rs = A.audit_length(p)
        assert rs['status'] == 'fail' and any('eval_span=4096' in x for x in rs['problems']), rs
        # A truncated cell's token count is n_seq * its scored span, not
        # n_seq * (L-1); a silently shortened slice must be named.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 2048,
                'truncated': True, 'n_seq': 16}}}}, f)
        rt = A.audit_length(p)
        assert rt['status'] == 'fail' and any('n_seq*eval_span' in x for x in rt['problems']), rt
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_audit_length_rejects_zero_target_cells():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        for _cell in ({'ppl': 30.0, 'n_tok': 0, 'eval_span': 4096,
                       'truncated': False, 'n_seq': 0},
                      {'ppl': 30.0, 'n_tok': 0, 'eval_span': 0,
                       'truncated': True, 'n_seq': 16}):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'by_len': {'4096': _cell}}}, f)
            r = A.audit_length(p)
            assert r['status'] == 'fail', r
            assert any(('n_tok' in x and 'positive' in x)
                       or ('n_seq' in x and 'positive' in x) for x in r['problems']), r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_audit_length_missing_n_seq_flag():
    # Same fail-closed rule as a missing truncation marker: without n_seq the
    # audit cannot prove the arms scored the same document rows.
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False}}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail' and any('n_seq' in x for x in r['problems']), r
        assert r['by_length']['4096']['same_n_seq_across_records'] is False, r
        # One record with n_seq must not mask another record that omits it.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False,
                                                    'n_seq': 16}}},
                       'b::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail' and r2['by_length']['4096']['same_n_seq_across_records'] is False, r2
        # Two row-count fields (n_seq and len(ppl_per_seq)) must describe the
        # same rows; silently preferring one would hide a truncated vector.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095,
                'truncated': False, 'n_seq': 16,
                'ppl_per_seq': [30.0] * 8}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail' and any('disagrees' in x for x in r3['problems']), r3
        # An invalid explicit n_seq is a problem even when a per-document
        # vector supplies a plausible length.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095,
                'truncated': False, 'n_seq': True,
                'ppl_per_seq': [30.0] * 16}}}}, f)
        r4 = A.audit_length(p)
        assert r4['status'] == 'fail' and any('non-integer' in x for x in r4['problems']), r4
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_audit_length_rejects_unverifiable_counts():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        # A non-numeric n_tok/eval_span, a non-integer length key and a
        # malformed cell must all fail closed instead of being skipped.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {
                '4096': {'ppl': 30.0, 'n_tok': 'lots', 'eval_span': 4095,
                         'truncated': False, 'n_seq': 16},
                'oops': {'ppl': 1.0},
                '512': 'not-a-cell'}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail', r
        assert any('n_tok' in x for x in r['problems']), r
        assert any('integer length' in x for x in r['problems']), r
        assert any('not a JSON object' in x for x in r['problems']), r
        # Booleans are not counts either.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': True, 'eval_span': 4095,
                'truncated': False, 'n_seq': True}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail', r2
        # A numeric string is not an exact token count, even if it equals the
        # number the audit would otherwise accept.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': str(16 * 4095), 'eval_span': 4095,
                'truncated': False, 'n_seq': 16}}}}, f)
        r_str = A.audit_length(p)
        assert r_str['status'] == 'fail', r_str
        assert any('n_tok' in x for x in r_str['problems']), r_str
        # Non-hashable malformed cells must be a recorded failure, not an
        # uncaught TypeError from inserting them into a comparison set.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': [1, 2], 'eval_span': {'bad': 1},
                'truncated': [False], 'n_seq': 1}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail', r3
        assert any('n_tok' in x for x in r3['problems']), r3
        assert any('eval_span' in x for x in r3['problems']), r3
        assert any('truncated' in x for x in r3['problems']), r3
        # An existing-but-empty artifact must not pass the audit, and a
        # malformed by_len container must fail closed instead of raising.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        r4 = A.audit_length(p)
        assert r4['status'] == 'not_verifiable_locally', r4
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': []}}, f)
        r5 = A.audit_length(p)
        assert r5['status'] == 'fail' and any('by_len' in x for x in r5['problems']), r5
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_global_multiplicity_audit_basic():
    # Known BH case: p=[0.01, 0.02, 0.5] at m=3 -> q=[0.03, 0.03, 0.5].
    r = L.global_multiplicity_audit([('a', 0.01), ('b', 0.02), ('c', 0.5)])
    assert r['m'] == 3, r
    ent = {e['name']: e for e in r['entries']}
    assert abs(ent['a']['q_bh'] - 0.03) < 1e-12 and abs(ent['b']['q_bh'] - 0.03) < 1e-12, r
    assert abs(ent['c']['q_bh'] - 0.5) < 1e-12, r
    # 0.01: Bonferroni 0.03 < 0.05 -> globally controlled.
    assert ent['a']['global_status'] == 'globally_controlled', r
    # 0.02: BH-rejected but Bonferroni 0.06 >= 0.05 -> fdr_only.
    assert ent['b']['global_status'] == 'fdr_only', r
    # 0.5: raw >= alpha.
    assert ent['c']['global_status'] == 'not_significant', r
    assert r['summary']['n_raw_significant'] == 2 and r['summary']['n_bh_rejected'] == 2, r
    # Ten nominal hits out of a large displayed family: BH still rejects them
    # (q = 0.04) but Bonferroni (0.4) does not -> fdr_only, not family-wise.
    r = L.global_multiplicity_audit([(f'c{i}', 0.04) for i in range(10)])
    assert all(e['global_status'] == 'fdr_only' for e in r['entries']), r
    assert r['summary']['n_raw_significant'] == 10 and r['summary']['n_bh_rejected'] == 10, r
    assert r['summary']['n_bonferroni_significant'] == 0, r
    assert abs(r['entries'][0]['q_bh'] - 0.04) < 1e-12, r
    # A single nominal hit inside a large displayed family of nulls is
    # family_only: raw p < alpha but the cross-panel FDR view does not reject.
    r = L.global_multiplicity_audit([(f'c{i}', (0.04 if i == 0 else 0.9)) for i in range(100)])
    ent = {e['name']: e for e in r['entries']}
    assert ent['c0']['global_status'] == 'family_only', ent['c0']
    assert ent['c0']['bh_rejected'] is False and ent['c0']['q_bh'] >= 0.05, ent['c0']
    assert all(e['global_status'] == 'not_significant' for k, e in ent.items() if k != 'c0'), r
    assert r['summary']['n_family_only'] == 1, r
    # Family labels ride along for display and never change the decision.
    r = L.global_multiplicity_audit([('a', 0.01, 'P0R'), ('b', 0.02, 'P1T')])
    assert {e['family'] for e in r['entries']} == {'P0R', 'P1T'}, r
    r = L.global_multiplicity_audit([('a', 0.01), ('b', 0.02)], families=['P0R', 'P1T'])
    assert {e['family'] for e in r['entries']} == {'P0R', 'P1T'}, r


def test_global_multiplicity_audit_fail_closed():
    # Duplicate names double-count one comparison: hard error.
    try:
        L.global_multiplicity_audit([('a', 0.01), ('a', 0.02)])
    except ValueError as e:
        assert 'duplicate' in str(e).lower(), e
    else:
        raise AssertionError('duplicate comparison names must be refused')
    # Invalid alpha is refused, not defaulted.
    for bad in (0.0, 1.0, -0.1, True, None, 'x'):
        try:
            L.global_multiplicity_audit([('a', 0.01)], alpha=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'alpha={bad!r} must be refused')
    # Empty family is a legal explicit state.
    r = L.global_multiplicity_audit([])
    assert r['m'] == 0 and r['summary']['n_raw_significant'] == 0 and r['entries'] == [], r
    # Missing/bool/out-of-range p keeps the slot (fail-closed family size).
    r = L.global_multiplicity_audit([('ok', 0.02), ('none', None), ('bool', True), ('neg', -0.1), ('big', 1.7), ('nan', float('nan'))])
    assert r['m'] == 6, r
    assert set(r['untestable']) == {'none', 'bool', 'neg', 'big', 'nan'}, r
    ent = {e['name']: e for e in r['entries']}
    assert all(ent[n]['global_status'] == 'untestable' for n in ('none', 'bool', 'neg', 'big', 'nan')), r
    # The valid entry is audited over the FULL displayed family (m=6), not
    # over a silently shrunk one.
    assert abs(ent['ok']['p_bonferroni'] - 0.12) < 1e-12, r
    assert r['summary']['n_untestable'] == 5, r
    # q is monotone in p and capped at 1 on random panels.
    import random as _rnd
    _rnd.seed(20261020)
    for _trial in range(30):
        m = _rnd.randint(1, 24)
        named = [(f'c{i}', _rnd.choice([None, _rnd.random()])) for i in range(m)]
        ra = L.global_multiplicity_audit(named)
        assert ra['m'] == m, (named, ra)
        pairs = sorted((e['p_raw'], e['q_bh']) for e in ra['entries'] if e['p_raw'] is not None)
        for (p1, q1), (p2, q2) in zip(pairs, pairs[1:]):
            assert p2 >= p1 and q2 >= q1 - 1e-12 and q2 <= 1.0 + 1e-12, (named, pairs)


def test_global_multiplicity_twin_consistency():
    # The report builder keeps a dependency-free local twin; the two must
    # agree exactly so the printed global table cannot drift from the library
    # the driver-side audits use.
    for named in _global_multiplicity_random_cases():
        a1 = L.global_multiplicity_audit(named, alpha=0.05)
        a2 = BR.global_multiplicity_adjust([(n, p) for n, p in named], alpha=0.05)
        assert a1['m'] == a2['m'], (named, a1, a2)
        by2 = {r['name']: r for r in a2['rows']}
        for e in a1['entries']:
            o = by2[e['name']]
            if e['p_raw'] is None:
                assert o['p_raw'] is None and o['global_status'] == 'untestable', (named, e, o)
            else:
                assert abs(e['p_raw'] - o['p_raw']) < 1e-12, (named, e, o)
                assert abs(e['q_bh'] - o['q_bh']) < 1e-9, (named, e, o)
                assert abs(e['p_bonferroni'] - o['p_bonferroni']) < 1e-12, (named, e, o)
                assert e['global_status'] == o['global_status'], (named, e, o)
        assert a1['summary']['n_raw_significant'] == a2['n_raw_significant'], (named, a1, a2)
        assert a1['summary']['n_bh_rejected'] == a2['n_bh_rejected'], (named, a1, a2)
        assert a1['summary']['n_bonferroni_significant'] == a2['n_bonferroni_significant'], (named, a1, a2)
    # NumPy booleans are flags too; the local twin must not turn them into 0/1.
    for _p in (np.bool_(True), np.bool_(False)):
        a1 = L.global_multiplicity_audit([('b', _p)])
        a2 = BR.global_multiplicity_adjust([('b', _p)])
        assert a1['entries'][0]['p_raw'] is None, a1
        assert a2['rows'][0]['p_raw'] is None, a2


def test_build_report_global_multiplicity_section():
    BR._reset_claim_registry()
    try:
        st = {'n': 6, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.01, 'p_holm': 0.02}
        ok = BR._sig_gate(st, label='P0X demo labelled contrast', family='P0X')
        assert ok is True, st
        st2 = {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25, 'p_holm': 0.5}
        assert BR._sig_gate(st2, label='P0Y demo underpowered contrast', family='P0Y') is False
        for _bad_alpha in (True, 0, 1, -0.1, 1.5, float('nan'), 'x'):
            assert BR._sig_gate(st, alpha=_bad_alpha) is False, _bad_alpha
        # An unlabeled gate decision is counted and flagged, never lost.
        assert BR._sig_gate(st) is True
        assert BR._UNREGISTERED_GATES == 1, BR._UNREGISTERED_GATES
        txt = BR.sec_global_multiplicity()
        assert 'P0X demo labelled contrast' in txt and 'P0Y demo underpowered contrast' in txt, txt
        assert 'q_BH' in txt and 'Benjamini-Hochberg' in txt or 'BH' in txt, txt
        assert 'P0X demo labelled contrast' in txt, txt
        # The globally-controlled wording rule is present so prose can cite it.
        assert '全局' in txt, txt
        # Flag line for the unregistered decision.
        assert '未带 label 注册' in txt, txt
        # Duplicate labels across sections must fail closed (family size
        # would double-count).
        try:
            BR._note_comparison('dup', 0.01)
            BR._note_comparison('dup', 0.02)
            BR.sec_global_multiplicity()
        except ValueError as e:
            assert 'duplicate' in str(e).lower(), e
        else:
            raise AssertionError('duplicate labels must be refused')
    finally:
        BR._reset_claim_registry()
    # No displayed comparisons -> explicit empty note, not a silent omission.
    txt = BR.sec_global_multiplicity()
    assert '没有显示任何带 p 值的比较' in txt, txt


def test_scale_corpus_binding_refuses_mixed_tokenizers():
    # Cross-scale effects (paired log PPL ratios) are only defined inside ONE
    # tokenisation; pooling across corpora/tokenizers is a units error, not a
    # heterogeneity finding.  The scale machinery must refuse it from the
    # provenance, not from the numeric pattern.
    pts = [{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2],
            'corpus_fp': 'fpA', 'vocab': 32768},
           {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35], 'seeds': [0, 1, 2],
            'corpus_fp': 'fpA', 'vocab': 32768}]
    m = L.meta_analyze_effects(pts)
    assert m['status'] == 'ok', m
    assert m['corpus_binding']['corpus_binding'] == 'shared', m
    # The scale points share seed identities, so the DL pooled CI's
    # independence assumption is disclosed and a paired per-seed common
    # effect is reported as the dependence-respecting sensitivity readout.
    assert m['seed_dependence'] == 'shared_seeds_paired', m
    pce = m['paired_common_effect']
    assert pce['status'] == 'ok' and pce['n_scales'] == 2, pce
    assert abs(pce['mean'] - 0.225) < 1e-12, pce
    assert pce['n'] == 3 and pce['seeds'] == [0, 1, 2], pce
    assert 'independen' in m['note'].lower(), m['note']
    # Undeclared seed identities cannot be treated as independent OR paired.
    mu = L.meta_analyze_effects([
        {'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3]},
        {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35]}])
    assert mu['seed_dependence'] == 'unattested', mu
    assert mu['paired_common_effect']['status'] == 'unattested_seed_identity', mu
    t = L.scaling_trend(pts)
    assert t.get('status') != 'refused_corpus_mismatch', t
    # Same ladder, second tier measured on a different tokenisation.
    pts_mixed = [pts[0], dict(pts[1], corpus_fp='fpB')]
    m = L.meta_analyze_effects(pts_mixed)
    assert m['status'] == 'refused_corpus_mismatch' and m['poolable'] is False, m
    assert 'tokenisation' in m['note'], m
    assert len(m['per_point']) == 2, 'per-scale rows stay; only pooling is refused'
    t = L.scaling_trend(pts_mixed)
    assert t['status'] == 'refused_corpus_mismatch' and 'slope' not in t, t
    # A vocab difference alone is enough to refuse (PPL units differ).
    m = L.meta_analyze_effects([dict(pts[0], corpus_fp='fpA', vocab=32768),
                                dict(pts[1], corpus_fp='fpA', vocab=8000)])
    assert m['status'] == 'refused_corpus_mismatch', m
    # Missing fingerprints are disclosed as unverifiable, never as shared.
    m = L.meta_analyze_effects([{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2]},
                                {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35], 'seeds': [0, 1, 2]}])
    assert m['corpus_binding']['corpus_binding'] == 'unverifiable', m
    assert m['status'] == 'ok', 'legacy points without fingerprints stay analysable, just unattested'
    # A shared corpus fingerprint does not attest the tokenizer when only one
    # point declares its vocab size; the missing width could be different.
    partial_vocab = [dict(pts[0], vocab=32768), {k: v for k, v in pts[1].items()
                                                if k != 'vocab'}]
    m = L.meta_analyze_effects(partial_vocab)
    assert m['corpus_binding']['corpus_binding'] == 'unverifiable', m
    assert m['corpus_binding']['n_vocab_declared'] == 1, m
    assert 'incomplete' in m['corpus_binding']['refusal_reason'], m
    # The ladder audit fails CLOSED on a corpus/tokenizer mismatch.
    def _tier(name, corpus_fp=None, vocab=None):
        cfg = {'steps': 100, 'batch_size': 4, 'seq_len': 512, 'n_train_tokens': 204800,
               'lr': 3e-4, 'warmup': 10, 'seeds': [0, 1]}
        if corpus_fp is not None:
            cfg['corpus_fp'] = corpus_fp
        if vocab is not None:
            cfg['vocab'] = vocab
        return {'name': name, 'cfg': cfg, 'params': 1000}
    a = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpA', 8000)])
    assert a['status'] in ('ok', 'warn') and a['corpus_binding']['corpus_fp'] == 'shared', a
    b = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpB', 8000)])
    assert b['status'] == 'fail' and any('corpus fingerprint' in x for x in b['problems']), b
    c = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpA', 32000)])
    assert c['status'] == 'fail' and any('vocab size' in x for x in c['problems']), c
    d = L.scale_ladder_audit([_tier('S1'), _tier('S2')])
    assert d['status'] != 'fail' and d['corpus_binding']['corpus_fp'] == 'unverifiable', d


def test_scale_point_provenance_roundtrip():
    # _scale_points carries the per-point corpus provenance through both
    # input forms; the m0 analysis path builds these from record data_version.
    # A malformed effects field must not silently delete the declared tier;
    # None remains the explicit declared-empty form.
    for _bad_eff in (1.0, True, 'abc', b'ab'):
        try:
            L._scale_points([{'label': 'Sbad', 'scale': 1,
                              'effects': _bad_eff, 'seeds': [0]}])
        except ValueError as e:
            assert 'effects must be an iterable' in str(e), (_bad_eff, e)
        else:
            raise AssertionError(f'malformed effects {_bad_eff!r} was not refused')
    pts = L._scale_points([
        {'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2], 'seeds': [0, 1],
         'corpus_fp': 'fpA', 'vocab': 8000},
        (2, [0.3, 0.4], [0, 1], 'fpB'),
        {'label': 'S3', 'scale': 3, 'effects': [0.5, 0.6], 'seeds': [0, 1]},
    ])
    assert pts[0]['corpus_fp'] == 'fpA' and pts[0]['vocab'] == 8000, pts
    assert pts[1]['corpus_fp'] == 'fpB' and pts[1]['vocab'] is None, pts
    assert pts[2]['corpus_fp'] is None, pts

