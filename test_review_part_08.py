"""CPU review checks (split group 08; t_overfit_loss_contract_and_ignore_index .. t_kernel_det_scatter_precision_and_nonfinite_topk)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_overfit_loss_contract_and_ignore_index():
    ids = np.arange(4096, dtype=np.int64) % 256
    r = L.overfit_sanity_check('full', seed=0, steps=120, d=64, n_layers=2,
                               n_heads=4, d_head=16, seq_len=32, batch_size=2,
                               vocab=256)
    assert r['passed'] is True, r
    assert r['loss_contract']['initial']['passed'] is True, r['loss_contract']
    assert r['loss_contract']['final']['passed'] is True, r['loss_contract']
    assert r['pass_criteria']['loss_reduction_matches_mean_ce'] is True, r['pass_criteria']
    assert r['code'] == L.code_stamp() and r['device'] == 'cpu', r
    r2 = L.overfit_sanity_check('full', seed=0, steps=20, d=32, n_layers=1,
                                n_heads=4, d_head=8, seq_len=16, batch_size=2,
                                vocab=256, train_ids=ids, ignore_index=-100,
                                device='cpu')
    assert r2['ignore_index'] == -100, r2
    assert r2['label_contract']['ignore_index'] == -100, r2['label_contract']
    assert r2['loss_contract']['initial']['passed'] is True, r2['loss_contract']
    for bad in (0, -1, True, 1.5):
        try:
            L.overfit_sanity_check('full', steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'overfit_sanity_check accepted steps={bad!r}')
    try:
        L.overfit_sanity_check('full', ignore_index=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError('overfit_sanity_check accepted a non-integral ignore_index')


def test_ce_loss_default_ignore_matches_pytorch_reduction():
    ids = torch.arange(64, dtype=torch.long).reshape(2, 32) % 32
    tg = torch.arange(1, 65, dtype=torch.long).reshape(2, 32) % 32
    tg = tg.clone()
    tg[0, :8] = -100
    model = L.SmallGPT(32, 32, 1, 4, 8, 32, L.make_layer_cfgs(1, 'full'), mlp_ratio=2)
    loss = float(model.ce_loss(ids, tg, 32).detach())
    with torch.no_grad():
        lg = model.head(model.trunk(ids))
        ref = float(torch.nn.functional.cross_entropy(
            lg.reshape(-1, 32), tg.reshape(-1), reduction='mean'))
    assert math.isclose(loss, ref, rel_tol=1e-5, abs_tol=1e-6), (loss, ref)


def test_repro_probe_config_binds_device_and_lr():
    cpu = L._repro_guard_base(seed=0, lr=0.003, device=torch.device('cpu'))
    cuda = L._repro_guard_base(seed=0, lr=0.003, device=torch.device('cuda'))
    low_lr = L._repro_guard_base(seed=0, lr=0.001, device=torch.device('cpu'))
    assert cpu != cuda, 'the reproducibility probe cache must bind the device'
    assert cpu != low_lr, 'the reproducibility probe cache must bind the learning rate'
    r = L.reproducibility_check('full', seed=0, steps=2, lr=0.001,
                                device=torch.device('cpu'))
    assert r['passed'] is True and r['lr'] == 0.001 and r['device'] == 'cpu', r
    rs = L.reproducibility_check('full', seed=0, steps=2, device='cpu')
    assert rs['passed'] is True and rs['device'] == 'cpu', rs
    assert rs['code'] == L.code_stamp(), rs
    for bad in (0, -1, True, 1.5):
        try:
            L.reproducibility_check('full', steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_check accepted steps={bad!r}')
    for bad_tol in (-1.0, float('nan')):
        try:
            L.reproducibility_check('full', tol=bad_tol)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_check accepted tol={bad_tol!r}')


def test_seed_contract_cuda_device_inventory():
    r = L.seed_contract_check(seed=5)
    assert r['passed'] is True and r['code'] == L.code_stamp(), r
    if not torch.cuda.is_available():
        assert r['checks']['cuda_devices_seeded'] is None, r['checks']
    else:
        assert r['checks']['cuda_devices_seeded'] is True, r['checks']


def test_manifest_missing_curve_ref_is_incomplete():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        hp = L.run_hyperparameters(cfg, seed=0, d=64, n_layers=2, n_heads=4,
                                   d_head=16, vocab=97, mlp_ratio=4.0,
                                   data_version=prov['data']['data_version'])
        cell = {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                'code_stamp': L.code_stamp(), 'hyperparams': hp,
                'data_version': prov['data']['data_version'], 'rng_fp': 'r',
                'curve_summary': {'n_points': 2},
                'curve_ref': 'curves/missing.json'}
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['tracking_integrity']['missing_curve_refs'] == ['full::seed0'], e['tracking_integrity']
        assert e['tracking_integrity']['complete'] is False, e['tracking_integrity']
        os.makedirs(os.path.join(d, 'curves'), exist_ok=True)
        with open(os.path.join(d, 'curves', 'missing.json'), 'w', encoding='utf-8') as fh:
            json.dump({'losses': [3.0, 2.0]}, fh)
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['tracking_integrity']['missing_curve_refs'] == [], e['tracking_integrity']
        assert e['tracking_integrity']['complete'] is True, e['tracking_integrity']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_environment_snapshot_runtime_identity():
    env = L.environment_snapshot()
    assert env.get('hostname'), env
    assert env.get('process_id') == os.getpid(), env
    assert env.get('working_directory') == os.getcwd(), env


def test_ledger_curve_is_bounded():
    small = [1.0, 2.0, 3.0]
    got, n, down = L._ledger_curve(small)
    assert got == small and n == 3 and down is False, (got, n, down)
    vals = [float(i % 17) for i in range(50000)]
    vals[12345] = float('nan')
    vals[23456] = 1e9
    got, n, down = L._ledger_curve(vals, max_points=20000)
    assert n == 50000 and down is True and len(got) <= 20000, (n, down, len(got))
    assert any(isinstance(v, float) and math.isnan(v) for v in got), 'the ledger reduction dropped the NaN marker'
    assert 1e9 in got, 'the ledger reduction dropped the spike'
    pts = [[i, float(i)] for i in range(300)]
    pts[201][1] = float('nan')
    pg, pn, pd = L._ledger_curve_points(pts, max_points=100)
    assert pn == 300 and pd is True and len(pg) <= 100, (pn, pd, len(pg))
    assert any(int(a) == 201 and math.isnan(b) for a, b in pg), 'the point reduction dropped the NaN point'


def test_batch_contract_check_validation():
    ids = np.arange(5000, dtype=np.int64) % 997
    ok = L.batch_contract_check(ids, 32, 4, 'cpu', seed=7, vocab=997)
    assert ok['passed'] is True and ok['device'] == 'cpu', ok
    assert ok['code'] == L.code_stamp(), ok
    for bad in (1.5, True, -1, L.MAX_SEED + 1):
        try:
            L.batch_contract_check(ids, 32, 4, 'cpu', seed=bad, vocab=997)
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_contract_check accepted seed={bad!r}')
    for kw in ({'n_batches': 0}, {'n_batches': 1.5}, {'vocab': 0},
               {'seq_len': 0}, {'batch_size': True}):
        args = dict(seq_len=32, batch_size=4, seed=7, vocab=997, device='cpu')
        args.update(kw)
        try:
            L.batch_contract_check(ids, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_contract_check accepted {kw!r}')


def test_manifest_rejects_mismatched_cell_identity():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        hp = L.run_hyperparameters(cfg, seed=0, d=64, n_layers=2, n_heads=4,
                                   d_head=16, vocab=97, mlp_ratio=4.0,
                                   data_version=prov['data']['data_version'])
        cell = {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                'code_stamp': L.code_stamp(), 'hyperparams': hp,
                'data_version': 'DIFFERENT-DATA', 'rng_fp': 'r',
                'run_cfg': 'DIFFERENT-CFG',
                'curve_summary': {'n_points': 2}}
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        integ = e['tracking_integrity']
        assert integ['run_cfg_mismatch'] == ['full::seed0'], integ
        assert integ['data_version_mismatch'] == ['full::seed0'], integ
        assert integ['complete'] is False, integ
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_determinism_label_cpu_strict():
    old_env = os.environ.get('CSA_DETERMINISTIC')
    old_cudnn = torch.backends.cudnn.deterministic
    old_bench = torch.backends.cudnn.benchmark
    old_strict = torch.are_deterministic_algorithms_enabled()
    try:
        os.environ['CSA_DETERMINISTIC'] = '1'
        L._pin_cuda_determinism()
        assert L.determinism_label() in ('strict', 'warn'), L.determinism_label()
        if not torch.cuda.is_available():
            assert L.determinism_label() == 'strict', L.determinism_label()
    finally:
        if old_env is None:
            os.environ.pop('CSA_DETERMINISTIC', None)
        else:
            os.environ['CSA_DETERMINISTIC'] = old_env
        torch.backends.cudnn.deterministic = old_cudnn
        torch.backends.cudnn.benchmark = old_bench
        if not old_strict:
            torch.use_deterministic_algorithms(False)


def test_target_encoder_keeps_duplicate_rows_in_one_fold():
    # Exact duplicate feature rows are one observation/annotation cluster.  If
    # the row-OOF fold assignment splits them, changing one duplicate's label
    # changes the encoding of its twin: the twin is encoded from its own
    # duplicate label.  All duplicates must therefore share a fold.
    import data_guard as DG
    X = np.array([['a'], ['a'], ['b'], ['b']], dtype=object)
    y = np.array([1.0, 1.0, 2.0, 2.0])
    for seed, n_folds in ((0, 3), (0, 2), (5, 4)):
        oof = DG.LeakageSafeTargetEncoder(
            smoothing=0.0, n_folds=n_folds, seed=seed).fit_transform(X, y)
        y2 = y.copy()
        y2[0] += 1000.0
        oof2 = DG.LeakageSafeTargetEncoder(
            smoothing=0.0, n_folds=n_folds, seed=seed).fit_transform(X, y2)
        assert np.allclose(oof2[1], oof[1]), (seed, n_folds, oof, oof2)
        assert np.allclose(oof2[0], oof[0]), (seed, n_folds, oof, oof2)
    # A table with no duplicates keeps the historical per-row fold assignment.
    uniq = np.array([['a'], ['b'], ['c'], ['d']], dtype=object)
    old = DG.LeakageSafeTargetEncoder(n_folds=2, seed=7)
    old_folds = old._row_folds(4)
    new_folds = old._row_folds(4, keys=DG._row_keys(uniq))
    assert np.array_equal(old_folds, new_folds)


def test_missing_group_ids_are_one_identity():
    # None, NaN and NaT are three spellings of a missing group id.  Treating
    # them as different groups lets the same missing group straddle splits, and
    # a real tuple id spelled like the old tuple sentinel could impersonate it.
    import data_guard as DG
    missing = [None, float('nan'), np.float64('nan'),
               np.datetime64('NaT', 'ns'), np.timedelta64('NaT', 'ns')]
    keys = {DG._group_key(v) for v in missing}
    assert len(keys) == 1, keys
    # The historical tuple sentinel is kept for compatibility, but the old
    # NaT/None-specific spellings no longer create extra missing groups.
    assert DG._group_key(('__nat__',)) not in keys
    assert DG._group_key(('__missing__',)) not in keys
    g = np.asarray([None, np.float64('nan')], dtype=object)
    try:
        DG.assert_no_split_leakage([0], [1], np.array([], dtype=np.int64),
                                   groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('a missing group split across sides must fail')
    # A single missing group cannot be silently cut into populated folds.
    g4 = np.asarray([None, np.float64('nan'), np.datetime64('NaT', 'ns'),
                     'x'], dtype=object)
    try:
        DG.grouped_chronological_split_indices(4, g4, 0.5, 0.25, 0.25)
    except ValueError as e:
        assert 'no rows for the requested' in str(e), e
    else:
        raise AssertionError('a missing group must not be cut across folds')


def test_split_and_prepare_refuses_ignored_options():
    # Cleaning options without clean_train=True were silently ignored, so a
    # caller asking to drop invalid labels or outliers got an unfitted/unchanged
    # training split.  Preprocessor options were likewise ignored when
    # preprocess=False.  Both are now hard errors.
    import data_guard as DG
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6.0)
    for bad in ({'drop_invalid_labels': True},
                {'drop_nonfinite_features': True},
                {'outlier_iqr': 1.5},
                {'target_outlier_iqr': 1.5},
                {'label_conflicts': 'keep'},
                {'label_conflicts': 'raise'}):
        try:
            DG.split_and_prepare(X, y, **bad)
        except ValueError as e:
            assert 'require clean_train=True' in str(e), (bad, e)
        else:
            raise AssertionError(f'split_and_prepare silently ignored {bad!r}')
    try:
        DG.split_and_prepare(X, preprocess=False, scale='robust')
    except ValueError as e:
        assert 'preprocess=False' in str(e), e
    else:
        raise AssertionError('preprocess=False must not ignore preprocessor options')
    try:
        DG.split_and_prepare(X, quality_kwargs={'clip_z': 2.0})
    except ValueError as e:
        assert 'quality_kwargs' in str(e), e
    else:
        raise AssertionError('quality_kwargs without an audit must be refused')
    # The same options are accepted when their stage is actually enabled.
    out = DG.split_and_prepare(X, y, clean_train=True,
                               drop_invalid_labels=True)
    assert out['cleaning']['n_invalid_label_rows_dropped'] == 0, out['cleaning']
    raw = DG.split_and_prepare(X, preprocess=False)
    assert raw['preprocessor'] is None


def test_persisted_stats_reject_silently_degraded_transforms():
    # A persisted artifact is untrusted input.  Malformed clip bounds used to
    # disable clipping, inconsistent categorical indices could write the wrong
    # one-hot slot, and non-finite target/count tables produced invalid
    # features instead of refusing to load.
    import json
    import data_guard as DG
    X = np.arange(12.0).reshape(4, 3)
    base = json.loads(json.dumps(DG.LeakageSafePreprocessor().fit(X).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(clip_lo=1.0, clip_hi=0.0),
            lambda s: s['features'][0].update(clip_lo=float('nan')),
            lambda s: s['features'][0].update(clip_hi=None),
            lambda s: s['features'][0].pop('clip_lo'),
            lambda s: s['features'][0].update(clip_hi=True),
            lambda s: s.update(n_train=True),
            lambda s: s.pop('scale')):
        bad = json.loads(json.dumps(base))
        mutate(bad)
        try:
            DG.LeakageSafePreprocessor.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed preprocessor stats accepted: {bad}')
    Xc = np.asarray([['a'], ['b'], [None]], dtype=object)
    y = np.asarray([1.0, 2.0, 3.0])
    cat = json.loads(json.dumps(
        DG.LeakageSafeCategoricalEncoder().fit(Xc).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(n_categories=99),
            lambda s: s['features'][0].update(categories='ab'),
            lambda s: s['features'][0].update(keys=['only-one']),
            lambda s: s['features'][0].update(unknown_index=0)):
        bad = json.loads(json.dumps(cat))
        mutate(bad)
        try:
            DG.LeakageSafeCategoricalEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed categorical stats accepted: {bad}')
    freq = json.loads(json.dumps(
        DG.LeakageSafeFrequencyEncoder().fit(Xc).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(n_observed=99),
            lambda s: s['features'][0]['entries'][0].update(count=-1),
            lambda s: s['features'][0]['entries'][0].update(count=0)):
        bad = json.loads(json.dumps(freq))
        mutate(bad)
        try:
            DG.LeakageSafeFrequencyEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed frequency stats accepted: {bad}')
    target = json.loads(json.dumps(
        DG.LeakageSafeTargetEncoder().fit(Xc, y).get_stats()))
    for mutate in (
            lambda s: s['features'][0]['lookup'].update({'x': float('nan')}),
            lambda s: s['features'][0].update(prior=float('inf')),
            lambda s: s['features'][0].update(n_reference_labels=99,
                                              n_reference_rows=1)):
        bad = json.loads(json.dumps(target))
        mutate(bad)
        try:
            DG.LeakageSafeTargetEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed target stats accepted: {bad}')


def test_baseline_audit_direction_and_explicit_field_fallbacks():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _art(direction, select_max=False):
        ppls = [10.0, 11.0]
        rows = [{'config': dict(c), 'ppl': v} for c, v in zip(trials, ppls)]
        best = max(rows, key=lambda r: r['ppl']) if select_max else min(
            rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': direction,
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}

    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[], final_seeds=[])
    good = L.baseline_tuning_audit(_art('lower_is_better'), protocol=protocol)
    assert good['tuning_verified'] is True, good
    # Flipping the artifact's direction makes a maximum-PPL trial look like
    # the argmin.  The protocol declares lower-is-better, so that is not a
    # tuned winner.
    flipped = L.baseline_tuning_audit(_art('higher_is_better', select_max=True),
                                      protocol=protocol)
    assert flipped['tuning_verified'] is False, flipped
    assert any('differs from the declared' in p or 'incompatible with PPL' in p
               for p in flipped['problems']), flipped
    # An explicitly empty/blank field is malformed evidence, not an absent key
    # that may inherit the protocol default.
    blank = _art('lower_is_better')
    blank['selection_metric'] = ''
    rb = L.baseline_tuning_audit(blank, protocol=protocol)
    assert rb['tuning_verified'] is False and any(
        'not a metric name' in p for p in rb['problems']), rb
    blank_split = _art('lower_is_better')
    blank_split['selection_split'] = ''
    rs = L.baseline_tuning_audit(blank_split, protocol=protocol)
    assert rs['tuning_verified'] is False and any(
        'validation' in p for p in rs['problems']), rs
    # A non-object protocol is malformed evidence, not an AttributeError.
    rp = L.baseline_tuning_audit(_art('lower_is_better'), protocol=['not', 'a', 'map'])
    assert rp['status'] == 'fail' and any(
        'protocol must be an object' in p for p in rp['problems']), rp
    # The artifact must be audited against the split it was declared under.
    # A protocol that declares test selection cannot certify a validation
    # artifact just because the artifact itself says "validation".
    rsplit = L.baseline_tuning_audit(
        _art('lower_is_better'),
        protocol=dict(protocol, selection_split='test'))
    assert rsplit['tuning_verified'] is False and any(
        'differs from the declared' in p for p in rsplit['problems']), rsplit


def test_simple_floor_rejects_undeclared_arms_and_blank_fields():
    rng = np.random.default_rng(11)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert L.simple_baseline_audit(art, protocol=proto)['baseline_verified'] is True
    # A copied extra arm can otherwise be chosen as the best floor while being
    # outside the declared equal-search design.
    extra = json.loads(json.dumps(art))
    extra['baselines']['extra'] = json.loads(json.dumps(extra['baselines']['unigram']))
    extra['baselines']['extra']['selected_metric_value'] = 1e-9
    for _t in extra['baselines']['extra']['trials']:
        _t['ppl'] = 1e-9
    extra['best_baseline'] = 'extra'
    extra['best_ppl'] = 1e-9
    rex = L.simple_baseline_audit(extra, protocol=proto)
    assert rex['baseline_verified'] is False and any(
        'undeclared arm' in p for p in rex['problems']), rex
    blank = json.loads(json.dumps(art))
    blank['selection_metric'] = ''
    rbl = L.simple_baseline_audit(blank, protocol=proto)
    assert rbl['baseline_verified'] is False and any(
        'not a metric name' in p for p in rbl['problems']), rbl
    blank_dir = json.loads(json.dumps(art))
    blank_dir['direction'] = ''
    rbd = L.simple_baseline_audit(blank_dir, protocol=proto)
    assert rbd['baseline_verified'] is False and any(
        'direction' in p for p in rbd['problems']), rbd
    rp = L.simple_baseline_audit(art, protocol=['not', 'a', 'map'])
    assert rp['status'] == 'fail' and any(
        'protocol must be an object' in p for p in rp['problems']), rp
    bad_arms = dict(proto, arms=['not-a-map'])
    rba = L.simple_baseline_audit(art, protocol=bad_arms)
    assert rba['baseline_verified'] is False and any(
        'arms field is not an object' in p for p in rba['problems']), rba
    # The recorded fit policy must be the one the code actually implements;
    # otherwise the artifact describes a data path that never ran.
    try:
        L.simple_baseline_fit(train, val, vocab,
                              protocol=dict(proto, fit_token_policy='seeded_sample'))
    except ValueError as _e:
        assert 'deterministic_prefix' in str(_e), _e
    else:
        raise AssertionError('an unimplemented fit token policy must be refused')
    bad_policy = json.loads(json.dumps(art))
    bad_policy['data']['fit_token_policy'] = 'seeded_sample'
    rbp = L.simple_baseline_audit(bad_policy, protocol=proto)
    assert rbp['baseline_verified'] is False and any(
        'fit_token_policy' in p for p in rbp['problems']), rbp
    rsplit = L.simple_baseline_audit(
        art, protocol=dict(proto, selection_split='test'))
    assert rsplit['baseline_verified'] is False and any(
        'differs from the declared' in p for p in rsplit['problems']), rsplit
    # The reader-facing estimand label cannot describe a different sample
    # from the nested rows the artifact actually scored.
    wrong_rows = json.loads(json.dumps(art))
    wrong_rows['best_ppl_estimation_rows'] = (
        'full_validation'
        if art['best_ppl_estimation_rows'] == 'nested_validation_holdout'
        else 'nested_validation_holdout')
    rwr = L.simple_baseline_audit(wrong_rows, protocol=proto)
    assert rwr['baseline_verified'] is False and any(
        'best_ppl_estimation_rows' in p for p in rwr['problems']), rwr


def test_simple_baseline_bound_floor_identity_and_rows():
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert L.simple_baseline_audit(art, protocol=proto)['baseline_verified'] is True
    # The reader-facing floor identity must not fall back to the legacy alias
    # when `best_baseline` is explicitly blank/null.
    for _bad_name in ('', None):
        bad = json.loads(json.dumps(art))
        bad['best_baseline'] = _bad_name
        r = L.simple_baseline_audit(bad, protocol=proto)
        assert r['baseline_verified'] is False and any(
            'best_baseline' in p for p in r['problems']), (_bad_name, r)
    # A present-but-non-finite floor value is malformed evidence, not an
    # absent legacy field.
    bad_ppl = json.loads(json.dumps(art))
    bad_ppl['best_ppl'] = float('nan')
    rp = L.simple_baseline_audit(bad_ppl, protocol=proto)
    assert rp['baseline_verified'] is False and any(
        'best_ppl' in p for p in rp['problems']), rp
    # Explicit nulls in the per-arm selection cannot silently fall back to a
    # different field.
    _arm = art['best_baseline']
    bad_sel = json.loads(json.dumps(art))
    bad_sel['baselines'][_arm]['selected'] = None
    rs = L.simple_baseline_audit(bad_sel, protocol=proto)
    assert rs['baseline_verified'] is False and any(
        'selected config' in p for p in rs['problems']), rs
    bad_val = json.loads(json.dumps(art))
    bad_val['baselines'][_arm]['selected_metric_value'] = None
    rv = L.simple_baseline_audit(bad_val, protocol=proto)
    assert rv['baseline_verified'] is False and any(
        'selected ppl' in p for p in rv['problems']), rv
    # The estimation-rows label needs actual nested evidence; without it the
    # claim is unverifiable rather than true.
    no_data = json.loads(json.dumps(art))
    no_data.pop('data', None)
    rd = L.simple_baseline_audit(no_data, protocol=proto)
    assert rd['baseline_verified'] is False and any(
        'cannot be verified' in p for p in rd['problems']), rd
    # Per-arm per-row accounting in a nested artifact must describe the held
    # out fold, and an unknown marker is not a sample identity.
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    val8 = rng.integers(0, vocab, size=(8, 15))
    nested = L.simple_baseline_fit(train, val8, vocab, protocol=proto,
                                   selection_rows=sel, evaluation_rows=ev,
                                   row_split=meta)
    assert L.simple_baseline_audit(nested, protocol=proto)['baseline_verified'] is True
    # Present-but-malformed row-split evidence must fail closed, not be
    # ignored and replaced by a setdefault value.
    for _field, _bad in (('n_rows', 0), ('n_rows', 1.5),
                         ('n_selection_rows', 0),
                         ('n_evaluation_rows', float('nan')),
                         ('selection_row_range', [0]),
                         ('selection_row_range', [-1, 1]),
                         ('evaluation_row_range', 'bad')):
        _m = json.loads(json.dumps(meta))
        _m[_field] = _bad
        try:
            L.simple_baseline_fit(train, val8, vocab, protocol=proto,
                                  selection_rows=sel, evaluation_rows=ev,
                                  row_split=_m)
        except ValueError:
            pass
        else:
            raise AssertionError(f'simple_baseline_fit accepted row_split {_field}={_bad!r}')
    wrong_rows = json.loads(json.dumps(nested))
    wrong_rows['baselines'][_arm]['reported_estimation_rows'] = 'full_validation'
    rr = L.simple_baseline_audit(wrong_rows, protocol=proto)
    assert rr['baseline_verified'] is False and any(
        'not the held-out fold' in p for p in rr['problems']), rr
    unknown = json.loads(json.dumps(nested))
    unknown['baselines'][_arm]['reported_estimation_rows'] = 'some_fold'
    ru = L.simple_baseline_audit(unknown, protocol=proto)
    assert ru['baseline_verified'] is False and any(
        'reported_estimation_rows' in p for p in ru['problems']), ru


def test_metric_audit_recommended_availability_and_exact_floor_support():
    y = np.array([0] * 1000 + [1])
    score = np.r_[np.linspace(0.0, 0.4, 1000), 0.9]
    m = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                 positive_label=1)
    m.update(task='binary_classification', variant='x')
    ok = L.metric_policy_audit({'x': m})
    assert not any('recommended_primary_available' in w for w in ok['warnings']), ok
    m_bad = json.loads(json.dumps(m))
    m_bad['pr_auc'] = None
    m_bad['recommended_primary_available'] = True
    rb = L.metric_policy_audit({'x': m_bad})
    assert rb['status'] == 'warn' and any(
        'recommended_primary_available=True' in w for w in rb['warnings']), rb
    # A fractional/boolean support is not a count and cannot reproduce the
    # majority-class floor.
    m_sup = json.loads(json.dumps(m))
    _k = next(k for k in m_sup['per_class']
              if abs(float(m_sup['per_class'][k]['support']) - 1.0) < 1e-12)
    m_sup['per_class'][_k]['support'] = 1.5
    rs = L.metric_policy_audit({'x': m_sup})
    assert rs['status'] == 'warn' and any(
        'exact non-negative count' in w or 'trivial-baseline' in w
        for w in rs['warnings']), rs
    # Micro one-vs-rest PR-AUC pools K decisions per sample: its chance floor
    # is 1/K, not a class prevalence.  A missing/wrong floor must not pass.
    ym = np.repeat([0, 1, 2], [90, 9, 1])
    sm = np.random.default_rng(3).random((100, 3))
    sm /= sm.sum(axis=1, keepdims=True)
    mm = L.classification_metrics(ym, y_score=sm, labels=[0, 1, 2])
    assert abs(mm['pr_auc_micro_chance'] - 1.0 / 3.0) < 1e-12, mm
    assert abs(mm['pr_auc_micro_lift_over_chance']
               - (mm['pr_auc_micro'] - 1.0 / 3.0)) < 1e-12, mm
    mm.update(task='multiclass_classification', variant='mc')
    assert L.metric_policy_audit({'mc': mm})['status'] == 'ok'
    mm_bad = json.loads(json.dumps(mm))
    mm_bad['pr_auc_micro_chance'] = 0.9
    rm = L.metric_policy_audit({'mc': mm_bad})
    assert rm['status'] == 'warn' and any(
        'pr_auc_micro_chance' in w for w in rm['warnings']), rm


def test_build_report_malformed_panel_is_excluded():
    import build_report as BR
    td = tempfile.mkdtemp(prefix='br_shape_')
    old_repo = BR.REPO
    old_loaded = BR._LOADED
    try:
        BR.REPO = td
        BR._LOADED = {}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a', 'seed': 0, 'ppl': 1.0}], f)
        with open(os.path.join(td, 'aggregate.json'), 'w', encoding='utf-8') as f:
            json.dump('not an aggregate', f)
        assert BR.summary(td) == {}
        assert BR.agg(td) == {}
        assert BR.per_seed_records(td) == {}
        assert BR._legacy_warm_tags_cached(td) == {}
    finally:
        BR.REPO = old_repo
        BR._LOADED = old_loaded
        shutil.rmtree(td, ignore_errors=True)


def test_single_factor_design_gate_passes_and_fails_closed():
    # The design-time single-factor gate must (1) verify the SHIPPED design
    # and (2) refuse both drift directions when the arm configs drift away
    # from the declared intervention key sets.
    import dataclasses as _dc
    rep = L.single_factor_design_audit(verbose=False)
    assert rep['status'] == 'ok', rep['violations'] + rep['undeclared']
    assert rep['checked_pairs'] == len(L.declared_ablation_pairs()) + sum(
        len(c) for c in L.BASELINE_DESIGN.values()), rep['checked_pairs']
    # the method-vs-matched-dense baselines and the hybrid_hierarchy swap are
    # architecture-level (kind in the diff), never knob-level single factors
    assert rep['architecture_level_pairs'], rep
    for label in rep['architecture_level_pairs']:
        assert 'kind' in rep['pairs'][label]['actual_keys'], label
    # A multi-family bundle (the dynamic-HCA pairs move segmentation together
    # with comp_lambda_mult) is system-level, never a knob-level single factor;
    # every reported single_factor pair must really sit inside one family.
    assert rep['system_level_pairs'], rep
    for label in rep['single_factor_pairs']:
        assert L.cfg_keys_single_factor(rep['pairs'][label]['actual_keys']), label
    for label in rep['system_level_pairs']:
        assert not L.cfg_keys_single_factor(rep['pairs'][label]['actual_keys']), label
        assert rep['pairs'][label]['level'] == 'system_level', label
    # Removing the explicit system-level declaration must fail closed instead
    # of silently re-labelling the bundle as a single-factor ablation.
    _saved_system = dict(L.CONTRAST_SYSTEM_LEVEL_DECLARATIONS)
    try:
        L.CONTRAST_SYSTEM_LEVEL_DECLARATIONS.clear()
        _rep_bad = L.single_factor_design_audit(verbose=False)
        assert _rep_bad['status'] == 'fail', _rep_bad
        assert any('multi-factor' in v for v in _rep_bad['violations']), _rep_bad['violations']
    finally:
        L.CONTRAST_SYSTEM_LEVEL_DECLARATIONS.update(_saved_system)
    # the seeded-random control family: salt counts INSIDE indexer_mode's family
    assert L.cfg_keys_single_factor({'indexer_mode', 'random_salt'})
    assert L.cfg_keys_single_factor({'index_topk'})
    assert not L.cfg_keys_single_factor({'kind'})
    assert not L.cfg_keys_single_factor({'indexer_mode', 'index_topk'})

    # drift 1: a second factor smuggled into one arm -> extra key -> refuse
    def _factory_extra(n_layers, variant):
        cfgs = L.make_layer_cfgs(n_layers, variant)
        if variant == 'csa_fix_nosink':
            return [_dc.replace(c, index_topk=8) for c in cfgs]
        return cfgs
    rep1 = L.single_factor_design_audit(factory=_factory_extra, verbose=False)
    assert rep1['status'] == 'fail', rep1['violations']
    assert any('index_topk' in v and 'csa_fix_nosink' in v for v in rep1['violations']), rep1['violations']

    # drift 2: the declared intervention disappears -> missing key -> refuse
    def _factory_gone(n_layers, variant):
        cfgs = L.make_layer_cfgs(n_layers, variant)
        if variant == 'csa_fix_nosink':
            return L.make_layer_cfgs(n_layers, 'csa_fixed')
        return cfgs
    rep2 = L.single_factor_design_audit(factory=_factory_gone, verbose=False)
    assert rep2['status'] == 'fail', rep2['violations']
    assert any('use_sink' in v and 'disappeared' in v for v in rep2['violations']), rep2['violations']

    # drift 3: a new declared contrast without a key declaration -> fail-closed
    _design = dict(L.ABLATION_DESIGN)
    _design['new_component'] = [('csa_fixed', 'csa_fix_topk8', 'undeclared contrast')]
    rep3 = L.single_factor_design_audit(design=_design, baseline_design={}, verbose=False)
    assert rep3['status'] == 'fail' and rep3['undeclared'], rep3['undeclared']
    assert any('csa_fix_topk8' in u for u in rep3['undeclared']), rep3['undeclared']

    # the open-block matrix must vary only along its two declared axes
    ob = L.open_block_matrix_audit(verbose=False)
    assert ob['status'] == 'ok' and ob['pairs'], ob['violations']
    for label, entry in ob['pairs'].items():
        assert not entry['extra_keys'], (label, entry)


def test_frozen_manifest_mechanism_flags_are_verified_single_factor():
    # The plan preflight must DERIVE each frozen comparison's single-factor
    # verdict and refuse a mechanism_claim_allowed=true that is not backed by
    # the arm construction (cfg diff family + matching objectives + matched
    # parameter totals).
    rep = V12.plan_check(os.path.join(REPO, 'confirmation_frozen.json'))
    assert rep['ok'], rep['errors']
    _cmp = rep['comparisons']
    _seeded = [c for k, c in _cmp.items() if 'randidx_seeded' in k]
    assert _seeded and _seeded[0]['single_factor'] is True, _seeded
    assert _seeded[0]['mechanism_claim_allowed'] is True and _seeded[0]['claim_flag_consistent'] is True
    _main = [c for k, c in _cmp.items() if k.startswith('main:')]
    assert _main and _main[0]['single_factor'] is False and 'kind' in _main[0]['cfg_diff_keys'], _main
    assert all(c['claim_flag_consistent'] for c in _cmp.values()), _cmp
    # every shipped arm carries the objective descriptor the verdict used
    assert all(isinstance(p.get('train_objective'), dict) for p in rep['predicted'].values()), \
        [a for a, p in rep['predicted'].items() if not isinstance(p.get('train_objective'), dict)]

    # an inconsistent flag on the kind-swap main comparison must refuse
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        man = json.load(f)
    _names = [c.get('name') for c in man['primary_comparisons']]
    man['primary_comparisons'][_names.index('main: csa_fixed_rope - full_rope @ L4096 (test)')] = dict(
        man['primary_comparisons'][_names.index('main: csa_fixed_rope - full_rope @ L4096 (test)')],
        mechanism_claim_allowed=True)
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, 'bad_flag.json')
        with open(bad, 'w', encoding='utf-8') as f:
            json.dump(man, f)
        _buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_buf):
                V12.plan_check(bad)
        except SystemExit:
            out = _buf.getvalue()
            assert 'NOT single-factor' in out and "mechanism_claim_allowed=true" in out, out[-600:]
        else:
            raise AssertionError('plan_check must refuse a mechanism flag the arms do not back')
    # the audit-gates entry point reports the same consistency verdicts
    _arep = A.audit_single_factor()
    assert _arep['status'] == 'ok', _arep
    _fm = _arep['frozen_manifest']
    assert _fm['status'] == 'ok' and all(c['claim_flag_cfg_consistent'] for c in _fm['comparisons'].values()), _fm


def test_seq2k_contrasts_carry_factor_declarations():
    # The v9/v10 seq2k paired contrasts must declare their factor structure:
    # the topk endpoint contrast is a single-factor dose response, the two m1
    # contrasts are multi-factor block-geometry bundles (block_size/overlap/
    # index_topk all change together), and add_paired writes the declaration
    # into the comparison record (omitted or not).
    import explib.resume as V9
    decl = V9.SEQ2K_CONTRAST_FACTORS
    assert decl['seq2k(bs1): topk8 - topk512']['single_factor'] is True, decl
    assert decl['seq2k(bs1): topk8 - topk512']['factors'] == ['index_topk'], decl
    for name in ('seq2k(bs1): topk8 - m1', 'seq2k(bs1): topk512 - m1'):
        assert decl[name]['single_factor'] is False, name
        assert set(decl[name]['factors']) == {'block_size', 'overlap', 'index_topk'}, name
    # the declaration must cover exactly the contrasts v9/v10 quote
    for name in ('seq2k(bs1): topk8 - m1', 'seq2k(bs1): topk512 - m1',
                 'seq2k(bs1): topk8 - topk512'):
        assert name in decl, name
    # The declarations are not prose: the audit recomputes each arm diff and
    # refuses a `single_factor=True` that the actual configs do not back.
    _sweep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert _sweep['status'] == 'ok', _sweep['violations']
    assert all(e['verified'] for e in _sweep['pairs'].values()), _sweep['pairs']
    _drift = {k: dict(v) for k, v in decl.items()}
    _drift['seq2k(bs1): topk8 - topk512'] = dict(
        _drift['seq2k(bs1): topk8 - topk512'],
        factors=['index_topk', 'block_size'])
    _sweep_bad = L.sweep_contrast_factor_audit(_drift, verbose=False)
    assert _sweep_bad['status'] == 'fail', _sweep_bad
    assert not _sweep_bad['pairs']['seq2k(bs1): topk8 - topk512']['verified']
    # add_paired attaches the declaration to a real paired record
    knobs = L.training_knobs(amp_dtype=None, eager_batch=False)
    def _rec(ppl):
        return {'run_cfg': 'steps1500_sl2048_bs1_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_v8192_wu50_cl0.05_dlm10_mt4_mr4_ee250_es128_det0_df0_cs0',
                'train_knobs': knobs, 'steps': 1500, 'tokens_seen': 8000000, 'ppl': ppl}
    panel = {'csa_fixed_topk8': {0: _rec(40.0), 1: _rec(41.0)},
             'csa_fix_m1': {0: _rec(42.0), 1: _rec(43.0)}}
    comps = {}
    out = V9.add_paired(comps, 'seq2k(bs1): topk8 - m1', panel, 'csa_fixed_topk8', 'csa_fix_m1',
                        factors=decl['seq2k(bs1): topk8 - m1'])
    assert out is not None and 'omitted' not in out, out
    assert out['single_factor'] is False and 'factors' in out and 'factors_note' in out, out
    assert out['factor_declaration_verified'] is True, out
    assert out['n'] == 2 and out['seeds'] == [0, 1], out
    # and to an omitted record (missing arm) the declaration still attaches
    comps2 = {}
    V9.add_paired(comps2, 'seq2k(bs1): topk8 - topk512', {}, 'csa_fixed_topk8', 'csa_fixed_topk512',
                  outdir='results_lm_v9_seq2k',
                  factors=decl['seq2k(bs1): topk8 - topk512'])
    rec = comps2['seq2k(bs1): topk8 - topk512']
    assert 'omitted' in rec and rec.get('single_factor') is True and rec.get('factors') == ['index_topk'], rec
    assert rec.get('factor_declaration_verified') is True, rec
    # A wrong metadata label is cleared, never propagated as a single factor.
    _bad = dict(decl['seq2k(bs1): topk8 - topk512'], factors=['block_size'])
    _panel = {'csa_fixed_topk8': {0: _rec(40.0)}, 'csa_fixed_topk512': {0: _rec(41.0)}}
    _out = V9.add_paired({}, 'wrong', _panel, 'csa_fixed_topk8', 'csa_fixed_topk512',
                         factors=_bad)
    assert _out['single_factor'] is False, _out
    assert _out['factor_declaration_verified'] is False, _out


def test_v7_rope_contrasts_carry_factor_declarations():
    # The v7 rope/sink/selection contrasts must declare their factor structure
    # and the audit must recompute it from the real v7 arm configs.  In
    # particular the CSA+RoPE vs CSA-absPE contrast moves one positional-
    # encoding family, not an undeclared second knob.
    decl = V7.ROPE_CONTRAST_FACTORS
    for name in ('CSA+RoPE vs CSA absPE', 'CSA+RoPE vs dense+RoPE',
                 'hybrid+RoPE vs dense+RoPE',
                 'v7 scratch: seeded-random selection vs learned',
                 'v7 scratch: all-block reading vs learned',
                 'v7 scratch: no-sink vs learned',
                 'v7 scratch: sink-matched dense vs dense'):
        assert name in decl, name
    pos = decl['CSA+RoPE vs CSA absPE']
    assert pos['single_factor'] is True, pos
    assert set(pos['factors']) == {'qk_norm', 'rope', 'rope_dim'}, pos
    assert L.cfg_keys_single_factor(set(pos['factors'])), pos
    assert not L.cfg_keys_single_factor({'qk_norm', 'rope', 'rope_dim', 'index_topk'})
    _rep = L.sweep_contrast_factor_audit(
        decl, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _rep['status'] == 'ok', _rep['violations']
    assert all(e['verified'] for e in _rep['pairs'].values()), _rep['pairs']
    assert _rep['pairs']['CSA+RoPE vs CSA absPE']['actual_single_factor'] is True
    assert _rep['pairs']['CSA+RoPE vs dense+RoPE']['actual_single_factor'] is False

    _drift = {k: dict(v) for k, v in decl.items()}
    _drift['CSA+RoPE vs CSA absPE'] = dict(
        _drift['CSA+RoPE vs CSA absPE'], factors=['rope', 'rope_dim'])
    _bad = L.sweep_contrast_factor_audit(
        _drift, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _bad['status'] == 'fail', _bad
    assert not _bad['pairs']['CSA+RoPE vs CSA absPE']['verified'], _bad['pairs']
    _mislabel = {k: dict(v) for k, v in decl.items()}
    _mislabel['CSA+RoPE vs dense+RoPE'] = dict(
        _mislabel['CSA+RoPE vs dense+RoPE'], single_factor=True)
    _bad2 = L.sweep_contrast_factor_audit(
        _mislabel, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _bad2['status'] == 'fail', _bad2
    assert not _bad2['pairs']['CSA+RoPE vs dense+RoPE']['verified'], _bad2['pairs']

    _verified = {'CSA+RoPE vs CSA absPE': {'single_factor': True, 'n': 2}}
    V7.attach_contrast_factor_metadata(
        _verified, decl, {'pairs': {'CSA+RoPE vs CSA absPE': {'verified': True}}})
    assert _verified['CSA+RoPE vs CSA absPE']['single_factor'] is True
    assert _verified['CSA+RoPE vs CSA absPE']['factor_declaration_verified'] is True
    _unverified = {'CSA+RoPE vs CSA absPE': {'single_factor': True, 'n': 2}}
    V7.attach_contrast_factor_metadata(
        _unverified, decl, {'pairs': {'CSA+RoPE vs CSA absPE': {'verified': False}}})
    _unverified_rec = _unverified['CSA+RoPE vs CSA absPE']
    assert _unverified_rec['single_factor'] is False, _unverified_rec
    assert _unverified_rec['declared_single_factor'] is True, _unverified_rec
    assert _unverified_rec['factor_declaration_verified'] is False, _unverified_rec
    assert _unverified_rec['factors'] == pos['factors'], _unverified_rec
    _missing = {'CSA+RoPE vs CSA absPE': {'single_factor': True, 'n': 2}}
    V7.attach_contrast_factor_metadata(_missing, decl, {'status': 'fail'})
    assert _missing['CSA+RoPE vs CSA absPE']['single_factor'] is False, _missing
    _system = {'CSA+RoPE vs dense+RoPE': {'single_factor': False, 'n': 2}}
    V7.attach_contrast_factor_metadata(
        _system, decl, {'pairs': {'CSA+RoPE vs dense+RoPE': {'verified': False}}})
    assert _system['CSA+RoPE vs dense+RoPE']['single_factor'] is False, _system


def test_niah_contrast_factors_are_machine_verified():
    import copy as _copy
    rep = L.niah_contrast_factor_audit(
        factory=V7.make_layer_cfgs_v7, verbose=False)
    assert rep['status'] == 'ok', rep['violations']
    for tag in ('randidx', 'allblocks', 'csa_randtrain', 'csa_alltrain'):
        entry = rep['pairs'][tag]
        assert entry['verified'] is True, entry
        assert entry['declared_single_factor'] is True, entry
        assert entry['actual_single_factor'] is True, entry
        assert entry['actual_keys'] == sorted(L.NIAH_CONTRAST_FACTORS[tag]['factors']), entry
    _dense = rep['pairs']['dense']
    assert _dense['verified'] is True and _dense['actual_single_factor'] is False, _dense
    assert _dense['actual_keys'] == ['kind'], _dense
    assert L.NIAH_INFERENCE_ARM_OVERRIDES['randidx'] == {'indexer_mode': 'random'}
    assert L.NIAH_INFERENCE_ARM_OVERRIDES['allblocks'] == {'index_topk': 10 ** 6}

    _base = V7.AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=16,
                           index_topk=32, rope=True, rope_dim=16, qk_norm=True)
    _target = _copy.deepcopy(_base)

    def _diff(a, b):
        return {k for k in a.__dataclass_fields__ if getattr(a, k) != getattr(b, k)}

    L.apply_niah_inference_arm(_target, _base, 'randidx')
    assert _diff(_target, _base) == {'indexer_mode'}, _diff(_target, _base)
    L.apply_niah_inference_arm(_target, _base, 'allblocks')
    assert _diff(_target, _base) == {'index_topk'}, _diff(_target, _base)
    L.apply_niah_inference_arm(_target, _base, 'learned')
    assert _diff(_target, _base) == set(), _diff(_target, _base)

    _bad = _copy.deepcopy(L.NIAH_CONTRAST_FACTORS)
    _bad['randidx'] = dict(_bad['randidx'], factors=['indexer_mode', 'index_topk'])
    _bad_rep = L.niah_contrast_factor_audit(
        _bad, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _bad_rep['status'] == 'fail' and not _bad_rep['pairs']['randidx']['verified'], _bad_rep
    _bad2 = _copy.deepcopy(L.NIAH_CONTRAST_FACTORS)
    _bad2['allblocks'] = dict(_bad2['allblocks'], override='missing_arm')
    _bad2_rep = L.niah_contrast_factor_audit(
        _bad2, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _bad2_rep['status'] == 'fail', _bad2_rep
    _bad3 = _copy.deepcopy(L.NIAH_CONTRAST_FACTORS)
    _bad3['csa_alltrain'] = dict(_bad3['csa_alltrain'], factors=['index_topk', 'use_sink'])
    _bad3_rep = L.niah_contrast_factor_audit(
        _bad3, factory=V7.make_layer_cfgs_v7, verbose=False)
    assert _bad3_rep['status'] == 'fail', _bad3_rep

    _rec = {'single_factor': True}
    V10._attach_niah_factor_metadata(
        _rec, 'randidx', {'pairs': {'randidx': {'verified': False}}})
    assert _rec['single_factor'] is False and _rec['declared_single_factor'] is True, _rec
    assert _rec['factor_declaration_verified'] is False, _rec
    assert _rec['factors'] == ['indexer_mode'], _rec
    _rec_ok = {'single_factor': True}
    V10._attach_niah_factor_metadata(
        _rec_ok, 'allblocks', rep)
    assert _rec_ok['single_factor'] is True, _rec_ok
    assert _rec_ok['factor_declaration_verified'] is True, _rec_ok


def test_driver_pair_capacity_controls():
    # A driver-level method-vs-dense contrast must not quote a p-value as a
    # method advantage when the dense arm has a different parameter count.
    # `add_paired` in both v8 and v9 now reuses the methodology-audit capacity
    # definitions and marks the contrast instead of letting capacity masquerade
    # as the module.
    import explib.scale as V8
    import explib.resume as V9
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _rec(seed, ppl, params):
        return {'variant': 'x', 'seed': seed, 'run_cfg': fp, 'steps': 1,
                'tokens_seen': 1, 'ppl': ppl, 'params': params,
                'train_knobs': knobs}

    panel = {}
    for seed in (0, 1, 2):
        panel.setdefault('method', {})[seed] = _rec(seed, 49.0, 1200)
        panel.setdefault('tiny', {})[seed] = _rec(seed, 50.0, 1000)
        panel.setdefault('match', {})[seed] = _rec(seed, 50.0, 1200)
    for fn in (V9.add_paired, V8._add_paired):
        comp = {}
        fn(comp, 'cap', panel, 'method', 'tiny', require_param_match=True)
        st = comp['cap']
        assert st['param_controlled'] is False, st
        assert st['capacity_confounded'] is True, st
        assert st['method_claim_allowed'] is False, st
        assert abs(st['param_relative_diff'] - 200.0 / 1200.0) < 1e-12, st
        fn(comp, 'ok', panel, 'method', 'match', require_param_match=True)
        st = comp['ok']
        assert st['param_controlled'] is True, st
        assert st['capacity_confounded'] is False, st
        assert st['method_claim_allowed'] is True, st
        # Without the opt-in the numbers remain available as a raw
        # comparison; the capacity readout is still attached.
        fn(comp, 'raw', panel, 'method', 'tiny')
        st = comp['raw']
        assert st['param_controlled'] is False and 'method_claim_allowed' not in st, st
        # A one-sided param_breakdown is UNVERIFIABLE, not a pass: the total
        # count matches, but the capacity detail cannot be attested for both
        # arms.  Treating None as "not False" (the old `and _cap is not False`
        # shortcut) silently promoted it to controlled.
        one_sided = {}
        for seed in (0, 1, 2):
            one_sided.setdefault('method', {})[seed] = dict(
                _rec(seed, 49.0, 1200),
                param_breakdown={'non_embedding': 800, 'active': 700})
            one_sided.setdefault('match_nb', {})[seed] = _rec(seed, 50.0, 1200)
        for fn in (V9.add_paired, V8._add_paired):
            comp = {}
            fn(comp, 'unverifiable', one_sided, 'method', 'match_nb',
               require_param_match=True)
            st = comp['unverifiable']
            assert st['param_capacity_controlled'] is None, st
            assert st['param_controlled'] is None, st
            assert st['capacity_confounded'] is True, st
            assert st['method_claim_allowed'] is False, st


def test_indexer_chunked_partial_tile_preserves_row_stride():
    # idx_score writes each tile compactly with row stride tc, not the
    # allocation's T_c stride.  For R >= 2 the final short chunk must be read
    # back from that compact prefix; scores[:, :tc] keeps stride T_c and reads
    # stale rows for every row after the first.
    import types
    import csa_kernels as C
    R, T, nIH, hd, Bm, k = 2, 5, 1, 2, 8193, 2

    class _Launch:
        def __getitem__(self, _grid):
            return self
        def __call__(self, qI, kI, w, scores, t0, TF, tc, Bm_, *args, **kwargs):
            flat = scores.reshape(-1, int(Bm_))
            for r in range(R):
                for t in range(int(tc)):
                    flat[r * int(tc) + t].fill_(float(1000 * r + int(t0) + t))

    fake = types.SimpleNamespace(
        triton=types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        idx_score=_Launch())
    seen = []

    def _helper(scores, last_tok, t0, T_, Bm_, k_out, need_soft, out_dtype):
        rr, tt = int(scores.shape[0]), int(scores.shape[1])
        for r in range(rr):
            for t in range(tt):
                seen.append((r, int(t0) + t, float(scores[r, t, 0])))
        return (torch.zeros((rr, tt, k_out), dtype=torch.int32),
                torch.zeros((rr, tt, k_out), dtype=torch.bool), None)

    _old = (C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked)
    try:
        C._kernels = lambda: fake
        C._IDX_SCORES_CAP_BYTES = 3 * R * Bm * 4  # T_c = 3 for T = 5
        C._indexer_topk_chunked = _helper
        C._indexer_fwd_triton(
            torch.randn(R, T, nIH, hd), torch.randn(R, Bm, nIH, hd),
            torch.randn(nIH, hd), torch.zeros(R, Bm, dtype=torch.long),
            k, need_soft=False)
    finally:
        C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked = _old
    assert len(seen) == R * T, seen
    bad = [(r, t, got) for r, t, got in seen
           if got != float(1000 * r + t)]
    assert not bad, bad[:6]


def test_categorical_shift_preserves_equal_length_tuple_categories():
    import data_guard as DG
    # np.asarray(list_of_equal_length_tuples) is 2-D; flattening it splits each
    # tuple into two categories.  A tuple is one category identity.
    same = DG.categorical_shift_report([('a', 'b')] * 4, [('a', 'b')] * 4)
    assert same['n_reference'] == 4 and same['n_holdout'] == 4, same
    assert same['js_divergence'] < 1e-12, same
    shifted = DG.categorical_shift_report([('a', 'b')] * 4, [('a', 'c')] * 4)
    assert shifted['n_reference'] == 4, shifted
    assert shifted['unseen_rate'] == 1.0, shifted


def test_preprocessor_audit_rejects_truncated_stats():
    import data_guard as DG
    rng = np.random.default_rng(11)
    Xtr = rng.normal(size=(30, 2))
    fit = DG.LeakageSafePreprocessor().fit(Xtr)
    stats = fit.get_stats()
    stats['features'] = stats['features'][:-1]

    class _Truncated:
        fitted_ = True
        def get_stats(self):
            return stats

    rep = DG.audit_preprocessor_fit(_Truncated(), Xtr)
    assert rep['stats_match_train'] is False, rep
    assert rep['status'] != 'train_only_verified', rep


def test_single_factor_audit_rejects_nonobject_manifest():
    import audit_gates as A
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'manifest.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rep = A.audit_single_factor(manifest_path=path)
    assert rep['status'] == 'fail', rep
    assert rep['frozen_manifest']['status'] == 'fail', rep


def test_factorial_design_gate_verifies_parallel_cells():
    import copy
    rep = L.factorial_design_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert set(rep['factorials']) == set(L.FACTORIAL_DESIGN), rep['factorials']
    for name, rec in rep['factorials'].items():
        assert rec['complete'] is True, (name, rec)
        assert rec['full_factorial']['complete'] is True, (name, rec)
        assert len(rec['cells']) == 4, (name, rec)
        for factor, diffs in rec['factor_diffs'].items():
            assert diffs['at_other_control'] == diffs['at_other_active'], (name, factor, diffs)
            assert 'kind' not in diffs['at_other_control'], (name, factor, diffs)
    # A cell that changes TWO factors makes one axis non-parallel; the
    # interaction would then compare different interventions.
    bad = copy.deepcopy(L.FACTORIAL_DESIGN['dynamic_blocking_x_attention_sink'])
    bad['cells'][('active', 'active')] = 'csa_dyn_zerocont'
    rbad = L.factorial_design_audit(
        {'bad': bad}, present=set(L.RUN_ABL['variants']))
    assert rbad['status'] == 'fail', rbad
    assert any('not parallel' in v for v in rbad['violations']), rbad['violations']
    # Reusing one arm in two cells is not a factorial either.
    dup = copy.deepcopy(L.FACTORIAL_DESIGN['dynamic_blocking_x_learned_selection'])
    dup['cells'][('active', 'active')] = dup['cells'][('control', 'control')]
    rdup = L.factorial_design_audit(
        {'dup': dup}, present=set(L.RUN_ABL['variants']))
    assert rdup['status'] == 'fail', rdup
    assert any('distinct' in v or 'full-factorial' in v for v in rdup['violations']), rdup['violations']
    # An arm absent from the canonical panel cannot be read as a cell.
    rmiss = L.factorial_design_audit(
        present=set(L.RUN_ABL['variants']) - {'csa_dyn_randidx'})
    assert rmiss['status'] == 'fail', rmiss
    assert any('not scheduled' in v for v in rmiss['violations']), rmiss['violations']
    # A scaled model must be audited at its actual depth, not a hard-coded 6.
    rdeep = L.factorial_design_audit(n_layers=2)
    assert rdeep['status'] == 'ok', rdeep['violations']
    assert rdeep['n_layers'] == 2, rdeep
    # Two distinct dict keys that normalise to the same (row, column) cell
    # must not let one silently overwrite the other.
    class _AliasKey(tuple):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    alias = copy.deepcopy(L.FACTORIAL_DESIGN['dynamic_blocking_x_attention_sink'])
    alias['cells'][_AliasKey(('control', 'control'))] = \
        alias['cells'][('control', 'control')]
    ralias = L.factorial_design_audit(
        {'alias': alias}, present=set(L.RUN_ABL['variants']))
    assert ralias['status'] == 'fail', ralias
    assert any('duplicates normalized cell' in v
               for v in ralias['violations']), ralias['violations']


def test_factorial_interaction_test_is_paired_and_isolated():
    # The interaction is the seed-level difference of differences; a constant
    # +0.05 must be detected with the exact sign-flip test, not by comparing
    # two independent significance verdicts.
    recs = _factorial_records(0.05)
    r = L.factorial_interaction_audit(recs=recs)
    assert r['status'] == 'ok', r
    fac = r['factorials']['dynamic_blocking_x_learned_selection']
    it = fac['interaction']
    assert it['available'] is True and it['n'] == 7, it
    assert abs(float(it['mean']) - 0.05) < 1e-9, it
    assert abs(float(it['ratio_of_ratios']) - math.exp(0.05)) < 1e-9, it
    assert it['significant'] is True, it
    assert it['p_holm'] == it['p_exact_signflip'], it
    assert it['direction_consistent'] is True, it
    assert r['significant_interactions'] == ['dynamic_blocking_x_learned_selection'], r
    # The interaction family is its own declared family: only the factorial
    # with arms present is counted; the other two are untouched.
    assert r['family']['m'] == 1, r['family']
    # A negative interaction is a real result but contradicts the
    # pre-registered sub-additive direction; it must not be read as support.
    neg = L.factorial_interaction_audit(recs=_factorial_records(-0.05))
    nit = neg['factorials']['dynamic_blocking_x_learned_selection']['interaction']
    assert nit['significant_family'] is True and nit['direction_consistent'] is False, nit
    assert neg['direction_conflict_interactions'] == ['dynamic_blocking_x_learned_selection'], neg
    # Missing one cell keeps the family slot (m=1) but reports no interaction.
    incomplete = {k: v for k, v in recs.items() if k[0] != 'csa_fix_randidx'}
    ri = L.factorial_interaction_audit(recs=incomplete)
    ifac = ri['factorials']['dynamic_blocking_x_learned_selection']
    assert ifac['status'] == 'incomplete' and ifac['interaction']['available'] is False, ifac
    assert ri['family']['m'] == 1, ri['family']
    assert any('family slot is kept at p=1' in w for w in ri['warnings']), ri['warnings']
    # A run_cfg mismatch drops every seed rather than mixing two recipes.
    mixed = dict(recs)
    for s in range(7):
        mixed[('csa_dyn_randidx', '', s)] = dict(
            mixed[('csa_dyn_randidx', '', s)], run_cfg='different_cfg')
    rm = L.factorial_interaction_audit(recs=mixed)
    mfac = rm['factorials']['dynamic_blocking_x_learned_selection']
    assert mfac['status'] == 'no_data', mfac
    assert any('no complete 2x2 seed' in w for w in rm['warnings']), rm['warnings']
    # A capacity difference is not silently treated as a mechanism effect.
    cap = L.factorial_interaction_audit(
        recs=_factorial_records(0.05, params={'csa_dynamic': 5000}))
    cit = cap['factorials']['dynamic_blocking_x_learned_selection']['interaction']
    assert cit['mechanism_claim_allowed'] is False, cit
    assert cit['significant_family'] is True, cit
    assert any('capacity control' in w for w in cap['warnings']), cap['warnings']


def test_factorial_interaction_fails_closed_on_malformed_records_and_names_missing_seeds():
    import copy
    # A cells declaration that is not a mapping must produce a warning, not an
    # AttributeError from `.values()`.
    bad = {'bad': {'factors': ('dynamic_blocking', 'attention_sink'),
                   'cells': ['not', 'a', 'mapping']}}
    r = L.factorial_interaction_audit(design=bad, recs=_factorial_records(0.05))
    assert r['status'] == 'incomplete', r
    assert any('cells must be a mapping' in w for w in r['warnings']), r['warnings']
    # A non-string (and therefore unhashable) arm id must not reach set().
    unl = copy.deepcopy(L.FACTORIAL_DESIGN['dynamic_blocking_x_attention_sink'])
    unl['cells'][('active', 'active')] = ['csa_dyn_zerocont']
    ru = L.factorial_interaction_audit(
        design={'bad': unl}, recs=_factorial_records(0.05))
    assert ru['status'] == 'incomplete', ru
    assert any('non-empty, hashable arm ids' in w for w in ru['warnings']), ru['warnings']
    # A seed present in only some cells is named in dropped, not silently
    # excluded from the denominator.
    recs = _factorial_records(0.05)
    del recs[('csa_fix_randidx', '', 3)]
    rm = L.factorial_interaction_audit(recs=recs)
    fac = rm['factorials']['dynamic_blocking_x_learned_selection']
    assert fac['n_complete'] == 6, fac
    assert any(int(d.get('seed')) == 3 and 'csa_fix_randidx' in d.get('reason', '')
               for d in fac['dropped']), fac['dropped']


def test_factorial_design_gate_rejects_value_drift_and_absent_interventions():
    import dataclasses
    base = L.AttnCfg(kind='csa', dynamic=False, block_size=4,
                     overlap=L.OVERLAP, index_topk=32)

    def _dyn(block):
        return dataclasses.replace(base, dynamic=True,
                                    chunking='cosine_learnable',
                                    target_block_tokens=block)

    def _design(factory):
        return {'drift': {
            'factors': ('dynamic_blocking', 'attention_sink'),
            'cells': {('control', 'control'): 'a0',
                      ('active', 'control'): 'a1',
                      ('control', 'active'): 'b0',
                      ('active', 'active'): 'b1'},
            'hypothesis': 'dynamic_blocking_x_attention_sink',
        }}

    def value_drift(_n, variant):
        if variant == 'a0':
            return [base]
        if variant == 'a1':
            return [_dyn(4)]
        if variant == 'b0':
            return [dataclasses.replace(base, use_sink=False)]
        return [dataclasses.replace(_dyn(8), use_sink=False)]

    rv = L.factorial_design_audit(
        _design(value_drift), factory=value_drift,
        present={'a0', 'a1', 'b0', 'b1'})
    assert rv['status'] == 'fail', rv
    assert any('same key names' in v and 'different values' in v
               for v in rv['violations']), rv['violations']

    def absent_dynamic(_n, variant):
        if variant in ('a0', 'a1'):
            return [base]
        return [dataclasses.replace(base, use_sink=False)]

    ra = L.factorial_design_audit(
        _design(absent_dynamic), factory=absent_dynamic,
        present={'a0', 'a1', 'b0', 'b1'})
    assert ra['status'] == 'fail', ra
    assert any('intervention is absent' in v for v in ra['violations']), ra['violations']

    # A factory that returns None is a reported construction failure, not a
    # TypeError escaping the audit.
    err, per = L._design_pair_cfg_keys('a', 'b', lambda _n, _v: None, 2)
    assert isinstance(err, Exception) and per == {}, (err, per)


def test_factorial_interaction_matches_stripped_arms_and_disjoint_designs():
    import copy
    recs = _factorial_records(0.05)
    base = copy.deepcopy(
        L.FACTORIAL_DESIGN['dynamic_blocking_x_learned_selection'])
    # A single declared factorial is fully identified by its own four arms;
    # it must not be skipped just because those arms are also the global
    # intersection of a one-element design list.
    rone = L.factorial_interaction_audit(design={'only': base}, recs=recs)
    assert rone['status'] == 'ok', rone
    one = rone['factorials']['only']
    assert one['n_complete'] == 7 and one['interaction']['available'] is True, one
    assert rone['family']['m'] == 1, rone['family']
    # A single declared factorial is identified by its own arms even when the
    # panel is partial: it must be reported incomplete (keeping its p=1 family
    # slot), not silently downgraded to not_applicable.
    partial = {k: v for k, v in recs.items() if k[0] != 'csa_fix_randidx'}
    rpart = L.factorial_interaction_audit(design={'only': base}, recs=partial)
    assert rpart['status'] == 'incomplete', rpart
    pfac = rpart['factorials']['only']
    assert pfac['status'] == 'incomplete' and pfac['interaction']['available'] is False, pfac
    assert rpart['family']['m'] == 1, rpart['family']
    assert any('family slot is kept at p=1' in w for w in rpart['warnings']), rpart['warnings']
    # With multiple declared designs, a panel containing only their shared arms
    # is genuinely ambiguous and must still be skipped rather than attributed
    # to an arbitrary factorial.
    shared_only = {k: v for k, v in recs.items()
                   if k[0] in ('csa_fixed', 'csa_dynamic')}
    amb = L.factorial_interaction_audit(
        design={'one': base,
                'two': copy.deepcopy(
                    L.FACTORIAL_DESIGN['dynamic_blocking_x_content_pooling'])},
        recs=shared_only)
    assert amb['status'] == 'not_applicable' and not amb['factorials'], amb
    # Arm ids are stripped by the design gate; the record-level audit must
    # resolve the same records rather than silently reporting no_data.
    ws = copy.deepcopy(base)
    ws['cells'] = {k: '  %s  ' % v for k, v in ws['cells'].items()}
    rws = L.factorial_interaction_audit(design={'padded': ws}, recs=recs)
    assert rws['status'] == 'ok', rws
    wfac = rws['factorials']['padded']
    assert wfac['n_complete'] == 7 and wfac['interaction']['available'] is True, wfac
    assert wfac['cells']['control|control'] == 'csa_fix_randidx', wfac['cells']
    # Adding a future factorial with disjoint arms must not hide the measured
    # design (the old global-intersection test returned not_applicable).
    future = {
        'factors': ('future_a', 'future_b'),
        'cells': {k: 'future_arm_%d' % i
                  for i, k in enumerate(L._FACTORIAL_CELL_KEYS)},
    }
    rdis = L.factorial_interaction_audit(
        design={'present': base, 'future': future}, recs=recs)
    assert rdis['status'] == 'ok', rdis
    assert set(rdis['factorials']) == {'present'}, rdis['factorials']
    assert rdis['factorials']['present']['n_complete'] == 7, rdis['factorials']
    assert rdis['family']['m'] == 1, rdis['family']


def test_factorial_interaction_look_history_binds_cells_and_tags():
    import copy
    name = 'dynamic_blocking_x_learned_selection'
    key = L._interaction_look_key(name)
    recs5 = _factorial_records(0.05, n=5)
    recs7 = _factorial_records(0.05, n=7)
    s5 = L.factorial_look_snapshot(recs5)[key]
    s7 = L.factorial_look_snapshot(recs7)[key]
    # The snapshot carries the estimand identity, not just numbers.
    assert s5['cells']['control|control'] == 'csa_fix_randidx', s5
    assert set(s5['tags']) == {str(i) for i in range(5)}, s5

    def read(*contrasts, final):
        history = {'looks': [{'look': i, 'contrasts': {key: c}}
                             for i, c in enumerate(contrasts, 1)]}
        return L._interaction_look_analysis(
            L.looks_by_contrast(history), key, final['seeds'], final['deltas'],
            final_tags=final['tags'], final_cells=final['cells'])

    # Re-reading the same look is deduped, not counted as optional stopping.
    seq, stale, n = read(s5, final=s5)
    assert n == 1 and not stale and seq['look_adjusted'] is False, (seq, stale, n)
    # A protocol-tag change is stale even when the deltas are byte-identical.
    retagged = copy.deepcopy(s5)
    retagged['tags'] = {k: 'v2' for k in retagged['tags']}
    seq, stale, n = read(retagged, final=s5)
    assert n == 1 and stale, (seq, stale, n)
    assert stale[0]['reason'] == 'protocol tag changed across looks', stale
    # So is a change in the four cell arms (the estimand identity).
    recelled = copy.deepcopy(s5)
    recelled['cells']['control|control'] = 'other_arm'
    seq, stale, n = read(recelled, final=s5)
    assert n == 1 and stale, (seq, stale, n)
    assert stale[0]['reason'] == 'cell arms changed across looks', stale
    # A malformed duplicate-seed look is excluded as stale, never crashes the
    # sequential reader.
    dup = copy.deepcopy(s5)
    dup['seeds'] = [s5['seeds'][0], s5['seeds'][0]]
    dup['deltas'] = [s5['deltas'][0], s5['deltas'][0]]
    seq, stale, n = read(dup, final=s5)
    assert n == 1 and stale, (seq, stale, n)
    # Two genuine top-up looks remain two looks and are look-adjusted.
    seq, stale, n = read(s5, s7, final=s7)
    assert n == 2 and not stale and seq['look_adjusted'] is True, (seq, stale, n)


def test_classification_threshold_metadata_matches_predictions():
    y = np.array([0, 1, 0, 1])
    score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    argmax = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                      threshold=0.99)
    assert argmax['decision_threshold_source'] == 'argmax', argmax
    assert argmax['decision_threshold'] is None, argmax
    pred_only = L.classification_metrics(y, y_pred=np.array([0, 1, 0, 1]),
                                         labels=[0, 1])
    assert pred_only['threshold_free_metrics'] == [], pred_only
    assert 'roc_auc' not in pred_only and 'pr_auc' not in pred_only, pred_only


def test_baseline_claim_gate_requires_every_rung():
    report = {'baselines': {'compression_vs_matched_dense': [
        {'with': 'csa_dynamic', 'without': 'full_sw128_matched'}]}}
    empty = L.baseline_claim_gate(report)
    assert empty['applicable'] is True and empty['claim_allowed'] is False, empty
    assert empty['rungs']['tuned_count_floor']['status'] == 'unverified', empty
    full = dict(
        report,
        simple_baseline={'tuned_floor_claim_allowed': True,
                         'comparable_to_panel': True,
                         'method_arms_beat_floor': True,
                         'method_arm_floor_status': 'all_method_arms_beat_floor'},
        baseline_tuning={'tuning_verified': True, 'tuned_claim_allowed': True},
        tuned_verified_baselines=['compression_vs_matched_dense'],
        untuned_verified_baselines=[], direction_conflict_baselines=[])
    gate = L.baseline_claim_gate(full)
    assert gate['claim_allowed'] is True and gate['status'] == 'ok', gate
    below_floor = dict(full, simple_baseline=dict(
        full['simple_baseline'],
        method_arms_beat_floor=False,
        method_arm_floor_status='method_arm_does_not_beat_floor'))
    g2 = L.baseline_claim_gate(below_floor)
    assert g2['claim_allowed'] is False and g2['status'] == 'fail', g2
    untuned = dict(full, tuned_verified_baselines=[],
                   untuned_verified_baselines=['compression_vs_matched_dense'])
    g3 = L.baseline_claim_gate(untuned)
    assert g3['claim_allowed'] is False and g3['status'] == 'warn', g3
    conflict = dict(full, tuned_verified_baselines=[],
                    untuned_verified_baselines=[],
                    direction_conflict_baselines=['compression_vs_matched_dense'])
    g4 = L.baseline_claim_gate(conflict)
    assert g4['claim_allowed'] is False and g4['status'] == 'fail', g4
    no_baseline = L.baseline_claim_gate({'baselines': {}})
    assert no_baseline['status'] == 'not_applicable' and no_baseline['claim_allowed'] is None, no_baseline


def test_tuned_baseline_hypothesis_is_attached_to_the_panel():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    hp = {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50}

    def _summary(effect):
        s = {}
        for i in range(7):
            s[f'csa_dynamic::seed{i}'] = dict(
                variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=50.0 + effect, params=1000, train_knobs=knobs, hyperparams=dict(hp))
            s[f'full_sw128_matched::seed{i}'] = dict(
                variant='full_sw128_matched', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0, params=1000, train_knobs=knobs,
                hyperparams=dict(hp))
            s[f'full_matched::seed{i}'] = dict(
                variant='full_matched', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0, params=1000, train_knobs=knobs,
                hyperparams=dict(hp))
        return s

    old_bt, old_sb = L.baseline_tuning_audit, L.simple_baseline_audit
    cov = {a: a for a in ('csa_dynamic', 'full_sw128_matched', 'full_matched')}
    L.baseline_tuning_audit = lambda **kw: {
        'status': 'ok', 'tuning_verified': True, 'tuned_claim_allowed': True,
        'tuned_claim_reasons': [], 'coverage': cov,
        'selected_configs': {a: dict(hp) for a in cov}, 'warnings': []}
    L.simple_baseline_audit = lambda **kw: {
        'status': 'not_verifiable_locally', 'baseline_verified': False,
        'warnings': []}
    try:
        supported = L.methodology_audit(_summary(-1.0), verbose=False)
        falsified = L.methodology_audit(_summary(+1.0), verbose=False)
    finally:
        L.baseline_tuning_audit, L.simple_baseline_audit = old_bt, old_sb
    th = supported['baseline_tuning']['tuned_hypothesis']
    assert th['observed_status'] == 'supported' and th['direction_consistent'] is True, th
    assert supported['baseline_claim_gate']['rungs']['tuned_matched_dense']['passed'] is True
    assert supported['baseline_claim_gate']['claim_allowed'] is False, (
        'the count floor is still unverified, so the full claim must stay closed')
    fh = falsified['baseline_tuning']['tuned_hypothesis']
    assert fh['observed_status'] == 'falsified' and fh['direction_consistent'] is False, fh
    assert falsified['baseline_claim_gate']['status'] == 'fail', (
        falsified['baseline_claim_gate'])


def test_data_protocol_is_field_driven_and_scale_safe():
    # The protocol is resolved from fields, so a later scale-up changes values
    # rather than a code path; malformed/absent declared fields fail closed.
    big = L.resolve_data_protocol({'dataset': 'wikitext', 'seq_len': 16384,
                                   'n_train_tokens': 100_000_000, 'vocab_size': 32768,
                                   'val_seqs': 1024, 'cache_dir': './big'})
    assert big['seq_len'] == 16384 and big['vocab_size'] == 32768, big
    assert big['cache_dir'] == './big'
    for bad in ({'seq_len': True}, {'vocab_size': 0}, {'n_train_tokens': -1}):
        try:
            L.resolve_data_protocol(dict({'dataset': 'wikitext'}, **bad))
            raise AssertionError(f'{bad} must be rejected')
        except ValueError:
            pass
    try:
        L.resolve_data_protocol({'dataset': 'future_corpus', 'seq_len': 4096})
    except Exception:
        # Unknown corpora are allowed only when the loader is supplied; a bare
        # partial protocol is never silently completed with current-scale values.
        pass
    # require= forces a field to be explicit, even when a default exists.
    try:
        L.resolve_data_protocol({'dataset': 'wikitext'}, require=('vocab_size',))
        raise AssertionError('require must reject an implicit default')
    except ValueError:
        pass
    cfg = L.apply_data_protocol({'dataset': 'wikitext', 'outdir': 'x'})
    assert cfg['seq_len'] == L.DATA_PROTOCOL_DEFAULTS['wikitext']['seq_len'], cfg


def test_split_and_prepare_emits_per_split_fingerprints():
    import data_guard as DG
    X = np.arange(60, dtype=np.float64).reshape(30, 2)
    y = np.arange(30, dtype=np.int64)
    a = DG.split_and_prepare(X, y, train_frac=0.6, val_frac=0.2, test_frac=0.2,
                             check_cross_split_duplicates=False)
    fps = a['split_fingerprints']
    assert set(fps) == {'train', 'val', 'test'}, fps
    assert all(v.get('content_fp') for v in fps.values()), fps
    assert fps['train']['index_fp'] != fps['val']['index_fp'], fps
    b = DG.split_and_prepare(X + 1, y, train_frac=0.6, val_frac=0.2,
                             test_frac=0.2, check_cross_split_duplicates=False)
    assert b['split_fingerprints']['train']['content_fp'] != fps['train']['content_fp'], \
        'changed content must change the per-split fingerprint'


def test_chunking_protocol_and_prefix_invariance():
    c = L.AttnCfg(kind='csa', chunking='cosine_learnable', dynamic=True)
    assert L.chunk_similarity_source(c) == 'fused'
    x = torch.randn(11, 4)
    try:
        L.chunk_similarity(x, c, fused=None)
        raise AssertionError('fused source without a fused tensor must fail closed')
    except RuntimeError:
        pass
    c2 = L.AttnCfg(kind='csa', chunking='cosine_learnable', dynamic=True,
                   cut_signal_source='context')
    assert L.chunk_similarity_source(c2) == 'context'
    assert torch.equal(L.chunk_similarity(x, c2), L.cosine_similarity_consecutive(x))
    proto = L.chunking_protocol(c2)
    assert proto['cut_signal_source'] == 'context' and proto['cut_rule_strict'] is True, proto
    # A causal cut table yields prefix-invariant block starts for both the
    # no-cap and capped geometries.
    rng = np.random.default_rng(2)
    cuts = rng.random(80) < 0.25
    for mb in (None, 8):
        rep = L.segmentation_prefix_invariance(cuts, 81, 2, mb)
        assert rep['ok'], (mb, rep['mismatches'][:1])
    # The causal read gate stays strict at the diagonal.
    pos = torch.tensor([0, 1, 2])
    lt = torch.tensor([0, 1, 2])
    assert L.block_readable(pos, lt).diagonal().tolist() == [False, False, False]


def test_bnd_rand_definitions_are_tagged_and_never_pooled():
    gt = np.zeros(20, dtype=bool)
    gt[[3, 10]] = True
    prov = {}
    p, r, f, rp = L.boundary_alignment([3, 10], gt, 20, provenance=prov)
    assert prov.get('bnd_rand_kind') == 'matched_cut_count', prov
    assert 0.0 <= rp <= 1.0
    prov2 = {}
    p2, r2, f2, rp2 = L.boundary_alignment([3, 10], gt, 20,
                                           rand_kind='per_sequence_independent',
                                           provenance=prov2)
    assert prov2.get('bnd_rand_kind') == 'per_sequence_independent', prov2
    assert prov2.get('bnd_rand_exact') is False, prov2
    # At P >= n_pos the independent process has p=1, so every eligible slot is
    # cut: expected precision is matched/n_pos, NOT matched/P.
    _rp_cap = L._random_cut_precision(
        30, None, 19, 1, 2, gt_bounds=[3, 10],
        kind='per_sequence_independent')
    assert abs(_rp_cap - 2.0 / 19.0) < 1e-12, _rp_cap
    try:
        L.boundary_alignment([3], gt, 20, rand_kind='nonsense')
        raise AssertionError('unknown bnd_rand kind must be refused')
    except ValueError:
        pass
    # Aggregating cells that mix the two definitions must not average their
    # excesses into one series.
    def rec(seed, kind):
        cell = {'len_mean': 4.0, 'bnd_prec': 0.5, 'bnd_rec': 0.4,
                'bnd_f1': 0.44, 'bnd_rand': 0.2, 'bnd_excess': 0.3,
                'bnd_rand_exact': (kind == 'matched_cut_count'),
                'bnd_rand_kind': kind}
        return {'variant': 'csa_dynamic', 'seed': seed, 'ppl': 10.0,
                'run_cfg': 'same', 'steps': 1, 'tokens_seen': 10,
                'params': 10, 'stats': {'L0_csa_dyn': cell}}
    summ = {'csa_dynamic::seed0': rec(0, 'matched_cut_count'),
            'csa_dynamic::seed1': rec(1, 'per_sequence_independent')}
    agg = L.aggregate(summ)['csa_dynamic']
    assert 'boundary_rand_kind_conflict' in agg, agg
    assert 'boundary_excess_dyn' not in agg, agg


def test_protocol_group_and_code_semantics_pairing_gate():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = {'variant': 'a', 'seed': 0, 'run_cfg': fp, 'steps': 1,
            'tokens_seen': 1, 'ppl': 10.0, 'data_fp': 'd', 'val_fp': 'v',
            'eval_fp': 'e', 'protocol': 'p1', '_code': 'CODE'}
    b = dict(base, variant='b')
    assert L.pair_reason(base, b) is None
    why = L.pair_reason(base, dict(b, protocol='p2'))
    assert why and 'different protocol' in why, why
    why = L.pair_reason(base, dict(b, _code='CODE2'))
    assert why and 'different code semantics' in why, why
    try:
        L.pair_reason(base, {k: v for k, v in b.items() if k != 'protocol'})
        raise AssertionError('one-sided protocol must be unverifiable')
    except Exception:
        pass
    # aggregate must not borrow an untagged baseline for a tagged arm.
    arm = {'variant': 'csa_dynamic', 'seed': 0, 'run_cfg': 'same',
           'steps': 1, 'tokens_seen': 10, 'ppl': 9.0, 'params': 10,
           'protocol': 'p1'}
    full = {'variant': 'full', 'seed': 0, 'run_cfg': 'same', 'steps': 1,
            'tokens_seen': 10, 'ppl': 10.0, 'params': 10}
    agg = L.aggregate({'csa_dynamic::seed0': arm, 'full::seed0': full})
    assert 'dPPL_vs_full_mean' not in agg.get('csa_dynamic#p1', {}), agg


def test_test_split_purpose_and_content_fingerprints():
    # Only the frozen final evaluation may read the held-out split.
    try:
        L.load_wikitest(8, n_seqs=1, purpose='diagnostic')
        raise AssertionError('a diagnostic test-split read must be refused')
    except ValueError:
        pass
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
        tmeta = {'vocab': 256, 'seq_len': 8, 'split_strategy': 'official_test',
                 'shuffle_at_split': False, 'tokenizer_fit_split': 'train',
                 'src_fp': L._ids_fp(te), 'test_fp': L._ids_fp(te),
                 'read_purpose': 'final_evaluation',
                 'split_fingerprints': {'test': {'ids_fp': L._ids_fp(te)}}}
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(tmeta, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_fingerprint_verified') is True, rec
        assert rec['status'] != 'fail', rec
        # A replaced test file with stale metadata must fail closed.
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te + 1)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('test id-cache fingerprint' in w for w in out['warnings']), out
        # Restore; a recorded read purpose other than the frozen evaluation is
        # itself a protocol failure.
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        tmeta['read_purpose'] = 'diagnostic'
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(tmeta, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('read_purpose' in w for w in out['warnings']), out


def test_test_split_is_read_only_once():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
        lock = V12._check_or_write_freeze_lock(td, man, 'FP', stage='accessed')
        assert lock.get('test_reads') == 1, lock
        # A second P5T preaccess would re-evaluate the same frozen slice.
        try:
            V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
            raise AssertionError('a second test-split access must be refused')
        except SystemExit as e:
            assert 'only once' in str(e) or 'implicit validation' in str(e), str(e)
        # Analysis-style calls (already carrying the slice fingerprint) are
        # not a new read.
        V12._check_or_write_freeze_lock(td, man, 'FP', stage='accessed')
        # A consciously declared new round is recorded, not silent.
        os.environ['V12_REFREEZE'] = '1'
        os.environ['V12_ALLOW_TEST_REREAD'] = '1'
        try:
            lock = V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
        finally:
            del os.environ['V12_REFREEZE']
            del os.environ['V12_ALLOW_TEST_REREAD']
        assert lock.get('test_reads') == 2, lock
        assert any(h.get('stage') == 'reread' for h in lock.get('history') or []), lock


def test_segmentation_prefix_check_honours_small_budget_and_scalar_prefix():
    # The diagnostic must remain usable when a caller asks for fewer checks
    # than the 32 short prefixes: the old np.linspace count went negative and
    # raised instead of returning a partial report.
    rng = np.random.default_rng(7)
    cuts = rng.random(200) < 0.2
    for mc in (0, 1, 5, 10, 31, 32, 33, 64):
        rep = L.segmentation_prefix_invariance(cuts, 201, 2, None,
                                               max_checks=mc)
        assert rep['ok'], (mc, rep)
        assert rep['checked_prefixes'] <= max(0, mc), (mc, rep)
    rep = L.segmentation_prefix_invariance(cuts, 201, 2, None, prefixes=17)
    assert rep['ok'] and rep['checked_prefixes'] == 1, rep


def test_loader_docstrings_survive_parameter_validation():
    # Executable statements before the first string literal silently turn a
    # function docstring into a dead expression; the scale loaders must keep
    # their documented contract introspectable.
    assert L.load_fineweb_edu.__doc__ and 'FineWeb-Edu' in L.load_fineweb_edu.__doc__
    assert L._classify_group_status.__doc__ and 'verified' in L._classify_group_status.__doc__


def test_frozen_vocab_width_is_a_validated_positive_integer():
    m = {'binding': {'training_recipe_signature': {'v': 8192},
                     'per_arm_recipe_signature': {'a': {'v': 8192}}}}
    assert V12._frozen_vocab_size(m) == 8192, m
    assert V12._frozen_vocab_size(m, {'vocab_size': 8192}) == 8192
    assert V12._frozen_vocab_size({}) is None
    for bad in (True, False, 0, -1, 8192.5, float('inf'), float('nan'), 'x'):
        try:
            V12._frozen_vocab_size({'binding': {'training_recipe_signature':
                                               {'v': bad}}})
            raise AssertionError(f'frozen vocab width {bad!r} was accepted')
        except SystemExit:
            pass
    try:
        V12._frozen_vocab_size(
            {'binding': {'training_recipe_signature': {'v': 8192},
                         'per_arm_recipe_signature': {'a': {'v': 16384}}}})
        raise AssertionError('conflicting frozen vocabulary widths were accepted')
    except SystemExit:
        pass


def test_budget_ledger_enforces_token_parameter_read_controls():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        m = json.load(f)
    r = L.comparison_budget_ledger(m)
    assert r['status'] == 'ok', r['violations']
    # n_train_tokens is the served-cache size, NOT steps*bs*sl.  The ledger
    # must compare the CONSUMED token budget across arms and record both.
    tb = r['token_budget']
    assert tb['train_cache_tokens'] == 8000000, tb
    assert tb['consumed_tokens'] == 3000 * 12 * 512 == 18432000, tb
    assert tb['shared_by_all_arms'] is True, tb
    assert abs(tb['definitions']['epochs_over_cache'] - 2.304) < 1e-9, tb
    cmps = r['comparisons']
    main = cmps['main: csa_fixed_rope - full_rope @ L4096 (test)']
    assert main['token_matched'] is True and main['param_matched'] is True, main
    assert main['read_matched'] is False, main           # sparse vs full read
    assert main['mechanism_claim_allowed'] is False, main
    seeded = next(v for k, v in cmps.items() if 'randidx_seeded' in k)
    assert seeded['token_matched'] and seeded['param_matched'], seeded
    assert seeded['read_matched'] is True, seeded        # equal top-k
    assert seeded['dedicated_control'] == 'selection_control', seeded
    allb = next(v for k, v in cmps.items() if 'allblocks' in k)
    assert allb['read_matched'] is False, allb
    assert allb['mechanism_claim_allowed'] is False, allb

    # A mechanism claim on the main sparse-vs-dense comparison must be refused:
    # it is parameter-matched but not read/compute matched, and the cfg diff is
    # an architecture swap, not a single factor.
    bad = copy.deepcopy(m)
    for c in bad['primary_comparisons']:
        if c.get('name', '').startswith('main:'):
            c['mechanism_claim_allowed'] = True
    rb = L.comparison_budget_ledger(bad)
    assert rb['status'] == 'fail', rb
    assert any('read budget is not matched' in v or 'single-factor' in v
               for v in rb['violations']), rb['violations']

    # A per-arm token override silently changes the treatment budget and must
    # break the shared-token premise.
    tok = copy.deepcopy(m)
    tok['binding']['per_arm_recipe_signature']['full_rope']['steps'] = 1500
    rt = L.comparison_budget_ledger(tok)
    assert rt['status'] == 'fail', rt
    assert any('do not share one consumed token budget' in p
               for p in rt['problems']), rt['problems']


def test_evidence_level_audit_keeps_exploratory_exploratory():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        main = json.load(f)
    ev = L.evidence_level_audit(main, manifest_path=os.path.join(REPO, 'confirmation_frozen.json'))
    assert ev['status'] == 'ok', ev['violations']
    assert ev['manifest_level'] == 'confirmatory', ev
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    de = L.evidence_level_audit(dyn, manifest_path=os.path.join(REPO, 'confirmation_frozen_dynamic.json'))
    assert de['status'] == 'ok', de['violations']
    assert de['manifest_level'] == 'draft', de
    # A draft comparison cannot be promoted by label.
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['evidence_level'] = 'confirmatory'
    be = L.evidence_level_audit(bad, manifest_path='confirmation_frozen_dynamic.json')
    assert be['status'] == 'fail', be
    # A v6-v11 panel that claims confirmatory is a hard contradiction: those
    # panels were selected on the same validation split they report.
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'results_lm_v7_rope')
        os.makedirs(panel)
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'evidence_level': 'confirmatory', 'arm': {'ppl': 1.0}}, f)
        pe = L.evidence_level_audit(main, manifest_path='confirmation_frozen.json',
                                    panel_paths=[panel])
        assert pe['status'] == 'fail', pe
        assert any('same validation split' in v for v in pe['violations']), pe['violations']
        # A registry panel with no summary is still exploratory, not a pass for
        # confirmatory wording.
        empty = os.path.join(td, 'results_v10_mech')
        os.makedirs(empty)
        ee = L.evidence_level_audit(main, manifest_path='confirmation_frozen.json',
                                    panel_paths=[empty])
        assert ee['status'] == 'ok', ee
        assert ee['panels']['results_v10_mech']['declared_level'].startswith('exploratory'), ee


def test_dynamic_design_audit_requires_hypotheses_and_controls():
    import copy
    with open(os.path.join(REPO, 'confirmation_frozen_dynamic.json'), encoding='utf-8') as f:
        dyn = json.load(f)
    d = L.dynamic_design_audit(dyn, manifest_path=os.path.join(REPO, 'confirmation_frozen_dynamic.json'))
    assert d['status'] == 'ok', d['violations']
    assert d['runnable'] is False and d['promotion_ready'] is False, d
    assert {'H1_segmentation_quality', 'H2_retrieval_precision',
            'H3_long_range_coverage'} <= set(d['hypotheses']), d['hypotheses']
    # Every required dedicated control is present.
    for role in ('token_budget', 'parameter_proxy', 'read_budget',
                 'selection_control', 'segmentation_control', 'hca_control'):
        assert isinstance(d['controls'].get(role), dict), (role, d['controls'])
    # Missing hypothesis / missing control arm / a draft mechanism claim all
    # fail closed before any GPU spend.
    bad = copy.deepcopy(dyn)
    bad['hypotheses'].pop('H1_segmentation_quality')
    assert L.dynamic_design_audit(bad)['status'] == 'fail'
    bad = copy.deepcopy(dyn)
    bad['arms'].remove('csa_dynamic_rope_randidx_seeded')
    assert L.dynamic_design_audit(bad)['status'] == 'fail'
    bad = copy.deepcopy(dyn)
    bad['primary_comparisons'][0]['mechanism_claim_allowed'] = True
    assert L.dynamic_design_audit(bad)['status'] == 'fail'
    # A frozen/runnable claim without the measured calibration + passed power
    # gate must not be promoted.
    bad = copy.deepcopy(dyn)
    bad['status'] = 'frozen'
    bad['frozen_at'] = '2026-10-22'
    bad['precondition']['runnable'] = True
    assert L.dynamic_design_audit(bad)['status'] == 'fail'
    # The ledger must agree that the dynamic design's token budgets are shared
    # and that the read-rich/dense arms are not the mechanism controls.
    led = L.comparison_budget_ledger(dyn)
    assert led['status'] == 'ok', led['violations']
    assert led['token_budget']['shared_by_all_arms'] is True, led['token_budget']
    hca = next(v for k, v in led['comparisons'].items() if 'HCA' in k)
    assert hca['read_matched'] is False and hca['mechanism_claim_allowed'] is False, hca


def test_dynamic_control_arms_construct_and_isolate_factors():
    # The dynamic design is rope-only; these are the arms the draft names.
    for arm in ('csa_dynamic_rope', 'csa_dynamic_rope_randidx_seeded',
                'csa_dynamic_rope_task', 'hybrid_dynamic_rope',
                'csa_fixed_rope_topk32'):
        cfgs = V7.make_layer_cfgs_v7(6, arm)
        assert len(cfgs) == 6 and all(getattr(c, 'rope', False) for c in cfgs), (arm, cfgs)
    keys, _ = L._design_pair_cfg_keys('csa_dynamic_rope',
                                      'csa_dynamic_rope_randidx_seeded',
                                      V7.make_layer_cfgs_v7, 6)
    assert set(keys) == {'indexer_mode', 'random_salt'}, keys
    keys, _ = L._design_pair_cfg_keys('csa_dynamic_rope',
                                      'csa_dynamic_rope_task',
                                      V7.make_layer_cfgs_v7, 6)
    assert set(keys) == {'boundary_policy'}, keys
    keys, _ = L._design_pair_cfg_keys('csa_fixed_rope_randidx_seeded',
                                      'csa_dynamic_rope_randidx_seeded',
                                      V7.make_layer_cfgs_v7, 6)
    assert set(keys) == {'dynamic', 'chunking', 'target_block_tokens'}, keys
    # HCA is a kind swap: architecture-level, never a single-factor mechanism.
    keys, _ = L._design_pair_cfg_keys('csa_dynamic_rope', 'hybrid_dynamic_rope',
                                      V7.make_layer_cfgs_v7, 6)
    assert 'kind' in set(keys), keys


def test_v12_plan_emits_design_preflight():
    rep = V12.plan_check(os.path.join(REPO, 'confirmation_frozen.json'))
    assert rep['ok'], rep['errors']
    dp = rep.get('design_preflight') or {}
    led = dp.get('budget_ledger') or {}
    ev = dp.get('evidence_level') or {}
    assert led.get('status') == 'ok', led.get('violations')
    assert led['token_budget']['shared_by_all_arms'] is True, led['token_budget']
    assert ev.get('status') == 'ok' and ev.get('manifest_level') == 'confirmatory', ev
    evd = dp.get('evaluation_design') or {}
    assert evd.get('n_units') == 48, evd
    assert (evd.get('population') or {}).get('claim_allowed') is False, evd
    # The dynamic draft must be auditable but explicitly non-runnable.
    dr = V12.dynamic_plan_check(os.path.join(REPO, 'confirmation_frozen_dynamic.json'))
    assert dr['ok'] and dr['runnable'] is False and dr['promotion_ready'] is False, dr


def test_kernel_det_scatter_precision_and_nonfinite_topk():
    import csa_kernels as CK
    # The deterministic scatter must honour the requested dtype and must not
    # erase a later segment behind a large global prefix.
    vals = torch.tensor([1e8, 1.0, 1.0], dtype=torch.float64)
    idx = torch.tensor([0, 1, 1])
    out = CK._det_scatter_add_multi(2, idx, [vals], 'cpu', torch.float64)[0]
    assert out.dtype == torch.float64, out.dtype
    assert torch.equal(out, torch.tensor([1e8, 2.0], dtype=torch.float64)), out
    idx = torch.cat([torch.zeros(64, dtype=torch.int64),
                     torch.ones(1000, dtype=torch.int64)])
    vals = torch.cat([torch.full((64,), 1e12), torch.ones(1000)])
    out = CK._det_scatter_add_multi(2, idx, [vals], 'cpu', torch.float32)[0]
    assert abs(float(out[1]) - 1000.0) < 1.0, out
    # NaN input must delegate to torch.sort instead of leaving buf[..., 0]
    # uninitialised (which crashed at k=1 and returned [5,5] at k=2).
    x = torch.tensor([[float('nan'), 1., 2., 3., 4., 5.]])
    for k in (1, 2, 3):
        got_v, got_i = CK._stable_topk_desc(x, k)
        exp_v, exp_i = torch.sort(x, dim=-1, descending=True, stable=True)
        assert torch.equal(got_i, exp_i[..., :k]), (k, got_i, exp_i)
        assert torch.equal(got_v.nan_to_num(0), exp_v[..., :k].nan_to_num(0)), (k, got_v, exp_v)
    # +/-inf is a routine indexer mask, not a NaN.  The fast rank/scatter path
    # must handle it without falling back to an O(Bm) full sort.
    x_inf = torch.tensor([[1.0, 0.5, float('-inf'), float('-inf')]])
    _orig_sort = torch.sort
    _calls = []

    def _spy_sort(*args, **kwargs):
        _calls.append(tuple(args[0].shape))
        return _orig_sort(*args, **kwargs)

    torch.sort = _spy_sort
    try:
        got_v, got_i = CK._stable_topk_desc(x_inf, 2)
    finally:
        torch.sort = _orig_sort
    exp_v, exp_i = _orig_sort(x_inf, dim=-1, descending=True, stable=True)
    assert not _calls, _calls
    assert torch.equal(got_i, exp_i[..., :2]), (got_i, exp_i)
    assert torch.equal(got_v, exp_v[..., :2]), (got_v, exp_v)


def test_kernel_dense_arange_checks_budget_before_allocating():
    import csa_kernels as CK
    # The stride-0 lane table must remain available at long context.
    lane = CK._full_arange_cached(1, 3, 4, 'cpu', contiguous=False)
    assert lane.shape == (1, 3, 4)
    # The batch/sequence axes are expanded (a size-1 axis may still report a
    # non-zero stride); the lane axis is the real arange.
    assert lane.stride()[1] == 0 and lane.stride()[2] == 1, lane.stride()
    # A dense table whose logical size exceeds the cache budget is refused
    # before `.contiguous()` allocates it (with Bm~T/block_size this would be
    # an O(T^2) allocation).
    try:
        CK._full_arange_cached(1, 100000, 50000, 'cpu', contiguous=True)
    except RuntimeError as exc:
        assert 'contiguous=False' in str(exc), exc
    else:
        raise AssertionError('an oversized dense arange table must be refused')



def test_sweep_factor_protocol_declarations_do_not_crash():
    # A protocol-factor declaration (the v7 warmup cut point) has no AttnCfg
    # arms.  The audit used to read local ``a``/``b`` before assigning them,
    # raising UnboundLocalError instead of returning a verdict.
    decl = {'warm50_vs_100': {'factors': ['warm_steps'], 'single_factor': True,
                              'protocol_factor': 'warm_steps',
                              'a_level': 50, 'b_level': 100}}
    rep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert rep['status'] == 'ok', rep
    entry = rep['pairs']['warm50_vs_100']
    assert entry['mode'] == 'protocol' and entry['verified'] is True, entry
    assert entry['a_level'] == 50 and entry['b_level'] == 100, entry
    # Arms must not be smuggled into a protocol declaration, and the levels
    # must be distinct integers.
    bad = {
        'bad_arms': dict(decl['warm50_vs_100'], a='csa_dynamic', b='csa_fixed'),
        'bad_levels': dict(decl['warm50_vs_100'], a_level=50, b_level=50),
    }
    rep = L.sweep_contrast_factor_audit(bad, verbose=False)
    assert rep['status'] == 'fail', rep
    assert rep['pairs']['bad_arms']['verified'] is False, rep
    assert rep['pairs']['bad_levels']['verified'] is False, rep
    assert any('must not also name' in v for v in rep['violations']), rep
    assert any('distinct integer' in v for v in rep['violations']), rep


def test_interaction_coverage_names_duplicate_factorials_and_out_of_scope_waivers():
    design = {k: [] for k in ('a', 'b', 'c', 'd')}
    factorials = {
        'first': {'factors': ('a', 'b')},
        'second': {'factors': ('b', 'a')},
    }
    r = L.interaction_coverage_audit(
        design=design, factorials=factorials, scope=['a', 'b'],
        waivers={('c', 'd'): 'not part of the core interaction family'})
    assert r['status'] == 'ok', r
    assert r['coverage_status'] == 'complete', r
    assert r['full_pairwise_measured'] is True, r
    assert r['duplicate_pairs'] == {"['a', 'b']": ['first', 'second']}, r['duplicate_pairs']
    assert [x['pair'] for x in r['waivers_outside_scope']] == [['c', 'd']], r
    assert any('outside the declared interaction scope' in w for w in r['warnings']), r
    # A malformed scope is a reported failure, not a silently shrunken denominator.
    bad = L.interaction_coverage_audit(
        design=design, factorials=factorials, scope=['a', 'a'])
    assert bad['status'] == 'fail', bad
    assert any('at least two distinct factors' in p for p in bad['problems']), bad
    # A malformed factorial cannot silently count as "not covering" only.
    badfac = L.interaction_coverage_audit(
        design=design, factorials={'bad': {'factors': ['a']}}, scope=['a', 'b'])
    assert badfac['status'] == 'fail', badfac
    assert any('no two distinct factor names' in p
               for p in badfac['problems']), badfac


def test_factorial_design_audit_accepts_custom_fixture_arms():
    import dataclasses
    base = L.AttnCfg(kind='csa', dynamic=False, block_size=4,
                     overlap=L.OVERLAP, index_topk=32)

    def _dyn():
        return dataclasses.replace(base, dynamic=True,
                                   chunking='cosine_learnable',
                                   target_block_tokens=4)

    def factory(_n, variant):
        return {
            'a0': [base],
            'a1': [_dyn()],
            'b0': [dataclasses.replace(base, use_sink=False)],
            'b1': [dataclasses.replace(_dyn(), use_sink=False)],
        }[variant]

    design = {'fixture': {
        'factors': ('dynamic_blocking', 'attention_sink'),
        'cells': {('control', 'control'): 'a0',
                  ('active', 'control'): 'a1',
                  ('control', 'active'): 'b0',
                  ('active', 'active'): 'b1'},
        'hypothesis': 'dynamic_blocking_x_attention_sink',
    }}
    r = L.factorial_design_audit(
        design, factory=factory, present={'a0', 'a1', 'b0', 'b1'})
    assert r['status'] == 'ok', (r['violations'], r['warnings'])
    assert r['interaction_coverage']['coverage_status'] == 'complete', r
    assert r['component_anchor']['status'] == 'not_applicable', r['component_anchor']
    assert not r['violations'], r['violations']
