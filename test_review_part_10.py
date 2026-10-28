"""CPU review checks (split group 10; t_source_tree_manifest_tracks_exact_bytes .. t_sesoi_practical_decision_primitives)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_source_tree_manifest_tracks_exact_bytes():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, 'docs'))
        os.makedirs(os.path.join(td, 'results_should_be_skipped'))
        with open(os.path.join(td, 'exp_lib.py'), 'w', encoding='utf-8') as f:
            f.write('A = 1\n')
        with open(os.path.join(td, 'pyproject.toml'), 'w', encoding='utf-8') as f:
            f.write('[project]\n')
        with open(os.path.join(td, 'docs', 'design_notes.md'), 'w', encoding='utf-8') as f:
            f.write('note\n')
        with open(os.path.join(td, 'test_should_be_skipped.py'), 'w', encoding='utf-8') as f:
            f.write('NOT_MEASUREMENT = True\n')
        with open(os.path.join(td, 'results_should_be_skipped', 'x.py'), 'w', encoding='utf-8') as f:
            f.write('NOT_SOURCE = True\n')
        m1 = L.source_tree_manifest(root=td)
        m2 = L.source_tree_manifest(root=td)
        assert m1['sha256'] == m2['sha256'], 'two identical scans must agree'
        assert set(m1['files']) == {'exp_lib.py', 'pyproject.toml'}, m1['files']
        assert m1['n_files'] == 2 and 'files' not in L.source_tree_summary(root=td)
        # A content change must change both the per-file digest and the aggregate.
        with open(os.path.join(td, 'exp_lib.py'), 'a', encoding='utf-8') as f:
            f.write('B = 2\n')
        m3 = L.source_tree_manifest(root=td)
        assert m3['sha256'] != m1['sha256']
        assert (m3['files']['exp_lib.py']['sha256']
                != m1['files']['exp_lib.py']['sha256'])
        # A removal must be visible, not silently treated as an unchanged tree.
        os.unlink(os.path.join(td, 'pyproject.toml'))
        m4 = L.source_tree_manifest(root=td)
        assert 'pyproject.toml' not in m4['files']
        assert m4['sha256'] != m3['sha256']
        # An explicit include list is restricted to the requested files.
        one = L.source_tree_manifest(root=td, include=['exp_lib.py'])
        assert set(one['files']) == {'exp_lib.py'}, one
        try:
            L.source_tree_manifest(root=td, include=['../escape.py'])
        except ValueError:
            pass
        else:
            raise AssertionError('include path escaping the root must be rejected')


def test_dependency_contract_parses_lock_and_detects_mismatch():
    from importlib import metadata as _md
    with tempfile.TemporaryDirectory() as td:
        lock = os.path.join(td, 'requirements.lock')
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('# comment\n--extra-index-url https://example.invalid/simple\n')
            f.write(f'numpy=={_md.version("numpy")}\n')
            f.write(f'torch=={_md.version("torch")}\n')
            f.write('PyYAML==' + _md.version('pyyaml') + '\n')
        c = L.dependency_contract(lock)
        assert c['status'] in ('ok', 'partial'), c
        assert c['n_pinned'] == 3 and not c['mismatches'] and not c['missing'], c
        assert c['options'] == ['--extra-index-url https://example.invalid/simple'], c
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('numpy==0.0.0\n')
        c = L.dependency_contract(lock)
        assert c['status'] == 'mismatch' and c['mismatches'], c
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('definitely-not-installed-xyz==1.0\n')
        c = L.dependency_contract(lock)
        assert c['status'] == 'mismatch' and c['missing'], c
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('numpy>=2\n')
        c = L.dependency_contract(lock)
        assert c['status'] == 'invalid' and c['invalid'], c
        c = L.dependency_contract(os.path.join(td, 'absent.lock'))
        assert c['status'] == 'missing_lock', c


def test_dependency_contract_binds_interpreter_and_platform():
    """Package pins alone are not an environment: the interpreter must match.

    A lock is only the verified numerical stack when it declares which Python
    implementation/patch and platform produced it.  Legacy locks without the
    clauses stay usable but report `unverified`; a declared mismatch fails the
    contract instead of silently accepting another stack.
    """
    import platform
    from importlib import metadata as _md
    with tempfile.TemporaryDirectory() as td:
        lock = os.path.join(td, 'requirements.lock')
        good = {
            'python-version': platform.python_version(),
            'python-implementation': platform.python_implementation(),
            'platform': f'{platform.system()} {platform.machine()}'.strip(),
        }

        def _write(declared):
            with open(lock, 'w', encoding='utf-8') as f:
                f.write(f'numpy=={_md.version("numpy")}\n')
                for k, v in declared.items():
                    f.write(f'# {k}: {v}\n')

        _write(good)
        parsed = L.parse_requirements_lock(lock)
        assert parsed['environment'] == good, parsed
        c = L.dependency_contract(lock)
        assert c['interpreter']['status'] == 'match', c['interpreter']
        assert c['interpreter']['declared'] == good, c['interpreter']
        assert c['status'] in ('ok', 'partial'), c
        for key, bad in (('python-version', '0.0.0'),
                         ('python-implementation', 'NotCPython'),
                         ('platform', 'NotAPlatform')):
            declared = dict(good, **{key: bad})
            _write(declared)
            c = L.dependency_contract(lock)
            assert c['interpreter']['status'] == 'mismatch', (key, c['interpreter'])
            assert c['status'] == 'mismatch', (key, c)
            assert any(p.startswith(key) for p in c['interpreter']['problems']), (
                key, c['interpreter']['problems'])
        _write({})
        c = L.dependency_contract(lock)
        assert c['interpreter']['status'] == 'unverified', c['interpreter']
        assert c['interpreter']['problems'] == [], c['interpreter']
        assert c['interpreter']['missing'] == list(L._LOCK_ENV_KEYS), c['interpreter']
        assert c['status'] == 'partial', c
        # A lock that declares only SOME clauses has not identified the full
        # numerical stack: matching declared clauses must not turn the
        # interpreter contract green while implementation/platform are absent.
        _write({'python-version': good['python-version']})
        c = L.dependency_contract(lock)
        assert c['interpreter']['status'] == 'match', c['interpreter']
        assert c['interpreter']['missing'] == ['python-implementation', 'platform'], c['interpreter']
        assert c['status'] == 'partial', c
        assert 'python-implementation' in c['note'], c
        with open(lock, 'w', encoding='utf-8') as f:
            f.write(f'numpy=={_md.version("numpy")}\n')
            f.write('# python-version: 3.12.15\n')
            f.write('# python-version: 3.13.0\n')
        c = L.dependency_contract(lock)
        assert c['status'] == 'invalid' and c['invalid'], c
        # An empty clause is an invalid declaration, not an absent one: silently
        # dropping it would let the remaining clauses look complete.
        with open(lock, 'w', encoding='utf-8') as f:
            f.write(f'numpy=={_md.version("numpy")}\n')
            f.write('# python-version:\n')
        parsed = L.parse_requirements_lock(lock)
        assert parsed['environment'] == {}, parsed
        assert any('empty environment' in msg for msg in parsed['invalid']), parsed
        c = L.dependency_contract(lock)
        assert c['status'] == 'invalid' and c['invalid'], c


def test_requirements_ci_uses_the_verified_lock():
    """CI must install the lock itself, not a second, drifting pin list."""
    path = os.path.join(REPO, 'requirements-ci.txt')
    with open(path, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f]
    requirements = [ln for ln in lines if ln and not ln.startswith('#')]
    assert requirements == ['-r requirements.lock'], (
        'requirements-ci.txt must reference requirements.lock and contain no '
        f'direct pins (found {requirements})')
    parsed = L.parse_requirements_lock(os.path.join(REPO, 'requirements.lock'))
    assert set(parsed['environment']) == {
        'python-version', 'python-implementation', 'platform'}, parsed['environment']
    assert not parsed['invalid'], parsed['invalid']


def test_audit_source_and_dependency_gates():
    cur = L.source_tree_manifest()
    with tempfile.TemporaryDirectory() as td:
        snap = os.path.join(td, 'plan_snapshot.json')
        with open(snap, 'w', encoding='utf-8') as f:
            json.dump({'source_tree': cur}, f)
        r = A.audit_source_integrity(snapshot_path=snap)
        assert r['status'] == 'ok', r
        bad = json.loads(json.dumps(cur))
        bad['sha256'] = '0' * 64
        with open(snap, 'w', encoding='utf-8') as f:
            json.dump({'source_tree': bad}, f)
        r = A.audit_source_integrity(snapshot_path=snap)
        assert r['status'] == 'fail' and r['problems'], r
        r = A.audit_source_integrity(
            snapshot_path=os.path.join(td, 'missing_plan.json'))
        assert r['status'] == 'not_verifiable_locally', r
        lock = os.path.join(td, 'requirements.lock')
        from importlib import metadata as _md
        with open(lock, 'w', encoding='utf-8') as f:
            f.write(f'numpy=={_md.version("numpy")}\n')
        r = A.audit_dependencies(lock_path=lock)
        assert r['status'] == 'ok' and r['contract_status'] in ('ok', 'partial'), r
        with open(lock, 'w', encoding='utf-8') as f:
            f.write('numpy==0.0.0\n')
        r = A.audit_dependencies(lock_path=lock)
        assert r['status'] == 'fail' and r['mismatches'], r


def test_requirements_lock_is_a_complete_environment():
    parsed = L.parse_requirements_lock(os.path.join(REPO, 'requirements.lock'))
    assert parsed['exists'] and not parsed['invalid'], parsed
    assert parsed['pins'], parsed
    # The five direct runtime deps must all be pinned, and the lock must carry
    # the transitive closure rather than regressing to a five-line top-level
    # list (the exact failure mode this hardening fixes).
    for name in ('torch', 'numpy', 'tokenizers', 'datasets', 'matplotlib'):
        assert name in parsed['pins'], (name, sorted(parsed['pins']))
    assert len(parsed['pins']) >= 20, (
        f'requirements.lock pins only {len(parsed["pins"])} distributions; '
        'it must be a full transitive closure')


def test_freeze_lock_binds_the_measurement_source():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        src_a = _fake_source_tree('a')
        src_b = _fake_source_tree('b')
        lock = V12._check_or_write_freeze_lock(
            td, man, None, stage='preaccess', source_tree=src_a)
        assert lock['source_tree_sha256'] == src_a['sha256'], lock
        assert lock['source_tree']['files']['exp_lib.py']['sha256'] == src_a['sha256']
        lock = V12._check_or_write_freeze_lock(
            td, man, 'FP', stage='accessed', source_tree=src_a)
        assert lock['source_tree_sha256'] == src_a['sha256'], lock
        # A test evaluation from a different measurement tree is refused even
        # though the manifest hash and CODE_SEMANTICS stamp still match.
        try:
            V12._check_or_write_freeze_lock(
                td, man, lock.get('test_ids_fp'), stage='accessed', source_tree=src_b)
            raise AssertionError('a test evaluation under changed source must be refused')
        except SystemExit as e:
            assert 'measurement source changed' in str(e), str(e)
            assert 'exp_lib.py' in str(e), str(e)
        # An analysis-only run may disclose the drift explicitly; the lock
        # keeps the ORIGINAL attestation so the source gate still fails.
        os.environ['V12_ALLOW_SOURCE_DRIFT'] = '1'
        try:
            lock = V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='analysis', source_tree=src_b)
        finally:
            del os.environ['V12_ALLOW_SOURCE_DRIFT']
        assert lock['source_tree_sha256'] == src_a['sha256'], lock
        assert lock['source_tree_drift'][-1]['current_sha256'] == src_b['sha256'], lock
        # Even after a disclosed analysis drift, a test evaluation still
        # requires the frozen tree (or a conscious re-freeze).
        try:
            V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='accessed', source_tree=src_b)
            raise AssertionError('a disclosed analysis drift must not unlock P5T')
        except SystemExit as e:
            assert 'measurement source changed' in str(e), str(e)
        # A conscious re-freeze rebinds the source and keeps the old digest.
        os.environ['V12_REFREEZE'] = '1'
        try:
            lock = V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='analysis', source_tree=src_b)
        finally:
            del os.environ['V12_REFREEZE']
        assert lock['source_tree_sha256'] == src_b['sha256'], lock
        assert lock['superseded'][-1]['source_tree_sha256'] == src_a['sha256'], lock
        assert any(h.get('stage') == 'refreeze-source-tree'
                   for h in lock.get('history') or []), lock
        # A malformed/incomplete injected manifest is never accepted as an
        # attestation.
        try:
            V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='analysis', source_tree={'sha256': ''})
            raise AssertionError('a source tree without a digest must be refused')
        except SystemExit:
            pass


def test_freeze_lock_source_audit_and_legacy_boundary():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        cur = L.source_tree_manifest()
        V12._check_or_write_freeze_lock(
            td, man, None, stage='preaccess', source_tree=cur)
        V12._check_or_write_freeze_lock(
            td, man, 'FP', stage='accessed', source_tree=cur)
        lp = os.path.join(td, '.freeze_lock.json')
        r = A.audit_source_integrity(
            snapshot_path=os.path.join(td, 'missing_plan.json'), lock_path=lp)
        assert r['status'] == 'ok' and r['expected_sha256'] == cur['sha256'], r
        # The audit compares the lock's frozen bytes with the current tree.
        with open(lp, encoding='utf-8') as f:
            lock = json.load(f)
        lock['source_tree']['sha256'] = '0' * 64
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump(lock, f)
        r = A.audit_source_integrity(
            snapshot_path=os.path.join(td, 'missing_plan.json'), lock_path=lp)
        assert r['status'] == 'fail' and r['problems'], r
        # A pre-source-binding lock is unverifiable, never "source is fine".
        lock.pop('source_tree', None)
        lock.pop('source_tree_sha256', None)
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump(lock, f)
        r = A.audit_source_integrity(
            snapshot_path=os.path.join(td, 'missing_plan.json'), lock_path=lp)
        assert r['status'] == 'not_verifiable_locally', r
        assert 'predates source-tree binding' in r['note'], r
        # An analysis run that declared source drift is a visible failure even
        # when the current tree happens to hash to the recorded digest again.
        lock['source_tree'] = cur
        lock['source_tree_sha256'] = cur['sha256']
        lock['source_tree_drift'] = [{'stage': 'analysis', 'current_sha256': '1' * 64}]
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump(lock, f)
        r = A.audit_source_integrity(
            snapshot_path=os.path.join(td, 'missing_plan.json'), lock_path=lp)
        assert r['status'] == 'fail', r
        assert r['declared_source_drift'], r


def test_freeze_lock_persists_declared_drift():
    # The disclosed-drift branch used to set _dirty and then have it
    # unconditionally reset, so the event lived only in the returned dict.
    # The on-disk lock — the artifact the source audit reads — stayed clean.
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        src_a = _fake_source_tree('a')
        src_b = _fake_source_tree('b')
        V12._check_or_write_freeze_lock(
            td, man, None, stage='preaccess', source_tree=src_a)
        V12._check_or_write_freeze_lock(
            td, man, 'FP', stage='accessed', source_tree=src_a)
        lp = os.path.join(td, '.freeze_lock.json')
        os.environ['V12_ALLOW_SOURCE_DRIFT'] = '1'
        try:
            V12._check_or_write_freeze_lock(
                td, man, 'FP', stage='analysis', source_tree=src_b)
        finally:
            del os.environ['V12_ALLOW_SOURCE_DRIFT']
        with open(lp, encoding='utf-8') as f:
            disk = json.load(f)
        assert disk.get('source_tree_drift'), disk
        assert disk['source_tree_drift'][-1]['current_sha256'] == src_b['sha256']
        # The original attestation is retained (the round is disclosed, not
        # re-frozen).
        assert disk['source_tree_sha256'] == src_a['sha256'], disk
        # The audit must read the drift from the lock even when the expected
        # tree came from the plan snapshot (which is a different artifact).
        snap = os.path.join(td, 'plan_snapshot.json')
        with open(snap, 'w', encoding='utf-8') as f:
            json.dump({'source_tree': src_a}, f)
        r = A.audit_source_integrity(snapshot_path=snap, lock_path=lp)
        assert r['status'] == 'fail', r
        assert r.get('declared_source_drift'), r
        # Discovery through results_dirs must give the same verdict.
        r2 = A.audit_source_integrity(snapshot_path=snap, results_dirs=[td])
        assert r2['status'] == 'fail' and r2.get('declared_source_drift'), r2


def test_frozen_vocab_accepts_legacy_recipe_key():
    # _recipe_signature accepts the pre-v2 `training_recipe` alias; the
    # vocabulary binding must use the same accepted shape or P5T resolves the
    # test cache from the local width instead of the frozen one.
    assert V12._frozen_vocab_size(
        {'binding': {'training_recipe': {'v': 16384}}}) == 16384
    assert V12._frozen_vocab_size(
        {'binding': {'training_recipe_signature': {'v': 16384}}}) == 16384
    # Conflicting spellings are corrupt evidence, not "pick the new one".
    try:
        V12._frozen_vocab_size({'binding': {
            'training_recipe': {'v': 16384},
            'training_recipe_signature': {'v': 8192}}})
        raise AssertionError('conflicting frozen vocabulary widths must be refused')
    except SystemExit as e:
        assert 'conflicting vocabulary' in str(e), str(e)


def test_row_keys_are_per_row_not_per_block():
    # Duplicate identity must be a pure function of the row.  The old integer
    # branch chose float-vs-int hashing for a whole 64 MiB block, so one
    # float64-inexact row poisoned every row in that block and an equal row in
    # the next block got a different key (missed duplicate/leak).
    import data_guard as DG
    old_block = DG._HASH_BLOCK_BYTES
    DG._HASH_BLOCK_BYTES = 16  # 2 rows per block for a 1-column int64 table
    try:
        X = np.array([[5], [2 ** 60 + 1], [5], [3]], dtype=np.int64)
        got = DG.duplicate_mask(X)
        assert list(got) == [False, False, True, False], got
        # Exact and object spellings of the same rows share one key; a large
        # integer stays distinct from its rounded float image.
        native = DG._row_keys(np.array([[5], [2 ** 60 + 1]], dtype=np.int64))
        obj = DG._row_keys(np.array([[5], [2 ** 60 + 1]], dtype=object))
        assert native == obj, (native, obj)
        assert (DG._row_keys(np.array([[5]], dtype=np.int64))
                == DG._row_keys(np.array([[5.0]], dtype=np.float64)))
    finally:
        DG._HASH_BLOCK_BYTES = old_block


def test_nonmonotone_pooling_uses_original_positions():
    # When block ids are not monotone the rows are sorted by block, but the
    # position bias is a function of the ORIGINAL token position within its
    # block, not of the sorted lane rank.
    X = torch.tensor([[0.], [1.], [0.], [1.]])
    Z = torch.zeros_like(X)
    B = torch.tensor([[0.], [0.5], [1.0], [1.5]])
    bid = torch.tensor([0, 1, 1, 0])
    comp = L.pool_blocks_single(X, Z, B, bid, n_blocks=2,
                                monotonic=False)[0].ravel()
    # block 0 = original tokens {0, 2} -> bias rows {0, 1}; block 1 =
    # original tokens {1, 3} -> bias rows {0, 1}.
    assert torch.allclose(comp, torch.tensor([0.81757444, 0.37754068]),
                          atol=1e-6), comp
    # A merely reversed block order is still contiguous: sorting must recover
    # the same per-block offsets as the already-monotone path.
    mono = L.pool_blocks_single(X, Z, B, torch.tensor([0, 0, 1, 1]),
                                n_blocks=2, monotonic=True)[0]
    rev = L.pool_blocks_single(X, Z, B, torch.tensor([1, 1, 0, 0]),
                               n_blocks=2, monotonic=False)[0]
    assert torch.allclose(mono, rev, atol=1e-6), (mono, rev)


def test_analyze_errors_topk_one_has_no_margin():
    # With one logit the top1/top2 margin is unmeasured; a synthetic 0 margin
    # made every non-rare token "ambiguous_context" and made the blind-spot
    # cause unreachable.
    V = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            x = torch.zeros((h.shape[0], h.shape[1], V))
            x[..., 0] = 10.0
            return x

    vb = np.full((2, 4), 3, dtype=np.int64)  # model always predicts 0
    r = L.analyze_errors(_Stub(), vb, V, torch.device('cpu'), topk=1,
                         top_n=0, top_tokens=0)
    cc = r['cause_counts_all']
    assert cc['ambiguous_context']['n'] == 0, cc
    assert cc['target_outside_topk']['n'] == r['n_targets'], cc


def test_token_frequency_table_chunks_exactly():
    # The rare-target table must not need an int64 copy of the whole corpus.
    # Chunked integer additions have to return the exact bincount vector, also
    # across several chunk boundaries, for every integer storage width the
    # scale corpora use.
    V = 11
    ids = np.array([0, 1, 1, 2, 5, 5, 5, 10, 3, 4, 4, 0, 7, 8, 9, 2, 2, 6,
                    1, 10], dtype=np.int64)
    expected = np.bincount(ids, minlength=V)
    for dt in (np.uint8, np.uint16, np.int32, np.int64):
        got = L._token_frequency_table(ids.astype(dt), V, chunk=3)
        assert got.dtype == np.int64, got.dtype
        assert np.array_equal(got, expected), (dt, got, expected)
    # Empty corpora and a chunk larger than the data are exact too.
    assert np.array_equal(L._token_frequency_table(np.empty(0, np.int32), V),
                          np.zeros(V, dtype=np.int64))
    assert np.array_equal(L._token_frequency_table(ids, V, chunk=1 << 20),
                          expected)
    for bad in (np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.bool_)):
        try:
            L._token_frequency_table(bad, V)
        except TypeError:
            pass
        else:
            raise AssertionError(f'non-integer ids {bad.dtype} must be refused')
    for bad_v, bad_chunk in ((0, 3), (V, 0)):
        try:
            L._token_frequency_table(ids, bad_v, chunk=bad_chunk)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid vocab/chunk must be refused')


def test_metric_threshold_and_manifest_status_guards():
    # Logit scores are a supported 1-D score type; a finite cut of 2.0 is a
    # legal operating point and must not be rejected as "not in [0, 1]".
    y = np.array([0, 1, 0, 1, 1, 0])
    logits = np.array([-2.0, 3.0, -1.0, 2.0, 1.0, -0.5])
    m = L.classification_metrics(y, y_score=logits, labels=[0, 1],
                                 threshold=2.0)
    rec = dict(m, variant='r', task='binary_classification')
    aud = L.metric_policy_audit({'r': rec})
    assert aud['status'] == 'ok', aud
    # Non-finite probability scores are unavailable, not an "ok" calibration
    # with NaN Brier/ECE.
    cal = L._binary_calibration(np.array([0, 1]),
                                np.array([0.5, float('nan')]))
    assert cal['status'] == 'unavailable', cal

    # A declared terminal status must not be contradicted by
    # completion.complete=True.
    with tempfile.TemporaryDirectory() as td:
        cfg = {'seeds': [], 'variants': [], 'outdir': td,
               'n_train_tokens': 1, 'seq_len': 1}
        for status in ('incomplete', 'failed'):
            L.write_run_manifest(td, cfg, 'FP', {'data': {}}, {},
                                 status=status, planned_cells=[])
            e = json.load(open(os.path.join(td, 'run_manifest.json'),
                               encoding='utf-8'))['panels']['FP']
            assert e['status'] == status, e
            assert e['completion']['status'] == status, e['completion']
            assert e['completion']['complete'] is False, e['completion']


def test_effect_modification_across_conditions():
    # One paired effect per seed at each condition: the interaction is the
    # within-seed CHANGE, not a comparison of two separate p-values.
    seeds = list(range(12))
    conds = ['512', '1024', '2048', '4096']
    axis = {'512': 9.0, '1024': 10.0, '2048': 11.0, '4096': 12.0}
    grow = {c: [0.05 * (j + 1) + 0.10 * i for j in seeds]
            for i, c in enumerate(conds)}
    r = L.effect_modification(grow, reference='512', condition_values=axis,
                              seeds=seeds)
    assert r['status'] == 'ok' and r['n_complete'] == 12, r
    assert r['reference'] == '512' and r['conditions'] == conds, r
    assert set(r['interactions']) == {'1024', '2048', '4096'}, r['interactions']
    for i, c in enumerate(conds[1:], start=1):
        cell = r['interactions'][c]
        assert abs(cell['mean'] - 0.10 * i) < 1e-12, (c, cell)
        assert abs(cell['ratio'] - math.exp(0.10 * i)) < 1e-12, (c, cell)
        assert cell['p_exact_signflip'] == L.signflip_p_floor(12), (c, cell)
        # Holm over the three non-reference conditions: m*p on the first step.
        assert cell['p_holm'] is not None and cell['p_holm'] > cell['p_exact_signflip'], (c, cell)
    assert r['interaction_family']['m'] == 3, r['interaction_family']
    assert abs(r['trend']['mean'] - 0.10) < 1e-12, r['trend']
    assert r['trend']['p_exact_signflip'] == L.signflip_p_floor(12), r['trend']
    assert r['trend']['ci95_low'] > 0, r['trend']
    assert 'not evidence' in r['note'], r['note']
    # A constant effect has zero interaction and zero slope: the correct
    # reading is "no detected effect modification", never "equivalent".
    const = {c: [0.05 * (j + 1) for j in seeds] for c in conds}
    rc = L.effect_modification(const, reference='512', condition_values=axis,
                               seeds=seeds)
    assert abs(rc['interactions']['4096']['mean']) < 1e-15, rc
    assert rc['interactions']['4096']['p_exact_signflip'] == 1.0, rc
    assert rc['interactions']['4096']['significant'] is False, rc
    assert abs(rc['trend']['mean']) < 1e-15 and rc['trend']['p_exact_signflip'] == 1.0, rc
    # A seed missing at ONE condition is dropped from the whole panel and
    # named; the simple effects and the interaction must use the same set.
    miss = {c: list(v) for c, v in grow.items()}
    miss['2048'][3] = float('nan')
    rm = L.effect_modification(miss, reference='512', condition_values=axis,
                               seeds=seeds)
    assert rm['n_complete'] == 11 and rm['n_dropped'] == 1, rm
    assert rm['dropped_seeds'] == [3], rm['dropped_seeds']
    assert all(c['n'] == 11 for c in rm['simple_effects'].values()), rm['simple_effects']
    assert all(c['n'] == 11 for c in rm['interactions'].values()), rm['interactions']
    # Numeric keys work too (the condition label is normalised to str for the
    # report/tables, but the reference resolution is by value).
    rnum = L.effect_modification({512: grow['512'], 1024: grow['1024']},
                                 reference=512, seeds=seeds)
    assert rnum['reference'] == '512' and rnum['trend'] is None, rnum
    assert 'no numeric condition axis' in rnum['trend_reason'], rnum['trend_reason']
    r2ax = L.effect_modification({512: grow['512'], 1024: grow['1024']},
                                 reference=512, seeds=seeds,
                                 condition_values={'512': 9, '1024': 10})
    assert r2ax['trend'] is None and 'at least three' in r2ax['trend_reason'], r2ax
    # A non-monotone or incomplete axis is refused as a slope, while the
    # interaction itself (which does not need an axis) still runs.
    rbad_axis = L.effect_modification(
        grow, reference='512',
        condition_values={'512': 10, '1024': 9, '2048': 11, '4096': 12},
        seeds=seeds)
    assert rbad_axis['trend'] is None and 'strictly increasing' in rbad_axis['trend_reason'], rbad_axis
    rmiss_axis = L.effect_modification(
        grow, reference='512',
        condition_values={'512': 9, '1024': 10, '2048': 11}, seeds=seeds)
    assert rmiss_axis['trend'] is None and 'no condition_values' in rmiss_axis['trend_reason'], rmiss_axis
    # No data: reported, not crashed into a fabricated p.
    r0 = L.effect_modification({'a': [float('nan')], 'b': [float('nan')]},
                               reference='a', seeds=[0])
    assert r0['status'] == 'no_data' and r0['n_complete'] == 0, r0
    # Malformed inputs fail closed.
    for bad in (lambda: L.effect_modification({}, reference='a'),
                lambda: L.effect_modification({'a': '12', 'b': '12'}),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1]}),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, reference='c'),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, seeds=[0]),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, seeds=[0, 0]),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, seeds=[0, True]),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, alpha=0),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, min_n=-1),
                lambda: L.effect_modification({'a': [1, 2], 'b': [1, 2]}, min_n=2.5)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('malformed effect_modification input must raise ValueError')
    # Booleans are not measurements; dropping them must also drop the seed.
    rb = L.effect_modification({'a': [True, 2.0], 'b': [1.0, 1.0]}, seeds=[0, 1])
    assert rb['n_complete'] == 1 and rb['dropped_seeds'] == [0], rb


def test_length_interaction_analysis_and_markdown():
    lens = [512, 1024, 2048, 4096]
    recs_a, recs_b = {}, {}
    for s in range(12):
        recs_a[s] = {'by_len': {
            str(Ln): {'ppl': 10.0 * math.exp(0.05 * (s + 1) + 0.10 * i),
                      'truncated': False}
            for i, Ln in enumerate(lens)}}
        recs_b[s] = {'by_len': {
            str(Ln): {'ppl': 10.0 * math.exp(0.05 * (s + 1)),
                      'truncated': False}
            for i, Ln in enumerate(lens)}}
    li = V12._length_interaction_analysis(recs_a, recs_b, list(range(12)), lens,
                                          primary_len=4096)
    assert li is not None, li
    assert li['reference_length'] == 512 and li['primary_length'] == 4096, li
    assert li['n_complete'] == 12 and li['interactions']['4096']['mean'] > 0, li
    assert 'log(PPL_a)' in li['effect_definition'], li
    assert 'never changes' in li['role'] and 'post-hoc' in li['role'], li
    # A seed whose L2048 cell is unusable is dropped from EVERY length, not
    # only from the 2048 interaction.
    bad_a = {s: {'by_len': {k: dict(v) for k, v in r['by_len'].items()}}
             for s, r in recs_a.items()}
    bad_a[2]['by_len']['2048']['truncated'] = True
    li2 = V12._length_interaction_analysis(bad_a, recs_b, list(range(12)), lens,
                                           primary_len=4096)
    assert li2['n_complete'] == 11 and li2['dropped_seeds'] == [2], li2
    assert all(c['n'] == 11 for c in li2['interactions'].values()), li2
    # The reference falls back to the shortest length when training length 512
    # is not in the panel.
    li3 = V12._length_interaction_analysis(recs_a, recs_b, list(range(12)),
                                           [1024, 2048], primary_len=2048)
    assert li3['reference_length'] == 1024, li3
    # Fewer than two lengths/seeds -> no panel, never a one-point "trend".
    assert V12._length_interaction_analysis(recs_a, recs_b, list(range(12)),
                                            [512], primary_len=512) is None
    assert V12._length_interaction_analysis(recs_a, recs_b, [0], lens,
                                            primary_len=4096) is None
    # Rendering: the interaction table and the "not two significance verdicts"
    # caveat must both survive, and an absent block must render nothing.
    md = V12._length_interaction_markdown('cmp', li)
    assert md and any('L4096' in line for line in md), md
    assert any('交互' in line and 'Holm' in line for line in md), md
    assert any('不显著 ≠' in line for line in md), md
    assert any('不改变' in line for line in md), md
    assert any('趋势' in line for line in md), md
    assert V12._length_interaction_markdown('cmp', None) == []
    assert V12._length_interaction_markdown('cmp', {}) == []


def test_wikitext_article_unit_primitives():
    """L151 — the raw-row unit is a line; articles are heading-delimited."""
    rows = ['\r', ' = Alpha = \r', ' = = Section = = \r', ' A paragraph. \r',
            ' = Beta = ', ' Another paragraph. ']
    g = L.wikitext_article_groups(rows)
    assert g['n_articles'] == 2, g
    assert g['n_nonempty_rows'] == 5, g
    assert g['group_sizes'] == [3, 2], g
    assert g['group_has_heading'] == [True, True], g
    assert g['valid'] is True and g['boundary_rule'] == 'top_level_heading', g
    assert g['row_group'][0] == -1 and g['row_group'][1] == 0, g
    assert g['row_group'][4] == 1, g
    # A blank line inside an article (between sections) must not split it.
    assert ' = = Section = = ' in L.wikitext_article_text(rows, g['groups'][0]), g
    # Preamble rows are reported, not silently folded into the first article.
    g2 = L.wikitext_article_groups([' intro ', ' = Alpha = ', ' body '])
    assert g2['n_articles'] == 2 and g2['preamble_groups'] == 1, g2
    assert g2['valid'] is False, g2
    # No top-level heading: the corpus is one group and explicitly invalid.
    g3 = L.wikitext_article_groups([' a ', ' b ', ' c '])
    assert g3['n_articles'] == 1 and g3['valid'] is False, g3
    assert g3['has_headings'] is False, g3
    # Non-string rows are refused instead of coerced (a bytes row would hash
    # differently from the loader's str row).
    try:
        L.wikitext_article_groups([' a ', b' b '])
    except TypeError:
        pass
    else:
        raise AssertionError('article grouping must refuse non-string rows')
    # Stable article ids: same rows -> same id; any edited line -> new id.
    a1 = L.wikitext_article_ids(rows, g['groups'])
    a2 = L.wikitext_article_ids(rows, g['groups'])
    assert a1 == a2 and len(a1) == 2 and a1[0] != a1[1], (a1, a2)
    edited = list(rows)
    edited[3] = ' A changed paragraph. '
    assert L.wikitext_article_ids(edited, g['groups'])[0] != a1[0]
    # A cache may not claim a unit it does not persist: article claims without
    # article sidecars are refused, as is a contradictory line/article pair.
    _tr = np.arange(16, dtype=np.int32)
    _vb = np.zeros((2, 8), dtype=np.int64)
    _bnd = np.zeros_like(_vb, dtype=bool)
    for _meta, _needle in (
            ({'document_unit': 'paragraph'}, 'document_unit'),
            ({'document_unit': 'article'}, 'article-level sidecar'),
            ({'document_unit': 'line', 'document_unit_is_article': True},
             'document_unit_is_article')):
        _p = L._cached_split_problems(_meta, _tr, _vb, _bnd, 512,
                                      fingerprints_trusted=True,
                                      ranges_trusted=True)
        assert any(_needle in x for x in _p), (_meta, _p)
    # Per-token run recovery: only equal-id runs are merged.
    spans = L.line_spans_from_token_doc_ids(np.asarray([7, 7, 7, 9, 9, 1]))
    assert spans == [(0, 3, 7), (3, 5, 9), (5, 6, 1)], spans
    for bad in (np.asarray([[1, 2]]), np.asarray([1.0, 2.0]), np.asarray([-1, 2])):
        try:
            L.line_spans_from_token_doc_ids(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'line spans accepted {bad!r}')

    # The tokenizer must be round-trippable: a byte-level BPE without its
    # inverse silently decodes to the BPE surface and loses space-prefixed
    # punctuation from boundary_token_ids.
    corpus = ([' hello . hello . hello . '] * 50
              + ['abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,:;!?'] * 8)
    tok = L.build_tokenizer(corpus, 512)
    assert tok.decoder is not None, 'build_tokenizer must attach the ByteLevel decoder'
    text = ' hello . '
    assert tok.decode(tok.encode(text).ids) == text, (tok.decode(tok.encode(text).ids), text)
    bnd = L.boundary_token_ids(tok, tok.token_to_id('<eos>'))
    enc = tok.encode(text)
    surf = [tok.id_to_token(i) for i in enc.ids]
    spaced_punct = [i for i, s in zip(enc.ids, surf)
                    if s.startswith('Ġ') and s[1:].strip()
                    and all(c in L.PUNCT_CHARS for c in s[1:].strip())]
    assert spaced_punct, surf
    assert all(i in bnd for i in spaced_punct), (spaced_punct, surf, sorted(bnd))
    # A legacy tokenizer JSON (decoder null) is still auditable in memory.
    tok.decoder = None
    assert L.ensure_bytelevel_decoder(tok) is True
    assert tok.decode(tok.encode(text).ids) == text


def test_document_unit_audit_detects_line_vs_article():
    """L152 — the audit names the served unit and refuses document claims."""
    td = tempfile.mkdtemp(prefix='docunit_')
    try:
        info = _write_document_unit_fixture(td)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_document_unit(td, raw_dir=info['raw'])
        rec = rep['caches'][os.path.abspath(td)]
        assert rep['status'] == 'warnings', rep
        assert rec['unit'] == 'line' and rec['document_level_claims_allowed'] is False
        assert rec['test']['n_lines'] == 6 and rec['test']['n_articles'] == 2, rec['test']
        assert rec['test']['frozen_prefix']['distinct_articles'] == 2, rec['test']
        assert rec['test']['claims']['document_level_claims_allowed'] is False
        assert rec['test']['claims']['population_claim_allowed'] is False
        assert rec['test']['claims']['n_independent_articles_in_slice'] == 2
        assert rec['article_leakage']['raw_train_articles'] == 2
        assert rec['article_leakage']['raw_validation_articles'] == 1
        assert rec['article_leakage']['articles_with_served_lines_in_both_splits'] == 0
        # The loader caps the validation stream at val_seqs*seq_len + seq_len
        # but serves only the first n_val*seq_len tokens, so a normal cache's
        # per-token sidecar is ONE extra sequence longer than val_batch.  The
        # audit must trim that unserved tail instead of refusing to decode.
        _tok_map = np.load(info['val_token_doc_path'])
        np.save(info['val_token_doc_path'],
                np.concatenate([_tok_map, _tok_map[-8:]]))
        with contextlib.redirect_stdout(io.StringIO()):
            rep_tail = A.audit_document_unit(td, raw_dir=info['raw'])
        _vrec = rep_tail['caches'][os.path.abspath(td)]['validation']
        assert _vrec['status'] == 'ok', _vrec
        assert _vrec['trimmed_unserved_tokens'] == 8, _vrec
        assert _vrec['unit'] == 'line', _vrec
        # A shorter map is still a hard error: some served token would have no
        # document identity.
        from tokenizers import Tokenizer as _Tok
        _tok_obj = _Tok.from_file(os.path.join(td, f"tok_{info['tag']}.json"))
        _short = {'problems': []}
        _va = np.load(os.path.join(td, f"val_batch_{info['tag']}.npy"))
        assert A._unit_decode_runs(
            _va, _tok_map[:-1], _tok_obj, L, 'validation', _short) is None, _short
        assert any('not fully mapped' in p for p in _short['problems']), _short
        np.save(info['val_token_doc_path'], _tok_map)
        # Missing raw data is a disclosed non-verification, not "clean".
        with contextlib.redirect_stdout(io.StringIO()):
            rep_noraw = A.audit_document_unit(td, raw_dir=os.path.join(td, 'absent'))
        rec_noraw = rep_noraw['caches'][os.path.abspath(td)]
        assert rec_noraw['article_leakage']['status'] == 'not_verifiable_locally'
        assert rec_noraw['document_level_claims_allowed'] is False

        # Article-unit classification: each run is a whole article and starts
        # with a heading; a line unit has non-heading runs.
        art_texts = [' = Alpha = \n Alpha body. ', ' = Beta = \n Beta body. ']
        art_runs = [((i * 10, i * 10 + 5,
                      int(L._doc_id_u64(L._doc_exact_fp(t)))), t)
                    for i, t in enumerate(art_texts)]
        assert A._unit_classify_runs(art_runs, L)['unit'] == 'article'
        line_texts = [' = Alpha = ', ' Alpha body. ', ' = Beta = ', ' Beta body. ']
        line_runs = [((i * 5, i * 5 + 5,
                       int(L._doc_id_u64(L._doc_exact_fp(t)))), t)
                     for i, t in enumerate(line_texts)]
        assert A._unit_classify_runs(line_runs, L)['unit'] == 'line'

        # An article whose served lines straddle train and validation is
        # invisible to the exact line index; the raw article title makes it
        # visible and fails the gate.
        arr = np.load(info['val_doc_path'])
        extra = L._doc_id_u64(L._doc_exact_fp(' Alpha has another line. '))
        np.save(info['val_doc_path'], np.append(arr, np.asarray([extra], dtype=np.uint64)))
        with open(os.path.join(info['raw'], 'wiki.valid.raw'), 'a', encoding='utf-8') as f:
            f.write(' = Alpha = \n Alpha has another line. \n')
        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = A.audit_document_unit(td, raw_dir=info['raw'])
        rec2 = rep2['caches'][os.path.abspath(td)]
        assert rep2['status'] == 'fail', rep2
        assert rec2['article_leakage']['articles_with_served_lines_in_both_splits'] >= 1
        assert rec2['document_level_claims_allowed'] is False
        assert any('article leakage' in p for p in rep2['problems']), rep2['problems']
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_article_unit_primitive_and_reconstruction():
    """L153 — article ids are recovered from a frozen line map, without raw."""
    rows = ['', ' = Alpha = ', ' Alpha body. ', ' = = Section = = ',
            ' More alpha. ', '', ' = Beta = ', ' Beta body. ']
    rid = L.wikitext_article_row_ids(rows)
    assert rid['status'] == 'ok' and rid['n_articles'] == 2, rid
    assert rid['valid_for_clustering'] is True, rid
    assert rid['n_lines'] == 6, rid
    aid = rid['row_article_id']
    assert aid[1] == aid[2] == aid[3] == aid[4], rid
    assert aid[6] == aid[7] and aid[1] != aid[6], rid
    # A blank/uncovered row is its own cluster, never folded into an article.
    assert aid[0] not in (aid[1], aid[6]), rid
    assert aid[5] not in (aid[1], aid[6]), rid
    # Cluster identity is the article, not its body bytes: editing a paragraph
    # keeps the cluster, editing/removing the top-level title does not.
    edited = list(rows)
    edited[2] = ' A changed line. '
    assert L.wikitext_article_row_ids(edited)['row_article_id'][1] == aid[1]
    retitled = list(rows)
    retitled[1] = ' = Gamma = '
    assert L.wikitext_article_row_ids(retitled)['row_article_id'][1] != aid[1]
    # A corpus with no top-level heading is returned but not claimed valid.
    nohead = L.wikitext_article_row_ids([' a ', ' b ', ' c '])
    assert nohead['status'] == 'ok' and nohead['valid_for_clustering'] is False

    td = tempfile.mkdtemp(prefix='articleunit_')
    old_env = os.environ.get('WT103_CACHE_DIR')
    try:
        info = _write_document_unit_fixture(td)
        ttag = info['ttag']
        sl = 8
        test = [' = Alpha = ', ' Alpha is a test article. ',
                ' It has two lines. ', ' = Beta = ',
                ' Beta is another article. ', ' Another line here. ']
        ids = np.load(os.path.join(td, f'test_ids_{ttag}.npy'))
        line = np.load(os.path.join(td, f'test_token_doc_ids_{ttag}.npy'))
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(td, f'test_tok_{ttag}.json'))
        rep = L.wikitext_token_article_ids_from_line_map(ids, line, tok)
        assert rep['status'] == 'ok', rep
        assert rep['n_lines'] == 6 and rep['n_articles'] == 2, rep
        assert rep['pseudoreplication_factor'] == 3.0, rep
        assert len(set(rep['line_article_ids'].tolist())) == 2, rep
        assert np.unique(rep['token_article_ids']).size == 2, rep
        # The article id is the article, not one of its line ids.
        assert set(rep['line_article_ids'].tolist()) != set(rep['line_ids']), rep
        # Cache-build path (raw rows + lengths) must agree with the decode path.
        lengths = []
        full = L._tokenise_row_chunks([test], 10 ** 9, 'test', tok,
                                      tok.token_to_id('<eos>'),
                                      lengths_out=lengths)
        te_rows = [r for r in test if r.strip()]
        ta = L.wikitext_token_article_ids_for_rows(
            te_rows, lengths, int(ids.size))
        assert ta['status'] == 'ok', ta
        assert np.array_equal(ta['token_article_ids'],
                              rep['token_article_ids'].reshape(-1)), ta
        assert int(full.size) >= int(ids.size)
        # A legacy tokenizer (decoder null) is repaired in memory, not trusted
        # as BPE surface text.
        tok.decoder = None
        rep_legacy = L.wikitext_token_article_ids_from_line_map(ids, line, tok)
        assert rep_legacy['status'] == 'ok', rep_legacy
        assert np.array_equal(rep_legacy['token_article_ids'],
                              rep['token_article_ids']), rep_legacy
        # A flat map with a wrong total size is refused, never reshaped by
        # guessing where a line boundary moved.
        bad = L.wikitext_token_article_ids_from_line_map(
            ids, line[:-1], tok)
        assert bad['status'] == 'shape_mismatch', bad
        # The persisted test tokenizer is loadable without a test read.  The
        # fixture's tiny BPE fit stops below its requested width, so use the
        # actual serialized width and mirror the file under that tag.
        vw = int(tok.get_vocab_size())
        shutil.copy(os.path.join(td, f'test_tok_{ttag}.json'),
                    os.path.join(td, f'test_tok_v2_sl{sl}_v{vw}_ns{ids.shape[0]}.json'))
        tk = L.load_wikitext_test_tokenizer(td, n_seqs=ids.shape[0],
                                            seq_len=sl, vocab_size=vw)
        assert tk is not None and tk.get_vocab_size() == vw
        shutil.copy(os.path.join(td, f'test_tok_{ttag}.json'),
                    os.path.join(td, f'test_tok_v2_sl{sl}_v{vw + 1}_ns{ids.shape[0]}.json'))
        try:
            L.load_wikitext_test_tokenizer(td, n_seqs=ids.shape[0],
                                           seq_len=sl, vocab_size=vw + 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError('a tokenizer width mismatch must be refused')
        # The driver helper resolves the same article map through the cache.
        os.environ['WT103_CACHE_DIR'] = td
        art, diag = V12._resolve_test_article_ids(
            ids, line.reshape(ids.shape), n_seqs=ids.shape[0], seq_len=sl,
            vocab_size=vw)
        assert art is not None and diag['status'] == 'ok', diag
        assert np.array_equal(art, rep['token_article_ids']), diag
        assert diag['n_articles'] == 2 and diag['n_lines'] == 6, diag
    finally:
        if old_env is None:
            os.environ.pop('WT103_CACHE_DIR', None)
        else:
            os.environ['WT103_CACHE_DIR'] = old_env
        shutil.rmtree(td, ignore_errors=True)


def test_article_level_eval_and_analysis():
    """L154 — article clustering is additive and preferred by the analysis."""
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64,
                       _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(11).integers(0, 256, size=(3, 65)),
                     dtype=np.int64)
    # Four line-level documents inside two articles.
    doc = np.empty_like(ids, dtype=np.uint64)
    doc[:, :20] = 11
    doc[:, 20:40] = 22
    doc[:, 40:] = 33
    art = np.empty_like(ids, dtype=np.uint64)
    art[:, :20] = 101
    art[:, 20:] = 202
    base = V7.eval_length_gen(model, ids, [32], dev, max_pos=None,
                              doc_ids=doc)[32]
    both = V7.eval_length_gen(model, ids, [32], dev, max_pos=None,
                              doc_ids=doc, article_ids=art)[32]
    # The frozen line-level fields are bit-identical; article fields are
    # additive evidence, not a replacement.
    assert both['ppl'] == base['ppl'], (both['ppl'], base['ppl'])
    assert both['nll_per_doc'] == base['nll_per_doc'], both
    assert both['ppl_per_doc'] == base['ppl_per_doc'], both
    assert both['n_docs'] == base['n_docs'] and both['doc_granularity'] == 'document'
    assert both['article_granularity'] == 'article' and both['n_articles'] == 2, both
    assert sum(both['n_tokens_per_article']) == both['n_tok'], both
    assert abs(math.fsum(both['nll_per_article']) - both['nll_sum']) <= \
        1e-5 * max(1.0, abs(both['nll_sum'])), both
    assert len(set(both['article_ids'])) == 2, both
    # Admission accepts the additive cell and rejects inconsistent article
    # vectors instead of letting them reach the bootstrap.
    assert V12._by_len_shape_reason({'by_len': {32: both}}, 3) is None
    for _mut, _needle in (
            (lambda c: c.update(n_tokens_per_article=[46, 46]), 'sums to'),
            (lambda c: c.update(article_ids=['a', 'a']), 'duplicate'),
            (lambda c: c.update(article_ids=[['a'], ['b']]), 'unhashable'),
            (lambda c: c.pop('nll_per_article'), 'missing'),
            (lambda c: c.update(article_granularity='line'), 'article_granularity'),
            (lambda c: c.update(n_articles=1), 'n_articles')):
        bad = _copy.deepcopy(both)
        _mut(bad)
        why = V12._by_len_shape_reason({'by_len': {32: bad}}, 3)
        assert why is not None and _needle in why, (why, _needle, bad)
    for bad_map in (art[:, :-1], art.astype(np.float64), -(art.astype(np.int64))):
        try:
            V7.eval_length_gen(model, ids, [32], dev, max_pos=None,
                               article_ids=bad_map)
        except ValueError:
            pass
        else:
            raise AssertionError('eval_length_gen accepted a bad article map')

    # Analysis prefers the article unit when every record carries it.
    cells_a = _article_cell([2.0, 6.0], [4, 10], ['l1', 'l2'],
                            [2.0, 6.0], [7, 7], ['a1', 'a2'])
    cells_b = _article_cell([2.5, 6.5], [4, 10], ['l1', 'l2'],
                            [2.5, 6.5], [7, 7], ['a1', 'a2'])
    recs_a = {s: {'by_len': {512: _copy.deepcopy(cells_a)}} for s in (0, 1)}
    recs_b = {s: {'by_len': {512: _copy.deepcopy(cells_b)}} for s in (0, 1)}
    out = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    assert out is not None and out['granularity'] == 'article', out
    assert out['unit'] == 'article' and out['n_units'] == 2, out
    assert out['n_articles'] == 2 and out['n_docs'] == 2, out
    assert out['pooled_ci_n_units'] == 2, out
    assert out['n_line_docs'] == 2 and out['pseudoreplication_factor'] == 1.0, out
    assert 'small_cluster_warning' in out, out
    want = math.exp(8.0 / 14.0) - math.exp(9.0 / 14.0)
    assert abs(out['pooled_delta_ppl'] - want) < 1e-12, (out, want)
    assert '文章' in V12._doc_ci_markdown_cell(out), out
    # If only some records carry article cells, do not mix units: fall back
    # to the line-level document map and say so.
    recs_doc_b = _copy.deepcopy(recs_b)
    for s in (0, 1):
        for _k in ('nll_per_article', 'n_tokens_per_article',
                   'ppl_per_article', 'article_ids', 'article_granularity',
                   'n_articles'):
            recs_doc_b[s]['by_len'][512].pop(_k, None)
    fallback = V12._doc_level_analysis(recs_a, recs_doc_b, [0, 1], 512)
    assert fallback is not None and fallback['granularity'] == 'document', fallback
    assert fallback['article_fallback_reason'], fallback
    # Article ids that differ across seeds are not the same units.
    recs_b_ids = _copy.deepcopy(recs_b)
    for s in (0, 1):
        recs_b_ids[s]['by_len'][512]['article_ids'] = ['a1', f'a{s + 3}']
    mixed_ids = V12._doc_level_analysis(recs_a, recs_b_ids, [0, 1], 512)
    assert mixed_ids is not None and mixed_ids['comparable'] is False, mixed_ids


def test_project_packaging_is_complete():
    """The source tree must be installable, not just importable in-place."""
    import tomllib
    import subprocess
    import glob as _glob
    cfg_path = os.path.join(REPO, 'pyproject.toml')
    with open(cfg_path, 'rb') as fh:
        cfg = tomllib.load(fh)
    mods = cfg.get('tool', {}).get('setuptools', {}).get('py-modules')
    assert isinstance(mods, list) and mods, 'pyproject.toml must declare py-modules explicitly'
    expected = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in _glob.glob(os.path.join(REPO, '*.py'))
        if not os.path.basename(p).startswith('test_'))
    assert sorted(mods) == expected, (
        'pyproject [tool.setuptools] py-modules is out of date: '
        f'missing={sorted(set(expected) - set(mods))} '
        f'extra={sorted(set(mods) - set(expected))}')
    assert not any(m.startswith('test_') for m in mods), 'test scripts are not runtime modules'
    assert cfg['project'].get('readme') == 'README.md'
    assert cfg['project'].get('license'), 'the package must declare its license'
    assert os.path.exists(os.path.join(REPO, 'MANIFEST.in'))
    # Actually build/install a minimal source copy: setuptools discovery used to
    # fail with "Multiple top-level packages discovered" and no unit test saw
    # it because in-place imports never invoke the build backend.
    try:
        import setuptools  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise _SkipTest(f'setuptools is unavailable ({type(e).__name__}: {e})')
    pkg_cfg = cfg.get('tool', {}).get('setuptools', {}).get('packages', {}).get('find', {})
    includes = pkg_cfg.get('include') or []
    assert includes, 'pyproject.toml must declare the split packages explicitly'
    pkg_dirs = set()
    for pattern in includes:
        for path in _glob.glob(os.path.join(REPO, pattern)):
            if not os.path.isdir(path):
                continue
            for cur, dirs, names_ in os.walk(path):
                if '__init__.py' in names_:
                    pkg_dirs.add(os.path.relpath(cur, REPO))
    assert pkg_dirs, 'no packages found for the declared include patterns'
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'src')
        os.makedirs(src)
        names = ['pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in',
                 'requirements.txt', 'requirements.lock'] + [m + '.py' for m in mods]
        for name in names:
            p = os.path.join(REPO, name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(src, name))
        for pkg in sorted(pkg_dirs):
            shutil.copytree(os.path.join(REPO, pkg), os.path.join(src, pkg), dirs_exist_ok=True)
        target = os.path.join(td, 'installed')
        proc = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-deps',
             '--no-build-isolation', '--target', target, src],
            capture_output=True, text=True)
        assert proc.returncode == 0, (
            'pip install . failed:\n' + (proc.stdout or '')[-2000:]
            + (proc.stderr or '')[-4000:])
        for m in mods:
            assert os.path.exists(os.path.join(target, m + '.py')), (
                f'{m}.py missing from the installed package')
        for pkg in sorted(pkg_dirs):
            assert os.path.isfile(os.path.join(target, pkg, '__init__.py')), (
                f'{pkg}/__init__.py missing from the installed package')
        assert not os.path.exists(os.path.join(target, 'test_review_fixes.py')), (
            'test scripts must stay source-only')


def test_dataset_source_pinning_and_audit():
    """The upstream corpus commit is pinned, recorded, and independently audited."""
    assert L.WIKITEXT_HF_REVISION == 'b08601e04326c79dfdd32d625aee71d232d685c3'
    assert L.FINEWEB_EDU_HF_REVISION == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
    assert L.wikitext_dataset_source()['revision_pinned'] is True
    assert L.wikitext_dataset_source()['config'] == 'wikitext-103-raw-v1'
    # Moving references are refused; only a full commit is a data identity.
    old = os.environ.get('WT103_DATASET_REVISION')
    try:
        for bad in ('main', 'abc123', 'z' * 40, 'A' * 39):
            os.environ['WT103_DATASET_REVISION'] = bad
            try:
                L.wikitext_dataset_source()
            except ValueError:
                pass
            else:
                raise AssertionError(f'WT103_DATASET_REVISION={bad!r} must be refused')
        os.environ['WT103_DATASET_REVISION'] = 'a' * 40
        assert L.wikitext_dataset_source()['revision'] == 'a' * 40
    finally:
        if old is None:
            os.environ.pop('WT103_DATASET_REVISION', None)
        else:
            os.environ['WT103_DATASET_REVISION'] = old
    good = {'dataset_source': L.wikitext_dataset_source()}
    p, w = L._dataset_source_problems(good, L.wikitext_dataset_source(), 'x')
    assert p == [] and w == [], (p, w)
    legacy_p, legacy_w = L._dataset_source_problems({}, L.wikitext_dataset_source(), 'x')
    assert legacy_p == [] and legacy_w, (legacy_p, legacy_w)
    bad = {'dataset_source': dict(L.wikitext_dataset_source(), revision='b' * 40)}
    p, _w = L._dataset_source_problems(bad, L.wikitext_dataset_source(), 'x')
    assert p and 'differs from the pinned' in p[0], p
    local = {'dataset_source': L.local_dataset_source('raw')}
    p, w = L._dataset_source_problems(local, L.wikitext_dataset_source(), 'x')
    assert p == [] and w, (p, w)
    malformed = {'dataset_source': {'provider': 'hub', 'revision': 'main'}}
    p, _w = L._dataset_source_problems(malformed, L.wikitext_dataset_source(), 'x')
    assert p, p
    pair = L.dataset_source_pair_problems(
        {'dataset_source': dict(L.wikitext_dataset_source(), revision='a' * 40)},
        [{'dataset_source': dict(L.wikitext_dataset_source(), revision='b' * 40)}])
    assert pair[0], pair
    # The audit is independent of the loader: a matching cache pair is ok, a
    # mismatched frozen test cache is a hard failure, and a legacy cache is
    # explicitly unverifiable rather than silently clean.
    with tempfile.TemporaryDirectory() as td:
        def _write(name, meta):
            with open(os.path.join(td, name), 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
        base = {'vocab': 8192, 'fit_fp': 'f' * 16,
                'dataset_source': L.wikitext_dataset_source()}
        _write('meta_v5_sl512.json', base)
        _write('test_meta_v2_sl4097.json',
               dict(base, dataset_source=L.wikitext_dataset_source()))
        out = A.audit_data_source(td)
        assert out['status'] == 'ok', out
        assert out['n_pinned'] == 2 and out['n_legacy'] == 0, out
        # A cache directory may hold several corpora.  A FineWeb train cache
        # must not be paired with the WikiText test slice, and its different
        # upstream commit is not a mixed-snapshot warning.
        _write('meta_v2_sl512_fwsample-10BT.json',
               {'vocab': 8192, 'source': 'sample-10BT',
                'dataset_source': L.fineweb_edu_dataset_source()})
        out = A.audit_data_source(td)
        assert out['status'] == 'ok', out
        assert out['n_pinned'] == 3, out
        assert not any('more than one upstream revision' in w
                       for w in out['warnings']), out
        assert not any('paired train revision' in x
                       for x in out['problems']), out
        _write('test_meta_v2_sl4097.json', dict(
            base, dataset_source=dict(L.wikitext_dataset_source(),
                                      revision='c' * 40)))
        out = A.audit_data_source(td)
        assert out['status'] == 'fail', out
        assert any('does not match' in x for x in out['problems']), out
        _write('test_meta_v2_sl4097.json', {'vocab': 8192})
        out = A.audit_data_source(td)
        assert out['status'] == 'not_verifiable_locally', out
        assert out['n_legacy'] == 1, out
        assert any('predates the pinned' in x for x in out['warnings']), out


def test_local_raw_cache_records_dataset_source():
    """A cold local-raw build must record an explicit, non-pinned source."""
    old = os.environ.get('WT103_RAW_TXT')
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, 'raw')
        os.makedirs(raw)
        for split, n in (('train', 400), ('valid', 200)):
            with open(os.path.join(raw, f'wiki.{split}.raw'), 'w',
                      encoding='utf-8') as fh:
                for i in range(n):
                    fh.write(f'= {split} {i} =\n\nThis is a {split} sentence '
                             f'number {i} with some words.\n\n')
        cache = os.path.join(td, 'cache')
        os.makedirs(cache)
        os.environ['WT103_RAW_TXT'] = raw
        try:
            a = L.load_wikitext(16, 128, 64, val_seqs=4, cache_dir=cache)
            meta_path = os.path.join(cache, 'meta_v5_sl16_cap128_v64_vs4.json')
            assert os.path.exists(meta_path), os.listdir(cache)
            with open(meta_path, encoding='utf-8') as fh:
                meta = json.load(fh)
            src = meta.get('dataset_source')
            assert isinstance(src, dict), src
            _samp = meta.get('sampling')
            assert isinstance(_samp, dict), meta
            assert _samp['train']['design'] == 'source_order_prefix', _samp
            assert _samp['train']['representativeness_claim_allowed'] is False, _samp
            assert _samp['train']['source_rows'] == 1600, _samp
            assert _samp['train']['rows_served'] > 0, _samp
            assert _samp['validation']['source_rows'] == 800, _samp
            assert src.get('provider') == 'local' and src.get('revision') is None, src
            assert src.get('revision_pinned') is False, src
            # A cache hit must serve exactly the same bytes and not fabricate a
            # Hub revision for a local dump.
            b = L.load_wikitext(16, 128, 64, val_seqs=4, cache_dir=cache)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            _p, _w = L._dataset_source_problems(
                meta, L.wikitext_dataset_source(), 'local')
            assert _p == [] and _w, (_p, _w)
        finally:
            if old is None:
                os.environ.pop('WT103_RAW_TXT', None)
            else:
                os.environ['WT103_RAW_TXT'] = old


def test_document_stratified_plan_is_reproducible_and_auditable():
    sizes = [1 + (i * 7) % 23 for i in range(120)]
    p1 = L.document_stratified_plan(sizes, 24, seed=11, n_strata=6)
    p2 = L.document_stratified_plan(sizes, 24, seed=11, n_strata=6)
    assert p1 == p2, (p1.get('plan_fp'), p2.get('plan_fp'))
    assert p1['n_units'] == 120 and p1['n_select'] == 24, p1
    assert p1['selected'] == sorted(p1['selected']), p1['selected']
    assert len(set(p1['selected'])) == 24, p1['selected']
    assert all(0 <= i < 120 for i in p1['selected']), p1['selected']
    assert p1['selection_without_replacement'] is True, p1
    assert p1['all_strata_represented'] is True, p1['strata']
    assert sum(s['n_units'] for s in p1['strata']) == 120, p1['strata']
    assert sum(s['n_selected'] for s in p1['strata']) == 24, p1['strata']
    assert all(s['n_selected'] >= 1 for s in p1['strata']), p1['strata']
    assert abs(p1['unit_coverage_fraction'] - 24 / 120) < 1e-12, p1
    assert L.sampling_plan_problems(p1) == [], L.sampling_plan_problems(p1)
    # A different source snapshot must not reuse the same selected units.
    p3 = L.document_stratified_plan(sizes, 24, seed=11, n_strata=6,
                                    source_fp='snapshot-b')
    assert p3['selected'] != p1['selected'], ('source_fp ignored', p1['selected'])
    assert L.sampling_plan_problems(p3) == [], L.sampling_plan_problems(p3)
    # Tampering with the persisted selection invalidates the fingerprint.
    bad = json.loads(json.dumps(p1))
    bad['selected'][0] = (bad['selected'][0] + 1) % 120
    if bad['selected'] != sorted(set(bad['selected'])):
        bad['selected'] = sorted(set(bad['selected']))
    bad['selected'] = sorted(set(bad['selected']))
    if len(bad['selected']) != bad['n_select']:
        bad['selected'] = bad['selected'][:-1] + [i for i in range(120)
                                                 if i not in bad['selected']][:1]
    _bad_problems = L.sampling_plan_problems(bad)
    assert any('plan_fp' in x or 'stratum n_selected' in x or 'selected has' in x
               for x in _bad_problems), _bad_problems
    # A plan cannot claim more strata than units (which would imply an empty
    # stratum); the fingerprint is recomputed so only the structure fails.
    fake = json.loads(json.dumps(p1))
    fake['n_strata'] = fake['n_units'] + 1
    fake['strata'].append({'index': len(fake['strata']), 'n_units': 0,
                           'n_selected': 0})
    fake['plan_fp'] = L._sampling_plan_fp(fake)
    _fake_problems = L.sampling_plan_problems(fake)
    assert any('n_strata' in x or 'no units' in x for x in _fake_problems), _fake_problems
    # Public error paths are explicit, not silent coercions.
    for kwargs in ({'n_select': 0}, {'n_select': 121}, {'seed': True},
                   {'n_strata': 0}, {'allocation': 'nope'}):
        try:
            _kw = {'n_select': 24}
            _kw.update(kwargs)
            L.document_stratified_plan(sizes, **_kw)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'bad plan input accepted: {kwargs}')
    try:
        L.document_stratified_plan([1.0, float('nan')], 1)
    except ValueError:
        pass
    else:
        raise AssertionError('NaN size accepted')
    picked = L.select_sampled_items(list(range(120)), p1)
    assert picked == p1['selected'], picked
    # Indexable containers are indexed directly (no full copy); generators
    # still work through the list fallback.
    assert L.select_sampled_items(range(120), p1) == p1['selected']
    assert L.select_sampled_items(iter(range(120)), p1) == p1['selected']
    # A capped size-weighted allocation stays exact and fast; the remainder
    # is distributed in bulk instead of one unit per pass.
    alloc = L._sampling_allocation(
        400, 400, [100, 100, 100, 100], 'size_weighted', 1,
        stratum_size_sums=[10 ** 9, 1, 1, 1])
    assert int(alloc.sum()) == 400, alloc
    assert int(alloc[0]) == 100 and (alloc >= 1).all(), alloc
    assert (alloc <= 100).all(), alloc
    # An infeasible per-stratum floor is an explicit error, never silently
    # lowered to the stratum's population.
    try:
        L._sampling_allocation(12, 12, [1, 5, 6], 'equal', 2)
    except ValueError as e:
        assert 'floor' in str(e), e
    else:
        raise AssertionError('infeasible min_per_stratum silently accepted')


def test_wikitext_article_sampling_selects_whole_articles():
    rows = []
    for a in range(8):
        rows.append(f' = Article {a} = ')
        rows.extend([f'body {a}.{j}' for j in range(a % 4)])
        rows.append('')
    rep = L.wikitext_article_sampling_plan(rows, 4, seed=3, n_strata=2)
    assert rep['status'] == 'ok' and rep['n_articles_available'] == 8, rep
    groups = L.wikitext_article_groups(rows)['groups']
    selected_articles = set(rep['selected_article_indices'])
    expected_rows = sorted(i for a in selected_articles for i in groups[a])
    assert rep['selected_row_indices'] == expected_rows, rep
    assert len(rep['selected_article_ids']) == 4, rep
    assert L.sampling_plan_problems(rep['plan']) == [], rep['plan']
    # No headings means no article unit; the sampler refuses instead of
    # treating each line as a pseudo-article.
    try:
        L.wikitext_article_sampling_plan(['a', 'b', 'c'], 2)
    except ValueError as e:
        assert 'top-level headings' in str(e), e
    else:
        raise AssertionError('line-only rows accepted as articles')


def test_sample_design_audit_reports_prefix_and_probability_plans():
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, 'raw')
        cache = os.path.join(td, 'cache')
        os.makedirs(raw)
        os.makedirs(cache)
        train_rows = []
        for a in range(16):
            train_rows.extend([f' = Train {a} = ', f'body {a}', f'more {a}'])
        val_rows = [' = Val 0 = ', 'v0', ' = Val 1 = ', 'v1']
        with open(os.path.join(raw, 'wiki.train.raw'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(train_rows) + '\n')
        with open(os.path.join(raw, 'wiki.valid.raw'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(val_rows) + '\n')
        tag = 'v5_sl8_cap100_v80_vs4'
        _id = lambda r: int(L._doc_id_u64(L._doc_exact_fp(r)))
        np.save(os.path.join(cache, f'train_doc_ids_{tag}.npy'),
                np.asarray([_id(r) for r in train_rows[:24]], dtype=np.uint64))
        np.save(os.path.join(cache, f'val_doc_ids_{tag}.npy'),
                np.asarray([_id(r) for r in val_rows], dtype=np.uint64))
        prefix = {
            'train': {'design': 'source_order_prefix', 'source_rows': len(train_rows),
                      'rows_served': 24, 'tokens_served': 100, 'token_cap': 100,
                      'cap_reached': True, 'representativeness_claim_allowed': False,
                      'population_claim_allowed': False},
            'validation': {'design': 'source_order_prefix',
                           'source_rows': len(val_rows), 'rows_served': len(val_rows),
                           'tokens_served': 100, 'token_cap': 100,
                           'cap_reached': True,
                           'representativeness_claim_allowed': False,
                           'population_claim_allowed': False},
        }
        meta_path = os.path.join(cache, f'meta_{tag}.json')
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': prefix}, fh)
        rep = A.audit_sample_design(cache, raw_dir=raw)
        assert rep['status'] == 'warnings', rep
        train = rep['caches'][os.path.abspath(cache)]['train']
        assert train['sampling_class'] == 'non_probability_prefix', train
        assert train['representativeness_claim_allowed'] is False, train
        assert train['row_coverage_fraction'] == 24 / len(train_rows), train
        rc = train['raw_coverage']
        assert rc['status'] == 'ok' and rc['raw_articles'] == 16, rc
        assert rc['article_coverage_fraction'] == 0.5, rc
        assert 'wiki.test.raw' in rep['note'], rep
        # An optimistic claim on a prefix is a hard failure, not a footnote.
        bad = json.loads(json.dumps(prefix))
        bad['train']['representativeness_claim_allowed'] = True
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': bad}, fh)
        rep_bad = A.audit_sample_design(cache, raw_dir=raw)
        assert rep_bad['status'] == 'fail', rep_bad
        assert any('optimistic claim' in p for p in rep_bad['problems']), rep_bad
        # A valid stratified plan is accepted; a population claim below the
        # 64-unit floor is refused even when representativeness is claimed.
        plan = L.document_stratified_plan([2] * 16, 8, seed=5, n_strata=4,
                                          source_fp='fixture')
        strat = {
            'train': {'design': 'stratified_document', 'plan': plan,
                      'source_fp': 'fixture', 'source_rows': 16,
                      'rows_served': 8, 'tokens_served': 100,
                      'token_cap': 100, 'cap_reached': True,
                      'representativeness_claim_allowed': True,
                      'population_claim_allowed': True},
            'validation': prefix['validation'],
        }
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': strat}, fh)
        rep_strat = A.audit_sample_design(cache, raw_dir=raw)
        assert rep_strat['status'] == 'fail', rep_strat
        assert any('64-unit floor' in p for p in rep_strat['problems']), rep_strat
        strat['train']['population_claim_allowed'] = False
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': strat}, fh)
        rep_ok = A.audit_sample_design(cache, raw_dir=raw)
        train = rep_ok['caches'][os.path.abspath(cache)]['train']
        assert train['status'] == 'ok', train
        assert train['representativeness_claim_allowed'] is True, train
        # A population claim is stronger than representativeness: metadata
        # that forbids the latter cannot authorise the former.
        _pop_plan = L.document_stratified_plan(
            [2] * 80, 80, seed=5, n_strata=4, source_fp='fixture80')
        _pop_block = {
            'train': {'design': 'stratified_document', 'plan': _pop_plan,
                      'source_fp': 'fixture80', 'source_rows': 80,
                      'rows_served': 80, 'tokens_served': 100,
                      'token_cap': 100, 'cap_reached': True,
                      'representativeness_claim_allowed': False,
                      'population_claim_allowed': True},
            'validation': prefix['validation'],
        }
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': _pop_block}, fh)
        rep_pop = A.audit_sample_design(cache, raw_dir=raw)
        assert rep_pop['status'] == 'fail', rep_pop
        assert any('requires representativeness_claim_allowed=true' in p
                   for p in rep_pop['problems']), rep_pop
        _pop_block['train']['representativeness_claim_allowed'] = True
        _pop_block['train']['population_claim_allowed'] = False
        _pop_block['train']['rows_served'] = 81
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': _pop_block}, fh)
        rep_rows = A.audit_sample_design(cache, raw_dir=raw)
        assert rep_rows['status'] == 'fail', rep_rows
        assert any('exceeds source_rows' in p for p in rep_rows['problems']), rep_rows
        # Restore a valid train block before the test-only read check.
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump({'vocab': 80, 'seq_len': 8, 'sampling': strat}, fh)
        # Test metadata is audited without ever opening wiki.test.raw.
        test_meta = {'vocab': 80, 'seq_len': 8, 'sampling': {
            'test': {'design': 'source_order_prefix', 'source_rows': 64,
                     'rows_served': 24, 'tokens_served': 192,
                     'token_cap': 192, 'cap_reached': True,
                     'representativeness_claim_allowed': False,
                     'population_claim_allowed': False}}}
        with open(os.path.join(cache, 'test_meta_v5_sl8_v80_ns24.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(test_meta, fh)
        rep_test = A.audit_sample_design(cache, raw_dir=raw)
        trec = rep_test['caches'][os.path.abspath(cache)]['test']
        assert trec['raw_coverage']['status'] == 'not_applicable', trec
        assert 'wiki.test.raw' in trec['raw_coverage']['reason'], trec


def test_g0_certificate_requires_real_booleans():
    # The G0 gate must fail closed on malformed certificates: a non-empty
    # string such as "false" is truthy, so a plain truthiness test would let
    # a corrupted/hand-edited certificate authorize every scale panel.
    base = {
        'status': 'pass',
        'numeric_pass': True,
        'benchmark_pass': True,
        'training_tost_pass': True,
        'code_semantics': L.CODE_SEMANTICS,
        'kernel_stamp': L._kernel_stamp(),
    }
    assert L._g0_mismatch(dict(base)) is None, L._g0_mismatch(dict(base))
    for key in ('numeric_pass', 'benchmark_pass', 'training_tost_pass'):
        for bogus in ('false', 'true', 1, 0, [], {}):
            cert = dict(base)
            cert[key] = bogus
            why = L._g0_mismatch(cert)
            assert why is not None, (key, bogus)


def test_design_preflight_missing_manifest_fails_closed():
    # A missing/truncated frozen manifest is a hard design failure, not a
    # "not_verifiable" warning that still exits 0: the CLI treats only
    # status=fail as a red gate, so the violations list alone is not enough.
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, 'missing_manifest.json')
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_preflight(manifest_path=missing)
        assert rep['status'] == 'fail', rep.get('status')
        assert any('missing/unreadable' in v for v in rep['violations']), rep
        bad = os.path.join(td, 'truncated_manifest.json')
        with open(bad, 'w', encoding='utf-8') as f:
            f.write('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            rep_bad = A.audit_design_preflight(manifest_path=bad)
        assert rep_bad['status'] == 'fail' and rep_bad['violations'], rep_bad


def test_p5t_health_gate_uses_a_status_whitelist():
    # The one-shot test gate may only proceed for statuses the policy names.
    # An unrecognized/renamed/uppercase status must fail closed; the old
    # blacklist let any string outside _HEALTH_BLOCKING pass as non-blocking.
    assert V12._health_blocks('healthy') is False
    for status in sorted(V12._HEALTH_NONBLOCKING):
        assert V12._health_blocks(status) is False, status
    for status in ('unknown', 'fine', 'HEALTHY', 'degraded', 'nonfinite',
                   'overfitting', 'validation_degrading', 'insufficient_data',
                   'unmonitored', None):
        assert V12._health_blocks(status) is True, status
    why = V12._training_health_gate(
        {}, 'arm', 0, {'curve_health': {'status': 'fine'}}, {})
    assert why and 'refusing' in why, why


def test_source_audit_rejects_lock_snapshot_disagreement():
    # The snapshot records the approved plan; a freeze lock records the tree
    # actually used at first test access.  A stale/mis-copied snapshot must
    # not make the audit green while the lock attests a different tree.
    cur = L.source_tree_manifest()
    with tempfile.TemporaryDirectory() as td:
        snap = os.path.join(td, 'plan_snapshot.json')
        with open(snap, 'w', encoding='utf-8') as f:
            json.dump({'source_tree': cur}, f)
        lock_b = _fake_source_tree('b')
        lp = os.path.join(td, '.freeze_lock.json')
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump({'manifest_sha256': 'AAA',
                       'source_tree': lock_b,
                       'source_tree_sha256': lock_b['sha256']}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            bad = A.audit_source_integrity(snapshot_path=snap, lock_path=lp)
        assert bad['status'] == 'fail', bad
        assert bad.get('source_lock_mismatches'), bad
        assert any('different source tree' in p for p in bad['problems']), bad
        # The same disagreement is found through result-dir discovery.
        with contextlib.redirect_stdout(io.StringIO()):
            bad2 = A.audit_source_integrity(snapshot_path=snap, results_dirs=[td])
        assert bad2['status'] == 'fail' and bad2.get('source_lock_mismatches'), bad2
        # A matching lock keeps the normal byte-identical verdict.
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump({'manifest_sha256': 'AAA',
                       'source_tree': cur,
                       'source_tree_sha256': cur['sha256']}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            good = A.audit_source_integrity(snapshot_path=snap, lock_path=lp)
        assert good['status'] == 'ok' and not good['problems'], good
        # Two discovered locks bound to different trees cannot both describe
        # the same measurement source.
        td_b = os.path.join(td, 'results_b')
        os.mkdir(td_b)
        lp_b = os.path.join(td_b, '.freeze_lock.json')
        with open(lp_b, 'w', encoding='utf-8') as f:
            json.dump({'manifest_sha256': 'BBB',
                       'source_tree': lock_b,
                       'source_tree_sha256': lock_b['sha256']}, f)
        td_a = os.path.join(td, 'results_a')
        os.mkdir(td_a)
        lp_a = os.path.join(td_a, '.freeze_lock.json')
        with open(lp_a, 'w', encoding='utf-8') as f:
            json.dump({'manifest_sha256': 'AAA',
                       'source_tree': cur,
                       'source_tree_sha256': cur['sha256']}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            mixed = A.audit_source_integrity(snapshot_path=snap,
                                             results_dirs=[td_a, td_b])
        assert mixed['status'] == 'fail' and mixed.get('source_lock_mismatches'), mixed


def test_eval_length_truncated_beyond_panel_is_refused():
    # A truncated request still needs the full Ln prefix.  When the panel is
    # shorter than Ln, clamping to the panel width silently changes the causal
    # context instead of reporting the unavailable length.
    torch.manual_seed(0)
    cfgs = L.make_layer_cfgs(1, 'full_rope')
    model = V7.SmallGPTRoPE(32, 16, 1, 2, 8, 20, cfgs).eval()
    ids = np.asarray(np.random.default_rng(3).integers(
        0, 32, size=(2, 20)), dtype=np.int64)
    dev = torch.device('cpu')
    out = V7.eval_length_gen(model, ids, [24, 32], dev, max_pos=16)
    for Ln in (24, 32):
        assert 'error' in out[Ln], (Ln, out[Ln])
        assert out[Ln].get('truncated') is False, out[Ln]
    # A length that really fits keeps the intended last-_eff_max slice.
    valid = V7.eval_length_gen(model, ids, [16, 20], dev, max_pos=8)
    for Ln in (16, 20):
        cell = valid[Ln]
        assert 'ppl' in cell and cell['truncated'] is True, (Ln, cell)
        assert cell['eval_span'] == 7, (Ln, cell)


def test_article_cluster_map_survives_a_capped_heading():
    # The loader's token cap can cut a raw row.  The cold-build helper and the
    # decode-from-cache helper must produce the SAME article cluster map; a
    # truncated heading must not be folded into the previous article.
    rows = [' = Art 0 = ', ' = Art 1 = ']
    tok = L.build_tokenizer(rows, 128)
    eos = tok.token_to_id('<eos>')
    lengths = []
    ids = L._tokenise_row_chunks([rows], 10 ** 9, 'test', tok, eos,
                                 lengths_out=lengths)
    line_map = np.repeat(
        [L._doc_id_u64(L._doc_exact_fp(r)) for r in rows],
        np.asarray(lengths, dtype=np.int64))
    first_len = int(lengths[0])
    for cap in range(first_len + 1, int(ids.size)):
        cold = L.wikitext_token_article_ids_for_rows(rows, lengths, cap)
        cached = L.wikitext_token_article_ids_from_line_map(
            ids[:cap], line_map[:cap], tok)
        assert cold['status'] == cached['status'] == 'ok', (cap, cold, cached)
        assert np.array_equal(cold['token_article_ids'],
                              cached['token_article_ids']), cap
        assert cached['last_run_partial'] is True, cap
        # The partial heading is its own line-level unit, not article 0.
        assert len(np.unique(cached['token_article_ids'])) == 2, (cap, cached)
    # Blank rows have identical stripped content; each still needs its own
    # cluster id, otherwise unrelated blank lines are one pseudo-document.
    rid = L.wikitext_article_row_ids(['', ' = A = ', '', ''])
    blanks = rid['row_article_id'][[0, 2, 3]]
    assert len(set(int(x) for x in blanks)) == 3, rid['row_article_id']


def test_p5t_checkpoint_identity_binds_arm_seed_and_weights():
    sd = {'w': torch.tensor([1.0, 2.0])}
    sha = L.state_dict_sha256(sd)
    base = {'variant': 'arm_a', 'seed': 0,
            'state_dict_sha256': sha, 'sd': sd}
    assert V12._checkpoint_identity_problem(base, 'arm_a', 0) is None
    cases = [
        (dict(base, variant='arm_b'), 'arm_b'),
        (dict(base, seed=1), 'seed'),
        (dict(base, state_dict_sha256='0' * 64), 'digest'),
        ({k: v for k, v in base.items() if k != 'state_dict_sha256'},
         'state_dict_sha256'),
        (dict(base, sd={'w': torch.tensor([9.0, 9.0])}), 'edited'),
    ]
    for ckpt, needle in cases:
        why = V12._checkpoint_identity_problem(ckpt, 'arm_a', 0)
        assert why, (needle, ckpt)
    # The requested identity itself is checked, not just the stored one.
    assert V12._checkpoint_identity_problem(base, 'arm_a', 1), 'seed mismatch'


def test_kernel_fp64_and_negative_topk_guards():
    import csa_kernels as C
    q64 = torch.zeros((1, 2, 1, 16), dtype=torch.float64)
    assert C._attn_impl_for_shape('triton', q64, 8) == 'ref'
    k64 = torch.zeros((1, 2, 4), dtype=torch.float64)
    assert C._indexer_impl_for_shape('triton', k64) == 'ref'
    # A negative topk is clamped to zero like the eager helper, rather than
    # requesting a tensor with a negative dimension.
    R, T, Bm, nIH, hd = 1, 2, 2, 1, 4
    qI = torch.randn(R, T, nIH, hd)
    kI = torch.randn(R, Bm, nIH, hd)
    w_idx = torch.randn(R, T, nIH)
    last_tok = torch.tensor([[1, 2]], dtype=torch.int64)
    idx, valid, soft = C._indexer_fwd_ref(
        qI, kI, w_idx, last_tok, -1, need_soft=False)
    assert tuple(idx.shape) == (R, T, 0), idx.shape
    assert tuple(valid.shape) == (R, T, 0), valid.shape
    assert soft is None


def test_interaction_sesoi_design_gate():
    rep = L.interaction_sesoi_audit()
    assert rep['status'] == 'ok', rep
    assert set(rep['factorials']) == set(L.FACTORIAL_DESIGN), rep
    for name, rec in rep['factorials'].items():
        assert rec['declared'] is True and rec['predicted_sign'] == 1, (name, rec)
        assert math.isfinite(rec['margin_log_ratio']), (name, rec)
        assert abs(rec['margin_log_ratio'] - math.log(1.01)) < 1e-12, (name, rec)
    # Absolute log-ratio margins are accepted and stay on the log scale.
    spec = dict(L.INTERACTION_SESOI)
    spec['dynamic_blocking_x_attention_sink'] = {
        'absolute_log_ratio': 0.02, 'rationale': 'unit probe'}
    absrep = L.interaction_sesoi_audit(sesoi=spec)
    assert absrep['status'] == 'ok', absrep
    assert abs(absrep['factorials']['dynamic_blocking_x_attention_sink']
               ['margin_log_ratio'] - 0.02) < 1e-15, absrep
    # A missing margin cannot silently fall back to "not significant".
    partial = {k: v for k, v in L.INTERACTION_SESOI.items()
               if k != 'dynamic_blocking_x_attention_sink'}
    missing = L.interaction_sesoi_audit(sesoi=partial)
    assert missing['status'] == 'fail', missing
    assert any('no pre-registered interaction SESOI' in p
               for p in missing['problems']), missing
    # A typo is surfaced instead of leaving the real factorial unmargined.
    extra = dict(L.INTERACTION_SESOI)
    extra['dynamic_blocking_x_attention_sin'] = {'relative_pct': 1.0}
    ex = L.interaction_sesoi_audit(sesoi=extra)
    assert any('not present in the design' in w for w in ex['warnings']), ex
    # Malformed/ambiguous margins are design defects.
    for bad in ({'dynamic_blocking_x_learned_selection': {'relative_pct': 0.0}},
                {'dynamic_blocking_x_learned_selection': {'relative_pct': -1.0}},
                {'dynamic_blocking_x_learned_selection': {'relative_pct': float('nan')}},
                {'dynamic_blocking_x_learned_selection': {'relative_pct': True}},
                {'dynamic_blocking_x_learned_selection': {
                    'relative_pct': 1.0, 'absolute_log_ratio': 1.0}},
                {'dynamic_blocking_x_learned_selection': {
                    'absolute_log_ratio': None}}):
        out = L.interaction_sesoi_audit(sesoi=bad)
        assert out['status'] == 'fail', (bad, out)
    # No factorial design means the interaction test is simply not applicable.
    assert L.interaction_sesoi_audit(design={})['status'] == 'not_applicable'
    # The CLI wrapper returns the same verdict and does not read results.
    with contextlib.redirect_stdout(io.StringIO()):
        cli = A.audit_interaction_sesoi()
    assert cli['status'] == 'ok', cli
    with contextlib.redirect_stdout(io.StringIO()):
        cli_bad = A.audit_interaction_sesoi(spec=partial)
    assert cli_bad['status'] == 'fail', cli_bad


def test_factorial_interaction_sesoi_decisions():
    name = 'dynamic_blocking_x_learned_selection'
    margin = math.log(1.01)

    def run(vals, **kw):
        return L.factorial_interaction_audit(
            recs=_factorial_records(vals), **kw)

    # The predicted positive interaction above the margin is supported only
    # after the interaction-family correction.
    r = run([0.05] * 7)
    f = r['factorials'][name]
    assert f['practical_decision'] == 'hypothesis_supported', f
    assert f['interaction']['significant_family'] is True, f['interaction']
    assert f['interaction']['p_tost_holm'] is not None, f['interaction']
    assert r['hypothesis_supported_interactions'] == [name], r
    assert r['equivalent_interactions'] == [], r
    # Opposite sign, above margin: a real direction conflict, not support.
    neg = run([-0.05] * 7)
    nf = neg['factorials'][name]
    assert nf['practical_decision'] == 'direction_conflict', nf
    assert neg['direction_conflict_interactions'] == [name], neg
    assert name not in neg['indeterminate_interactions'], neg
    # Detectable but inside the margin is sub-SESOI, never "useful" and never
    # "additive": a tiny p-value cannot turn it into either.
    sub = run([0.005] * 7)
    sf = sub['factorials'][name]
    assert sf['practical_decision'] == 'sub_sesoi', sf
    assert sf['interaction']['sesoi']['practical_significant'] is False, sf
    assert sub['sub_sesoi_interactions'] == [name], sub
    assert sub['equivalent_interactions'] == [], sub
    # A precise zero interaction inside the margin is positive evidence of
    # practical additivity, with a family-adjusted TOST p.
    add = run([0.002 if i % 2 else -0.002 for i in range(7)])
    af = add['factorials'][name]
    assert af['practical_decision'] == 'additive', af
    assert af['interaction']['p_holm'] == 1.0, af['interaction']
    assert math.isfinite(float(af['interaction']['p_tost_holm'])), af
    assert add['equivalent_interactions'] == [name], add
    assert add['sesoi']['family']['m'] == 1, add['sesoi']
    # A noisy, non-significant interaction is indeterminate, not additive.
    ind = run([0.025 if i % 2 else -0.015 for i in range(7)])
    assert ind['factorials'][name]['practical_decision'] == 'indeterminate', ind
    assert ind['indeterminate_interactions'] == [name], ind
    assert ind['equivalent_interactions'] == [], ind
    # n<min_n fails closed even if the point estimate is exactly zero.
    weak = L.factorial_interaction_audit(
        recs=_factorial_records([0.0, 0.0], n=2), min_n=3)
    assert weak['factorials'][name]['practical_decision'] == 'underpowered', weak
    # Missing a cell cannot be called additive or indeterminate; it is
    # not_measured and keeps its declared TOST family slot.
    incomplete = {k: v for k, v in _factorial_records(0.05).items()
                  if k[0] != 'csa_fix_randidx'}
    inc = L.factorial_interaction_audit(recs=incomplete)
    assert inc['factorials'][name]['practical_decision'] == 'not_measured', inc
    assert inc['not_measured_interactions'] == [name], inc
    # Empty SESOI mapping disables the practical layer explicitly.
    off = run([0.05] * 7, sesoi={})
    assert off['factorials'][name]['practical_decision'] == 'not_declared', off
    assert off['not_declared_interactions'] == [name], off
    assert off['equivalent_interactions'] == [], off
    assert off['sesoi']['family']['m'] == 0, off['sesoi']
    # A malformed-but-declared partial SESOI still keeps an unbeatable slot
    # for the undeclared factorial; the TOST family must not shrink and buy
    # significance for the one factorial that happens to have a margin.
    import copy
    base = copy.deepcopy(L.FACTORIAL_DESIGN[name])
    two = {'a': copy.deepcopy(base), 'b': copy.deepcopy(base)}
    partial_two = L.factorial_interaction_audit(
        design=two, recs=_factorial_records(
            [0.002 if i % 2 else -0.002 for i in range(7)]),
        sesoi={'a': {'relative_pct': 1.0}})
    assert partial_two['sesoi']['family']['m'] == 2, partial_two['sesoi']
    assert partial_two['factorials']['b']['practical_decision'] == 'not_declared', partial_two
    assert partial_two['not_declared_interactions'] == ['b'], partial_two
    malformed_cells = {'good': copy.deepcopy(base), 'bad': copy.deepcopy(base)}
    malformed_cells['bad']['cells'][('active', 'active')] = (
        malformed_cells['bad']['cells'][('active', 'control')])
    md = L.factorial_interaction_audit(
        design=malformed_cells, recs=_factorial_records(
            [0.002 if i % 2 else -0.002 for i in range(7)]),
        sesoi={k: {'relative_pct': 1.0} for k in malformed_cells})
    assert md['status'] == 'incomplete', md
    assert md['factorials']['bad']['status'] == 'incomplete', md['factorials']
    assert md['family']['m'] == 2, md['family']
    assert md['sesoi']['family']['m'] == 2, md['sesoi']
    # The no-complete-2x2-seed branch must not append the same name twice.
    allbad = {k: (dict(v, run_cfg='other_cfg')
                  if k[0] == 'csa_fix_randidx' else v)
              for k, v in _factorial_records(0.05).items()}
    rbad = L.factorial_interaction_audit(recs=allbad, sesoi={})
    assert rbad['not_declared_interactions'] == [name], rbad
    assert rbad['factorials'][name]['status'] == 'no_data', rbad
    # A malformed SESOI fails the audit instead of silently dropping a margin.
    bad = run([0.05] * 7, sesoi={name: {'relative_pct': 0.0}})
    assert bad['status'] == 'fail' and bad['sesoi']['error'], bad
    # Repeating the same factorial three times exercises the Holm correction
    # on both the superiority and TOST families; m must be the declared 3,
    # not the one factorial we happened to inspect.
    three = {f'copy{i}': copy.deepcopy(base) for i in range(3)}
    r3 = L.factorial_interaction_audit(
        design=three, recs=_factorial_records(
            [0.002 if i % 2 else -0.002 for i in range(7)]),
        sesoi={k: {'relative_pct': 1.0} for k in three})
    assert r3['family']['m'] == 3 and r3['sesoi']['family']['m'] == 3, r3
    for rec in r3['factorials'].values():
        assert rec['practical_decision'] == 'additive', rec
        assert rec['interaction']['p_holm'] == 1.0, rec['interaction']
        assert (rec['interaction']['p_tost_holm']
                >= rec['interaction']['sesoi']['tost_p_tost']), rec['interaction']


def test_interaction_practical_decision_primitives():
    margin = math.log(1.01)
    r = L.interaction_practical_decision(
        [0.002, -0.002, 0.002, -0.002, 0.002, -0.002],
        margin=margin, direction=1)
    assert r['decision'] == 'additive' and r['claim'] == 'additive_within_sesoi', r
    assert r['tost_equivalent'] is True and r['tost_p_tost'] < 0.05, r
    assert r['margin_ratio'] is not None and abs(r['margin_ratio'] - 1.01) < 1e-12, r
    # A one-pass iterable feeds both the superiority and TOST tests.
    again = L.interaction_practical_decision(
        iter([0.002, -0.002, 0.002, -0.002, 0.002, -0.002]),
        margin=margin, direction=1)
    assert again['n'] == 6 and again['decision'] == 'additive', again
    # No margin means no additivity claim.
    off = L.interaction_practical_decision([0.002, -0.002, 0.0], sesoi={})
    assert off['decision'] == 'not_declared' and off['margin'] is None, off
    # Relative specs are converted on the log-ratio scale.
    rel = L.interaction_practical_decision(
        [0.002, -0.002, 0.002, -0.002, 0.002, -0.002],
        sesoi={'relative_pct': 1.0}, direction=1)
    assert rel['decision'] == 'additive', rel
    assert abs(rel['margin'] - margin) < 1e-12, rel
    # Malformed family p-values fail closed: a raw TOST alone cannot certify
    # equivalence when the supplied family p is invalid.
    malformed = L.interaction_practical_decision(
        [0.002, -0.002, 0.002, -0.002, 0.002, -0.002],
        margin=margin, direction=1, p_holm_tost='false')
    assert malformed['decision'] != 'additive', malformed
    malformed2 = L.interaction_practical_decision(
        [0.05] * 7, margin=margin, direction=1, p_holm='false')
    assert malformed2['decision'] != 'hypothesis_supported', malformed2
    # A malformed superiority-family p must also block an additive claim: the
    # same fail-closed rule applies even when the raw TOST interval is tight.
    malformed3 = L.interaction_practical_decision(
        [0.002, -0.002, 0.002, -0.002, 0.002, -0.002],
        margin=margin, direction=1, p_holm='false')
    assert malformed3['decision'] != 'additive', malformed3
    assert 'superiority p is malformed' in malformed3['reason'], malformed3
    # Direction and margin are validation gates, not coercions.
    for bad in (1.5, -0.5, float('nan'), True):
        try:
            L.interaction_practical_decision([0.1, 0.2, 0.3], margin=margin,
                                             direction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'fractional/bool interaction direction accepted: {bad!r}')
    for bad_margin in (0.0, -0.1, float('nan'), float('inf'), True):
        try:
            L.interaction_practical_decision([0.1, 0.2, 0.3], margin=bad_margin)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid interaction margin accepted: {bad_margin!r}')
    # The normaliser accepts its own output and rejects ambiguous units.
    norm = L._normalise_interaction_sesoi_design(
        {'x': {'relative_pct': 2.0}})['x']
    again2 = L._normalise_interaction_sesoi_design({'x': norm})['x']
    assert again2['relative_pct'] == 2.0 and again2['absolute_log_ratio'] is None, again2
    assert abs(norm['margin_log_ratio'] - math.log1p(0.02)) < 1e-15, norm


def test_sesoi_practical_decision_primitives():
    # Equivalence is a positive claim: a small, precise effect inside the
    # pre-registered margin is 'equivalent', not 'not significant'.
    r = L.ablation_practical_decision(
        [0.02, -0.01, 0.0, 0.01, -0.02, 0.0], margin=0.5,
        alpha=0.05, min_n=3, direction=-1)
    assert r['decision'] == 'equivalent' and r['claim'] == 'module_removable_within_sesoi', r
    assert r['tost_equivalent'] is True and r['tost_p_tost'] < 0.05, r
    # A detectable effect smaller than the margin is sub-SESOI, never
    # deliberately 'useful', even though the sign-flip p is tiny.
    r2 = L.ablation_practical_decision(
        [0.05, 0.051, 0.052, 0.053, 0.054, 0.055], margin=0.1,
        alpha=0.05, min_n=3, direction=-1)
    assert r2['decision'] == 'sub_sesoi' and r2['practical_significant'] is False, r2
    # Same magnitude in the declared-benefit direction is only 'superior' when
    # it exceeds the margin; this one does not, so it stays sub-SESOI too.
    r3 = L.ablation_practical_decision(
        [-0.05, -0.051, -0.052, -0.053, -0.054, -0.055], margin=0.1,
        alpha=0.05, min_n=3, direction=-1)
    assert r3['decision'] == 'sub_sesoi', r3
    # Opposite-direction, above-margin effect is harmful.
    r4 = L.ablation_practical_decision(
        [0.8, 0.81, 0.79, 0.82, 0.8, 0.81], margin=0.1,
        alpha=0.05, min_n=3, direction=-1)
    assert r4['decision'] == 'inferior' and r4['claim'] == 'module_harmful', r4
    # No pre-registered margin: no equivalence verdict at all.
    r5 = L.ablation_practical_decision([0.0, 0.0, 0.0, 0.0, 0.0], sesoi=None)
    assert r5['decision'] == 'not_declared' and r5['margin'] is None, r5
    # n=2 cannot support equivalence even if the point estimate is tiny.
    r6 = L.ablation_practical_decision([0.0, 0.0], margin=0.5, min_n=3)
    assert r6['decision'] == 'underpowered', r6
    # Relative margins translate against the without-arm PPL.
    spec = L._normalise_sesoi_design({'x': {'relative_pct': 2.0}})['x']
    margin, why = L._sesoi_margin_ppl(spec, 50.0)
    assert why == '' and abs(margin - 1.0) < 1e-12, (margin, why)
    # A normalised spec (both unit keys present, one None) can be normalised
    # again; the public helper accepts its own output.
    again = L._normalise_sesoi_design({'x': spec})['x']
    assert again['relative_pct'] == 2.0 and again['absolute_ppl'] is None, again
    assert L._sesoi_margin_ppl(spec, 0.0)[0] is None, 'a zero baseline cannot define a relative margin'
    # The public helper can derive the absolute margin from a relative spec.
    r7 = L.ablation_practical_decision([0.01, -0.01, 0.0, 0.02, -0.02, 0.0],
                                       sesoi={'relative_pct': 1.0},
                                       baseline_ppl=50.0, min_n=3)
    assert r7['decision'] == 'equivalent' and abs(r7['margin'] - 0.5) < 1e-12, r7
    # Malformed declarations fail closed instead of silently dropping a margin.
    for bad in ({'x': {'relative_pct': 0.0}},
                {'x': {'relative_pct': -1.0}},
                {'x': {'relative_pct': float('nan')}},
                {'x': {'relative_pct': True}},
                {'x': {'relative_pct': 1.0, 'absolute_ppl': 1.0}},
                {'x': {'absolute_ppl': None}}):
        try:
            L._normalise_sesoi_design(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed SESOI must be rejected: {bad!r}')

