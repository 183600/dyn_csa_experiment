"""Numerical-backend environment identity (env_fp_v3) and its gates.

``env_fp_v2`` binds the lock bytes and every pinned distribution.  These checks
pin down the complementary digest: the BLAS/LAPACK/SIMD backend inside the
pinned wheels and the versions of unpinned runtime distributions must move the
digest; host/process identity must not; a v3 digest is preferred only when both
sides carry it; and the record/audit plumbing exposes it without rewriting the
already-valid v2 identity of archived records.
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _env():
    return copy.deepcopy(L.environment_snapshot())


def _record(*, env='env', env_v2='v2', env_v3='v3', ppl=10.0, tokens=1000):
    prov = {
        'code': {
            'code_semantics': 'v11.128',
            'code_stamp': 'CS',
            'source_tree': {'sha256': 'a' * 64, 'n_files': 3, 'bytes': 100,
                            'scope': 'synthetic'},
        },
        'environment': {'python': '3.12.15', 'torch': '2.14.1+cpu'},
        'data': {
            'dataset': 'wikitext', 'data_fp': 'data', 'val_fp': 'val',
            'data_version': 'dv', 'corpus_fp': 'corpus',
        },
    }
    rec = {
        'variant': 'csa_dynamic', 'seed': 0, 'run_cfg': 'cfg_csCS',
        '_code': 'CS', 'exec_fp': env, 'init_core_fp': 'init', 'rng_fp': 'rng',
        'data_fp': 'data', 'val_fp': 'val', 'data_version': 'dv',
        'corpus_fp': 'corpus', 'ppl': float(ppl), 'n_tokens': int(tokens),
        'train_knobs': L.training_knobs(amp_dtype='fp32', eager_batch=False),
        'train_objective': {
            'comp_reg_applies': True, 'comp_lambda': 0.05,
            'comp_lambda_effective': 0.05, 'delta_lr_effective': 10.0,
            'boundary_policy': ['threshold'],
            'loss_terms': ['ce', 'comp_reg'],
        },
        'provenance': prov,
        'synthesized': False,
    }
    if env_v2 is not None:
        rec['env_fp_v2'] = env_v2
    if env_v3 is not None:
        rec['env_fp_v3'] = env_v3
    return rec


def test_snapshot_records_numerical_backend_and_v3():
    env = _env()
    assert env['env_fp_v3_schema'] == L.ENVIRONMENT_FP_V3_SCHEMA == 3
    assert isinstance(env['env_fp_v3'], str) and len(env['env_fp_v3']) == 16
    backend = env['numerical_backend']
    assert backend['numpy']['version'] == np.__version__
    # A normal Linux wheel exposes at least the BLAS or the runtime dispatch;
    # the fields may be None on an unusual build, but the block must exist.
    assert 'blas' in backend['numpy'] and 'lapack' in backend['numpy']
    assert 'build_report_sha256' in backend['torch']
    assert set(('mkl', 'mkldnn', 'openmp')).issubset(backend['torch'])


def test_v3_binds_blas_lapack_and_simd_backend():
    env = _env()
    base = L.environment_fingerprint_v3(env)
    assert len(base) == 16 and base == base.lower()

    mutations = [
        ('numpy.blas.version', lambda e: e['numerical_backend']['numpy']['blas'].__setitem__('version', '0.0.0-other')),
        ('numpy.lapack', lambda e: e['numerical_backend']['numpy'].__setitem__('lapack', {'name': 'mkl', 'version': '1.0'})),
        ('numpy.simd_runtime_dispatch', lambda e: e['numerical_backend']['numpy'].__setitem__('simd_runtime_dispatch', ['AVX2'])),
        ('numpy.build_config_sha256', lambda e: e['numerical_backend']['numpy'].__setitem__('build_config_sha256', '0' * 16)),
        ('torch.build_report_sha256', lambda e: e['numerical_backend']['torch'].__setitem__('build_report_sha256', '0' * 16)),
        ('torch.mkl', lambda e: e['numerical_backend']['torch'].__setitem__('mkl', not e['numerical_backend']['torch'].get('mkl'))),
        ('torch.openmp_version', lambda e: e['numerical_backend']['torch'].__setitem__('openmp_version', '1998')),
    ]
    for name, mutate in mutations:
        changed = _env()
        mutate(changed)
        assert L.environment_fingerprint_v3(changed) != base, (
            f'changing {name} did not change env_fp_v3')


def test_v3_binds_unpinned_runtime_distribution_versions():
    env = _env()
    base = L.environment_fingerprint_v3(env)
    changed = _env()
    changed['dependency_contract']['unpinned_versions'] = {'scipy': '1.14.0'}
    assert L.environment_fingerprint_v3(changed) != base
    upgraded = copy.deepcopy(changed)
    upgraded['dependency_contract']['unpinned_versions']['scipy'] = '1.15.0'
    assert L.environment_fingerprint_v3(upgraded) != L.environment_fingerprint_v3(changed)
    # Host/process identity and run-control flags are still not stack identity.
    for key, value in (('hostname', 'another-box'), ('process_id', 999999),
                       ('working_directory', '/tmp/elsewhere'),
                       ('pythonhashseed', '123'),
                       ('cuda_visible_devices', '3')):
        changed = _env()
        changed[key] = value
        assert L.environment_fingerprint_v3(changed) == base, key
    changed = _env()
    changed['env_flags'] = {'CSA_CPU_THREADS': '7'}
    assert L.environment_fingerprint_v3(changed) == base


def test_dependency_contract_exposes_unpinned_versions_for_tooling_free_stack():
    contract = L.dependency_contract()
    assert contract['status'] in ('ok', 'partial'), contract
    assert set(contract['unpinned_versions']) == set(contract['unpinned_installed'])
    for name in ('pip', 'setuptools', 'wheel', 'ruff', 'mypy', 'black'):
        assert name not in contract['unpinned_installed'], (
            f'declared tooling {name!r} must not count as an unpinned runtime '
            'distribution')
    for name, version in contract['unpinned_versions'].items():
        assert isinstance(name, str) and name
        assert version is None or isinstance(version, str)


def test_v3_requires_a_mapping_and_is_stable():
    try:
        L.environment_fingerprint_v3('not-a-snapshot')
    except TypeError as e:
        assert 'mapping' in str(e)
    else:
        raise AssertionError('a non-mapping snapshot must be refused')
    assert L.environment_fingerprint_v3() == L.environment_fingerprint_v3()
    # v3 is a separate generation, not a redefinition of v2.
    assert L.environment_fingerprint_v3() != L.environment_fingerprint_v2()


def test_pairing_prefers_v3_and_one_sided_v3_falls_back_to_v2():
    reason = L.execution_environment_reason(
        _record(env='same', env_v2='v2', env_v3='a' * 16),
        _record(env='same', env_v2='v2', env_v3='b' * 16))
    assert reason and 'env_fp_v3' in reason and 'BLAS' in reason

    # One-sided v3 is a provenance-generation difference, not a stack change:
    # the v2 digest remains the comparable identity (seed completion).
    assert L.execution_environment_reason(
        _record(env='same', env_v2='v2', env_v3='a' * 16),
        _record(env='same', env_v2='v2', env_v3=None)) is None
    # But a real v2 difference is still refused when v3 cannot arbitrate.
    reason = L.execution_environment_reason(
        _record(env='same', env_v2='one', env_v3='a' * 16),
        _record(env='same', env_v2='two', env_v3=None))
    assert reason and 'env_fp_v2' in reason


def test_reproduction_diagnosis_compares_v3_when_both_sides_have_it():
    same = L.reproduction_diagnosis(_record(env_v3='a' * 16),
                                    _record(env_v3='a' * 16))
    fields = {f['field']: f for f in same['dimensions']['environment']['fields']}
    assert fields['env_fp_v3']['status'] == 'match'
    assert same['verdict'] == 'identical'

    diff = L.reproduction_diagnosis(_record(env_v3='a' * 16),
                                    _record(env_v3='b' * 16))
    assert diff['dimensions']['environment']['status'] == 'mismatch'
    assert diff['verdict'] == 'environment_changed'
    assert [f for f in diff['dimensions']['environment']['fields']
            if f['field'] == 'env_fp_v3'][0]['status'] == 'mismatch'

    one = L.reproduction_diagnosis(_record(env_v3='a' * 16),
                                   _record(env_v3=None))
    assert one['dimensions']['environment']['status'] == 'unverified'
    assert one['verdict'] == 'provenance_incomplete'
    assert any('predates env_fp_v3' in note
               for note in one['dimensions']['environment'].get('notes') or [])

    # Legacy comparisons that never had v3 are unchanged: no extra field.
    legacy = L.reproduction_diagnosis(_record(env_v3=None), _record(env_v3=None))
    assert legacy['verdict'] == 'identical'
    assert 'env_fp_v3' not in {f['field']
                               for f in legacy['dimensions']['environment']['fields']}


def test_hyperparameters_and_run_manifest_carry_v3():
    hp = L.run_hyperparameters(
        {}, seed=0, d=8, n_layers=1, n_heads=1, d_head=8, vocab=97,
        mlp_ratio=2.0, amp_dtype='fp32', eager_batch=False)
    assert hp['env_fp_v3'] == L.environment_fingerprint_v3()
    assert hp['env_fp_v3_schema'] == L.ENVIRONMENT_FP_V3_SCHEMA

    d = tempfile.mkdtemp()
    try:
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        summary = {'full::seed0': {
            'ppl': 10.0, 'steps': 10, 'tokens_seen': 320, 'params': 100,
            'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc',
            '_code': L.code_stamp(), 'losses': [3.0, 2.3],
            'ppl_history': [[10, 10.0]], 'hyperparams': hp,
            'data_fp': prov['data'].get('train_fp'),
            'data_version': prov['data']['data_version'],
            'env_fp_v3': hp['env_fp_v3'],
        }}
        path = L.write_run_manifest(d, cfg, 'FP', prov, summary)
        entry = json.load(open(path, encoding='utf-8'))['panels']['FP']
        assert entry['cells']['full::seed0']['env_fp_v3'] == hp['env_fp_v3']
        assert entry['tracking_integrity']['with_env_fp_v3'] == 1
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def _write_panel(root, records):
    panel = os.path.join(root, 'results_v3')
    os.makedirs(panel, exist_ok=True)
    summary = {}
    for i, rec in enumerate(records):
        r = copy.deepcopy(rec)
        r['variant'] = 'csa_dynamic'
        r['seed'] = i
        summary[f'csa_dynamic::seed{i}'] = r
    with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh)
    return panel


def test_environment_identity_audit_reports_v3_without_downgrading_legacy():
    local_v2 = L.environment_fingerprint_v2()
    local_v3 = L.environment_fingerprint_v3()
    with tempfile.TemporaryDirectory() as td:
        panel = _write_panel(td, [
            _record(env_v2=local_v2, env_v3=local_v3),
            _record(env_v2=local_v2, env_v3='other-v3'),
        ])
        out = A.audit_environment_identity([panel])
    assert out['status'] == 'warning', out
    assert out['n_with_v3'] == 2 and out['n_match_v3'] == 1
    assert out['n_different_v3'] == 1
    assert any('env_fp_v3' in w for w in out['warnings'])

    # v2 still matches for records that predate v3; their absence must not
    # turn a healthy v2 panel into a warning.
    with tempfile.TemporaryDirectory() as td:
        panel = _write_panel(td, [_record(env_v2=local_v2, env_v3=None)])
        out = A.audit_environment_identity([panel])
    assert out['status'] == 'ok', out
    assert out['n_with_v3'] == 0 and out['n_missing_v3'] == 1


def test_environment_identity_audit_merges_v3_conflicts():
    local_v2 = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'results_v3_conflict')
        os.makedirs(panel)
        manifest = {'cells': {
            'csa_dynamic::seed0': {'env_fp_v2': local_v2, 'env_fp_v3': 'a' * 16,
                                   'seed': 0},
        }}
        with open(os.path.join(panel, 'run_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh)
        summary = {'csa_dynamic::seed0': {'env_fp_v2': local_v2,
                                          'env_fp_v3': 'b' * 16,
                                          'seed': 0, 'ppl': 10.0}}
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh)
        out = A.audit_environment_identity([panel])
    conflicts = out['panels']['results_v3_conflict']['conflicts']
    assert conflicts and conflicts[0].get('field') == 'env_fp_v3', out


def test_release_identity_records_both_strict_generations():
    import archive_release as AR
    ident = AR._release_identity(
        L.source_tree_manifest(include=['exp_lib.py']))
    assert ident['env_fp_v2'] == L.environment_fingerprint_v2()
    assert ident['env_fp_v3'] == L.environment_fingerprint_v3()
    backend = ident['numerical_backend']
    assert backend['numpy']['version'] == np.__version__
    assert backend['torch']['build_report_sha256'] is not None


def test_analysis_report_diff_binds_v3_backend():
    base = L.stamped_analysis('unit_report', {'answer': 1})
    same = L.stamped_analysis('unit_report', {'answer': 1})
    assert L.analysis_report_diff(base, same)['verdict'] == 'reproduced'
    other = copy.deepcopy(same)
    other['provenance']['environment']['env_fp_v3'] = '0' * 16
    rep = L.analysis_report_diff(base, other)
    assert rep['verdict'] == 'provenance_changed', rep
    assert rep['dimensions']['environment']['fields']['env_fp_v3'] == 'mismatch'
