"""Strict declared-environment identity (env_fp_v2) and its gates.

The legacy ``exec_fp`` binds six direct packages.  These checks pin down the
complementary digest: it must cover every lock-pinned distribution, the lock
bytes and the OS/driver block; it must ignore host/process identity; a
one-sided legacy record must stay unverified rather than become a match; and
the reproduction/audit callers must surface the difference.
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


def _env(**overrides):
    env = L.environment_snapshot()
    env = copy.deepcopy(env)
    env.update(overrides)
    return env


def _record(*, env='env', env_v2='v2', ppl=10.0, tokens=1000):
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
    return rec


def test_contract_exposes_every_lock_pin():
    contract = L.dependency_contract()
    parsed = L.parse_requirements_lock()
    assert contract['status'] in ('ok', 'partial'), contract
    assert contract['n_pinned'] == len(parsed['pins']) > 0
    assert set(contract['pinned_installed']) == set(parsed['pins'])
    # Every installed pin is a version string; a genuinely missing pin is
    # recorded as null rather than being dropped from the identity.
    for name, version in contract['pinned_installed'].items():
        assert version is None or isinstance(version, str), (name, version)


def test_environment_fingerprint_covers_transitive_pins():
    env = _env()
    base = L.environment_fingerprint_v2(env)
    # A transitive pin (not one of the six packages in the legacy exec_fp)
    # must move the digest.  Pick an installed non-direct dependency.
    direct = {'torch', 'numpy', 'tokenizers', 'datasets', 'matplotlib',
              'tensorflow'}
    candidates = [n for n, v in env['dependency_contract']['pinned_installed'].items()
                  if v is not None and n not in direct]
    assert candidates, 'no installed transitive pin to exercise the digest'
    changed = copy.deepcopy(env)
    name = candidates[0]
    changed['dependency_contract']['pinned_installed'][name] = '0.0.0-not-the-lock'
    assert L.environment_fingerprint_v2(changed) != base, (
        f'changing transitive pin {name!r} did not change env_fp_v2')


def test_environment_fingerprint_binds_lock_os_and_determinism():
    env = _env()
    base = L.environment_fingerprint_v2(env)

    changed = copy.deepcopy(env)
    changed['dependency_contract']['lock_sha256'] = 'b' * 64
    assert L.environment_fingerprint_v2(changed) != base

    changed = copy.deepcopy(env)
    changed['os_identity']['nvidia_driver'] = '999.99'
    assert L.environment_fingerprint_v2(changed) != base

    changed = copy.deepcopy(env)
    changed['deterministic_algorithms'] = not changed['deterministic_algorithms']
    assert L.environment_fingerprint_v2(changed) != base

    changed = copy.deepcopy(env)
    changed['cpu_threads'] = int(changed['cpu_threads']) + 1
    assert L.environment_fingerprint_v2(changed) != base


def test_environment_fingerprint_ignores_host_and_process_identity():
    env = _env()
    base = L.environment_fingerprint_v2(env)
    changed = copy.deepcopy(env)
    changed['hostname'] = 'another-box'
    changed['process_id'] = 999999
    changed['working_directory'] = '/tmp/elsewhere'
    changed['invocation'] = {'argv': ['other'], 'entrypoint': 'other'}
    changed['pythonhashseed'] = '12345'
    changed['env_flags'] = {'CSA_CPU_THREADS': '7'}
    changed['cuda_visible_devices'] = '3'
    assert L.environment_fingerprint_v2(changed) == base, (
        'host/process/run-control identity must not change the stack digest')


def test_environment_fingerprint_requires_a_mapping():
    try:
        L.environment_fingerprint_v2('not-a-snapshot')
    except TypeError as e:
        assert 'mapping' in str(e)
    else:
        raise AssertionError('a non-mapping snapshot must be refused')


def test_environment_fingerprint_is_stable_across_calls():
    a = L.environment_fingerprint_v2()
    b = L.environment_fingerprint_v2()
    assert a == b and len(a) == 16 and a == a.lower()


def test_execution_environment_reason_prefers_v2():
    reason = L.execution_environment_reason(
        _record(env='same', env_v2='a' * 16),
        _record(env='same', env_v2='b' * 16))
    assert reason and 'declared environment' in reason and 'env_fp_v2' in reason

    # v2 matches but the legacy digest differs: the legacy gate still refuses.
    reason = L.execution_environment_reason(
        _record(env='a', env_v2='same'), _record(env='b', env_v2='same'))
    assert reason and 'execution environment' in reason

    # Both v2 equal: no environment reason.
    assert L.execution_environment_reason(
        _record(env='same', env_v2='same'),
        _record(env='same', env_v2='same')) is None


def test_execution_environment_reason_one_sided_v2_is_not_a_mismatch():
    # A seed-completion run (new) paired with a legacy arm (old) must not be
    # refused by the pairing gate just because only one side carries v2.
    assert L.execution_environment_reason(
        _record(env='same', env_v2='new'), _record(env='same', env_v2=None)) is None
    # But two present-but-different v2 digests are a definite mismatch.
    assert L.execution_environment_reason(
        _record(env='same', env_v2='new'), _record(env='same', env_v2='old'))


def test_reproduction_diagnosis_adds_a_strict_environment_field():
    same = L.reproduction_diagnosis(_record(env_v2='v2'),
                                    _record(env_v2='v2'))
    fields = {f['field']: f for f in same['dimensions']['environment']['fields']}
    assert fields['exec_fp']['status'] == 'match'
    assert fields['env_fp_v2']['status'] == 'match'
    assert same['verdict'] == 'identical'

    diff = L.reproduction_diagnosis(_record(env_v2='one'),
                                    _record(env_v2='two'))
    assert diff['dimensions']['environment']['status'] == 'mismatch'
    assert diff['verdict'] == 'environment_changed'
    field = [f for f in diff['dimensions']['environment']['fields']
             if f['field'] == 'env_fp_v2'][0]
    assert field['status'] == 'mismatch'

    one = L.reproduction_diagnosis(_record(env_v2='one'),
                                   _record(env_v2=None))
    assert one['dimensions']['environment']['status'] == 'unverified'
    assert one['verdict'] == 'provenance_incomplete'
    assert any('predates env_fp_v2' in note
               for note in one['dimensions']['environment'].get('notes') or [])

    legacy = L.reproduction_diagnosis(_record(env_v2=None), _record(env_v2=None))
    assert legacy['verdict'] == 'identical'
    assert any('legacy six-package exec_fp' in note
               for note in legacy['dimensions']['environment'].get('notes') or [])


def test_run_hyperparameters_carries_the_same_v2_digest():
    hp = L.run_hyperparameters(
        {}, seed=0, d=8, n_layers=1, n_heads=1, d_head=8, vocab=97,
        mlp_ratio=2.0, amp_dtype='fp32', eager_batch=False)
    assert hp['env_fp_v2'] == L.environment_fingerprint_v2()
    assert hp['exec_fp'] == L.execution_fingerprint()


def _write_summary(root, records):
    panel = os.path.join(root, 'results_synthetic')
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


def test_environment_identity_audit_without_records_is_not_verifiable():
    with tempfile.TemporaryDirectory() as td:
        out = A.audit_environment_identity([td])
    assert out['status'] == 'not_verifiable_locally', out
    assert out['n_records'] == 0


def test_environment_identity_audit_compares_recorded_cells():
    local = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = _write_summary(td, [_record(env_v2=local), _record(env_v2='other')])
        out = A.audit_environment_identity([panel])
    assert out['status'] == 'warning', out
    assert out['n_panels'] == 1 and out['n_with_v2'] == 2
    assert out['n_match'] == 1 and out['n_different_from_local'] == 1
    assert out['n_missing_v2'] == 0
    assert any('different declared environment' in w for w in out['warnings'])


def test_environment_identity_audit_all_match_is_ok():
    local = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = _write_summary(td, [_record(env_v2=local), _record(env_v2=local)])
        out = A.audit_environment_identity([panel])
    assert out['status'] == 'ok', out
    assert out['n_match'] == 2 and out['n_different_from_local'] == 0


def test_environment_identity_audit_legacy_records_are_unverified():
    with tempfile.TemporaryDirectory() as td:
        panel = _write_summary(td, [_record(env_v2=None)])
        out = A.audit_environment_identity([panel])
    assert out['status'] == 'not_verifiable_locally', out
    assert out['n_missing_v2'] == 1 and out['n_with_v2'] == 0
    assert any('predate env_fp_v2' in w for w in out['warnings'])


def test_environment_identity_audit_reads_run_manifest_cells():
    local = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'results_manifest')
        os.makedirs(panel)
        manifest = {'cells': {
            'csa_dynamic::seed0': {'env_fp_v2': local, 'seed': 0},
            'csa_dynamic::seed1': {'hyperparams': {'env_fp_v2': 'other'}, 'seed': 1},
        }}
        with open(os.path.join(panel, 'run_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh)
        out = A.audit_environment_identity([panel])
    assert out['n_with_v2'] == 2 and out['n_match'] == 1
    assert out['n_different_from_local'] == 1


def test_environment_identity_audit_deduplicates_manifest_and_summary():
    local = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'results_dupe')
        os.makedirs(panel)
        manifest = {'cells': {
            'csa_dynamic::seed0': {'env_fp_v2': local, 'seed': 0},
        }}
        with open(os.path.join(panel, 'run_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh)
        # The same cell in summary.json is a legacy copy without v2: the
        # manifest's digest wins and the cell is counted once, not twice.
        summary = {'csa_dynamic::seed0': {'seed': 0, 'ppl': 10.0}}
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh)
        out = A.audit_environment_identity([panel])
    assert out['n_records'] == 1 and out['n_with_v2'] == 1, out
    assert out['n_match'] == 1 and out['n_missing_v2'] == 0


def test_environment_identity_audit_surfaces_conflicting_digests():
    local = L.environment_fingerprint_v2()
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'results_conflict')
        os.makedirs(panel)
        manifest = {'cells': {
            'csa_dynamic::seed0': {'env_fp_v2': local, 'seed': 0},
        }}
        with open(os.path.join(panel, 'run_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh)
        summary = {'csa_dynamic::seed0': {'env_fp_v2': 'other', 'seed': 0}}
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh)
        out = A.audit_environment_identity([panel])
    assert out['n_records'] == 1
    assert out['panels']['results_conflict']['conflicts'], out['panels']
    assert any('two different non-null env_fp_v2' in w for w in out['warnings'])


def test_environment_identity_audit_ignores_synthesized_records():
    rec = _record(env_v2=None)
    rec['synthesized'] = True
    with tempfile.TemporaryDirectory() as td:
        panel = _write_summary(td, [rec])
        out = A.audit_environment_identity([panel])
    assert out['n_records'] == 0 and out['n_missing_v2'] == 0, out
    assert out['status'] == 'not_verifiable_locally'


def test_environment_identity_is_wired_into_report_and_cli():
    assert hasattr(A, 'audit_environment_identity')
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '_audit_gates', '_parts', 'part_05_audit_confirmation_verdicts.py')
    body = open(src, encoding='utf-8').read()
    assert "'environment_identity': audit_environment_identity()" in body
    assert "'environment': audit_environment_identity" in body
    assert "'env_drift': audit_environment_identity" in body
