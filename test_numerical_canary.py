"""CPU checks for the numerical-stack canary (reproducibility hardening).

The canary hashes fixed torch operator outputs so two processes that declare
the same lock/`env_fp_v3` can still discover that their *observed* arithmetic
differs.  These tests exercise determinism, parameter binding, artifact
integrity, drift classification, the audit gate and CLI/Makefile wiring.
No training, GPU or split access is required.
"""
import contextlib
import copy
import inspect
import io
import json
import os
import tempfile

from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _retamper(artifact, op='matmul'):
    """Change one op digest and recompute every dependent digest.

    This is the strongest consistent edit an attacker can make without
    changing the measured bytes: the artifact hash and the arithmetic
    fingerprint both recompute, so only a real replay can expose the drift.
    """
    art = copy.deepcopy(artifact)
    arithmetic = art['result']['arithmetic']
    params = art['result']['params']
    arithmetic['ops'][op]['sha256'] = '0' * 64
    arithmetic['fingerprint'] = L._nc_sha_json({
        'schema': 1,
        'size': params['size'],
        'dtype': params['dtype'],
        'device_class': str(params['device']).split(':')[0],
        'input_fingerprint': arithmetic['input_fingerprint'],
        'ops': arithmetic['ops'],
    })
    art['content_sha256'] = L._nc_content_sha(art)
    return art


def test_canary_is_deterministic_and_json_safe():
    a = L.numerical_canary(seed=3, size=6)
    b = L.numerical_canary(seed=3, size=6)
    assert a['kind'] == 'numerical_canary' and a['status'] == 'ok', a
    assert a['params'] == {'seed': 3, 'size': 6, 'dtype': 'float32',
                           'device': 'cpu'}, a['params']
    assert a['schema'] == L.NC_SCHEMA
    assert a['arithmetic'] == b['arithmetic'], 'same-seed canary must be bitwise stable'
    assert a['environment'] == b['environment']
    assert set(a['arithmetic']['ops']) == set(L.NC_OP_NAMES)
    assert len(a['arithmetic']['fingerprint']) == 64
    assert len(a['arithmetic']['input_fingerprint']) == 64
    for name, row in a['arithmetic']['ops'].items():
        assert set(row) == set(L.NC_OP_KEYS), (name, row)
        assert len(row['sha256']) == 64, (name, row)
        assert isinstance(row['shape'], list) and row['finite'] is True, (name, row)
        assert row['dtype'] in ('float32', 'int64'), (name, row)
    # The record must survive a strict JSON writer: a diagnostics record that
    # needs NaN/Infinity tokens cannot be archived or replayed portably.
    json.dumps(a, allow_nan=False)
    assert L._nc_validate_result(a) == [], L._nc_validate_result(a)


def test_canary_binds_seed_size_and_dtype():
    base = L.numerical_canary(seed=0, size=5)
    other_seed = L.numerical_canary(seed=1, size=5)
    # The seed only feeds the explicit RNG op; inputs stay platform-independent.
    assert (base['arithmetic']['ops']['rng']['sha256']
            != other_seed['arithmetic']['ops']['rng']['sha256'])
    assert (base['arithmetic']['ops']['matmul']['sha256']
            == other_seed['arithmetic']['ops']['matmul']['sha256'])
    assert (base['arithmetic']['fingerprint']
            != other_seed['arithmetic']['fingerprint'])
    bigger = L.numerical_canary(seed=0, size=6)
    assert bigger['arithmetic']['fingerprint'] != base['arithmetic']['fingerprint']
    as_double = L.numerical_canary(seed=0, size=5, dtype='float64')
    assert as_double['params']['dtype'] == 'float64'
    assert as_double['arithmetic']['ops']['matmul']['dtype'] == 'float64'
    assert as_double['arithmetic']['fingerprint'] != base['arithmetic']['fingerprint']


def test_canary_rejects_invalid_parameters():
    for bad in (True, -1, L.MAX_SEED + 1, 1.5, 'x'):
        try:
            L.numerical_canary(seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'canary accepted seed={bad!r}')
    for bad in (1, L.NC_MAX_SIZE + 1, True, 2.5):
        try:
            L.numerical_canary(size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'canary accepted size={bad!r}')
    for bad in ('float16', 'int32', None, True):
        try:
            L.numerical_canary(dtype=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'canary accepted dtype={bad!r}')
    for bad in ('meta', 'mps', True):
        try:
            L.numerical_canary(device=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'canary accepted device={bad!r}')
    if not torch.cuda.is_available():
        try:
            L.numerical_canary(device='cuda')
        except ValueError as exc:
            assert 'CUDA is not available' in str(exc), exc
        else:
            raise AssertionError('canary accepted cuda on a CPU-only box')


def test_comparison_classifies_match_mismatch_and_not_comparable():
    a = L.numerical_canary(seed=2, size=5)
    b = L.numerical_canary(seed=2, size=5)
    same = L.compare_numerical_canaries(a, b)
    assert same['status'] == 'match' and same['arithmetic_match'] is True, same
    assert same['differing_ops'] == [] and same['environment_changes'] == []

    different = L.numerical_canary(seed=3, size=5)
    mismatch = L.compare_numerical_canaries(a, different)
    assert mismatch['status'] == 'mismatch' and mismatch['arithmetic_match'] is False
    assert 'rng' in mismatch['differing_ops'], mismatch

    # Environment-only changes are reported, but do not turn a matching probe
    # into a mismatch.
    env_only = copy.deepcopy(a)
    env_only['environment']['threads'] = {'torch': 999, 'interop': 0}
    changed = L.compare_numerical_canaries(a, env_only)
    assert changed['status'] == 'match' and changed['environment_changes'] == ['threads']

    # A different device class is not comparable, not a spurious drift.
    cuda_like = copy.deepcopy(a)
    cuda_like['params']['device'] = 'cuda:0'
    cuda_like['arithmetic']['fingerprint'] = L._nc_sha_json({
        'schema': 1, 'size': 5, 'dtype': 'float32', 'device_class': 'cuda',
        'input_fingerprint': cuda_like['arithmetic']['input_fingerprint'],
        'ops': cuda_like['arithmetic']['ops'],
    })
    not_comparable = L.compare_numerical_canaries(a, cuda_like)
    assert not_comparable['status'] == 'not_comparable', not_comparable
    assert not_comparable['not_comparable']['field'] == 'device_class'

    assert L.compare_numerical_canaries({}, a)['status'] == 'unverified'
    art = L.build_numerical_canary_artifact(result=a)
    bad_artifact = copy.deepcopy(art)
    bad_artifact['content_sha256'] = '0' * 64
    assert L.compare_numerical_canaries(bad_artifact, art)['status'] == 'unverified'


def test_artifact_round_trip_and_tamper_rejection():
    art = L.build_numerical_canary_artifact(seed=2, size=5,
                                            created_at='2026-01-01T00:00:00')
    assert art['kind'] == L.NC_ARTIFACT_KIND and art['schema'] == L.NC_SCHEMA
    assert set(art) == set(L.NC_ARTIFACT_KEYS)
    assert art['inputs'] == art['result']['params']
    verdict = L.validate_numerical_canary_artifact(art)
    assert verdict['status'] == 'ok', verdict
    assert verdict['problems'] == [] and verdict['drifted_ops'] == []

    # created_at is volatile: a rebuild of the same body is the same artifact.
    moved = copy.deepcopy(art)
    moved['created_at'] = '2027-02-02T00:00:00'
    assert L._nc_content_sha(moved) == art['content_sha256']

    # A digest edit without recomputation is a tamper failure.
    edited = copy.deepcopy(art)
    edited['result']['arithmetic']['ops']['matmul']['sha256'] = '0' * 64
    assert L.validate_numerical_canary_artifact(edited)['status'] == 'fail'

    # A consistent-looking arithmetic edit that keeps the old fingerprint is
    # still caught: the fingerprint must recompute from the recorded ops.
    consistent = copy.deepcopy(art)
    consistent['result']['arithmetic']['ops']['matmul']['sha256'] = '0' * 64
    consistent['content_sha256'] = L._nc_content_sha(consistent)
    verdict = L.validate_numerical_canary_artifact(consistent)
    assert verdict['status'] == 'fail'
    assert any('fingerprint does not recompute' in p for p in verdict['problems']), verdict

    # The strongest edit (all digests recomputed) can only be caught by replay.
    drift = L.validate_numerical_canary_artifact(_retamper(art))
    assert drift['status'] == 'drift', drift
    assert 'matmul' in drift['drifted_ops'], drift

    extra = copy.deepcopy(art)
    extra['smuggled'] = True
    extra['content_sha256'] = L._nc_content_sha(extra)
    assert L.validate_numerical_canary_artifact(extra)['status'] == 'fail'

    extra_ops = copy.deepcopy(art)
    extra_ops['result']['arithmetic']['ops']['extra'] = {
        'sha256': '0' * 64, 'shape': [], 'dtype': 'float32', 'finite': True}
    extra_ops['content_sha256'] = L._nc_content_sha(extra_ops)
    assert L.validate_numerical_canary_artifact(extra_ops)['status'] == 'fail'

    extra_prov = copy.deepcopy(art)
    extra_prov['provenance']['smuggled'] = True
    extra_prov['content_sha256'] = L._nc_content_sha(extra_prov)
    assert L.validate_numerical_canary_artifact(extra_prov)['status'] == 'fail'

    missing = copy.deepcopy(art)
    missing.pop('provenance')
    missing['content_sha256'] = L._nc_content_sha(missing)
    assert L.validate_numerical_canary_artifact(missing)['status'] == 'fail'


def test_writer_noop_conflict_and_overwrite():
    art = L.build_numerical_canary_artifact(seed=2, size=5)
    with tempfile.TemporaryDirectory() as td:
        first = L.write_numerical_canary_artifact(td, art)
        assert first['status'] == 'ok' and first['written'] is True, first
        second = L.write_numerical_canary_artifact(td, art)
        assert second['status'] == 'ok' and second['unchanged'] is True, second
        other = L.build_numerical_canary_artifact(seed=3, size=5)
        conflict = L.write_numerical_canary_artifact(td, other)
        assert conflict['conflict'] is True and conflict['written'] is False
        replaced = L.write_numerical_canary_artifact(td, other, overwrite=True)
        assert replaced['written'] is True and replaced['status'] == 'ok', replaced
        bad = L.write_numerical_canary_artifact(td, {'kind': 'x'})
        assert bad['status'] == 'fail' and bad['written'] is False
        assert os.path.isfile(os.path.join(td, 'numerical_canary.json'))


def test_audit_gate_missing_ok_drift_and_tamper():
    art = L.build_numerical_canary_artifact(seed=2, size=5)
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, 'absent.json')
        assert A.audit_numerical_canary(missing, verbose=False)['status'] \
            == 'not_verifiable_locally'
        path = os.path.join(td, 'numerical_canary.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(art, fh)
        ok = A.audit_numerical_canary(path, verbose=False)
        assert ok['status'] == 'ok', ok
        assert ok['arithmetic_match'] is True and ok['n_ops'] == len(L.NC_OP_NAMES)
        assert ok['source_tree_changed'] is False

        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(_retamper(art), fh)
        drift = A.audit_numerical_canary(path, verbose=False)
        assert drift['status'] == 'fail', drift
        assert drift['drifted_ops'] == ['matmul'], drift

        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        assert A.audit_numerical_canary(path, verbose=False)['status'] == 'fail'

        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(art['result'], fh)
        assert A.audit_numerical_canary(path, verbose=False)['status'] == 'fail'

        # The same gate is reachable through the artifact alias.
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(art, fh)
        assert A.audit_numerical_canary_artifact(path, verbose=False)['status'] == 'ok'


def test_cli_write_and_verify():
    with tempfile.TemporaryDirectory() as td:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = L._nc_cli(['--write', '--out', td, '--seed', '3', '--size', '6'])
        assert rc == 0, out.getvalue()[-2000:]
        path = os.path.join(td, 'numerical_canary.json')
        assert os.path.isfile(path)
        with contextlib.redirect_stdout(io.StringIO()):
            assert L._nc_cli(['--verify', path]) == 0
            assert L._nc_cli(['--write', '--out', td, '--seed', '4']) == 1
            assert L._nc_cli(['--write', '--out', td, '--seed', '4',
                              '--overwrite']) == 0
            assert L._nc_cli(['--verify', os.path.join(td, 'absent.json')]) == 2


def test_wiring_and_no_forbidden_mutation():
    assert 'numerical_canary' in inspect.getsource(A.run_report)
    assert 'numerical_canary' in inspect.getsource(A._run_cli)
    for name in ('numerical_canary', 'numerical_canary_provenance',
                 'build_numerical_canary_artifact',
                 'validate_numerical_canary_artifact',
                 'compare_numerical_canaries',
                 'write_numerical_canary_artifact'):
        assert callable(getattr(L, name)), name
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as handle:
        make = handle.read()
    assert 'canary:' in make and 'canary-check:' in make
    assert 'audit_gates.py numerical_canary' in make
    with open(os.path.join(REPO, 'explib', '__init__.py'), encoding='utf-8') as handle:
        assert 'part_82_numerical_canary.py' in handle.read()
    with open(os.path.join(REPO, '_audit_gates', '__init__.py'), encoding='utf-8') as handle:
        assert 'part_48_audit_numerical_canary.py' in handle.read()
    for name in ('README.md', 'README_zh.md', '.github'):
        assert os.path.exists(os.path.join(REPO, name)), name


def test_canary_malformed_fields_fail_closed_and_stay_strict_json():
    """A hand-edited canary is untrusted JSON.

    Python integers can exceed CPython's decimal-conversion limit and dtype
    fields can be unhashable.  Validation must report these as ordinary
    problems instead of raising while formatting the diagnostic.
    """
    huge = 10 ** 10000
    art = L.build_numerical_canary_artifact(seed=2, size=5,
                                            created_at='2026-01-01T00:00:00')
    cases = []
    for path, value in (
            (('kind',), huge),
            (('schema',), huge),
            (('result', 'kind'), huge),
            (('result', 'status'), huge),
            (('result', 'schema'), True),
            (('result', 'params', 'seed'), huge),
            (('result', 'params', 'size'), huge),
            (('result', 'params', 'dtype'), [huge]),
            (('result', 'params', 'device'), 'cpufoo'),
            (('result', 'arithmetic', 'schema'), True),
            (('result', 'arithmetic', 'ops'), {huge: {'sha256': 'x'}}),
            (('result', 'environment',), {huge: huge})):
        bad = copy.deepcopy(art)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        cases.append((path, bad))
    for path, bad in cases:
        verdict = L.validate_numerical_canary_artifact(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        json.dumps(verdict, allow_nan=False)

    # A comparison over a malformed (but not hashed) result must also stay
    # serialisable when its environment keys are unprintable integers.
    base = L.numerical_canary(seed=2, size=5)
    broken = copy.deepcopy(base)
    broken['environment'] = {huge: huge}
    compared = L.compare_numerical_canaries(broken, base)
    json.dumps(compared, allow_nan=False)


def test_canary_builder_and_cli_reject_malformed_inputs_cleanly():
    base = L.numerical_canary(seed=2, size=5)
    bad = copy.deepcopy(base)
    bad['params']['size'] = 10 ** 10000
    try:
        L.build_numerical_canary_artifact(result=bad)
    except ValueError as exc:
        assert 'malformed result' in str(exc), exc
    else:
        raise AssertionError('builder accepted a malformed result')
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            assert L._nc_cli(['--write', '--out', '']) == 2
