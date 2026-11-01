"""Fail-closed and scale guards for the newest Type S/M and exact-TOST code.

These checks exercise malformed manifests, absurd family sizes and tampered
nested companions.  A design/audit helper must report a failure (and remain
strict JSON) instead of raising or silently accepting a field it did not
replay.  CPU-only; no split or checkpoint is read.
"""
import copy
import json
import os
import tempfile

from test_support import *  # noqa: F401,F403


def _manifest():
    return {
        'seeds': list(range(12)),
        'primary_comparisons': [{'name': 'c0', 'a': 'a', 'b': 'b'}],
        'statistics': {'alpha': 0.05},
    }


def _pilot():
    return {
        'status': 'pass',
        'alpha': 0.05,
        'm_family': 1,
        'manifest_sha256': 'a' * 64,
        'comparisons': {'c0': {'deltas': [0.13, 0.11, 0.12, 0.10]}},
    }


def test_type_sm_malformed_pilot_fails_closed_and_stays_json():
    for bad in (True, 1, 1.0, 'pilot', [], {}, {'comparisons': []}):
        out = L.type_sm_from_manifest(_manifest(), bad)
        assert out['status'] == 'fail', (bad, out)
        if not isinstance(bad, dict):
            assert out['pilot_status'] == 'invalid', (bad, out)
            assert out['pilot'] is None, (bad, out)
        json.dumps(out, allow_nan=False)


def test_type_sm_manifest_and_pilot_metadata_cannot_break_strict_json():
    m = _manifest()
    m['manifest_sha256'] = float('nan')
    out = L.type_sm_from_manifest(m, _pilot())
    assert out['status'] == 'fail', out
    json.dumps(out, allow_nan=False)

    p = _pilot()
    p['status'] = float('inf')
    p['alpha'] = float('nan')
    p['m_family'] = b'1'
    p['manifest_sha256'] = 10 ** 400
    out = L.type_sm_from_manifest(_manifest(), p)
    assert out['status'] == 'fail', out
    assert out['pilot']['status'] is None
    assert out['pilot']['alpha'] is None
    assert out['pilot']['m_family'] is None
    assert out['pilot']['manifest_sha256'] is None
    json.dumps(out, allow_nan=False)


def test_type_sm_absurd_family_size_is_invalid_not_an_overflow():
    # float(10**400) raises OverflowError; a 10**10000-digit integer also
    # breaks Python's integer->string conversion.  Neither may escape as an
    # exception or as a non-JSON field in the result.
    for family_size in (10 ** 400, 10 ** 10000):
        out = L.type_sm_design(12, 1.0, 0.1, family_size=family_size)
        assert out['status'] == 'invalid', out
        assert out['family_size'] is None, out
        assert out['problems'], out
        json.dumps(out, allow_nan=False)
    out = L.type_sm_design_from_deltas(
        12, [0.1, 0.2], family_size=10 ** 10000)
    assert out['status'] == 'invalid', out
    json.dumps(out, allow_nan=False)


def test_exact_tost_floor_uses_the_shifted_vectors_not_a_theoretical_n():
    # One delta sits exactly on -margin, so the low shifted vector has an
    # exact zero and only n-1 non-zero entries.  Its one-sided floor is
    # 2**-(n-1), not 2**-n; at conf=0.95 alpha=0.025 equivalence is then
    # impossible even though the theoretical all-nonzero floor is below alpha.
    margin = 0.1
    deltas = [-margin] + [0.05] * 5
    out = L.exact_signflip_tost(deltas, margin, conf=0.95)
    assert out['status'] == 'ok', out
    assert out['p_floor_low'] == 1.0 / 32.0, out
    assert out['p_floor'] == out['p_floor_low'], out
    assert out['p_floor_by_n'] == out['p_floor'], out
    assert out['resolution_limited'] is True, out
    assert out['equivalent'] is False, out


def test_artifact_validators_do_not_echo_a_malformed_digest():
    for artifact, validate in (
            (L.build_exact_tost([0.001, -0.002], 0.01), L.validate_exact_tost),
            (L.build_type_sm_design(12, 1.0, 0.1), L.validate_type_sm_design)):
        bad = copy.deepcopy(artifact)
        bad['content_sha256'] = float('nan')
        verdict = validate(bad)
        assert verdict['status'] == 'fail', verdict
        assert verdict['content_sha256'] is None, verdict
        json.dumps(verdict, allow_nan=False)


def test_nested_exact_tost_gate_replays_every_companion_field():
    d = [0.001] * 6
    row = L.exact_signflip_tost(d, 0.1)
    assert row['equivalent'] is True
    row['t_equivalent'] = True
    row['agreement'] = True
    stats = {
        'comparisons': {
            'c0': {
                'equivalent': True,
                'practical_significance': {
                    'status': 'equivalent_within_sesoi',
                    'margin': 0.1,
                    'tost_conf': 0.90,
                    'per_seed_log_ratio': d,
                    'distribution_free': row,
                },
            },
        },
    }
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, 'good.json')
        with open(good, 'w', encoding='utf-8') as fh:
            json.dump(stats, fh)
        assert A.audit_exact_tost(good, verbose=False)['status'] == 'ok'
        # Each field is individually replayable; none may hide beside a valid
        # p_tost.  ``bogus`` covers the "extra field" branch, ``confirmed`` the
        # explicit frozen-decision branch.
        for key, value in (('p_floor_low', 0.5), ('mean', 99.0),
                           ('n_dropped', 9), ('bogus', 1),
                           ('confirmed', True)):
            bad = copy.deepcopy(stats)
            bad['comparisons']['c0']['practical_significance'][
                'distribution_free'][key] = value
            path = os.path.join(td, f'bad_{key}.json')
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(bad, fh)
            out = A.audit_exact_tost(path, verbose=False)
            assert out['status'] == 'fail', (key, out)
            json.dumps(out, allow_nan=False)
