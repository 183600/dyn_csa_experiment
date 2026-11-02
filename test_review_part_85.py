"""Global (cross-panel) exploratory FDR: BH/BY replay and resolution floors.

These checks exercise the new ``explib`` block, its replayable artifact and
the audit gate.  A malformed claim family must fail closed (or be reported
unavailable), not crash a report or silently shrink the declared m.  CPU-only;
no split, checkpoint or training run is read.
"""
import copy
import json
import os
import tempfile

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _ref_q(pvals, method):
    """Independent reference for BH/BY q-values (ascending p order)."""
    m = len(pvals)
    factor = (sum(1.0 / i for i in range(1, m + 1))
              if method == 'by' else 1.0)
    order = sorted(range(m), key=lambda i: (pvals[i], i))
    q = [0.0] * m
    running = 1.0
    for pos in range(m - 1, -1, -1):
        i = order[pos]
        running = min(running, (m * pvals[i] / (pos + 1)) * factor)
        q[i] = min(1.0, running)
    return q


def test_bh_matches_an_independent_reference():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
         0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34]
    claims = {f'c{i}': v for i, v in enumerate(p)}
    got = L.global_fdr(claims, method='bh')
    assert got['status'] == 'ok', got
    assert got['m_declared'] == len(p)
    ref = _ref_q(p, 'bh')
    for i, name in enumerate(claims):
        row = got['claims'][name]
        assert abs(row['q_bh'] - ref[i]) < 1e-12, (name, row, ref[i])
        assert row['q'] == row['q_bh'], row
        assert row['reject'] is (row['q_bh'] <= 0.05), row
    # The BH step-up rejects the small p's but not the 0.205 tail.
    assert got['n_reject'] == len(got['reject_names'])
    assert got['n_reject'] >= 1
    assert got['p_threshold'] == max(
        got['claims'][n]['p'] for n in got['reject_names'])
    json.dumps(got, allow_nan=False)


def test_by_is_the_harmonic_multiple_of_bh_in_the_unclipped_range():
    p = [0.001, 0.002, 0.003, 0.004]
    bh = L.global_fdr({f'c{i}': v for i, v in enumerate(p)}, method='bh')
    by = L.global_fdr({f'c{i}': v for i, v in enumerate(p)}, method='by')
    h = sum(1.0 / i for i in range(1, len(p) + 1))
    for name in bh['claims']:
        assert abs(by['claims'][name]['q_by']
                   - min(1.0, bh['claims'][name]['q_bh'] * h)) < 1e-12
    assert by['dependence'] == 'arbitrary'
    assert bh['dependence'] == 'independent_or_positive'


def test_ties_are_monotone_and_share_a_q_value():
    got = L.global_fdr({'a': 0.01, 'b': 0.01, 'c': 0.02, 'd': 0.5},
                       method='bh')
    a, b, c, d = (got['claims'][k] for k in 'abcd')
    assert a['q_bh'] == b['q_bh'], (a, b)
    assert a['q_bh'] <= c['q_bh'] <= d['q_bh'], (a, b, c, d)
    assert sorted([a['rank'], b['rank']]) == [1, 2]


def test_invalid_p_occupies_a_declared_slot_but_is_never_rejected():
    got = L.global_fdr({'a': 0.01, 'b': float('nan'), 'c': 'bad',
                        'd': True, 'e': 1.2}, method='bh')
    assert got['status'] == 'ok', got
    assert got['m_declared'] == 5
    assert got['m_valid'] == 1
    assert got['m_invalid'] == 4
    # One valid p in a DECLARED family of five has q=0.05, not q=0.01.
    assert abs(got['claims']['a']['q_bh'] - 0.05) < 1e-12, got['claims']['a']
    for name in 'bcde':
        row = got['claims'][name]
        assert row['q'] is None and row['reject'] is False, row
        assert row['reason'], row
    assert got['invalid_claims'] == ['b', 'c', 'd', 'e']
    json.dumps(got, allow_nan=False)


def test_no_valid_p_is_not_applicable_not_a_silent_pass():
    got = L.global_fdr({'a': float('inf'), 'b': -0.1})
    assert got['status'] == 'not_applicable', got
    assert got['claims']['a']['q'] is None
    json.dumps(got, allow_nan=False)


def test_confirmatory_tier_is_refused_and_duplicate_names_are_hard_errors():
    for tier in ('confirmatory', 'confirmatory_primary'):
        try:
            L.global_fdr({'a': {'p': 0.1, 'tier': tier}})
        except ValueError:
            pass
        else:
            raise AssertionError(f'tier {tier!r} must not enter the global FDR')
    try:
        L.global_fdr([('a', 0.1), ('a', 0.2)])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate claim names must be refused')


def test_alpha_and_method_validation_is_fail_closed():
    for alpha in (0.0, 1.0, -0.1, 1.1, True, '0.05', float('nan')):
        try:
            L.global_fdr({'a': 0.1}, alpha=alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'alpha={alpha!r} must be refused')
    for method in ('', 'fdr-x', None, 1, True):
        try:
            L.global_fdr({'a': 0.1}, method=method)
        except ValueError:
            pass
        else:
            raise AssertionError(f'method={method!r} must be refused')
    # Accepted aliases stay explicit, not fuzzy matching.
    assert L.global_fdr({'a': 0.1}, method='benjamini-hochberg')['method'] == 'bh'
    assert L.global_fdr({'a': 0.1}, method='dependent')['method'] == 'by'


def test_discrete_floor_turns_no_rejection_into_a_resolution_statement():
    # n_nonzero=2 gives the exact sign-flip floor 2/2**2 = 0.5.  Even a
    # perfect panel at that floor cannot reach alpha=0.05 after correction.
    claims = [
        {'name': 'a', 'p': 0.5, 'n_nonzero': 2},
        {'name': 'b', 'p': 0.5, 'n_nonzero': 2},
    ]
    got = L.global_fdr(claims, method='bh')
    assert got['claims']['a']['p_floor'] == 0.5, got['claims']['a']
    assert got['claims']['a']['at_floor'] is True
    assert got['claims']['a']['q_floor_bh'] == 0.5
    assert got['claims']['a']['resolution_limited'] is True
    assert got['family_resolution_limited_bh'] is True
    assert got['min_attainable_q_bh'] == 0.5
    assert got['resolution_complete'] is True
    # With 12 non-zero seeds the floor is 2/4096 and the family is resolvable.
    fine = L.global_fdr([
        {'name': 'a', 'p': 0.00048828125, 'n_nonzero': 12},
        {'name': 'b', 'p': 0.00048828125, 'n_nonzero': 12},
    ], method='bh')
    assert fine['claims']['a']['resolution_limited'] is False
    assert fine['family_resolution_limited_bh'] is False


def test_floor_above_p_is_clamped_with_an_explicit_warning():
    got = L.global_fdr({'a': {'p': 0.2, 'p_floor': 0.3, 'n_nonzero': 4}})
    row = got['claims']['a']
    assert row['p_floor'] == 0.2, row
    assert any('exceeds p' in w for w in got['warnings']), got['warnings']
    json.dumps(got, allow_nan=False)


def test_q_floor_is_withheld_when_a_trailing_floor_is_missing():
    # The floor for ``a`` cannot be bounded if a later claim has no floor:
    # replacing the missing floor with 1.0 would overstate the bound.
    got = L.global_fdr([
        {'name': 'a', 'p': 0.01, 'n_nonzero': 2},
        {'name': 'b', 'p': 0.02},
    ], method='bh')
    assert got['claims']['a']['q_floor_bh'] is None, got['claims']['a']
    assert got['claims']['b']['q_floor_bh'] is None, got['claims']['b']
    assert got['resolution_complete'] is False


def test_artifact_round_trip_and_created_at_does_not_change_identity():
    claims = [
        {'name': 'a', 'p': 0.004, 'panel': 'capacity', 'n': 12,
         'n_nonzero': 12},
        {'name': 'b', 'p': 0.031, 'panel': 'length', 'n': 12,
         'n_nonzero': 12},
    ]
    one = L.build_global_fdr(claims, family='exploratory', created_at='t1')
    two = L.build_global_fdr(claims, family='exploratory', created_at='t2')
    assert one['content_sha256'] == two['content_sha256']
    # A one-shot claim iterator must not be consumed twice by the builder.
    gen = (c for c in claims)
    three = L.build_global_fdr(gen, family='exploratory', created_at='t3')
    assert three['content_sha256'] == one['content_sha256']
    assert L.validate_global_fdr(three)['status'] == 'ok'
    assert L.validate_global_fdr(one)['status'] == 'ok'
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'global_fdr.json')
        L.write_global_fdr(path, one)
        with open(path, encoding='utf-8') as fh:
            loaded = json.load(fh)
        assert L.validate_global_fdr(loaded)['status'] == 'ok'
    assert L.validate_global_fdr(one)['result']['claims']['a']['q'] is not None
    json.dumps(one, allow_nan=False)


def test_artifact_tampering_and_schema_drift_fail():
    artifact = L.build_global_fdr(
        [{'name': 'a', 'p': 0.01, 'n_nonzero': 12},
         {'name': 'b', 'p': 0.2, 'n_nonzero': 12}], method='by')
    cases = []
    bad = copy.deepcopy(artifact)
    bad['result']['claims']['a']['q'] = 0.0001
    cases.append(('q', bad))
    bad = copy.deepcopy(artifact)
    bad['result']['claims']['a']['rank'] = 99
    cases.append(('rank', bad))
    bad = copy.deepcopy(artifact)
    bad['result']['n_reject'] = 99
    cases.append(('n_reject', bad))
    bad = copy.deepcopy(artifact)
    bad['result']['extra'] = True
    cases.append(('extra-result', bad))
    bad = copy.deepcopy(artifact)
    bad['extra-top'] = True
    cases.append(('extra-top', bad))
    bad = copy.deepcopy(artifact)
    bad['result']['claims']['a']['extra'] = True
    cases.append(('extra-claim', bad))
    bad = copy.deepcopy(artifact)
    bad['inputs']['claims'][0]['p'] = 0.02
    cases.append(('input-p', bad))
    bad = copy.deepcopy(artifact)
    bad['kind'] = 'something_else'
    cases.append(('kind', bad))
    bad = copy.deepcopy(artifact)
    bad['schema'] = 2
    cases.append(('schema', bad))
    bad = copy.deepcopy(artifact)
    del bad['content_sha256']
    cases.append(('missing-hash', bad))
    bad = copy.deepcopy(artifact)
    bad['content_sha256'] = '0' * 64
    cases.append(('wrong-hash', bad))
    for label, bad in cases:
        verdict = L.validate_global_fdr(bad)
        assert verdict['status'] == 'fail', (label, verdict)
        json.dumps(verdict, allow_nan=False)


def test_artifact_rejects_frozen_decision_fields_everywhere():
    artifact = L.build_global_fdr([{'name': 'a', 'p': 0.01}])
    for where in ('top', 'result', 'claim'):
        bad = copy.deepcopy(artifact)
        if where == 'top':
            bad['confirmed'] = True
        elif where == 'result':
            bad['result']['p_holm'] = 0.01
        else:
            bad['result']['claims']['a']['verdict'] = 'confirmed'
        verdict = L.validate_global_fdr(bad)
        assert verdict['status'] == 'fail', (where, verdict)
        assert verdict['problems'], verdict


def test_gate_missing_ok_not_applicable_and_tampered():
    missing = os.path.join(REPO, '_does_not_exist_global_fdr.json')
    out = A.audit_global_fdr(missing, verbose=False)
    assert out['status'] == 'not_verifiable_locally', out

    good = L.build_global_fdr([{'name': 'a', 'p': 0.01, 'n_nonzero': 12}])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'global_fdr.json')
        L.write_global_fdr(path, good)
        out = A.audit_global_fdr(path, verbose=False)
        assert out['status'] == 'ok', out
        assert out['m_declared'] == 1 and out['n_reject'] == out['n_reject']

        bad = copy.deepcopy(good)
        bad['result']['claims']['a']['q'] = 0.5
        bad_path = os.path.join(td, 'bad.json')
        L.write_global_fdr(bad_path, bad)
        out = A.audit_global_fdr(bad_path, verbose=False)
        assert out['status'] == 'fail', out

        # A self-consistent artifact whose whole family is unusable is a
        # not-applicable readout, not a failure and not a green pass.
        unusable = L.build_global_fdr([{'name': 'a', 'p': 0.01}])
        unusable['inputs']['claims'][0]['p'] = None
        unusable['result'] = L.global_fdr([{'name': 'a', 'p': None}])
        unusable['content_sha256'] = L._sha256_json(
            {k: v for k, v in unusable.items()
             if k not in ('created_at', 'content_sha256')})
        unusable_path = os.path.join(td, 'unusable.json')
        L.write_global_fdr(unusable_path, unusable)
        out = A.audit_global_fdr(unusable_path, verbose=False)
        assert out['status'] == 'not_applicable', out


def test_nominal_budget_and_absurd_counts_stay_strict_json():
    got = L.global_fdr({'a': 0.001, 'b': 0.004}, method='bh')
    assert got['false_discovery_budget'] == 0.05 * got['n_reject'], got
    extreme = L.global_fdr({
        'a': {'p': 0.1, 'n_nonzero': 10 ** 400},
        'b': {'p': 0.2, 'n': 10 ** 400},
    })
    assert extreme['status'] == 'ok', extreme
    assert extreme['claims']['a']['p_floor'] is None
    assert extreme['claims']['b']['n'] is None
    assert any('n_nonzero' in w for w in extreme['warnings'])
    json.dumps(extreme, allow_nan=False)
    # A 10**10000 alpha cannot be stringified by repr on Python 3.12; the
    # validation error must still be a clean ValueError, not a formatting bug.
    try:
        L.global_fdr({'a': 0.1}, alpha=10 ** 10000)
    except ValueError:
        pass
    else:
        raise AssertionError('an absurd alpha must be refused')


def test_parts_cli_makefile_and_report_wiring():
    import explib
    import _audit_gates
    assert 'part_78_global_fdr.py' in explib._PARTS
    assert 'part_44_audit_global_fdr.py' in _audit_gates._PARTS
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        makefile = fh.read()
    assert 'global-fdr:' in makefile
    assert 'audit_gates.py global_fdr' in makefile
    assert callable(A.audit_global_fdr)
    assert callable(A.audit_global_fdr_artifact)
    with open(os.path.join(
            REPO, '_audit_gates', '_parts',
            'part_05_audit_confirmation_verdicts.py'), encoding='utf-8') as fh:
        source = fh.read()
    assert "'global_fdr': audit_global_fdr" in source
    assert "res['global_fdr'] = audit_global_fdr" in source
    # The prohibited trees stay present and untouched.
    assert os.path.isdir(os.path.join(REPO, '.github'))
    assert os.path.exists(os.path.join(REPO, 'README.md'))
    assert os.path.exists(os.path.join(REPO, 'README_zh.md'))
