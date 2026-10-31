"""Regression checks for the prospective preparation-clearance gate.

These are zero-GPU checks.  They cover audit failure propagation, replay
artifact locality/identity, malformed-protocol tolerance, internally
inconsistent power/dependency evidence, and the sparse run-order balance
accounting needed when an arm grid grows.
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

import test_review_part_72 as T


def _rehash(clearance):
    clearance['content_sha256'] = L._prep_sha_obj(
        {k: v for k, v in clearance.items()
         if k not in ('created_at', 'revision', 'supersedes', 'reason',
                      'notes', 'content_sha256')})
    return clearance


def test_clearance_audit_fails_validation_without_replay_artifacts():
    # An extra check is outside the fixed vocabulary.  Even with a recomputed
    # body hash and no replay artifact present, the audit must not report a
    # warning-only pass for an invalid clearance.
    p = T._protocol()
    bad = copy.deepcopy(T._clearance(p))
    bad['checks']['bogus'] = {'name': 'bogus', 'required': False,
                              'status': 'ok'}
    _rehash(bad)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        with open(os.path.join(td, 'preparation_clearance.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(bad, fh)
        rep = L.preparation_clearance_audit(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'),
            events=[])
    assert rep['status'] == 'fail', rep
    assert any('unknown checks' in x for x in rep['problems']), rep['problems']


def test_preflight_summary_keeps_the_recorded_launch_decision_and_isolates_artifacts():
    # A clearance in a temporary directory must not import an unrelated
    # plan_snapshot from the repository root, and the summary must expose the
    # recorded launch decision rather than a missing replay key.
    p = T._protocol()
    c = T._clearance(p)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        L.write_preparation_clearance(td, c)
        summary = L.preparation_clearance_preflight_summary(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'))
        full = L.preparation_clearance_audit(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'), events=[])
    assert summary['launch_allowed'] is True, summary
    assert summary['n_problems'] == 0, summary
    assert summary['status'] == 'warnings', summary
    assert full['status'] != 'fail', full
    assert 'rehearsal' not in (full.get('replay', {}).get('replayed') or []), full


def test_clearance_build_tolerates_a_malformed_study_block():
    p = T._protocol()
    for study in (['not', 'an', 'object'], 'not-an-object', 3, None):
        q = dict(p)
        q['study'] = study
        c = L.build_preparation_clearance(
            q, run_order=T._order(T._manifest()),
            disposition=T._disposition(T._manifest()),
            power=T._power(T._manifest()), source_tree=T._tree(),
            rehearsal={'ok': True}, test_access_state=False,
            dependency_contract=T._dependency())
        assert c['status'] == 'fail', (study, c['status'])
        json.dumps(c, allow_nan=False)


def test_cyclic_rehearsal_is_a_failure_not_a_recursion_error():
    p = T._protocol()
    cyclic = {}
    cyclic['plan'] = cyclic
    c = T._clearance(p, rehearsal=cyclic)
    assert c['status'] == 'fail', c
    assert c['checks']['rehearsal']['status'] == 'fail', c['checks']['rehearsal']
    json.dumps(c, allow_nan=False)


def test_inconsistent_power_and_dependency_evidence_fails_closed():
    p = T._protocol()
    m = T._manifest()
    c = T._clearance(p, power=T._power(m, status='fail', all_pass=True))
    assert c['checks']['power']['status'] == 'fail', c['checks']['power']

    for dep in ({'status': 'ok'},
                {'status': 'ok', 'lock_sha256': 'a' * 64,
                 'mismatches': [{'name': 'torch'}]},
                {'status': 'ok', 'lock_sha256': 'not-a-digest'}):
        c = T._clearance(p, dependency_contract=dep)
        assert c['checks']['dependency']['status'] == 'fail', (dep, c['checks'])


def test_clearance_replay_detects_rehearsal_and_dependency_drift():
    p = T._protocol()
    c = T._clearance(p)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        L.write_preparation_clearance(td, c)
        path = os.path.join(td, 'preparation_clearance.json')
        proto = os.path.join(td, 'design_protocol.json')
        ok = L.preparation_clearance_audit(
            path=path, protocol_path=proto, events=[],
            rehearsal={'ok': True}, dependency_contract=T._dependency())
        assert ok['status'] != 'fail', ok

        rehearsal = L.preparation_clearance_audit(
            path=path, protocol_path=proto, events=[],
            rehearsal={'ok': True, 'note': 'different passing rehearsal'},
            dependency_contract=T._dependency())
        assert rehearsal['status'] == 'fail', rehearsal
        assert any('rehearsal_sha256' in x for x in rehearsal['problems']), (
            rehearsal['problems'])

        dependency = L.preparation_clearance_audit(
            path=path, protocol_path=proto, events=[],
            rehearsal={'ok': True},
            dependency_contract={'status': 'ok', 'lock_sha256': 'b' * 64})
        assert dependency['status'] == 'fail', dependency
        assert any('lock_sha256' in x for x in dependency['problems']), (
            dependency['problems'])


def test_run_order_balance_scales_to_a_large_arm_grid():
    arms = [f'arm{i}' for i in range(5000)]
    plan = {
        'kind': 'run_order', 'schema': 1,
        'design': {'name': 'randomized_complete_block_balanced',
                   'randomized': True, 'design_seed': 1, 'block': 'seed',
                   'position_balance': 'cyclic_base_rotation'},
        'variants': arms, 'seeds': ['0'], 'blocks': {'0': arms},
        'planned_cells': [],
    }
    rep = L.run_order_balance(plan)
    assert rep['status'] == 'ok', rep
    assert rep['position_balanced'] is True, rep
    assert rep['n_cells'] == len(arms), rep
