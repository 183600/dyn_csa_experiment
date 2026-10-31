"""CPU checks (split group 72): design registration is not launch readiness.

A ``design_protocol`` records what the study intends to do.  The separate
preparation artifacts (run order, disposition, pre-test power, dependency
lock, rehearsal and held-out state) each answer one question, and nothing
required them to describe the same experiment at the moment the first cell
started.  These checks exercise the ``preparation_clearance`` object: its
required checks, its strict/confirmatory versus draft behaviour, its hash and
timing rules, and its audit/replay.  No training, GPU or test split is read.
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

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO, 'confirmation_frozen.json')
_TREE = None


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _tree():
    global _TREE
    if _TREE is None:
        _TREE = L.source_tree_manifest()
    return _TREE


def _cells(manifest):
    return ['%s::seed%s' % (a, s) for a in manifest['arms']
            for s in manifest['seeds']]


def _order(manifest):
    pairs = [(a, s) for a in manifest['arms'] for s in manifest['seeds']]
    return L.build_run_order_plan(
        pairs, block_key=lambda c: c[1], seed=11,
        created_at='2026-10-31T00:00:00')


def _disposition(manifest):
    return L.build_run_disposition_plan(
        _cells(manifest), created_at='2026-10-31T00:00:00')


def _protocol(manifest=None, *, level=None, source_tree='current',
              created_at='2026-10-31T00:00:01'):
    manifest = copy.deepcopy(manifest or _manifest())
    tree = _tree() if source_tree == 'current' else source_tree
    return L.build_design_protocol(
        manifest, level=level, study_name='fixed-chunk confirmation',
        run_order=_order(manifest), disposition=_disposition(manifest),
        test_access={'method': 'single_look', 'max_looks': 1, 'declared': True},
        disjoint_declared=True, created_at=created_at, source_tree=tree)


def _power(manifest, *, status='pass', all_pass=True, q=0.95, min_power=0.8,
           value=0.9, n=12, power_ok=True):
    comps = {}
    for c in manifest['primary_comparisons']:
        comps[c['name']] = {
            'power_ok': power_ok, 'n': n,
            'power': {'power': {'q=0.95': value, 'q=0.90': value - 0.1}},
        }
    return {
        'kind': 'power_validation', 'status': status, 'all_pass': all_pass,
        'alpha': 0.05, 'm_family': len(manifest['primary_comparisons']),
        'power_q_for_test': q, 'min_power_for_test': min_power,
        'comparisons': comps,
    }


def _dependency():
    return {'status': 'ok', 'lock_sha256': 'a' * 64}


def _clearance(protocol, manifest=None, **overrides):
    manifest = manifest or _manifest()
    kwargs = dict(
        run_order=_order(manifest), disposition=_disposition(manifest),
        power=_power(manifest), source_tree=_tree(),
        rehearsal={'ok': True}, test_access_state=False,
        dependency_contract=_dependency(),
        created_at='2026-10-31T00:01:00')
    kwargs.update(overrides)
    return L.build_preparation_clearance(protocol, **kwargs)


def test_complete_clearance_binds_every_required_check():
    p = _protocol()
    assert p['sample_size']['power_gate_required'] is True
    c = _clearance(p)
    assert c['status'] == 'ok', (c['problems'], c['warnings'])
    assert c['launch_allowed'] is True
    assert set(c['required_checks']) == set(
        L.PREPARATION_CLEARANCE_POLICY['required_confirmatory'])
    assert c['study']['design_protocol_sha256'] == p['content_sha256']
    assert c['checks']['power']['status'] == 'ok'
    assert L.validate_preparation_clearance(c, protocol=p)['status'] == 'ok'

    # The identity excludes only timestamps/revision metadata, like the
    # protocol it binds.
    moved = copy.deepcopy(c)
    moved['created_at'] = '2027-01-01T00:00:00'
    assert moved['content_sha256'] == c['content_sha256']
    edited = copy.deepcopy(c)
    edited['checks']['power']['status'] = 'fail'
    assert L.validate_preparation_clearance(edited, protocol=p)['status'] == 'fail'

    blob = json.dumps(c, sort_keys=True)
    for forbidden in ('p_holm', 'p_exact_signflip', 'p_value', 'confirmed',
                      'verified', 'verdict'):
        assert forbidden not in blob, forbidden


def test_missing_confirmatory_preparation_fails_closed():
    p = _protocol()
    for key, needle in (
            ('run_order', 'run_order'),
            ('disposition', 'disposition'),
            ('power', 'power'),
            ('rehearsal', 'rehearsal'),
            ('test_access_state', 'held-out'),
            ('dependency_contract', 'dependency')):
        c = _clearance(p, **{key: None})
        assert c['status'] == 'fail', (key, c['status'])
        assert c['launch_allowed'] is False
        assert any(needle in x for x in c['problems']), (key, c['problems'][:5])
    # A draft gets named warnings rather than hard failures, but is still not
    # allowed to launch.
    m = _manifest()
    m['statistics'].pop('require_power_gate', None)
    draft = _protocol(m, level='draft')
    c = _clearance(
        draft, m, run_order=None, disposition=None, power=None,
        rehearsal=None, test_access_state=None, dependency_contract=None)
    assert c['status'] == 'warnings', c['problems']
    assert c['launch_allowed'] is False


def test_run_order_must_be_the_registered_complete_randomization():
    p = _protocol()
    order = _order(_manifest())

    missing = copy.deepcopy(order)
    missing['planned_cells'] = missing['planned_cells'][:-1]
    c = _clearance(p, run_order=missing)
    assert c['status'] == 'fail', c['problems']
    assert any('planned_cells' in x for x in c['problems']), c['problems']

    reordered = copy.deepcopy(order)
    reordered['planned_cells'] = list(reversed(reordered['planned_cells']))
    c = _clearance(p, run_order=reordered)
    assert c['status'] == 'fail', c['problems']
    assert any('sequence' in x for x in c['problems']), c['problems']

    wrong_seed = copy.deepcopy(order)
    wrong_seed['design']['design_seed'] = 99
    c = _clearance(p, run_order=wrong_seed)
    assert c['status'] == 'fail', c['problems']
    assert any('design_seed' in x for x in c['problems']), c['problems']

    # A one-arm-per-block plan is arm-major even if every planned cell is
    # present: two arms, one in each block, so arm and calendar time are
    # perfectly confounded.
    m = _manifest()
    one_arm = L.build_run_order_plan(
        [(m['arms'][0], 0), (m['arms'][1], 1)],
        block_key=lambda c: c[1], seed=1,
        created_at='2026-10-31T00:00:00')
    c = _clearance(p, run_order=one_arm)
    assert c['status'] == 'fail', c['problems']
    assert any('arm-major' in x.lower() or 'complete' in x
               for x in c['problems']), c['problems']


def test_disposition_must_match_the_registered_policy_and_cells():
    p = _protocol()
    m = _manifest()
    base = _disposition(m)

    changed = copy.deepcopy(base)
    changed['policy']['retention'] = 'last_success'
    changed['content_sha256'] = L._dispo_content_sha(
        changed['planned_cells'], changed['policy'])
    c = _clearance(p, disposition=changed)
    assert c['status'] == 'fail', c['problems']
    assert any('policy differs' in x for x in c['problems']), c['problems']

    short = copy.deepcopy(base)
    short['planned_cells'] = short['planned_cells'][:-1]
    short['content_sha256'] = L._dispo_content_sha(
        short['planned_cells'], short['policy'])
    c = _clearance(p, disposition=short)
    assert c['status'] == 'fail', c['problems']
    assert any('does not cover' in x for x in c['problems']), c['problems']


def test_power_gate_requires_a_matching_passed_artifact():
    p = _protocol()
    m = _manifest()

    c = _clearance(p, power=None)
    assert c['status'] == 'fail' and c['checks']['power']['status'] == 'fail'

    c = _clearance(p, power=_power(m, status='fail', all_pass=False))
    assert c['status'] == 'fail', c['problems']

    c = _clearance(p, power=_power(m, q=0.90))
    assert c['status'] == 'fail', c['problems']
    assert any('power_q' in x for x in c['problems']), c['problems']

    c = _clearance(p, power=_power(m, value=0.5))
    assert c['status'] == 'fail', c['problems']
    assert any('power at' in x for x in c['problems']), c['problems']

    c = _clearance(p, power=_power(m, n=4))
    assert c['status'] == 'fail', c['problems']
    assert any('min_n' in x for x in c['problems']), c['problems']

    # A calibration that passes on 12 seeds and matches q/alpha is accepted.
    assert _clearance(p, power=_power(m))['checks']['power']['status'] == 'ok'


def test_source_tree_drift_invalidates_the_clearance():
    p = _protocol()
    tree = copy.deepcopy(_tree())
    tree['files']['explib/_parts/part_10_paired_bootstrap_ci.py'][
        'sha256'] = '0' * 64
    c = _clearance(p, source_tree=tree)
    assert c['status'] == 'fail', c['problems']
    assert any('analysis implementation changed' in x for x in c['problems'])


def test_clearance_timing_is_prospective_not_reconstructed():
    p = _protocol()
    c = _clearance(p)
    early = [{'kind': 'cell', 'started_at': '2026-10-31T01:00:00'}]
    assert L.preparation_clearance_timing(c, early)['status'] == 'ok'
    late = [{'kind': 'cell', 'started_at': '2026-10-31T00:00:30'}]
    timing = L.preparation_clearance_timing(c, late)
    assert timing['status'] == 'fail', timing
    assert timing['reconstruction'] is True
    # An analysis event is a post-run read, not the start of execution.
    post = [{'kind': 'analysis', 'started_at': '2026-10-31T00:00:00'}]
    assert L.preparation_clearance_timing(c, post)['status'] != 'fail'


def test_clearance_audit_replays_and_detects_artifact_drift():
    p = _protocol()
    c = _clearance(p)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        assert L.write_preparation_clearance(td, c)['written'] is True
        for name, obj in (('run_order.json', _order(_manifest())),
                          ('run_disposition.json', _disposition(_manifest())),
                          ('power_validation.json', _power(_manifest()))):
            with open(os.path.join(td, name), 'w', encoding='utf-8') as fh:
                json.dump(obj, fh)
        events = [{'kind': 'cell', 'started_at': '2026-10-31T01:00:00'}]
        ok = L.preparation_clearance_audit(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'),
            events=events, rehearsal={'ok': True},
            dependency_contract=_dependency())
        assert ok['status'] == 'ok', (ok['problems'], ok['warnings'])
        assert ok['timing']['status'] == 'ok'

        # Change the plan on disk after the clearance: replay must notice.
        bad_order = _order(_manifest())
        bad_order['planned_cells'] = bad_order['planned_cells'][:-1]
        with open(os.path.join(td, 'run_order.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(bad_order, fh)
        drift = L.preparation_clearance_audit(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'),
            events=events, rehearsal={'ok': True},
            dependency_contract=_dependency())
        assert drift['status'] == 'fail', drift
        assert any('run_order' in x for x in drift['problems']), drift['problems']

        # A tampered body that does not reproduce its own hash fails even
        # before replay.
        tampered = copy.deepcopy(c)
        tampered['checks']['power']['status'] = 'fail'
        with open(os.path.join(td, 'preparation_clearance.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(tampered, fh)
        bad = L.preparation_clearance_audit(
            path=os.path.join(td, 'preparation_clearance.json'),
            protocol_path=os.path.join(td, 'design_protocol.json'),
            events=events)
        assert bad['status'] == 'fail', bad
        assert any('content_sha256' in x or 'power' in x
                   for x in bad['problems']), bad['problems']


def test_absent_clearance_is_not_a_pass():
    p = _protocol()
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        missing = os.path.join(td, 'preparation_clearance.json')
        rep = L.preparation_clearance_audit(
            path=missing,
            protocol_path=os.path.join(td, 'design_protocol.json'),
            events=[])
        assert rep['status'] == 'not_verifiable_locally', rep
        assert os.path.basename(rep.get('path') or '') == (
            'preparation_clearance.json')
        # The CLI gate delegates to the same function and remains a no-op on a
        # source-only checkout.
        assert hasattr(A, 'audit_preparation_clearance')
        gate = A.audit_preparation_clearance(
            path=missing, verbose=False)
        assert gate['status'] == 'not_verifiable_locally'


def test_writer_refuses_conflicts_and_accepts_a_superseding_revision():
    p = _protocol()
    c = _clearance(p)
    with tempfile.TemporaryDirectory() as td:
        first = L.write_preparation_clearance(td, c)
        assert first['written'] is True
        again = L.write_preparation_clearance(td, c)
        assert again['noop'] is True and again['written'] is False
        changed = _clearance(
            p, environment={'hostname': 'other-host', 'cuda_available': False})
        conflict = L.write_preparation_clearance(td, changed)
        assert conflict['conflict'] is True, conflict
        assert conflict['written'] is False
        revised = _clearance(
            p, environment={'hostname': 'other-host', 'cuda_available': False},
            revision=1, supersedes=[c['content_sha256']],
            reason='re-run the rehearsal on the same design')
        replaces = L.write_preparation_clearance(
            td, revised, keep_existing=False)
        assert replaces['written'] is True, replaces
        with open(os.path.join(td, 'preparation_clearance.json'),
                  encoding='utf-8') as fh:
            stored = json.load(fh)
        assert stored['revision'] == 1
        assert stored['supersedes'] == [c['content_sha256']]
