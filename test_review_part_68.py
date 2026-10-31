"""CPU review checks (split group 67): prospective design registration.

The design is only prospective if its decisions are bound together *before*
the first cell.  These checks cover ``build_design_protocol`` /
``validate_design_protocol`` / ``design_protocol_conformance`` /
``design_protocol_timing`` / ``write_design_protocol`` and the
``audit_gates.design_protocol`` replay gate.
"""
import contextlib
import copy
import inspect
import io
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


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _order(manifest, *, seed=11):
    cells = [(a, s) for a in manifest['arms'] for s in manifest['seeds']]
    return L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=seed,
        created_at='2026-10-31T00:00:00')


def _disposition(manifest):
    return L.build_run_disposition_plan(
        ['%s::seed%s' % (a, s) for a in manifest['arms']
         for s in manifest['seeds']],
        created_at='2026-10-31T00:00:00')


def _access():
    return {'method': 'single_look', 'max_looks': 1,
            'require_reason_for_reread': True, 'declared': True,
            'source': 'test'}


def _protocol(manifest=None, **kw):
    manifest = copy.deepcopy(manifest or _manifest())
    kw.setdefault('study_name', 'fixed-chunk confirmation')
    kw.setdefault('run_order', _order(manifest))
    kw.setdefault('disposition', _disposition(manifest))
    kw.setdefault('test_access', _access())
    kw.setdefault('disjoint_declared', True)
    kw.setdefault('created_at', '2026-10-31T00:00:01')
    return L.build_design_protocol(manifest, **kw)


def _rehash(p):
    p['commitments'] = {s: L._dp_sha_obj(p.get(s))
                        for s in L.DESIGN_PROTOCOL_SECTIONS}
    p['content_sha256'] = L.design_protocol_content_sha(p)
    return p


def test_complete_protocol_is_canonical_and_hash_bound():
    p = _protocol()
    v = L.validate_design_protocol(p)
    assert v['status'] == 'warnings', (v['problems'], v['warnings'])
    assert not v['problems'], v['problems']
    assert p['kind'] == L.DESIGN_PROTOCOL_KIND
    assert p['schema'] == L.DESIGN_PROTOCOL_SCHEMA
    assert p['study']['manifest_sha256'] == L._dp_manifest_sha(_manifest())
    assert p['sample_size']['family_size'] == 6
    assert p['sample_size']['planned_seeds'] == 12
    assert p['sample_size']['min_n'] == 8
    assert p['randomization']['design_seed'] == 11
    assert len(p['randomization']['planned_cells']) == 72
    assert p['operations']['disposition']['retention'] == 'first_success'
    # The timestamp/revision metadata is outside the content identity; the
    # scientific body is not.
    assert p['content_sha256'] == L.design_protocol_content_sha(p)
    moved = copy.deepcopy(p)
    moved['created_at'] = '2026-11-01T00:00:00'
    assert moved['content_sha256'] == L.design_protocol_content_sha(moved)
    changed = copy.deepcopy(p)
    changed['sample_size']['min_n'] = 9
    assert L.design_protocol_content_sha(changed) != p['content_sha256']
    assert set(p['commitments']) == set(L.DESIGN_PROTOCOL_SECTIONS)


def test_missing_design_decisions_fail_a_confirmatory_protocol():
    m = _manifest()
    p = L.build_design_protocol(m, study_name='unregistered',
                                created_at='2026-10-31T00:00:01')
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('randomization' in x for x in v['problems']), v
    assert any('operations.disposition' in x for x in v['problems']), v
    # The same missing pieces are only warnings for an exploratory/draft
    # study; the gate does not silently promote it to confirmatory.
    p2 = copy.deepcopy(p)
    p2['study']['level'] = 'draft'
    _rehash(p2)
    v2 = L.validate_design_protocol(p2)
    assert v2['status'] == 'warnings', v2


def test_resolution_floor_rejects_an_impossible_seed_plan():
    m = _manifest()
    m['seeds'] = [0, 1, 2, 3, 4]
    m['statistics']['n_seeds'] = 5
    m['statistics']['min_seeds_for_confirmation'] = 5
    p = _protocol(m)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('resolution floor' in x for x in v['problems']), v
    # 8 seeds still give 6*2/2**8 = 0.046875 < 0.05 for the frozen m=6 family.
    m2 = _manifest()
    p2 = _protocol(m2)
    assert L.validate_design_protocol(p2)['status'] in ('ok', 'warnings')


def test_tampered_body_no_longer_matches_its_registration():
    p = _protocol()
    p['research_question']['hypotheses'][0]['falsifier'] = 'anything I want'
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('content_sha256' in x for x in v['problems']), v
    p2 = _protocol()
    p2['operations']['disposition']['retention'] = 'last_success'
    _rehash(p2)
    v2 = L.validate_design_protocol(p2)
    assert v2['status'] == 'fail', v2
    assert any('first_success' in x for x in v2['problems']), v2


def test_conformance_detects_a_manifest_that_changed_after_registration():
    m = _manifest()
    p = _protocol(m)
    assert L.design_protocol_conformance(p, m)['status'] == 'ok'
    for mutator in (
            lambda x: x['arms'].append('new_arm'),
            lambda x: x['seeds'].append(99),
            lambda x: x['statistics'].__setitem__('alpha', 0.10),
            lambda x: x['primary_comparisons'].pop()):
        bad = copy.deepcopy(m)
        mutator(bad)
        c = L.design_protocol_conformance(p, bad)
        assert c['status'] == 'fail', c
        assert c['problems'], c


def test_timing_requires_the_protocol_before_the_first_event():
    p = _protocol()
    ok = L.design_protocol_timing(
        p, [{'kind': 'cell', 'started_at': '2026-10-31T00:00:02'}])
    assert ok['status'] == 'ok' and ok['reconstruction'] is False, ok
    late = L.design_protocol_timing(
        p, [{'kind': 'panel_start', 'started_at': '2026-10-30T23:59:59'},
            {'kind': 'cell', 'started_at': '2026-10-31T00:00:02'}])
    assert late['status'] == 'fail' and late['reconstruction'] is True, late
    assert 'reconstruction' in late['problems'][0], late
    none = L.design_protocol_timing(p, [])
    assert none['status'] == 'not_verifiable', none
    nav = L.design_protocol_timing(
        p, [{'kind': 'cell', 'started_at': 'not-a-time'}])
    assert nav['status'] == 'not_verifiable', nav
    # A post-run analysis/report event is not an execution start; an earlier
    # exploratory analysis must not make a later confirmatory registration
    # look retrospective.
    ignored = L.design_protocol_timing(
        p, [{'kind': 'analysis', 'created_at': '2026-10-30T00:00:00'},
            {'kind': 'cell', 'started_at': '2026-10-31T00:00:02'}])
    assert ignored['status'] == 'ok', ignored


def test_writer_never_silently_replaces_a_registration():
    p = _protocol()
    with tempfile.TemporaryDirectory() as td:
        first = L.write_design_protocol(td, p)
        assert first['written'] is True and first['conflict'] is False, first
        again = L.write_design_protocol(td, copy.deepcopy(p))
        assert again['written'] is False and again.get('noop') is True, again
        conflict = L.write_design_protocol(
            td, _protocol(study_name='a different study'))
        assert conflict['written'] is False and conflict['conflict'] is True, conflict
        with open(os.path.join(td, 'design_protocol.json'), encoding='utf-8') as fh:
            assert json.load(fh)['study']['name'] == 'fixed-chunk confirmation'
        revised = _protocol(study_name='a different study', revision=1,
                            supersedes=[p['content_sha256']],
                            reason='broaden the study name before the run')
        out = L.write_design_protocol(td, revised, keep_existing=False)
        assert out['written'] is True, out
        with open(os.path.join(td, 'design_protocol.json'), encoding='utf-8') as fh:
            got = json.load(fh)
        assert got['revision'] == 1 and got['supersedes'] == [p['content_sha256']]
        assert got['reason'] == 'broaden the study name before the run'


def test_audit_absence_is_not_a_pass_and_tampering_is_a_fail():
    with tempfile.TemporaryDirectory() as empty:
        absent = L.design_protocol_audit(
            path=os.path.join(empty, 'does-not-exist-dp.json'),
            manifest_path=MANIFEST, verbose=False)
    assert absent['status'] == 'not_verifiable_locally', absent
    assert absent['required_sections'] == list(L.DESIGN_PROTOCOL_SECTIONS)
    p = _protocol()
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        path = os.path.join(td, 'design_protocol.json')
        ok = L.design_protocol_audit(path=path, manifest_path=MANIFEST,
                                     verbose=False)
        assert ok['status'] == 'warnings', ok  # timing is not observable here
        assert ok['validation']['status'] in ('ok', 'warnings'), ok['validation']
        raw = json.load(open(path, encoding='utf-8'))
        raw['study']['alpha'] = 0.5
        json.dump(raw, open(path, 'w', encoding='utf-8'), allow_nan=False)
        bad = L.design_protocol_audit(path=path, manifest_path=MANIFEST,
                                      verbose=False)
        assert bad['status'] == 'fail', bad
        assert any('content_sha256' in x for x in bad['problems']), bad


def test_audit_replays_timing_and_manifest_conformance():
    p = _protocol()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'design_protocol.json')
        L.write_design_protocol(td, p)
        late = L.design_protocol_audit(
            path=path, manifest_path=MANIFEST,
            events=[{'kind': 'cell', 'started_at': '2026-10-31T00:00:00'}],
            verbose=False)
        assert late['status'] == 'fail', late
        assert any('reconstruction' in x for x in late['problems']), late
        changed = _manifest()
        changed['statistics']['alpha'] = 0.10
        bad_manifest = os.path.join(td, 'confirmation_frozen.json')
        json.dump(changed, open(bad_manifest, 'w', encoding='utf-8'))
        drift = L.design_protocol_audit(
            path=path, manifest_path=bad_manifest, verbose=False)
        assert drift['status'] == 'fail', drift
        assert any('manifest_sha256' in x for x in drift['problems']), drift


def test_gate_and_design_preflight_are_wired_without_creating_a_test():
    assert 'design_protocol' in inspect.getsource(A.run_report)
    assert 'design_protocol' in inspect.getsource(A._run_cli)
    with contextlib.redirect_stdout(io.StringIO()):
        gate = A.audit_design_protocol(verbose=False)
    assert gate['kind'] == 'design_protocol_audit'
    assert gate['status'] == 'not_verifiable_locally', gate
    with contextlib.redirect_stdout(io.StringIO()):
        dp = A.audit_design_preflight()
    assert dp['status'] == 'ok', dp['violations']
    block = dp.get('design_protocol') or {}
    assert block.get('status') == 'not_verifiable_locally', block
    p = _protocol()
    for forbidden in ('p_holm', 'p_exact_signflip', 'verified', 'confirmed',
                      'verdict', 'decision'):
        assert forbidden not in p, p


def test_planned_cell_grid_must_match_declared_arms_and_seeds():
    p = _protocol()
    p['randomization']['planned_cells'].append('ghost_arm::seed0')
    _rehash(p)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('undeclared arm' in x for x in v['problems']), v
    p2 = _protocol()
    p2['randomization']['planned_cells'].append('full_rope::seed99')
    _rehash(p2)
    v2 = L.validate_design_protocol(p2)
    assert v2['status'] == 'fail', v2
    assert any('not in sample_size.seeds' in x for x in v2['problems']), v2


def test_unknown_direction_label_is_not_silently_non_directional():
    m = _manifest()
    m['primary_comparisons'][0]['expected_sign'] = 'up-ish'
    p = _protocol(m)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('predicted_sign' in x for x in v['problems']), v
