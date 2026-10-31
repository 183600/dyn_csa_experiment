"""Known-truth rehearsal of the frozen decision path (preparation, zero GPU).

The plan gate admits synthetic cells and the calibration suite tests the
statistical primitives, but neither exercises the integration that turns a
stored per-seed record into a frozen verdict.  These checks plant known effects
into the real comparison family and verify that exact sign-flip + declared
family Holm + directional/coverage gates return the expected answer, and that
the gate refuses a rehearsal whose body or manifest binding was changed.
"""
import copy
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO, 'confirmation_frozen.json')


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _toy_manifest(*, n=12, min_n=4, alpha=0.05):
    return {
        'manifest_version': 3,
        'seeds': list(range(n)),
        'statistics': {
            'alpha': alpha,
            'min_seeds_for_confirmation': min_n,
            'require_complete_coverage': True,
        },
        'primary_comparisons': [
            {'name': 'toy:a-b', 'a': 'a', 'b': 'b',
             'expected_sign': 'positive'},
            {'name': 'toy:a-c', 'a': 'a', 'b': 'c',
             'expected_sign': 'negative'},
            {'name': 'toy:b-c', 'a': 'b', 'b': 'c',
             'expected_sign': None},
        ],
    }


def _rehash(rehearsal):
    body = {k: v for k, v in rehearsal.items()
            if k not in ('content_sha256', 'created_at', 'revision',
                         'supersedes', 'reason', 'notes')}
    rehearsal = copy.deepcopy(rehearsal)
    rehearsal['content_sha256'] = L._ar_sha_obj(body)
    return rehearsal


def test_rehearsal_binds_the_real_frozen_family_and_passes():
    r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    assert r['kind'] == 'analysis_rehearsal'
    assert r['status'] == 'ok', r['failures']
    assert r['n_failures'] == 0
    assert r['synthetic'] is True and r['test_split_read'] is False
    assert r['family_size'] == 6
    assert set(r['scenarios']) == {
        'null_balanced', 'forward_effect', 'reverse_effect',
        'underpowered', 'family_correction', 'omitted_comparison'}
    assert all(s['ok'] for s in r['scenarios'].values())
    assert all(s['synthetic'] is True for s in r['scenarios'].values())
    assert L.validate_analysis_rehearsal(r)['status'] == 'ok'
    assert L.validate_analysis_rehearsal(r, manifest=_manifest())['status'] == 'ok'
    assert L.analysis_rehearsal_summary(r)['ok'] is True

    # The live gate is the same object; no stored artifact is required on a
    # source-only checkout.
    gate = A.audit_analysis_rehearsal()
    assert gate['status'] == 'ok', gate['problems']
    assert gate['live']['n_scenarios'] == 6
    assert gate['test_split_read'] is False


def test_confirmation_plan_embeds_the_rehearsal():
    rep = V12.plan_check(MANIFEST)
    assert rep['ok'], rep['errors']
    ar = rep.get('analysis_rehearsal')
    assert isinstance(ar, dict) and ar.get('status') == 'ok', ar
    assert ar.get('synthetic') is True and ar.get('test_split_read') is False
    assert any('analysis rehearsal:' in n for n in rep['notes'])


def test_rehearsal_rejects_a_pipeline_that_always_rejects():
    original = L.exact_signflip_p

    def always_zero(deltas):
        return {'n': len(list(deltas)), 'p': 0.0, 'n_dropped': 0,
                'n_nonzero': len(list(deltas)), 'p_floor': 0.0,
                'method': 'rehearsal-mutation'}

    L.exact_signflip_p = always_zero
    try:
        r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    finally:
        L.exact_signflip_p = original
    # A null panel must not be declared a discovery, and a uniform reverse
    # effect must not be confused with the expected direction.
    assert r['status'] == 'fail'
    assert any('null_balanced' in f for f in r['failures']), r['failures']


def test_rehearsal_rejects_family_shrinkage():
    original = L.holm_adjust

    def no_correction(pairs, alpha=0.05):
        out = {}
        for i, (name, p) in enumerate(pairs):
            out[str(name)] = {'p_raw': p, 'p_holm': p, 'rank': i + 1,
                              'reject': bool(p < alpha), 'm_effective': 1}
        return out

    L.holm_adjust = no_correction
    try:
        r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    finally:
        L.holm_adjust = original
    assert r['status'] == 'fail'
    assert any('family_correction' in f for f in r['failures']), r['failures']
    # The same family slot count is part of the control.
    assert any('family_slot_not_shrunk' in f for f in r['failures'])


def test_rehearsal_validator_rejects_tamper_and_mislabel():
    r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    for mutate, needle in (
            (lambda x: x.update({'synthetic': False}), 'synthetic'),
            (lambda x: x.update({'test_split_read': True}), 'test_split_read'),
            (lambda x: x.update({'status': 'fail'}), 'status'),
            (lambda x: x.update({'n_failures': 1}), 'n_failures'),
            (lambda x: x['scenarios']['null_balanced'].update({'ok': False}),
             'null_balanced')):
        bad = copy.deepcopy(r)
        mutate(bad)
        verdict = L.validate_analysis_rehearsal(_rehash(bad))
        assert verdict['status'] == 'fail', verdict
        assert any(needle in p for p in verdict['problems']), verdict
    # A manifest mismatch is a hard binding failure even with a valid body.
    other = _toy_manifest()
    verdict = L.validate_analysis_rehearsal(_rehash(r), manifest=other)
    assert verdict['status'] == 'fail', verdict
    assert any('different frozen manifest' in p for p in verdict['problems'])
    # A body edit without recomputing the content hash is rejected first.
    edited = copy.deepcopy(r)
    edited['alpha'] = 0.10
    verdict = L.validate_analysis_rehearsal(edited)
    assert verdict['status'] == 'fail'
    assert any('does not reproduce' in p for p in verdict['problems'])


def test_rehearsal_handles_generic_and_absent_manifests():
    toy = _toy_manifest()
    r = L.build_analysis_rehearsal(toy, manifest_path='toy.json',
                                   created_at='2026-01-01T00:00:00')
    assert r['status'] == 'ok', r['failures']
    # Three declared slots can hold a raw-significant/Holm-null middle member,
    # so the family scenario is present and must pass.
    assert 'family_correction' in r['scenarios']
    assert len(r['scenarios']) == 6
    assert L.validate_analysis_rehearsal(r, manifest=toy)['status'] == 'ok'
    # An alpha so large that no discrete exact p can be raw-significant while
    # Holm-null omits the scenario instead of asserting an impossible trap.
    wide = L.build_analysis_rehearsal(
        _toy_manifest(alpha=0.10), manifest_path='toy.json',
        created_at='2026-01-01T00:00:00')
    assert wide['status'] == 'ok', wide['failures']
    assert 'family_correction' not in wide['scenarios']
    assert len(wide['scenarios']) == 5

    with tempfile.TemporaryDirectory() as td:
        missing = L.build_analysis_rehearsal(
            manifest_path=os.path.join(td, 'absent.json'))
        assert missing['status'] == 'not_verifiable', missing
        assert missing['synthetic'] is True
        assert missing['test_split_read'] is False


def test_rehearsal_writer_and_snapshot_gate():
    r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    toy = L.build_analysis_rehearsal(_toy_manifest(alpha=0.10),
                                     manifest_path='toy.json',
                                     created_at='2026-01-01T00:00:00')
    assert toy['status'] == 'ok', toy['failures']
    with tempfile.TemporaryDirectory() as td:
        first = L.write_analysis_rehearsal(td, r)
        assert first['written'] is True
        again = L.write_analysis_rehearsal(td, r)
        assert again['unchanged'] is True and again['written'] is False

        stored = os.path.join(td, 'analysis_rehearsal.json')
        gate = A.audit_analysis_rehearsal([stored])
        assert gate['status'] == 'ok', gate['problems']

        # A different valid rehearsal cannot silently replace the stored one,
        # and after an explicit overwrite it no longer names the frozen
        # manifest, so the gate must fail the binding rather than accept it.
        conflict = L.write_analysis_rehearsal(td, toy)
        assert conflict['conflict'] is True and conflict['written'] is False
        replaced = L.write_analysis_rehearsal(td, toy, overwrite=True)
        assert replaced['written'] is True
        mismatched = A.audit_analysis_rehearsal([stored])
        assert mismatched['status'] == 'fail', mismatched
        assert any('different frozen manifest' in p
                   for p in mismatched['problems']), mismatched

        snap = os.path.join(td, 'plan_snapshot.json')
        with open(snap, 'w', encoding='utf-8') as fh:
            json.dump({'plan': {'analysis_rehearsal': r}}, fh)
        embedded = A.audit_analysis_rehearsal([snap])
        assert embedded['status'] == 'ok', embedded['problems']

        bad = copy.deepcopy(r)
        bad['scenarios']['underpowered']['ok'] = False
        with open(snap, 'w', encoding='utf-8') as fh:
            json.dump({'plan': {'analysis_rehearsal': _rehash(bad)}}, fh)
        broken = A.audit_analysis_rehearsal([snap])
        assert broken['status'] == 'fail', broken
        assert any('underpowered' in p for p in broken['problems']), broken

        absent = A.audit_analysis_rehearsal([os.path.join(td, 'nope.json')])
        assert absent['status'] == 'not_verifiable', absent


def test_clearance_binds_the_embedded_rehearsal_without_copying_p_values():
    r = L.build_analysis_rehearsal(manifest_path=MANIFEST)
    protocol = {
        'study': {'level': 'confirmatory',
                  'manifest_sha256': r['manifest_sha256']},
        'analysis': {},
    }
    good = L._prep_rehearsal_check(
        protocol, {'ok': True, 'analysis_rehearsal': r})
    assert good['status'] == 'ok', good
    assert good['analysis_rehearsal']['status'] == 'ok'
    assert good['analysis_rehearsal']['content_sha256'] == r['content_sha256']
    blob = json.dumps(good, sort_keys=True)
    for forbidden in ('p_holm', 'p_exact_signflip', 'p_value'):
        assert forbidden not in blob, (forbidden, blob)

    bad = copy.deepcopy(r)
    bad['status'] = 'fail'
    failed = L._prep_rehearsal_check(
        protocol, {'ok': True, 'analysis_rehearsal': _rehash(bad)})
    assert failed['status'] == 'fail'
    assert any('known-truth analysis rehearsal' in p
               for p in failed['problems']), failed['problems']

    mismatch = dict(protocol, study={'level': 'confirmatory',
                                     'manifest_sha256': '0' * 64})
    wrong = L._prep_rehearsal_check(
        mismatch, {'ok': True, 'analysis_rehearsal': r})
    assert wrong['status'] == 'fail'
    assert any('different frozen manifest' in p for p in wrong['problems'])

    # A legacy/bare rehearsal is recorded but not silently upgraded.
    legacy = L._prep_rehearsal_check(protocol, {'ok': True})
    assert legacy['status'] == 'ok'
    assert legacy['analysis_rehearsal']['status'] == 'missing'


def test_rehearsal_is_wired_into_report_and_cli_aliases():
    src = inspect.getsource(A.run_report)
    assert "'analysis_rehearsal': audit_analysis_rehearsal()" in src
    assert A.audit_analysis_rehearsal is not None
