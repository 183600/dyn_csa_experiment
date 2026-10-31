"""CPU review checks (split group 69): design-protocol scale/robustness.

These pin defects in ``build_design_protocol`` / ``validate_design_protocol``
that a correctly sized fixture does not reach:

* the held-out access policy bypassed the shared ``test_access_policy``
  validator, so a typo or ``single_look``/``max_looks>1`` combination could be
  registered as a valid design decision;
* the exact Holm floor was computed as ``m * 2 / 2.0**n``, which raises
  ``OverflowError`` for a few thousand seeds instead of reporting the
  (vanishing) floor;
* duplicate seed identities or a seed list shorter than ``planned_seeds``
  could inflate the effective n used by the resolution check;
* the event reader treated ``panel_dirs=[]`` as "scan every ``results_*``",
  mixing repository history into an explicitly supplied event fixture and
  materialising every ledger record before looking for the first timestamp.
"""
import copy
import json
import math
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


def _protocol(manifest=None, **kw):
    manifest = copy.deepcopy(manifest or _manifest())
    cells = [(a, s) for a in manifest['arms'] for s in manifest['seeds']]
    kw.setdefault('study_name', 'fixed-chunk confirmation')
    kw.setdefault('run_order', L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=11,
        created_at='2026-10-31T00:00:00'))
    kw.setdefault('disposition', L.build_run_disposition_plan(
        ['%s::seed%s' % (a, s) for a in manifest['arms']
         for s in manifest['seeds']],
        created_at='2026-10-31T00:00:00'))
    kw.setdefault('test_access', {'method': 'single_look', 'max_looks': 1,
                                  'declared': True})
    kw.setdefault('disjoint_declared', True)
    kw.setdefault('created_at', '2026-10-31T00:00:01')
    return L.build_design_protocol(manifest, **kw)


def _rehash(p):
    p['commitments'] = {s: L._dp_sha_obj(p.get(s))
                        for s in L.DESIGN_PROTOCOL_SECTIONS}
    p['content_sha256'] = L.design_protocol_content_sha(p)
    return p


def test_invalid_held_out_access_policy_is_rejected_not_registered():
    # Every one of these would previously be copied into data.test_access and
    # validated only for "max_looks is an integer": an unknown method, a
    # single-look budget with two looks, and a misspelled key all passed.
    for bad in ({'method': 'bonferoni', 'max_looks': 5, 'declared': True},
                {'method': 'single_look', 'max_looks': 5, 'declared': True},
                {'max_look': 1, 'declared': True}):
        p = _protocol(test_access=bad)
        assert p['data']['test_access'].get('invalid') is True, p['data']
        v = L.validate_design_protocol(p)
        assert v['status'] == 'fail', (bad, v)
        assert any('test_access' in x for x in v['problems']), (bad, v)

    # A valid explicit declaration is recorded as declared even when the
    # caller did not repeat the provenance flag, and its source is preserved.
    p = _protocol(test_access={'method': 'single_look', 'max_looks': 1})
    assert p['data']['test_access']['declared'] is True
    assert p['data']['test_access']['source'] is None
    assert L.validate_design_protocol(p)['status'] in ('ok', 'warnings')

    # A hand-edited protocol cannot exceed the shared policy's budget either.
    too_many = _protocol()
    too_many['data']['test_access'].update(
        {'method': 'bonferroni', 'max_looks': 1001})
    too_many['stopping']['test_looks'] = 1001
    _rehash(too_many)
    v = L.validate_design_protocol(too_many)
    assert v['status'] == 'fail', v
    assert any('exceeds' in x for x in v['problems']), v


def test_missing_n_seeds_is_not_inferred_from_the_seed_list():
    # The manifest may carry a seed vector but omit statistics.n_seeds; the
    # protocol records missing decisions instead of turning the vector length
    # into a declaration the manifest never made.
    m = _manifest()
    del m['statistics']['n_seeds']
    p = _protocol(m)
    assert p['sample_size']['planned_seeds'] is None, p['sample_size']
    assert p['sample_size']['seeds'] == m['seeds']
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('planned_seeds' in x for x in v['problems']), v


def test_confirmatory_protocol_requires_an_explicit_held_out_budget():
    # The historical single-look default is a warning for an exploratory
    # draft, but a confirmatory registration must say so itself.
    p = _protocol(test_access=None)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('held-out access budget' in x for x in v['problems']), v
    draft = copy.deepcopy(p)
    draft['study']['level'] = 'draft'
    _rehash(draft)
    v2 = L.validate_design_protocol(draft)
    assert v2['status'] == 'warnings', v2
    assert any('held-out access budget' in x for x in v2['warnings']), v2


def test_resolution_floor_is_finite_for_a_large_seed_plan():
    # 2.0 ** 5000 is an OverflowError; the true floor is just 0.0.
    assert L._dp_resolution_floor(6, 5000) == 0.0
    assert L._dp_resolution_floor(6, 10 ** 100) == 0.0
    # A family size too large for a float is an unattainable floor, not a
    # crash, and still fails the alpha comparison.
    assert L._dp_resolution_floor(10 ** 400, 8) == float('inf')
    p = _protocol()
    p['sample_size']['min_n'] = 5000
    p['sample_size']['planned_seeds'] = 5000
    p['sample_size']['seeds'] = list(range(5000))
    _rehash(p)
    v = L.validate_design_protocol(p)
    assert v['status'] in ('ok', 'warnings'), v['problems']
    assert v['checks']['resolution_floor_at_min_n'] == 0.0
    assert v['checks']['resolution_floor_at_planned_n'] == 0.0


def test_duplicate_or_short_seed_list_cannot_inflate_planned_n():
    m = _manifest()
    m['seeds'] = list(range(8))
    m['statistics']['n_seeds'] = 8
    m['statistics']['min_seeds_for_confirmation'] = 8
    p = _protocol(m)
    assert L.validate_design_protocol(p)['status'] in ('ok', 'warnings')

    # 12 entries but only 8 independent seeds: the resolution check must not
    # treat this as n=12.
    forged = copy.deepcopy(p)
    forged['sample_size']['seeds'] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
    forged['sample_size']['planned_seeds'] = 12
    _rehash(forged)
    v = L.validate_design_protocol(forged)
    assert v['status'] == 'fail', v
    assert any('duplicate seed' in x for x in v['problems']), v

    short = copy.deepcopy(p)
    short['sample_size']['planned_seeds'] = 9
    _rehash(short)
    v = L.validate_design_protocol(short)
    assert v['status'] == 'fail', v
    assert any('planned_seeds' in x and 'seeds' in x for x in v['problems']), v


def test_non_finite_protocol_fails_closed_without_raising():
    p = _protocol()
    p['study']['name'] = math.nan
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('canonical JSON' in x for x in v['problems']), v

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'design_protocol.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(p, fh, allow_nan=True)
        audited = L.design_protocol_audit(
            path=path, manifest_path=None, panel_dirs=[], verbose=False)
        assert audited['status'] == 'fail', audited
        assert any('unreadable' in x for x in audited['problems']), audited


def test_missing_section_commitments_are_not_accepted():
    # Removing the per-section hashes leaves a self-consistent content hash
    # (they are redundant with it), so validation must name the missing
    # commitment block rather than silently skipping it.
    p = _protocol()
    p.pop('commitments')
    p['content_sha256'] = L.design_protocol_content_sha(p)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('commitments' in x for x in v['problems']), v


def test_absent_direction_is_not_silently_non_directional():
    # The frozen manifest uses an explicit ``expected_sign: null`` for its
    # sink-matched readout; that is a declared non-directional comparison.
    # An *absent* key, by contrast, is a missing design decision and must not
    # be converted into the same object.
    m = _manifest()
    p = _protocol(m)
    assert L.validate_design_protocol(p)['status'] in ('ok', 'warnings')

    missing = _manifest()
    missing['primary_comparisons'][0].pop('expected_sign')
    p2 = _protocol(missing)
    v = L.validate_design_protocol(p2)
    assert v['status'] == 'fail', v
    assert any('predicted_sign' in x for x in v['problems']), v

    hyp = _manifest()
    hyp['hypotheses'] = {'h1': {
        'statement': 'a statement without a direction',
        'endpoint': 'by_len.ppl@4096',
        'falsifier': 'the declared direction is not observed',
    }}
    p3 = _protocol(hyp)
    v3 = L.validate_design_protocol(p3)
    assert v3['status'] == 'fail', v3
    assert any('predicted_sign' in x for x in v3['problems']), v3


def test_conformance_accepts_an_embedded_disposition_plan():
    # A manifest may carry the complete run-disposition plan (kind +
    # planned_cells + policy) instead of only the policy block.  Build already
    # unwraps it; conformance must compare the policy rather than warning that
    # the wrapper is not a policy.
    m = _manifest()
    plan = L.build_run_disposition_plan(
        ['%s::seed%s' % (a, s) for a in m['arms'] for s in m['seeds']],
        created_at='2026-10-31T00:00:00')
    m['statistics']['disposition_policy'] = plan
    p = _protocol(m, disposition=plan)
    c = L.design_protocol_conformance(p, m)
    assert c['status'] == 'ok', c


def test_timing_accepts_a_stream_and_explicit_empty_panel_dirs_means_none(monkeypatch):
    p = _protocol()
    stream = ({'kind': 'cell', 'started_at': '2026-10-31T00:00:02'}
              for _ in (0,))
    out = L.design_protocol_timing(p, stream)
    assert out['status'] == 'ok', out
    assert L.design_protocol_timing(p, [])['warnings'][0].startswith(
        'no recorded events')

    with tempfile.TemporaryDirectory() as td:
        results = os.path.join(td, 'results_scale')
        os.makedirs(results)
        with open(os.path.join(results, 'experiments.jsonl'), 'w',
                  encoding='utf-8') as fh:
            # An earlier execution in the repository must not leak into a
            # caller-supplied fixture when panel_dirs=[] is explicit.
            fh.write(json.dumps({'kind': 'cell',
                                 'started_at': '2026-10-30T00:00:00'}) + '\n')
        monkeypatch.setattr(L, '_REPO', td)

        explicit = [{'kind': 'cell', 'started_at': '2026-10-31T00:00:02'}]
        evs = L._dp_read_events(events=explicit, panel_dirs=[])
        assert evs == explicit, evs
        # None still means "discover the default panel ledgers".
        assert len(L._dp_read_events(events=[], panel_dirs=None)) == 1

        L.write_design_protocol(td, p)
        path = os.path.join(td, 'design_protocol.json')
        audited = L.design_protocol_audit(
            path=path, manifest_path=MANIFEST, events=explicit,
            panel_dirs=[], verbose=False)
        assert audited['timing']['status'] == 'ok', audited['timing']
        assert audited['status'] in ('ok', 'warnings'), audited['problems']
