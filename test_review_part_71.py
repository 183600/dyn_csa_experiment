"""CPU checks (split group 71): a registration must bind the analysis code.

The design protocol already binds the manifest, run order, disposition and
held-out budget.  Those are declarations; the endpoint is computed by code.
A protocol whose manifest stays byte-identical while the analysis closure
changes is not a pre-registration of the executed analysis.  These checks
exercise the analysis-closure digest, its narrow scope (unrelated panel
drivers and the dynamic draft must not invalidate a fixed-chunk registration),
and the conformance/audit failure mode.  No training, GPU or test split is
read.
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


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _order(manifest):
    cells = [(a, s) for a in manifest['arms'] for s in manifest['seeds']]
    return L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=11,
        created_at='2026-10-31T00:00:00')


def _disposition(manifest):
    return L.build_run_disposition_plan(
        ['%s::seed%s' % (a, s) for a in manifest['arms']
         for s in manifest['seeds']],
        created_at='2026-10-31T00:00:00')


def _protocol(manifest=None, *, source_tree=None, created_at='2026-10-31T00:00:01'):
    manifest = copy.deepcopy(manifest or _manifest())
    return L.build_design_protocol(
        manifest,
        study_name='fixed-chunk confirmation',
        run_order=_order(manifest),
        disposition=_disposition(manifest),
        test_access={'method': 'single_look', 'max_looks': 1,
                     'declared': True},
        disjoint_declared=True,
        created_at=created_at,
        source_tree=source_tree)


def _rehash(p):
    p['commitments'] = {s: L._dp_sha_obj(p.get(s))
                        for s in L.DESIGN_PROTOCOL_SECTIONS}
    p['content_sha256'] = L.design_protocol_content_sha(p)
    return p


def _tree():
    return L.source_tree_manifest()


def test_registration_binds_the_analysis_closure():
    tree = _tree()
    p = _protocol(source_tree=tree)
    impl = p['analysis']['implementation']
    assert isinstance(impl, dict), impl
    assert impl['analysis_source_sha256'] == L._dp_subset_source_sha(
        tree['files'], L._dp_analysis_relpaths(tree['files']))
    assert impl['analysis_n_files'] >= 60, impl['analysis_n_files']
    assert impl['full_source_tree_sha256'] == tree['sha256']
    assert impl['split_registry_status'] == 'ok'
    assert impl['split_registry_n_parts'] == tree['split_registry']['n_parts']
    selected = set(L._dp_analysis_relpaths(tree['files']))
    # The endpoint implementation and the measurement library are bound.
    assert 'explib/confirmation/_parts/part_03_finaltest_stats.py' in selected
    assert 'explib/_parts/part_10_paired_bootstrap_ci.py' in selected
    assert 'exp_lib.py' in selected and '_split_loader.py' in selected
    # Other panels and their manifests are deliberately outside the hard
    # closure: they can change without rewriting this endpoint.
    assert not any(x.startswith('explib/rope/') for x in selected)
    assert 'run_gapfill.py' not in selected
    assert 'confirmation_frozen_dynamic.json' not in selected
    v = L.validate_design_protocol(p)
    assert not v['problems'], v['problems']
    assert v['status'] in ('ok', 'warnings'), v


def test_confirmatory_registration_without_code_binding_fails_closed():
    p = _protocol(source_tree={})
    # An empty/failed source-tree hash is exactly the missing decision: the
    # builder must not invent a digest, and a confirmatory protocol must not
    # claim to be registered without one.
    assert p['analysis']['implementation'] is None, p['analysis']
    _rehash(p)
    v = L.validate_design_protocol(p)
    assert v['status'] == 'fail', v
    assert any('analysis.implementation' in x for x in v['problems']), v
    draft = copy.deepcopy(p)
    draft['study']['level'] = 'draft'
    _rehash(draft)
    v2 = L.validate_design_protocol(draft)
    assert v2['status'] == 'warnings', v2
    assert any('analysis.implementation' in x for x in v2['warnings']), v2


def test_analysis_closure_digest_is_narrow_and_order_sensitive():
    tree = _tree()
    manifest = _manifest()
    p = _protocol(manifest, source_tree=tree)
    registered = p['analysis']['implementation']['analysis_source_sha256']

    # A file outside the analysis closure does not change the closure digest.
    unrelated = copy.deepcopy(tree)
    outside = next(rel for rel in unrelated['files']
                   if rel.startswith('explib/rope/'))
    unrelated['files'][outside]['sha256'] = '0' * 64
    assert L._dp_subset_source_sha(
        unrelated['files'],
        L._dp_analysis_relpaths(unrelated['files'])) == registered

    # A core analysis fragment does change it.
    core = copy.deepcopy(tree)
    inside = 'explib/_parts/part_10_paired_bootstrap_ci.py'
    core['files'][inside]['sha256'] = '1' * 64
    changed = L._dp_subset_source_sha(
        core['files'], L._dp_analysis_relpaths(core['files']))
    assert changed != registered

    # The split-package declaration (the executed order) is a bound file too:
    # reordering _PARTS changes the __init__.py digest and therefore the
    # analysis closure, even if every fragment file is byte-identical.
    reordered = copy.deepcopy(tree)
    init = 'explib/__init__.py'
    reordered['files'][init]['sha256'] = '2' * 64
    assert L._dp_subset_source_sha(
        reordered['files'],
        L._dp_analysis_relpaths(reordered['files'])) != registered


def test_changed_analysis_code_invalidates_conformance_not_the_manifest():
    manifest = _manifest()
    tree = _tree()
    p = _protocol(manifest, source_tree=tree)
    ok = L.design_protocol_conformance(p, manifest, source_tree=tree)
    assert ok['status'] == 'ok', ok
    assert ok['implementation']['status'] == 'ok', ok['implementation']

    # Same manifest, changed analysis closure.
    changed = copy.deepcopy(tree)
    changed['files']['explib/_parts/part_03_pool_geom_flat_single.py'][
        'sha256'] = '3' * 64
    changed['sha256'] = '4' * 64
    bad = L.design_protocol_conformance(p, manifest, source_tree=changed)
    assert bad['status'] == 'fail', bad
    assert bad['implementation']['status'] == 'fail', bad['implementation']
    assert any('analysis implementation changed' in x
               for x in bad['implementation']['problems']), bad

    # Same analysis closure, unrelated source drift: visible but not fatal.
    outside = next(rel for rel in tree['files'] if rel.startswith('explib/rope/'))
    peripheral = copy.deepcopy(tree)
    peripheral['files'][outside]['sha256'] = '5' * 64
    peripheral['sha256'] = '6' * 64
    warn = L.design_protocol_conformance(p, manifest, source_tree=peripheral)
    assert warn['status'] == 'warnings', warn
    assert warn['implementation']['status'] == 'warnings', warn['implementation']
    assert any('outside the analysis closure' in x
               for x in warn['implementation']['warnings']), warn


def test_registered_code_semantics_must_match_the_protocol():
    p = _protocol(source_tree=_tree())
    changed = copy.deepcopy(p)
    changed['analysis']['implementation']['code_semantics'] = 'not-the-same'
    _rehash(changed)
    v = L.validate_design_protocol(changed)
    assert v['status'] == 'fail', v
    assert any('implementation.code_semantics' in x for x in v['problems']), v

    c = L.design_protocol_conformance(changed, _manifest(), source_tree=_tree())
    assert c['status'] == 'fail', c
    assert any('code_semantics' in x for x in c['implementation']['problems']), c


def test_malformed_implementation_binding_is_not_a_pass():
    p = _protocol(source_tree=_tree())
    for mutator, needle in (
            (lambda x: x['analysis']['implementation'].__setitem__(
                'analysis_source_sha256', 'not-a-digest'),
             'analysis_source_sha256'),
            (lambda x: x['analysis']['implementation'].__setitem__(
                'analysis_n_files', 0), 'analysis_n_files'),
            (lambda x: x['analysis']['implementation'].__setitem__(
                'source_unreadable', [{'path': 'x.py'}]), 'source_unreadable')):
        bad = copy.deepcopy(p)
        mutator(bad)
        _rehash(bad)
        v = L.validate_design_protocol(bad)
        assert v['status'] == 'fail', (needle, v)
        assert any(needle in x for x in v['problems']), (needle, v)


def test_implementation_is_part_of_the_registration_identity():
    tree = _tree()
    p = _protocol(source_tree=tree)
    moved = copy.deepcopy(p)
    moved['created_at'] = '2026-11-01T00:00:00'
    assert moved['content_sha256'] == L.design_protocol_content_sha(moved)
    changed = copy.deepcopy(p)
    changed['analysis']['implementation']['analysis_n_files'] += 1
    _rehash(changed)
    assert changed['content_sha256'] != p['content_sha256']


def test_audit_surfaces_implementation_drift_without_reading_test_data():
    tree = _tree()
    p = _protocol(source_tree=tree)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        path = os.path.join(td, 'design_protocol.json')
        good = L.design_protocol_audit(
            path=path, manifest_path=MANIFEST, events=[], source_tree=tree,
            verbose=False)
        assert good['status'] == 'warnings', good  # timing is not observable
        assert good['implementation']['status'] == 'ok', good['implementation']

        changed = copy.deepcopy(tree)
        changed['files']['explib/confirmation/_parts/part_03_finaltest_stats.py'][
            'sha256'] = '7' * 64
        changed['sha256'] = '8' * 64
        drift = L.design_protocol_audit(
            path=path, manifest_path=MANIFEST, events=[], source_tree=changed,
            verbose=False)
        assert drift['status'] == 'fail', drift
        assert drift['implementation']['status'] == 'fail', drift['implementation']
        assert any('analysis implementation changed' in x
                   for x in drift['problems']), drift
        # The gate reports the drift, it does not create a test, p-value or
        # confirmation verdict.
        blob = json.dumps(drift, sort_keys=True)
        for forbidden in ('p_holm', 'p_exact_signflip', 'confirmed', 'verified',
                          'p_value'):
            assert forbidden not in blob, forbidden


def test_preflight_summary_surfaces_the_implementation_status():
    tree = _tree()
    p = _protocol(source_tree=tree)
    with tempfile.TemporaryDirectory() as td:
        L.write_design_protocol(td, p)
        summary = L.design_protocol_preflight_summary(
            path=os.path.join(td, 'design_protocol.json'),
            manifest_path=MANIFEST)
    assert summary['implementation'] == 'ok', summary
    assert summary['status'] == 'warnings', summary
