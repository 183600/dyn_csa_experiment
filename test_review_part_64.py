"""CPU review checks (split group 64): cross-panel evidence synthesis.

``ablation_coverage_across_panels`` counts verified panels; it deliberately
did not pool them.  These checks cover the new quantitative synthesis: it
include every estimable panel (not only the significant ones), refuses
poolability when the panels are not commensurable or their draws are
dependent, reports heterogeneity and a verified-only selection-bias readout,
and replays its own arithmetic in ``audit_gates.py panel_synthesis``.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _entry(deltas, seeds, *, pred=-1, without=100.0, with_arm='full',
           without_arm='full-A'):
    return {'controlled': True, 'fully_paired': True,
            'with': with_arm, 'without': without_arm,
            'kind': 'component', 'deltas': list(deltas), 'seeds': list(seeds),
            'without_mean': float(without),
            'relative_effect_pct': (100.0 * sum(deltas) / len(deltas) / without),
            'predicted_sign': pred, 'observed_sign': pred,
            'direction_consistent': True, 'practical_decision': 'superior'}


def _rep(entries, seeds, eval_fp, *, corpus='c1', verified=True,
         comp='A', min_n=3, alpha=0.05, conflict=False, harmful=False,
         incomplete=False):
    return {'components': {comp: entries},
            'verified_components': [comp] if verified else [],
            'direction_conflict_components': [comp] if conflict else [],
            'practically_harmful_components': [comp] if harmful else [],
            'ablation_completeness': {
                'components': {comp: {}},
                'complete_components': [] if incomplete else [comp],
                'incomplete_components': [comp] if incomplete else [],
                'missing_arms': {}},
            'component_draws': {comp: {
                'seeds': list(seeds), 'eval_fps': [eval_fp],
                'corpus_fps': [corpus]}},
            'min_n': min_n, 'alpha': alpha}


_DESIGN = {'A': [('full', 'full-A', 'component A is useful')]}


def _consistent_panels():
    return {
        'p1': _rep([_entry([-2.0, -2.1, -1.9], [0, 1, 2])], [0, 1, 2], 'e1'),
        'p2': _rep([_entry([-2.2, -1.8, -2.0], [10, 11, 12])],
                   [10, 11, 12], 'e2'),
        'p3': _rep([_entry([-1.9, -2.0, -2.1], [20, 21, 22])],
                   [20, 21, 22], 'e3'),
    }


def _block(syn, comp='A', name='full - full-A'):
    return syn['components'][comp]['contrasts'][name]


def test_cross_panel_synthesis_pools_every_estimable_panel():
    syn = L.cross_panel_synthesis(_consistent_panels(), _DESIGN)
    assert syn['status'] in ('ok', 'warn'), syn
    b = _block(syn)
    assert b['status'] == 'poolable', b
    assert b['claim_allowed'] is True, b
    assert b['evidence_tier'] == 'independent_replication', b
    assert b['meta']['seed_dependence'] == 'disjoint_seed_sets', b['meta']
    assert b['meta']['random_effects']['mean'] is not None
    assert b['meta']['heterogeneity']['I2_percent'] == 0.0, b['meta']
    assert b['meta']['prediction_interval'] is not None, b['meta']
    assert set(b['panels']) == {'p1', 'p2', 'p3'}, b['panels']
    assert b['leave_one_panel_out'] and b['sign_stable_leave_one_out'] is True, b
    assert b['meta']['corpus_binding']['corpus_binding'] == 'shared', b['meta']
    assert syn['poolable_components'] == ['A'], syn


def test_cross_panel_synthesis_refuses_heterogeneous_panels():
    reports = _consistent_panels()
    reports['p3'] = _rep([_entry([-8.0, -9.0, -7.0], [20, 21, 22])],
                         [20, 21, 22], 'e3')
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    assert b['status'] == 'heterogeneous', b
    assert b['claim_allowed'] is False, b
    assert b['meta']['heterogeneity']['I2_percent'] > 50.0, b['meta']
    assert syn['heterogeneous_components'] == ['A'], syn


def test_cross_panel_synthesis_refuses_mixed_direction_and_corpus():
    reports = _consistent_panels()
    reports['p2'] = _rep([_entry([-2.2, -1.8, -2.0], [10, 11, 12], pred=1)],
                         [10, 11, 12], 'e2')
    b = _block(L.cross_panel_synthesis(reports, _DESIGN))
    assert b['status'] == 'refused_direction_mismatch', b
    assert b['direction']['consistent'] is False, b
    assert not b.get('meta_input'), b

    reports = _consistent_panels()
    reports['p2'] = _rep([_entry([-2.2, -1.8, -2.0], [10, 11, 12])],
                         [10, 11, 12], 'e2', corpus='c2')
    b = _block(L.cross_panel_synthesis(reports, _DESIGN))
    assert b['meta']['corpus_binding']['corpus_binding'] == 'mixed', b
    assert b['claim_allowed'] is False, b
    assert b['status'] == 'not_estimable', b


def test_cross_panel_synthesis_does_not_select_on_significance():
    # Three panels clear the (synthetic) verified label with a strong effect;
    # one unverified panel measured no movement.  The all-panel estimate must
    # include the unverified panel, and the verified-only readout must be
    # reported separately as a selection sensitivity -- never as the headline.
    reports = {
        'p1': _rep([_entry([-5.0, -5.1, -4.9], [0, 1, 2])], [0, 1, 2], 'e1'),
        'p2': _rep([_entry([-5.0, -5.1, -4.9], [10, 11, 12])],
                   [10, 11, 12], 'e2'),
        'p3': _rep([_entry([-5.0, -5.1, -4.9], [20, 21, 22])],
                   [20, 21, 22], 'e3'),
        'p4': _rep([_entry([0.0, 0.1, -0.1], [30, 31, 32])],
                   [30, 31, 32], 'e4', verified=False),
    }
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    assert set(b['panels']) == {'p1', 'p2', 'p3', 'p4'}, b['panels']
    assert [p['label'] for p in b['meta_input']['points']] == ['p1', 'p2', 'p3', 'p4']
    sel = b['verified_only_sensitivity']
    assert sel['n_all'] == 4 and sel['n_verified'] == 3, sel
    assert sel['selection_on_significance'] is True, sel
    assert sel['verified_only_more_favourable'] is True, sel
    assert sel['benefit_verified_only'] > sel['benefit_all_panels'], sel
    assert syn['selection_sensitive_components'] == ['A'], syn


def test_cross_panel_synthesis_handles_dependent_draws():
    # Same training seeds: a paired common effect is available, but it is not
    # an independent replication.
    same = {}
    for name, fp in (('p1', 'e1'), ('p2', 'e2'), ('p3', 'e3')):
        same[name] = _rep([_entry([-2.0, -2.1, -1.9], [0, 1, 2])],
                          [0, 1, 2], fp)
    b = _block(L.cross_panel_synthesis(same, _DESIGN))
    assert b['meta']['seed_dependence'] == 'shared_seeds_paired', b['meta']
    assert b['evidence_tier'] == 'paired_same_seeds', b
    assert b['claim_allowed'] is False, b
    assert b['status'] == 'dependent_draws', b
    assert b['meta']['paired_common_effect']['status'] == 'ok', b['meta']

    # Disjoint seeds but one shared evaluation slice: new-seed replication,
    # not an independent evaluation.
    shared_eval = {}
    for i, seeds in enumerate(([0, 1, 2], [10, 11, 12], [20, 21, 22])):
        shared_eval[f'p{i + 1}'] = _rep(
            [_entry([-2.0, -2.1, -1.9], seeds)], seeds, 'same-eval')
    b = _block(L.cross_panel_synthesis(shared_eval, _DESIGN))
    assert b['evidence_tier'] == 'new_seeds_shared_evaluation', b
    assert b['claim_allowed'] is False, b

    # No draw identity at all must not be treated as independent.
    legacy = _consistent_panels()
    for rep in legacy.values():
        rep.pop('component_draws')
    b = _block(L.cross_panel_synthesis(legacy, _DESIGN))
    assert b['evidence_tier'] == 'unattested_evaluation', b
    assert b['claim_allowed'] is False, b


def test_cross_panel_synthesis_excludes_with_named_reasons():
    reports = _consistent_panels()
    # Too few seeds for the declared floor.
    reports['p4'] = _rep([_entry([-2.0, -2.1], [30, 31])], [30, 31], 'e4')
    # Missing seed identity.
    bad = _entry([-2.0, -2.1, -1.9], [0, 1, 2])
    bad.pop('seeds')
    reports['p5'] = _rep([bad], [40, 41, 42], 'e5')
    # Stale aggregate: the recorded percent does not match the deltas.
    stale = _entry([-2.0, -2.1, -1.9], [50, 51, 52])
    stale['relative_effect_pct'] = -9.0
    reports['p6'] = _rep([stale], [50, 51, 52], 'e6')
    # Duplicate entry for the same (with, without) pair inside one panel.
    dup = _entry([-2.0, -2.1, -1.9], [60, 61, 62])
    reports['p7'] = _rep([dup, dict(dup)], [60, 61, 62], 'e7')
    b = _block(L.cross_panel_synthesis(reports, _DESIGN))
    assert set(b['panels']) == {'p1', 'p2', 'p3'}, b['panels']
    assert 'n=2 < min_n=3' in b['excluded']['p4'], b['excluded']
    assert 'seed' in b['excluded']['p5'], b['excluded']
    assert 'inconsistent' in b['excluded']['p6'], b['excluded']
    assert 'duplicate' in b['excluded']['p7'], b['excluded']


def test_cross_panel_synthesis_degenerate_inputs_are_honest():
    one = {'p1': _rep([_entry([-2.0, -2.1, -1.9], [0, 1, 2])],
                      [0, 1, 2], 'e1')}
    b = _block(L.cross_panel_synthesis(one, _DESIGN))
    assert b['status'] == 'insufficient_panels', b
    assert b['claim_allowed'] is False, b

    empty = L.cross_panel_synthesis({}, _DESIGN)
    assert empty['status'] == 'not_verifiable_locally', empty
    assert 'no panel reports' in empty['reason'], empty

    unmeasured = {'p1': {'components': {}, 'min_n': 3, 'alpha': 0.05}}
    out = L.cross_panel_synthesis(unmeasured, _DESIGN)
    assert out['status'] == 'not_verifiable_locally', out
    assert out['components']['A']['status'] == 'not_measured', out


def test_cross_panel_synthesis_json_safe_and_deterministic():
    reports = _consistent_panels()
    a = L.cross_panel_synthesis(reports, _DESIGN)
    b = L.cross_panel_synthesis(dict(reports), _DESIGN)
    assert json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(
        b, allow_nan=False, sort_keys=True)
    # No numpy scalar/array leaks into a strict JSON writer.
    json.dumps(a, allow_nan=False)


def test_cross_panel_synthesis_is_embedded_in_ablation_coverage():
    reports = _consistent_panels()
    cov = L.ablation_coverage_across_panels(reports, _DESIGN)
    assert 'synthesis' in cov, cov
    assert cov['synthesis']['kind'] == 'cross_panel_synthesis', cov['synthesis']
    assert cov['synthesis']['components']['A']['contrasts'], cov['synthesis']
    # The label-counting verdict is unchanged by the synthesis block.
    assert cov['verified_anywhere'] == ['A'], cov
    assert cov['status'] == ('warn' if cov.get('warnings') else 'ok'), cov


def _write_report(td, report):
    p = os.path.join(td, 'audit_report.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(report, f, allow_nan=False)
    return p


def test_panel_synthesis_audit_replays_and_fails_closed():
    td = tempfile.mkdtemp(prefix='synthesis_')
    try:
        cov = L.ablation_coverage_across_panels(_consistent_panels(), _DESIGN)
        doc = {'methodology': {'ablation_coverage': cov}}
        p = _write_report(td, doc)
        rep = A.audit_panel_synthesis(p)
        assert rep['status'] == 'ok', rep
        assert rep['n_ok'] == 1 and rep['n_fail'] == 0, rep

        # A changed pooled mean must fail even though the input points agree.
        import copy
        bad = copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'meta']['random_effects']['mean'] = -3.0
        rep2 = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep2['status'] == 'fail', rep2
        assert any('random_effects' in x for x in rep2['problems']), rep2

        # A forged claim flag must fail.
        bad = copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'claim_allowed'] = False
        rep3 = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep3['status'] == 'fail', rep3
        assert any('claim_allowed' in x for x in rep3['problems']), rep3

        # A point input that no longer matches the rendered panel row fails.
        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['meta_input']['points'][0]['effects'][0] = -9.0
        rep4 = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep4['status'] == 'fail', rep4
        assert any('effects' in x for x in rep4['problems']), rep4

        # Tampered selection-bias readout fails.
        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['verified_only_sensitivity']['benefit_all_panels'] = -99.0
        rep5 = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep5['status'] == 'fail', rep5

        # Tampered panel dependence fails: the tier is recomputed from the
        # recorded draw identities instead of trusted.
        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['independence']['status'] = 'new_seeds_only'
        b['evidence_tier'] = 'new_seeds_shared_evaluation'
        b['claim_allowed'] = False
        rep5b = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep5b['status'] == 'fail', rep5b
        assert any('independence' in x for x in rep5b['problems']), rep5b

        # Tampered leave-one-panel-out robustness readout fails.
        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['leave_one_panel_out']['p3']['pooled_mean'] = 0.0
        b['leave_one_panel_out']['p3']['same_sign_as_full'] = False
        rep5c = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep5c['status'] == 'fail', rep5c
        assert any('leave_one_panel_out' in x for x in rep5c['problems']), rep5c

        # A report without a synthesis block is unverifiable, not a pass.
        rep6 = A.audit_panel_synthesis(_write_report(td, {'methodology': {}}))
        assert rep6['status'] == 'not_verifiable_locally', rep6

        # A bare synthesis report (no audit_report wrapper) is accepted.
        bare = _write_report(td, cov['synthesis'])
        rep7 = A.audit_panel_synthesis(bare)
        assert rep7['status'] == 'ok', rep7
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_panel_synthesis_audit_fails_closed_on_malformed_nested_fields():
    """A hand-edited report must be a named failure, never a TypeError.

    The audit reads several nested JSON objects/lists from an artifact.  A
    string where a direction/corpus block belongs, a scalar category list or a
    scalar per-seed effect vector is malformed evidence; the gate has to say so
    (``fail``) instead of crashing before it can report anything.
    """
    import copy
    td = tempfile.mkdtemp(prefix='synthesis_malformed_')
    try:
        cov = L.ablation_coverage_across_panels(_consistent_panels(), _DESIGN)
        doc = {'methodology': {'ablation_coverage': cov}}
        assert A.audit_panel_synthesis(_write_report(td, doc))['status'] == 'ok'

        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['direction'] = 'not-an-object'
        rep = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('direction' in x for x in rep['problems']), rep

        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['corpus'] = 1
        rep = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('corpus' in x for x in rep['problems']), rep

        bad = copy.deepcopy(doc)
        synth = bad['methodology']['ablation_coverage']['synthesis']
        synth['poolable_components'] = 1
        synth['heterogeneous_components'] = 'A'
        rep = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('poolable_components' in x for x in rep['problems']), rep
        assert any('heterogeneous_components' in x for x in rep['problems']), rep

        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['meta_input']['points'][0]['effects'] = 1
        rep = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('effects' in x for x in rep['problems']), rep
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_panel_synthesis_audit_is_reachable_from_the_cli_alias():
    assert A.audit_panel_synthesis is not None
    # The CLI dispatch table exposes the new gate under the documented aliases.
    src = open(_os.path.join(REPO, '_audit_gates', '_parts',
                             'part_05_audit_confirmation_verdicts.py'),
               encoding='utf-8').read()
    for alias in ("'panel_synthesis'", "'synthesis'", "'cross_panel'",
                  "'poolability'", "'evidence_synthesis'"):
        assert alias in src, alias
