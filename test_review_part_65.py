"""CPU review checks (split group 65): cross-panel effect-scale sensitivity.

``cross_panel_synthesis`` pools the relative percent effect but the shared
``meta_analyze_effects`` core documents log PPL ratios.  Percent and log are
different estimands (PPL is an exponential of mean NLL), so a pooled claim
can depend on that scale choice.  These checks cover the new readout: both
scales are recomputed from the same per-seed pairs, the Jensen gap is
reported, a scale-discordant family loses the pooled claim, and
``audit_gates.py panel_scale_sensitivity`` replays the arithmetic and the
gate from the recorded inputs.
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


def _rep(entries, seeds, eval_fp, *, corpus='c1', verified=True, comp='A',
         min_n=3, alpha=0.05):
    return {'components': {comp: entries},
            'verified_components': [comp] if verified else [],
            'direction_conflict_components': [],
            'practically_harmful_components': [],
            'ablation_completeness': {
                'components': {comp: {}},
                'complete_components': [comp],
                'incomplete_components': [],
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


def _discordant_panels():
    """A panel family where the percent interval excludes zero, the log one
    does not.  Found by exact enumeration of the two meta-analyses; both
    scales are poolable, so only the scale choice decides 'significant'."""
    effects = [
        [-6.512024639824981, -5.976485450569079, -1.7551114335401348],
        [-10.510141304396658, -1.9709599582890518, -8.806716896276473],
        [-7.979053170705706, 1.9132761184978317, -11.56040220154138],
    ]
    seeds = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    return {f'p{i + 1}': _rep([_entry(effects[i], seeds[i])], seeds[i],
                              f'e{i + 1}')
            for i in range(3)}


def _block(syn, comp='A', name='full - full-A'):
    return syn['components'][comp]['contrasts'][name]


def _write_report(td, report):
    p = os.path.join(td, 'audit_report.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(report, f, allow_nan=False)
    return p


def test_scale_sensitivity_reports_both_estimands_and_concordance():
    syn = L.cross_panel_synthesis(_consistent_panels(), _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    assert b['status'] == 'poolable' and b['claim_allowed'] is True, b
    assert sb['status'] == 'concordant', sb
    assert sb['percent']['poolable'] is True and sb['log_ratio']['poolable'] is True, sb
    assert abs(sb['percent']['mean'] - (-2.0)) < 1e-12, sb['percent']
    assert abs(sb['log_ratio']['mean'] - (-0.02020317008790245)) < 1e-12, sb['log_ratio']
    back = sb['log_ratio']['back_transformed_relative_pct']
    assert abs(back - (-2.0000453514870404)) < 1e-12, back
    # The test entries carry no per-seed arm PPLs, so the panel-mean baseline
    # is the only exact log transform; the record must say so.
    assert sb['log_ratio_basis'] == 'panel_mean_without', sb
    assert sb['per_seed_ppl_panels'] == [], sb
    assert abs(sb['jensen_gap_pct']) < 1e-4, sb['jensen_gap_pct']
    assert sb['direction_agree'] is True and sb['zero_exclusion_agree'] is True, sb
    assert sb['claim_allowed_on_both_scales'] is True, sb
    assert syn['scale_sensitivity']['status'] == 'ok', syn['scale_sensitivity']
    assert syn['scale_sensitivity']['concordant_components'] == ['A']
    assert syn['scale_sensitive_components'] == [], syn


def test_scale_sensitivity_reports_jensen_gap_without_overclaiming():
    # Heterogeneous panel means: the arithmetic mean of percent effects and
    # the percent implied by the mean log effect are different numbers.  Both
    # scales are reported; the common-effect claim is already closed by the
    # heterogeneity gate, and the scale readout must not reopen it.
    effects = [[-10, -10.2, -9.8], [-50, -52, -48], [-50, -55, -45]]
    seeds = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    reports = {f'p{i + 1}': _rep([_entry(effects[i], seeds[i])], seeds[i],
                                 f'e{i + 1}') for i in range(3)}
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    assert b['status'] == 'heterogeneous', b
    assert b['claim_allowed'] is False, b
    assert sb['percent']['poolable'] is False, sb['percent']
    assert sb['log_ratio']['poolable'] is False, sb['log_ratio']
    assert sb['jensen_gap_pct'] is not None, sb
    assert sb['jensen_gap_pct'] > 2.0, sb['jensen_gap_pct']
    assert sb['percent']['mean'] != sb['log_ratio']['back_transformed_relative_pct'], sb
    assert sb['claim_allowed_on_both_scales'] is False, sb


def test_scale_sensitivity_withholds_claim_when_intervals_disagree():
    syn = L.cross_panel_synthesis(_discordant_panels(), _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    # The percent-scale pooled interval excludes zero ...
    assert sb['percent']['poolable'] is True, sb['percent']
    assert sb['percent_excludes_zero'] is True, sb
    # ... the log-ratio interval does not.  The common effect is then a
    # scale-choice artifact, so the producer must withdraw the claim.
    assert sb['log_ratio']['poolable'] is True, sb['log_ratio']
    assert sb['log_excludes_zero'] is False, sb
    assert sb['zero_exclusion_agree'] is False, sb
    assert sb['status'] == 'precision_discordant', sb
    assert sb['claim_allowed_on_both_scales'] is False, sb
    assert b['claim_allowed'] is False, b
    assert b['status'] == 'scale_sensitive', b['status']
    assert any('effect scale' in r for r in b['reasons']), b['reasons']
    assert syn['scale_sensitive_components'] == ['A'], syn
    assert syn['scale_sensitivity']['status'] == 'warn', syn['scale_sensitivity']
    assert syn['scale_sensitivity']['scale_sensitive_components'] == ['A']


def test_scale_sensitivity_withholds_claim_when_poolability_differs():
    # Percent I2=48.8 (poolable), log I2=56.4 (not poolable): the effect
    # survives the heterogeneity gate on one scale only.  That is also a
    # scale-dependent admissibility, so the pooled claim must be withdrawn.
    effects = [
        [-70.9164757652584, -73.17965939093023, -76.24509994315153],
        [-58.774502922809916, -71.05807066192286, -72.03528355506486],
        [-73.98788169553279, -74.93812014802221, -76.03392487552189],
    ]
    seeds = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    reports = {f'p{i + 1}': _rep([_entry(effects[i], seeds[i])], seeds[i],
                                 f'e{i + 1}') for i in range(3)}
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    assert sb['percent']['poolable'] is True, sb['percent']
    assert sb['log_ratio']['poolable'] is False, sb['log_ratio']
    assert sb['status'] == 'poolability_discordant', sb
    assert sb['claim_allowed_on_both_scales'] is False, sb
    assert b['claim_allowed'] is False, b
    assert b['status'] == 'scale_sensitive', b['status']
    assert syn['scale_sensitive_components'] == ['A'], syn


def test_scale_sensitivity_uses_true_paired_log_ratio_when_available():
    # With per-seed arm PPLs the true log(with_i) - log(without_i) is
    # available.  It is NOT log1p(percent/100) here because the without
    # baselines vary across seeds; the record must use the paired basis and
    # its log input points must match the true ratios.
    with_vals = [[98.0, 99.5, 96.0], [101.0, 103.5, 99.0],
                 [95.0, 97.5, 93.0]]
    without_vals = [[100.0, 101.0, 99.0], [103.0, 104.0, 102.0],
                    [97.0, 98.0, 96.0]]
    seeds = [[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    reports = {}
    for i in range(3):
        deltas = [a - b for a, b in zip(with_vals[i], without_vals[i])]
        without_mean = sum(without_vals[i]) / len(without_vals[i])
        e = _entry(deltas, seeds[i], without=without_mean)
        e['per_seed_with'] = list(with_vals[i])
        e['per_seed_without'] = list(without_vals[i])
        reports[f'p{i + 1}'] = _rep([e], seeds[i], f'e{i + 1}')
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    assert b['claim_allowed'] is True, b
    assert sb['log_ratio_basis'] == 'paired_per_seed', sb
    assert sb['per_seed_ppl_panels'] == ['p1', 'p2', 'p3'], sb
    true_log = [math.log(a) - math.log(bb)
                for a, bb in zip(with_vals[0], without_vals[0])]
    got = [pt['effects'] for pt in sb['log_meta_input']['points']
           if pt['label'] == 'p1'][0]
    assert all(abs(x - y) < 1e-12 for x, y in zip(got, true_log)), (got, true_log)
    fallback = [math.log1p(e / 100.0) for e in b['panels']['p1']['effects']]
    assert any(abs(x - y) > 1e-8 for x, y in zip(got, fallback)), (got, fallback)
    # The audit replays the same basis from the retained per-seed PPLs.
    td = tempfile.mkdtemp(prefix='paired_basis_')
    try:
        doc = {'methodology': {'ablation_coverage': {'synthesis': syn}}}
        p = _write_report(td, doc)
        assert A.audit_panel_scale_sensitivity(p)['status'] == 'ok'
        # Editing only the retained per-seed PPL must fail even though the
        # stored log_effects vector still replays against itself.
        import copy as _copy
        bad = _copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'panels']['p1']['per_seed_with'][0] += 1.0
        rep = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('with-without' in x or 'log(with)' in x
                   for x in rep['problems']), rep
    finally:
        shutil.rmtree(td, ignore_errors=True)
    # One panel without the per-seed vectors makes the whole family fall back
    # to the mean-baseline transform, so the pool never mixes two estimands.
    mixed = dict(reports)
    mixed['p3'] = _rep([_entry(b['panels']['p3']['effects'],
                               seeds[2])], seeds[2], 'e3')
    sb2 = _block(L.cross_panel_synthesis(mixed, _DESIGN))['scale_sensitivity']
    assert sb2['log_ratio_basis'] == 'panel_mean_without', sb2
    assert sb2['per_seed_ppl_panels'] == [], sb2
    # Supplying only one of the two vectors is malformed, not a silent
    # fallback: the panel is excluded with a named reason.
    partial = _entry([-2.0, -2.1, -1.9], [0, 1, 2])
    partial['per_seed_with'] = [98.0, 99.0, 97.0]
    excluded = _block(L.cross_panel_synthesis(
        {'p1': _rep([partial], [0, 1, 2], 'e1')}, _DESIGN))
    assert 'supplied together' in excluded['excluded']['p1'], excluded


def test_scale_sensitivity_rejects_implied_nonpositive_ppl():
    # A -100% effect implies a zero with-arm PPL; log(with/without) is not
    # defined.  The gate refuses to fabricate a log ratio and says why.
    reports = _consistent_panels()
    reports['p4'] = _rep([_entry([-100.0, -99.0, -98.0], [30, 31, 32])],
                         [30, 31, 32], 'e4')
    syn = L.cross_panel_synthesis(reports, _DESIGN)
    b = _block(syn)
    sb = b['scale_sensitivity']
    assert sb['status'] == 'not_estimable', sb
    assert sb['log_ratio']['mean'] is None, sb['log_ratio']
    assert sb['claim_allowed_on_both_scales'] is False, sb
    assert any('non-positive with-arm PPL' in r for r in sb['reasons']), sb
    assert 'A' in syn['scale_sensitivity']['not_estimable_components'], syn


def test_scale_sensitivity_does_not_change_percent_pool_or_panel_verdicts():
    reports = _consistent_panels()
    cov = L.ablation_coverage_across_panels(reports, _DESIGN)
    syn = cov['synthesis']
    b = _block(syn)
    # The historical percent estimand is still the same vector.
    points = b['meta_input']['points']
    direct = L.meta_analyze_effects(points, alpha=0.05, min_seeds=3)
    assert b['meta']['random_effects'] == direct['random_effects'], b['meta']
    # The label-counting verdict is untouched by the scale block.
    assert cov['verified_anywhere'] == ['A'], cov
    assert cov['status'] == ('warn' if cov.get('warnings') else 'ok'), cov
    # And the gate is what closes the discordant claim: disable the helper
    # and the same input keeps the historical (percent-only) claim.
    saved = L.cross_panel_scale_contrast
    try:
        L.cross_panel_scale_contrast = None
        raw = L.cross_panel_synthesis(_discordant_panels(), _DESIGN)
        raw_block = _block(raw)
        assert raw_block['claim_allowed'] is True, raw_block
        assert raw_block['status'] == 'poolable', raw_block
        assert 'scale_sensitivity' not in raw_block, raw_block
    finally:
        L.cross_panel_scale_contrast = saved


def test_scale_sensitivity_is_deterministic_and_json_safe():
    a = L.cross_panel_synthesis(_discordant_panels(), _DESIGN)
    b = L.cross_panel_synthesis(_discordant_panels(), _DESIGN)
    assert json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(
        b, allow_nan=False, sort_keys=True)
    json.dumps(a, allow_nan=False)
    assert L.cross_panel_scale_sensitivity(a)['status'] == 'warn'


def test_scale_sensitivity_audit_replays_and_fails_closed():
    td = tempfile.mkdtemp(prefix='panel_scale_')
    try:
        cov = L.ablation_coverage_across_panels(_consistent_panels(), _DESIGN)
        doc = {'methodology': {'ablation_coverage': cov}}
        p = _write_report(td, doc)
        rep = A.audit_panel_scale_sensitivity(p)
        assert rep['status'] == 'ok', rep
        assert rep['n_checked'] == 1 and rep['n_ok'] == 1, rep

        # A changed pooled log mean must fail.
        import copy
        bad = copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'scale_sensitivity']['log_ratio']['mean'] = -0.5
        rep2 = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep2['status'] == 'fail', rep2
        assert any('log_ratio' in x for x in rep2['problems']), rep2

        # A changed status / agreement flag must fail.
        bad = copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'scale_sensitivity']['zero_exclusion_agree'] = False
        rep3 = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep3['status'] == 'fail', rep3

        # A log input point that is not log1p(percent/100) must fail.
        bad = copy.deepcopy(doc)
        sb = _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'scale_sensitivity']
        sb['log_meta_input']['points'][0]['effects'][0] = -9.0
        rep4 = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep4['status'] == 'fail', rep4
        assert any('log_meta_input' in x for x in rep4['problems']), rep4

        # A deleted block is unverifiable, not a pass.
        bad = copy.deepcopy(doc)
        _block(bad['methodology']['ablation_coverage']['synthesis']).pop(
            'scale_sensitivity')
        rep5 = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep5['status'] == 'not_verifiable_locally', rep5
        assert rep5['n_missing'] == 1, rep5

        # A bare synthesis object (no audit_report wrapper) is accepted.
        bare = _write_report(td, cov['synthesis'])
        assert A.audit_panel_scale_sensitivity(bare)['status'] == 'ok'
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_scale_sensitivity_audit_rejects_an_unapplied_gate():
    td = tempfile.mkdtemp(prefix='panel_scale_gate_')
    try:
        cov = L.ablation_coverage_across_panels(_discordant_panels(), _DESIGN)
        doc = {'methodology': {'ablation_coverage': cov}}
        p = _write_report(td, doc)
        # The honest discordant report passes both gates.
        assert A.audit_panel_scale_sensitivity(p)['status'] == 'ok'
        assert A.audit_panel_synthesis(p)['status'] == 'ok'

        import copy
        bad = copy.deepcopy(doc)
        b = _block(bad['methodology']['ablation_coverage']['synthesis'])
        b['claim_allowed'] = True
        b['status'] = 'poolable'
        rep = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep['status'] == 'fail', rep
        assert any('claim_allowed' in x for x in rep['problems']), rep
        assert any('contrast status' in x for x in rep['problems']), rep

        # The synthesis audit must apply the same gate on replay.
        rep2 = A.audit_panel_synthesis(_write_report(td, bad))
        assert rep2['status'] == 'fail', rep2

        # A confirmatory-looking key smuggled into the sensitivity block is
        # a fail even if every derived number still matches.
        bad = copy.deepcopy(doc)
        sb = _block(bad['methodology']['ablation_coverage']['synthesis'])[
            'scale_sensitivity']
        sb['confirmed'] = True
        rep3 = A.audit_panel_scale_sensitivity(_write_report(td, bad))
        assert rep3['status'] == 'fail', rep3
        assert any('confirmatory verdict' in x for x in rep3['problems']), rep3
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_scale_sensitivity_audit_fails_closed_on_malformed_meta_input():
    """A scalar/string ``meta_input`` is bad evidence, not an audit crash."""
    td = tempfile.mkdtemp(prefix='panel_scale_malformed_')
    try:
        cov = L.ablation_coverage_across_panels(_consistent_panels(), _DESIGN)
        doc = {'methodology': {'ablation_coverage': cov}}
        assert A.audit_panel_scale_sensitivity(_write_report(td, doc))[
            'status'] == 'ok'
        import copy
        for malformed in ('not-an-object', 1.5, True, [0]):
            bad = copy.deepcopy(doc)
            _block(bad['methodology']['ablation_coverage']['synthesis'])[
                'meta_input'] = malformed
            rep = A.audit_panel_scale_sensitivity(_write_report(td, bad))
            assert rep['status'] == 'fail', (malformed, rep)
            assert any('meta_input' in x for x in rep['problems']), (malformed, rep)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_scale_sensitivity_cli_aliases_are_reachable():
    assert A.audit_panel_scale_sensitivity is not None
    src = open(_os.path.join(REPO, '_audit_gates', '_parts',
                             'part_05_audit_confirmation_verdicts.py'),
               encoding='utf-8').read()
    for alias in ("'panel_scale_sensitivity'", "'scale_sensitivity'",
                  "'panel_scale'", "'synthesis_scale'",
                  "'cross_panel_scale'", "'scale_robustness'"):
        assert alias in src, alias
