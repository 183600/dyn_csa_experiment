"""CPU review checks (split group 11; t_sesoi_audit_gates_verified_and_equivalence .. t_paired_design_sensitivity_planning)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_sesoi_audit_gates_verified_and_equivalence():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    design = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed', 'x')]}

    def _summary(effect, noise=0.0, n=8):
        s = {}
        for i in range(n):
            s[f'csa_dynamic::seed{i}'] = dict(
                variant='csa_dynamic', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0 + effect + (noise if i % 2 else -noise),
                params=100, train_knobs=knobs)
            s[f'csa_fixed::seed{i}'] = dict(
                variant='csa_fixed', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0, params=100, train_knobs=knobs)
        return s

    # 2% improvement: above the 1% SESOI -> useful and verified.
    r = L.methodology_audit(_summary(-1.0, 0.001), design=design,
                            baselines={}, verbose=False)
    e = r['components']['dynamic_blocking'][0]
    assert e['practical_decision'] == 'superior', e
    assert 'dynamic_blocking' in r['verified_components'], r['verified_components']
    assert 'dynamic_blocking' in r['practically_useful_components'], r
    assert r['sesoi']['components']['dynamic_blocking']['status'] == 'useful', r['sesoi']
    # 0.2% improvement: statistically detectable but below the SESOI.  It must
    # not be counted as verified or practically useful.
    r2 = L.methodology_audit(_summary(-0.1, 0.001), design=design,
                             baselines={}, verbose=False)
    e2 = r2['components']['dynamic_blocking'][0]
    assert e2['practical_decision'] == 'sub_sesoi', e2
    assert 'dynamic_blocking' not in r2['verified_components'], r2['verified_components']
    assert 'dynamic_blocking' in r2['sub_sesoi_components'], r2['sub_sesoi_components']
    # A precise zero effect: TOST inside the 1% margin -> removable, NOT a
    # failed significance test and NOT verified.
    r3 = L.methodology_audit(_summary(0.0, 0.005), design=design,
                             baselines={}, verbose=False)
    e3 = r3['components']['dynamic_blocking'][0]
    assert e3['practical_decision'] == 'equivalent', e3
    assert 'dynamic_blocking' in r3['equivalent_components'], r3['equivalent_components']
    assert 'dynamic_blocking' not in r3['verified_components'], r3['verified_components']
    assert r3['sesoi']['components']['dynamic_blocking']['status'] == 'equivalent', r3['sesoi']
    # The equivalence p is family-adjusted and reported.
    assert math.isfinite(float(e3['sesoi']['tost_p_holm'])), e3['sesoi']
    # 1.6% in the wrong direction: harmful, above the SESOI.
    r4 = L.methodology_audit(_summary(0.8, 0.001), design=design,
                             baselines={}, verbose=False)
    assert r4['sesoi']['components']['dynamic_blocking']['status'] == 'harmful', r4['sesoi']
    # Passing an empty SESOI mapping restores the statistical-only behaviour:
    # the 0.2% effect is then verified (legacy), proving the gate is explicit.
    r5 = L.methodology_audit(_summary(-0.1, 0.001), design=design,
                             baselines={}, sesoi={}, verbose=False)
    assert 'dynamic_blocking' in r5['verified_components'], r5['verified_components']
    assert r5['sesoi']['declared_components'] == [], r5['sesoi']
    # An unmeasured declared contrast keeps its SESOI family slot (dummy p=1);
    # a partial ablation cannot shrink the equivalence correction burden.
    partial_design = {'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'x'),
        ('csa_dynamic', 'csa_fixed_missing', 'y')]}
    r6 = L.methodology_audit(_summary(0.0, 0.005), design=partial_design,
                             baselines={}, verbose=False)
    assert r6['sesoi']['family']['m'] == 2, r6['sesoi']['family']
    # Equivalence on a partial ablation cannot certify "the component is
    # removable" (the second declared contrast is missing).
    assert 'dynamic_blocking' not in r6['equivalent_components'], r6['equivalent_components']
    assert r6['sesoi']['components']['dynamic_blocking']['status'] != 'equivalent', r6['sesoi']
    # Practical superiority follows the FAMILY-corrected significance, not the
    # raw sign-flip p.  Here n=6 gives raw p=0.031 but 5 declared contrasts
    # leave Holm p=0.156, so the contrast must not be called useful.
    wide_design = {'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'x')] + [
        ('csa_dynamic', f'csa_fixed_missing{i}', 'y') for i in range(1, 5)]}
    r7 = L.methodology_audit(_summary(-1.0, 0.001, n=6), design=wide_design,
                             baselines={}, verbose=False)
    e7 = r7['components']['dynamic_blocking'][0]
    assert e7['significant_global'] is False, e7
    assert e7['practical_decision'] != 'superior', e7['sesoi']
    assert 'dynamic_blocking' not in r7['verified_components'], r7['verified_components']


def test_sesoi_declaration_gate():
    def _audit(**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_sesoi_declaration(**kw)

    ok = _audit()
    assert ok['status'] == 'ok' and ok['ablation_complete'] is True, ok
    # Every declared ablation component must have a pre-registered margin.
    partial = {'dynamic_blocking': {'relative_pct': 1.0}}
    bad = _audit(spec=partial)
    assert bad['status'] == 'fail' and any('no pre-registered SESOI' in p
                                           for p in bad['problems']), bad
    # A malformed declaration is a design defect, not a silently missing margin.
    bad2 = _audit(spec={'dynamic_blocking': {'relative_pct': 0.0}})
    assert bad2['status'] == 'fail', bad2
    # Extra/misspelled component names are surfaced (a typo would otherwise
    # leave the real component without a margin).
    extra = dict(L.ABLATION_SESOI)
    extra['dynamic_blockin'] = {'relative_pct': 1.0}
    out = _audit(spec=extra)
    assert any('not present in the ablation design' in w for w in out['warnings']), out


def test_sesoi_partial_panel_and_input_guards():
    # Component-level equivalence needs EVERY declared contrast to be powered.
    # One n=8 equivalent contrast must not certify a component whose second
    # declared contrast is only n=2 (underpowered).
    eq = {'with': 'a', 'without': 'b', 'controlled': True, 'n': 8,
          'significant_global': False, 'within_seed_noise': False,
          'sesoi': {'margin_ppl': 0.5, 'declared': True},
          'practical_decision': 'equivalent'}
    weak = {'with': 'a', 'without': 'c', 'controlled': True, 'n': 2,
            'significant_global': False, 'within_seed_noise': False,
            'sesoi': {'margin_ppl': 0.5, 'declared': True},
            'practical_decision': 'underpowered'}
    assert L._classify_group_status([eq, weak], min_n=3, alpha=0.05,
                                    m_family=2, n_declared=2) != 'equivalent'
    # Two protocol entries for ONE pair cannot stand in for a missing second
    # declared contrast; distinct pairs, not entry count, define coverage.
    dup = dict(eq)
    assert L._classify_group_status([eq, dup], min_n=3, alpha=0.05,
                                    m_family=2, n_declared=2) != 'equivalent'
    # The explicit declaration path is strict even if one pair has two entries.
    assert L._classify_group_status(
        [eq, dup], min_n=3, alpha=0.05, m_family=2, n_declared=2,
        declared_pairs=[('a', 'b'), ('a', 'c')]) != 'equivalent'
    # No measured contrast still leaves the declared family slots visible.
    fam = L._apply_sesoi_holm(
        {'c': []}, {'c': [('a', 'b', 'x')]}, {}, {},
        {'c': {'relative_pct': 1.0}}, alpha=0.05, min_n=3)
    assert fam['m'] == 1 and fam['declared_contrasts'] == 1, fam
    # At scale a contrast can carry several protocol tags.  Those extra
    # measured entries must not fill the p=1 slot of a DIFFERENT, missing
    # declared contrast: m = 2 measured tags + 1 missing declaration = 3.
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_'
          'cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_'
          'detcpu-eager_dfabc_csX')
    knobs = L.training_knobs()
    multi = {}
    for i in range(3):
        for tag in ('p1', 'p2'):
            multi[f'csa_dynamic::{tag}::seed{i}'] = dict(
                variant='csa_dynamic', seed=i, protocol=tag, run_cfg=fp,
                steps=1, tokens_seen=1, ppl=50.0, params=100,
                train_knobs=knobs)
            multi[f'csa_fixed::{tag}::seed{i}'] = dict(
                variant='csa_fixed', seed=i, protocol=tag, run_cfg=fp,
                steps=1, tokens_seen=1, ppl=50.0, params=100,
                train_knobs=knobs)
    design = {'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'x'),
        ('csa_dynamic', 'csa_fixed_missing', 'y')]}
    mr = L.methodology_audit(multi, design=design, baselines={}, verbose=False)
    assert mr['family']['effective_m'] == 3, mr['family']
    assert mr['sesoi']['family']['m'] == 3, mr['sesoi']['family']
    assert 'dynamic_blocking' not in mr['equivalent_components'], mr
    # A one-pass iterable must feed both the superiority and TOST tests.
    r = L.ablation_practical_decision(
        iter([0.01, -0.01, 0.0, 0.01, -0.01, 0.0]), margin=0.5)
    assert r['decision'] == 'equivalent' and r['n'] == 6, r
    # Direction is a sign, not a number to truncate.
    for bad in (1.5, -0.5, float('nan'), True):
        try:
            L.ablation_practical_decision([0.1, 0.2, 0.3], margin=0.1,
                                          direction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'fractional/bool direction accepted: {bad!r}')
    # An empty SESOI mapping means "declaration disabled", not a malformed
    # single-component spec.
    off = L.ablation_practical_decision([0.01, -0.01, 0.0], sesoi={})
    assert off['decision'] == 'not_declared' and off['margin'] is None, off
    # Truthy/non-boolean flags must not authorize a superiority or TOST claim.
    malformed = L._practical_decision_from_stats(
        {'n': 8, 'mean': 1.0, 'significant': 'false'},
        {'equivalent': 'false', 'p_tost': 0.0, 'ci': [0.0, 0.0]},
        margin=0.1, direction=-1, alpha=0.05, min_n=3)
    assert malformed['decision'] != 'superior', malformed
    assert malformed['decision'] != 'equivalent', malformed
    # Refreshing the family p must not drop the descriptive per-contrast fields.
    pd = L.ablation_practical_decision(
        [0.01, -0.01, 0.0, 0.01, -0.01, 0.0],
        sesoi={'relative_pct': 1.0}, baseline_ppl=50.0, label='x')
    entry = {'with': 'a', 'without': 'b', 'sesoi': pd,
             'significant_global': False, 'p_holm_global': 0.9}
    L._apply_sesoi_holm({'c': [entry]}, {'c': [('a', 'b', 'x')]}, {}, {},
                        {'c': {'relative_pct': 1.0}}, alpha=0.05, min_n=3)
    assert entry['sesoi']['label'] == 'x', entry['sesoi']
    assert entry['sesoi']['baseline_ppl'] == 50.0, entry['sesoi']
    assert abs(entry['sesoi']['relative_margin_pct'] - 1.0) < 1e-12, entry['sesoi']
    # A malformed design is a gate failure, never an exception from len().
    with contextlib.redirect_stdout(io.StringIO()):
        bad = A.audit_sesoi_declaration(
            spec={'dynamic_blocking': {'relative_pct': 1.0}},
            design={'dynamic_blocking': None})
    assert bad['status'] == 'fail', bad
    # The full methodology audit must also fail closed on a malformed design
    # instead of crashing on len()/unpacking.  Each shape below is a
    # declaration defect, not a reason to abort the whole panel audit.
    for _bad_design in ({'c': [('a', 'b')]}, {'c': None}, {'c': ['ab']},
                        {'c': [None]}, {'c': []}, {'c': 'not-a-list'},
                        ['not', 'a', 'mapping']):
        with contextlib.redirect_stdout(io.StringIO()):
            _rep = L.methodology_audit({}, design=_bad_design,
                                       baselines={}, verbose=False)
        assert _rep['warnings'], (_bad_design, _rep['warnings'])
    # A one-pass generator is walked by two audit passes (collection and
    # family scoping); it must be materialised once, not exhausted.
    _gen = {'c': (item for item in [('a', 'b', 'x')])}
    with contextlib.redirect_stdout(io.StringIO()):
        _gen_rep = L.methodology_audit({}, design=_gen, baselines={},
                                       verbose=False)
    assert 'c' in _gen_rep['components'], _gen_rep['components']
    # The prospective seed plan has the same contract (and counts a malformed
    # declaration as one slot rather than dropping it).
    assert L.methodology_seed_plan(6, design={'c': [('a', 'b')]},
                                   baselines={})['declared_contrasts'] == 1
    assert L.methodology_seed_plan(6, design={'c': None},
                                   baselines={})['declared_contrasts'] == 1
    assert L.methodology_seed_plan(6, design=['not', 'a', 'mapping'],
                                   baselines={})['declared_contrasts'] == 0


def test_baseline_sesoi_and_design_gate():
    # The baseline claim has its own pre-registered margin; a significant but
    # practically tiny advantage over a matched dense baseline is not enough.
    assert set(L.BASELINE_SESOI) == set(L.BASELINE_DESIGN), L.BASELINE_SESOI
    norm = L._normalise_sesoi_design(L.BASELINE_SESOI)
    assert norm['compression_vs_matched_dense']['relative_pct'] == 1.0, norm
    with contextlib.redirect_stdout(io.StringIO()):
        ok = A.audit_baseline_sesoi()
    assert ok['status'] == 'ok' and ok['all_arms_scheduled'] is True, ok
    with contextlib.redirect_stdout(io.StringIO()):
        missing = A.audit_baseline_sesoi(spec={})
    assert missing['status'] == 'fail', missing
    assert any('no pre-registered baseline SESOI' in p
               for p in missing['problems']), missing
    extra = dict(L.BASELINE_SESOI)
    extra['compression_vs_matched_dens'] = {'relative_pct': 1.0}
    with contextlib.redirect_stdout(io.StringIO()):
        typo = A.audit_baseline_sesoi(spec=extra)
    assert typo['status'] == 'ok', typo
    assert any('not present in BASELINE_DESIGN' in w
               for w in typo['warnings']), typo
    with contextlib.redirect_stdout(io.StringIO()):
        malformed = A.audit_baseline_sesoi(
            spec={'compression_vs_matched_dense': {'relative_pct': 0.0}})
    assert malformed['status'] == 'fail', malformed
    with contextlib.redirect_stdout(io.StringIO()):
        unscheduled = A.audit_baseline_sesoi(
            spec={'x': {'relative_pct': 1.0}},
            design={'x': [('a', 'not_scheduled', 'x')]})
    assert unscheduled['status'] == 'fail', unscheduled
    assert any('not scheduled' in p for p in unscheduled['problems']), unscheduled

    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _summary(effect, noise=0.001, n=8):
        s = {}
        for i in range(n):
            s[f'csa_dynamic::seed{i}'] = dict(
                variant='csa_dynamic', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0 + effect + (noise if i % 2 else -noise),
                params=1000, train_knobs=knobs)
            s[f'full_sw128_matched::seed{i}'] = dict(
                variant='full_sw128_matched', seed=i, run_cfg=fp, steps=1,
                tokens_seen=1, ppl=50.0, params=1001, train_knobs=knobs)
        return s

    # 0.2% below the matched dense baseline: statistically detectable, but
    # below the 1% baseline SESOI.  It must not become a verified baseline.
    small = L.methodology_audit(_summary(-0.1), verbose=False)
    es = small['baselines']['compression_vs_matched_dense'][0]
    assert es['sesoi']['declared'] is True and es['sesoi']['margin_ppl'] == 0.5, es
    assert es['practical_decision'] == 'sub_sesoi', es
    assert 'compression_vs_matched_dense' not in small['verified_baselines'], small
    assert 'compression_vs_matched_dense' in small['sub_sesoi_baselines'], small
    assert 'compression_vs_matched_dense' in small['sesoi']['sub_sesoi_components'], small
    # 2% below the same baseline: above SESOI, so the baseline claim passes.
    big = L.methodology_audit(_summary(-1.0), verbose=False)
    eb = big['baselines']['compression_vs_matched_dense'][0]
    assert eb['practical_decision'] == 'superior', eb
    assert 'compression_vs_matched_dense' in big['verified_baselines'], big
    # Explicitly disabling SESOI restores the historical statistical-only
    # verdict; the default is not silently applied when the caller opts out.
    legacy = L.methodology_audit(_summary(-0.1), sesoi={}, verbose=False)
    assert 'compression_vs_matched_dense' in legacy['verified_baselines'], legacy


def test_doc_unit_influence_primitives():
    """L173 — leave-one-unit-out influence targets the pooled estimand."""
    A = [[2.0, 3.0, 5.0, 7.0], [3.0, 4.0, 6.0, 8.0]]
    B = [[3.0, 3.0, 4.0, 6.0], [4.0, 4.0, 5.0, 7.0]]
    T = [10.0, 20.0, 30.0, 40.0]
    full = L._paired_nll_doc_ci(A, B, T)
    got = L.paired_nll_unit_influence(A, B, T,
                                     unit_ids=['u0', 'u1', 'u2', 'u3'])
    assert got['available'] is True and got['n_units'] == 4, got
    assert got['n_seeds'] == 2 and got['unit_ids'] == ['u0', 'u1', 'u2', 'u3'], got
    assert abs(got['full_delta_ppl'] - full['delta_ppl']) < 1e-12, got
    assert abs(got['full_delta_log_ratio'] - full['delta_log_ratio']) < 1e-12, got
    assert len(got['loo_delta_ppl']) == 4 and len(got['loo_change_ppl']) == 4, got
    Aa, Bb, Tt = np.asarray(A), np.asarray(B), np.asarray(T)
    for i in range(4):
        keep = [j for j in range(4) if j != i]
        one = L._paired_nll_doc_ci(Aa[:, keep], Bb[:, keep], Tt[keep])
        assert abs(got['loo_delta_ppl'][i] - one['delta_ppl']) < 1e-12, (i, got, one)
        assert abs(got['loo_delta_log_ratio'][i] - one['delta_log_ratio']) < 1e-12, (i, got)
    # The jackknife SE is the delete-one-unit variance of those exact folds.
    d = np.asarray(got['loo_delta_ppl'])
    want_se = math.sqrt(((len(d) - 1.0) / len(d)) * float(np.sum((d - d.mean()) ** 2)))
    assert abs(got['jackknife_se_ppl'] - want_se) < 1e-12, got
    # The exposed delta-method influence is exactly the term whose squares
    # give the existing cluster-robust SE; it is not a second, inconsistent
    # influence definition.
    _psi = np.asarray(got['delta_method_influence_ppl'])
    _se_from_psi = math.sqrt((len(_psi) / (len(_psi) - 1.0))
                             * float(np.sum(_psi ** 2)))
    assert abs(_se_from_psi - full['se_ppl']) < 1e-12, (got, full, _se_from_psi)
    lo, hi = got['jackknife_ci95_ppl']
    assert lo <= got['full_delta_ppl'] <= hi, got
    assert got['se_ratio_jackknife_to_cluster_ppl'] is not None, got
    # The most influential unit is the largest absolute LOO change on the
    # scale-free log effect, not necessarily the largest raw PPL difference.
    _lr = np.asarray(got['loo_delta_log_ratio'])
    assert got['most_influential_index'] == int(np.argmax(np.abs(_lr - got['full_delta_log_ratio']))), got
    assert got['top_units'][0]['unit_id'] == got['most_influential_unit'], got

    # One unit carries the whole effect: deleting it reverses the direction.
    A2 = [[1.0, 1.0, 1.0, 200.0]]
    B2 = [[2.0, 2.0, 2.0, 2.0]]
    T2 = [10.0, 10.0, 10.0, 10.0]
    out = L.paired_nll_unit_influence(A2, B2, T2,
                                      unit_ids=['a', 'b', 'c', 'd'])
    assert out['most_influential_unit'] == 'd', out
    assert out['fragile_to_one_unit'] is True, out
    assert out['single_unit_dominates'] is True, out
    assert out['sign_stable_log_ratio'] is False, out
    assert out['n_sign_flips_log_ratio'] >= 1, out
    assert any(u['unit_id'] == 'd' for u in out['top_units']), out

    # A uniform relative effect has zero LOO change: no single unit dominates.
    A3 = [[2.0, 2.0, 2.0, 2.0]]
    B3 = [[3.0, 3.0, 3.0, 3.0]]
    T3 = [10.0, 10.0, 10.0, 10.0]
    flat = L.paired_nll_unit_influence(A3, B3, T3)
    assert flat['max_abs_change_ppl'] == 0.0, flat
    assert flat['max_abs_change_log_ratio'] == 0.0, flat
    assert flat['fragile_to_one_unit'] is False, flat
    assert flat['sign_stable_log_ratio'] is True, flat
    assert flat['most_influential_tied_count'] == 4, flat
    assert flat['unit_ids'] == ['unit_0', 'unit_1', 'unit_2', 'unit_3'], flat
    assert flat['label_source'] == 'index_fallback', flat
    assert got['label_source'] == 'provided', got

    # An exactly null contrast has no direction to be fragile in: report
    # "undefined", never a fabricated stable/fragile boolean.
    null = L.paired_nll_unit_influence(A3, A3, T3)
    assert null['full_delta_log_ratio'] == 0.0, null
    assert null['sign_stable_log_ratio'] is None, null
    assert null['fragile_to_one_unit'] is None, null
    assert null['most_influential_unit'] is None, null

    # Fewer than three units cannot support a delete-one jackknife.
    short = L.paired_nll_unit_influence([[2.0, 3.0]], [[3.0, 4.0]], [10.0, 10.0])
    assert short['available'] is False and short['n_units'] == 2, short
    # A small unit can be so important that deleting it overflows the pooled
    # PPL: that fold must be named unavailable, not silently become inf/nan.
    holder = L.paired_nll_unit_influence([[0.0, 1000.0, 1000.0]],
                                         [[0.0, 0.0, 0.0]],
                                         [1000.0, 1.0, 1.0])
    assert holder['available'] is True and holder['jackknife_available'] is False, holder
    assert holder['loo_available'][0] is False, holder
    assert holder['loo_available'][1] is True, holder
    # An unavailable fold is JSON-safe null in every per-fold vector, never
    # inf/NaN that a strict writer would reject or a reader could mistake for
    # a point estimate.
    assert holder['loo_delta_ppl'][0] is None, holder
    assert holder['loo_change_ppl'][0] is None, holder
    assert holder['loo_delta_log_ratio'][0] is None, holder
    assert holder['loo_change_log_ratio'][0] is None, holder
    assert holder['loo_delta_ppl'][1] is not None, holder
    json.dumps(holder, allow_nan=False)

    # All LOO folds can be finite while their squared deviations exceed the
    # float range.  The jackknife must then be explicitly unavailable (no
    # inf SE/CI masquerading as a measured interval), and the object stays
    # JSON-compliant.
    extreme = L.paired_nll_unit_influence([[0.0, 500.0, 500.0]],
                                          [[0.0, 0.0, 0.0]],
                                          [10, 1, 1])
    assert extreme['available'] is True, extreme
    assert extreme['loo_available'] == [True, True, True], extreme
    assert extreme['jackknife_available'] is False, extreme
    assert extreme['jackknife_se_ppl'] is None, extreme
    assert extreme['jackknife_ci95_ppl'] is None, extreme
    assert extreme['jackknife_unavailable_reason'], extreme
    json.dumps(extreme, allow_nan=False)
    # The cluster-robust variance has the same overflow contract: a finite
    # point estimate with an unrepresentable variance is a point-only result,
    # never an inf interval.
    _pc = L._paired_nll_doc_ci([[700.0, 701.0, 699.0, 700.0]],
                               [[0.0, 0.0, 0.0, 0.0]], [1, 1, 1, 1])
    assert math.isfinite(_pc['delta_ppl']), _pc
    assert _pc['se_ppl'] is None and _pc['delta_ppl_ci95'] is None, _pc

    # Shape/token/id/option guards: a malformed id vector or fold must not
    # silently index the wrong unit.
    bad_inputs = [
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0]], [1.0, 1.0, 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [1.0, 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [1.0, 0.0, 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [1.0, float('nan'), 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [1.5, 1.0, 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [True, 1.0, 1.0]),
        ([[True, 2.0, 3.0]], [[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0]),
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, float('inf')]], [1.0, 1.0, 1.0]),
    ]
    for _a, _b, _t in bad_inputs:
        try:
            L.paired_nll_unit_influence(_a, _b, _t)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed influence input {(_a, _b, _t)!r}')
    for _ids in (['a', 'b'], ['a', 'a', 'c'], 'abc'):
        try:
            L.paired_nll_unit_influence([[1.0, 2.0, 3.0]], [[2.0, 2.0, 2.0]],
                                        [1.0, 1.0, 1.0], unit_ids=_ids)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted bad unit_ids={_ids!r}')
    for _kw in ({'top_k': 0}, {'top_k': 1.5}, {'top_k': True},
                {'conf': 0.0}, {'conf': True}):
        try:
            L.paired_nll_unit_influence([[1.0, 2.0, 3.0]], [[2.0, 2.0, 2.0]],
                                        [1.0, 1.0, 1.0], **_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted bad influence option {_kw!r}')


def test_doc_influence_analysis_and_markdown():
    """L174 — the v12 analysis records unit influence and the report caveats it."""
    docs = ['d0', 'd1', 'd2', 'd3']
    toks = [10, 10, 10, 10]
    nll_a = [1.0, 1.0, 1.0, 200.0]
    nll_b = [2.0, 2.0, 2.0, 2.0]
    recs_a = {s: {'by_len': {512: _document_cell(nll_a, toks, docs)}}
              for s in (0, 1)}
    recs_b = {s: {'by_len': {512: _document_cell(nll_b, toks, docs)}}
              for s in (0, 1)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    inf = got.get('unit_influence')
    assert isinstance(inf, dict) and inf.get('available') is True, got
    assert inf['unit_ids'] == docs, inf
    assert abs(inf['full_delta_ppl'] - got['pooled_delta_ppl']) < 1e-12, (inf, got)
    assert inf['most_influential_unit'] == 'd3', inf
    assert inf['fragile_to_one_unit'] is True, inf
    md = V12._doc_influence_markdown(
        {'comparisons': {'csa_vs_full': {'doc_sampling': got}}})
    text = '\n'.join(md)
    assert '单元影响诊断' in text and 'd3' in text, text
    assert '单单元主导' in text and '不改变上面的冻结判定' in text, text
    assert '不能把结论描述为' in text, text
    # A uniform effect is not flagged, and the table says so without turning
    # the absence of a single-unit flip into a population claim.
    flat_a = [2.0, 2.0, 2.0, 2.0]
    flat_b = [3.0, 3.0, 3.0, 3.0]
    fa = {s: {'by_len': {512: _document_cell(flat_a, toks, docs)}}
          for s in (0, 1)}
    fb = {s: {'by_len': {512: _document_cell(flat_b, toks, docs)}}
          for s in (0, 1)}
    flat = V12._doc_level_analysis(fa, fb, [0, 1], 512)
    ftext = '\n'.join(V12._doc_influence_markdown(
        {'comparisons': {'csa_vs_full': {'doc_sampling': flat}}}))
    assert '未见单单元翻转' in ftext and '⚠ 单单元主导' not in ftext, ftext
    # The article unit is preferred and its ids, not the line ids, label folds.
    cell_a = _article_cell([1.0, 1.0, 1.0, 200.0], toks,
                           ['l0', 'l1', 'l2', 'l3'],
                           [1.0, 1.0, 1.0, 200.0], toks,
                           ['a0', 'a1', 'a2', 'a3'])
    cell_b = _article_cell([2.0, 2.0, 2.0, 2.0], toks,
                           ['l0', 'l1', 'l2', 'l3'],
                           [2.0, 2.0, 2.0, 2.0], toks,
                           ['a0', 'a1', 'a2', 'a3'])
    art_a = {s: {'by_len': {512: dict(cell_a)}} for s in (0, 1)}
    art_b = {s: {'by_len': {512: dict(cell_b)}} for s in (0, 1)}
    art = V12._doc_level_analysis(art_a, art_b, [0, 1], 512)
    assert art['granularity'] == 'article', art
    assert art['unit_influence']['unit_ids'] == ['a0', 'a1', 'a2', 'a3'], art
    assert art['unit_influence']['most_influential_unit'] == 'a3', art
    # No influence block means no table (never a fabricated "stable" line);
    # a pooled CI without the block is named as unverified, not omitted.
    assert V12._doc_influence_markdown({}) == []
    assert V12._doc_influence_markdown(
        {'comparisons': {'x': {'doc_sampling': {'unit': 'document'}}}}) == []
    miss = V12._doc_influence_markdown({'comparisons': {'x': {'doc_sampling': {
        'unit': 'document', 'pooled_delta_ppl_ci95': [-1.0, 1.0],
        'unit_influence_unavailable_reason': 'unit ids are not unique'}}}})
    mtext = '\n'.join(miss)
    assert '单单元敏感性**未验证**' in mtext and 'unit ids are not unique' in mtext, mtext


def test_unit_influence_audit_gate():
    """L175 — the audit checks the influence block and warns on fragility."""
    nll_a = [[1.0, 1.0, 1.0, 200.0]]
    nll_b = [[2.0, 2.0, 2.0, 2.0]]
    toks = [10.0, 10.0, 10.0, 10.0]
    inf = L.paired_nll_unit_influence(nll_a, nll_b, toks,
                                      unit_ids=['a', 'b', 'c', 'd'])
    comp = {'doc_sampling': {
        'unit': 'document',
        'pooled_delta_ppl': inf['full_delta_ppl'],
        'pooled_delta_ppl_ci95': [-1.0, 1.0],
        'unit_influence': inf,
    }}
    stats = {'comparisons': {'x': comp}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_unit_influence(path=p)
        assert aud['status'] == 'warnings', aud
        assert aud['n_fragile'] == 1 and aud['n_ok'] == 0, aud
        assert any('x' in w and 'single-unit' in w for w in aud['warnings']), aud
        assert aud['comparisons']['x']['most_influential_unit'] == 'd', aud

        # The same block with a uniform effect passes the consistency gate.
        inf_ok = L.paired_nll_unit_influence(
            [[2.0, 2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0, 3.0]],
            [10.0, 10.0, 10.0, 10.0], unit_ids=['a', 'b', 'c', 'd'])
        stats_ok = {'comparisons': {'x': {
            'doc_sampling': {'unit': 'document',
                             'pooled_delta_ppl': inf_ok['full_delta_ppl'],
                             'pooled_delta_ppl_ci95': [-1.0, 1.0],
                             'unit_influence': inf_ok}}}}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stats_ok, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_ok = A.audit_unit_influence(path=p)
        assert aud_ok['status'] == 'ok' and aud_ok['n_ok'] == 1, aud_ok

        # Numeric strings are not measurements: stringified readings must fail
        # instead of replaying as a valid influence block.
        stringified = json.loads(json.dumps(stats_ok))
        _u = stringified['comparisons']['x']['doc_sampling']['unit_influence']
        _u['full_delta_ppl'] = str(_u['full_delta_ppl'])
        _u['loo_delta_ppl'][0] = str(_u['loo_delta_ppl'][0])
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stringified, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_string = A.audit_unit_influence(path=p)
        assert aud_string['status'] == 'fail', aud_string
        assert any('non-finite' in x for x in aud_string['problems']), aud_string

        # The availability flag is a boolean, not a truthy string.
        bad_avail = json.loads(json.dumps(stats_ok))
        bad_avail['comparisons']['x']['doc_sampling']['unit_influence'][
            'available'] = 'false'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad_avail, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_avail = A.audit_unit_influence(path=p)
        assert aud_avail['status'] == 'fail', aud_avail
        assert any('available' in x for x in aud_avail['problems']), aud_avail

        # A point estimate that no longer matches the pooled Δ is a hard fail.
        bad = json.loads(json.dumps(stats))
        bad['comparisons']['x']['doc_sampling']['unit_influence'][
            'full_delta_ppl'] += 1.0
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_bad = A.audit_unit_influence(path=p)
        assert aud_bad['status'] == 'fail', aud_bad
        assert any('does not reproduce' in x for x in aud_bad['problems']), aud_bad

        # Duplicate ids and non-finite fold values are also hard failures.
        bad_ids = json.loads(json.dumps(stats))
        bad_ids['comparisons']['x']['doc_sampling']['unit_influence'][
            'unit_ids'][1] = bad_ids['comparisons']['x']['doc_sampling'][
                'unit_influence']['unit_ids'][0]
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad_ids, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_ids = A.audit_unit_influence(path=p)
        assert aud_ids['status'] == 'fail', aud_ids
        assert any('not unique' in x for x in aud_ids['problems']), aud_ids
        bad_vec = json.loads(json.dumps(stats))
        bad_vec['comparisons']['x']['doc_sampling']['unit_influence'][
            'loo_delta_ppl'][0] = float('nan')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad_vec, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_vec = A.audit_unit_influence(path=p)
        assert aud_vec['status'] == 'fail', aud_vec
        assert any('non-finite' in x for x in aud_vec['problems']), aud_vec
        # A non-object comparison is corrupt, not silently absent.
        bad_comp = json.loads(json.dumps(stats_ok))
        bad_comp['comparisons']['bad'] = ['not', 'an', 'object']
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad_comp, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_comp = A.audit_unit_influence(path=p)
        assert aud_comp['status'] == 'fail', aud_comp
        assert any('not an object' in x for x in aud_comp['problems']), aud_comp

        # A stale pooled-CI artifact without the block is unverified, not clean.
        stale = {'comparisons': {'x': {
            'doc_sampling': {'unit': 'document',
                             'pooled_delta_ppl': 1.0,
                             'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stale, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_stale = A.audit_unit_influence(path=p)
        assert aud_stale['status'] == 'warnings', aud_stale
        assert aud_stale['n_missing'] == 1, aud_stale
        # A missing artifact is explicitly not verifiable, not a pass.
        with contextlib.redirect_stdout(io.StringIO()):
            aud_missing = A.audit_unit_influence(path=os.path.join(td, 'none.json'))
        assert aud_missing['status'] == 'not_verifiable_locally', aud_missing


def test_sequential_look_signflip_primitives():
    """L176 — max-over-looks sequential p is exact, directional and monotone."""
    d1 = [-1.0, -1.0, -1.0]
    one = L.sequential_signflip_p([(list(range(3)), d1)])
    assert one['n_looks'] == 1 and one['method'] == 'single-look', one
    assert abs(one['p_sequential'] - L.exact_signflip_p(d1)['p']) < 1e-12, one
    # Nested looks: the sequential p can never be smaller than the final-look
    # two-sided p (the final look is inside the max).
    d2 = [-1.0] * 8
    s2 = L.sequential_signflip_p([(list(range(3)), d2[:3]),
                                  (list(range(6)), d2[:6]),
                                  (list(range(8)), d2)])
    assert s2['p_sequential'] >= s2['looks'][-1]['p'] - 1e-12, s2
    assert s2['method'] == 'exact-max-over-looks-final-direction', s2
    # A concrete vector where the final-n p is < alpha but the sequential p is
    # larger: the same data at an extra look is not the same evidence.
    d3 = [-1.34, -1.18, -1.46, -1.21, -0.7, -1.1, 0.42, -0.48, 0.18, -0.04]
    s3 = L.sequential_signflip_p([(list(range(6)), d3[:6]),
                                  (list(range(10)), d3)])
    assert s3['looks'][-1]['p'] < s3['p_sequential'] - 1e-12, s3
    # An early look in the OPPOSITE direction must not create a sequential
    # claim for the final (near-null) direction.
    s4 = L.sequential_signflip_p([
        (list(range(6)), [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
                                   31.0, 31.0])])
    assert s4['p_sequential'] == 1.0, s4
    # Inconsistent seed deltas across looks (a re-run) fall back to the
    # conservative Bonferroni bound and name the seed; never silently combine.
    s5 = L.sequential_signflip_p([([0, 1, 2], [-1.0, -1.0, -1.0]),
                                  ([0, 1, 2, 3], [-1.0, -1.0, -2.0, -1.0])])
    assert 'inconsistent' in s5['method'] and s5['inconsistent_seeds'] == [2], s5
    assert abs(s5['p_sequential'] - min(1.0, 2 * s5['p_final_look'])) < 1e-12, s5
    # Too many seeds for joint enumeration -> valid Bonferroni over the final p.
    s6 = L.sequential_signflip_p([
        (list(range(17)), [-1.0] * 17),
        (list(range(18)), [-1.0] * 17 + [-0.5])])
    assert s6['method'] == 'bonferroni-over-looks', s6
    assert abs(s6['p_sequential'] - min(1.0, 2 * s6['p_final_look'])) < 1e-12, s6
    # Determinism: the enumeration must not depend on hash ordering.
    assert L.sequential_signflip_p([(list(range(6)), d3[:6]),
                                    (list(range(10)), d3)])['p_sequential'] == s3['p_sequential']
    # Input contract failures are explicit, not silent drops.
    for bad in ([([0, 0], [1.0, 2.0])],
                [([0, 1], [1.0])],
                [([], [])],
                [([0.5], [1.0])]):
        try:
            L.sequential_signflip_p(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed look series must be rejected: {bad!r}')
    # A non-finite delta is dropped and counted, matching exact_signflip_p's
    # contract instead of silently treating NaN as a measurement.
    drop = L.sequential_signflip_p([([0, 1], [1.0, float('nan')])])
    assert drop['looks'][0]['n'] == 1 and drop['looks'][0]['n_dropped'] == 1, drop


def test_look_history_recorded_and_audited():
    """L177 — look history is deduped, stale-aware, and drives the audit p."""
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    knobs = L.training_knobs()
    deltas = [-1.34, -1.18, -1.46, -1.21, -0.7, -1.1, 0.42, -0.48, 0.18, -0.04]
    design = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed', 'x')]}
    key = 'component::dynamic_blocking::csa_dynamic|csa_fixed||'

    def _summary(n):
        s = {}
        for i in range(n):
            s[f'csa_dynamic::seed{i}'] = dict(
                variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=50.0 + deltas[i], params=100, train_knobs=knobs)
            s[f'csa_fixed::seed{i}'] = dict(
                variant='csa_fixed', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=50.0, params=100, train_knobs=knobs)
        return s

    with tempfile.TemporaryDirectory() as td:
        recs6, _ = L._variant_seed_map(_summary(6))
        h1 = L.record_panel_look(td, recs6)
        assert h1['n_looks'] == 1, h1
        recs10, _ = L._variant_seed_map(_summary(10))
        h2 = L.record_panel_look(td, recs10)
        assert h2['n_looks'] == 2, h2
        # Re-aggregating unchanged data is not a new look.
        assert L.record_panel_look(td, recs10)['n_looks'] == 2
        hist = L.load_look_history(td)
        assert len(hist['looks']) == 2 and key in hist['looks'][-1]['contrasts'], hist
        by_key = L.looks_by_contrast(hist)
        assert len(by_key[key]) == 2 and by_key[key][-1]['seeds'] == list(range(10))

        rep = L.methodology_audit(_summary(10), design=design, baselines={},
                                  outdir=td, verbose=False)
        e = rep['components']['dynamic_blocking'][0]
        la = e['look_analysis']
        assert la['n_looks'] == 2 and e['look_adjusted'] is True, e
        assert abs(e['p_look_adjusted'] - la['p_sequential']) < 1e-12, e
        assert abs(e['p_holm_global'] - la['p_sequential']) < 1e-12, e
        assert abs(e['p_holm_global_naive'] - e['p_exact_signflip']) < 1e-12, e
        assert e['p_holm_global'] >= e['p_holm_global_naive'] - 1e-12, e
        assert rep['look_history']['panel_looks'] == 2, rep['look_history']
        assert rep['look_history']['n_contrasts_look_adjusted'] == 1, rep['look_history']
        # TOST equivalence is also corrected for the spent looks.
        _se = e.get('sesoi') or {}
        assert _se.get('n_looks') == 2 and _se.get('p_tost_look_adjusted') is not None, _se
        assert abs(float(_se['p_tost_look_adjusted'])
                   - min(1.0, 2.0 * float(_se['tost_p_tost']))) < 1e-12, _se

        # The independent gate recomputes the same sequential p and passes.
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_look_history(paths=[td])
        assert aud['status'] == 'ok', aud
        assert aud['n_panels'] == 1 and aud['n_looks'] == 2, aud
        assert aud['n_contrasts_adjusted'] == 1, aud

        # A re-run that changes an early seed's delta is STALE: the old look is
        # excluded and named, not combined into the joint null.
        stale_hist = json.loads(json.dumps(hist))
        stale_hist['looks'][0]['contrasts'][key]['deltas'][0] = -9.0
        with contextlib.redirect_stdout(io.StringIO()):
            rep_stale = L.methodology_audit(_summary(10), design=design,
                                            baselines={}, look_history=stale_hist,
                                            verbose=False)
        e_stale = rep_stale['components']['dynamic_blocking'][0]
        assert e_stale['look_analysis']['n_stale_looks'] >= 1, e_stale
        assert any('STALE' in w for w in rep_stale['warnings']), rep_stale['warnings']
        # The independent gate mirrors the stale exclusion: it warns about the
        # changed seed but still reproduces the audit's (remaining) sequential p.
        with open(os.path.join(td, 'look_history.json'), 'w', encoding='utf-8') as f:
            json.dump(stale_hist, f)
        L.methodology_audit(_summary(10), design=design, baselines={},
                            outdir=td, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_stale = A.audit_look_history(paths=[td])
        assert aud_stale['status'] == 'warnings', aud_stale
        assert any('stale look' in w for w in aud_stale['warnings']), aud_stale

        # A tampered recorded per-look p is a hard audit failure.
        bad = json.loads(json.dumps(hist))
        bad['looks'][0]['contrasts'][key]['p_exact_signflip'] = 0.999
        with open(os.path.join(td, 'look_history.json'), 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_bad = A.audit_look_history(paths=[td])
        assert aud_bad['status'] == 'fail', aud_bad
        assert any('does not recompute' in p for p in aud_bad['problems']), aud_bad

        # A numeric string that happens to equal the recomputed p is still not
        # a measurement: present-but-unparseable evidence must fail closed.
        string_p = json.loads(json.dumps(hist))
        _orig_p = string_p['looks'][0]['contrasts'][key]['p_exact_signflip']
        string_p['looks'][0]['contrasts'][key]['p_exact_signflip'] = str(_orig_p)
        with open(os.path.join(td, 'look_history.json'), 'w', encoding='utf-8') as f:
            json.dump(string_p, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_string_p = A.audit_look_history(paths=[td])
        assert aud_string_p['status'] == 'fail', aud_string_p
        assert any('not a finite number' in p
                   for p in aud_string_p['problems']), aud_string_p

        # Missing audit block: recorded looks cannot be cross-checked -> warn.
        with open(os.path.join(td, 'look_history.json'), 'w', encoding='utf-8') as f:
            json.dump(hist, f)
        os.remove(os.path.join(td, 'methodology_audit.json'))
        with contextlib.redirect_stdout(io.StringIO()):
            aud_missing = A.audit_look_history(paths=[td])
        assert aud_missing['status'] == 'warnings', aud_missing
        assert any('no look_analysis block' in w for w in aud_missing['warnings']), aud_missing
    # No look files at all is not a pass.
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            aud_empty = A.audit_look_history(paths=[td])
        assert aud_empty['status'] == 'not_verifiable_locally', aud_empty


def test_look_history_audit_flags_downgrade():
    """L178 — the audit surfaces a naive-significant / sequential-null claim."""
    deltas = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.5]
    key = 'component::dynamic_blocking::csa_dynamic|csa_fixed||'
    hist = {'schema': 1, 'looks': [
        {'look': 1, 'contrasts': {key: {
            'seeds': [0, 1, 2, 3, 4, 5], 'deltas': deltas[:6],
            'p_exact_signflip': L.exact_signflip_p(deltas[:6])['p']}}},
        {'look': 2, 'contrasts': {key: {
            'seeds': list(range(8)), 'deltas': deltas,
            'p_exact_signflip': L.exact_signflip_p(deltas)['p']}}},
    ]}
    seq = L.sequential_signflip_p(
        [{'seeds': hist['looks'][0]['contrasts'][key]['seeds'],
          'deltas': deltas[:6]},
         {'seeds': hist['looks'][1]['contrasts'][key]['seeds'],
          'deltas': deltas}])
    # A synthetic audit block whose naive verdict is significant but whose
    # sequential verdict is not; the gate must reproduce the p and flag it.
    audit = {'components': {'dynamic_blocking': [{
        'look_analysis': {'contrast': key, 'n_looks': 2,
                          'p_sequential': seq['p_sequential']},
        'significant_global_naive': True, 'significant_global': False,
        'p_holm_global_naive': 0.01, 'p_holm_global': seq['p_sequential'],
    }]}, 'baselines': {}}
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'look_history.json'), 'w', encoding='utf-8') as f:
            json.dump(hist, f)
        with open(os.path.join(td, 'methodology_audit.json'), 'w', encoding='utf-8') as f:
            json.dump(audit, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_look_history(paths=[td])
        assert aud['status'] == 'ok', aud
        assert len(aud['downgraded_claims']) == 1, aud
        assert aud['downgraded_claims'][0]['contrast'] == key, aud
        # A stale audit p (not what the recorded deltas imply) is a hard fail.
        audit['components']['dynamic_blocking'][0]['look_analysis'][
            'p_sequential'] = 0.001
        with open(os.path.join(td, 'methodology_audit.json'), 'w', encoding='utf-8') as f:
            json.dump(audit, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud_bad = A.audit_look_history(paths=[td])
        assert aud_bad['status'] == 'fail', aud_bad
        assert any('does not match the recomputed' in p for p in aud_bad['problems']), aud_bad


def test_look_history_accepts_raw_summary_keys():
    """L187 — the real aggregation path feeds string-keyed summaries to the
    look recorder; they must be normalized, not unpacked as (variant, tag, seed)."""
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    knobs = L.training_knobs()
    summary = {}
    for i in range(6):
        summary[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0 - (i + 1) / 10.0, params=100, train_knobs=knobs)
        summary[f'csa_fixed::seed{i}'] = dict(
            variant='csa_fixed', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs)
    design = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed')]}
    key = 'component::dynamic_blocking::csa_dynamic|csa_fixed||'
    with tempfile.TemporaryDirectory() as td:
        hist = L.record_panel_look(td, summary, design=design, baselines={})
        assert hist['n_looks'] == 1, hist
        snap = hist['looks'][0]['contrasts']
        assert key in snap, snap
        assert snap[key]['seeds'] == list(range(6)), snap[key]
        assert snap[key]['n'] == 6, snap[key]


def test_look_series_array_inputs_and_strict_audit_seeds():
    """L179 — array look inputs work; the audit refuses truncated seeds."""
    d = [-1.0, -1.0, -1.0]
    # A NumPy array is a legitimate sequence; `x or []` used to raise
    # "truth value ... is ambiguous" before the function could read it.
    arr = L.sequential_signflip_p([(np.asarray([0, 1, 2]),
                                    np.asarray(d))])
    plain = L.sequential_signflip_p([([0, 1, 2], d)])
    assert arr['p_sequential'] == plain['p_sequential'], (arr, plain)
    mapping = L.sequential_signflip_p([{'seeds': np.asarray([0, 1, 2]),
                                        'deltas': np.asarray(d)}])
    assert mapping['p_sequential'] == plain['p_sequential'], mapping
    # Non-integer/boolean recorded seeds are not silently truncated (1.5 and
    # True would both become seed 1 under plain int()).
    key = 'component::dynamic_blocking::csa_dynamic|csa_fixed||'
    d1, d2 = [-1.0, -2.0], [-1.0, -2.0, -3.0]
    for bad in ([0, 1.5], [0, True], [0, 'x']):
        hist = {'schema': 1, 'looks': [
            {'look': 1, 'contrasts': {key: {
                'seeds': bad, 'deltas': d1,
                'p_exact_signflip': L.exact_signflip_p(d1)['p']}}},
            {'look': 2, 'contrasts': {key: {
                'seeds': [0, 1, 2], 'deltas': d2,
                'p_exact_signflip': L.exact_signflip_p(d2)['p']}}}]}
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, 'look_history.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(hist, f)
            with contextlib.redirect_stdout(io.StringIO()):
                aud = A.audit_look_history(paths=[td])
        assert aud['status'] == 'fail', (bad, aud)
        assert any('non-integer seed' in p for p in aud['problems']), aud
    # A genuinely empty look is a malformed record, not an empty evidence set.
    hist['looks'][0]['contrasts'][key].update({'seeds': [], 'deltas': []})
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'look_history.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(hist, f)
        with contextlib.redirect_stdout(io.StringIO()):
            aud = A.audit_look_history(paths=[td])
    assert aud['status'] == 'fail' and any('empty seeds/deltas' in p
                                           for p in aud['problems']), aud


def test_full_width_cap_accounts_for_window_tables():
    """L180 — the full-width gate counts the O(n*w) window tables it allocates."""
    # At k=32, w=128 the hoisted masks/rows alone fit 64 MiB at n=39000, but
    # the cached window index/validity and block-offset tables push the real
    # footprint above it.  The chunked path is value-identical and must be
    # selected instead.
    assert L._attn_full_width_ok(39000, 32, 128, False) is False
    # Normal short-context cells still take the faster hoisted path.
    assert L._attn_full_width_ok(2048, 32, 128, False) is True
    assert L._attn_full_width_ok(128, 8, 16, False) is True


def test_dose_response_design_requires_ordered_single_axis_ladder():
    """L181 — a dose ladder is not a pair of endpoints with a label."""
    base = [
        {'label': 'k8', 'dose': 8, 'arm': 'k8', 'seeds': [0, 1, 2],
         'protocol': 'uniform/4097'},
        {'label': 'k32', 'dose': 32, 'arm': 'k32', 'seeds': [0, 1, 2],
         'protocol': 'uniform/4097'},
        {'label': 'k128', 'dose': 128, 'arm': 'k128', 'seeds': [0, 1, 2],
         'protocol': 'uniform/4097'},
    ]
    r = L.dose_response_design_audit(
        base, direction='increasing', dose_key='index_topk',
        factory=_dose_factory, n_layers=2)
    assert r['status'] == 'ok', r
    assert r['n_levels'] == 3 and r['n_seeds'] == 3, r
    assert r['dose_key_values'] == {'k8': 8.0, 'k32': 32.0, 'k128': 128.0}, r
    assert r['levels']['k32']['cfg_diff_keys'] == ['index_topk'], r
    # The wrapper accepts the explicit design object too.
    with contextlib.redirect_stdout(io.StringIO()):
        wrapped = A.audit_dose_response(
            design={'levels': base, 'direction': 'increasing',
                    'dose_key': 'index_topk'},
            factory=_dose_factory)
    assert wrapped['status'] == 'ok', wrapped
    # An extra moved key is a second factor, not a dose axis.
    extra = base[:2] + [{'label': 'mixed', 'dose': 128,
                         'arm': 'k32_plus_block', 'seeds': [0, 1, 2],
                         'protocol': 'uniform/4097'}]
    bad = L.dose_response_design_audit(
        extra, direction='increasing', dose_key='index_topk',
        factory=_dose_factory, n_layers=2)
    assert bad['status'] == 'fail', bad
    assert any('not isolated' in v and 'block_size' in v
               for v in bad['violations']), bad
    # Duplicate doses, mismatched seeds, a missing seed declaration, and a
    # two-point "ladder" are all design failures.
    dup = base[:2] + [{'label': 'dup', 'dose': 32, 'arm': 'k128',
                       'seeds': [0, 1, 2], 'protocol': 'uniform/4097'}]
    for levels, why in [
        (dup, 'duplicate dose'),
        (base[:2] + [{'label': 'k128', 'dose': 128, 'arm': 'k128',
                      'seeds': [0, 1], 'protocol': 'uniform/4097'}],
         'seed set'),
        ([{k: v for k, v in rec.items() if k != 'seeds'} for rec in base],
         'seed set'),
        (base[:2], 'at least 3'),
    ]:
        rep = L.dose_response_design_audit(
            levels, direction='increasing', dose_key='index_topk',
            factory=_dose_factory, n_layers=2)
        assert rep['status'] == 'fail', (why, rep)
        assert any(why in v for v in rep['violations']), (why, rep)
    # A protocol change between doses confounds the trend.
    proto = [dict(rec, protocol='uniform/4097' if rec['label'] != 'k32'
                  else 'uniform/2049') for rec in base]
    rep = L.dose_response_design_audit(
        proto, direction='increasing', dose_key='index_topk',
        factory=_dose_factory, n_layers=2)
    assert rep['status'] == 'fail' and any('protocol' in v
                                           for v in rep['violations']), rep
    # The declared dose order must agree with the actual key values (inverse
    # coding is fine; a scrambled ladder is not).
    scrambled = [
        {'label': 'a', 'dose': 8, 'arm': 'k8', 'seeds': [0], 'protocol': 'p'},
        {'label': 'b', 'dose': 32, 'arm': 'k128', 'seeds': [0], 'protocol': 'p'},
        {'label': 'c', 'dose': 128, 'arm': 'k32', 'seeds': [0], 'protocol': 'p'},
    ]
    rep = L.dose_response_design_audit(
        scrambled, direction='increasing', dose_key='index_topk',
        factory=_dose_factory, n_layers=2, min_seeds=1)
    assert rep['status'] == 'fail' and any('not monotonic' in v
                                           for v in rep['violations']), rep
    # An auto/post-hoc direction is not representable.
    rep = L.dose_response_design_audit(
        base, direction='auto', dose_key='index_topk',
        factory=_dose_factory, n_layers=2)
    assert rep['status'] == 'fail' and any('direction' in v
                                           for v in rep['violations']), rep
    # The shape is a pre-registered design choice, not an analysis-time option.
    rep = L.dose_response_design_audit(
        base, direction='increasing', dose_key='index_topk',
        factory=_dose_factory, n_layers=2, shape='u-shape')
    assert rep['status'] == 'fail' and any('shape' in v
                                           for v in rep['violations']), rep
    # Vector-valued config fields beside the scalar dose axis must not make
    # the diff comparison raise (a multi-element tensor is not truth-testable).
    def _tensor_factory(n, arm):
        return [{'kind': 'csa', 'block_size': 4,
                 'index_topk': {'k8': 8, 'k32': 32, 'k128': 128}[arm],
                 'mask': torch.tensor([1.0, 2.0])}
                for _ in range(int(n))]

    tr = L.dose_response_design_audit(
        base, direction='increasing', dose_key='index_topk',
        factory=_tensor_factory, n_layers=2)
    assert tr['status'] == 'ok', tr


def test_dose_response_trend_is_exact_directional_and_shape_gated():
    """L182 — one pre-registered ordered test, not endpoint cherry-picking."""
    def levels(fn, n=6, doses=(0, 1, 2)):
        return [{'label': str(d), 'dose': float(d),
                 'values': {s: fn(s, d) for s in range(n)}}
                for d in doses]

    pos = L.dose_response_trend(
        levels(lambda s, d: 10.0 + (0.5 + 0.2 * s) * d),
        direction='increasing')
    assert pos['status'] == 'ok' and pos['verdict'] == 'monotone_trend_supported', pos
    assert pos['claim_allowed'] is True, pos
    assert pos['p_exact_signflip'] == 2.0 / 2 ** 6, pos
    assert pos['monotonicity']['all_mean_steps_in_direction'] is True, pos
    assert pos['monotonicity']['strict_monotone_seeds'] == 6, pos
    assert pos['slope_stats']['n'] == 6, pos
    # The same data support the opposite pre-registered direction only as a
    # conflict, never as a positive claim.
    neg = L.dose_response_trend(
        levels(lambda s, d: 10.0 - (0.5 + 0.2 * s) * d),
        direction='increasing')
    assert neg['verdict'] == 'direction_conflict', neg
    assert neg['claim_allowed'] is False, neg
    # A monotone shape requires every adjacent mean step, while a linear
    # trend can still be reported without upgrading it to monotone.
    bent = levels(lambda s, d: (0.0, 10.0, 9.0)[d])
    lin = L.dose_response_trend(bent, direction='increasing', shape='linear')
    mono = L.dose_response_trend(bent, direction='increasing', shape='monotone')
    assert lin['verdict'] == 'linear_trend_supported' and lin['claim_allowed'], lin
    assert mono['verdict'] == 'shape_conflict' and not mono['claim_allowed'], mono
    # Incomplete cases are named, not silently averaged; five seeds cannot
    # clear the exact-test resolution floor at alpha=.05.
    missing = L.dose_response_trend(
        levels(lambda s, d: None if (s == 0 and d == 1)
               else 10.0 + (0.5 + 0.2 * s) * d),
        direction='increasing')
    assert missing['n_complete_seeds'] == 5, missing
    assert missing['excluded_seeds'][0]['seed'] == 0, missing
    assert missing['verdict'] == 'resolution_limited', missing
    # Non-finite values are excluded with a warning rather than becoming a
    # fake zero.
    nonfinite = levels(lambda s, d: float('nan') if (s == 1 and d == 2)
                       else 10.0 + (0.5 + 0.2 * s) * d)
    nf = L.dose_response_trend(nonfinite, direction='increasing')
    assert nf['n_complete_seeds'] == 5 and any('non-finite' in w
                                               for w in nf['warnings']), nf
    # A two-point contrast is a named endpoint comparison, not a trend.
    two = L.dose_response_trend(
        levels(lambda s, d: 10.0 + s * d, n=6, doses=(0, 1)),
        direction='increasing')
    assert two['status'] == 'insufficient_levels' and not two['claim_allowed'], two
    # Protocol mismatch blocks the causal reading even if p is small.
    proto = levels(lambda s, d: 10.0 + (0.5 + 0.2 * s) * d)
    proto[0]['protocol'] = 'eval/2049'
    proto[1]['protocol'] = 'eval/4097'
    proto[2]['protocol'] = 'eval/4097'
    pc = L.dose_response_trend(proto, direction='increasing')
    assert pc['verdict'] == 'protocol_conflict' and not pc['claim_allowed'], pc
    # An unknown direction/shape is a caller error, not an automatic choice.
    for kw in ({'direction': 'up'}, {'direction': 'increasing', 'shape': 'u-shape'}):
        try:
            L.dose_response_trend(levels(lambda s, d: 10.0 + s * d), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid trend parameter accepted: {kw}')


def test_layer_cfg_diff_handles_vector_fields():
    """L185 — dataclass config diffs must not truth-test elementwise vectors."""
    import dataclasses as _dc

    @_dc.dataclass
    class _VectorCfg:
        kind: str = 'csa'
        index_topk: int = 32
        mask: object = None

    a = _VectorCfg(mask=np.array([1.0, 2.0]))
    same = _VectorCfg(mask=np.array([1.0, 2.0]))
    assert L.layer_cfg_diff(a, same) == {}
    assert L.layer_cfg_diff(
        _VectorCfg(mask=torch.tensor([1.0, 2.0])),
        _VectorCfg(mask=torch.tensor([1.0, 2.0]))) == {}
    assert set(L.layer_cfg_diff(
        a, _VectorCfg(mask=torch.tensor([1.0, 3.0])))) == {'mask'}
    # Broadcast-incompatible shapes are still a difference, not a crash.
    assert set(L.layer_cfg_diff(
        a, _VectorCfg(mask=np.array([1.0, 2.0, 3.0])))) == {'mask'}

    def _factory(n_layers, arm):
        return [_VectorCfg(index_topk=8 if arm == 'k8' else 128,
                           mask=torch.tensor([1.0, 2.0]))
                for _ in range(int(n_layers))]

    keys, per_layer = L._design_pair_cfg_keys('k8', 'k128', _factory, 2)
    assert keys == ['index_topk'], (keys, per_layer)
    assert set(per_layer) == {0, 1}, per_layer
    assert all(set(d) == {'index_topk'} for d in per_layer.values()), per_layer


def test_statistical_calibration_gates_and_suite():
    """L183 — the statistical machinery is calibrated before its output counts."""
    assert abs(L._binomial_sf(1, 10, 0.5) - (1.0 - 0.5 ** 10)) < 1e-15
    assert abs(L._binomial_cdf(0, 10, 0.5) - 0.5 ** 10) < 1e-15
    assert L._binomial_sf(0, 10, 0.5) == 1.0
    assert L._binomial_cdf(10, 10, 0.5) == 1.0
    try:
        L._binomial_sf(11, 10, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range binomial count accepted')
    # A conservative procedure passes; a grossly anti-conservative procedure
    # fails the one-sided control in the predicted direction.
    assert L.calibration_type_i_gate(0, 200, alpha=0.05)['status'] == 'ok'
    bad = L.calibration_type_i_gate(200, 200, alpha=0.05)
    assert bad['status'] == 'fail' and bad['p_anti_conservative'] < 1e-6, bad
    # A test with no power cannot be rescued by passing the null control.
    zero = L.calibration_power_gate(0, 100, power_floor=0.60)
    assert zero['status'] == 'fail', zero
    assert L.calibration_power_gate(100, 100, power_floor=0.60)['status'] == 'ok'
    assert L.calibration_power_gate(60, 100, power_floor=0.60)['status'] == 'ok'
    for bad_args in ((-1, 10), (11, 10), (1.5, 10), (True, 10), (1, 10.5), (1, 0)):
        try:
            L.calibration_type_i_gate(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid calibration counts accepted: {bad_args!r}')
    for bad_kw in ({'alpha': True}, {'alpha': 0.0}, {'alpha': 1.0},
                   {'control_level': 0.0}, {'warning_level': 0.0001},
                   {'control_level': 0.02, 'warning_level': 0.01}):
        try:
            L.calibration_type_i_gate(5, 100, **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid calibration gate parameter accepted: {bad_kw!r}')
    # The suite is deterministic and must not consume or reseed the global RNG.
    before = np.random.get_state()
    r1 = L.statistical_calibration_suite(n_sim=120)
    after = np.random.get_state()
    assert before[0] == after[0], 'calibration changed the global RNG algorithm'
    assert np.array_equal(before[1], after[1]), 'calibration consumed the global RNG state'
    assert before[2:] == after[2:], 'calibration changed the global RNG position'
    r2 = L.statistical_calibration_suite(n_sim=120)
    assert r1['status'] == 'ok', r1
    assert r1['tests'] == r2['tests'], 'calibration is not deterministic'
    assert r1['policy_fp'] == r2['policy_fp'] == L._calibration_policy_fp(r1['policy']), r1
    assert r1['policy']['n_sim'] == 120 and r1['policy']['simulation_seed'] == 20261023, r1
    assert L._calibration_policy_fp(L.statistical_calibration_suite(n_sim=40)['policy']) != r1['policy_fp']
    expected = {'exact_signflip_enumeration', 'exact_signflip_mitm',
                'saddlepoint_vs_exact', 'saddlepoint_tail',
                'sequential_looks', 'tost_paired', 'holm_family',
                'interval_seed_t', 'interval_paired_bootstrap',
                'interval_unit_bootstrap', 'interval_combined_seed_doc'}
    assert set(r1['tests']) == expected, sorted(r1['tests'])
    for name, rec in r1['tests'].items():
        if name == 'saddlepoint_vs_exact':
            assert rec['status'] == 'ok', rec
            assert rec['max_abs_error'] <= rec['max_abs_tolerance'], rec
            assert rec['mean_abs_error'] <= rec['mean_abs_tolerance'], rec
        elif 'coverage_control' in rec:
            cov, wid = rec['coverage_control'], rec['width_control']
            assert cov['status'] in ('ok', 'warnings'), (name, rec)
            assert wid['status'] == 'ok', (name, rec)
            assert 0.0 <= cov['rate'] <= 1.0, rec
            assert 0.0 < wid['mean_width'] <= wid['max_width'], rec
        else:
            for part in ('null_control', 'positive_control'):
                assert rec[part]['status'] in ('ok', 'warnings'), (name, rec)
                assert 0.0 <= rec[part]['rate'] <= 1.0, rec
            if 'methods' in rec:
                assert rec.get('path_ok') is True, (name, rec)
    for bad in ({'n_sim': 19}, {'n_sim': 1.5}, {'seed': -1},
                {'seed': True}, {'power_floor': 1.0},
                {'control_level': 0.0}, {'warning_level': 0.0001}):
        kw = {'n_sim': 40}
        kw.update(bad)
        try:
            L.statistical_calibration_suite(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid suite parameter accepted: {bad!r}')


def test_statistical_calibration_is_wired_into_the_audit():
    """L184 — the calibration is reachable from the audit and the report."""
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_statistical_calibration(n_sim=100, seed=7)
    assert rep['status'] == 'ok', rep
    assert rep['kind'] == 'statistical_calibration' and rep['deterministic'] is True
    assert rep['seed'] == 7 and rep['n_sim'] == 100
    assert set(rep['violations']) == set(), rep
    for name in ('exact_signflip_enumeration', 'exact_signflip_mitm',
                 'saddlepoint_tail', 'sequential_looks', 'tost_paired',
                 'holm_family'):
        assert rep['tests'][name]['null_control']['rate'] <= 0.10, (name, rep)
        assert rep['tests'][name]['positive_control']['rate'] >= 0.60, (name, rep)
    for name in ('interval_seed_t', 'interval_paired_bootstrap',
                 'interval_unit_bootstrap', 'interval_combined_seed_doc'):
        _rec = rep['tests'][name]
        assert _rec['coverage_control']['status'] in ('ok', 'warnings'), (name, rep)
        assert _rec['width_control']['status'] == 'ok', (name, rep)
    import inspect
    src = inspect.getsource(A.run_report)
    assert "'calibration': audit_statistical_calibration()" in src, src
    assert 'calibration' in (A.__doc__ or ''), 'calibration is not documented in the audit CLI'
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            A.audit_statistical_calibration(n_sim=19)
    except ValueError:
        pass
    else:
        raise AssertionError('audit accepted an under-sized calibration simulation')


def test_interval_calibration_gates_and_bootstrap():
    """L186 — interval coverage is calibrated and vacuous widths are rejected."""
    # Coverage gates: nominal passes; a gross shortfall fails; over-coverage
    # is conservative and passes; malformed counts/levels fail closed.
    ok = L.calibration_interval_coverage_gate(950, 1000, nominal=0.95)
    assert ok['status'] == 'ok' and abs(ok['rate'] - 0.95) < 1e-12, ok
    bad = L.calibration_interval_coverage_gate(700, 1000, nominal=0.95)
    assert bad['status'] == 'fail' and bad['p_under_coverage'] < 0.001, bad
    conservative = L.calibration_interval_coverage_gate(990, 1000, nominal=0.95)
    assert conservative['status'] == 'ok', conservative
    for bad_args in ((-1, 10), (11, 10), (1.5, 10), (True, 10), (1, 10.5)):
        try:
            L.calibration_interval_coverage_gate(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid coverage counts accepted: {bad_args!r}')
    for bad_kw in ({'nominal': True}, {'nominal': 0.0}, {'nominal': 1.0},
                   {'control_level': 0.0}, {'warning_level': 0.0001},
                   {'control_level': 0.02, 'warning_level': 0.01}):
        try:
            L.calibration_interval_coverage_gate(95, 100, **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid coverage gate parameter accepted: {bad_kw!r}')
    # Width gate: a finite positive width under the cap passes; zero, negative,
    # NaN/inf and over-cap widths fail.  Coverage alone would accept them.
    assert L.calibration_interval_width_gate(1.0, 2.0)['status'] == 'ok'
    for bad_w in (0.0, -1.0, float('nan'), float('inf'), 2.0001):
        assert L.calibration_interval_width_gate(bad_w, 2.0)['status'] == 'fail', bad_w
    assert L.calibration_interval_width_gate(True, 2.0)['status'] == 'fail'
    # The suite now calibrates interval constructors on both properties.
    r = L.statistical_calibration_suite(n_sim=80)
    assert r['status'] == 'ok', r
    for name in ('interval_seed_t', 'interval_paired_bootstrap',
                 'interval_unit_bootstrap', 'interval_combined_seed_doc'):
        rec = r['tests'][name]
        assert rec['coverage_control']['status'] in ('ok', 'warnings'), rec
        assert rec['width_control']['status'] == 'ok', rec
        assert 0.0 < rec['width_control']['mean_width'] <= rec['width_control']['max_width'], rec
    # The paired and unit bootstrap intervals are studentized, deterministic,
    # and fall back explicitly on a degenerate (zero-variance) input.
    d = [0.1, -0.2, 0.3, 0.05, 0.15, -0.05]
    p1 = L.paired_bootstrap_ci(d, n_boot=200, seed=11)
    p2 = L.paired_bootstrap_ci(d, n_boot=200, seed=11)
    assert p1 == p2 and p1[0] < p1[1], (p1, p2)
    unit = L.bootstrap_mean_ci([0.1, -0.2, 0.3, 0.05, 0.15, -0.05],
                               n_boot=200, seed=11)
    assert 'studentized' in unit['method'], unit
    assert unit['n_boot_studentized'] > 0, unit
    const = L.bootstrap_mean_ci([2.0] * 8, n_boot=100, seed=11)
    assert const['ci_low'] == const['ci_high'] == 2.0, const
    assert 'fallback' in const['method'], const


def test_analysis_choice_concordance_and_audit():
    """L188 — multi-specification agreement is explicit, replayed and caveated."""
    primary = {'spec': 'seed_mean', 'unit': 'seed', 'scale': 'ppl_delta',
               'estimand': 'mean (a-b)', 'point': -1.0, 'ci95': [-2.0, -0.5]}
    stable_alts = [
        {'spec': 'seed_median', 'unit': 'seed', 'scale': 'ppl_delta',
         'point': -0.5, 'ci95': [-1.5, -0.1]},
        {'spec': 'unit_log_ratio', 'unit': 'article', 'scale': 'log_ratio',
         'point': -0.2, 'ci95': [-0.4, -0.05]},
    ]
    stable = L.analysis_choice_concordance(primary, stable_alts)
    assert stable['status'] == 'stable', stable
    assert stable['direction_stable'] is True and stable['fragile'] is False, stable
    assert stable['precision_concordant'] is True, stable
    assert stable['n_specs'] == 3 and stable['n_available'] == 3, stable
    assert stable['n_same_as_primary'] == 3 and stable['n_opposite_to_primary'] == 0, stable
    assert stable['n_excluding_zero'] == 3 and stable['n_crossing_zero'] == 0, stable
    er = stable['effect_range_ppl']
    assert er['min'] == -1.0 and er['max'] == -0.5 and er['n'] == 2, stable
    assert er['relative_span'] == 0.5, stable
    # A PPL-delta range must not be divided by a primary point on another
    # scale (log-ratio here): that would publish a meaningless mixed-unit ratio.
    mixed_scale = L.analysis_choice_concordance(
        {'spec': 'unit_log_ratio', 'unit': 'article', 'scale': 'log_ratio',
         'point': -0.2, 'ci95': [-0.4, -0.05]},
        [{'spec': 'seed_mean', 'unit': 'seed', 'scale': 'ppl_delta',
          'point': -1.0},
         {'spec': 'unit_mean', 'unit': 'article', 'scale': 'ppl_delta',
          'point': -1.5}])
    _mser = mixed_scale['effect_range_ppl']
    assert _mser['min'] == -1.5 and _mser['max'] == -1.0, mixed_scale
    assert _mser['span'] == 0.5 and _mser['relative_span'] is None, mixed_scale
    assert not any(k.startswith('p_') for k in stable), stable
    # A second interval that crosses zero while the direction still agrees is
    # precision sensitivity, not a direction contradiction.
    precision = L.analysis_choice_concordance(
        primary, [{'spec': 'unit_mean', 'point': -0.2, 'ci95': [-1.0, 0.5],
                   'scale': 'ppl_delta', 'unit': 'article'}])
    assert precision['status'] == 'precision_sensitive', precision
    assert precision['direction_stable'] is True and precision['fragile'] is True, precision
    assert precision['precision_concordant'] is False, precision
    assert precision['crossing_zero_specs'] == ['unit_mean'], precision
    # Opposite point estimates are a hard direction contradiction even when
    # both intervals exclude zero.
    unstable = L.analysis_choice_concordance(
        primary, [{'spec': 'unit_mean', 'point': 0.5, 'ci95': [0.1, 1.0],
                   'scale': 'ppl_delta', 'unit': 'article'}])
    assert unstable['status'] == 'direction_unstable', unstable
    assert unstable['n_positive'] == 1 and unstable['n_negative'] == 1, unstable
    assert unstable['n_opposite_to_primary'] == 1, unstable
    # Zero and unavailable specifications are named, not silently dropped.
    zero = L.analysis_choice_concordance(
        {'spec': 'seed_mean', 'point': 0.0, 'ci95': [-1.0, 1.0]},
        [{'spec': 'unit_mean', 'point': 0.0},
         {'spec': 'trimmed', 'available': False, 'reason': 'n<5'}])
    assert zero['status'] == 'zero_effect', zero
    assert zero['n_available'] == 2 and zero['n_unavailable'] == 1, zero
    assert zero['n_zero'] == 2, zero
    zero_primary = L.analysis_choice_concordance(
        {'spec': 'seed_mean', 'point': 0.0},
        [{'spec': 'unit_mean', 'point': 1.0}])
    assert zero_primary['status'] == 'direction_unstable', zero_primary
    assert zero_primary['n_opposite_to_primary'] == 1, zero_primary
    thin = L.analysis_choice_concordance(
        primary, [{'spec': 'trimmed', 'available': False, 'reason': 'n<5'}])
    assert thin['status'] == 'insufficient_specs' and thin['n_available'] == 1, thin
    # A CI-less specification still contributes to direction but not to the
    # precision concordance; one CI alone is not enough to claim agreement.
    sign_only = L.analysis_choice_concordance(
        primary, [{'spec': 'unit_median', 'point': -0.1}])
    assert sign_only['status'] == 'stable', sign_only
    assert sign_only['n_with_ci'] == 1 and sign_only['precision_concordant'] is None, sign_only
    # Fail closed: malformed rows, duplicate names and a non-finite primary
    # cannot produce a reassuring status.
    for bad_args in (
            (primary, [{'spec': 'x', 'point': -0.1}, {'spec': 'x', 'point': -0.2}]),
            (primary, [{'spec': 'x', 'point': True}]),
            (primary, [{'spec': 'x', 'point': -0.1, 'ci95': [1.0, -1.0]}]),
            (primary, [{'spec': 'x', 'point': -0.1, 'ci95': [0.0]}]),
            (primary, [{'spec': 'x', 'point': '-0.1'}]),
            (primary, [{'spec': 'x', 'point': -0.1,
                        'ci95': ['-0.2', '-0.05']}]),
            (primary, [{'spec': 'x', 'point': -0.1, 'unit': 3}]),
            (primary, [{'spec': 'x', 'point': -0.1,
                        'estimand': {'not': 'a label'}}]),
            (primary, [{'available': 'no', 'point': -0.1}]),
            ({'spec': 'p', 'available': False}, []),
            ({'spec': 'p', 'point': float('nan')}, []),
            ('not-a-dict', []),
            (primary, 'not-an-iterable')):
        try:
            L.analysis_choice_concordance(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed analysis choice {bad_args!r}')
    for bad_zt in (True, -1.0, float('nan'), float('inf')):
        try:
            L.analysis_choice_concordance(primary, [], zero_tol=bad_zt)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted zero_tol={bad_zt!r}')
    snapshot = json.dumps([primary, stable_alts], sort_keys=True)
    L.analysis_choice_concordance(primary, stable_alts)
    assert json.dumps([primary, stable_alts], sort_keys=True) == snapshot, \
        'concordance mutated its inputs'

    def _stats(cc, mean=-1.0):
        return {'comparisons': {'x': {
            'mean': mean,
            'doc_sampling': {'analysis_choice_concordance': cc}}}}

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')

        def _audit(obj):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(obj, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_analysis_choices(path=p)

        ok = _audit(_stats(stable))
        assert ok['status'] == 'ok' and ok['n_ok'] == 1, ok
        assert ok['n_fragile'] == 0 and not ok['problems'], ok
        unstable_cc = L.analysis_choice_concordance(
            primary, [{'spec': 'unit_mean', 'point': 0.5,
                       'ci95': [0.1, 1.0]}])
        warn = _audit(_stats(unstable_cc))
        assert warn['status'] == 'warnings' and warn['n_fragile'] == 1, warn
        assert any('disagree on direction' in w for w in warn['warnings']), warn
        bad_status = json.loads(json.dumps(_stats(stable)))
        bad_status['comparisons']['x']['doc_sampling'][
            'analysis_choice_concordance']['status'] = 'precision_sensitive'
        bad = _audit(bad_status)
        assert bad['status'] == 'fail', bad
        assert any('does not reproduce' in x for x in bad['problems']), bad
        bad_mean = _audit(_stats(stable, mean=-2.0))
        assert bad_mean['status'] == 'fail', bad_mean
        assert any('frozen per-seed mean' in x for x in bad_mean['problems']), bad_mean
        # Type-blind replay is not a reproduction: an aggregate flag written as
        # 1 and a numeric string mean must both fail.
        bool_aggregate = json.loads(json.dumps(_stats(stable)))
        bool_aggregate['comparisons']['x']['doc_sampling'][
            'analysis_choice_concordance']['direction_stable'] = 1
        bool_bad = _audit(bool_aggregate)
        assert bool_bad['status'] == 'fail', bool_bad
        assert any('direction_stable' in x for x in bool_bad['problems']), bool_bad
        string_mean = _audit(_stats(stable, mean='-1.0'))
        assert string_mean['status'] == 'fail', string_mean
        assert any('frozen mean is missing' in x for x in string_mean['problems']), string_mean
        stale = _audit({'comparisons': {'x': {
            'mean': -1.0, 'doc_sampling': {'per_doc_delta_mean': -1.0}}}})
        assert stale['status'] == 'not_verifiable_locally', stale
        assert stale['n_missing'] == 1 and stale['warnings'], stale
        no_ds = _audit({'comparisons': {'x': {'mean': -1.0}}})
        assert no_ds['status'] == 'not_verifiable_locally', no_ds
        assert no_ds['n_missing'] == 1 and no_ds['warnings'], no_ds
        corrupt = _audit({'comparisons': {'x': {
            'mean': -1.0,
            'doc_sampling': {'analysis_choice_concordance': ['bad']}}}})
        assert corrupt['status'] == 'fail', corrupt
        assert any('not a JSON object' in x for x in corrupt['problems']), corrupt
        assert corrupt['n_missing'] == 0, corrupt
        # A non-object comparison is corrupt, not silently absent.
        nonobj = _audit({'comparisons': {'x': ['not', 'an', 'object']}})
        assert nonobj['status'] == 'fail', nonobj
        assert any('not an object' in x for x in nonobj['problems']), nonobj
        # Replay from the stored rows cannot re-derive every label, but a
        # self-inconsistent derived field must still fail rather than be
        # hidden by the aggregate counts.
        bad_row = json.loads(json.dumps(_stats(stable)))
        bad_row['comparisons']['x']['doc_sampling'][
            'analysis_choice_concordance']['specifications'][1][
                'is_primary'] = True
        row_bad = _audit(bad_row)
        assert row_bad['status'] == 'fail', row_bad
        assert any('specification row 1' in x for x in row_bad['problems']), row_bad
        unhashable_name = json.loads(json.dumps(_stats(stable)))
        unhashable_name['comparisons']['x']['doc_sampling'][
            'analysis_choice_concordance']['specifications'][1]['spec'] = []
        name_bad = _audit(unhashable_name)
        assert name_bad['status'] == 'fail', name_bad
        assert any('specification names' in x for x in name_bad['problems']), name_bad
        bad_specs = json.loads(json.dumps(_stats(stable)))
        bad_specs['comparisons']['x']['doc_sampling'][
            'analysis_choice_concordance']['specifications'][1]['spec'] = \
            bad_specs['comparisons']['x']['doc_sampling'][
                'analysis_choice_concordance']['specifications'][0]['spec']
        dup = _audit(bad_specs)
        assert dup['status'] == 'fail', dup
        gone = os.path.join(td, 'none.json')
        with contextlib.redirect_stdout(io.StringIO()):
            absent = A.audit_analysis_choices(path=gone)
        assert absent['status'] == 'not_verifiable_locally', absent
    import inspect
    assert 'analysis_choices' in (A.__doc__ or ''), 'analysis-choice audit is not documented'
    assert "'analysis_choices': audit_analysis_choices()" in \
        inspect.getsource(A.run_report), 'analysis-choice audit is not wired into the report'
    assert 'analysis_choices' in inspect.getsource(A), 'analysis-choice CLI is not registered'

    # Integration: a distribution where the token-pooled mean is positive but
    # the equal-weighted median is negative must be reported as a direction
    # contradiction, not hidden behind either summary.
    docs = ['d0', 'd1', 'd2', 'd3']
    toks = [10, 10, 10, 10]
    nll_a = [1.0, 1.0, 1.0, 20.0]
    nll_b = [2.0, 2.0, 2.0, 2.0]
    recs_a = {s: {'by_len': {512: _document_cell(nll_a, toks, docs)}}
              for s in (0, 1)}
    recs_b = {s: {'by_len': {512: _document_cell(nll_b, toks, docs)}}
              for s in (0, 1)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1], 512)
    cc = got.get('analysis_choice_concordance')
    assert isinstance(cc, dict), got
    assert cc['status'] == 'direction_unstable', cc
    assert cc['n_specs'] == 8 and cc['n_available'] == 7, cc
    assert cc['primary_sign'] == 1 and cc['n_opposite_to_primary'] >= 1, cc
    assert any(s['spec'] == 'unit_median_ppl_delta' and s['sign'] == -1
               for s in cc['specifications']), cc
    json.dumps(cc, allow_nan=False)
    md = '\n'.join(V12._analysis_choice_markdown(
        {'comparisons': {'x': {'doc_sampling': got}}}))
    assert '分析口径一致性' in md and '方向翻转' in md, md
    assert '不改变上面的冻结判定' in md and '不能用来挑选最有利口径' in md, md
    assert '不能把结论描述为' in md, md
    flat_cell = _document_cell([2.0, 2.0, 2.0, 2.0], toks, docs)
    flat_a = {s: {'by_len': {512: dict(flat_cell)}} for s in (0, 1)}
    flat_b = {s: {'by_len': {512: dict(flat_cell)}} for s in (0, 1)}
    flat = V12._doc_level_analysis(flat_a, flat_b, [0, 1], 512)
    fcc = flat.get('analysis_choice_concordance')
    assert fcc['status'] == 'zero_effect', fcc
    _wb = next(s for s in fcc['specifications']
               if s['spec'] == 'unit_win_balance_oriented')
    assert _wb['point'] == 0.0, _wb
    assert V12._analysis_choice_markdown({}) == []
    miss_md = V12._analysis_choice_markdown(
        {'comparisons': {'x': {'doc_sampling': {'per_doc_delta_mean': -1.0}}}})
    assert any('未验证' in line and 'x' in line for line in miss_md), miss_md
    # A corrupt external artifact must render as unverified, never crash the
    # report while trying to describe it.
    corrupt_md = '\n'.join(V12._analysis_choice_markdown(
        {'comparisons': {'x': {'doc_sampling': {
            'analysis_choice_concordance': {
                'status': 'stable', 'primary_sign': 'not-a-number',
                'effect_range_ppl': ['not', 'an', 'object'],
                'specifications': 3}}}}}))
    assert '未验证' in corrupt_md and 'x' in corrupt_md, corrupt_md
    # A string "false" is not a boolean flag: it must not be rendered as a
    # fragility warning, and the corrupt block must be marked unverified.
    false_fragile = json.loads(json.dumps(stable))
    false_fragile['fragile'] = 'false'
    ff_md = '\n'.join(V12._analysis_choice_markdown(
        {'comparisons': {'x': {'doc_sampling': {
            'analysis_choice_concordance': false_fragile}}}}))
    assert '未验证' in ff_md, ff_md
    assert '**解读提示**' not in ff_md and '依赖分析口径' not in ff_md, ff_md


def test_wikitest_missing_train_source_fails_with_runtime_error():
    """A dataset loader returning None is a missing source, not UnboundLocal."""
    import unittest.mock as mock
    old = os.environ.get('WT103_RAW_TXT')
    with tempfile.TemporaryDirectory() as td:
        os.environ['WT103_RAW_TXT'] = os.path.join(td, 'absent_raw')
        try:
            with mock.patch('datasets.load_dataset', return_value=None):
                try:
                    L.load_wikitest(8, n_seqs=1, vocab_size=64,
                                    cache_dir=os.path.join(td, 'cache'))
                except RuntimeError as e:
                    assert 'no train source' in str(e), e
                else:  # pragma: no cover
                    raise AssertionError('a missing HF train split was accepted')
        finally:
            if old is None:
                os.environ.pop('WT103_RAW_TXT', None)
            else:
                os.environ['WT103_RAW_TXT'] = old


def test_release_identity_binds_measurement_source():
    """A release must carry the exact measurement source it was built from.

    The archive used to record the git HEAD only.  A dirty tree, a moved
    branch, or a source edit made after a run could therefore leave an
    "auditable" release whose code cannot be reconstructed.  `make` now packs
    the source-tree manifest's files and `verify` recomputes the digest from
    those packed bytes, independently of exp_lib and of Git.
    """
    import archive_release as AR
    import tarfile
    with tempfile.TemporaryDirectory(dir=REPO) as td:
        rel = os.path.basename(td)
        results = os.path.join(td, 'results_x')
        os.makedirs(results)
        with open(os.path.join(results, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'csa_fixed::seed0': {'ppl': 10.0}}, f)
        out = os.path.join(rel, 'archive')
        assert AR.cmd_make('identity_test', [os.path.join(rel, 'results_x')],
                           outdir=out) == 0
        path = os.path.join(out, 'release_identity_test.tar.gz')
        assert os.path.isfile(path)
        with tarfile.open(path, 'r:gz') as tf:
            man_name = next(n for n in tf.getnames()
                            if n.endswith('.MANIFEST.sha256'))
            man = json.load(tf.extractfile(man_name))
        ident = man['identity']
        assert ident['schema'] == AR._SOURCE_IDENTITY_SCHEMA, ident
        assert ident['code_semantics'] == L.CODE_SEMANTICS, ident
        st = ident['source_tree']
        assert st['sha256'] == ident['source_tree_sha256'], st
        assert AR._release_identity_problems(man) == [], AR._release_identity_problems(man)
        for rel_src, meta in st['files'].items():
            assert rel_src in man['files'], f'source file not packed: {rel_src}'
            assert man['files'][rel_src]['sha256'] == meta['sha256'], rel_src
            assert man['files'][rel_src]['bytes'] == meta['bytes'], rel_src
        assert 'exp_lib.py' in st['files'] and 'archive_release.py' in st['files'], st
        assert any(k.endswith('results_x/summary.json')
                   for k in man['files']), man['files'].keys()
        assert AR.cmd_verify(path) == 0
        # Identity is a binding, not a label: a source file missing from the
        # packed map or a wrong aggregate digest must be refused.
        bad = json.loads(json.dumps(man))
        bad['files'].pop('exp_lib.py')
        assert any('not packed' in p for p in AR._release_identity_problems(bad)), bad
        bad2 = json.loads(json.dumps(man))
        bad2['identity']['source_tree_sha256'] = '0' * 64
        assert AR._release_identity_problems(bad2), bad2
        bad3 = json.loads(json.dumps(man))
        bad3['identity']['code_semantics'] = ''
        assert any('code_semantics' in p for p in AR._release_identity_problems(bad3)), bad3
        # End-to-end: tampering with the identity block makes verify fail even
        # though every packed file hash still matches.
        tampered = path + '.tampered.tar.gz'
        with tarfile.open(path, 'r:gz') as src, tarfile.open(tampered, 'w:gz') as dst:
            for member in src.getmembers():
                data = src.extractfile(member)
                if member.name == man_name:
                    _m = json.loads(data.read())
                    _m['identity']['source_tree_sha256'] = '0' * 64
                    _b = json.dumps(_m).encode('utf-8')
                    member.size = len(_b)
                    dst.addfile(member, io.BytesIO(_b))
                else:
                    dst.addfile(member, data)
        assert AR.cmd_verify(tampered) == 1


def test_train_val_passage_overlap_gate():
    rng = np.random.default_rng(20261025)
    tr = rng.integers(0, 100, size=200000, dtype=np.int64)
    va = rng.integers(0, 100, size=(16, 128), dtype=np.int64)
    kw = dict(dataset='wikitext', max_scan_tokens=100000, sample_rows=16,
              strict=False)
    rep = L.audit_corpus_split(tr, va, 100, **kw)
    po = rep['passage_overlap']
    assert rep['status'] == 'ok', rep
    assert po['status'] == 'ok' and po['val_shingle_overlap'] == 0.0, po
    assert po['train_tokens_scanned_fraction'] >= 0.25, po
    # The memo key must include the new policy knobs, not only token bytes.
    rep2 = L.audit_corpus_split(
        tr, va, 100, dataset='wikitext', max_scan_tokens=100000,
        sample_rows=16, strict=False, min_scan_fraction=0.01,
        max_val_shingle_overlap=0.99)
    assert rep2['passage_overlap']['policy']['max_val_shingle_overlap'] == 0.99, rep2
    assert rep2['passage_overlap']['policy']['min_scan_fraction'] == 0.01, rep2

    # A contiguous train passage copied into val, with one edit so the exact
    # row scan stays clean.  The near-duplicate gate must still catch it.
    near = va.copy()
    near[4] = tr[1000:1128]
    near[4, 64] = (near[4, 64] + 1) % 100
    rep = L.audit_corpus_split(tr, near, 100, **kw)
    po = rep['passage_overlap']
    assert rep['exact_row_overlap_in_sample'] == 0, rep
    assert po['status'] == 'fail', po
    assert po['max_val_row_overlap'] > 0.5, po
    assert rep['status'] == 'fail', rep
    assert any('passage overlap' in h for h in rep['hard_failures']), rep

    # Exact copy: both the row gate and the passage gate fire.
    exact = va.copy()
    exact[4] = tr[1000:1128]
    rep = L.audit_corpus_split(tr, exact, 100, **kw)
    assert rep['status'] == 'fail', rep
    assert rep['passage_overlap']['status'] == 'fail', rep

    # Recurring corpus boilerplate is not split contamination: the
    # train-internal baseline is high and the excess is zero.
    b0 = np.arange(1000, 1064, dtype=np.int64)
    b1 = np.arange(2000, 2064, dtype=np.int64)
    b2 = np.arange(3000, 3064, dtype=np.int64)
    boiler_train = np.tile(np.concatenate([b0, b1, b2]), 2000)[:200000]
    boiler_val = np.tile(b0, (16, 2))
    rep = L.audit_corpus_split(boiler_train, boiler_val, 5000, **kw)
    po = rep['passage_overlap']
    assert po['train_internal_shingle_overlap'] > 0.9, po
    assert po['excess_over_val_baseline'] == 0.0, po
    assert po['status'] == 'ok', po
    assert rep['exact_row_overlap_in_sample'] == 0, rep

    # A clean sample over a small train fraction must not be certified.
    low = L.audit_corpus_split(tr, va, 100, dataset='wikitext',
                               max_scan_tokens=10000, sample_rows=16,
                               strict=False)
    assert low['train_tokens_scanned_fraction'] < 0.25, low
    assert low['exact_row_scan_sufficient'] is False, low
    assert low['passage_overlap']['status'] == 'not_verifiable', low
    assert low['status'] == 'warnings', low
    assert any('clean sample is not a clean-split proof' in w
               for w in low['warnings']), low

    # Rows shorter than the shingle width are explicitly not applicable, not
    # silently clean.
    short = L.audit_corpus_split(
        np.arange(1000, dtype=np.int64),
        np.arange(1000, 1008, dtype=np.int64).reshape(2, 4), 2000,
        dataset='wikitext', max_scan_tokens=1000, sample_rows=2, strict=False)
    assert short['passage_overlap']['status'] == 'not_applicable', short

    # The offline split audit must report the same statistic and fail closed.
    with tempfile.TemporaryDirectory() as d:
        tag = 'passage'
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        meta = {'vocab': 100, 'seq_len': 128, 'split_strategy': 'ordered',
                'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                       max_passage_scan_tokens=100000)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['passage_overlap']['status'] == 'ok', rec
        assert out['status'] == 'ok', out
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), exact)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                       max_passage_scan_tokens=100000)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['passage_overlap']['status'] == 'fail', rec
        assert rec['status'] == 'fail', rec
        assert any('passage overlap' in w for w in out['warnings']), out
        # A clean scan over a small train fraction is not certified offline
        # either; the per-cache status must not stay 'ok'.
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                       max_passage_scan_tokens=10000)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['passage_overlap']['status'] == 'not_verifiable', rec
        assert rec['status'] == 'warnings', rec

    # NaN/negative policy values must not silently disable the new gate.
    for kw_bad in ({'passage_shingle': 0}, {'passage_shingle': True},
                   {'min_scan_fraction': float('nan')},
                   {'max_val_shingle_overlap': -0.1},
                   {'max_val_row_shingle_overlap': 1.5},
                   {'max_passage_excess_overlap': 2.0}):
        try:
            L.audit_corpus_split(tr, va, 100, strict=False, **kw_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_corpus_split accepted {kw_bad!r}')
    for kw_bad in ({'passage_shingle': 0},
                   {'min_passage_scan_fraction': float('nan')},
                   {'max_val_shingle_overlap': -1.0},
                   {'max_val_row_shingle_overlap': 1.5},
                   {'max_passage_excess_overlap': 2.0}):
        try:
            A.audit_split_discipline(cache_dirs=[], **kw_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_split_discipline accepted {kw_bad!r}')


def test_student_t_large_df_tail():
    z = 1.959963984540054
    assert abs(L._student_t_sf(z, 1e12) - 0.025000000000138661) < 5e-15
    assert abs(L._student_t_sf(z, 1e9) - 0.025000000138647571) < 5e-14
    assert abs(L._student_t_sf(1.0, 1e8) - 0.15865525514131067) < 5e-14
    assert abs(L._student_t_sf(2.0, 1e12) - 0.022750131948314185) < 5e-15
    assert abs(L._t_critical(0.95, 1e12) - 1.9599639845424261) < 1e-14
    assert abs(L._t_critical(0.95, 1e9) - 1.9599639869123251) < 1e-13
    assert L._student_t_sf(0.0, 1e12) == 0.5
    assert abs(L._student_t_sf(-2.0, 1e12)
               - (1.0 - L._student_t_sf(2.0, 1e12))) < 1e-15
    assert abs(L._student_t_sf(20.0, 1e20) - 2.7536241186063325e-89) < 1e-100
    assert abs(L._student_t_sf(20.0, float('inf'))
               - L._student_t_sf(20.0, 1e20)) < 1e-100
    # The 1/df expansion is only switched on once its neglected term is below
    # the beta path's error.  At df=1e4, |t|=10 the old dispatch used the
    # expansion and was ~1e-4 relative low; the beta path is ~1e-11 there.
    for _df, _ref in ((1e4, 9.8164037143314743e-24),
                      (1e5, 7.8165076501170029e-24)):
        _got = L._student_t_sf(10.0, _df)
        assert abs(_got / _ref - 1.0) < 1e-8, (_df, _got, _ref)
    # In the 1e7 <= df < 1e15 / |t| > 10 region the incomplete-beta fallback
    # loses up to ~2e-2 relative accuracy; the 1/df expansion must be used
    # there too (reference values from high-precision mpmath).
    assert abs(L._student_t_sf(11.0, 1e10)
               - 1.9106602853626618e-28) < 1e-40
    assert abs(L._student_t_sf(20.0, 1e12)
               - 2.7536242293012506e-89) < 1e-100
    assert abs(L._student_t_sf(11.0, 1e14)
               - 1.9106595745697621e-28) < 1e-40
    assert abs(L._student_t_sf(15.0, 1e8)
               - 3.6714349561714935e-51) < 1e-60
    assert abs(L._student_t_sf(20.0, 1e7)
               - 2.7647156034788507e-89) < 1e-99


def test_paired_design_sensitivity_planning():
    base = L.paired_design_sensitivity(
        12, 6, alpha=0.05, target_power=0.80,
        pilot_deltas=[-1.0] * 10 + [-0.5, -2.0], expected_sign='negative')
    assert base['status'] == 'ok', base['violations']
    assert base['n_seeds'] == 12 and base['m_family'] == 6
    assert abs(base['p_floor'] - L.signflip_p_floor(12)) < 1e-15, base['p_floor']
    assert abs(base['min_attainable_p_holm']
               - L.holm_min_attainable_p(12, 6, 0.05)) < 1e-15
    assert base['pilot']['q_sign_rate'] == 1.0, base['pilot']
    assert base['pilot']['direction_ok'] is True, base['pilot']
    assert 0.9 < base['min_q_for_target_power'] < 0.99, base['min_q_for_target_power']
    assert abs(base['power_curve']['q=0.950']
               - L.signflip_power(12, 6, 0.95, 0.05)) < 1e-15
    assert base['mde']['absolute'] > 0.0, base['mde']
    assert base['precision']['ci95_halfwidth'] > 0.0, base['precision']
    assert base['pilot']['required_n_for_target_power'] <= 12, base['pilot']

    tight = L.paired_design_sensitivity(
        12, 6, target_power=0.80, pilot_sd=1.0, target_halfwidth=0.1)
    assert tight['status'] == 'ok', tight['violations']
    assert tight['precision']['meets_target'] is False, tight['precision']
    assert tight['precision']['n_required_for_target_halfwidth'] > 12, tight['precision']
    loose = L.paired_design_sensitivity(
        12, 6, target_power=0.80, pilot_sd=1.0, target_halfwidth=1.0)
    assert loose['precision']['meets_target'] is True, loose['precision']
    assert loose['precision']['n_required_for_target_halfwidth'] <= 12, loose['precision']

    # The CI half-width depends on sd/n only.  A malformed target_power must
    # not suppress the precision block or its required-n readout.
    bad_tp = L.paired_design_sensitivity(
        12, 6, target_power=1.0, pilot_sd=1.0, target_halfwidth=0.5)
    assert bad_tp['status'] == 'fail', bad_tp
    assert bad_tp['precision']['ci95_halfwidth'] > 0.0, bad_tp['precision']
    assert (bad_tp['precision']['n_required_for_target_halfwidth']
            is not None), bad_tp['precision']

    # The half-width search is binary; verify it returns the true minimum and
    # handles the n_max boundary without enumerating every n.
    for _sd, _th, _cap in ((1.0, 0.1, 512), (2.5, 0.05, 1000),
                           (0.3, 1.0, 64)):
        _got = L._n_for_target_halfwidth(_sd, _th, n_max=_cap)
        _want = None
        for _n in range(2, _cap + 1):
            _tc = L._t_critical(0.95, float(_n - 1))
            if _tc * _sd / math.sqrt(_n) <= _th:
                _want = _n
                break
        assert _got == _want, (_sd, _th, _cap, _got, _want)
    _edge = L._t_critical(0.95, 9999.0) / 100.0
    assert L._n_for_target_halfwidth(1.0, _edge, n_max=10000) == 10000
    assert L._n_for_target_halfwidth(1.0, _edge * 0.999, n_max=10000) is None

    big = L.paired_design_sensitivity(
        12, 6, target_power=0.80, pilot_sd=1.0, expected_sign='positive',
        sesoi=10.0, require_sesoi_power=True)
    assert big['status'] == 'ok' and big['sesoi']['power_ok'] is True, big
    small = L.paired_design_sensitivity(
        12, 6, target_power=0.80, pilot_sd=1.0, expected_sign='positive',
        sesoi=0.01, require_sesoi_power=True)
    assert small['status'] == 'fail', small
    assert small['sesoi']['status'] == 'underpowered', small['sesoi']
    assert any('underpowered at the pre-registered SESOI' in v
               for v in small['violations']), small['violations']
    advisory = L.paired_design_sensitivity(
        12, 6, target_power=0.80, pilot_sd=1.0, expected_sign='positive',
        sesoi=0.01)
    assert advisory['status'] == 'ok' and advisory['sesoi']['status'] == 'underpowered', advisory
    assert any('SESOI' in w for w in advisory['warnings']), advisory['warnings']

    nd = L.paired_design_sensitivity(
        12, 6, pilot_deltas=[-1.0, 1.0, -2.0], expected_sign=None, sesoi=1.0)
    assert nd['status'] == 'ok' and nd['sesoi']['status'] == 'no_direction', nd
    bare = L.paired_design_sensitivity(12, 6)
    assert bare['status'] == 'ok' and bare['pilot']['status'] == 'not_provided', bare
    assert any('no pilot deltas or SD' in w for w in bare['warnings']), bare['warnings']
    empty = L.paired_design_sensitivity(12, 6, pilot_deltas=[])
    assert empty['status'] == 'ok' and empty['pilot']['status'] == 'not_provided', empty

    for kwargs in (dict(alpha=float('nan')), dict(target_power=1.0),
                   dict(pilot_sd=-1.0), dict(sesoi=True),
                   dict(expected_sign='maybe'), dict(require_sesoi_power='yes'),
                   dict(n_max=1), dict(pilot_deltas='abc'),
                   dict(target_halfwidth=0.0)):
        bad = L.paired_design_sensitivity(12, 6, **kwargs)
        assert bad['status'] == 'fail', (kwargs, bad)
    assert L.paired_design_sensitivity(True, 6)['status'] == 'fail'
    assert L.paired_design_sensitivity(6, 4)['status'] == 'fail'
    assert any('minimum attainable Holm p' in v
               for v in L.paired_design_sensitivity(6, 4)['violations'])

