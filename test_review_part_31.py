"""CPU checks for the missing-data sensitivity / planned-frame audit.

The panel decision rule already refuses to promote an incomplete paired panel.
These checks pin the quantitative readout that rule leaves out: sharp
partial-identification bounds over a declared range, the tipping value that
would erase the observed sign or SESOI margin, separate missing vs
control-gate-excluded accounting, and a replay gate that rejects a hand-edited
robustness claim.  No test reads a result panel or a held-out split.
"""
import contextlib
import copy
import io
import json
import math
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


_RUN_CFG = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
            'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _summary(deltas, arms=('a', 'b'), base=50.0):
    out = {}
    for i, d in enumerate(deltas):
        out[f'{arms[0]}::seed{i}'] = dict(
            variant=arms[0], seed=i, run_cfg=_RUN_CFG, steps=1,
            tokens_seen=1, params=100, ppl=float(base) + float(d))
        out[f'{arms[1]}::seed{i}'] = dict(
            variant=arms[1], seed=i, run_cfg=_RUN_CFG, steps=1,
            tokens_seen=1, params=100, ppl=float(base))
    return out


def _planned(arms, seeds):
    return [f'{a}::seed{s}' for a in arms for s in seeds]


def _raises(fn, why):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(why)


def test_complete_frame_has_no_missing_sensitivity():
    r = L.missingness_sensitivity([1.0, 1.2, 0.8], seeds=[0, 1, 2],
                                  planned_seeds=[0, 1, 2])
    assert r['status'] == 'complete'
    assert r['n_planned'] == 3 and r['n_missing'] == 0 and r['n_excluded'] == 0
    assert r['missing_seeds'] == [] and r['excluded_seeds'] == []
    assert _close(r['overall_mean_bounds'][0], r['observed_mean'])
    assert _close(r['overall_mean_bounds'][1], r['observed_mean'])
    assert r['scenarios'] == []
    assert r['robust_zero'] is True


def test_tipping_point_and_sharp_bounds():
    r = L.missingness_sensitivity([1.0, 1.0, 1.0], seeds=[0, 1, 2],
                                  planned_seeds=[0, 1, 2, 3],
                                  margin=0.5, delta_range=[-1.0, 1.0])
    # (n/u) * mean = 3 * 1.0: one unobserved seed needs -3.0 to erase the sign.
    assert _close(r['unobserved_mean_for_zero'], -3.0)
    assert _close(r['tipping_ratio_for_zero'], 3.0)
    # margin 0.5 is crossed when the missing seed contributes 0.5 - 3*0.5 = -1.0.
    assert _close(r['unobserved_mean_for_threshold'], -1.0)
    assert r['tipping_in_range_zero'] is False
    assert r['tipping_in_range_threshold'] is True
    # x in [-1, 1] -> overall mean in [mean + (1/4)(-1 - mean), ...] = [0.5, 1.0]
    assert _close(r['overall_mean_bounds'][0], 0.5)
    assert _close(r['overall_mean_bounds'][1], 1.0)
    assert r['robust_zero'] is True and r['robust_margin'] is False
    assert r['status'] == 'sign_robust_margin_not'
    # The range endpoints are imputed scenarios, not measurements.
    assert r['scenarios'] and all(s['imputed'] is True for s in r['scenarios'])


def test_range_crossing_zero_is_flagged_as_not_robust():
    r = L.missingness_sensitivity([2.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                                  seeds=[0, 1, 2, 3, 4, 5],
                                  planned_seeds=[0, 1, 2, 3, 4, 5, 6])
    assert _close(r['observed_mean'], 1.0 / 6.0)
    assert r['tipping_in_range_zero'] is True
    assert r['status'] == 'not_robust_within_range'
    assert r['overall_mean_bounds'][0] <= 0.0 <= r['overall_mean_bounds'][1]
    assert any('erases the observed sign' in w for w in r['warnings'])


def test_excluded_cells_are_not_called_missing():
    r = L.missingness_sensitivity([1.0, 2.0, 3.0], seeds=[0, 1, 2],
                                  planned_seeds=[0, 1, 2, 3],
                                  excluded_seeds=[3])
    assert r['n_missing'] == 0 and r['n_excluded'] == 1
    assert r['n_unobserved'] == 1 and r['missing_seeds'] == []
    assert r['excluded_seeds'] == [3]
    # The excluded seed has no delta, so it still widens the identification
    # bounds; it is simply named as a control-gate exclusion, not an absent run.
    assert r['overall_mean_bounds'][0] < r['observed_mean']


def test_missingness_scale_invariance():
    base = [1.0, 1.1, 0.9, 1.05]
    a = L.missingness_sensitivity(base, seeds=[0, 1, 2, 3],
                                  planned_seeds=[0, 1, 2, 3, 4],
                                  delta_range=[-0.2, 1.2], margin=0.3)
    scale = 1e6
    b = L.missingness_sensitivity([x * scale for x in base],
                                  seeds=[0, 1, 2, 3],
                                  planned_seeds=[0, 1, 2, 3, 4],
                                  delta_range=[-0.2 * scale, 1.2 * scale],
                                  margin=0.3 * scale)
    assert a['status'] == b['status']
    assert _close(a['unobserved_mean_for_zero'] * scale,
                  b['unobserved_mean_for_zero'])
    assert _close(a['overall_mean_bounds'][0] * scale,
                  b['overall_mean_bounds'][0])
    assert a['robust_zero'] == b['robust_zero']
    assert a['robust_margin'] == b['robust_margin']


def test_missingness_fail_closed_on_malformed_frames():
    good = dict(seeds=[0, 1], planned_seeds=[0, 1, 2])
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, seeds=[0, 0])),
            'duplicate observed seeds')
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, planned_seeds=[1, 2])),
            'observed seed outside the plan')
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, excluded_seeds=[1])),
            'a seed cannot be observed and excluded')
    _raises(lambda: L.missingness_sensitivity([1, True], **good),
            'boolean delta')
    _raises(lambda: L.missingness_sensitivity([1, float('nan')], **good),
            'NaN delta')
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, margin=-1.0)),
            'negative margin')
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, delta_range=[2, 1])),
            'reversed range')
    _raises(lambda: L.missingness_sensitivity([1, 2], **dict(good, planned_seeds=[])),
            'empty plan')
    _raises(lambda: L.missingness_sensitivity([1], **good),
            'delta/seed length mismatch')


def test_panel_coverage_separates_absent_and_unusable_cells():
    summary = _summary([1.0, 1.0, 1.0], arms=('a', 'b'))
    # Drop b::seed2.  a is complete, b is not: the absent seed sets differ.
    summary.pop('b::seed2')
    c = L.panel_missingness_audit(
        summary, planned_cells=_planned(('a', 'b'), [0, 1, 2]))
    assert c['status'] == 'incomplete'
    assert c['unequal_arm_coverage'] is True
    assert c['by_arm']['a']['missing_seeds'] == []
    assert c['by_arm']['b']['missing_seeds'] == [2]
    assert c['n_planned_cells'] == 6 and c['n_missing_cells'] == 1
    assert _close(c['coverage_rate'], 5.0 / 6.0)
    # A present record with no usable primary PPL is "unusable", not absent.
    summary2 = _summary([1.0, 1.0, 1.0], arms=('a', 'b'))
    summary2['b::seed2']['ppl'] = None
    c2 = L.panel_missingness_audit(
        summary2, planned_cells=_planned(('a', 'b'), [0, 1, 2]))
    assert c2['n_missing_cells'] == 0
    assert c2['n_unusable_cells'] == 1
    assert c2['by_arm']['b']['unusable_seeds'] == [2]


def test_no_plan_is_reported_not_assumed_complete():
    c = L.panel_missingness_audit(_summary([1.0, 1.0, 1.0]))
    assert c['status'] == 'no_plan'
    assert c['n_planned_cells'] is None and c['coverage_rate'] is None
    assert any('no planned-cell list' in w for w in c['warnings'])


def test_methodology_audit_attaches_missingness_without_promoting():
    deltas = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2]
    summary = _summary(deltas)
    design = {'compA': [('a', 'b', 'x')]}
    before = L.methodology_audit(summary, design=design, baselines={},
                                 verbose=False)
    after = L.methodology_audit(
        summary, design=design, baselines={}, verbose=False,
        planned_cells=_planned(('a', 'b'), range(7)))
    e = after['components']['compA'][0]
    assert e['missingness']['n_missing'] == 1
    assert e['missingness']['missing_seeds'] == [6]
    assert e['missingness']['status'] in ('robust_beyond_margin',
                                          'sign_robust_margin_not',
                                          'not_robust_within_range')
    # The new descriptive block must not move any decision.
    assert after['verified_components'] == before['verified_components']
    assert after['significant_components'] == before['significant_components']
    assert after['family'] == before['family']
    assert after['missingness']['n_contrasts'] == 1


def test_missingness_gate_replays_and_rejects_tampering():
    import audit_gates as A
    summary = _summary([2.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    outdir = tempfile.mkdtemp()
    L.methodology_audit(
        summary, design={'compA': [('a', 'b', 'x')]}, baselines={},
        outdir=outdir, verbose=False,
        planned_cells=_planned(('a', 'b'), range(7)))
    with contextlib.redirect_stdout(io.StringIO()):
        ok = A.audit_missingness(outdir)
    assert ok['status'] == 'ok' and ok['n_checked'] == 1, ok
    path = os.path.join(outdir, 'methodology_audit.json')
    with open(path, encoding='utf-8') as fh:
        original = json.load(fh)
    # A hand-edited bound/status must not survive the replay.
    bad = copy.deepcopy(original)
    bad['components']['compA'][0]['missingness']['overall_mean_bounds'] = [123.0, 456.0]
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(bad, fh)
    with contextlib.redirect_stdout(io.StringIO()):
        failed = A.audit_missingness(outdir)
    assert failed['status'] == 'fail', failed
    assert any('overall_mean_bounds' in p for p in failed['problems'])
    bad = copy.deepcopy(original)
    bad['components']['compA'][0]['missingness']['status'] = 'complete'
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(bad, fh)
    with contextlib.redirect_stdout(io.StringIO()):
        failed2 = A.audit_missingness(outdir)
    assert failed2['status'] == 'fail', failed2
    assert any('status' in p for p in failed2['problems'])


def test_missingness_gate_treats_legacy_artifact_as_unverified():
    import audit_gates as A
    outdir = tempfile.mkdtemp()
    path = os.path.join(outdir, 'methodology_audit.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'components': {'c': [{'deltas': [1.0, 2.0]}]}}, fh)
    with contextlib.redirect_stdout(io.StringIO()):
        out = A.audit_missingness(outdir)
    assert out['status'] == 'warn', out
    assert out['n_not_verifiable'] == 1
    assert out['n_ok'] == 0


def test_missingness_gate_is_registered_in_cli_and_report():
    import inspect
    import audit_gates as A
    src = inspect.getsource(A)
    assert "'missingness': audit_missingness" in src
    assert "'missingness': audit_missingness()" in src
