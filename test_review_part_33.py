"""CPU checks for the paired evaluation-trajectory stability readout.

The frozen endpoint is the fixed final evaluation.  These checks pin the
descriptive question next to it: does the per-eval-step history on the same
paired seeds end on the same sign as the quoted endpoint, or is the final
ordering a late crossing / a still-narrowing gap?  No test reads a result
panel, a held-out split or a trained checkpoint.
"""
import json
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


def _raises(fn, why):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(why)


def _hist(*rows):
    return [list(r) for r in rows]


def test_stable_trajectory_and_primary_agreement():
    ha = {0: _hist((100, 50.0), (200, 40.0), (300, 30.0)),
          1: _hist((100, 60.0), (200, 50.0), (300, 40.0))}
    hb = {0: _hist((100, 52.0), (200, 43.0), (300, 34.0)),
          1: _hist((100, 62.0), (200, 54.0), (300, 44.0))}
    r = L.paired_trajectory_stability(ha, hb, primary_delta=-4.0)
    assert r['status'] == 'stable', r
    assert r['n_shared_steps'] == 3 and r['shared_steps'] == [100, 200, 300]
    assert r['final_step'] == 300 and r['final_sign'] == -1
    assert r['tail_sign_stable'] is True
    assert r['n_sign_changes'] == 0 and r['last_sign_change_step'] is None
    assert r['n_seeds_final_sign_matches'] == 2
    assert r['primary_vs_history_sign'] is True
    assert r['gap_trend'] == 'widening'
    assert r['warnings'] == []


def test_late_crossing_is_not_relabelled_stable():
    ha = {0: _hist((100, 40.0), (200, 30.0), (300, 23.0)),
          1: _hist((100, 44.0), (200, 33.0), (300, 25.0))}
    # At 100/200 arm a is better; at 300 it is worse: the final sign is new.
    hb = {0: _hist((100, 42.0), (200, 32.0), (300, 22.0)),
          1: _hist((100, 46.0), (200, 35.0), (300, 24.0))}
    r = L.paired_trajectory_stability(ha, hb, primary_delta=-1.0)
    assert r['status'] == 'late_crossing', r
    assert r['tail_sign_stable'] is False
    assert r['n_sign_changes'] == 1 and r['last_sign_change_step'] == 300
    assert any('late crossing' in w for w in r['warnings'])
    # History says + at the end; the primary delta is negative: disagreement
    # must be named even though the crossing is already flagged.
    assert r['primary_vs_history_sign'] is False


def test_final_point_only_and_zero_final():
    ha = {0: _hist((100, 40.0), (200, 38.0), (300, 30.0))}
    hb = {0: _hist((100, 40.0), (200, 38.0), (300, 32.0))}
    r = L.paired_trajectory_stability(ha, hb)
    assert r['status'] == 'final_point_only', r
    assert r['tail_sign_stable'] is False
    assert any('only the final point' in w for w in r['warnings'])

    ha = {0: _hist((100, 40.0), (200, 38.0), (300, 30.0))}
    hb = {0: _hist((100, 41.0), (200, 39.0), (300, 30.0))}
    r = L.paired_trajectory_stability(ha, hb)
    assert r['status'] == 'zero_final', r
    assert r['final_sign'] == 0


def test_narrowing_gap_is_flagged_descriptively():
    # Negative gap at every step, but it shrinks toward zero: log ratios
    # -0.2, -0.1, -0.02.  This is not a new test and must not change status.
    ha = {0: _hist((100, 80.0), (200, 60.0), (300, 50.0))}
    hb = {0: _hist((100, 97.7712), (200, 66.3145), (300, 51.0101))}
    r = L.paired_trajectory_stability(ha, hb)
    assert r['final_sign'] == -1
    assert r['gap_trend'] == 'narrowing', r
    assert r['tail_gap_slope_log_per_1k'] > 0.0
    assert any('narrowing' in w for w in r['warnings'])


def test_log_and_raw_signs_can_diverge_without_hiding_it():
    # Per seed the raw delta and the log ratio have the same sign, but the
    # seed with the large reference PPL dominates the raw mean while the log
    # mean weights the two ratios equally.  The final raw sign is negative and
    # the final log sign is positive; the block must name the scale choice.
    ha = {0: _hist((100, 90.0), (200, 90.0), (300, 90.0)),
          1: _hist((100, 1.2), (200, 1.2), (300, 1.2))}
    hb = {0: _hist((100, 100.0), (200, 100.0), (300, 100.0)),
          1: _hist((100, 1.0), (200, 1.0), (300, 1.0))}
    r = L.paired_trajectory_stability(ha, hb, primary_delta=-4.9)
    assert r['final_delta_ppl_sign'] == -1
    assert r['final_sign'] == 1
    assert r['scale_sensitive_final'] is True
    assert r['primary_vs_history_sign'] is True
    assert r['primary_vs_history_log_sign'] is False
    assert any('scale-sensitive' in w for w in r['warnings'])


def test_history_short_of_endpoint_is_flagged():
    ha = {0: _hist((100, 50.0), (200, 40.0))}
    hb = {0: _hist((100, 52.0), (200, 43.0))}
    r = L.paired_trajectory_stability(ha, hb, endpoint_step=300)
    assert r['final_step'] == 200
    assert r['endpoint_step'] == 300
    assert r['final_step_is_endpoint'] is False
    assert any('does not cover the reported endpoint' in w
               for w in r['warnings'])
    r2 = L.paired_trajectory_stability(ha, hb, endpoint_step=200)
    assert r2['final_step_is_endpoint'] is True


def test_balanced_grid_and_skipped_seeds_are_named():
    ha = {0: _hist((100, 50.0), (200, 40.0), (300, 30.0)),
          1: _hist((100, 60.0), (200, 50.0), (300, 40.0), (400, 35.0)),
          2: _hist((100, 70.0), (200, 60.0), (300, 50.0))}
    hb = {0: _hist((100, 52.0), (200, 42.0), (300, 32.0)),
          1: _hist((100, 62.0), (200, 52.0), (300, 42.0), (400, 37.0))}
    r = L.paired_trajectory_stability(ha, hb)
    assert r['seeds'] == [0, 1]
    assert [x['seed'] for x in r['skipped_seeds']] == [2]
    assert r['shared_steps'] == [100, 200, 300]
    assert r['steps_dropped_unbalanced'] == [400]
    assert [x['seed'] for x in r['skipped_seeds']] == [2]
    assert all(row['n_seeds'] == 2 for row in r['series'])
    assert any('dropped' in w for w in r['warnings'])
    assert any('skipped' in w for w in r['warnings'])


def test_scale_shift_does_not_change_shape_or_sign():
    base_a = {0: _hist((100, 50.0), (200, 40.0), (300, 30.0))}
    base_b = {0: _hist((100, 52.0), (200, 43.0), (300, 34.0))}
    a = L.paired_trajectory_stability(base_a, base_b)
    scaled_a = {0: [[s, p * 1e6] for s, p in base_a[0]]}
    scaled_b = {0: [[s, p * 1e6] for s, p in base_b[0]]}
    b = L.paired_trajectory_stability(scaled_a, scaled_b)
    assert a['status'] == b['status'] == 'stable'
    assert a['final_sign'] == b['final_sign'] == -1
    assert _close(a['final_mean_log_ratio'], b['final_mean_log_ratio'], 1e-12)
    assert _close(a['tail_gap_slope_log_per_1k'],
                  b['tail_gap_slope_log_per_1k'], 1e-12)
    assert _close(a['final_mean_delta_ppl'] * 1e6,
                  b['final_mean_delta_ppl'])


def test_malformed_history_fails_closed():
    good = {0: _hist((100, 50.0), (200, 40.0))}
    _raises(lambda: L.paired_trajectory_stability([], good), 'list is not a mapping')
    _raises(lambda: L.paired_trajectory_stability({0: [(100, 50), (100, 40)]}, good),
            'duplicate step')
    _raises(lambda: L.paired_trajectory_stability({0: [(100, 0.0)]}, {0: [(100, 1.0)]}),
            'non-positive PPL')
    _raises(lambda: L.paired_trajectory_stability({0: [(100, True)]}, {0: [(100, 1.0)]}),
            'boolean PPL')
    _raises(lambda: L.paired_trajectory_stability({0: [(1.5, 2.0)]}, {0: [(1, 1.0)]}),
            'non-integer step')
    _raises(lambda: L.paired_trajectory_stability({0: [(100, 2.0)]}, {0: [(100, 1.0)]},
                                                  tail_points=1), 'tail_points < 2')
    _raises(lambda: L.paired_trajectory_stability(good, good, seeds=[0, 0]),
            'duplicate requested seeds')
    _raises(lambda: L.paired_trajectory_stability(good, good, primary_delta='x'),
            'non-numeric primary delta')


def test_insufficient_history_is_explicit():
    ha = {0: _hist((100, 50.0), (200, 40.0))}
    hb = {0: _hist((150, 51.0), (250, 41.0))}
    r = L.paired_trajectory_stability(ha, hb)
    assert r['status'] == 'insufficient_history'
    assert r['n_shared_steps'] == 0
    assert r['shared_steps'] == []
    assert any('not identifiable' in w or 'cannot be checked' in w
               for w in r['warnings'])


def _summary(with_history=True, flips=False):
    out = {}
    for s in (0, 1, 2):
        a_hist = [[100, 50.0], [200, 40.0], [300, 30.0 + s]]
        b_hist = [[100, 52.0], [200, 43.0], [300, 34.0 + s]]
        if flips:
            # Final history sign reverses while the primary PPL stays as-is.
            a_hist[-1][1] = 40.0 + s
        rec_a = dict(variant='a', seed=s, run_cfg=_RUN_CFG, steps=300,
                     tokens_seen=300, params=100, ppl=30.0 + s)
        rec_b = dict(variant='b', seed=s, run_cfg=_RUN_CFG, steps=300,
                     tokens_seen=300, params=100, ppl=34.0 + s)
        if with_history:
            rec_a['ppl_history'] = a_hist
            rec_b['ppl_history'] = b_hist
        out[f'a::seed{s}'] = rec_a
        out[f'b::seed{s}'] = rec_b
    return out


def test_methodology_audit_attaches_without_changing_verdicts():
    design = {'comp': [('a', 'b', 'trajectory wiring')]}
    plain = L.methodology_audit(_summary(False), design=design, baselines={},
                                write=False, verbose=False, sesoi={})
    with_h = L.methodology_audit(_summary(True), design=design, baselines={},
                                 write=False, verbose=False, sesoi={})
    e_plain = plain['components']['comp'][0]
    e_hist = with_h['components']['comp'][0]
    assert e_plain['trajectory_stability']['status'] == 'insufficient_history'
    assert e_hist['trajectory_stability']['status'] == 'stable'
    assert e_hist['trajectory_stability']['primary_vs_history_sign'] is True
    # The diagnostic is descriptive: every decision-bearing number is identical.
    for key in ('mean', 'std', 'p_exact_signflip', 'p_holm_global',
                'significant_global', 'verified', 'deltas'):
        assert e_plain.get(key) == e_hist.get(key), key
    assert plain['verified_components'] == with_h['verified_components']
    assert plain['significant_components'] == with_h['significant_components']
    assert plain['family'] == with_h['family']
    assert with_h['trajectory_stability']['n_contrasts'] == 1
    assert with_h['trajectory_stability']['n_stable'] == 1
    assert plain['trajectory_stability']['n_contrasts'] == 1
    assert plain['trajectory_stability']['n_stable'] == 0


def test_methodology_audit_flags_late_crossing_and_disagreement():
    design = {'comp': [('a', 'b', 'trajectory wiring')]}
    rep = L.methodology_audit(_summary(True, flips=True), design=design,
                              baselines={}, write=False, verbose=False, sesoi={})
    tr = rep['trajectory_stability']
    assert tr['n_late_crossing'] == 1, tr
    assert tr['n_primary_history_disagreement'] == 1, tr
    assert tr['late_crossing_contrasts'] == [
        'component::a - b'], tr
    assert any('late-crossing' in w for w in rep['warnings'])
    assert any('disagrees with the primary-endpoint sign' in w
               for w in rep['warnings'])
    # Again: the warning is descriptive, not a verdict change.
    assert rep['components']['comp'][0]['p_exact_signflip'] is not None


def test_audit_gate_replays_and_rejects_tamper():
    design = {'comp': [('a', 'b', 'trajectory gate')]}
    with tempfile.TemporaryDirectory() as tmp:
        L.methodology_audit(_summary(True), design=design, baselines={},
                            outdir=tmp, write=True, verbose=False, sesoi={})
        path = os.path.join(tmp, 'methodology_audit.json')
        clean = A.audit_trajectory(tmp)
        assert clean['status'] == 'ok', clean['problems']
        data = json.load(open(path))
        tr = data['components']['comp'][0]['trajectory_stability']
        assert tr['status'] == 'stable'
        # A hand-edited final sign is not replayed as a clean panel.
        tr['final_sign'] = 1
        json.dump(data, open(path, 'w'))
        tampered = A.audit_trajectory(tmp)
        assert tampered['status'] == 'fail'
        assert any('final_sign' in p for p in tampered['problems'])
        # A legacy artifact without the block is unverifiable, never a pass.
        data = json.load(open(path))
        del data['trajectory_stability']
        del data['components']['comp'][0]['trajectory_stability']
        json.dump(data, open(path, 'w'))
        legacy = A.audit_trajectory(tmp)
        assert legacy['status'] == 'not_verifiable_locally' or legacy.get('n_not_verifiable') == 1
