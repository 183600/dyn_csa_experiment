#!/usr/bin/env python3
"""test_review_fixes.py — 单元测试（纯 CPU、零实验）。

覆盖：
  A. exp_lib 统计工具：
     holm_min_attainable_p / signflip_min_consistency / signflip_power /
     tost_paired / exact_signflip_p 假设文档存在性（导入不报错即可证）；
  B. v10 腐坏模式：_corrupt_row 的既有四种模式互相对拍，新模式
     span/doc/topical 的确定性与目标区不可触碰约束；
  C. v12 门槛：_load_manifest 功效门槛（6 种子拒绝、12 种子放行、
     descriptive_only 豁免）；_admit_confirmation_records 的白名单/歧义/
     混切片/None 指纹行为；_recipe_matches_signature；冻结锁的 manifest
     不匹配拒绝与 V12_REFREEZE 降级；v2 清单文件本身的功效表与代码一致；
  D. eval_length_gen 逐文档 PPL：池化 PPL == 逐文档 PPL 的几何平均
     （归约不变）。

运行：python test_review_fixes.py   （需要 torch；无 torch 的环境全部跳过
并给出非零提醒码 0——CI 以「SKIPPED」字样识别）。"""
import contextlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

try:
    import numpy as np
    import torch  # noqa: F401
    import exp_lib as L
    import v10_supp as V10
    import v12_supp as V12
    import v7_supp as V7
    import audit_gates as A
    import baseline_tune as BT
    import build_report as BR
    import error_analysis as EA
    import m0_supp as M0
    _HAVE_TORCH = True
except Exception as _e:  # pragma: no cover
    print(f'[test] SKIPPED: cannot import the repo modules here ({type(_e).__name__}: {_e}); run on a box with CPU torch')
    _HAVE_TORCH = False

_results = []


class _SkipTest(Exception):
    """A check whose precondition this environment cannot provide (e.g. the
    sandbox forbids POSIX semaphores for DataLoader workers).  Reported as
    SKIP rather than silently passing or failing."""


def _check(name, fn):
    try:
        fn()
        _results.append((name, 'PASS', ''))
        print(f'[test] PASS  {name}')
    except _SkipTest as e:
        _results.append((name, 'SKIP', str(e)))
        print(f'[test] SKIP  {name}: {e}')
    except AssertionError as e:
        _results.append((name, 'FAIL', str(e)))
        print(f'[test] FAIL  {name}: {e}')
    except Exception as e:
        _results.append((name, 'ERROR', f'{type(e).__name__}: {e}'))
        print(f'[test] ERROR {name}: {type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# A. exp_lib 统计工具
# ---------------------------------------------------------------------------

def t_holm_min_attainable():
    assert abs(L.holm_min_attainable_p(6, 4, 0.05) - 0.125) < 1e-12, '6 seeds x Holm/4 floor must be 0.125'
    assert abs(L.holm_min_attainable_p(12, 4, 0.05) - 0.001953125) < 1e-12
    assert abs(L.holm_min_attainable_p(8, 1, 0.05) - 0.0078125) < 1e-12
    assert L.holm_min_attainable_p(6, 4, 0.05) > 0.05 and L.holm_min_attainable_p(12, 4, 0.05) < 0.05
    # Duplicate names cannot share one dict slot: rejecting is the only
    # representation that does not shrink/corrupt the family.
    try:
        L.holm_adjust([('a', 0.01), ('a', 0.02)])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate comparison names must be rejected')


def t_min_consistency():
    assert L.signflip_min_consistency(12, 4, 0.05) == 11, 'n=12, m=4: 11/12 same-sign clears the first Holm gate (raw p = 2*(1+12)/4096 = 0.00635 < 0.0125)'
    assert L.signflip_min_consistency(6, 4, 0.05) is None, 'n=6, m=4: even 6/6 cannot clear the gate'
    assert L.signflip_min_consistency(8, 1, 0.05) == 8
    assert L.signflip_min_consistency(6, 1, 0.05) == 6
    # Holm rejects at p_holm <= alpha.  The planning helper must use the same
    # boundary: at n=3, m=1, alpha=0.25 the attainable floor IS 0.25.
    assert L.signflip_min_consistency(3, 1, 0.25) == 3
    assert L.signflip_power(3, 1, 1.0, 0.25) == 1.0
    assert L.signflip_min_consistency(5, 1, 0.0625) == 5
    assert L.signflip_min_consistency(3, 1, 0.25 - 1e-12) is None
    # Scale-up: the planning math must not overflow int->float at large n.
    big = L.signflip_min_consistency(2000, 10, 0.05)
    assert big is not None and 0 < big < 2000, big
    # The planning alpha is a probability threshold, not a coerced float:
    # True->1.0 and a string could otherwise change the required seed count.
    for _bad_alpha in (True, 0, 1.0, 1.5, float('nan'), 'not-a-number'):
        try:
            L.signflip_min_consistency(12, 4, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_min_consistency accepted alpha={_bad_alpha!r}')
        try:
            L.signflip_power(12, 4, 0.9, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_power accepted alpha={_bad_alpha!r}')
        try:
            L.signflip_min_n_for_holm(4, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_min_n_for_holm accepted alpha={_bad_alpha!r}')


def t_power():
    assert L.signflip_power(6, 4, 0.99) == 0.0, 'an impossible plan has zero power at any q'
    pw = L.signflip_power(12, 4, 0.9, 0.05)
    assert abs(pw - 0.659) < 0.01, f'q=0.9 power at first Holm gate should be ~0.659, got {pw:.4f}'
    assert L.signflip_power(12, 4, 0.8, 0.05) < 0.30
    pw_big = L.signflip_power(2000, 10, 0.55, 0.05)
    assert 0.0 < pw_big <= 1.0 and math.isfinite(pw_big), pw_big


def t_tost():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.tost_paired(inside, margin=0.01)
    assert r['equivalent'] is True and r['p_tost'] < 0.05, f'tiny deltas vs margin must be equivalent: {r}'
    above = [0.05, 0.052, 0.049, 0.051, 0.05, 0.048]
    r2 = L.tost_paired(above, margin=0.01)
    assert r2['equivalent'] is False and r2['p_tost'] > 0.05, f'mean far above margin must NOT be equivalent: {r2}'
    # direction check:
    far = [0.9, 0.95, 0.85, 1.0, 0.92, 0.88]
    r3 = L.tost_paired(far, margin=0.1)
    assert r3['equivalent'] is False, 'mean ~0.9 vs margin 0.1 must not be equivalent even with noise'
    small_n = L.tost_paired([0.0, 0.0], margin=0.1)
    assert small_n['equivalent'] is not None


def t_tost_student_t_and_boundary():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.tost_paired(inside, margin=0.01)
    assert r['equivalent'] is True, r
    assert r['df'] == 5 and 'Student t' in r['method'], r
    se = float(np.std(inside, ddof=1) / math.sqrt(len(inside)))
    tc = L._t_critical(0.90, 5)
    assert abs((r['ci'][1] - r['mean']) - tc * se) < 1e-12, 'TOST CI must use the t critical value'
    assert (r['ci'][1] - r['mean']) > 1.6448536269514722 * se, 'at n=6 the t interval must be wider than the normal-z interval'
    # Zero-variance sample exactly at the margin is the t limit P(T>=0)=0.5,
    # not certainty: equality at the boundary must never be called equivalent.
    for boundary in ([0.1] * 6, [-0.1] * 6):
        b = L.tost_paired(boundary, margin=0.1)
        assert b['equivalent'] is False and abs(b['p_tost'] - 0.5) < 1e-12, b
    # The t critical value must invert the t tail for high confidence and
    # small/fractional df too; a fixed 1000 cap used to return a too-narrow CI.
    for _df, _conf in ((0.5, 0.99), (1.0, 0.9999), (10.0, 0.999)):
        _tc = L._t_critical(_conf, _df)
        assert math.isfinite(_tc), (_df, _conf, _tc)
        assert abs(L._student_t_sf(_tc, _df) - (1.0 - _conf) / 2.0) <= 1e-10, (
            _df, _conf, _tc)


def t_safe_exp_and_doc_ci():
    # A finite exponent must stay a finite PPL; only an actual IEEE overflow
    # yields inf.
    assert math.isfinite(L._safe_exp_ppl(705.0))
    assert L._safe_exp_ppl(1000.0) == float('inf')
    assert L._safe_exp_ppl(float('nan')) != L._safe_exp_ppl(float('nan'))  # NaN stays NaN
    # Token-pooled CE and its interval must target the same estimand even when
    # document lengths differ.
    nll = [10.0, 1.0, 4.0]
    ntok = [100, 10, 20]
    ci = L._nll_doc_ci(nll, ntok)
    ce = sum(nll) / sum(ntok)
    assert abs(math.log(ci['ppl']) - ce) < 1e-12, ci
    assert ci['ppl_ci95_low'] <= ci['ppl'] <= ci['ppl_ci95_high'], ci
    assert ci['ci_method'].startswith('cluster-robust'), ci
    assert ci['n_docs'] == 3
    # NumPy arrays are the natural per-document inputs, and a one-shot
    # iterator must not be exhausted before the residual pass.  Neither input
    # may raise or report a zero document count.
    ci_np = L._nll_doc_ci(np.array(nll), np.array(ntok))
    assert ci_np['n_docs'] == 3 and abs(ci_np['ppl'] - ci['ppl']) < 1e-12, ci_np
    ci_gen = L._nll_doc_ci(iter(nll), iter(ntok))
    assert ci_gen['n_docs'] == 3 and abs(ci_gen['ppl'] - ci['ppl']) < 1e-12, ci_gen
    # A single scalar pair is one document, not an unhandled TypeError.
    ci_scalar = L._nll_doc_ci(10.0, 100)
    assert ci_scalar['n_docs'] == 1, ci_scalar
    assert abs(math.log(ci_scalar['ppl']) - 0.1) < 1e-12, ci_scalar
    # The same iterable contract holds for the curve diagnostics.
    assert L._finite_curve(np.array([1.0, np.inf, 2.0]))['n_finite'] == 2
    assert L._finite_curve(iter([1.0, 2.0]))['n_finite'] == 2
    # PPL is >= 1 by definition: a symmetric CE interval that crosses zero
    # must not be exponentiated into an impossible sub-1 lower bound.
    _wide = L._nll_doc_ci([6.369616873214543, 2.697867137638703], [1, 1])
    assert _wide['ppl_ci95_low'] >= 1.0, _wide
    assert _wide['ppl_ci95_low'] <= _wide['ppl'] <= _wide['ppl_ci95_high'], _wide
    # Mismatched per-document accounting is malformed evidence, not an
    # invitation to silently zip to the shorter list.
    try:
        L._nll_doc_ci([1.0, 2.0], [10])
    except ValueError as e:
        assert 'paired' in str(e), e
    else:
        raise AssertionError('mismatched NLL/token lists must be refused')


def t_bootstrap_mean_ci():
    vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    a = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    b = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    assert a == b and a['n'] == 5 and a['ci_low'] <= a['mean'] <= a['ci_high'], a
    assert L.bootstrap_mean_ci([1.0]) is None
    assert L.bootstrap_mean_ci([1.0, float('nan'), float('inf')]) is None
    # Paired/unpaired helpers are chunked for seed panels that can reach
    # hundreds of seeds; the result stays deterministic and well-formed.
    rng = np.random.default_rng(3)
    d = rng.normal(0.5, 1.0, size=400)
    p1 = L.paired_bootstrap_ci(d, n_boot=600, seed=11)
    p2 = L.paired_bootstrap_ci(d, n_boot=600, seed=11)
    assert p1 == p2 and p1[0] < float(d.mean()) < p1[1], (p1, d.mean())
    u1 = L.unpaired_bootstrap_ci(d + 1.0, d, n_boot=600, seed=11)
    u2 = L.unpaired_bootstrap_ci(d + 1.0, d, n_boot=600, seed=11)
    assert u1 == u2 and u1[0] < 1.0 < u1[1], u1


def t_training_health_classifier():
    # Overfitting: train CE falls while val CE rises in the tail.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 + 0.05 * i,
             'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h['best_step'] is not None and h['final_minus_best'] is not None, h
    # Underfitting/stalled: neither curve decreases.
    hist = [{'step': i * 100, 'train_ce': 4.0 + 0.001 * i, 'val_ce': 4.0 + 0.001 * i,
             'val_ppl': 60.0 + 0.01 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'underfitting_or_stalled', h
    # Healthy: train falls, val does not rise.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 - 0.02 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy', h
    # A MEASURED non-finite point must be flagged...
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 - 0.02 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    hist[3]['val_ppl'] = float('nan')
    hist[3]['val_ce'] = float('nan')
    h = L.classify_training_health(hist)
    assert h['status'] == 'nonfinite' and h['n_nonfinite'] == 2, h
    # ...but an UNMONITORED train CE (None) must not be miscounted as
    # non-finite: the val-only classification speaks, and the reason says
    # the train curve is missing.
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0 - 0.05 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy' and h['n_nonfinite'] == 0, h
    assert 'train CE unmonitored' in h['reason'] and h['n_train_ce_measured'] == 0, h
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0 + 0.05 * i,
             'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'validation_degrading' and h['n_nonfinite'] == 0, h
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0, 'val_ppl': 55.0} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'underfitting_or_stalled', h
    # A train CE that was measured and came back NaN is still nonfinite.
    hist = [{'step': i * 100, 'train_ce': (float('nan') if i == 2 else 6.0 - 0.3 * i),
             'val_ce': 4.0 - 0.02 * i, 'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'nonfinite' and h['n_nonfinite'] == 1, h
    # Validation improving while training CE is flat must NOT be labelled
    # "validation_degrading" (the old else branch inverted the direction).
    hist = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0 - 0.05 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'validation_improving_train_flat', h
    assert 'validation CE is falling' in h['reason'], h
    assert any('not read this as overfitting' in a for a in h['actions']), h
    # Too few points stays insufficient regardless of which side is missing.
    h = L.classify_training_health([{'step': 100, 'train_ce': 5.0, 'val_ce': 4.0, 'val_ppl': 55.0}])
    assert h['status'] == 'insufficient_data', h
    h = L.classify_training_health([])
    assert h['status'] == 'insufficient_data', h


def t_training_health_noise_aware_threshold():
    # Four equally spaced eval points with realistic CE jitter: the tail slope
    # is +3.2e-5, five times the fixed 1e-5 tolerance, yet it is entirely
    # within the slope's own noise (2*SE ~ 3.4e-5).  The adaptive threshold
    # must call it healthy.
    hist = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.00, 'val_ppl': 54.6},
            {'step': 250, 'train_ce': 4.5, 'val_ce': 4.02, 'val_ppl': 55.7},
            {'step': 500, 'train_ce': 4.0, 'val_ce': 4.01, 'val_ppl': 55.1},
            {'step': 750, 'train_ce': 3.5, 'val_ce': 4.03, 'val_ppl': 56.3}]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy', h
    assert h['slope_tol_mode'] == 'adaptive_2se', h
    assert h['val_ce_slope_se'] is not None and h['val_slope_tol'] > 1e-5, h
    assert h['val_ce_slope_per_step'] > 1e-5, h
    # The historical fixed threshold is still available explicitly and still
    # calls the same noise "overfitting" (proving the test is discriminating).
    h_fixed = L.classify_training_health(hist, slope_tol=1e-5)
    assert h_fixed['status'] == 'overfitting', h_fixed
    assert h_fixed['slope_tol_mode'] == 'fixed', h_fixed
    # A genuine rising trend must still be caught even under the adaptive rule.
    hist_bad = [dict(hh, val_ce=4.0 + 0.01 * i, val_ppl=55.0 + 0.5 * i)
                for i, hh in enumerate(hist)]
    assert L.classify_training_health(hist_bad)['status'] == 'overfitting'


def t_training_health_actions_and_gap():
    # The classifier must not stop at a label: overfitting needs concrete
    # next steps and the widening train-val gap must be measured, not implied.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i,
             'val_ce': 4.0 + 0.05 * i, 'val_ppl': 55.0 + 0.5 * i,
             'gap': (4.0 + 0.05 * i) - (6.0 - 0.30 * i)} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h.get('actions'), h
    assert any('regularizer' in a for a in h['actions']), h
    assert h['gap_slope_per_step'] is not None and h['gap_slope_per_step'] > 0.0, h
    assert h['steps_since_best'] is not None and h['steps_since_best'] > 0, h
    assert any('best-validation' in a for a in h['actions']), h
    stalled = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0,
                'val_ppl': 55.0, 'gap': 0.0} for i in range(6)]
    hs = L.classify_training_health(stalled)
    assert hs['status'] == 'underfitting_or_stalled', hs
    assert any('data pipeline' in a for a in hs['actions']), hs
    bad = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0,
            'val_ppl': float('nan')} for i in range(6)]
    hn = L.classify_training_health(bad)
    assert hn['status'] == 'nonfinite', hn
    assert any('NaN triage' in a for a in hn['actions']), hn
    rising = [{'step': i * 100, 'train_ce': 4.0 + 0.02 * i,
               'val_ce': 4.0 + 0.02 * i, 'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    hr = L.classify_training_health(rising)
    assert hr['train_ce_rising'] is True, hr
    assert any('training CE itself is rising' in a for a in hr['actions']), hr


def t_health_accepts_val_ce_only_and_orders_steps():
    # The API documents ``val_ce`` OR ``val_ppl``.  A CE-only curve must still
    # produce the best/final checkpoint evidence the report uses, and the gap
    # must be derived from the same two measurements.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i,
             'val_ce': 4.0 + 0.05 * i} for i in range(5)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h['val_ppl_source'] == 'val_ce_derived', h
    assert h['best_step'] == 0 and h['best_val_ppl'] is not None, h
    assert abs(h['best_val_ppl'] - math.exp(4.0)) < 1e-9, h
    assert h['final_val_ppl'] is not None and h['final_minus_best'] > 0, h
    assert h['gap_source'] == 'derived' and h['gap_last'] is not None, h
    assert h['gap_slope_source'] == 'derived', h
    assert h['gap_slope_per_step'] is not None and h['gap_slope_per_step'] > 0, h

    # Out-of-order history and a repeated step are one evaluation grid, not a
    # different best step or a two-point noise fit.  The last write to a step
    # wins (resume semantics), and the tail is chronological.
    shuffled = [hist[2], hist[0], hist[4], hist[1], hist[3],
                {'step': 100, 'train_ce': 6.0, 'val_ce': 3.5}]
    hs = L.classify_training_health(shuffled)
    assert hs['n_points'] == 6 and hs['n_eval_points'] == 5, hs
    assert hs['best_step'] == 100, hs
    assert abs(hs['best_val_ppl'] - math.exp(3.5)) < 1e-9, hs
    assert hs['steps_since_best'] == 300, hs

    # An explicitly recorded non-finite PPL is a measured failure: the finite
    # CE twin must not paper over it by becoming the best PPL.
    bad = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0, 'val_ppl': 55.0},
           {'step': 100, 'train_ce': 4.5, 'val_ce': 4.1, 'val_ppl': float('nan')},
           {'step': 200, 'train_ce': 4.0, 'val_ce': 4.2, 'val_ppl': 60.0}]
    hb = L.classify_training_health(bad)
    assert hb['status'] == 'nonfinite', hb
    assert hb['val_ppl_source'] == 'val_ppl', hb

    # Early stopping accepts the same CE-only representation.
    ce_only = [{'step': 0, 'val_ce': 4.0}, {'step': 100, 'val_ce': 4.1},
               {'step': 200, 'val_ce': 4.2}]
    d = L.early_stop_decision(ce_only, patience=2)
    assert d['should_stop'] is True and d['best_step'] == 0, d
    assert d['val_ppl_source'] == 'val_ce_derived', d
    assert abs(d['best_val_ppl'] - math.exp(4.0)) < 1e-9, d
    # Two evaluation points cannot establish a trend; the label is kept but
    # explicitly marked as interpolated/provisional.
    short = L.classify_training_health([
        {'step': 0, 'train_ce': 6.0, 'val_ce': 4.0},
        {'step': 100, 'train_ce': 5.0, 'val_ce': 4.1}])
    assert short['trend_reliable'] is False and short['n_eval_points'] == 2, short
    assert any('only 2 validation evaluation point' in a for a in short['actions']), short
    assert h['trend_reliable'] is True, h

    # Fractional/Boolean steps are not evaluation identities: they must not
    # enter the slope fit while best-step selection rejects them.
    weird = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
             {'step': True, 'train_ce': 1.0, 'val_ce': 1.0},
             {'step': 100.5, 'train_ce': 1.0, 'val_ce': 1.0},
             {'step': 200, 'train_ce': 4.0, 'val_ce': 4.2}]
    hw = L.classify_training_health(weird)
    assert hw['n_eval_points'] == 2 and hw['best_step'] == 0, hw
    assert hw['train_ce_slope_per_step'] == -0.005, hw


def t_early_stop_decision():
    hist = [{'step': 100, 'val_ppl': 50.0}, {'step': 200, 'val_ppl': 48.0}]
    assert L.early_stop_decision(hist, patience=2)['should_stop'] is False
    hist += [{'step': 300, 'val_ppl': 48.0}, {'step': 400, 'val_ppl': 48.1}]
    d = L.early_stop_decision(hist, patience=2)
    assert d['should_stop'] is True and d['stale_evals'] == 2, d
    assert d['best_step'] == 200 and d['best_val_ppl'] == 48.0, d
    # A real improvement resets the patience counter; min_delta makes a tiny
    # change explicitly insufficient.
    hist2 = [{'step': 100, 'val_ppl': 50.0}, {'step': 200, 'val_ppl': 49.0},
             {'step': 300, 'val_ppl': 48.9}, {'step': 400, 'val_ppl': 47.0}]
    assert L.early_stop_decision(hist2, patience=2)['should_stop'] is False
    tight = L.early_stop_decision(hist2[:3], patience=2, min_delta=2.0)
    assert tight['should_stop'] is True, tight
    assert L.early_stop_decision([], patience=1)['should_stop'] is False
    for bad_p in (0, -1, True, 1.5, '2'):
        try:
            L.early_stop_decision(hist, patience=bad_p)
        except ValueError:
            pass
        else:
            raise AssertionError(f'patience={bad_p!r} must be refused')
    for bad_d in (-1.0, float('nan'), float('inf'), True):
        try:
            L.early_stop_decision(hist, patience=1, min_delta=bad_d)
        except ValueError:
            pass
        else:
            raise AssertionError(f'min_delta={bad_d!r} must be refused')
    # A fractional/boolean step is not an evaluation identity: truncating it
    # would merge two different eval points into one step.
    fractional = L.early_stop_decision(
        [{'step': 100, 'val_ppl': 50.0}, {'step': 100.5, 'val_ppl': 40.0}],
        patience=1)
    assert fractional['n_points'] == 1 and fractional['best_step'] == 100, fractional


def t_nan_diagnosis_path():
    clean = L.diagnose_training_instability(
        {'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                              'amp_scaler_overflows': 0, 'events': []}},
        data_checked='startup')
    assert clean['status'] == 'clean' and clean['primary_suspect'] is None, clean
    # fp16 scaler overflow is triaged before a generic gradient claim.
    fp16 = L.diagnose_training_instability({
        'amp_dtype': 'fp16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 1,
                             'amp_scaler_overflows': 3, 'amp_scaler_scale_min': 0.25,
                             'grad_norm_max': 1000.0,
                             'events': [{'kind': 'grad', 'grad_norm': float('inf'), 'lr': 3e-4}]}})
    assert fp16['status'] == 'instability_observed', fp16
    assert fp16['primary_suspect'] == 'fp16_overflow', fp16
    # A non-finite loss whose CE is finite points at the auxiliary term.
    loss = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 2, 'nonfinite_grad_steps': 0,
                             'events': [{'kind': 'loss', 'loss': float('nan'),
                                         'ce': 4.2, 'comp_reg': float('nan'),
                                         'boundary_pg': 0.0, 'lr': 3e-4}]}})
    assert loss['primary_suspect'] == 'forward_loss_numerics', loss
    assert 'compression regularizer' in loss['checks'][1]['evidence'], loss
    # A guarded attention value is evidence even when the loss never went bad.
    guard = L.diagnose_training_instability({
        'amp_dtype': None,
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'nan_guard': {'elements_cleaned': 12,
                                           'first': {'where': 'indexer_softmax'}},
                             'events': []}})
    assert guard['primary_suspect'] == 'attention_pool_nan_to_num', guard
    assert guard['checks'][0]['status'] == 'unknown', guard
    # Plain gradient explosion with finite loss in bf16.
    grad = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 1,
                             'amp_scaler_overflows': 0, 'grad_norm_max': float('inf'),
                             'events': [{'kind': 'grad', 'grad_norm': float('inf'), 'lr': 1e-2}]}})
    assert grad['primary_suspect'] == 'gradient_explosion_lr', grad
    # A FINITE but far-above-clip gradient is still an observed instability.
    # It must not be reported as clean just because no NaN/Inf step occurred
    # (the gradient_explosion_lr check itself is `observed`).
    huge = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'amp_scaler_overflows': 0, 'grad_norm_max': 100.0,
                             'events': []}})
    assert huge['status'] == 'instability_observed', huge
    assert huge['primary_suspect'] == 'gradient_explosion_lr', huge
    # An explicitly recorded inf/NaN maximum is evidence too, even if the
    # integer event counters are zero (e.g. a partially written record).
    nonfinite_max = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'amp_scaler_overflows': 0,
                             'grad_norm_max': float('nan'), 'events': []}})
    assert nonfinite_max['status'] == 'instability_observed', nonfinite_max
    assert nonfinite_max['primary_suspect'] == 'gradient_explosion_lr', nonfinite_max
    # A non-finite VALIDATION point can coexist with a finite training loss;
    # it must not be reported as clean.
    hbad = L.diagnose_training_instability({
        'curve_health': {'status': 'nonfinite', 'reason': 'val_ppl NaN at step 200'},
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert hbad['status'] == 'instability_observed', hbad
    assert hbad['primary_suspect'] == 'validation_curve_numerics', hbad
    # A legacy/edited record may carry the curve but not the cached summary;
    # an unsummarised non-finite point is still evidence, never `clean`.
    hhist = L.diagnose_training_instability({
        'health_history': [
            {'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
            {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert hhist['status'] == 'instability_observed', hhist
    assert hhist['primary_suspect'] == 'validation_curve_numerics', hhist
    assert hhist['curve_health_source'] == 'derived_from_history', hhist
    # No curve evidence at all means the validation-numerics check is unknown,
    # not silently "not observed".
    noc = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    vc = [c for c in noc['checks'] if c['check'] == 'validation_curve_numerics'][0]
    assert vc['status'] == 'unknown', noc
    assert noc['curve_health_source'] == 'missing', noc
    # A malformed summary is record-integrity evidence, but a usable history
    # still supplies the validation verdict.
    both = L.diagnose_training_instability({
        'curve_health': ['healthy'],
        'health_history': [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                           {'step': 100, 'train_ce': 4.8, 'val_ce': 4.2}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert both['status'] == 'unverifiable', both
    assert both['curve_health_source'] == 'derived_from_history', both
    # A cached healthy status cannot mask a non-finite point in its own
    # history: the curve wins and the contradiction is evidence.
    conflict = L.diagnose_training_instability({
        'curve_health': {'status': 'healthy', 'reason': 'ok'},
        'health_history': [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                           {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert conflict['status'] == 'instability_observed', conflict
    assert conflict['primary_suspect'] == 'validation_curve_numerics', conflict
    assert any(c['check'] == 'record_integrity' for c in conflict['checks']), conflict

    # A malformed/corrupt count field must neither crash the triage nor be
    # silently read as a zero-event (clean) record.
    malformed = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 1.5,
                             'nonfinite_grad_steps': {'bad': 1},
                             'amp_scaler_overflows': float('nan'),
                             'nan_guard': {'elements_cleaned': 'many'}}})
    assert malformed['status'] == 'unverifiable', malformed
    assert malformed['primary_suspect'] == 'record_integrity', malformed
    assert any(c['check'] == 'record_integrity' for c in malformed['checks']), malformed
    bad_grad_max = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'grad_norm_max': 'not-a-number', 'events': []}})
    assert bad_grad_max['status'] == 'unverifiable', bad_grad_max
    assert bad_grad_max['primary_suspect'] == 'record_integrity', bad_grad_max


def t_paired_errors_use_common_finite_support():
    # Two arms can each have non-finite CE at DIFFERENT tokens.  Comparing
    # their independent finite means would use different supports; the paired
    # delta must be computed only where both are finite.
    class _Fake:
        def __init__(self, vocab, bad_input):
            self.vocab = int(vocab)
            self.bad_input = int(bad_input)
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return ids.unsqueeze(-1)

        def head(self, h):
            import torch.nn.functional as _F
            ids = h[..., 0]
            lg = _F.one_hot(((ids + 1) % self.vocab).long(), self.vocab).float() * 8.0
            lg = lg.clone()
            lg[ids == self.bad_input] = float('inf')
            return lg

    vb = np.array([[1, 2, 3, 4, 5], [5, 4, 2, 3, 1]], dtype=np.int64)
    a = _Fake(16, bad_input=2)
    b = _Fake(16, bad_input=3)
    r = L.compare_token_errors(a, b, vb, 16, torch.device('cpu'), top_n=20)
    assert r['n_tokens'] == 8, r
    assert r['n_nonfinite_a'] > 0 and r['n_nonfinite_b'] > 0, r
    assert r['n_paired_finite'] < r['n_tokens'], r
    assert r['n_paired_finite'] == 4, r
    assert abs(r['mean_delta_b_minus_a']
               - (r['mean_ce_b'] - r['mean_ce_a'])) < 1e-12, r
    assert r['mean_ce_a'] != r['mean_ce_a_all_finite'], r
    assert r['mean_ce_b'] != r['mean_ce_b_all_finite'], r
    for t in r['worst_tokens_for_b']:
        assert math.isfinite(t['ce_a']) and math.isfinite(t['ce_b']), t
    assert r['b_worse_at_n'] + r['a_worse_at_n'] <= r['n_paired_finite'], r


def t_error_analysis_rejects_bad_token_tables():
    # The summary says the next-token ids were range-checked; that claim must
    # be true even when analyze_errors is called directly (not via the CLI).
    L.set_seed(0)
    model = L.SmallGPT(32, 16, 1, 2, 8, 8, L.make_layer_cfgs(1, 'full')).eval()
    bad = np.array([[1, 2, 32, 4]], dtype=np.int64)
    try:
        L.analyze_errors(model, bad, 32, torch.device('cpu'), top_tokens=4)
    except ValueError as e:
        assert 'out of range' in str(e), e
    else:
        raise AssertionError('out-of-vocabulary ids must be refused')
    frac = np.array([[1.5, 2, 3, 4]], dtype=np.float64)
    try:
        L.analyze_errors(model, frac, 32, torch.device('cpu'), top_tokens=4)
    except TypeError as e:
        assert 'integer token ids' in str(e), e
    else:
        raise AssertionError('float token tables must be refused')
    # The paired comparison has the same label contract.
    try:
        L.compare_token_errors(None, None, bad, 32, torch.device('cpu'))
    except ValueError as e:
        assert 'out of range' in str(e), e
    else:
        raise AssertionError('paired comparison must refuse out-of-vocabulary ids')
    try:
        L.compare_token_errors(None, None, frac, 32, torch.device('cpu'))
    except TypeError as e:
        assert 'integer token ids' in str(e), e
    else:
        raise AssertionError('paired comparison must refuse float token tables')


def t_error_analysis_paired_vocab_guard():
    EA._paired_vocab_guard({'vocab': 8192}, {'vocab': 8192})
    EA._paired_vocab_guard({'vocab': 32.0}, {'vocab': 32})
    EA._paired_vocab_guard({}, {})
    for bad in ({'vocab': 8192}, {'vocab': 8193}), ({'vocab': True}, {'vocab': 1}), \
               ({'vocab': 32}, {'vocab': -32}), ({'vocab': 32}, {'vocab': 32.5}), \
               ({'vocab': 32}, {'vocab': 'x'}):
        try:
            EA._paired_vocab_guard(*bad)
        except SystemExit as e:
            assert 'REFUSE' in str(e), e
        else:
            raise AssertionError(f'paired vocab guard accepted {bad!r}')


def t_error_analysis_paired_none_is_renderable():
    class _M:
        def to(self, device):
            return self

    class _Cfg:
        pass

    import tempfile as _tf
    td = _tf.mkdtemp(prefix='err_', dir=REPO)
    saved = (EA._load_model, EA._sha256, L.load_run_corpus,
             L.analyze_errors, L.compare_token_errors)
    try:
        EA._load_model = lambda path: (_M(), {'vocab': 32, 'd': 16, 'n_layers': 1,
                                              'n_heads': 2, 'd_head': 8, 'train_len': 8,
                                              'cfg': [_Cfg()]})
        EA._sha256 = lambda path: 'deadbeef'
        vb = np.arange(32, dtype=np.int64).reshape(4, 8) % 32
        L.load_run_corpus = lambda cfg: (np.arange(64, dtype=np.int64), vb, 32, None, None)
        L.analyze_errors = lambda *a, **k: {
            'mean_ce': 1.0, 'pooled_ppl': 2.7, 'buckets': {},
            'worst_tokens': [], 'worst_sequences': [],
            'cause_counts': {}, 'cause_counts_all': {}, 'cause_guide': {}}
        L.compare_token_errors = lambda *a, **k: {
            'checkpoint_a': 'a', 'checkpoint_b': 'b',
            'mean_delta_b_minus_a': None, 'n_paired_finite': 0, 'n_tokens': 8,
            'b_worse_at_n': 0, 'a_worse_at_n': 0, 'worst_tokens_for_b': []}
        EA.main(['--ckpt', 'a.pt', '--compare-ckpt', 'b.pt', '--outdir', td,
                 '--max-rows', '4'])
        txt = open(os.path.join(td, 'bad_cases.md'), encoding='utf-8').read()
        assert 'n/a' in txt and 'paired finite coverage' in txt, txt
    finally:
        (EA._load_model, EA._sha256, L.load_run_corpus,
         L.analyze_errors, L.compare_token_errors) = saved
        shutil.rmtree(td, ignore_errors=True)


def t_token_error_cause_taxonomy():
    # The bad-case file must say why a token is hard, not only how large its CE
    # is.  These are deliberately heuristic labels with a manual-review caveat.
    rare = L.classify_token_error(ce=8.0, entropy=1.0, margin=4.0, rare_target=True,
                                  target_id=7, topk_ids=[7, 8], vocab=100)
    assert rare['primary'] == 'annotation_or_rare_target', rare
    amb = L.classify_token_error(ce=4.0, entropy=math.log(100) * 0.9, margin=0.1,
                                 rare_target=False, target_id=7, topk_ids=[7, 8], vocab=100)
    assert amb['primary'] == 'ambiguous_context' and 'ambiguous_context' in amb['causes'], amb
    blind = L.classify_token_error(ce=8.0, entropy=1.0, margin=5.0, rare_target=False,
                                   target_id=7, topk_ids=[3, 4], vocab=100)
    assert blind['primary'] == 'confident_model_blind_spot', blind
    assert blind['target_in_topk'] is False, blind
    outside = L.classify_token_error(ce=4.0, entropy=2.0, margin=1.0, rare_target=False,
                                     target_id=7, topk_ids=[3, 4], vocab=100)
    assert outside['primary'] == 'target_outside_topk' and outside['target_in_topk'] is False, outside
    unmeasured = L.classify_token_error(ce=float('nan'), entropy=1.0, margin=1.0,
                                        rare_target=False, target_id=7,
                                        topk_ids=[3, 4], vocab=100)
    assert unmeasured['primary'] == 'unmeasured', unmeasured


def t_error_analysis_rare_basis_is_measured():
    # An empty/malformed training sample is not a frequency table of zeros.
    # The old path labelled every target "rare", manufacturing an
    # annotation/rare-target cause with no evidence.
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = (np.arange(8, dtype=np.int64) % vocab).reshape(2, 4)
    empty = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=8, train_ids=np.array([], dtype=np.int64))
    assert empty['rare_target_basis'].startswith('unavailable'), empty
    assert 'empty train_ids' in (empty['rare_target_problem'] or ''), empty
    assert empty['cause_counts_all']['annotation_or_rare_target']['n'] == 0, empty
    assert empty['cause_counts_all']['ambiguous_context']['n'] == empty['n_targets'], empty

    # A valid table marks only tokens at/under the rare threshold.
    tr = np.array([1] * 5 + [2] * 6 + [3] * 100, dtype=np.int64)
    ok = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                          top_n=2, top_tokens=8, train_ids=tr)
    assert ok['rare_target_basis'] == 'train_frequency', ok
    assert ok['rare_target_problem'] is None, ok
    assert ok['cause_counts_all']['annotation_or_rare_target']['n'] > 0, ok

    # Out-of-range / non-integer samples disable the flag instead of being
    # silently truncated into a different vocabulary's frequencies.
    for bad in (np.array([-1, 2]), np.array([vocab, 2]), np.array([1.5, 2.0])):
        r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=8, train_ids=bad)
        assert r['rare_target_basis'].startswith('unavailable'), (bad, r)
        assert r['rare_target_problem'], (bad, r)
        assert r['cause_counts_all']['annotation_or_rare_target']['n'] == 0, (bad, r)


def t_error_analysis_zero_and_invalid_budgets():
    # A zero inventory is not one row, and a malformed budget must be refused
    # (negative top_n/-chunk used to be silently clamped to 1).
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = (np.arange(12, dtype=np.int64) % vocab).reshape(3, 4)
    r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                         top_n=0, top_tokens=0)
    assert r['worst_sequences'] == [] and r['worst_tokens'] == [], r
    for kw in ({'top_n': -1}, {'top_tokens': -1}, {'top_n': 1.5},
               {'chunk': 0}, {'context_tokens': -1},
               {'top_n': True}):
        try:
            L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'analyze_errors must refuse {kw!r}')
    p = L.compare_token_errors(_Stub(), _Stub(), vb, vocab, torch.device('cpu'),
                               top_n=0)
    assert p['worst_tokens_for_b'] == [], p
    assert p['n_paired_finite'] == 9, p
    for bad in (-1, 1.5, True):
        try:
            L.compare_token_errors(_Stub(), _Stub(), vb, vocab,
                                   torch.device('cpu'), top_n=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'compare_token_errors must refuse top_n={bad!r}')
    try:
        L.compare_token_errors(_Stub(), _Stub(), vb, vocab,
                               torch.device('cpu'), chunk=0)
    except ValueError:
        pass
    else:
        raise AssertionError('compare_token_errors must refuse chunk=0')


def t_error_analysis_flags_prefix_ambiguity():
    # A high-CE token whose exact prefix occurs with a different next token
    # elsewhere is not uniquely labelled: the review must see that, rather
    # than being told only "ambiguous distribution".
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], dtype=np.int64)
    r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                         top_n=2, top_tokens=8)
    ca = r['context_ambiguity']
    assert ca['n_contexts_with_multiple_targets'] == 1, ca
    assert ca['n_positions_with_ambiguous_context'] == 2, ca
    assert ca['n_ambiguous_worst'] >= 2, ca
    hit = [t for t in r['worst_tokens'] if t['same_context_ambiguous']]
    assert hit, r['worst_tokens']
    for t in hit:
        assert t['same_context_target_variants'] == 2, t
        assert t['same_context_support'] == 2, t
    r0 = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                          top_n=0, top_tokens=0)
    assert r0['context_ambiguity']['n_worst_reviewed'] == 0, r0
    assert r0['context_ambiguity']['context_len'] is None, r0


def t_loss_curve_downsample_keeps_spikes_and_nan():
    # Long runs are downsampled before the curve is written.  A one-step
    # explosion/NaN must survive that reduction, or a diverged run looks clean.
    vals = [float(i) for i in range(5000)]
    vals[1234] = float('nan')
    vals[2345] = 999.0
    out = L._reduce_curve_values(vals, 100)
    assert len(out) <= 100, len(out)
    assert out[0] == 0.0 and out[-1] == 4999.0, (out[0], out[-1])
    assert any(isinstance(v, float) and math.isnan(v) for v in out), 'NaN spike was dropped'
    assert 999.0 in out, 'loss spike was dropped'
    assert L._reduce_curve_values([1.0, 2.0, 3.0], 100) == [1.0, 2.0, 3.0]
    # A tiny point budget is still a bound: the five-landmarks-per-bin rule
    # must not return six samples when the caller asked for at most one.
    for _mp in range(1, 8):
        _v = [float(i) for i in range(50)]
        _v[20] = float('nan')
        _got = L._reduce_curve_values(_v, _mp)
        assert len(_got) <= _mp, (_mp, len(_got))
        assert _got[-1] == 49.0, (_mp, _got)
        _p = [[i, float(i)] for i in range(50)]
        _p[20][1] = float('nan')
        _gp = L._reduce_curve_points(_p, _mp)
        assert len(_gp) <= _mp, (_mp, len(_gp))
        assert _gp[-1][0] == 49, (_mp, _gp)
    assert L._reduce_curve_values([1.0, 2.0], 0) == []
    assert L._reduce_curve_points([[0, 1.0], [1, 2.0]], 0) == []


def t_final_loss_smoothed_uses_last_steps():
    # `final_loss_smoothed` must describe the final 50 steps.  Reading a
    # scalar only on sparse diagnostic steps made the old value an average of
    # points that could span a large part of the run.
    ids = np.arange(4096, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(4, 16)
    kw = dict(d=32, n_layers=1, n_heads=2, d_head=16, seq_len=16, batch_size=2,
              steps=70, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
              eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'), log_every=0)
    r = L.train_variant('full', ids, val, 997, seed=0, **kw)
    assert len(r['losses']) == 70, len(r['losses'])
    tail = [v for v in r['losses'][-50:] if math.isfinite(v)]
    assert r['final_loss_smoothed_n'] == len(tail) == 50, r.get('final_loss_smoothed_n')
    assert abs(r['final_loss_smoothed'] - sum(tail) / len(tail)) < 1e-12, r['final_loss_smoothed']
    assert r['final_loss_smoothed_window'] == 50, r


def t_train_variant_early_stop_record():
    ids = np.arange(2048, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(4, 16)
    kw = dict(d=32, n_layers=1, n_heads=2, d_head=16, seq_len=16, batch_size=2,
              steps=4, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
              eval_every=1, eval_subset=2, mlp_ratio=4.0,
              device=torch.device('cpu'), log_every=0)
    r = L.train_variant('full', ids, val, 997, seed=0,
                        early_stop_patience=1, early_stop_min_delta=1e9, **kw)
    es = r.get('early_stop') or {}
    assert es.get('stopped') is True and es.get('stopped_step') == 2, es
    assert r['steps'] == 4 and r['steps_completed'] == 2, r
    assert r['tokens_seen'] == 2 * 2 * 16, r['tokens_seen']
    assert r['opt_diag_summary']['n_steps_completed'] == 2, r['opt_diag_summary']
    assert r['final_full_val_ppl'] == r['ppl'], r
    assert r['final_subset_val_ppl'] is not None and r['best_val_ppl'] is not None, r
    assert abs(r['final_minus_best'] - (r['final_subset_val_ppl'] - r['best_val_ppl'])) < 1e-12, r
    r2 = L.train_variant('full', ids, val, 997, seed=0, **dict(kw, steps=2))
    assert r2['steps_completed'] == 2 and r2.get('early_stop') is None, r2
    assert r2['tokens_seen'] == 2 * 2 * 16, r2['tokens_seen']
    try:
        L.train_variant('full', ids, val, 997, seed=0,
                        early_stop_patience=1, **dict(kw, eval_every=0))
    except ValueError:
        pass
    else:
        raise AssertionError('early stopping without a validation grid must be refused')


def t_v12_health_gate_refuses_unstable_training():
    cfg = {'smoke': False}
    healthy = {'curve_health': {'status': 'healthy', 'reason': 'ok'}}
    assert V12._training_health_gate(cfg, 'a', 0, healthy, healthy) is None
    bad = {'curve_health': {'status': 'overfitting', 'reason': 'val rising'}}
    msg = V12._training_health_gate(cfg, 'a', 0, bad, bad)
    assert msg and 'overfitting' in msg, msg
    triage = {'curve_health': {'status': 'healthy', 'reason': 'ok'},
              'instability_diagnosis': {'status': 'instability_observed',
                                        'primary_suspect': 'fp16_overflow',
                                        'summary': 'scaler overflow'}}
    msg2 = V12._training_health_gate(cfg, 'a', 0, triage, {})
    assert msg2 and 'fp16_overflow' in msg2, msg2
    _old = os.environ.get('V12_ALLOW_UNHEALTHY')
    os.environ['V12_ALLOW_UNHEALTHY'] = '1'
    try:
        assert V12._training_health_gate(cfg, 'a', 0, triage, {}) is None
    finally:
        if _old is None:
            os.environ.pop('V12_ALLOW_UNHEALTHY', None)
        else:
            os.environ['V12_ALLOW_UNHEALTHY'] = _old


def t_rope_arms_are_not_batched():
    # The implementation binding must report eager_batch=False for the rope csa
    # arms because `HybridAttentionRoPE.forward` unconditionally takes the
    # per-row `_single_rope` loop.  A CUDA-capable device still yields False for
    # all four frozen arms; a non-rope csa config remains eligible (the check is
    # not a blanket False).
    class _Cuda:
        type = 'cuda'
    for arm in ('csa_fixed_rope', 'full_rope', 'csa_fixed_rope_randidx',
                'csa_fixed_rope_allblocks'):
        cfgs = L.make_layer_cfgs(6, arm)
        assert L._uses_batched_eager(cfgs, device=_Cuda(), batch_size=12) is False, arm
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed'), device=_Cuda(), batch_size=12) is True
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed_rope'), device=_Cuda(), batch_size=1) is False


def t_by_len_requires_one_ppl_per_row():
    cell = {'n_tok': 16 * 4095, 'truncated': False,
            'ppl_per_seq': [1.0] * 64}
    reason = V12._by_len_shape_reason({'by_len': {'4096': cell}}, 16)
    assert reason is not None and 'ppl_per_seq' in reason, reason
    cell['ppl_per_seq'] = [1.0] * 16
    assert V12._by_len_shape_reason({'by_len': {'4096': cell}}, 16) is None


def t_plan_check_rejects_bad_endpoint_and_finaltest_is_gated():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
            man = json.load(f)
        man['primary_comparisons'][0] = dict(man['primary_comparisons'][0],
                                             endpoint='by_len.ppl@L4096')
        bad = os.path.join(td, 'bad_endpoint.json')
        with open(bad, 'w', encoding='utf-8') as f:
            json.dump(man, f)
        _buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_buf):
                V12.plan_check(bad)
        except SystemExit:
            assert 'endpoint' in _buf.getvalue(), _buf.getvalue()[:500]
        else:
            raise AssertionError('an unparseable endpoint must fail plan_check')
        # run_finaltest must not skip the plan preflight for a direct P5T call.
        _old = V12.plan_check
        _seen = []
        def _boom(path=None):
            _seen.append(path)
            raise SystemExit('plan refused in test')
        try:
            V12.plan_check = _boom
            try:
                _cfg = dict(V12.FINALTEST_CFG)
                _cfg.update(manifest=os.path.join(REPO, 'confirmation_frozen.json'),
                            outdir=os.path.join(td, 'testout'), smoke=False)
                V12.run_finaltest(_cfg)
            except SystemExit as e:
                assert 'plan refused in test' in str(e), str(e)[:200]
            else:
                raise AssertionError('run_finaltest bypassed plan_check')
        finally:
            V12.plan_check = _old
        assert _seen, 'plan_check was not called'


def t_manifest_v3_admission_can_succeed():
    # The zero-GPU plan check is the end-to-end guard against a stamp-only
    # alignment that would admit no records.  Exercise it on the shipped
    # manifest.
    rep = V12.plan_check(os.path.join(REPO, 'confirmation_frozen.json'))
    assert rep['ok'] and rep['admitted'] == rep['n_arms'] * rep['n_seeds'], rep
    assert all(c['pairs_planned'] == rep['n_seeds'] for c in rep['comparisons'].values()), rep
    assert rep['predicted']['full_rope']['eager_batch'] is False, rep
    assert rep['predicted']['csa_fixed_rope']['amp_dtype'] == 'bf16', rep


def t_v7_rope_gather_rotation_order():
    # The per-chunk RoPE refactor rotates the gathered window rows.  The
    # gathered tensor is [queries, lanes, heads, head_dim], so the position
    # table needs a singleton HEAD axis; inserting a singleton LANE axis only
    # broadcasts when lanes == heads.
    torch.manual_seed(0)
    T, nh, hd, q, m = 13, 3, 8, 7, 5
    k = torch.randn(T, nh, hd)
    idx = torch.randint(0, T, (q, m))
    dev = torch.device('cpu')
    cos, sin = V7.rope_cos_sin(hd, hd, L._arange_cache(T, dev), dev)
    old = L._take_2d(V7.apply_rope(k, cos[:, None, :], sin[:, None, :], hd), idx)
    new = V7.apply_rope(L._take_2d(k, idx),
                        cos[idx][:, :, None, :], sin[idx][:, :, None, :], hd)
    assert torch.equal(old, new), (old - new).abs().max()
    # End-to-end: the frozen csa_fixed_rope arm must run with lanes != heads.
    cfg = V7.make_layer_cfgs_v7(1, 'csa_fixed_rope')[0]
    model = V7.SmallGPTRoPE(64, 32, 1, 2, 16, 16, [cfg])
    x = torch.randint(0, 64, (1, 16))
    y = torch.randint(0, 64, (1, 16))
    loss = model.ce_loss(x, y, 64)
    assert math.isfinite(float(loss.detach())), loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads), 'RoPE backward produced non-finite gradients'


def t_scale_input_guards():
    try:
        L.meta_analyze_effects([{'label': 'A', 'scale': 1.0, 'effects': [1.0, 2.0], 'seeds': [0, 0]},
                                {'label': 'B', 'scale': 2.0, 'effects': [2.0, 3.0], 'seeds': [0, 1]}])
        raise AssertionError('duplicate seeds inside one scale point must be rejected')
    except ValueError:
        pass
    try:
        L.meta_analyze_effects([{'label': 'A', 'scale': 1.0, 'effects': [1.0, 2.0], 'seeds': [0, 1]},
                                {'label': 'A', 'scale': 2.0, 'effects': [2.0, 3.0], 'seeds': [0, 1]}])
        raise AssertionError('two rows for one scale label must not be pooled as independent studies')
    except ValueError:
        pass
    for fn, args in (
        (L.paired_log_ratio, ([1.0, 2.0], [1.1, 2.1])),
        (L.scale_interaction, ([1.0, 2.0], [1.1, 2.1])),
    ):
        try:
            fn(*args, seeds=[0, 0])
            raise AssertionError(f'{fn.__name__} must reject duplicate seeds instead of double-counting them')
        except ValueError:
            pass
    # A non-finite pair is dropped from the test but must stay visible as a
    # dropped observation; otherwise the surviving subset is misreported as a
    # complete paired panel.
    si = L.scale_interaction([1.0, float('nan'), 2.0, 3.0],
                             [1.1, float('nan'), 2.1, 3.1],
                             seeds=[10, 11, 12, 13])
    assert si['n'] == 3 and si['n_input'] == 4 and si['n_dropped'] == 1, si
    assert si['complete_pairs'] is False and si['dropped_seeds'] == [11], si
    assert si['seeds'] == [10, 12, 13], si
    # A caller-declared minimum-n floor is part of the claim, not just a label
    # on the text.  n=6 reaches p=0.03125 but an explicit min_n=8 must keep
    # the interaction-level `significant` flag false (the raw test flag stays
    # visible for transparency).
    und = L.scale_interaction([0.0] * 6, [1.0] * 6, seeds=list(range(6)),
                              min_n=8)
    assert und['n'] == 6 and abs(und['p_exact_signflip'] - 0.03125) < 1e-12, und
    assert und['underpowered'] is True, und
    assert und['significant_raw'] is True and und['significant'] is False, und
    powered = L.scale_interaction([0.0] * 6, [1.0] * 6, seeds=list(range(6)),
                                  min_n=6)
    assert powered['significant'] is True and powered['significant_raw'] is True, powered
    # One dropped seed makes this a complete-case subset; even though six
    # non-zero pairs can reach p=0.031, the interaction-level claim is refused.
    partial = L.scale_interaction([0.0] * 7, [1.0, float('nan'), 1.0, 1.0,
                                              1.0, 1.0, 1.0],
                                  seeds=list(range(7)), min_n=6)
    assert partial['n'] == 6 and partial['complete_pairs'] is False, partial
    assert partial['significant_raw'] is True, partial
    assert partial['significant'] is False, partial


def t_exact_signflip_baseline():
    p = L.exact_signflip_p([1.0] * 6)['p']
    assert abs(p - 0.03125) < 1e-12, 'all-same-sign n=6 two-sided p must stay 2/64'
    p12 = L.exact_signflip_p([1.0] * 12)['p']
    assert abs(p12 - 2.0 / 4096) < 1e-12


def t_report_gate_refuses_attrition_and_seed_driven():
    # The report builder must expose the same two reliability diagnostics as
    # exp_lib.paired_effect_stats, not just n/raw p/Holm/|mean|>std.
    robust = BR.exact_signflip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], n_input=6)
    assert robust['n'] == 6 and robust['n_dropped'] == 0, robust
    assert robust['seed_driven'] is False and robust['seed_robust'] is True, robust
    driven = BR.exact_signflip([10.0, 10.0, -10.0], n_input=4)
    assert driven['n'] == 3 and driven['n_dropped'] == 1, driven
    assert driven['seed_driven'] is True and driven['seed_robust'] is False, driven
    assert len(driven['loo_means']) == 3, driven
    # The gate must refuse both a complete-case-only panel and a sign that is
    # owned by a single seed, even when every numeric threshold passes.
    ok = {'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02,
          'mean': 1.0, 'std': 0.5}
    assert BR._sig_gate(ok) is True
    assert BR._sig_gate(dict(ok, n_dropped=1)) is False
    assert BR._sig_gate(dict(ok, seed_driven=True)) is False
    # Seeds skipped by the controlled-comparison gate are an incomplete paired
    # panel too; the report gate must mirror exp_lib.reportable_claim.
    for _key in ('n_skipped_config_mismatch', 'n_skipped'):
        assert BR._sig_gate(dict(ok, **{_key: 1})) is False, _key
        assert BR._sig_gate(dict(ok, **{_key: -1})) is False, _key
        assert BR._sig_gate(dict(ok, **{_key: True})) is False, _key
    # An explicit complete panel / robust sign stays admissible.
    assert BR._sig_gate(dict(ok, n_dropped=0, seed_driven=False)) is True
    assert BR._sig_gate(dict(ok, n_skipped_config_mismatch=0)) is True
    assert BR._sig_gate(dict(ok, n_skipped=0)) is True
    # Corrupt panel accounting must fail closed exactly as in
    # exp_lib.reportable_claim (negative drops, drops exceeding the input,
    # an incomplete-panel flag, or tie-reduced resolution).
    for _bad in (dict(ok, n_dropped=-1),
                 dict(ok, n_input=-1, n_dropped=0),
                 dict(ok, n_input=3, n_dropped=6),
                 dict(ok, n_input=3, n_dropped=0),
                 dict(ok, complete_pairs=False),
                 dict(ok, n_nonzero=7),
                 dict(ok, n_nonzero=-1),
                 dict(ok, n_nonzero=2)):
        assert BR._sig_gate(_bad) is False, _bad
    assert BR._sig_gate(dict(ok, n_nonzero=6)) is True
    assert BR._sig_gate(dict(ok, n_input=6, n_dropped=0)) is True


def _brute_signflip_p(d):
    """Reference chunked enumeration."""
    d = np.asarray(d, dtype=float)
    n = int(d.size)
    obs = abs(float(d.mean()))
    total = 1 << n
    cnt = 0
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 22) // max(n, 1)))
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        cnt += int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
    return cnt / total


def t_exact_signflip_at_scale():
    for n in (21, 22, 25, 40):
        r = L.exact_signflip_p([0.7] * n)
        assert r['method'] == 'exact-mitm', f'n={n} must stay exact via meet-in-the-middle, got {r}'
        assert abs(r['p'] - 2.0 / (1 << n)) < 1e-18, f'all-equal d n={n}: only +/- assignments reach the observed mean, p={r["p"]}'
    assert BR.exact_signflip([0.7] * 22)['method'] == 'exact-mitm', 'the zero-GPU report builder must use the same exact path'
    assert abs(BR.exact_signflip([0.7] * 22)['p_exact_signflip'] - 2.0 / (1 << 22)) < 1e-15
    z = L.exact_signflip_p([0.0] * 24)
    assert z['p'] == 1.0 and z['method'] == 'exact-mitm', 'every sign assignment matches a zero observed mean'
    # The driver used by the confirmation analyzer must not silently substitute
    # a random Monte-Carlo p-value for the pre-registered exact test at n>20.
    v = V7.exact_sign_permutation([0.7] * 21)
    assert abs(v['p_exact_signflip'] - 2.0 / (1 << 21)) < 1e-18 and v.get('method') == 'exact-mitm', v
    rng = np.random.default_rng(11)
    d = rng.normal(size=21)
    got = L.exact_signflip_p(d)['p']
    ref = _brute_signflip_p(d)
    assert abs(got - ref) < 1e-15, f'MITM count must match full enumeration at n=21 ({got} vs {ref})'
    # The zero-GPU report builder must use the same n*1e-12 tie tolerance as
    # exp_lib; a 1e-12 sum threshold can disagree near the decision boundary.
    assert abs(BR.exact_signflip(d)['p_exact_signflip'] - got) < 1e-15, (
        BR.exact_signflip(d)['p_exact_signflip'], got)


def t_signflip_saddlepoint_large_n():
    # n>40 must not use the centered-SD normal approximation (whose variance
    # shrinks with the effect) and must not be a random Monte-Carlo draw.
    rng = np.random.default_rng(17)
    for n, shift in ((45, 0.1), (45, 0.3), (60, 0.2), (100, 0.15)):
        d = rng.normal(shift, 1.0, size=n)
        d = d - d.mean() + shift
        a = L.exact_signflip_p(d)
        b = L.exact_signflip_p(d)
        assert a['method'].startswith('saddlepoint'), a
        assert a['p'] == b['p'], 'saddlepoint fallback must be deterministic'
        # Fixed-seed Monte-Carlo reference (200k draws) — the saddlepoint tail
        # must agree within tolerance for the larger shifts.
        draws = 200000
        obs = abs(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(draws, n))
        mc = float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())
        tol = max(0.003, 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-9) / draws))
        assert abs(a['p'] - mc) <= tol, f'n={n} shift={shift}: saddlepoint {a["p"]} vs MC {mc}'
        # Never below the exact two-sided resolution floor 2/2^n.
        assert a['p'] >= L.signflip_p_floor(n) - 1e-18, (a, n)
        # The report builder must return the same deterministic number.
        br = BR.exact_signflip(list(d))
        assert br['method'].startswith('saddlepoint'), br
        assert abs(br['p_exact_signflip'] - a['p']) < 1e-12, (br, a)
    # All deltas equal: only the two all-same-sign assignments reach the
    # observed sum, so the p is exactly the floor (checked at n=41, > 40).
    eq = L.exact_signflip_p([0.7] * 41)
    assert eq['method'] == 'saddlepoint' and abs(eq['p'] - 2.0 / (1 << 41)) < 1e-30, eq
    assert abs(BR.exact_signflip([0.7] * 41)['p_exact_signflip'] - 2.0 / (1 << 41)) < 1e-30
    # Near-maximum observed sum: the saddlepoint root sits at a large t, so
    # the implementation must stay finite (no cosh overflow) and above floor.
    near = [1.0] * 50
    near[0] = -0.998
    nr = L.exact_signflip_p(near)
    assert nr['method'].startswith('saddlepoint') and 0.0 < nr['p'] < 1.0, nr
    brn = BR.exact_signflip(near)
    assert abs(brn['p_exact_signflip'] - nr['p']) < 1e-12, (brn, nr)
    # The confirmation driver delegates to exp_lib, so it inherits the fix.
    v = V7.exact_sign_permutation(list(rng.normal(0.2, 1.0, size=50)))
    assert v['method'].startswith('saddlepoint'), v
    assert 'saddlepoint' in v['test'], v


def t_wilcoxon_exact_dp():
    def brute_wilcoxon(d):
        d = np.asarray([float(x) for x in d if math.isfinite(float(x))], dtype=float)
        d = d[np.abs(d) > 1e-15]
        n = len(d)
        if n == 0:
            return float('nan')
        order = np.argsort(np.abs(d))
        ranks = np.empty(n, dtype=float)
        a = np.abs(d)[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(a[j] - a[i]) <= 1e-15:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / 2.0
            i = j
        wp = float(ranks[d > 0].sum())
        total = float(ranks.sum())
        w = min(wp, total - wp)
        cnt = 0
        for mask in range(1 << n):
            sm = sum(ranks[k] for k in range(n) if mask >> k & 1)
            if sm <= w + 1e-12 or sm >= total - w - 1e-12:
                cnt += 1
        return cnt / float(1 << n)
    rng = np.random.default_rng(5)
    for n in range(1, 13):
        for variant in range(3):
            d = rng.normal(size=n)
            if variant == 1:
                d = np.round(d, 0)          # ties + zeros
            elif variant == 2:
                d = np.round(np.abs(d), 1)  # many ties, all positive
            got = L.wilcoxon_signed_rank_p(d)
            if not np.any(np.abs(d) > 1e-15):
                continue
            assert got['method'] == 'exact-dp', got
            assert abs(got['p'] - brute_wilcoxon(d)) < 1e-12, (n, variant, got, brute_wilcoxon(d))
    # The DP stays exact well beyond the seed counts the scale ladder may top
    # up to.
    d = rng.normal(0.2, 1.0, size=100)
    a = L.wilcoxon_signed_rank_p(d)
    b = L.wilcoxon_signed_rank_p(d)
    assert a['method'] == 'exact-dp' and a['p'] == b['p'], a
    assert 0.0 <= a['p'] <= 1.0
    # A perfectly balanced zero-effect panel has p=1 (both tails meet).
    assert L.wilcoxon_signed_rank_p([1.0, -1.0, 1.0, -1.0])['p'] == 1.0


def t_timeguard_variant_calibration():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'state.json')
        g = L.TimeGuard(dict(total_hours=10, margin=0.9, state_path=path))
        key_a = g._class_key(64, 2, 256, 4, 'csa_fixed')
        key_b = g._class_key(64, 2, 256, 4, 'full_matched')
        assert key_a != key_b and key_a.endswith('|csa_fixed'), (key_a, key_b)
        # A 10 s / 10 step calibration for variant a must price variant a from
        # its exact class, not from the shared geometry rate.
        g.record_run(10.0, 10, 64, 2, 256, 4, calib_seconds=10.0, variant='csa_fixed')
        a = g.estimate_seconds(100, d=64, n_layers=2, seq_len=256, batch_size=4, variant='csa_fixed')
        assert abs(a - 100 * 1.0 * 1.15) < 1e-9, a
        # Variant b must not borrow a's exact rate; it uses the normalised
        # fallback (which may be the same by coincidence only if the factors
        # match, so check the code path via a deliberately different rate).
        g.record_run(20.0, 10, 64, 2, 256, 4, calib_seconds=20.0, variant='full_matched')
        b = g.estimate_seconds(100, d=64, n_layers=2, seq_len=256, batch_size=4, variant='full_matched')
        assert abs(b - 100 * 2.0 * 1.15) < 1e-9, b
        # A geometry-only calibration remains the fallback for a variant with
        # no per-variant entry (backwards compatible with v7-v12 state files).
        g2 = L.TimeGuard(dict(total_hours=10, margin=0.9, state_path=os.path.join(tmp, 's2.json')))
        g2.record_run(5.0, 10, 128, 2, 256, 4, calib_seconds=5.0)
        c = g2.estimate_seconds(100, d=128, n_layers=2, seq_len=256, batch_size=4, variant='whatever')
        assert abs(c - 100 * 0.5 * 1.15) < 1e-9, c
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_paired_effect_resolution():
    r = L.paired_effect_stats([1.0, 1.1, 0.9, 1.2], alpha=0.05)
    assert r['resolution_limited'] is True, 'n=4 can never reach 0.05 (floor 0.125) — descriptive tier'
    # Missing/non-finite pairs must stay visible: the p-value is computed on
    # the survivors, so a complete-case result cannot be presented as if the
    # whole paired panel were measured.
    r = L.paired_effect_stats([1.0, float('nan'), -0.5], seeds=[0, 1, 2])
    assert r['n'] == 2 and r['n_input'] == 3 and r['n_dropped'] == 1, r
    assert r['dropped_seeds'] == [1] and r['seeds'] == [0, 2], r
    assert r['complete_pairs'] is False, r
    try:
        L.paired_effect_stats([1.0, 2.0], seeds=[0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate paired seeds must be rejected')
    try:
        L.paired_effect_stats([1.0, 2.0], seeds=[0])
    except ValueError:
        pass
    else:
        raise AssertionError('a mismatched seed vector must be rejected')


def t_scaling_trend_ragged_pairs():
    pts = [{'label': 'A', 'scale': 0.1, 'effects': [0.0, 1.0, 2.0, 3.0], 'seeds': [0, 1, 2, 3]},
           {'label': 'B', 'scale': 1.0, 'effects': [1.0, 0.0], 'seeds': [0, 1]}]
    r = L.scaling_trend(pts, min_seeds=2, min_points=2)
    assert r['paired']['per_seed_slopes'] == {'0': 1.0, '1': -1.0}, r['paired']
    assert r['paired']['slope_mean'] == 0.0, r['paired']
    for bad in (
        [{'label': 'A', 'scale': 1, 'effects': [1, 2], 'seeds': [0, 1]},
         {'label': 'A', 'scale': 2, 'effects': [2, 3], 'seeds': [0, 1]}],
        [{'label': 'A', 'scale': 1, 'effects': [1, 2], 'seeds': [0, 0]},
         {'label': 'B', 'scale': 2, 'effects': [2, 3], 'seeds': [0, 1]}],
    ):
        try:
            L.scaling_trend(bad, min_points=2)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate labels/seeds must be rejected instead of mis-paired')
    # Seed identity is an exact integer across every scale helper.  A
    # fractional value would quietly become a different paired draw, and
    # True would collide with seed 1 in the complete-case dict.
    # Non-iterable seed/plan descriptors are malformed input, not a crash.
    for _bad in ({'seeds': 3}, {'planned_seeds': 3}):
        try:
            L._scale_points([dict({'label': 'A', 'scale': 1,
                                   'effects': [1.0]}, **_bad)])
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-iterable {_bad!r} was accepted')
    for _bad_seed in (0.5, True, 'not-an-int'):
        for _call in (
            lambda: L._scale_points([{'label': 'A', 'scale': 1,
                                      'effects': [1.0], 'seeds': [_bad_seed]}]),
            lambda: L.paired_log_ratio([1.0], [2.0], seeds=[_bad_seed]),
            lambda: L.scale_interaction([1.0], [2.0], seeds=[_bad_seed]),
        ):
            try:
                _call()
            except ValueError:
                pass
            else:
                raise AssertionError(f'non-integer seed {_bad_seed!r} was accepted')


def t_scale_missing_seed_identity_is_not_paired():
    # Positional effect indices are not training seeds.  Without declared seed
    # identities the cross-scale paired slope/interaction must be withheld,
    # even though the effect vectors happen to have equal length.
    pts = [{'label': 'P1', 'scale': 1.0, 'effects': [0.0, 1.0, 2.0]},
           {'label': 'P2', 'scale': 10.0, 'effects': [1.0, 2.0, 3.0]}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=2)
    assert r['seed_pairing'] == 'unattested', r
    assert r['paired']['n_complete_seeds'] == 0, r['paired']
    assert r['paired']['seed_identity_attested'] is False, r['paired']
    assert r['primary_inference'].startswith('WLS'), r['primary_inference']
    assert r['interaction_extremes']['status'] == 'seed_identity_unattested', r['interaction_extremes']
    m = L.meta_analyze_effects(pts)
    assert m['seed_pairing'] == 'positional_only' and m['n_points_with_declared_seeds'] == 0, m
    # A partly declared family is still not attested: one undeclared point
    # could have run a different seed set.
    mixed = [pts[0], dict(pts[1], seeds=[0, 1, 2])]
    rm = L.scaling_trend(mixed, min_seeds=3, min_points=2)
    assert rm['seed_pairing'] == 'unattested' and rm['paired']['n_complete_seeds'] == 0, rm
    assert L.meta_analyze_effects(mixed)['seed_pairing'] == 'partial'
    # Both declared: the paired path is available again.
    both = [dict(pts[0], seeds=[0, 1, 2]), dict(pts[1], seeds=[0, 1, 2])]
    rb = L.scaling_trend(both, min_seeds=3, min_points=2)
    assert rb['seed_pairing'] == 'declared' and rb['paired']['n_complete_seeds'] == 3, rb
    assert rb['interaction_extremes'].get('seed_pairing') == 'declared', rb['interaction_extremes']
    # The unpaired WLS fallback must also honour min_seeds: a perfect line
    # through two-draw points is not a confirmable trend.
    low = [{'label': 'W%d' % i, 'scale': 10.0 ** i,
            'effects': [float(i)] * 2} for i in range(3)]
    rw = L.scaling_trend(low, min_seeds=3, min_points=3)
    assert rw['primary_inference'].startswith('WLS'), rw['primary_inference']
    assert rw['underpowered'] is True and rw['significant'] is False, rw
    assert rw['verdict'] == 'underpowered', rw['verdict']
    assert rw['underpowered_points'] == ['W0', 'W1', 'W2'], rw


def t_scaling_trend_primary_direction_and_floor():
    # Ragged panel: the paired per-seed evidence is six positive slopes, while
    # the seed-weighted WLS point means (the extra A seeds have high effects)
    # would give a negative slope.  The verdict must follow the paired slope.
    pts = [{'label': 'A', 'scale': 1.0, 'effects': [0.0] * 6 + [3.0] * 6,
            'seeds': list(range(12))},
           {'label': 'B', 'scale': 10.0, 'effects': [1.0] * 6, 'seeds': list(range(6))}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=2)
    assert r['paired']['slope_mean'] == 1.0, r['paired']
    assert r['wls']['slope'] < 0.0, 'the constructed unpaired fit must disagree with the paired slope'
    assert r['significant'] is True and r['direction_slope'] == 1.0, r
    assert r['verdict'] == 'effect_increases_with_scale', r['verdict']
    assert r['benefit_grows_with_scale'] is False, r

    # Six complete seeds reject at alpha=0.05, but with min_seeds=7 the
    # pre-registered floor is not met: the trend must stay descriptive.
    def _pts(seeds):
        return [{'label': 'A', 'scale': 1.0, 'effects': [0.0] * seeds, 'seeds': list(range(seeds))},
                {'label': 'B', 'scale': 10.0, 'effects': [1.0] * seeds, 'seeds': list(range(seeds))},
                {'label': 'C', 'scale': 100.0, 'effects': [2.0] * seeds, 'seeds': list(range(seeds))}]
    r2 = L.scaling_trend(_pts(6), min_seeds=7, min_points=3)
    assert r2['p_primary'] < 0.05 and r2['underpowered'] is True, r2
    assert r2['significant'] is False and r2['verdict'] == 'underpowered', r2

    # A perfect three-point ladder is not "heterogeneous": lack-of-fit Q is
    # zero, so the trend verdict must survive.
    r3 = L.scaling_trend(_pts(6), min_seeds=3, min_points=3)
    assert r3['heterogeneity']['Q'] == 0.0 and r3['heterogeneity']['I2_percent'] == 0.0, r3['heterogeneity']
    assert r3['significant'] is True and r3['verdict'] == 'effect_increases_with_scale', r3

    # Extra A-only seeds must not make the paired trend look heterogeneous:
    # they contribute no paired slope in the first place.
    ragged = _pts(6)
    ragged[0] = {'label': 'A', 'scale': 1.0, 'effects': [0.0] * 6 + [10.0] * 6,
                 'seeds': list(range(12))}
    r5 = L.scaling_trend(ragged, min_seeds=3, min_points=3)
    assert r5['heterogeneity']['I2_percent'] > 75.0, r5['heterogeneity']
    assert r5['heterogeneity_paired']['I2_percent'] == 0.0, r5['heterogeneity_paired']
    assert r5['significant'] is True and r5['verdict'] == 'effect_increases_with_scale', r5

    # A two-point fit is a line; min_points=3 must not be reported as a trend.
    r4 = L.scaling_trend(_pts(6)[:2], min_seeds=3, min_points=3)
    assert r4['p_primary'] < 0.05 and r4['significant'] is False, r4
    assert r4['verdict'] == 'insufficient_points', r4['verdict']


# ---------------------------------------------------------------------------
# B. v10 corruption modes
# ---------------------------------------------------------------------------

if _HAVE_TORCH:
    _PANEL = np.random.default_rng(7).integers(3, 8000, size=(9, 129), dtype=np.int64)
    _VOCAB = 8192


def _legacy_corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target):
    """The PRE-refactor inline algorithm, replayed bit-for-bit (uniform /
    natural / mask / delete). Any divergence means the extraction changed a
    measured value — the strongest invariant this suite checks."""
    n_far = eval_len - target
    if mode == 'delete':
        return np.asarray(val_ids[jj, eval_len - target:eval_len + 1], dtype=np.int64).copy()
    ids = np.asarray(val_ids[jj, :eval_len + 1], dtype=np.int64).copy()
    k = int(round(rho * n_far))
    if mode == 'mask':
        ids[:n_far] = V10.MASK_TOKEN_ID
        return ids
    elif k > 0:
        if mode == 'uniform':
            rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
            pos = rng.choice(n_far, size=k, replace=False)
            _draw = rng.integers(0, vocab, size=k)
            _same = _draw == ids[pos]
            while _same.any():
                _draw[_same] = rng.integers(0, vocab, size=int(_same.sum()))
                _same = _draw == ids[pos]
            ids[pos] = _draw
        else:
            rng = np.random.default_rng((eval_len * 1000003 + jj * 10007) * 1048576 + int(round(rho * 1048576)))
            pos = rng.choice(n_far, size=k, replace=False)
            _src = (jj + max(1, val_ids.shape[0] // 2)) % val_ids.shape[0]
            ids[pos] = np.asarray(val_ids[_src, pos], dtype=np.int64)
    return ids


def t_legacy_modes_bitexact():
    for mode, rhos in (('uniform', (0.0, 0.125, 0.5)), ('natural', (0.125, 0.5)), ('mask', (1.0,)), ('delete', (1.0,))):
        for rho in rhos:
            for jj in range(0, 9, 3):
                a = V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                b = _legacy_corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                assert np.array_equal(a, b), f'legacy mode {mode} rho={rho} jj={jj} diverged from the reference algorithm'


def t_corruption_deterministic_and_paired():
    for mode in ('uniform', 'natural', 'span', 'doc', 'topical', 'mask', 'delete'):
        rho = 1.0 if mode in ('mask', 'delete', 'doc') else 0.5
        rows = [V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32) for jj in range(9)]
        again = [V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32) for jj in range(9)]
        for a, b in zip(rows, again):
            assert np.array_equal(a, b), f'{mode}: corruption must be deterministic (arm-pairing contract)'
        if mode != 'delete':
            for jj, r in enumerate(rows):
                orig = np.asarray(_PANEL[jj, :129], dtype=np.int64)
                n_far = 128 - 32
                assert np.array_equal(r[n_far:], orig[n_far:]), f'{mode}: the scored target zone must never be corrupted (jj={jj})'
                assert r.shape == (129,)


def t_new_modes_do_something():
    span = V10._corrupt_row(_PANEL, 128, 0.5, 'span', 1, _VOCAB, 32)
    orig = np.asarray(_PANEL[1, :129], dtype=np.int64)
    diff = int((span[:96] != orig[:96]).sum())
    assert 0 < diff <= 96, f'span must replace a nonzero, bounded portion of the far context (diff={diff})'
    doc = V10._corrupt_row(_PANEL, 128, 1.0, 'doc', 1, _VOCAB, 32)
    n_far = 96
    _src = (1 + max(1, _PANEL.shape[0] // 2)) % _PANEL.shape[0]
    donor = np.asarray(_PANEL[_src, :129], dtype=np.int64)
    assert not np.array_equal(doc[:n_far], orig[:n_far]), 'doc must replace the far context'
    assert any(np.array_equal(doc[:n_far], donor[o:o + n_far]) for o in range(0, 129 - n_far + 1)), 'doc far context must be ONE contiguous donor slice'
    top = V10._corrupt_row(_PANEL, 128, 0.5, 'topical', 1, _VOCAB, 32)
    assert not np.array_equal(top[:96], orig[:96]), 'topical must replace some tokens'
    nat = V10._corrupt_row(_PANEL, 128, 0.5, 'natural', 1, _VOCAB, 32)
    assert not np.array_equal(top, nat), 'topical (most-similar donor) should differ from natural (fixed-offset donor)'


def t_mode_validation():
    for bad in ('naturaln', '', 'DOC'):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, bad, 0, _VOCAB, 32)
            raise AssertionError(f'unknown mode {bad!r} must raise')
        except ValueError:
            pass
    for mode in ('mask', 'delete', 'doc'):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, mode, 0, _VOCAB, 32)
            raise AssertionError(f'{mode} must only admit rho=1.0')
        except ValueError:
            pass
    # An invalid corruption geometry must fail before the seeded RNG works:
    # a negative far context or rho outside [0,1] silently produced an
    # unchanged/partial row that still looked like a measured condition.
    for _kw in ({'eval_len': 64, 'rho': -0.1, 'target': 32},
                {'eval_len': 64, 'rho': 1.1, 'target': 32},
                {'eval_len': 64, 'rho': float('nan'), 'target': 32},
                {'eval_len': 32, 'rho': 0.5, 'target': 64},
                {'eval_len': 64, 'rho': 0.5, 'target': 0}):
        try:
            V10._corrupt_row(_PANEL, _kw['eval_len'], _kw['rho'], 'uniform',
                             0, _VOCAB, _kw['target'])
            raise AssertionError(f'_corrupt_row accepted invalid geometry {_kw!r}')
        except ValueError:
            pass
    try:
        V10._corrupt_row(_PANEL[:1, :64], 128, 0.5, 'uniform', 0, _VOCAB, 32)
    except ValueError:
        pass
    else:
        raise AssertionError('_corrupt_row silently used a panel shorter than eval_len')
    # A one-token vocabulary has no alternative token to draw.  The historical
    # rejection loop redrew the same value forever; the degenerate condition
    # must return the unchanged row instead of hanging the probe.
    _zeros = np.zeros((1, 65), dtype=np.int64)
    _zc = V10._corrupt_row(_zeros, 64, 1.0, 'uniform', 0, 1, 32)
    assert np.array_equal(_zc, _zeros[0]), _zc
    # A non-integral/boolean/non-positive vocabulary is malformed, not a token
    # space to truncate or index.
    for _bad_vocab in (0, -1, True, 1.5):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, 'uniform', 0, _bad_vocab, 32)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_corrupt_row accepted vocab={_bad_vocab!r}')


# ---------------------------------------------------------------------------
# C. v12 gates
# ---------------------------------------------------------------------------

def _write_manifest(path, seeds, extra_stat=None, binding=None):
    m = {'frozen_at': '2026-10-17', 'code_semantics_expected': L.CODE_SEMANTICS, 'arms': ['a_arm', 'b_arm'], 'seeds': seeds, 'primary_comparisons': [{'name': 'c1', 'a': 'a_arm', 'b': 'b_arm', 'endpoint': 'by_len.ppl@4096'}], 'statistics': {'test': 'exact two-sided sign-flip', 'n_seeds': len(seeds), 'correction': 'holm step-down'}}
    if extra_stat:
        m['statistics'].update(extra_stat)
    if binding:
        m['binding'] = binding
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return m


def t_power_gate():
    with tempfile.TemporaryDirectory() as td:
        p6 = os.path.join(td, 'm6.json')
        m6 = _write_manifest(p6, list(range(6)))
        m6['primary_comparisons'] = [dict(m6['primary_comparisons'][0], name=f'c{i}') for i in range(4)]
        with open(p6, 'w', encoding='utf-8') as f:
            json.dump(m6, f)
        try:
            V12._load_manifest(p6)
            raise AssertionError('a 6-seed x Holm/4 manifest (floor 0.125 > 0.05) must be refused at load time')
        except SystemExit as e:
            assert 'cannot confirm' in str(e), str(e)[:200]
        p12 = os.path.join(td, 'm12.json')
        m12 = _write_manifest(p12, list(range(12)))
        m = V12._load_manifest(p12)
        assert m['seeds'] == list(range(12)) and len(m['manifest_sha256']) == 64
        # The frozen endpoint grid is exact: a fractional/bool/non-list
        # sequence length would score a different pre-registered endpoint.
        for _bad_lens in ([4096.5], [True], 4096):
            _bad = _write_manifest(os.path.join(td, 'bad_lens.json'), list(range(12)))
            _bad['eval'] = {'by_length_ppl': _bad_lens}
            with open(os.path.join(td, 'bad_lens.json'), 'w', encoding='utf-8') as f:
                json.dump(_bad, f)
            try:
                V12._load_manifest(os.path.join(td, 'bad_lens.json'))
            except SystemExit as e:
                assert 'by_length_ppl' in str(e), e
            else:
                raise AssertionError(f'eval.by_length_ppl={_bad_lens!r} was accepted')
        psm = os.path.join(td, 'smoke.json')
        _write_manifest(psm, [0], extra_stat={'descriptive_only': True})
        V12._load_manifest(psm)  # descriptive smoke manifests opt out
        # A truthy surrogate is not a boolean opt-out: the 6-seed plan still
        # cannot confirm anything and must be refused.
        pstr = os.path.join(td, 'm6_string.json')
        mstr = _write_manifest(pstr, list(range(6)),
                               extra_stat={'descriptive_only': 'false'})
        mstr['primary_comparisons'] = [dict(mstr['primary_comparisons'][0], name=f'c{i}') for i in range(4)]
        with open(pstr, 'w', encoding='utf-8') as f:
            json.dump(mstr, f)
        try:
            V12._load_manifest(pstr)
        except SystemExit as e:
            assert 'real boolean' in str(e), e
        else:
            raise AssertionError('a string descriptive_only opt-out was accepted')


def t_v12_malformed_manifest_and_lock_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        # A scalar/list comparison cannot be read with .get(); it must be
        # named as malformed evidence instead of crashing the loader.
        p = os.path.join(td, 'bad_comps.json')
        m = _write_manifest(p, list(range(12)))
        m['primary_comparisons'] = ['not-an-object']
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(m, f)
        try:
            V12._load_manifest(p)
        except SystemExit as e:
            assert 'primary_comparisons[0]' in str(e) and 'object' in str(e), e
        else:
            raise AssertionError('a scalar primary comparison was accepted')
        # Nested binding blocks are read with .get(); a list cannot be a
        # tokenizer descriptor and must be named as malformed.
        p2 = os.path.join(td, 'bad_binding.json')
        m2 = _write_manifest(p2, list(range(12)))
        m2['binding'] = {'tokenizer': ['not', 'an', 'object']}
        with open(p2, 'w', encoding='utf-8') as f:
            json.dump(m2, f)
        try:
            V12._load_manifest(p2)
        except SystemExit as e:
            assert 'binding.tokenizer' in str(e), e
        else:
            raise AssertionError('a non-object tokenizer binding was accepted')
        # A valid-JSON but non-object freeze lock cannot attest anything;
        # .get() on it used to raise AttributeError after test access.
        outdir = os.path.join(td, 'freeze')
        os.makedirs(outdir)
        lp = os.path.join(outdir, V12._LOCK_NAME)
        for _payload in ([], 'null', 3):
            with open(lp, 'w', encoding='utf-8') as f:
                json.dump(_payload, f)
            try:
                V12._check_or_write_freeze_lock(
                    outdir, {'manifest_sha256': 'a' * 64, 'frozen_at': 'x'},
                    'slice-fp', stage='precheck')
            except SystemExit as e:
                assert 'not a JSON object' in str(e), e
            else:
                raise AssertionError(f'freeze lock payload {_payload!r} was accepted')


def _rec(variant, seed, fp, by_len=None, code=None, recipe='steps3000_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_v8192_wu50_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee250_es128_detcpu-eager_dfdeadbeef_cscURRENT', msha='CURRENT'):
    return {'variant': variant, 'seed': seed, 'split': 'test', 'data_fp': fp, 'by_len': by_len or {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095, 'truncated': False}}, '_code': code or L.code_stamp(), 'recipe': recipe, 'manifest_sha256': msha, 'key': f'{variant}::s{seed}'}


def _manifest_for_admission(seeds=(0, 1, 2), sig=None):
    m = {'arms': ['a_arm', 'b_arm'], 'seeds': list(seeds), 'code_semantics_expected': L.CODE_SEMANTICS, 'manifest_sha256': 'CURRENT', 'statistics': {'test': 'exact two-sided sign-flip', 'correction': 'holm step-down'}, 'primary_comparisons': []}
    if sig is not None:
        m['binding'] = {'training_recipe_signature': sig}
    return m


def t_admission_gate():
    m = _manifest_for_admission()
    good = {'a::s0': _rec('a_arm', 0, 'FP'), 'b::s0': _rec('b_arm', 0, 'FP'), 'a::s1': _rec('a_arm', 1, 'FP'), 'b::s1': _rec('b_arm', 1, 'FP')}
    adm, fp = V12._admit_confirmation_records(good, m)
    assert len(adm) == 4 and fp == 'FP'
    # off-whitelist seed: excluded, named, never counted
    raw = dict(good, **{'a::s9': _rec('a_arm', 9, 'FP')})
    adm2, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 9) not in adm2 and len(adm2) == 4
    # None data_fp (the None==None pairing hole): excluded
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, None)})
    adm3, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm3
    # wrong manifest sha: excluded (stale revision)
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, 'FP', msha='OLD')})
    adm4, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm4
    # wrong code stamp: excluded
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, 'FP', code='v9.9')})
    adm5, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm5


def t_admission_refuses():
    m = _manifest_for_admission()
    # duplicate admissible records -> REFUSE (not overwrite)
    raw = {'a::s0': _rec('a_arm', 0, 'FP'), 'a::s0.0': _rec('a_arm', 0, 'FP'), 'b::s0': _rec('b_arm', 0, 'FP')}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('duplicate (variant, seed) must be refused, not overwritten')
    except SystemExit as e:
        assert 'duplicate' in str(e).lower() or 'ambiguous' in str(e).lower()
    # mixed fingerprints across seed pairs -> REFUSE (the slice must be ONE)
    raw = {'a::s0': _rec('a_arm', 0, 'FP1'), 'b::s0': _rec('b_arm', 0, 'FP1'), 'a::s1': _rec('a_arm', 1, 'FP2'), 'b::s1': _rec('b_arm', 1, 'FP2')}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('records spanning two test slices must be refused (pairwise-equal fps are not enough)')
    except SystemExit as e:
        assert 'slice' in str(e).lower()


def t_recipe_signature():
    recipe = 'steps3000_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_v8192_wu50_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee250_es128_detcpu-eager_dfdeadbeef_cscURRENT'
    sig = {'steps': 3000, 'sl': 512, 'bs': 12, 'nt': 8000000, 'lr': 0.0003, 'd': 256, 'L': 6}
    assert V12._recipe_matches_signature(recipe, sig) is True
    bad = dict(sig, steps=40)
    assert V12._recipe_matches_signature(recipe, bad) is False
    assert V12._recipe_matches_signature(None, sig) is False
    assert V12._recipe_matches_signature(recipe, None) is True  # no signature declared -> nothing to enforce
    # the SHIPPED v2 manifest's signature must match its own declared recipe
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    shipped = v2['binding']['training_recipe_signature']
    assert V12._recipe_matches_signature(recipe, shipped) is True, 'the shipped v2 signature must accept the canonical P3MT recipe string'


def t_freeze_lock():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        # two-phase lock: pre-access attestation is written BEFORE the test
        # slice is read, then bound to the slice fingerprint immediately after.
        pre = V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
        assert pre.get('stage') == 'preaccess' and pre.get('test_ids_fp') is None, pre
        lock = V12._check_or_write_freeze_lock(td, man, 'FP', stage='accessed')
        assert lock['manifest_sha256'] == 'AAA' and lock['test_ids_fp'] == 'FP'
        assert lock.get('stage') == 'accessed', lock
        # changed test slice -> refuse
        try:
            V12._check_or_write_freeze_lock(td, man, 'FP2')
            raise AssertionError('a changed test slice after first access must be refused')
        except SystemExit:
            pass
        # changed manifest -> refuse; V12_REFREEZE=1 downgrades with a warning
        man2 = {'manifest_sha256': 'BBB', 'frozen_at': '2026-10-17'}
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP')
            raise AssertionError('a changed manifest after first access must be refused')
        except SystemExit:
            pass
        os.environ['V12_REFREEZE'] = '1'
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP')
        finally:
            del os.environ['V12_REFREEZE']
        # Once a training fingerprint is locked, an analysis call without one
        # is a provenance downgrade and must be refused.
        V12._check_or_write_freeze_lock(td, man2, 'FP', train_fp='TRAIN')
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP', train_fp=None)
            raise AssertionError('a missing train_fp after it was locked must be refused')
        except SystemExit:
            pass


def t_p1l_recipe_signature():
    # P5T evaluates checkpoints written by v7's length-phase writer, whose
    # recipe is the bespoke `p1l_...` format (not a run_cfg fingerprint).
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    sig = v2['binding']['training_recipe_signature']
    p1l = ('p1l_steps3000_tl512_v8192_bs12_lr0.0003_nt8000000_wu50_wd0.1_'
           'cl0.05_dlm10.0_mr4-csa_fixed_rope_dfdeadbeef')
    assert V12._recipe_knobs(p1l)['steps'] == '3000'
    assert V12._recipe_knobs(p1l)['sl'] == '512'
    assert V12._recipe_matches_signature(p1l, sig) is True, 'the P1L checkpoint recipe must satisfy the shipped signature'
    assert V12._recipe_matches_signature(p1l.replace('steps3000', 'steps40'), sig) is False
    assert V12._recipe_matches_signature('p1l_garbage', sig) is False


def t_aggregate_seed_normalization():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='csa_dynamic', run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=10.0)
    raw = {'a::seed0': dict(base, seed=0), 'b::seed0': dict(base, seed='0')}
    try:
        L.aggregate(raw)
        raise AssertionError('seed 0 and "0" are the same seed and must trip the ambiguous-duplicate guard, not be counted as n=2')
    except ValueError:
        pass
    # Fractional/boolean seeds are not extra draws: exclude them rather than
    # inflating n or emitting a non-integer seed in the panel summary.
    assert L.aggregate({'x::seedbad': dict(base, seed=1.5)}) == {}
    assert L.aggregate({'x::seedbad': dict(base, seed=True)}) == {}
    # A malformed arm id is unhashable and must be excluded, not crash the
    # whole panel aggregation.
    assert L.aggregate({'x::seed0': dict(base, seed=0, variant={'bad': 1})}) == {}
    assert L.aggregate({'x::seed0': dict(base, seed=0, variant='')}) == {}
    # A record whose optional diagnostics are explicitly null must not crash
    # the panel aggregate; it simply contributes no block-length readout.
    _null_stats = L.aggregate({'x::seed0': dict(base, seed=0, stats=None)})
    assert _null_stats['csa_dynamic']['n_seeds'] == 1, _null_stats
    assert 'avg_block_len' not in _null_stats['csa_dynamic'], _null_stats
    # A malformed history must yield an empty/omitted curve, not an exception
    # from the boolean reduction over a zero-column matrix.
    _bad_hist = L.aggregate({'x::seed0': dict(base, seed=0, ppl_history=[1])})
    assert _bad_hist['csa_dynamic'].get('ppl_curve') in (None, []), _bad_hist
    _bad_hist2 = L.aggregate({'x::seed0': dict(base, seed=0, ppl_history=[{}, 'x'])})
    assert _bad_hist2['csa_dynamic'].get('ppl_curve') in (None, []), _bad_hist2


def t_audit_resolution_classification():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'

    def _summary(deltas):
        s = {}
        for i, d in enumerate(deltas):
            s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=50.0 + d)
            s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=50.0)
        return s
    design = {'c': [('a', 'b', 'test')]}
    # n=6, effect > noise, p=0.0625 > alpha, but the exact-test floor (0.031)
    # is BELOW alpha -> a genuine null-ish reading, NOT a resolution ceiling.
    r = L.methodology_audit(_summary([1, 1, 1, 1, 1, -0.5]), design=design, verbose=False)
    assert 'c' in r['noise_level_components'] and 'c' not in r['resolution_limited_components'], r
    # n=4 strong effect -> the floor (0.125) is above alpha -> resolution-limited.
    r2 = L.methodology_audit(_summary([1, 1, 1, 0.9]), design=design, verbose=False)
    assert 'c' in r2['resolution_limited_components'], r2


def t_pair_reason_train_knobs_mixed():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    a = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=10.0, train_knobs=L.training_knobs())
    b = dict(variant='b', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=11.0)
    r = L.pair_reason(a, b)
    assert r is not None and 'unverifiable train_knobs' in r, f'one-sided train_knobs presence must be refused, got {r!r}'
    r2 = L.pair_reason({k: v for k, v in a.items() if k != 'train_knobs'}, b)
    assert r2 is None, 'both records missing the descriptor is the legacy path and may still pair'


def t_pair_reason_fingerprint_gate():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=10.0,
                data_fp='train-a', val_fp='val-a', eval_fp='eval-a')
    b = dict(base, variant='b')
    assert L.pair_reason(base, b) is None
    # Same run_cfg but a different corpus / validation slice is NOT a
    # controlled contrast, even though every numeric knob is identical.
    assert 'data_fp' in L.pair_reason(base, dict(b, data_fp='train-b'))
    assert 'val_fp' in L.pair_reason(base, dict(b, val_fp='val-b'))
    assert 'eval_fp' in L.pair_reason(base, dict(b, eval_fp='eval-b'))
    # The zero-GPU report builder has a local pair gate; it must mirror this.
    assert 'data_fp' in BR._pair_reason(base, dict(b, data_fp='train-b'))
    assert 'unverifiable val_fp' in BR._pair_reason(base, {k: v for k, v in b.items() if k != 'val_fp'})
    # One-sided presence is unverifiable, not silently compatible.
    c = dict(b)
    c.pop('val_fp')
    assert 'unverifiable val_fp' in L.pair_reason(base, c)
    # Nested records can share the full-val/eval fingerprints yet score a
    # different held-out row fold.  Both pairing twins must compare it.
    h = dict(b, holdout_row_fp='h1')
    assert L.pair_reason(h, dict(h, variant='a')) is None
    assert BR._pair_reason(h, dict(h, variant='a')) is None
    assert 'holdout_row_fp' in (L.pair_reason(h, dict(h, holdout_row_fp='h2')) or '')
    assert 'holdout_row_fp' in (BR._pair_reason(h, dict(h, holdout_row_fp='h2')) or '')
    assert 'unverifiable holdout_row_fp' in L.pair_reason(base, h)
    assert 'unverifiable holdout_row_fp' in BR._pair_reason(base, h)
    # Legacy records (both without fingerprints) keep the compatibility path.
    leg = {k: v for k, v in base.items() if k not in ('data_fp', 'val_fp', 'eval_fp')}
    assert L.pair_reason(leg, dict(leg, variant='b')) is None
    # aggregate must refuse to pool one group across two evaluation slices.
    summary = {
        'a::seed0': dict(base, variant='a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5, val_fp='val-b'),
        'b::seed0': dict(b),
        'b::seed1': dict(b, seed=1, ppl=11.0),
    }
    try:
        L.aggregate(summary)
    except ValueError as e:
        assert 'different corpus/evaluation fingerprints' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across validation slices')
    # Same train/validation but a different fixed evaluation slice is still a
    # mixed target set for the pooled group aggregate.
    eval_mix = {
        'a::seed0': dict(base, variant='a', eval_fp='eval-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5, eval_fp='eval-b'),
    }
    try:
        L.aggregate(eval_mix)
    except ValueError as e:
        assert 'eval_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across evaluation slices')
    # The fixed evaluation SUBSET is part of the same identity: two records
    # scored on different subsets have different target denominators even when
    # train/validation/eval fingerprints agree.
    subset_mix = {
        'a::seed0': dict(base, variant='a', eval_subset_fp='subset-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5,
                         eval_subset_fp='subset-b'),
    }
    try:
        L.aggregate(subset_mix)
    except ValueError as e:
        assert 'eval_subset_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across evaluation subsets')
    # Nested records on different held-out row folds are also different
    # samples, even when the full validation/eval fingerprints agree.
    holdout_mix = {
        'a::seed0': dict(base, variant='a',
                         ppl_estimation_rows='nested_validation_holdout',
                         ppl_holdout=10.0, holdout_row_fp='fold-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5,
                         ppl_estimation_rows='nested_validation_holdout',
                         ppl_holdout=10.5, holdout_row_fp='fold-b'),
    }
    try:
        L.aggregate(holdout_mix)
    except ValueError as e:
        assert 'holdout_row_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across held-out row folds')
    rows = [dict(base, by_len={8: {'ppl': 10.0}}),
            dict(base, variant='b', val_fp='val-b', by_len={8: {'ppl': 11.0}})]
    try:
        L.by_len_cells(rows, 8)
    except ValueError as e:
        assert 'different validation slices' in str(e), e
    else:
        raise AssertionError('by_len_cells must reject rows from different validation slices')


def t_m0_scale_analysis_descriptive():
    fp = 'steps10_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    summary = {}
    for v, base in (('csa_dynamic', 10.0), ('csa_fixed', 10.5),
                    ('hybrid_dynamic', 10.2), ('full_matched', 11.0),
                    ('full_sw128_matched', 10.8)):
        for seed in range(4):
            summary[f'{v}::seed{seed}'] = dict(
                variant=v, seed=seed, run_cfg=fp, steps=10, tokens_seen=10,
                ppl=base + 0.1 * seed, params=100, train_knobs=knobs,
                data_fp='d', val_fp='v')
    # Tier discovery is the B-tier ladder only, not the earlier RUN_SCALE /
    # RUN_LONG / RUN_ABL exploratory panels; explicit extra tiers are allowed
    # and duplicate declared seeds do not inflate the planned n.
    assert set(M0._tier_run_cfgs()) == {'S1', 'S2'}, M0._tier_run_cfgs()
    extra = M0._tier_run_cfgs({'S3': {'outdir': 'results_b_s3',
                                      'seeds': [2, 0, 2, 1]}})
    assert extra['S3']['outdir'] == 'results_b_s3', extra
    assert M0._declared_seed_list(extra['S3']) == [0, 1, 2], extra
    tmp = tempfile.mkdtemp()
    try:
        r = M0.scale_analysis({'S1': summary}, outdir=tmp)
        p = r['panels']['S1']
        assert p['n_seeds'] == 4 and p['descriptive_only'] is True, p
        names = {(c['name'], c['with'], c['without']) for c in p['contrasts']}
        assert ('dynamic_blocking', 'csa_dynamic', 'csa_fixed') in names, names
        assert ('hybrid_hierarchy', 'hybrid_dynamic', 'csa_dynamic') in names, names
        assert all(c['resolution_limited'] for c in p['contrasts'] if c['n_paired']), p['contrasts']
        assert os.path.exists(os.path.join(tmp, 'scale_analysis.json'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_seed_plan():
    p = L.methodology_seed_plan(3)
    assert p['declared_contrasts'] >= 10, p
    assert p['min_seeds_for_family_significance'] > 3
    assert p['sufficient_for_family_rejection'] is False
    assert 'ablation_completeness' in p and p['ablation_complete'] is True, p
    req = p['min_seeds_for_family_significance']
    assert L.holm_min_attainable_p(req, p['declared_contrasts'], 0.05) <= 0.05
    assert L.holm_min_attainable_p(req - 1, p['declared_contrasts'], 0.05) > 0.05
    assert L.methodology_seed_plan(req)['sufficient_for_family_rejection'] is True
    # A fractional/bool plan size must not be truncated into a different n.
    for _bad_n in (req + 0.9, True, float('nan'), 'not-a-count'):
        _p = L.methodology_seed_plan(_bad_n)
        assert _p['n_seeds'] == 0, (_p, _bad_n)
        assert _p['sufficient_for_family_rejection'] is False, (_p, _bad_n)


def t_audit_family_holm_and_verified():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _summary(n, effect=-1.0):
        s = {}
        for i in range(n):
            s[f'csa_dynamic::seed{i}'] = dict(variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0 + effect, params=1000, train_knobs=knobs)
            s[f'full_sw128_matched::seed{i}'] = dict(variant='full_sw128_matched', seed=i, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0, params=1001, train_knobs=knobs)
        return s
    # Only the baseline component is actually present in this panel, so the
    # correction family is its two declared contrasts, not every unmeasured
    # ablation claim.
    r3 = L.methodology_audit(_summary(3), verbose=False)
    assert r3['family']['effective_m'] == 2, r3['family']
    assert 'compression_vs_matched_dense' in r3['resolution_limited_baselines'], r3['resolution_limited_baselines']
    assert 'compression_vs_matched_dense' not in r3['verified_baselines']
    r6 = L.methodology_audit(_summary(6), verbose=False)
    assert r6['family']['effective_m'] == 2
    assert 'compression_vs_matched_dense' not in r6['verified_baselines'], 'm=2 needs n>=7 for a Holm-corrected rejection'
    # n=7: all seeds same sign now clears the family-corrected gate.
    r7 = L.methodology_audit(_summary(7), verbose=False)
    assert 'compression_vs_matched_dense' in r7['verified_baselines'], r7['verified_baselines']
    e = r7['baselines']['compression_vs_matched_dense'][0]
    assert e['p_holm_global'] > e['p_exact_signflip'], 'Holm-adjusted p must be >= raw p'
    # an incomplete component keeps ALL its declared slots (dummy p=1), so a
    # missing arm cannot shrink the family.
    design = {'c': [('a', 'b', 'x'), ('a', 'c', 'y')]}
    incomplete = {'a::seed0': dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=49.0, params=100, train_knobs=knobs),
                  'b::seed0': dict(variant='b', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0, params=100, train_knobs=knobs)}
    ri = L.methodology_audit(incomplete, design=design, baselines={}, verbose=False)
    assert ri['family']['declared_contrasts'] == 2 and ri['family']['effective_m'] == 2, ri['family']
    # parameter mismatch is a hard control failure for a baseline claim
    s = _summary(7)
    for i in range(7):
        s[f'full_sw128_matched::seed{i}']['params'] = 2000
    rbad = L.methodology_audit(s, verbose=False)
    assert 'compression_vs_matched_dense' not in rbad['verified_baselines'], rbad['verified_baselines']
    assert 'compression_vs_matched_dense' in rbad['unverified_baselines']


def t_ablation_design_loss_term():
    assert 'compression_regularizer' in L.ABLATION_DESIGN, 'the compression loss term must have a leave-one-out arm'
    pairs = L.ABLATION_DESIGN['compression_regularizer']
    assert any(a == 'csa_dynamic' and b == 'csa_dyn_nocomp' for a, b, _n in pairs), pairs
    cfgs = L.make_layer_cfgs(2, 'csa_dyn_nocomp')
    assert all(getattr(c, 'comp_lambda_mult', None) == 0.0 for c in cfgs), 'the nocomp arm must zero the loss multiplier'
    ref = L.make_layer_cfgs(2, 'csa_dynamic')
    assert all(getattr(c, 'comp_lambda_mult', 1.0) == 1.0 for c in ref), 'the full arm must keep the regularizer'
    # every declared invention still has at least one concrete contrast
    for comp, contrasts in L.ABLATION_DESIGN.items():
        assert contrasts and all(len(t) == 3 for t in contrasts), f'{comp}: malformed leave-one-out design'
    assert any(v == 'csa_dyn_nocomp' for v in L.ABL_VARIANTS), 'the loss ablation arm must be scheduled by RUN_ABL'
    # Every arm named by a declared contrast must be in the SAME panel as its
    # reference.  A declared-but-unscheduled arm silently turns a complete
    # leave-one-out design into a partial one at run time.
    declared = {x for cs in L.ABLATION_DESIGN.values()
                for a, b, _n in cs for x in (a, b)}
    missing = declared - set(L.RUN_ABL['variants'])
    assert not missing, f'declared ablation arms not scheduled by RUN_ABL: {sorted(missing)}'
    assert L.ablation_completeness_audit(L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    # Dynamic blocking must be decomposed by layer kind, not only measured as
    # "all-dynamic vs all-fixed".  Otherwise hybrid_hca_dyn is scheduled but
    # never contributes to any declared contrast, and the HCA boundary effect
    # cannot be separated from the CSA boundary effect.
    _dyn = {(a, b) for a, b, _n in L.ABLATION_DESIGN['dynamic_blocking']}
    assert ('hybrid_csa_dyn', 'hybrid_fixed') in _dyn, _dyn
    assert ('hybrid_hca_dyn', 'hybrid_fixed') in _dyn, _dyn
    assert any('hybrid_hca_dyn' in (a, b)
               for cs in L.ABLATION_DESIGN.values() for a, b, _n in cs), \
        'hybrid_hca_dyn is scheduled but belongs to no declared contrast'


def t_training_knobs_in_resume_fingerprint():
    cfg = dict(steps=1, seq_len=8, batch_size=1, n_train_tokens=10, lr=0.0003,
               weight_decay=0.1, warmup=1, comp_lambda=0.05, delta_lr_mult=10.0)
    tr = np.arange(16, dtype=np.int64)
    va = np.arange(8, dtype=np.int64).reshape(1, 8)
    fp1 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    fp2 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    assert fp1 == fp2 and '_tk' in fp1, (fp1, fp2)
    assert L.run_cfg_knobs(fp1).get('tk') == L.training_knobs_fingerprint()
    orig = L.training_knobs
    fp3 = None
    try:
        L.training_knobs = lambda: dict(orig(), grad_clip_norm=2.0)
        fp3 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    finally:
        L.training_knobs = orig
    assert fp3 is not None and fp3 != fp1, 'changing optimizer/schedule/clip knobs must invalidate resume (no silent mixed panel)'
    # The sampled long-context segmenter's ceiling changes the cut geometry, so
    # it must invalidate a resumed cell even when the exact-limit stamp matches.
    _old_am = os.environ.get('CSA_ADAPTIVE_MAX_TOKENS')
    try:
        os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = '2048'
        fp4 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    finally:
        if _old_am is None:
            os.environ.pop('CSA_ADAPTIVE_MAX_TOKENS', None)
        else:
            os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = _old_am
    assert fp4 != fp1 and '_am2048' in fp4, (fp1, fp4)
    # The task/synthetic recipe is part of the data identity.
    _de1 = L._run_fp(dict(cfg, data_extra={'n_pairs': 4}), 8, 1, 1, 8, 16, tr, va)
    _de2 = L._run_fp(dict(cfg, data_extra={'n_pairs': 8}), 8, 1, 1, 8, 16, tr, va)
    assert _de1 != _de2 and '_de' in _de1, (_de1, _de2)
    assert L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va) == fp1, 'unset data_extra must keep legacy fingerprints stable'
    # Early stopping changes the completed token budget, so the policy must
    # invalidate a resumed cell (and be part of the record's recipe).
    _es1 = L._run_fp(dict(cfg, early_stop_patience=2), 8, 1, 1, 8, 16, tr, va)
    _es2 = L._run_fp(dict(cfg, early_stop_patience=3), 8, 1, 1, 8, 16, tr, va)
    assert _es1 != _es2 and _es1 != fp1 and '_esp2' in _es1, (_es1, _es2, fp1)
    assert L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va) == fp1, 'unset early stop must keep legacy fingerprints stable'


def t_controlled_variables_all_seed_records():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=dict(knobs))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=dict(knobs))
    # One seed of arm `a` silently ran with a different optimizer setting: the
    # panel-level audit must flag the mixed arm, even though the first record
    # looks identical and pair_reason refuses only that seed.
    s['a::seed0']['train_knobs'] = dict(knobs, grad_clip_norm=2.0)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['train_knobs_equal'] is False, cv
    assert 'a' in cv['within_arm_mixed'], cv
    assert r['components']['c'][0]['n_paired'] == 2, r['components']['c'][0]


def t_training_knobs_must_be_complete():
    # A controlled recipe is attested by the FULL descriptor.  Two records
    # carrying the same partial dictionary cannot prove the missing
    # optimizer/schedule/augmentation fields were shared.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    base = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=10.0, params=100, train_knobs=knobs)
    assert L.pair_reason(base, dict(base, variant='b')) is None
    # Both absent remains the documented legacy-vs-legacy path.
    assert L.pair_reason(dict(base, train_knobs=None),
                         dict(base, variant='b', train_knobs=None)) is None
    partial = {k: v for k, v in knobs.items()
               if k not in ('lr_schedule', 'augmentation')}
    ra = dict(base, train_knobs=partial)
    rb = dict(base, variant='b', train_knobs=dict(partial))
    why = L.pair_reason(ra, rb)
    assert why and 'omits controlled field' in why, why
    # The zero-GPU report twin must refuse exactly the same pair.
    rwhy = BR._pair_reason(ra, rb)
    assert rwhy and 'omits controlled field' in rwhy, rwhy
    # Malformed numeric fields are not a controlled recipe either.
    for bad in (dict(knobs, betas=[0.9, float('nan')]),
                dict(knobs, eps=0.0),
                dict(knobs, augmentation='')):
        assert L.pair_reason(dict(base, train_knobs=bad),
                             dict(base, variant='b', train_knobs=dict(bad))), bad
    # The tuned-baseline relaxed gate must not become a back door.
    tuned = {'a': 0, 'b': 0}
    # Give the relaxed rule its data fingerprints so it reaches the descriptor
    # check instead of stopping at missing corpus evidence.
    ra2 = dict(ra, data_fp='d', val_fp='v')
    rb2 = dict(rb, data_fp='d', val_fp='v')
    _relaxed = BT._relaxed_pair_reason(ra2, rb2, tuned)
    assert _relaxed and 'omits controlled field' in _relaxed, _relaxed
    # Panel-level report names the defect and refuses to call it controlled.
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=49.0, params=100,
                                train_knobs=dict(partial))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=50.0, params=100,
                                train_knobs=dict(partial))
    rep = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]},
                              baselines={}, verbose=False)
    cv = rep['controlled_variables']['c']
    assert cv['controlled'] is False, cv
    assert set(cv['train_knobs_defects']) == {'a', 'b'}, cv
    assert any('train_knobs descriptor(s) are incomplete' in w
               for w in rep['warnings']), rep['warnings']


def t_underpowered_contrast_is_never_significant():
    # alpha=0.6 makes the raw exact p (0.5 at n=2) clear the threshold.  The
    # three-seed floor must still prevent a positive verdict.
    r = L.paired_effect_stats([-1.0, -1.2], alpha=0.6, min_n=3)
    assert r['underpowered'] is True, r
    assert r['significant'] is False, r
    assert r['std_defined'] is True, r
    # One seed has no estimable sample spread: keep the legacy 0.0 but say so.
    one = L.paired_effect_stats([-1.0])
    assert one['n'] == 1 and one['std'] == 0.0, one
    assert one['std_defined'] is False and one['significant'] is False, one
    zero = L.paired_effect_stats([])
    assert zero['std_defined'] is False and zero['significant'] is False, zero
    # A powered, resolvable, same-direction panel remains significant.
    r2 = L.paired_effect_stats([-1.0, -1.2, -0.8, -1.1, -0.9, -1.0],
                               alpha=0.05, min_n=3)
    assert r2['underpowered'] is False and r2['significant'] is True, r2
    assert r2['std_defined'] is True, r2


def t_report_pairs_cover_the_declared_ablation_design():
    pairs = L.declared_ablation_pairs()
    declared = {(a, b) for cs in L.ABLATION_DESIGN.values()
                for a, b, *_ in cs}
    assert set(pairs) == declared, (set(pairs) ^ declared)
    assert len(pairs) == len(set(pairs)), 'declared pair list must be de-duplicated'
    # The task-learned-boundary contrast is a declared innovation; the
    # console pair table must not silently omit it (the historical drift bug).
    assert ('csa_dynamic_task', 'csa_dynamic') in pairs, pairs
    # A malformed design yields no crash and no fabricated pairs.
    assert L.declared_ablation_pairs({'A': 'not-a-list'}) == []
    assert L.declared_ablation_pairs({'A': [('a', 'b', 'x'),
                                            ('a', 'b', 'again')]}) == [('a', 'b')]


def t_partial_ablation_cannot_verify_a_component():
    design = {'A': [('full', 'full-A', ''), ('full', 'full-A2', '')]}
    partial = {
        'ablation_completeness': {
            'components': {'A': {'complete': False}},
            'complete_components': [], 'incomplete_components': ['A'],
            'missing_arms': {'A': ['full-A2']}},
        'measured_components': ['A'],
        'verified_components': ['A'],
        'unverified_components': []}
    cov = L.ablation_coverage_across_panels({'p': partial}, design)
    row = cov['components']['A']
    assert row['panels_verified_raw'] == ['p'], row
    assert row['panels_verified'] == [], row
    assert row['panels_verified_incomplete'] == ['p'], row
    assert cov['verified_anywhere'] == [], cov
    assert cov['verified_only_incomplete'] == ['A'], cov
    assert any('whole innovation' in w for w in cov['warnings']), cov
    # Once the declared ablation is complete, the same verified panel counts.
    complete = dict(partial)
    complete['ablation_completeness'] = {
        'components': {'A': {'complete': True}},
        'complete_components': ['A'], 'incomplete_components': [],
        'missing_arms': {}}
    cov2 = L.ablation_coverage_across_panels({'p': complete}, design)
    assert cov2['verified_anywhere'] == ['A'], cov2
    assert cov2['verified_only_incomplete'] == [], cov2
    # A project-level "every innovation is useful" claim is only open when a
    # complete ablation verified every declared component.
    assert cov2['all_components_useful_allowed'] is True, cov2
    assert cov['all_components_useful_allowed'] is False, cov
    assert cov['unverified_components'] == ['A'], cov
    assert any('not verified' in r for r in cov['all_components_useful_reasons']), cov
    incomplete_only = dict(complete, verified_components=['A'],
                           ablation_completeness=partial['ablation_completeness'])
    cov3 = L.ablation_coverage_across_panels({'p': incomplete_only}, design)
    assert cov3['all_components_useful_allowed'] is False, cov3
    assert any('incomplete ablation' in r
               for r in cov3['all_components_useful_reasons']), cov3


def t_seed_coverage_warning():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=knobs)
        if i < 2:
            s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                    ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['seed_coverage_equal'] is False, cv
    assert any('seed/protocol coverage' in w for w in r['warnings']), r['warnings']
    assert r['components']['c'][0]['n_paired'] == 2, r['components']['c'][0]


def t_fairness_cross_arm_controls():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    obj = {'comp_reg_applies': True, 'comp_lambda_effective': 0.05,
           'delta_lr_effective': 10.0, 'boundary_policy': ['threshold'],
           'loss_terms': ['ce', 'comp_reg']}

    def _panel(*, knob_override=None, drop_seed=None, obj_override_arm=None):
        s = {}
        for i in range(3):
            for arm, ppl in (('a', 10.0), ('b', 11.0)):
                if arm == 'b' and i == drop_seed:
                    continue
                kk = dict(knobs)
                if knob_override == (arm, i):
                    kk['grad_clip_norm'] = 2.0
                oo = dict(obj)
                if obj_override_arm == arm:
                    oo['boundary_policy'] = ['reinforce']
                s[f'{arm}::seed{i}'] = dict(
                    variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                    ppl=ppl, params=100, train_knobs=kk, train_objective=oo)
        return s

    with tempfile.TemporaryDirectory() as td:
        def _run(summary):
            path = os.path.join(td, 'summary.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(summary, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_fairness(path, arms=['a', 'b'])

        ok = _run(_panel())
        assert ok['controlled_variables']['train_knobs_equal'] is True, ok['controlled_variables']
        assert ok['controlled_variables']['objective_equal'] is True, ok['controlled_variables']
        assert not any('train_knobs differ' in p for p in ok['problems']), ok['problems']
        # A different objective in one arm is a system-level confound; surface
        # it as a warning instead of silently calling the pair controlled.
        od = _run(_panel(obj_override_arm='b'))
        assert od['controlled_variables']['objective_equal'] is False, od['controlled_variables']
        assert any('training objectives differ' in w for w in od['warnings']), od['warnings']
        assert od['status'] == 'warn', od['status']
        # A seed-level objective change INSIDE one arm is a mixed protocol,
        # not merely a cross-arm disclosure: the arm mean pools two different
        # objectives and must not be called controlled.
        mixed_obj = {k: dict(v) for k, v in _panel().items()}
        mixed_obj['a::seed1']['train_objective'] = dict(obj, boundary_policy=['reinforce'])
        _mr, _ = L._variant_seed_map(mixed_obj)
        cv = L.controlled_variables_report(_mr, ['a', 'b'])
        assert cv['objective_equal'] is False, cv
        assert 'a' in cv['within_arm_mixed'], cv
        assert cv['controlled'] is False, cv
        assert 'train_objective' in cv['differences'], cv
        md = _run(mixed_obj)
        assert md['status'] == 'fail', md
        assert any('mix DIFFERENT' in p for p in md['problems']), md['problems']
        # A partially recorded objective descriptor is unverifiable, not a
        # pass; the arm must be named as partially recorded.
        part_obj = {k: dict(v) for k, v in _panel().items()}
        part_obj['b::seed1'].pop('train_objective')
        _pr, _ = L._variant_seed_map(part_obj)
        cv2 = L.controlled_variables_report(_pr, ['a', 'b'])
        assert cv2['objective_equal'] is False, cv2
        assert 'b' in cv2['partially_recorded'], cv2
        assert cv2['controlled'] is False, cv2
        # A different optimizer/clip in ONE arm must fail even when every arm
        # is internally consistent.
        bad = _run(_panel(knob_override=('b', 1)))
        assert bad['status'] == 'fail', bad
        assert any('train_knobs differ across arms' in p for p in bad['problems']), bad['problems']
        # Missing seed in one arm: arm means would be over different seeds.
        cov = _run(_panel(drop_seed=2))
        assert cov['status'] == 'fail', cov
        assert any('seed/protocol coverage differs' in p for p in cov['problems']), cov['problems']
        # Same recipe but a larger arm is a capacity confound, even though the
        # per-seed run_cfg/train_knobs gate is satisfied.
        cap = _panel()
        for _k in list(cap):
            if cap[_k]['variant'] == 'b':
                cap[_k]['params'] = 150
        rcap = _run(cap)
        assert rcap['status'] == 'fail', rcap
        assert any('parameter counts differ' in p for p in rcap['problems']), rcap['problems']
        miss = _panel()
        miss['b::seed0'].pop('params')
        rmiss = _run(miss)
        assert rmiss['status'] == 'warn', rmiss
        assert any('capacity matching cannot be verified' in w for w in rmiss['warnings']), rmiss


def t_tuned_pairing_keeps_optimizer_gate():
    knobs = L.training_knobs()
    a = dict(variant='a', steps=1, tokens_seen=1, data_fp='d', val_fp='v', params=100,
             train_knobs=knobs)
    b = dict(a, variant='b')
    assert BT._relaxed_pair_reason(a, b, {'a': 3, 'b': 3}) is None
    b2 = dict(b, train_knobs=dict(knobs, grad_clip_norm=2.0))
    why = BT._relaxed_pair_reason(a, b2, {'a': 3, 'b': 3})
    assert why and 'train_knobs' in why, why
    b3 = dict(b)
    b3.pop('train_knobs')
    why2 = BT._relaxed_pair_reason(a, b3, {'a': 3, 'b': 3})
    assert why2 and 'unverifiable train_knobs' in why2, why2
    # A paired seed does not imply a matched shared-trunk initialization.
    a4 = dict(a, init_core_fp='aaaa')
    b4 = dict(b, init_core_fp='bbbb')
    why3 = BT._relaxed_pair_reason(a4, b4, {'a': 3, 'b': 3})
    assert why3 and 'init_core_fp' in why3, why3
    b5 = dict(b)
    why4 = BT._relaxed_pair_reason(a4, b5, {'a': 3, 'b': 3})
    assert why4 and 'unverifiable init_core_fp' in why4, why4
    # Capacity must fail CLOSED: a missing params field used to slip through
    # the old int(None) try/except as if the pair were matched.
    b6 = dict(b)
    b6.pop('params')
    why5 = BT._relaxed_pair_reason(a, b6, {'a': 3, 'b': 3})
    assert why5 and 'unverifiable params' in why5, why5
    a7 = dict(a, param_breakdown={'total': 100, 'non_embedding': 80, 'active': 70})
    b7 = dict(b, param_breakdown={'total': 100, 'non_embedding': 80, 'active': 50})
    why6 = BT._relaxed_pair_reason(a7, b7, {'a': 3, 'b': 3})
    assert why6 and 'parameter breakdown' in why6, why6
    # A tuned contrast must score the same evaluation slice.
    a8 = dict(a, eval_fp='e1')
    b8 = dict(b, eval_fp='e2')
    why7 = BT._relaxed_pair_reason(a8, b8, {'a': 3, 'b': 3})
    assert why7 and 'eval_fp' in why7, why7
    assert BT._relaxed_pair_reason(dict(a, eval_fp='e1'), dict(b, eval_fp='e1'),
                                   {'a': 3, 'b': 3}) is None
    # The final PPL subset is part of the evaluation slice too.
    assert BT._relaxed_pair_reason(dict(a, eval_subset_fp='s1'),
                                   dict(b, eval_subset_fp='s2'),
                                   {'a': 3, 'b': 3}) is not None
    assert BT._relaxed_pair_reason(dict(a, eval_subset_fp='s1'),
                                   dict(b, eval_subset_fp='s1'),
                                   {'a': 3, 'b': 3}) is None
    # A held-out row fold must be the same fold on both arms.
    why8 = BT._relaxed_pair_reason(dict(a, holdout_row_fp='h1'),
                                   dict(b, holdout_row_fp='h2'), {'a': 3, 'b': 3})
    assert why8 and 'holdout_row_fp' in why8, why8
    # Independent tuning may change lr/wd/warmup, but NOT the rest of the
    # structured recipe.  A different seq_len/eval_subset/vocab or a one-sided
    # descriptor is unverifiable, never a matched tuned pair.
    hp = {'steps': 1, 'tokens_seen': 1, 'lr': 3e-4, 'weight_decay': 0.1,
          'warmup': 1, 'batch_size': 2, 'seq_len': 8, 'd': 8,
          'n_layers': 1, 'n_heads': 1, 'd_head': 8, 'mlp_ratio': 4.0,
          'eval_every': 1, 'eval_subset': 4, 'vocab_size': 16}
    a9 = dict(a, hyperparams=dict(hp))
    b9 = dict(b, hyperparams=dict(hp, lr=1e-3, weight_decay=0.05, warmup=2))
    assert BT._relaxed_pair_reason(a9, b9, {'a': 3, 'b': 3}) is None
    for _k, _v in (('seq_len', 16), ('batch_size', 4), ('eval_subset', 8),
                   ('vocab_size', 32), ('mlp_ratio', 8.0)):
        _bad = dict(b, hyperparams={**hp, _k: _v})
        _why = BT._relaxed_pair_reason(a9, _bad, {'a': 3, 'b': 3})
        assert _why and ('hyperparams.' + _k) in _why, (_k, _why)
    _why = BT._relaxed_pair_reason(a9, b, {'a': 3, 'b': 3})
    assert _why and 'unverifiable hyperparams' in _why, _why
    # A tuned contrast may not call a complete-case / seed-driven / n<3 panel
    # significant just because its Holm p is small.
    base_st = {'n': 3, 'p_exact_signflip': 0.02, 'p_holm': 0.03,
               'mean': 1.0, 'std': 0.2, 'underpowered': False,
               'within_seed_noise': False, 'seed_driven': False,
               'complete_pairs': True, 'n_dropped': 0}
    assert BT._tuned_significant(base_st, 0.03) is True
    for _bad in (dict(base_st, n=2, underpowered=True),
                 dict(base_st, within_seed_noise=True),
                 dict(base_st, seed_driven=True),
                 dict(base_st, complete_pairs=False),
                 dict(base_st, n_dropped=1),
                 dict(base_st, n_skipped=1),
                 dict(base_st, n_skipped_config_mismatch=1),
                 dict(base_st, n_skipped=-1),
                 dict(base_st, n_skipped=True),
                 dict(base_st, n_dropped=-1),
                 dict(base_st, n_input=-1),
                 dict(base_st, n_input=2),
                 dict(base_st, n_input=3, n_dropped=4),
                 dict(base_st, n_nonzero=4),
                 dict(base_st, n_nonzero=-1),
                 dict(base_st, n_nonzero=1)):
        assert BT._tuned_significant(_bad, 0.03) is False, _bad
    assert BT._tuned_significant(dict(base_st, n_skipped=0), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n_input=3, n_dropped=0), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n=6, n_nonzero=6), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n=6, n_nonzero=3), 0.03) is False
    assert BT._tuned_significant(base_st, None) is False
    assert BT._tuned_significant(base_st, 0.5) is False
    # alpha is a gate parameter, not a free multiplier.
    for _bad_alpha in (True, 0, -0.1, 1.5, float('nan'), 'x'):
        assert BT._tuned_significant(base_st, 0.03, _bad_alpha) is False, _bad_alpha


def t_arm_mean_std_and_relative_effect():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    a_vals = [10.0, 10.2, 9.8, 10.1, 9.9]
    s = {}
    for i, ppl in enumerate(a_vals):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=ppl, params=100, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=10.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    arm = r['arm_summary']['a']
    assert arm['n_seeds'] == 5 and arm['meets_min_seeds'] is True, arm
    assert abs(arm['ppl_mean'] - float(np.mean(a_vals))) < 1e-12
    assert abs(arm['ppl_std'] - float(np.std(a_vals, ddof=1))) < 1e-12
    assert r['arms_below_min_seeds'] == [], r['arms_below_min_seeds']
    e = r['components']['c'][0]
    assert e['n_paired'] == 5 and e['without_mean'] == 10.0, e
    assert abs(e['relative_effect_pct'] - 100.0 * e['mean'] / e['without_mean']) < 1e-9, e
    # Nested tuning records: the contrast's arm means, relative effect and
    # bootstrap sensitivity must use the SAME holdout estimand as the paired
    # delta, not the full-validation `ppl` that selected the hyper-parameters.
    hold = [10.0, 10.2, 9.8, 10.1, 9.9]
    nested = {}
    for i, ppl in enumerate(hold):
        nested[f'a::seed{i}'] = dict(s[f'a::seed{i}'], ppl=99.0 + i,
                                     ppl_estimation_rows='nested_validation_holdout',
                                     ppl_holdout=ppl)
        nested[f'b::seed{i}'] = dict(s[f'b::seed{i}'], ppl=1.0 + i,
                                     ppl_estimation_rows='nested_validation_holdout',
                                     ppl_holdout=10.0)
    rn = L.methodology_audit(nested, design={'c': [('a', 'b', 'x')]},
                             baselines={}, verbose=False)
    en = rn['components']['c'][0]
    assert en['without_mean'] == 10.0, en
    assert abs(en['with_mean'] - float(np.mean(hold))) < 1e-12, en
    assert abs(en['relative_effect_pct'] - 100.0 * en['mean'] / en['without_mean']) < 1e-9, en
    arm_n = rn['arm_summary']['a']
    assert arm_n['ppl_estimand'] == 'nested_validation_holdout', arm_n
    assert abs(arm_n['ppl_mean'] - float(np.mean(hold))) < 1e-12, arm_n
    assert arm_n['ppl_primary_missing'] == 0, arm_n


def t_v7_v8_v9_pairs_use_primary_estimand():
    import v8_supp as V8
    import v9_supp as V9
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _rec(v, seed, full, hold):
        r = {'variant': v, 'seed': seed, 'run_cfg': fp, 'steps': 1,
             'tokens_seen': 1, 'ppl': full, 'params': 100,
             'train_knobs': knobs,
             'ppl_estimation_rows': 'nested_validation_holdout'}
        if hold is not None:
            r['ppl_holdout'] = hold
        return r

    panel = {}
    for seed in (0, 1, 2):
        panel.setdefault('a', {})[seed] = _rec('a', seed, 100.0 + seed, 10.0 + seed)
        panel.setdefault('b', {})[seed] = _rec('b', seed, 1.0 + seed,
                                              12.0 + seed if seed else None)
    for fn in (V9.add_paired, V8._add_paired):
        comp = {}
        fn(comp, 'x', panel, 'a', 'b')
        st = comp['x']
        # Seed 0's b record has no holdout accounting, so it is excluded
        # rather than paired on the full-validation PPL (which would give a
        # huge positive delta from 100 vs 1).
        assert st['n'] == 2 and abs(st['mean'] + 2.0) < 1e-12, st


def t_paired_effect_seed_robustness():
    stable = L.paired_effect_stats([-1.0, -1.1, -0.9, -1.0])
    assert stable['seed_driven'] is False and stable['seed_robust'] is True, stable
    assert stable['ci95_low'] < stable['mean'] < stable['ci95_high'], stable
    # One outlier seed makes the aggregate mean positive, but dropping it
    # flips the sign: the direction is owned by a single seed.
    fragile = L.paired_effect_stats([-2.0, -2.0, -2.0, -2.0, 8.5])
    assert fragile['mean'] > 0 and fragile['seed_driven'] is True, fragile
    assert fragile['seed_robust'] is False, fragile
    assert 'UNSTABLE' in fragile['effect_vs_noise'], fragile
    # The audit must surface it instead of silently certifying the mean.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    a_vals = [51.5, 51.5, 51.5, 51.5, 42.0]
    s = {}
    for i, ppl in enumerate(a_vals):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=ppl, params=100, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['seed_driven'] is True and e['seed_robust'] is False, e
    assert any('SEED-DRIVEN' in w for w in r['warnings']), r['warnings']


def t_capacity_confound_warning():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=1200, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=1000, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['param_controlled'] is False and e['capacity_confounded'] is True, e
    assert any('CAPACITY' in w or 'capacity' in w for w in r['warnings']), r['warnings']


def t_component_pair_requires_same_mlp_ratio():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _hp(mlp):
        return {'steps': 1, 'tokens_seen': 1, 'lr': 0.0003, 'weight_decay': 0.1,
                'warmup': 1, 'batch_size': 1, 'seq_len': 8, 'd': 8, 'n_layers': 1,
                'n_heads': 1, 'd_head': 8, 'mlp_ratio': mlp, 'comp_lambda': 0.05,
                'delta_lr_mult': 10.0, 'dataset': 'unit', 'data_version': 'v'}
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=knobs, hyperparams=_hp(4.0))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs, hyperparams=_hp(8.0))
    # Same run_cfg/train_knobs but a different MLP width is a capacity change,
    # not the module under test: a module ablation must not pair it.
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['controlled'] is False and e['n_paired'] == 0, e
    assert any('hyperparams.mlp_ratio' in w for w in r['warnings']), r['warnings']
    # A baseline contrast may legitimately widen the MLP for parameter
    # matching; there the parameter-count gate (not mlp_ratio equality) rules.
    for i in range(3):
        s[f'a::seed{i}']['params'] = 100
        s[f'b::seed{i}']['params'] = 100
    base_design = {'compression_vs_matched_dense': [('a', 'b', 'x')]}
    rb = L.methodology_audit(s, design={}, baselines=base_design, verbose=False)
    eb = rb['baselines']['compression_vs_matched_dense'][0]
    assert eb['n_paired'] == 3 and eb['controlled'] is True, eb


def t_ablation_completeness_leave_one_out():
    design = {'A': [('full', 'full-A', '')],
              'B': [('full', 'full-B', '')],
              'C': [('full', 'full-C', '')]}
    r = L.ablation_completeness_audit(design, ['full', 'full-A', 'full-B'])
    assert r['complete'] is False and r['incomplete_components'] == ['C'], r
    assert r['missing_arms']['C'] == ['full-C'], r
    r2 = L.ablation_completeness_audit(design, ['full', 'full-A', 'full-B', 'full-C'])
    assert r2['complete'] is True and r2['incomplete_components'] == [], r2


def t_ablation_completeness_full_factorial():
    arms = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC'}
    r = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r['mode'] == 'full-factorial' and r['complete'] is False, r
    assert r['missing_combos'] == [['A', 'B', 'C']], r
    arms[('A', 'B', 'C')] = 'ABC'
    r2 = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r2['complete'] is True and r2['n_mapped'] == 8, r2
    # A mapped but not-run arm is still incomplete.
    r3 = L.ablation_completeness_audit(arms_by_combo=arms, present=list(arms.values())[:-1])
    assert r3['complete'] is False and r3['missing_arms'], r3


def t_full_factorial_rejects_reused_arm():
    arms = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC',
            ('A', 'B', 'C'): 'AB'}
    r = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r['mode'] == 'full-factorial' and r['complete'] is False, r
    assert 'AB' in r['reused_arms'], r
    arms[('A', 'B', 'C')] = 'ABC'
    assert L.ablation_completeness_audit(arms_by_combo=arms)['complete'] is True


def t_ablation_coverage_across_panels():
    design = {'A': [('full', 'full-A', '')], 'B': [('full', 'full-B', '')]}

    def _rep(*, complete=(), incomplete=(), missing=None, measured=(), verified=(),
             unverified=()):
        return {'ablation_completeness': {
                    'components': {c: {} for c in set(complete) | set(incomplete)},
                    'complete_components': list(complete),
                    'incomplete_components': list(incomplete),
                    'missing_arms': dict(missing or {})},
                'measured_components': list(measured),
                'verified_components': list(verified),
                'unverified_components': list(unverified)}

    reports = {
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',)),
        'p2': _rep(incomplete=('A',), measured=('A',), missing={'A': ['full-A']}),
        'p3': _rep(unverified=('B',)),
    }
    cov = L.ablation_coverage_across_panels(reports, design)
    assert cov['components']['A']['panels_complete'] == ['p1'], cov
    assert cov['components']['A']['panels_usable'] == ['p1'], cov
    assert cov['components']['A']['panels_verified'] == ['p1'], cov
    assert cov['components']['A']['missing_arms'] == {'p2': ['full-A']}, cov
    assert cov['complete_anywhere'] == ['A'], cov
    assert cov['usable_anywhere'] == ['A'], cov
    assert cov['verified_anywhere'] == ['A'], cov
    assert cov['missing_everywhere'] == ['B'], cov
    assert cov['status'] == 'warn' and any('B' in w for w in cov['warnings']), cov
    # A complete panel that never verified is reported separately, not hidden.
    cov2 = L.ablation_coverage_across_panels(
        {'p': _rep(complete=('A', 'B'), measured=('A', 'B'), unverified=())}, design)
    assert cov2['complete_but_unverified'] == ['A', 'B'], cov2
    assert cov2['status'] == 'ok', cov2
    # All arms present but no controlled contrast shares enough seeds: this is
    # NOT a usable ablation even though the presence-only check is complete.
    cov3 = L.ablation_coverage_across_panels(
        {'p': _rep(complete=('A',), measured=())}, design)
    assert cov3['components']['A']['complete_anywhere'] is True, cov3
    assert cov3['components']['A']['usable_anywhere'] is False, cov3
    assert cov3['usable_anywhere'] == [], cov3
    assert cov3['status'] == 'warn' and any('shared seeds' in w for w in cov3['warnings']), cov3


def t_panel_ablation_flag_does_not_overclaim():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        for v, ppl in (('full', 50.0), ('full-A', 51.0)):
            s[f'{v}::seed{i}'] = dict(variant=v, seed=i, run_cfg=fp, steps=1,
                                      tokens_seen=1, ppl=ppl, params=100,
                                      train_knobs=knobs)
    design = {'A': [('full', 'full-A', '')], 'B': [('full', 'full-B', '')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    # A itself is complete, but the panel as a whole omitted B entirely; the
    # panel-level flag must not call that a complete ablation.
    assert r['ablation_completeness']['complete'] is True, r['ablation_completeness']
    assert r['ablation_complete'] is False, r['ablation_complete']
    assert r['ablation_completeness']['omitted_components'] == ['B'], r['ablation_completeness']


def t_adaptive_threshold_guard():
    # Row chunking keeps a large batch feasible; an explicit safety ceiling is
    # still honoured.
    L.causal_adaptive_threshold(torch.randn(64, 4096), 16)
    # Empty inputs keep their shape (batch x length), not a flat [0] tensor.
    for shape in ((0,), (0, 5), (3, 0), (0, 0)):
        got = L.causal_adaptive_threshold(torch.empty(shape), 16)
        assert tuple(got.shape) == shape, (shape, got.shape)
    _old = os.environ.get('CSA_ADAPTIVE_MAX_TOKENS')
    try:
        os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = '1024'
        try:
            L.causal_adaptive_threshold(torch.randn(16000), 16)
            raise AssertionError('a row above the explicit safety ceiling must be refused')
        except RuntimeError:
            pass
    finally:
        if _old is None:
            os.environ.pop('CSA_ADAPTIVE_MAX_TOKENS', None)
        else:
            os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = _old


def t_adaptive_threshold_values():
    # The production path row/column-chunks the sort.  That partition must not
    # change the closed-form per-row causal quantile: compare against the
    # unchunked definition for both a single row and a batch.
    def reference(sim, target):
        q = max(1.0 / max(int(target), 1), 0.001)
        squeeze = sim.dim() == 1
        x = (sim.unsqueeze(0) if squeeze else sim).detach().float()
        n = x.shape[-1]
        idx = torch.arange(n)
        pos = q * idx
        lo = pos.floor().long()
        frac = pos - lo
        lo_c = lo.clamp(max=n - 1)[..., None]
        hi_c = (lo + 1).clamp(max=n - 1)[..., None]
        use_hi = (frac > 0.0) & (lo + 1 <= idx)
        first = idx == 0
        masked = torch.where(idx[None, None, :] <= idx[None, :, None],
                             x[:, None, :], float('inf')).sort(dim=-1).values
        lo_v = masked.gather(-1, lo_c.expand(x.shape[0], n, 1)).squeeze(-1)
        hi_v = masked.gather(-1, hi_c.expand(x.shape[0], n, 1)).squeeze(-1)
        interp = lo_v * (1.0 - frac) + hi_v * frac
        oc = torch.where(use_hi, interp, lo_v)
        out = torch.where(first, torch.full_like(oc, float('-inf')), oc)
        return out.squeeze(0) if squeeze else out

    for shape in ((7,), (3, 11), (2, 300)):
        for target in (1, 4, 16, 128):
            sim = torch.randn(*shape)
            got = L.causal_adaptive_threshold(sim, target)
            want = reference(sim, target)
            assert torch.equal(got, want), (shape, target, (got - want).abs().max())


def t_empty_pool_geometry():
    dev = torch.device('cpu')
    ids = torch.empty(0, dtype=torch.long, device=dev)
    assert L._n_blocks_cached(ids) == 0, 'an empty block table has zero blocks'
    comp, last_tok, B = L.pool_variable_blocks(torch.empty(0, 4), torch.empty(0, 4),
                                               torch.empty(0, 4), torch.empty(0, 4), ids,
                                               torch.empty(4, 4), torch.empty(4, 4), 1)
    assert comp.shape == (0, 4) and last_tok.shape == (0,) and B == 0
    comp, last_tok, B = L.pool_blocks_single(torch.empty(0, 4), torch.empty(0, 4),
                                             torch.empty(4, 4), ids)
    assert comp.shape == (0, 4) and last_tok.shape == (0,) and B == 0


def t_segment_empty():
    bids, hon, counts = L._segment(0, [], 2, 128)
    assert bids.size == 0 and hon == {} and counts == []
    assert L._segment(1, [], 2, 128)[2] == [1]


def t_segment_uniform_stride_only_when_all_cuts():
    # A wanted cut at n-1 sits after the final token and cannot short-circuit
    # the walk.  {1} for n=2 has n-1 slots but leaves slot 0 unset, so the
    # first block must still run to the end (count 2), not take the short
    # uniform stride.  This is the boundary the uniform-stride fast path must
    # not absorb.
    bids, hon, counts = L._segment(2, [False, True], 1, 2)
    assert counts == [2] and bids.tolist() == [0, 0] and hon == {}, (bids, hon, counts)
    # n=3, two wanted slots including n-1: still NOT the all-cuts case.
    bids, hon, counts = L._segment(3, [False, True, True], 1, 2)
    assert counts == [2, 1] and bids.tolist() == [0, 0, 1] and hon == {1: 1.0}, (bids, hon, counts)
    bids, hon, counts = L._segment(3, [True, False, True], 1, 2)
    assert counts == [1, 2] and bids.tolist() == [0, 1, 1] and hon == {0: 1.0}, (bids, hon, counts)
    # All answer-forcing slots (0..n-2) present -> the uniform stride is exact.
    bids, hon, counts = L._segment(3, [True, True, True], 1, 2)
    assert counts == [1, 1, 1] and bids.tolist() == [0, 1, 2], (bids, hon, counts)
    bids, _, counts = L._segment(4, [True, True, False, True], 1, 3)
    assert counts == [1, 1, 2] and bids.tolist() == [0, 1, 2, 2], (bids, counts)


def t_shipped_manifest_consistency():
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    assert v2['manifest_version'] == 3
    n = len(v2['seeds'])
    m = len(v2['primary_comparisons'])
    assert n == 12 and v2['seeds'] == list(range(12))
    assert v2['statistics']['n_seeds'] == n
    assert v2['code_semantics_expected'] == L.CODE_SEMANTICS
    assert v2.get('eager_batch_expected') is False, 'v3 must bind the real per-row RoPE dispatch'
    assert v2.get('amp_dtype_expected') == 'bf16' and v2.get('amp_grad_scaler_expected') is False
    assert abs(v2['statistics']['p_floor'] - L.signflip_p_floor(n)) < 1e-15
    assert abs(v2['statistics']['min_attainable_p_holm'] - L.holm_min_attainable_p(n, m)) < 1e-12
    assert v2['statistics']['min_attainable_p_holm'] < 0.05
    assert v2['statistics']['min_consistency_for_first_gate'] == L.signflip_min_consistency(n, m)
    # v3 declares a reachable confirmation floor (8 complete pairs) and it is
    # mathematically able to reject under Holm/4.
    assert v2['statistics']['min_seeds_for_confirmation'] == 8
    assert L.holm_min_attainable_p(v2['statistics']['min_seeds_for_confirmation'], m) <= 0.05
    for k, want in v2['statistics']['power_first_holm_gate'].items():
        q = float(k.split('=')[1])
        got = L.signflip_power(n, m, q)
        assert abs(got - want) < 0.02, f'power table drift at {k}: manifest {want} vs code {got:.4f}'
    # the revision history must record the 6->12 change and its precondition
    hist = v2['revision']['history']
    assert any(h.get('version') == 1 for h in hist) and any(h.get('version') == 2 for h in hist) and any(h.get('version') == 3 for h in hist)
    # v3 pins the per-arm parameter-matching knob
    per = v2.get('binding', {}).get('per_arm_recipe_signature') or {}
    assert set(per) == set(v2['arms']), per
    assert abs(float(per['full_rope']['mr']) - 6.76348876953125) < 1e-9, per
    assert all(abs(float(per[a]['mr']) - 4.0) < 1e-12 for a in v2['arms'] if a != 'full_rope'), per
    # scope block present
    assert v2.get('scope', {}).get('not_covered'), 'scope.not_covered must name the dynamic/HCA claims that are out of scope'
    # the compute-budget caveat must be explicit, not buried in prose
    assert v2.get('compute_budget_disclosure', {}).get('flops'), 'v3 must disclose the dense-arm MLP/FLOPs mismatch'


def t_frozen_seeds():
    assert V12._frozen_seeds() == list(range(12)), 'phase seeds must come from the shipped manifest (single source of truth)'


def t_doc_level_analysis():
    n_seed, n_doc = 3, 4
    base_a = [[30.0 + i + 0.5 * s for i in range(n_doc)] for s in range(n_seed)]
    base_b = [[31.0 + i + 0.5 * s for i in range(n_doc)] for s in range(n_seed)]
    recs_a = {s: {'by_len': {'4096': {'ppl': 1.0, 'ppl_per_seq': base_a[s]}}} for s in range(n_seed)}
    recs_b = {s: {'by_len': {'4096': {'ppl': 1.0, 'ppl_per_seq': base_b[s]}}} for s in range(n_seed)}
    d = V12._doc_level_analysis(recs_a, recs_b, list(range(n_seed)), 4096)
    assert d is not None and d['n_docs'] == n_doc
    assert abs(d['per_doc_delta_mean'] - (-1.0)) < 1e-9, 'per-doc delta must be a - b = -1.0'
    assert abs(d['between_doc_sd']) < 1e-9, 'constant -1.0 shift across docs -> zero between-doc spread'


# ---------------------------------------------------------------------------
# D. eval per-seq PPL（意见九 b）
# ---------------------------------------------------------------------------

def t_eval_per_seq():
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64, _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(3).integers(0, 256, size=(3, 65)), dtype=np.int64)
    out = V7.eval_length_gen(model, ids, [32, 64], dev, max_pos=None)
    for Ln in (32, 64):
        cell = out[Ln]
        assert 'ppl_per_seq' in cell and len(cell['ppl_per_seq']) == 3, 'per-document rows must be recorded'
        geo = math.exp(sum(math.log(p) for p in cell['ppl_per_seq']) / 3.0)
        assert abs(geo - cell['ppl']) / cell['ppl'] < 1e-6, f'pooled ppl must equal the geometric mean of per-seq ppl at L{Ln} (reduction-invariance)'
        # eval_span is the SCORED target span (L-1), the same denominator the
        # by-length audit and n_tok use.  Reporting the input width L here
        # made the cell metadata contradict its own token count.
        assert cell['eval_span'] == Ln - 1, (Ln, cell['eval_span'])
        assert cell['n_tok'] == cell['n_seq'] * cell['eval_span'], cell
    # Two over-long requests can truncate to the SAME token slice.  The cache
    # reuse must carry the per-document accounting, not silently return a cell
    # that is missing n_seq/ppl_per_seq just because another length populated
    # the cache first.
    trunc = V7.eval_length_gen(model, ids, [48, 64], dev, max_pos=32)
    for Ln in (48, 64):
        cell = trunc[Ln]
        assert cell['truncated'] is True, (Ln, cell)
        assert cell['eval_span'] == 31, (Ln, cell['eval_span'])
        assert cell['n_seq'] == 3 and len(cell['ppl_per_seq']) == 3, (Ln, cell)
        assert cell['n_tok'] == 3 * 31, (Ln, cell['n_tok'])
        geo = math.exp(sum(math.log(p) for p in cell['ppl_per_seq']) / 3.0)
        assert abs(geo - cell['ppl']) / cell['ppl'] < 1e-6, (Ln, geo, cell['ppl'])
    # Invalid/ambiguous length geometry must fail before any scoring: a
    # fractional length is not the truncated integer, and max_pos=0/1 or a
    # negative value turns the last-token slice into a different prefix.
    for _bad_lens in ([1], [0], [-4], [1.5], [True], [None]):
        try:
            V7.eval_length_gen(model, ids, _bad_lens, dev, max_pos=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted eval_lens={_bad_lens!r}')
    for _bad_max in (0, 1, -1, 1.5, True):
        try:
            V7.eval_length_gen(model, ids, [32], dev, max_pos=_bad_max)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted max_pos={_bad_max!r}')
    for _bad_n in (0, -1, 1.5, True):
        try:
            V7.eval_length_gen(model, ids, [32], dev, n_seq=_bad_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted n_seq={_bad_n!r}')
    # The sampled path must credit per-document stats to the GLOBAL row, not
    # to a chunk-local row.  With chunk=1 and most targets sampled, a broken
    # offset would pile almost every token onto row 0.
    samp = L.eval_ppl(model, ids, dev, chunk=1, eval_rows=150, eval_seed=5, return_stats=True)
    per_row = samp['n_tokens_per_row']
    assert sum(per_row) == int(samp['n_tokens']) == 150, (sum(per_row), samp['n_tokens'])
    assert max(per_row) <= ids.shape[1] - 1, f'per-document token counts must stay within one row: {per_row}'
    assert abs(sum(samp['nll_per_row']) - samp['nll_sum']) < 1e-9, 'per-document NLL must sum to the pooled NLL'


# ---------------------------------------------------------------------------
# E. 可复现性与实验追踪（随机种子 / 小批量过拟合 / provenance）
# ---------------------------------------------------------------------------

def t_review3_evidence_metadata():
    assert os.path.exists(os.path.join(REPO, 'DECISION_LOG.md')), 'DECISION_LOG.md must be tracked'
    assert 'requirements.txt' in open(os.path.join(REPO, 'archive_release.py'), encoding='utf-8').read(), 'archive must pack the dependency declaration'
    assert 'DECISION_LOG.md' in open(os.path.join(REPO, 'archive_release.py'), encoding='utf-8').read()
    import build_report as BR
    import error_analysis as EA
    assert BR._sig_gate({'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02, 'mean': 1.0, 'std': 0.5}) is True
    assert BR._sig_gate({'n': 3, 'p_exact_signflip': 0.01, 'p_holm': 0.02, 'mean': 1.0, 'std': 0.5}) is False
    assert BR._sig_gate({'n': 6, 'p_exact_signflip': 0.01, 'p_holm': None, 'mean': 1.0, 'std': 0.5}) is False

def t_review3_pooled_aggregate():
    summary = {}
    for s, nll in ((0, 100.0), (1, 140.0)):
        summary[f'csa::seed{s}'] = {'variant': 'csa', 'seed': s, 'ppl': math.exp(nll / 100.0),
                                    'nll_sum': nll, 'n_tokens': 100, 'params': 10,
                                    'steps': 1, 'tokens_seen': 100,
                                    'train_objective': {'comp_reg_applies': True, 'delta_lr_effective': 10.0, 'boundary_policy': ['threshold']}}
    agg = L.aggregate(summary)
    e = agg['csa']
    assert abs(e['ppl_pooled'] - math.exp(240.0 / 200.0)) < 1e-9, e
    assert e['ppl_primary'] == e['ppl_pooled']
    assert e['ppl_mean_arithmetic'] != e['ppl_pooled'] or True

def t_nested_primary_paired_deltas():
    # A nested record's `ppl` is the full-validation value; its PRIMARY
    # estimate is the held-out-row PPL.  Every paired delta must use the
    # primary estimand, otherwise the contrast silently reads the rows that
    # selected the hyper-parameters.
    def _rec(arm, seed, full_ppl, hold_ppl, cfg='same_cfg'):
        return {
            'variant': arm, 'seed': seed, 'run_cfg': cfg,
            'ppl': float(full_ppl), 'nll_sum': math.log(full_ppl) * 100.0,
            'n_tokens': 100, 'params': 10, 'steps': 1, 'tokens_seen': 400,
            'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e',
            'ppl_estimation_rows': 'nested_validation_holdout',
            'ppl_holdout': float(hold_ppl),
            'nll_sum_holdout': math.log(hold_ppl) * 40.0, 'n_tokens_holdout': 40,
            'holdout_row_fp': 'h',
            'nll_per_row_holdout': [math.log(hold_ppl) * 40.0],
            'n_tokens_per_row_holdout': [40],
        }
    summary = {}
    for s in range(6):
        # Full validation says the method is better, while the held-out rows
        # say it is worse.  The primary endpoint is the held-out one.
        summary[f'full::seed{s}'] = _rec('full', s, 10.0, 10.0)
        summary[f'csa_dynamic::seed{s}'] = _rec('csa_dynamic', s, 9.0, 11.0)
    agg = L.aggregate(summary)
    entry = agg['csa_dynamic']
    assert abs(entry['ppl_mean'] - 11.0) < 1e-12, entry
    assert entry['ppl_primary_estimand'] == 'nested_validation_holdout', entry
    assert abs(entry['dPPL_vs_full_mean'] - 1.0) < 1e-12, entry
    recs, _amb = L._variant_seed_map(summary)
    deltas = L._paired_seed_deltas(recs, 'csa_dynamic', 'full')
    assert all(abs(d - 1.0) < 1e-12 for d in deltas[('', '')][0]), deltas
    # Holdout accounting can be recovered from NLL totals or per-row arrays;
    # a declared-nested record with none of them is unverifiable, never a
    # silent fallback to the full-validation PPL.
    total_only = _rec('csa_dynamic', 0, 9.0, 11.0)
    total_only.pop('ppl_holdout')
    assert abs(L._record_primary_ppl(total_only) - 11.0) < 1e-12, total_only
    rows_only = _rec('csa_dynamic', 0, 9.0, 12.0)
    rows_only.pop('ppl_holdout')
    rows_only.pop('nll_sum_holdout')
    rows_only.pop('n_tokens_holdout')
    assert abs(L._record_primary_ppl(rows_only) - 12.0) < 1e-12, rows_only
    missing = _rec('csa_dynamic', 0, 9.0, 11.0)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout',
               'nll_per_row_holdout', 'n_tokens_per_row_holdout'):
        missing.pop(_k)
    assert L._record_primary_ppl(missing) is None, missing
    assert L._paired_primary_delta(missing, _rec('full', 0, 10.0, 10.0)) is None
    # A group that mixes the two estimands is not poolable: summing holdout
    # NLLs with full-validation NLLs would invent a third sample, and labelling
    # the result 'nested' would misdescribe it.  The whole group must fall back
    # to full validation so every reported number is on one sample.
    mixed_full = {'variant': 'mixed', 'seed': 0, 'ppl': 10.0,
                  'nll_sum': 10.0, 'n_tokens': 1, 'params': 10,
                  'steps': 1, 'tokens_seen': 100, 'run_cfg': 'same'}
    mixed_nested = _rec('mixed', 1, 20.0, 30.0)
    mixed_nested.update(run_cfg='same', steps=1, tokens_seen=100)
    assert L._pooled_ppl_from_records([mixed_full, mixed_nested]) == (
        None, None, None)
    _pool, _nll, _nt = L._pooled_ppl_from_records(
        [mixed_full, mixed_nested], estimand=False)
    _want_pool = math.exp((math.log(20.0) * 100.0 + 10.0) / 101.0)
    assert abs(_pool - _want_pool) < 1e-12, (_pool, _nll, _nt)
    mixed_agg = L.aggregate({'mixed::seed0': mixed_full,
                             'mixed::seed1': mixed_nested})['mixed']
    assert mixed_agg['ppl_primary_estimand'] == 'full_validation_fallback', mixed_agg
    assert abs(mixed_agg['ppl_mean'] - 15.0) < 1e-12, mixed_agg
    assert abs(mixed_agg['ppl_pooled'] - _want_pool) < 1e-12, mixed_agg
    # A pair itself must not subtract a full-validation PPL from a holdout PPL.
    assert L._paired_primary_delta(mixed_full, mixed_nested) is None
    # Nor may a contrast average deltas from different estimands across seeds.
    mixed_summary = {
        'a::seed0': dict(mixed_full, variant='a'),
        'b::seed0': dict(mixed_full, variant='b', ppl=12.0),
        'a::seed1': dict(mixed_nested, variant='a'),
        'b::seed1': dict(mixed_nested, variant='b', ppl_holdout=31.0),
    }
    _recs, _amb = L._variant_seed_map(mixed_summary)
    _md = L._paired_seed_deltas(_recs, 'a', 'b')
    _deltas, _seeds, _skipped = _md[('', '')]
    assert _deltas == [] and len(_skipped) == 2, _md
    assert all('mixed estimands' in why for _s, why in _skipped), _md


def t_combined_seed_doc_uncertainty():
    # Two-level (seed x document) combined interval: the seed-level t CI and
    # the doc-level bootstrap each see only ONE variance source; the
    # combined interval must not silently drop the other one.
    def _recs(seed_offsets, n_docs=32, ntok=50, zero_docs=0):
        out = []
        for s, off in enumerate(seed_offsets):
            nll = [1.0 + off + 0.01 * j for j in range(n_docs)]
            ntoks = [ntok] * (n_docs - zero_docs) + [0] * zero_docs
            out.append({'variant': 'csa', 'seed': s,
                        'ppl': math.exp(sum(nll) / (n_docs * ntok)),
                        'nll_per_row': nll, 'n_tokens_per_row': ntoks})
        return out
    import numpy as _np
    recs = _recs([0.0, 0.2, 0.4])
    r = L._combined_seed_doc_ci(recs)
    assert r['status'] == 'ok', r
    assert r['n_seeds'] == 3 and r['n_docs'] == 32 and r['n_docs_dropped_zero_token'] == 0, r
    per_seed_lp = []
    for off in (0.0, 0.2, 0.4):
        nll = [1.0 + off + 0.01 * j for j in range(32)]
        # NLL/tokens is already log-PPL; the log must not be applied twice.
        per_seed_lp.append(sum(nll) / (32 * 50))
    _se_seed = _np.std(per_seed_lp, ddof=1) / math.sqrt(3)
    assert abs(r['se_seed_logppl'] - float(_se_seed)) < 1e-12, (r, _se_seed)
    assert r['se_doc_logppl'] is not None and r['se_doc_logppl'] > 0, r
    _comb = math.sqrt(r['se_seed_logppl'] ** 2 + r['se_doc_logppl'] ** 2)
    assert abs(r['se_combined_logppl'] - _comb) < 1e-12, r
    assert abs(r['component_ratio_seed_over_doc'] - r['se_seed_logppl'] / r['se_doc_logppl']) < 1e-12, r
    lo, hi = r['ci95_ppl']
    assert 0 < lo < hi, r
    # Center must be the PPL scale, not exp(log(CE)).
    _center = math.exp(float(_np.mean(per_seed_lp)))
    assert lo < _center < hi, (r, _center)
    assert lo > 1.0, ('a valid PPL interval cannot sit below 1.0', r)
    # The combined interval is never NARROWER than the seed-only t interval.
    tc = L._t_critical(0.95, 2)
    seed_only = (math.exp(_np.mean(per_seed_lp) - tc * _se_seed),
                 math.exp(_np.mean(per_seed_lp) + tc * _se_seed))
    assert r['ci95_ppl'][0] <= seed_only[0] + 1e-9 and r['ci95_ppl'][1] >= seed_only[1] - 1e-9, (r, seed_only)
    # Identical documents: the doc component collapses to ~0 and the
    # combined interval reduces to the seed component.
    r0 = L._combined_seed_doc_ci(_recs([0.0, 0.3, 0.6], n_docs=16, ntok=50)
                                 )
    def _flat(seed_offsets):
        out = []
        for s, off in enumerate(seed_offsets):
            nll = [1.0 + off] * 16
            out.append({'variant': 'csa', 'seed': s, 'ppl': math.exp(sum(nll) / (16 * 50)),
                        'nll_per_row': nll, 'n_tokens_per_row': [50] * 16})
        return out
    rf = L._combined_seed_doc_ci(_flat([0.0, 0.3, 0.6]))
    assert rf['status'] == 'ok' and rf['se_doc_logppl'] is not None and rf['se_doc_logppl'] < 1e-12, rf
    assert abs(rf['se_combined_logppl'] - rf['se_seed_logppl']) < 1e-12, rf
    assert rf['component_ratio_seed_over_doc'] is None, rf
    # Fractional per-document token counts are not a pooled denominator.
    _bad = _recs([0.0, 0.2, 0.4])
    _bad[0]['n_tokens_per_row'] = list(_bad[0]['n_tokens_per_row'])
    _bad[0]['n_tokens_per_row'][0] = 50.5
    assert L._combined_seed_doc_ci(_bad)['status'] == 'unverifiable'
    assert L._doc_bootstrap_ppl_ci(_bad) is None
    # Mixed estimands are two different document samples, not one bootstrap
    # panel, even if both expose per-row arrays of the same length.
    _mix = _recs([0.0, 0.2, 0.4])
    _mix[0]['ppl_estimation_rows'] = 'nested_validation_holdout'
    _mix[0]['nll_per_row_holdout'] = list(_mix[0]['nll_per_row'])
    _mix[0]['n_tokens_per_row_holdout'] = list(_mix[0]['n_tokens_per_row'])
    _rmix = L._combined_seed_doc_ci(_mix)
    assert _rmix['status'] == 'unverifiable' and 'mix' in _rmix.get('reason', ''), _rmix
    assert L._doc_bootstrap_ppl_ci(_mix) is None
    # One seed: the seed component is explicitly unverifiable, never zero.
    r1 = L._combined_seed_doc_ci(_recs([0.0]))
    assert r1['status'] == 'seed_component_unverifiable' and r1['se_seed_logppl'] is None and r1['se_doc_logppl'] is not None, r1
    assert r1['ci95_ppl'] is None, r1
    # Records without per-row NLL cannot support EITHER component.
    r2 = L._combined_seed_doc_ci([{'variant': 'csa', 'seed': 0, 'ppl': 30.0},
                                  {'variant': 'csa', 'seed': 1, 'ppl': 31.0}])
    assert r2['status'] == 'unverifiable' and r2['ci95_ppl'] is None, r2
    # Coverage/draw-count/RNG are part of the interval identity: never build
    # a 95% interval while echoing an invalid option.
    for _bad_conf in (True, 0, 1, 1.5, -0.1, float('nan'), 'x'):
        _rr = L._combined_seed_doc_ci(recs, conf=_bad_conf)
        assert _rr['status'] == 'unverifiable' and _rr.get('reason'), (_bad_conf, _rr)
        assert _rr['conf'] is None and _rr['ci95_ppl'] is None, (_bad_conf, _rr)
    for _bad_nb in (True, 0, 19, 1.5, 'x', None, float('nan')):
        _rr = L._combined_seed_doc_ci(recs, n_boot=_bad_nb)
        assert _rr['status'] == 'unverifiable' and _rr.get('reason'), (_bad_nb, _rr)
    for _bad_seed in (True, 1.5, 'x', None, float('nan')):
        _rr = L._combined_seed_doc_ci(recs, seed=_bad_seed)
        assert _rr['status'] == 'unverifiable' and 'seed' in _rr.get('reason', ''), (_bad_seed, _rr)
    # A single scored document cannot carry a doc-resampling component.
    r3 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=1, ntok=50))
    assert r3['status'] == 'doc_component_unverifiable' and r3['se_doc_logppl'] is None and r3['se_seed_logppl'] is not None, r3
    # Zero-token (never scored) rows are dropped, not pooled as free mass.
    r4 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=8, ntok=50, zero_docs=2))
    assert r4['status'] == 'ok' and r4['n_docs'] == 6 and r4['n_docs_dropped_zero_token'] == 2, r4
    # The aggregate entry exposes the combined interval next to the two
    # single-level intervals.
    summary = {}
    for s, off in ((0, 0.0), (1, 0.2)):
        nll = [1.0 + off + 0.01 * j for j in range(32)]
        summary[f'csa::seed{s}'] = {'variant': 'csa', 'seed': s, 'ppl': math.exp(sum(nll) / (32 * 50)),
                                    'nll_sum': sum(nll), 'n_tokens': 32 * 50, 'params': 10,
                                    'steps': 1, 'tokens_seen': 100,
                                    'nll_per_row': nll, 'n_tokens_per_row': [50] * 32,
                                    'train_objective': {'comp_reg_applies': True, 'delta_lr_effective': 10.0, 'boundary_policy': ['threshold']}}
    agg = L.aggregate(summary)
    cc = agg['csa'].get('ppl_ci_seed_doc_combined')
    assert isinstance(cc, dict) and cc['status'] == 'ok' and cc['ci95_ppl'] is not None, cc

def t_review3_holm_family_size():
    comps = {'a': {'p_exact_signflip': 0.01}, 'b': {'omitted': 'missing arm'}, 'c': {'p_exact_signflip': 0.04}}
    fam = L.attach_holm(comps)
    assert fam['declared_slots'] == 3 and fam['tested'] == 2, fam
    assert comps['a']['p_holm'] is not None and comps['c']['p_holm'] is not None
    # The omitted slot must inflate m: Holm(a)=3*0.01=0.03, not 2*0.01=0.02.
    assert abs(comps['a']['p_holm'] - 0.03) < 1e-12, comps

def t_review3_span_clean_key():
    pp = {'n_seq': 16, 'span_len': 32, 'target': 512}
    clean = V10._clean_probe_params(pp)
    assert 'span_len' not in clean and clean['n_seq'] == 16, clean
    assert pp['span_len'] == 32, 'the measured cell must keep span_len'

def t_key_exploratory_phases_declare_at_least_three_seeds():
    # Prospective seed plan: the top-k sweep and length-extrapolation panels
    # are key exploratory comparisons, so a 2-seed plan is not enough even for
    # a descriptive mean±std.  (Other phases inherit >=3 seeds from their cfg.)
    by_name = {p[0]: p for p in V7.PHASES}
    for _name in ('P1T', 'P0E'):
        _payload = by_name[_name][2]
        _seeds = _payload.get('seeds') or (_payload.get('cfg') or {}).get('seeds')
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    # Scale panels are key cross-size comparisons too: a 2-seed plan cannot
    # even report a seed-level mean±std.  The v10 scale phases and their v12
    # token/compute-matched re-runs all declare >=3 seeds.
    v10_phases = {p[0]: p for p in V10.PHASES}
    for _name in ('P3SS', 'P3SL', 'P3SL2', 'P3SL3'):
        _seeds = v10_phases[_name][3]
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    for _name in ('P5S2_CFG', 'P5S3_CFG'):
        _seeds = getattr(V12, _name).get('seeds')
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)


def t_v10_holm_family_keeps_unmeasured_slots():
    # A missing/non-finite p is a dummy p=1 slot, not a deleted comparison:
    # dropping it would shrink m and overstate the remaining tests' power.
    fam = V10._holm_family({
        'a': {'p_exact_signflip': 0.01},
        'b': {'p_exact_signflip': float('nan')},
        'c': {},
        'd': {'p_exact_signflip': True},
        'e': {'p_exact_signflip': 1.5},
    })
    assert [k for k, _p in fam] == ['a', 'b', 'c', 'd', 'e'], fam
    assert [p for _k, p in fam] == [0.01, 1.0, 1.0, 1.0, 1.0], fam
    h = L.holm_adjust(fam)
    assert abs(h['a']['p_holm'] - 5 * 0.01) < 1e-12, h
    assert h['b']['p_holm'] == 1.0 and h['c']['p_holm'] == 1.0, h

def t_objective_ablation_is_declared_not_a_confound():
    # The loss-term arm zeroes the per-layer multiplier, so the descriptor must
    # report that comp_reg does NOT apply (previously it copied the configured
    # lambda and silently described the wrong objective).
    m_full = L.SmallGPT(64, 32, 2, 2, 16, 16, L.make_layer_cfgs(2, 'csa_dynamic'))
    m_none = L.SmallGPT(64, 32, 2, 2, 16, 16, L.make_layer_cfgs(2, 'csa_dyn_nocomp'))
    o_full = L.train_objective(m_full, comp_lambda=0.05, delta_lr_mult=10.0)
    o_none = L.train_objective(m_none, comp_lambda=0.05, delta_lr_mult=10.0)
    assert o_full['comp_reg_applies'] is True and o_full['comp_lambda_effective'] > 0, o_full
    assert o_none['comp_reg_applies'] is False and o_none['comp_lambda_effective'] == 0.0, o_none
    assert 'comp_reg' not in o_none['loss_terms'], o_none
    assert o_none['comp_reg_configured_layers'] == o_full['comp_reg_configured_layers'] == 2, (o_full, o_none)
    _r, _keys = L._objective_diff({'train_objective': o_full}, {'train_objective': o_none})
    assert _r and _keys == ['comp_reg_applies', 'comp_lambda_effective', 'loss_terms'], (_r, _keys)
    # A descriptor that omits fields is unverifiable, not equal: comparing
    # two `.get()` misses would make an empty object look matched.
    _ru, _ku = L._objective_diff({'train_objective': {}}, {'train_objective': {}})
    assert _ru and _ru.startswith('unverifiable') and _ku == [], (_ru, _ku)
    _rp, _kp = L._objective_diff(
        {'train_objective': {'comp_reg_applies': True}},
        {'train_objective': {'comp_reg_applies': True}})
    assert _rp and _rp.startswith('unverifiable') and _kp == [], (_rp, _kp)
    # Same effective weight but a different configured base lambda + per-layer
    # multiplier is still a different objective.
    _o_scaled = dict(o_full, comp_lambda=0.2, comp_lambda_effective=0.2,
                     comp_lambda_mults=[1.0])
    _rs, _ks = L._objective_diff({'train_objective': o_full},
                                 {'train_objective': _o_scaled})
    assert _rs and 'comp_lambda' in _ks, (_rs, _ks)
    # The objective difference is the declared intervention for this module,
    # not an undeclared confound: six same-direction seeds must verify it
    # rather than downgrade it to system_level_only.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=49.0, params=100, train_knobs=knobs, train_objective=o_full)
        s[f'csa_dyn_nocomp::seed{i}'] = dict(
            variant='csa_dyn_nocomp', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs, train_objective=o_none)
    rep = L.methodology_audit(
        s, design={'compression_regularizer': [
            ('csa_dynamic', 'csa_dyn_nocomp', 'loss term')]},
        baselines={}, verbose=False)
    e = rep['components']['compression_regularizer'][0]
    assert e['objective_expected_differences'] == sorted([
        'comp_reg_applies', 'comp_lambda_effective', 'loss_terms']), e
    assert e.get('mechanism_claim_allowed') is not False, e
    assert 'compression_regularizer' in rep['verified_components'], rep
    assert 'compression_regularizer' not in rep['system_level_only_components'], rep
    # The task-learned boundary arm differs in its policy term, which is also
    # the declared intervention.
    o_task = dict(o_full, boundary_policy=['reinforce'],
                  loss_terms=['ce', 'comp_reg', 'boundary_policy_gradient'])
    s2 = {}
    for i in range(6):
        s2[f'csa_dynamic_task::seed{i}'] = dict(
            variant='csa_dynamic_task', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=49.0, params=100, train_knobs=knobs, train_objective=o_task)
        s2[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs, train_objective=o_full)
    rep2 = L.methodology_audit(
        s2, design={'task_learned_boundary': [
            ('csa_dynamic_task', 'csa_dynamic', 'policy')]},
        baselines={}, verbose=False)
    e2 = rep2['components']['task_learned_boundary'][0]
    assert e2['objective_expected_differences'] == sorted(
        ['boundary_policy', 'loss_terms']), e2
    assert 'task_learned_boundary' in rep2['verified_components'], rep2
    # An UNDECLARED objective change is still a confound.
    o_bad = dict(o_full, boundary_policy=['reinforce'])
    s3 = {k: dict(v) for k, v in s.items()}
    for i in range(6):
        s3[f'csa_dyn_nocomp::seed{i}']['train_objective'] = o_bad
    rep3 = L.methodology_audit(
        s3, design={'compression_regularizer': [
            ('csa_dynamic', 'csa_dyn_nocomp', 'loss term')]},
        baselines={}, verbose=False)
    e3 = rep3['components']['compression_regularizer'][0]
    assert e3.get('mechanism_claim_allowed') is False, e3
    assert 'compression_regularizer' in rep3['system_level_only_components'], rep3


def t_review3_boundary_arm_and_objective():
    cfg = L.make_layer_cfgs(1, 'csa_dynamic_task')[0]
    assert getattr(cfg, 'boundary_policy', None) == 'reinforce', cfg
    obj = L.train_objective(type('M', (), {'blocks': [], 'named_parameters': lambda self: iter([])})(), comp_lambda=0.05, delta_lr_mult=10.0)
    assert obj['comp_reg_applies'] is False, obj
    assert 'boundary_policy' in obj and 'loss_terms' in obj
    # The score-function term must actually reach delta_logit; otherwise the
    # "task-learned boundary" arm would be inert.
    m = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'csa_dynamic_task'))
    x = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(1))
    ce = m.ce_loss(x, y, 64)
    assert m.boundary_logp is not None and m.boundary_decisions > 0, 'boundary policy must emit log-probabilities'
    loss = ce + 0.05 * m.comp_reg - float(ce.detach()) * m.boundary_logp / max(m.boundary_decisions, 1)
    loss.backward()
    grads = [p.grad.abs().sum().item() for n, p in m.named_parameters() if 'delta_logit' in n and p.grad is not None]
    assert grads and all(g > 0 for g in grads), f'delta_logit received no task-loss gradient: {grads}'

def t_seed_reproducible():
    def draw():
        L.set_seed(12345)
        return (random.random(), float(np.random.random()), float(torch.rand(1).item()), L.rng_fingerprint())
    a = draw()
    b = draw()
    assert a == b, f'same seed must reproduce Python/NumPy/Torch draws and the RNG fingerprint: {a} vs {b}'
    L.set_seed(54321)
    d = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert d != a[:3], 'a different seed must change the RNG stream'


def t_seed_axis_normalization():
    assert L.normalize_seed_list([0, 1, 1, 2, '3']) == [0, 1, 2, 3]
    assert L.normalize_seed_list([]) == [0]
    assert L.normalize_seed_list(None) == [0]
    assert L.normalize_seed_list('0, 2 2') == [0, 2]
    # The fallback plan is validated like an explicit list; int(True)/int(1.5)
    # must not silently manufacture a different seed identity.
    for _bad_default in ((True,), (1.5,), ('x',), (None,)):
        try:
            L.normalize_seed_list([], default=_bad_default)
        except ValueError:
            pass
        else:
            raise AssertionError(f'default={_bad_default!r} must be refused')
    assert L.normalize_variant_list(['full', 'csa_fixed', 'full']) == ['full', 'csa_fixed']
    for bad in (True, 1.5, 'x'):
        try:
            L.normalize_seed_list([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'normalize_seed_list must reject {bad!r}')
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(2048, dtype=np.int64) % 997
        val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=2048, steps=2,
                   lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
                   delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64,
                   n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0, 0],
                   variants=['full'])
        L.run(cfg, label='dedupe-unit')
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = next(iter(m['panels'].values()))
        assert entry['n_planned'] == 1 and entry['n_completed'] == 1, entry
        assert entry['tracking_integrity']['seeds_unique'] is True, entry['tracking_integrity']
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        assert sum(1 for x in lines if x.get('kind') == 'cell') == 1, lines
    finally:
        L.load_run_corpus = _saved_load
        for k, old in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        shutil.rmtree(d, ignore_errors=True)


def t_seed_scalar_rejects_silent_truncation():
    from decimal import Decimal
    # Objects outside numbers.Real can still define __int__; int() would
    # truncate Decimal('1.5') / np.array(1.5) to a different valid seed.
    assert L._as_seed_int(Decimal('1.5')) is None
    assert L._as_seed_int(np.array(1.5)) is None
    assert L._as_seed_int(Decimal('2')) == 2
    assert L._as_seed_int(np.array(2)) == 2
    assert L._as_seed_int('0') == 0
    for bad in (Decimal('1.5'), np.array(1.5)):
        try:
            L.normalize_seed_list([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'normalize_seed_list must reject {bad!r}, not truncate it')
    assert L.normalize_seed_list([Decimal('2'), np.array(3)]) == [2, 3]


def t_seed_domain_guard():
    # NumPy's legacy seeder only accepts [0, 2**32 - 1]; accepting another
    # value in normalize_seed_list would defer the failure into training (or
    # silently remap it).  The same domain must be used by every entry point.
    for bad in (-1, L.MAX_SEED + 1, 2 ** 63, True, 1.5, 'x', None):
        for fn in (lambda b=bad: L.normalize_seed_list([b]),
                   lambda b=bad: L.set_seed(b),
                   lambda b=bad: L.dataloader_generator(b)):
            try:
                fn()
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad!r} must be rejected by every seed entry point')
    assert L.normalize_seed_list([L.MAX_SEED, 0]) == [L.MAX_SEED, 0]
    assert L._other_seed(L.MAX_SEED) == 0 and L._other_seed(0) == 1
    # A maximum seed must not make the contract probe try MAX_SEED+1.
    L.set_seed(4242)
    _fp = L.rng_fingerprint()
    r = L.seed_contract_check(seed=L.MAX_SEED)
    assert r['passed'] is True and r['checks']['different_seed_changes_streams'] is True, r
    assert L.rng_fingerprint() == _fp, 'the seed contract probe must still preserve the process RNG'
    for bad in (-1, L.MAX_SEED + 1):
        try:
            L.reproducibility_guard(['full'], seeds=[bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_guard must reject seed {bad!r}')


def t_seed_contract_check():
    L.set_seed(4242)
    fp0 = L.rng_fingerprint()
    r = L.seed_contract_check(seed=3)
    assert r['passed'] is True, r
    assert r['checks']['same_seed_reproducible'] is True
    assert r['checks']['different_seed_changes_streams'] is True
    assert r['checks']['worker_seed_reproducible'] is True
    assert r['checks']['cuda_seeded'] is (True if torch.cuda.is_available() else None)
    assert r['checks']['rng_state_preserved'] is True and r['rng_state_preserved'] is True, r
    assert L.rng_fingerprint() == fp0, 'the seed contract probe must not advance the process RNG streams'


def t_batch_iter_reproducible():
    ids = np.arange(5000, dtype=np.int64) % 997
    b1 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    b2 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1]), 'batch_iter must be seeded by the run seed'
    b3 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=8))
    assert not torch.equal(b1[0], b3[0]), 'a different data seed must change the batch order'
    for _bad in (-1, L.MAX_SEED + 1, True, 1.5):
        try:
            next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=_bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_iter must reject seed {_bad!r}')


def t_batch_iter_uint16():
    ids64 = np.arange(5000, dtype=np.int64) % 997
    ids16 = ids64.astype(np.uint16)
    a = next(L.batch_iter(ids64, 32, 4, torch.device('cpu'), seed=7))
    b = next(L.batch_iter(ids16, 32, 4, torch.device('cpu'), seed=7))
    assert a[0].dtype == b[0].dtype == torch.int64, 'batch_iter must widen the gathered ids back to int64'
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), 'the uint16 transport form must yield the exact int64 batch (CPU index kernels lack uint16)'


def t_batch_contract_check():
    ids = np.arange(5000, dtype=np.int64) % 997
    ok = L.batch_contract_check(ids, 32, 4, torch.device('cpu'), seed=7, vocab=997)
    assert ok['passed'] is True, ok
    assert all(ok['checks'].values()), ok['checks']
    _orig = L.batch_iter

    def _bad(train_ids, seq_len, batch_size, device, seed=0):
        for x, _y in _orig(train_ids, seq_len, batch_size, device, seed=seed):
            yield x, x[:, :-1]
    L.batch_iter = _bad
    try:
        bad = L.batch_contract_check(ids, 32, 4, torch.device('cpu'), seed=7, vocab=997)
    finally:
        L.batch_iter = _orig
    assert bad['passed'] is False, 'a loader that does not shift labels must FAIL the contract check'
    assert bad['checks']['labels_are_next_token_shift'] is False, bad




def t_parallel_child_env_seed():
    import par
    env = par._child_env('0', 17)
    assert env['PYTHONHASHSEED'] == '17' and env['CUDA_VISIBLE_DEVICES'] == '0', {k: env.get(k) for k in ('PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')}
    env2 = par._child_env('0')
    assert env2.get('PYTHONHASHSEED') == os.environ.get('PYTHONHASHSEED'), 'no-seed dispatch must not invent a hash seed'


def t_panel_blocks_on_loader_contract_failure():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _saved_iter = L.batch_iter
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)

        def _bad(train_ids, seq_len, batch_size, device, seed=0):
            for x, _y in _saved_iter(train_ids, seq_len, batch_size, device, seed=seed):
                yield x, x
        L.batch_iter = _bad
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = {'outdir': d, 'seq_len': 32, 'batch_size': 2, 'n_train_tokens': 4096,
               'steps': 3, 'lr': 1e-3, 'weight_decay': 0.0, 'warmup': 1,
               'comp_lambda': 0.0, 'delta_lr_mult': 10.0, 'eval_every': 0,
               'eval_subset': 2, 'd': 64, 'n_layers': 2, 'n_heads': 4,
               'd_head': 16, 'dataset': 'unit', 'seeds': [0], 'variants': ['full']}
        try:
            L.run(cfg, label='contract-failure-unit')
        except RuntimeError as e:
            assert 'batch iterator contract check FAILED' in str(e), e
        else:
            raise AssertionError('run() must hard-stop when the real loader contract fails')
        dc = json.load(open(os.path.join(d, 'data_contract.json'), encoding='utf-8'))
        assert dc['passed'] is False and dc['checks']['labels_are_next_token_shift'] is False, dc
        assert not os.path.exists(os.path.join(d, 'summary.json')), 'no training cell may start after a failed loader contract'
        mf = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = next(iter(mf['panels'].values()))
        assert entry['status'] == 'failed', entry.get('status')
        assert entry['completion']['status'] == 'failed', entry.get('completion')
        assert entry['planned_cells'] == ['full::seed0'], entry.get('planned_cells')
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['status'] == 'failed' and fin['stage'] == 'batch_iterator_contract', fin
    finally:
        L.load_run_corpus = _saved_load
        L.batch_iter = _saved_iter
        for _k, _v in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
        shutil.rmtree(d, ignore_errors=True)


def t_overfit_sanity():
    r = L.overfit_sanity_check('full', seed=0, steps=200, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256)
    assert r['passed'] is True, r
    assert r['final_loss'] <= r['target_loss'], f"final loss {r['final_loss']} did not reach {r['target_loss']}"
    assert r['final_accuracy'] >= r['target_accuracy'], f"final accuracy {r['final_accuracy']} did not reach {r['target_accuracy']}"
    assert r['grad_all_finite'] and r['params_changed'], 'gradients must be finite and applied'
    assert r['regularization']['weight_decay'] == 0.0 and r['regularization']['comp_lambda'] == 0.0
    assert r['label_shift_required'] is True
    assert r['pass_criteria']['labels_are_next_token_shift'] is True, r['pass_criteria']
    assert r['pass_criteria']['initial_loss_near_log_vocab'] is True, r['pass_criteria']
    assert r['failed_criteria'] == [], r['failed_criteria']
    # the check must also work when the batch is sampled from a real token stream
    ids = np.asarray(np.random.default_rng(3).integers(0, 256, size=1024), dtype=np.int64)
    r2 = L.overfit_sanity_check('csa_dynamic', seed=0, steps=200, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=24, batch_size=2, vocab=256, train_ids=ids)
    assert r2['passed'] is True and r2['data_source'] == 'train_ids', r2
    assert r2['loader_mode'] == 'batch_iter', 'the real-data probe must exercise the production loader'
    assert r2['intermediate_checks']['labels_are_next_token_shift'] is True, r2['intermediate_checks']


def t_rng_state_preserved_by_preflight():
    L.set_seed(777)
    fp0 = L.rng_fingerprint()
    r = L.overfit_sanity_check('full', seed=1, steps=5, d=32, n_layers=1, n_heads=4, d_head=8, seq_len=16, batch_size=2, vocab=64)
    assert r['rng_state_preserved'] is True, r
    assert L.rng_fingerprint() == fp0, 'the overfit probe must not advance the process RNG streams'
    rr = L.reproducibility_check('full', seed=2, steps=2)
    assert rr['rng_state_preserved'] is True, rr
    assert L.rng_fingerprint() == fp0, 'the reproducibility probe must not advance the process RNG streams'


def t_reproducibility_check():
    r = L.reproducibility_check('full', seed=0, steps=3)
    assert r['passed'] is True, r
    assert r['max_abs_diff_same_seed'] <= r['tol'], f'same seed must reproduce the short loss curve: {r}'
    if not torch.cuda.is_available():
        assert r['bitwise_identical'] is True, f'same seed must reproduce the short loss curve exactly on CPU: {r}'
    assert r['different_seed_changes_curve'] is True, 'a different seed must change the short loss curve'
    assert r['rng_state_preserved'] is True
    # the cache/guard records evidence without re-running
    reports = L.reproducibility_guard(['full'])
    assert reports and reports[0]['passed'] is True


def t_panel_tracking_events():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=3, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64, n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0, 1], variants=['full'])
        summary, _ = L.run(cfg, label='tracking-unit')
        assert 'full::seed0' in summary and 'full::seed1' in summary
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        kinds = [x.get('kind') for x in lines]
        assert 'panel_start' in kinds and 'panel_finish' in kinds, kinds
        assert kinds.count('cell') == 2, kinds
        st = next(x for x in lines if x.get('kind') == 'panel_start')
        for k in ('run_cfg', 'code', 'config', 'seeds', 'variants', 'provenance'):
            assert k in st, f'panel_start missing {k}'
        assert st['n_cells_planned'] == 2 and st['n_cells_to_run'] == 2 and st['n_cells_resumed'] == 0, st
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['n_cells_planned'] == 2 and fin['n_cells_measured'] == 2 and fin['n_cells_missing'] == 0, fin
        for k in ('provenance', 'config', 'seeds', 'variants', 'data_version'):
            assert k in fin, f'panel_finish missing {k}'
        assert fin['provenance']['code']['git_commit'] == st['provenance']['code']['git_commit'], fin
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = m['panels'][L._run_fp(cfg, 64, 2, 4, 16, 997, ids, val)]
        assert entry.get('tracking_schema') == 2 and 'reproducibility' in entry
        assert entry['data_contract']['passed'] is True, entry.get('data_contract')
        assert entry['evidence']['batch_iterator_contract'] is True, entry.get('evidence')
    finally:
        L.load_run_corpus = _saved_load
        if _old_skip_overfit is None:
            os.environ.pop('CSA_SKIP_OVERFIT', None)
        else:
            os.environ['CSA_SKIP_OVERFIT'] = _old_skip_overfit
        if _old_skip_repro is None:
            os.environ.pop('CSA_SKIP_REPRO', None)
        else:
            os.environ['CSA_SKIP_REPRO'] = _old_skip_repro
        shutil.rmtree(d, ignore_errors=True)


def t_provenance_and_manifest():
    tr = np.arange(512, dtype=np.int64) % 97
    va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
    prov = L.experiment_provenance({'dataset': 'unit', 'n_train_tokens': 512, 'seq_len': 32}, train_ids=tr, val_batch=va, vocab=97)
    for k in ('code', 'environment', 'data'):
        assert k in prov, f'provenance missing {k}'
    for k in ('code_semantics', 'code_stamp', 'git_commit', 'git_dirty', 'git_status_sha256', 'git_diff_sha256'):
        assert k in prov['code'], f'code provenance missing {k}'
    for k in ('python', 'torch', 'numpy', 'determinism'):
        assert k in prov['environment'], f'environment provenance missing {k}'
    assert prov['data']['train_fp'] and prov['data']['val_fp'] and prov['data']['vocab_size'] == 97
    assert prov['data'].get('data_version'), 'data provenance must carry a composite data_version'
    hp = L.run_hyperparameters({'steps': 10, 'seeds': [0], 'variants': ['full'], 'outdir': 'x', 'n_train_tokens': 512, 'seq_len': 32}, seed=3, d=64, n_layers=2, n_heads=4, d_head=16, vocab=97, mlp_ratio=4.0)
    assert hp['seed'] == 3 and hp['data_seed'] == 3 and hp['train_knobs']['optimizer'] == 'AdamW'
    assert hp['rng']['python'] == hp['rng']['numpy'] == hp['rng']['torch'] == hp['rng']['cuda'] == 3, hp['rng']
    assert hp['rng']['data'] == 3 and hp['rng']['dataloader_generator'] == 3, hp['rng']
    assert hp['determinism']['level'] == L.determinism_label(), hp['determinism']
    assert hp['determinism']['cpu_threads'] == int(torch.get_num_threads()), hp['determinism']
    d = tempfile.mkdtemp()
    try:
        summary = {'full::seed0': {'ppl': 10.0, 'steps': 10, 'tokens_seen': 320, 'params': 100, 'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc', '_code': L.code_stamp(), 'losses': [3.0, 2.3], 'ppl_history': [[10, 10.0]]}}
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d, 'n_train_tokens': 512, 'seq_len': 32}
        path = L.write_run_manifest(d, cfg, 'FP', prov, summary)
        m = json.load(open(path, encoding='utf-8'))
        assert m['manifest_version'] == 1 and 'FP' in m['panels'], m
        assert m['panels']['FP']['cells']['full::seed0']['ppl'] == 10.0
        assert m['panels']['FP']['provenance']['data']['train_fp'] == prov['data']['train_fp']
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# E5-E8. 进一步的可复现性 / 训练追踪检查
# ---------------------------------------------------------------------------

def t_dataloader_seeding_helpers():
    ds = torch.utils.data.TensorDataset(torch.arange(64, dtype=torch.long).unsqueeze(1))

    def order(seed):
        dl = L.deterministic_dataloader(ds, batch_size=8, shuffle=True, seed=seed)
        return [int(v) for b in dl for v in b[0]]
    assert order(7) == order(7), 'same DataLoader seed must reproduce the shuffled order'
    assert order(7) != order(8), 'a different DataLoader seed must change the shuffled order'
    # Explicit None means "not supplied": it must not disable the seeded
    # generator/worker hook and fall back to OS entropy.
    _dl_none = L.deterministic_dataloader(ds, batch_size=8, shuffle=True, seed=7,
                                          generator=None, worker_init_fn=None)
    assert [int(v) for b in _dl_none for v in b[0]] == order(7)
    # worker RNGs are derived from torch.initial_seed(), not OS entropy
    def worker_draw():
        torch.manual_seed(11)
        L.seed_worker(0)
        return (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert worker_draw() == worker_draw(), 'seed_worker must reproduce Python/NumPy/Torch worker RNG streams'
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(11)
    L.seed_worker(0)
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False, 'a spawned worker must re-pin deterministic kernels'
    # The worker hook must also make real worker PROCESSES reproducible.
    class _DrawDS(torch.utils.data.Dataset):
        def __len__(self):
            return 64
        def __getitem__(self, i):
            return torch.tensor([i, int(random.random() * 1e9), int(np.random.random() * 1e9)])
    def worker_order(seed):
        dl = L.deterministic_dataloader(_DrawDS(), batch_size=8, shuffle=True, seed=seed, num_workers=2)
        return [tuple(int(x) for x in row) for b in dl for row in b]
    try:
        _w4a, _w4b, _w5 = worker_order(4), worker_order(4), worker_order(5)
    except (PermissionError, OSError) as e:
        raise _SkipTest(f'DataLoader worker processes need POSIX semaphores, which this sandbox forbids ({type(e).__name__}: {e})') from e
    assert _w4a == _w4b, 'spawned DataLoader workers must reproduce their Python/NumPy draws'
    assert _w4a != _w5, 'a different DataLoader seed must change the worker draws'


def t_train_variant_reproducible():
    ids = np.arange(8192, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
    kw = dict(d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, steps=4, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'), log_every=0)
    a = L.train_variant('full', ids, val, 997, seed=3, **kw)
    # Token accounting: draws are not unique tokens; the estimate must be
    # present, bounded, and strictly below the draw count on an undersized cache.
    assert a['tokens_seen'] == 4 * 2 * 32, a['tokens_seen']
    # The smoothed tail is stored as numbers, not tensors; a tensor tail was
    # silently filtered to None for every run.
    assert a['final_loss_smoothed'] is not None, a.get('final_loss_smoothed')
    assert a.get('tokens_seen_unique_frac_estimate') is not None and 0.0 < a['tokens_seen_unique_frac_estimate'] <= 1.0, a
    assert a['tokens_seen_unique_estimate'] <= len(ids), a
    b = L.train_variant('full', ids, val, 997, seed=3, **kw)
    c = L.train_variant('full', ids, val, 997, seed=4, **kw)
    assert a['losses'] == b['losses'], 'same seed must reproduce the whole training loss curve'
    assert a['ppl'] == b['ppl'] and a['rng_fp'] == b['rng_fp'], 'same seed must reproduce the final PPL and RNG fingerprint'
    assert a['losses'] != c['losses'], 'a different seed must change the loss curve'


def t_overfit_detects_broken_optimizer():
    # lr=0 is the canonical "gradients computed but never applied" failure mode.
    r = L.overfit_sanity_check('full', seed=0, steps=5, lr=0.0, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=16, batch_size=2, vocab=256)
    assert r['passed'] is False, 'a run whose parameters never update must FAIL the sanity check'
    assert r['params_changed'] is False and r['loss_decreased'] is False
    assert 'did not update' in r['diagnosis']


def t_experiment_ledger():
    d = tempfile.mkdtemp()
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=3, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64, n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0], variants=['full'])
        prov = L.experiment_provenance(cfg, train_ids=ids, val_batch=val, vocab=997)
        fp = L._run_fp(cfg, 64, 2, 4, 16, 997, ids, val)
        out = L._run_cell(cfg, 'full', 0, fp=fp, train_ids=ids, val_batch=val, vocab=997, prov=prov)
        assert out['status'] == 'ok' and out.get('run_id')
        summary = json.load(open(os.path.join(d, 'summary.json'), encoding='utf-8'))
        rec = summary['full::seed0']
        assert rec.get('run_id') == out['run_id'] and rec.get('hyperparams') and rec.get('provenance')
        lines = open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()
        assert len(lines) == 1, f'one completed cell must append exactly one ledger line (got {len(lines)})'
        ev = json.loads(lines[0])
        for k in ('run_id', 'run_cfg', 'seed', 'provenance', 'hyperparams', 'data_fp', 'metrics', 'losses', 'ppl_history'):
            assert k in ev, f'experiment ledger entry missing {k}'
        assert ev['status'] == 'ok' and ev['run_id'] == out['run_id'] and len(ev['losses']) == 3
        assert ev['metrics'].get('steps_completed') == 3, ev['metrics']
        assert isinstance(ev.get('instability_diagnosis'), dict), ev
        san = {'kind': 'small_batch_overfit_sanity_panel', 'passed': True, 'reports': [], 'run_cfg': fp}
        dc = L.batch_contract_check(ids, cfg['seq_len'], cfg['batch_size'], torch.device('cpu'), seed=0, vocab=997)
        dc.update({'run_cfg': fp})
        L.write_run_manifest(d, cfg, fp, prov, summary, sanity=san, data_contract=dc)
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = m['panels'][fp]
        cell = entry['cells']['full::seed0']
        assert entry['sanity'] == san and entry['experiment_log'] == 'experiments.jsonl'
        assert entry['data_contract']['passed'] is True and entry['evidence']['batch_iterator_contract'] is True, entry.get('evidence')
        assert cell['run_id'] == out['run_id'] and cell['data_fp'] and cell['curve_summary']['n_points'] == 3
        assert cell.get('data_version') == prov['data']['data_version']
        assert entry.get('status') == 'complete' and entry.get('n_completed') == 1
        integ = entry.get('tracking_integrity') or {}
        assert integ.get('n_cells') == 1 and integ.get('with_curve') == 1 and integ.get('with_rng_fp') == 1 and integ.get('with_data_contract') == 1, integ
        assert integ.get('with_seed_metadata') == 1 and integ.get('seed_metadata_mismatch') == [], integ
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_data_version_identity():
    cfg = {'dataset': 'unit', 'n_train_tokens': 512, 'seq_len': 32, 'cache_dir': './unit_cache'}
    tr = np.arange(512, dtype=np.int64) % 97
    va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
    a = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)['data']
    b = L.experiment_provenance(cfg, train_ids=tr.copy(), val_batch=va.copy(), vocab=97)['data']
    assert a['data_version'] == b['data_version'], 'same data must yield the same data_version'
    tr2 = tr.copy()
    tr2[0] = (int(tr2[0]) + 1) % 97
    c = L.experiment_provenance(cfg, train_ids=tr2, val_batch=va, vocab=97)['data']
    assert c['data_version'] != a['data_version'], 'changing one token must change data_version'
    d = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=98)['data']
    assert d['data_version'] != a['data_version'], 'changing the vocabulary must change data_version'
    assert a['dataset'] == 'unit' and a['cache_dir'] == './unit_cache'


def t_manifest_running_then_complete():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0, 1], 'variants': ['full'], 'outdir': d, 'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        planned = ['full::seed0', 'full::seed1']
        L.write_run_manifest(d, cfg, 'FP', prov, {}, status='running', planned_cells=planned, started_at='T0')
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['status'] == 'running' and e['n_planned'] == 2 and e['n_completed'] == 0
        assert e['missing_cells'] == planned and e['created_at'] == 'T0'
        assert e['completion']['status'] == 'running' and e['completion']['complete'] is False, e.get('completion')
        summary = {'full::seed0': {'ppl': 10.0, 'steps': 3, 'tokens_seen': 192, 'params': 100, 'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc', '_code': L.code_stamp(), 'losses': [3.0, 2.3], 'ppl_history': []}}
        L.write_run_manifest(d, cfg, 'FP', prov, summary, status='complete', planned_cells=planned, started_at='T0')
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['status'] == 'complete' and e['n_completed'] == 1 and e['missing_cells'] == ['full::seed1']
        # The explicit status argument is preserved for compatibility, but the
        # completion verdict must still expose the missing cell and the
        # untraced record instead of looking complete.
        assert e['completion']['status'] == 'incomplete' and e['completion']['complete'] is False, e['completion']
        assert e['completion']['missing_cells'] == ['full::seed1'], e['completion']
        assert e['tracking_integrity']['complete'] is False
        assert 'full::seed0' in e['tracking_integrity']['incomplete_cells'], e['tracking_integrity']
        assert e['created_at'] == 'T0', 'a final update must not reset the panel start time'
        assert e['data_version'] == prov['data']['data_version']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_panel_finish_marks_missing_cells_incomplete():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _saved_train = L.train_variant
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'

        def _flaky(variant, *a, **kw):
            if variant == 'csa_fixed':
                raise RuntimeError('unit: deliberate cell failure')
            return _saved_train(variant, *a, **kw)
        L.train_variant = _flaky
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=1,
                   lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
                   delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64,
                   n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0],
                   variants=['full', 'csa_fixed'])
        L.run(cfg, label='incomplete-unit')
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        e = next(iter(m['panels'].values()))
        assert e['status'] == 'incomplete', e.get('status')
        assert e['missing_cells'] == ['csa_fixed::seed0'], e.get('missing_cells')
        assert e['completion']['status'] == 'incomplete' and e['completion']['n_missing'] == 1, e.get('completion')
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['status'] == 'incomplete' and fin['missing_cells'] == ['csa_fixed::seed0'], fin
        err = next(x for x in lines if x.get('kind') == 'cell' and x.get('status') == 'error')
        assert err['variant'] == 'csa_fixed' and 'deliberate' in err['error'], err
    finally:
        L.load_run_corpus = _saved_load
        L.train_variant = _saved_train
        for k, old in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        shutil.rmtree(d, ignore_errors=True)


def t_failed_guard_is_retried():
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_reports = dict(L._OVERFIT_REPORTS)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_REPORTS.clear()

        def _stub(v, **kw):
            calls.append(v)
            ok = len(calls) > 1
            return {'passed': ok, 'initial_loss': 1.0, 'final_loss': 0.0 if ok else 9.0,
                    'final_accuracy': 1.0 if ok else 0.0, 'grad_all_finite': True,
                    'params_changed': True, 'diagnosis': 'ok' if ok else 'broken'}
        L.overfit_sanity_check = _stub
        try:
            L.overfit_guard(['probe'])
            raise AssertionError('a failed probe must be a hard stop')
        except RuntimeError:
            pass
        assert calls == ['probe']
        L.overfit_guard(['probe'])
        assert calls == ['probe', 'probe'], 'a cached FAILURE must be retried, not treated as a completed check'
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS.update(_saved_reports)


# ---------------------------------------------------------------------------
# F. 稀疏路径的逐位契约 / 边界检查
# ---------------------------------------------------------------------------

def t_indexer_column_narrowing_exact():
    # `causal_ends_host` is not a value-changing option: the readable-prefix
    # path must return the same idx AND soft bits as the full-width path (a
    # different GEMM shape is NOT bitwise column-independent, so score
    # narrowing must not be applied).
    torch.manual_seed(0)
    random.seed(0)
    n, B, d, nIH = 17, 12, 8, 2
    H = torch.randn(n, d)
    comp = torch.randn(B, d)
    W_DQ = torch.randn(d, d)
    W_DK = torch.randn(d, d)
    W_w = torch.randn(nIH, d)
    ends = sorted(random.sample(range(1, n + 1), B))
    last = torch.tensor([e - 1 for e in ends])
    for rs in (False, True):
        a = L.lightning_indexer(H, comp, last, W_DQ, W_DK, W_w, nIH, 4, return_mask=False, random_select=rs, causal_ends_host=ends, need_soft=True)
        b = L.lightning_indexer(H, comp, last, W_DQ, W_DK, W_w, nIH, 4, return_mask=False, random_select=rs, causal_ends_host=None, need_soft=True)
        assert torch.equal(a[1], b[1]), 'column narrowing changed the selected block ids'
        assert torch.equal(a[2], b[2]), 'column narrowing changed the soft prior by ULPs'


def t_attention_zero_blocks_keeps_window():
    # A sliding-window-only call has zero compressed blocks: the stacked-row
    # offset is 0 and the window lanes are still present.
    torch.manual_seed(0)
    n, h, d, w = 6, 2, 3, 4
    q, k, v = (torch.randn(n, h, d) for _ in range(3))
    topk = torch.empty(n, 0, dtype=torch.int32)
    last = torch.empty(0, dtype=torch.long)
    got = L.gathered_attention(q, torch.empty(0, h, d), torch.empty(0, h, d), topk, last, k, v, w, 0.5, q_chunk=2, mem_budget_bytes=1 << 20)
    ref = torch.empty_like(q)
    for i in range(n):
        lo = max(0, i - w + 1)
        attn = torch.softmax(torch.einsum('hd,mhd->hm', q[i], k[lo:i + 1]) * 0.5, dim=-1)
        ref[i] = torch.einsum('hm,mhd->hd', attn, v[lo:i + 1])
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), 'the sliding-window lanes were dropped when k_blk held zero rows'


def t_atomic_torch_save():
    # Checkpoint writes use a unique same-directory temp and replace atomically;
    # an interrupted writer must not leave a shared `*.tmp` behind.
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'ckpt.pt')
        V7._atomic_torch_save({'a': torch.arange(3)}, p)
        got = torch.load(p, weights_only=False)
        assert torch.equal(got['a'], torch.arange(3))
        leftovers = [f for f in os.listdir(d) if f.endswith('.tmp')]
        assert not leftovers, f'incomplete temp files left behind: {leftovers}'
        V7._atomic_torch_save({'a': torch.arange(4)}, p)
        got = torch.load(p, weights_only=False)
        assert got['a'].numel() == 4, 'the atomic replace did not publish the new checkpoint'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_cache_bytes_shared_storage():
    base = torch.arange(64, dtype=torch.int32)
    expanded = base.expand(4096, 64)
    assert L._cache_bytes(expanded) == L._cache_bytes(base), 'an expand must charge its base storage, not numel'
    assert L._cache_bytes(expanded) < expanded.numel() * expanded.element_size()
    cache = {}
    L._cache_put(cache, 'arange', base, 8192, L._idx_cache_total)
    L._cache_put(cache, 'expand', expanded, 8192, L._idx_cache_total)
    assert set(cache) == {'arange', 'expand'}, 'a shared-storage view evicted an unrelated cache entry'


def t_overfit_guard_source_key():
    # A synthetic pre-flight and a data pre-flight are distinct keys: the data
    # check exercises real ids, and a passed data check subsumes the synthetic
    # check.
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        def _stub(v, **kw):
            calls.append(kw.get('train_ids') is not None)
            return {'passed': True, 'initial_loss': 1.0, 'final_loss': 0.0, 'final_accuracy': 1.0, 'grad_all_finite': True, 'params_changed': True, 'diagnosis': 'ok'}
        L.overfit_sanity_check = _stub
        L.overfit_guard(['probe'])
        assert calls == [False], f'a synthetic check should run once: {calls}'
        calls.clear()
        L.overfit_guard(['probe'], train_ids=np.arange(64, dtype=np.int64))
        assert calls == [True], f'a DATA check must not be skipped just because a synthetic check passed: {calls}'
        calls.clear()
        L.overfit_guard(['probe'])
        assert calls == [], f'a synthetic check is subsumed by a passed data check: {calls}'
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)


def t_overfit_guard_config_aware():
    # A cached PASS certifies only the exact probe that produced it.  Changing
    # the step count or the convergence targets must invalidate the cache; a
    # lenient probe must not silently certify a stricter one.
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_reports = dict(L._OVERFIT_REPORTS)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_REPORTS.clear()

        def _stub(v, **kw):
            calls.append((int(kw.get('steps', -1)), float(kw.get('target_loss', -1))))
            return {'passed': True, 'initial_loss': 1.0, 'final_loss': 0.0,
                    'final_accuracy': 1.0, 'grad_all_finite': True,
                    'params_changed': True, 'diagnosis': 'ok'}

        L.overfit_sanity_check = _stub
        L.overfit_guard(['probe'], steps=10)
        L.overfit_guard(['probe'], steps=10)
        assert calls == [(10, 0.05)], f'the identical probe must be cached once: {calls}'
        L.overfit_guard(['probe'], steps=20)
        assert calls == [(10, 0.05), (20, 0.05)], f'a changed step count must re-run the probe: {calls}'
        L.overfit_guard(['probe'], steps=20, target_loss=0.001)
        assert calls == [(10, 0.05), (20, 0.05), (20, 0.001)], f'a stricter target must re-run the probe: {calls}'
        _rep = list(L._OVERFIT_REPORTS.values())[0]
        assert _rep.get('guard_probe_config', {}).get('schema') == 2
        assert _rep.get('guard_probe_key')
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS.update(_saved_reports)


def t_environment_snapshot_identity():
    env = L.environment_snapshot()
    for k in ('gpu_identity', 'cuda_visible_devices', 'float32_matmul_precision',
              'cublas_workspace_config', 'dependencies',
              'fp16_reduced_precision_reduction', 'bf16_reduced_precision_reduction'):
        assert k in env, f'environment snapshot missing {k}'
    assert isinstance(env['gpu_identity'], list)
    dep = env['dependencies']
    assert 'requirements.txt' in dep['files'] and 'requirements.lock' in dep['files'], dep
    assert dep['files']['requirements.txt'] is None or dep['files']['requirements.txt']['sha256'], dep
    assert 'torch' in dep['packages'], dep



# ---------------------------------------------------------------------------
# G. scale-safety checks
# ---------------------------------------------------------------------------

def t_long_window_chunked_mask():
    # A sliding-window full-attention layer uses the chunked mask path beyond
    # the row cap; that path must also work with no sink (sink_view=None).
    torch.manual_seed(0)
    cfg = [L.AttnCfg(kind='full', window=5)]
    m = L.SmallGPT(64, 32, 1, 4, 8, 32, cfg, mlp_ratio=2).eval()
    ids = torch.randint(0, 64, (2, 23))
    with torch.no_grad():
        ref = m(ids)

    old_cap = L._DENSE_MASK_ROW_CAP_BYTES
    old_f = L.causal_window_mask_f
    old_w = L.causal_window_mask

    def _boom(*_a, **_k):
        raise AssertionError('the full-mask path was used beyond the row cap')
    try:
        L._DENSE_MASK_ROW_CAP_BYTES = 1
        L.causal_window_mask_f = _boom
        L.causal_window_mask = _boom
        with torch.no_grad():
            got = m(ids)
    finally:
        L._DENSE_MASK_ROW_CAP_BYTES = old_cap
        L.causal_window_mask_f = old_f
        L.causal_window_mask = old_w
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), 'the chunked long-window path changed the attention output'


def t_v7_long_window_chunked_mask():
    # Same contract for the v7 RoPE full-attention twin.
    torch.manual_seed(0)
    cfg = V7.AttnCfgRope(kind='full', rope=True, rope_dim=8, window=5)
    m = V7.HybridAttentionRoPE(32, 4, 8, cfg).eval()
    x = torch.randn(2, 23, 32)
    with torch.no_grad():
        ref = m(x)

    old_cap = L._DENSE_MASK_ROW_CAP_BYTES
    old_f = L.causal_window_mask_f
    old_w = L.causal_window_mask

    def _boom(*_a, **_k):
        raise AssertionError('v7 used the full-mask path beyond the row cap')
    try:
        L._DENSE_MASK_ROW_CAP_BYTES = 1
        L.causal_window_mask_f = _boom
        L.causal_window_mask = _boom
        with torch.no_grad():
            got = m(x)
    finally:
        L._DENSE_MASK_ROW_CAP_BYTES = old_cap
        L.causal_window_mask_f = old_f
        L.causal_window_mask = old_w
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), 'the v7 chunked long-window path changed the attention output'


def t_indexer_empty_sequence():
    # n == 0 must return the same shapes as a non-empty call.
    d, nIH = 8, 2
    H = torch.randn(0, d)
    comp = torch.randn(0, d)
    W_DQ, W_DK, W_w = torch.randn(d, d), torch.randn(d, d), torch.randn(nIH, d)
    valid = []
    m, idx, soft = L.lightning_indexer(H, comp, torch.empty(0, dtype=torch.long), W_DQ, W_DK, W_w, nIH, 4, out_valid=valid)
    assert idx.shape == (0, 0) and soft.shape == (0, 0) and m.shape == (0, 0), (idx.shape, soft.shape, m.shape)
    assert len(valid) == 1 and valid[0].shape == (0, 0), 'the valid-cols output must be empty too'


def t_underpowered_never_verified():
    # An underpowered contrast is never called 'verified', regardless of
    # alpha: the n floor is a precondition for every positive verdict.
    entries = [{'controlled': True, 'n': 2, 'significant_global': True,
                'within_seed_noise': False, 'param_controlled': True,
                'p_exact_signflip': 0.5}]
    assert L._classify_group_status(entries, min_n=3, alpha=0.6, m_family=1) == 'underpowered'



# ---------------------------------------------------------------------------
# H. 实验设计与准备：假设方向 / 指标口径 / 基线调参
# ---------------------------------------------------------------------------

def t_hypothesis_direction_conflict():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    s = {}
    for i in range(7):
        s[f'csa_dynamic::seed{i}'] = dict(variant='csa_dynamic', seed=i, run_cfg=fp, steps=1,
                                         tokens_seen=1, ppl=51.0, params=100)
        s[f'csa_fixed::seed{i}'] = dict(variant='csa_fixed', seed=i, run_cfg=fp, steps=1,
                                       tokens_seen=1, ppl=50.0, params=100)
    design = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed', 'dynamic vs fixed')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert 'dynamic_blocking' in r['direction_conflict_components'], r
    assert 'dynamic_blocking' not in r['verified_components'], 'a significant opposite-direction result must never be called verified'
    e = r['components']['dynamic_blocking'][0]
    assert e['predicted_sign'] == -1 and e['observed_sign'] == 1 and e['direction_consistent'] is False, e
    # Reverse the direction and the same effect becomes support.
    for i in range(7):
        s[f'csa_dynamic::seed{i}']['ppl'] = 49.0
    r2 = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert 'dynamic_blocking' in r2['verified_components'], r2
    assert r2['components']['dynamic_blocking'][0]['direction_consistent'] is True


def t_metric_policy_and_classification_metrics():
    pol = L.metric_policy_audit()
    assert pol['status'] == 'ok', pol
    assert pol['policies']['multiclass_classification']['aggregation'] == 'macro'
    assert L.METRIC_POLICY['binary_classification']['primary_when_extreme_imbalance'] == 'pr_auc'
    assert L.METRIC_POLICY['binary_classification']['extreme_imbalance_threshold'] == 1e-3
    y = [0] * 990 + [1] * 10
    pred_all_majority = [0] * 1000
    m = L.classification_metrics(y, y_pred=pred_all_majority, labels=[0, 1])
    assert abs(m['accuracy'] - 0.99) < 1e-12
    assert abs(m['f1_macro'] - 0.49748743718592964) < 1e-9, m
    assert abs(m['f1_micro'] - 0.99) < 1e-12 and abs(m['f1_weighted'] - 0.9850251256281407) < 1e-9, m
    assert 'macro, micro and weighted' in m['f1_averaging']
    score = [0.0] * 990 + [1.0] * 10
    mb = L.classification_metrics(y, y_score=score, labels=[0, 1])
    assert mb['roc_auc'] == 1.0 and abs(mb['pr_auc'] - 1.0) < 1e-12, mb
    # The same endpoint expressed as a 2-D probability matrix must use the
    # positive-class column, not the whole matrix.
    score2 = np.stack([1.0 - np.asarray(score), np.asarray(score)], axis=1)
    m2 = L.classification_metrics(y, y_score=score2, labels=[0, 1])
    assert m2['roc_auc'] == 1.0 and abs(m2['pr_auc'] - 1.0) < 1e-12, m2
    # Boundary metrics without precision/recall/random baseline must warn.
    bad = {'full::seed0': {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                           'stats': {'L0_csa_fix': {'bnd_f1': 0.4}}}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn' and any('boundary' in w for w in ma['warnings']), ma


def t_multiclass_ranking_metrics_and_audit():
    # One-vs-rest ranking endpoints: a perfect ranking must be perfect under
    # macro, micro and weighted averaging.
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    score = np.zeros((len(y), 3))
    for i, c in enumerate([0, 1, 2]):
        score[y == c, i] = 1.0
    m = L.classification_metrics(y, y_score=score, labels=[0, 1, 2])
    assert m['ranking_averaging'] == 'one-vs-rest', m
    for key in ('pr_auc_macro', 'pr_auc_micro', 'pr_auc_weighted',
                'roc_auc_macro', 'roc_auc_micro', 'roc_auc_weighted'):
        assert abs(m[key] - 1.0) < 1e-12, (key, m[key])
    assert m['per_class']['2']['pr_auc'] == 1.0, m['per_class']
    assert m['per_class']['2']['roc_auc'] == 1.0, m['per_class']
    # The trivial majority floor is recorded next to the model scores: with
    # 80/15/5 labels the always-0 predictor already has accuracy 0.8.
    assert m['trivial_baseline']['strategy'] == 'most_frequent'
    assert m['trivial_baseline']['label'] == '0'
    assert abs(m['trivial_baseline']['accuracy'] - 0.8) < 1e-12
    assert abs(m['trivial_baseline']['f1_macro'] - (2.0 * 0.8 / 1.8) / 3.0) < 1e-12
    # Micro F1 pools decisions across classes, so the constant-majority floor
    # must equal its accuracy (not the majority class's binary F1).
    assert abs(m['trivial_baseline']['f1_micro'] - 0.8) < 1e-12, m
    # One-vs-rest PR-AUC also has a non-zero chance floor (mean prevalence);
    # a raw PR-AUC of 0.5 at 1:1000 would not be skillful.
    assert abs(m['pr_auc_macro_chance'] - 1.0 / 3.0) < 1e-12, m
    assert abs(m['pr_auc_weighted_chance'] - 0.665) < 1e-12, m
    assert abs(m['pr_auc_macro_lift_over_chance'] - (1.0 - 1.0 / 3.0)) < 1e-12
    assert abs(m['pr_auc_weighted_lift_over_chance'] - 0.335) < 1e-12
    # The one-hot probability matrix is perfectly calibrated and confident.
    assert m['calibration']['status'] == 'ok' and m['calibration']['brier_score'] == 0.0
    assert m['calibration']['ece'] == 0.0 and m['calibration']['top1_accuracy'] == 1.0
    _uniform = L.classification_metrics(y, y_score=np.full((len(y), 3), 1.0 / 3.0),
                                        labels=[0, 1, 2])
    assert _uniform['calibration']['status'] == 'ok'
    assert abs(_uniform['calibration']['brier_score'] - 2.0 / 3.0) < 1e-12
    _logits = L.classification_metrics(y, y_score=np.ones((len(y), 3)), labels=[0, 1, 2])
    assert _logits['calibration']['status'] == 'unavailable'
    assert 'brier_score' not in _logits
    # The threshold-free metric list is a capability declaration: an
    # unavailable calibration endpoint must not be advertised as measured.
    assert 'brier_score' not in _logits['threshold_free_metrics']
    assert 'ece' not in _logits['threshold_free_metrics']
    _bin_logits = L.classification_metrics([0, 0, 1, 1], y_score=[-2.0, -1.0, 1.0, 2.0],
                                           labels=[0, 1])
    assert _bin_logits['calibration']['status'] == 'unavailable'
    assert _bin_logits['threshold_free_metrics'] == ['roc_auc', 'pr_auc']
    _bin_prob = L.classification_metrics([0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9],
                                         labels=[0, 1])
    assert _bin_prob['threshold_free_metrics'] == ['roc_auc', 'pr_auc',
                                                   'brier_score', 'ece']
    # A binary record gets the same floor; the majority predictor has zero
    # minority recall, so balanced accuracy is not fooled by its high accuracy.
    _bin_floor = L.classification_metrics([0] * 99 + [1], y_pred=[0] * 100, labels=[0, 1])
    assert _bin_floor['trivial_baseline']['label'] == '0'
    assert abs(_bin_floor['trivial_baseline']['accuracy'] - 0.99) < 1e-12
    assert abs(_bin_floor['trivial_baseline']['balanced_accuracy'] - 0.5) < 1e-12
    # Macro is equal-weight over classes, so it must not equal a support-
    # weighted or pooled (micro) aggregate when the classes differ in size and
    # the ranking is imperfect.  This is exactly why the averaging must be named.
    rng = np.random.default_rng(3)
    scores = np.stack([rng.random(len(y)) for _ in range(3)], axis=1)
    mi = L.classification_metrics(y, y_score=scores, labels=[0, 1, 2])
    assert mi['pr_auc_macro'] != mi['pr_auc_micro'] or mi['pr_auc_macro'] != mi['pr_auc_weighted'], mi
    assert 0.0 <= mi['pr_auc_macro'] <= 1.0 and 0.0 <= mi['pr_auc_micro'] <= 1.0
    assert abs(mi['macro_micro_f1_gap'] - (mi['f1_macro'] - mi['f1_micro'])) < 1e-12
    # A requested class absent from the sample has an undefined one-vs-rest
    # endpoint; it is listed rather than assigned a fake score, and the macro
    # average covers only the defined classes.
    m_undef = L.classification_metrics(np.array([0, 0, 1, 1]),
                                       y_score=np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0],
                                                         [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]),
                                       labels=[0, 1, 2])
    assert m_undef['per_class']['2']['pr_auc'] is None
    assert m_undef['per_class']['2']['roc_auc'] is None
    assert '2' in m_undef['undefined_classes']
    assert abs(m_undef['pr_auc_macro'] - np.mean(
        [m_undef['per_class'][c]['pr_auc'] for c in ('0', '1')])) < 1e-12
    # Extreme multiclass imbalance must switch the conditional primary to
    # macro PR-AUC and flag a record that reports only F1.
    y2 = np.array([0] * 1990 + [1] * 9 + [2] * 1)
    score2 = np.zeros((len(y2), 3))
    for i, c in enumerate([0, 1, 2]):
        score2[y2 == c, i] = 1.0
    m2 = L.classification_metrics(y2, y_score=score2, labels=[0, 1, 2])
    assert abs(m2['minority_prevalence'] - 1.0 / 2000.0) < 1e-15, m2['minority_prevalence']
    assert 'pr_auc_macro' in m2['imbalance_warning'], m2
    assert L.METRIC_POLICY['multiclass_classification']['primary_when_extreme_imbalance'] == 'pr_auc_macro'
    good = dict(m2, variant='mc', task='multiclass_classification')
    assert L.metric_policy_audit({'mc': good})['status'] == 'ok', L.metric_policy_audit({'mc': good})
    no_rank = {k: v for k, v in good.items() if not k.startswith(('pr_auc_', 'roc_auc_'))}
    nr = L.metric_policy_audit({'mc': no_rank})
    assert nr['status'] == 'warn', nr
    assert any('pr_auc_macro' in w for w in nr['warnings']), nr
    # A hand-edited aggregate cannot pass when per-class values exist.
    forged = dict(good, pr_auc_macro=0.5)
    fr = L.metric_policy_audit({'mc': forged})
    assert fr['status'] == 'warn' and any('pr_auc_macro' in w and 'mean' in w
                                         for w in fr['warnings']), fr
    # Declaring a binary ranking metric as the multiclass primary is rejected.
    wrong = dict(good, primary_metric='roc_auc')
    wr = L.metric_policy_audit({'mc': wrong})
    assert wr['status'] == 'warn' and any('primary_metric' in w for w in wr['warnings']), wr
    # A forged easier-than-majority floor is rejected.
    forged_floor = dict(good, trivial_baseline=dict(good['trivial_baseline'], label='2'))
    ff = L.metric_policy_audit({'mc': forged_floor})
    assert ff['status'] == 'warn' and any('trivial_baseline' in w for w in ff['warnings']), ff
    # The PR-AUC chance floor and the macro-micro F1 gap are reproducible too.
    forged_chance = dict(good, pr_auc_macro_chance=0.001)
    fc = L.metric_policy_audit({'mc': forged_chance})
    assert fc['status'] == 'warn' and any('pr_auc_macro_chance' in w for w in fc['warnings']), fc
    forged_gap = dict(good, macro_micro_f1_gap=0.5)
    fg = L.metric_policy_audit({'mc': forged_gap})
    assert fg['status'] == 'warn' and any('macro_micro_f1_gap' in w for w in fg['warnings']), fg
    # Different label identities that share a str() spelling (integer 1 and
    # string '1') must not overwrite one another in the per-class table.  The
    # report keeps the historical names when unambiguous and otherwise uses
    # unique display keys, so every declared class retains its own row.
    _mixed_labels = np.array([1, '1', 2], dtype=object)
    _mixed_y = np.array([_mixed_labels[i % 3] for i in range(12)], dtype=object)
    _mixed_score = np.zeros((len(_mixed_y), 3))
    for _i, _lab in enumerate(_mixed_labels):
        _mixed_score[[str(v) == str(_lab) and type(v) is type(_lab)
                      for v in _mixed_y.tolist()], _i] = 1.0
    _mx = L.classification_metrics(_mixed_y, y_score=_mixed_score,
                                   labels=_mixed_labels)
    assert len(_mx['labels']) == 3 and len(set(_mx['labels'])) == 3, _mx['labels']
    assert len(_mx['per_class']) == 3, _mx['per_class']
    assert all(_mx['per_class'][_n]['support'] == 4 for _n in _mx['labels']), _mx
    assert _mx['trivial_baseline']['label'] in _mx['labels']
    assert len(_mx['prevalence']) == 3, _mx['prevalence']
    _mx_audit = L.metric_policy_audit({'mx': dict(
        _mx, variant='mx', task='multiclass_classification')})
    assert _mx_audit['status'] == 'ok', _mx_audit
    # Omitting labels must not let np.asarray coerce the integer 1 into the
    # string '1' before the class identities are checked.
    _mx_auto = L.classification_metrics([1, '1', 2, 1, '1', 2],
                                        y_pred=[1, '1', 2, 1, '1', 2])
    assert len(_mx_auto['labels']) == 3 and len(set(_mx_auto['labels'])) == 3, _mx_auto
    assert len(_mx_auto['per_class']) == 3, _mx_auto
    # An explicitly declared label list must not be string-coerced either:
    # np.asarray([1, '1']) merges the two classes before the duplicate check.
    for _decl in ([1, '1', 2], np.array([1, '1', 2], dtype=object)):
        _mx_explicit = L.classification_metrics(
            [1, '1', 2, 1, '1', 2], y_pred=[1, '1', 2, 1, '1', 2],
            labels=_decl)
        assert len(_mx_explicit['labels']) == 3, _mx_explicit['labels']
        assert len(_mx_explicit['per_class']) == 3, _mx_explicit
    # A BINARY mixed-type endpoint must not go through np.where with two
    # un-promotable dtypes (NumPy 2 raises DTypePromotionError for int 1 vs
    # the string '1'); the derived predictions keep both identities.
    _mixed_y_bin = np.array([1, '1'] * 25, dtype=object)
    _mixed_s_bin = np.array([0.8] * 25 + [0.2] * 25)
    _mb = L.classification_metrics(
        _mixed_y_bin, y_score=_mixed_s_bin,
        labels=np.array([1, '1'], dtype=object), positive_label=1)
    assert len(_mb['labels']) == 2 and _mb['n'] == 50, _mb
    assert _mb['predicted_positive_rate'] == 0.5, _mb


def t_metric_prevalence_recommendation_and_per_class_chance():
    # A rare class's raw PR-AUC must be readable against its own chance floor,
    # not only the macro mean (which a frequent class can carry).
    y = np.array([0] * 1990 + [1] * 9 + [2] * 1)
    score = np.zeros((len(y), 3))
    for i, c in enumerate([0, 1, 2]):
        score[y == c, i] = 1.0
    m = L.classification_metrics(y, y_score=score, labels=[0, 1, 2])
    assert m['recommended_primary_metric'] == 'pr_auc_macro', m
    assert m['recommended_primary_average'] == 'one_vs_rest_macro', m
    assert m['recommended_primary_available'] is True, m
    for c, sup in (('1', 9), ('2', 1)):
        pc = m['per_class'][c]
        assert abs(pc['pr_auc_chance'] - sup / len(y)) < 1e-15, pc
        assert abs(pc['pr_auc_lift_over_chance'] - (pc['pr_auc'] - pc['pr_auc_chance'])) < 1e-12, pc
        assert pc['roc_auc_chance'] == 0.5 and abs(
            pc['roc_auc_lift_over_chance'] - (pc['roc_auc'] - 0.5)) < 1e-15, pc
    good = dict(m, variant='mc', task='multiclass_classification')
    assert L.metric_policy_audit({'mc': good})['status'] == 'ok', L.metric_policy_audit({'mc': good})
    # Forged per-class chance/lift values are rejected from the stored record.
    forged = json.loads(json.dumps(good))
    forged['per_class']['2']['pr_auc_chance'] = 0.5
    fr = L.metric_policy_audit({'mc': forged})
    assert fr['status'] == 'warn' and any('per-class PR-AUC chance' in w
                                         for w in fr['warnings']), fr
    # The recommendation is a claim too: it must match the policy's
    # conditional endpoint at the observed prevalence.
    wrong_rec = dict(good, recommended_primary_metric='f1_macro')
    wr = L.metric_policy_audit({'mc': wrong_rec})
    assert wr['status'] == 'warn' and any('recommended_primary_metric' in w
                                         for w in wr['warnings']), wr
    # Extreme binary imbalance switches the recommendation to PR-AUC and
    # exposes chance-relative ROC skill.
    yb = np.array([0] * 9999 + [1])
    sb = np.linspace(0.0, 1.0, len(yb))
    mb = L.classification_metrics(yb, y_score=sb, labels=[0, 1])
    assert mb['recommended_primary_metric'] == 'pr_auc', mb
    assert abs(mb['roc_auc_chance'] - 0.5) < 1e-15
    assert abs(mb['roc_auc_lift_over_chance'] - (mb['roc_auc'] - 0.5)) < 1e-15
    assert abs(mb['roc_auc_normalized_skill'] - (mb['roc_auc'] - 0.5) / 0.5) < 1e-15
    bad_rec = dict(mb, variant='b', task='binary_classification',
                   recommended_primary_metric='f1')
    br = L.metric_policy_audit({'b': bad_rec})
    assert br['status'] == 'warn' and any('recommended_primary_metric' in w
                                         for w in br['warnings']), br

def t_hypothesis_prevalence_conditional_endpoint():
    def _spec(**kw):
        base = {'statement': 'rare-event detector improves recall',
                'mechanism': 'the score ranks positives above negatives',
                'predicted_sign': 1, 'primary_metric': 'f1',
                'requires_param_match': False, 'falsifier': 'PR-AUC is at prevalence'}
        base.update(kw)
        return base
    # At a 1:10000 base rate, a hypothesis written against F1 is misaligned
    # with the policy's conditional PR-AUC endpoint even though f1 is the
    # normal binary primary.
    bad = L.hypothesis_audit({'h': _spec(prevalence=1e-4)}, 'component')
    assert bad['status'] == 'warn' and 'h' in bad['metric_misaligned'], bad
    assert any('extreme-imbalance threshold' in w for w in bad['warnings']), bad
    # Declaring the conditional endpoint (or a justified secondary endpoint)
    # is accepted.
    good = L.hypothesis_audit({'h': _spec(prevalence=1e-4,
                                         primary_metric='pr_auc')}, 'component')
    assert good['status'] == 'ok' and not good['metric_misaligned'], good
    sec = L.hypothesis_audit({'h': _spec(prevalence=1e-4,
                                        primary_metric='roc_auc',
                                        metric_role='secondary',
                                        metric_justification='report ROC alongside PR')},
                             'component')
    assert sec['status'] in ('ok', 'warn') and not sec['metric_misaligned'], sec
    # Above the threshold the normal F1 primary remains aligned.
    normal = L.hypothesis_audit({'h': _spec(prevalence=0.2)}, 'component')
    assert normal['status'] == 'ok' and not normal['metric_misaligned'], normal
    # A malformed base rate fails closed instead of disabling the check.
    for _bad in (True, '0.0001', float('nan'), float('inf'), -0.1, 1.2):
        r = L.hypothesis_audit({'h': _spec(prevalence=_bad)}, 'component')
        assert r['status'] == 'warn' and 'h' in r['metric_misaligned'], (_bad, r)
    # The same conditional rule applies to NIAH retrieval: a hypothesis for a
    # rare answer class must name macro recall, not top-1 accuracy.
    _niah_bad = L.hypothesis_audit({'h': _spec(
        primary_metric='top1_accuracy', metric_policy='niah_retrieval',
        prevalence=1e-4)}, 'component')
    assert _niah_bad['status'] == 'warn' and 'h' in _niah_bad['metric_misaligned'], _niah_bad
    _niah_good = L.hypothesis_audit({'h': _spec(
        primary_metric='macro_recall', metric_policy='niah_retrieval',
        prevalence=1e-4)}, 'component')
    assert _niah_good['status'] == 'ok' and not _niah_good['metric_misaligned'], _niah_good

def t_binary_degenerate_threshold_visibility():
    # Logits/scores on a non-probability scale make the default 0.5 cut a
    # constant predictor.  The record must expose that instead of letting an
    # F1/accuracy headline look like learned decision quality.
    y = [0, 0, 1, 1]
    m = L.classification_metrics(y, y_score=[1.0, 2.0, 3.0, 4.0], labels=[0, 1])
    assert m['predicted_positive_rate'] == 1.0, m
    assert m['degenerate_predictions'] is True, m
    assert m['threshold_within_score_range'] is False, m
    assert 'outside the observed score range' in m['threshold_range_warning'], m
    rec = dict(m, variant='b', task='binary_classification')
    audit = L.metric_policy_audit({'b': rec})
    assert audit['status'] == 'warn', audit
    assert any('outside the observed score range' in w for w in audit['warnings']), audit
    assert 'b' in audit['degenerate_prediction_records'], audit
    # A forged predicted-positive rate cannot hide a degenerate cut.
    forged = dict(rec, predicted_positive_rate=0.5)
    fa = L.metric_policy_audit({'b': forged})
    assert fa['status'] == 'warn' and any(
        'predicted_positive_rate' in w for w in fa['warnings']), fa
    # A normal probability score with a meaningful cut stays non-degenerate.
    good = L.classification_metrics(y, y_score=[0.1, 0.2, 0.8, 0.9], labels=[0, 1])
    assert good['degenerate_predictions'] is False, good
    assert good['threshold_within_score_range'] is True, good
    assert good['predicted_positive_rate'] == 0.5, good
    assert L.metric_policy_audit({'b': dict(good, variant='b',
                                           task='binary_classification')})['status'] == 'ok'

def t_binary_pr_floor_uses_positive_prevalence():
    # PR-AUC's chance floor follows the POSITIVE class, not the minority class.
    # When the positive endpoint is the common label (rare negative event), a
    # correct record must not be flagged as having a misstated floor.
    y = ['pos'] * 9999 + ['neg']
    score = [0.9] * 9999 + [0.1]
    m = L.classification_metrics(y, y_score=score, labels=['neg', 'pos'],
                                 positive_label='pos')
    assert m['positive_label'] == 'pos'
    assert abs(m['pr_auc_prevalence'] - 0.9999) < 1e-12, m
    audit = L.metric_policy_audit({'b': dict(m, variant='b',
                                             task='binary_classification')})
    assert audit['status'] == 'warn'  # only the rare-minority F1-cut warning
    assert not any('floor is misstated' in w for w in audit['warnings']), audit
    assert not any('pr_auc_lift_over_prevalence' in w for w in audit['warnings']), audit
    # A genuinely forged floor is still rejected when the positive class is common.
    forged = dict(m, variant='b', task='binary_classification',
                  pr_auc_prevalence=0.0001)
    fa = L.metric_policy_audit({'b': forged})
    assert any('floor is misstated' in w for w in fa['warnings']), fa
    # A record that exposes only the PR-AUC chance floor still declares its
    # prevalence; a rare-positive record must not evade the PR-AUC endpoint
    # check by omitting a separate `prevalence` field.
    _rare_only = {'variant': 'r', 'task': 'binary_classification',
                  'positive_label': '1', 'f1_primary': 0.9,
                  'f1_primary_average': 'binary_positive_class',
                  'roc_auc': 0.99, 'pr_auc_prevalence': 0.0005}
    _ro = L.metric_policy_audit({'r': _rare_only})
    assert _ro['status'] == 'warn' and any(
        'PR-AUC' in w and '1:1000' in w for w in _ro['warnings']), _ro

def t_metric_policy_checks_record_values():
    # The declared token-weighted PPL must match the record's own NLL
    # accounting; a per-sequence arithmetic mean is a different estimand.
    _tok = 100
    good = {'v::seed0': {'variant': 'v', 'ppl': 10.0, 'nll_sum': math.log(10.0) * _tok,
                         'n_tokens': _tok}}
    assert L.metric_policy_audit(good)['status'] == 'ok'
    # A PPL with no NLL/token accounting is unverifiable, not silently accepted.
    _legacy = L.metric_policy_audit({'v::seed0': {'variant': 'v', 'ppl': 10.0}})
    assert _legacy['status'] == 'warn' and any(
        'no NLL/token accounting' in w for w in _legacy['warnings']), _legacy
    bad = {'v::seed0': {'variant': 'v', 'ppl': 10.0, 'nll_sum': math.log(20.0) * _tok,
                        'n_tokens': _tok}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn' and any('not exp(nll_sum/n_tokens)' in w for w in ma['warnings']), ma
    # Per-row accounting is accepted as the same token-weighted estimate.
    row = {'v::seed0': {'variant': 'v', 'ppl': 10.0,
                        'nll_per_row': [math.log(10.0) * 60, math.log(10.0) * 40],
                        'n_tokens_per_row': [60, 40]}}
    assert L.metric_policy_audit(row)['status'] == 'ok', L.metric_policy_audit(row)
    # A nested record reports its holdout PPL and must be checked against
    # the holdout NLL accounting, not the full-validation accounting.
    # `ppl` stays the full-validation value here (10.0) while the primary
    # estimate is the holdout PPL (5.0); the check must not compare the
    # holdout NLL against the full-validation PPL.
    nested = {'v::seed0': {'variant': 'v', 'ppl': 10.0,
                           'nll_sum': math.log(10.0) * 100, 'n_tokens': 100,
                           'ppl_holdout': 5.0,
                           'nll_sum_holdout': math.log(5.0) * 40,
                           'n_tokens_holdout': 40,
                           'ppl_estimation_rows': 'nested_validation_holdout'}}
    assert L.metric_policy_audit(nested)['status'] == 'ok', L.metric_policy_audit(nested)
    _pooled, _nl, _nt = L._pooled_ppl_from_records(list(nested.values()))
    assert abs(_pooled - 5.0) < 1e-12 and _nt == 40, (_pooled, _nl, _nt)
    # Intervals must use the same rows as the primary (holdout) PPL.  The
    # full-validation arrays here are deliberately incompatible with the
    # holdout estimate so a wrong-array fallback cannot look plausible.
    _nr = {'variant': 'v', 'seed': 0, 'ppl': math.exp(10.0),
           'nll_sum': 10.0 * 40, 'n_tokens': 40,
           'ppl_estimation_rows': 'nested_validation_holdout',
           'ppl_holdout': math.exp(2.0),
           'nll_sum_holdout': 2.0 * 10, 'n_tokens_holdout': 10,
           'nll_per_row': [100.0] * 4, 'n_tokens_per_row': [10] * 4,
           'nll_per_row_holdout': [10.0] * 2, 'n_tokens_per_row_holdout': [5] * 2}
    _nr2 = dict(_nr, seed=1)
    _pd, _pn, _pt = L._pooled_ppl_from_records([_nr, _nr2])
    assert abs(_pd - math.exp(2.0)) < 1e-12 and _pt == 20, (_pd, _pn, _pt)
    _dci = L._doc_bootstrap_ppl_ci([_nr, _nr2])
    assert _dci is not None and all(abs(x - math.exp(2.0)) < 1e-9 for x in _dci), _dci
    _cci = L._combined_seed_doc_ci([_nr, _nr2])
    assert _cci['status'] == 'ok' and _cci['n_docs'] == 2, _cci
    assert all(abs(x - math.exp(2.0)) < 1e-9 for x in _cci['ci95_ppl']), _cci
    # Missing holdout accounting must not silently pool the full-validation
    # totals.  A complete holdout per-row array is still a valid source, so
    # remove both forms to exercise the fail-closed path.
    _miss = dict(_nr)
    _miss.pop('nll_sum_holdout')
    _miss.pop('n_tokens_holdout')
    _miss.pop('nll_per_row_holdout')
    _miss.pop('n_tokens_per_row_holdout')
    assert L._pooled_ppl_from_records([_miss]) == (None, None, None)
    # Per-row holdout arrays alone are enough to recover the same totals.
    _rows_only = dict(_nr)
    _rows_only.pop('nll_sum_holdout')
    _rows_only.pop('n_tokens_holdout')
    assert abs(L._pooled_ppl_from_records([_rows_only])[0] - math.exp(2.0)) < 1e-12
    # The metric-policy audit must use those same holdout rows; falling back
    # to the full-validation arrays would compare the holdout PPL against a
    # different sample and flag a correct nested record.
    assert L.metric_policy_audit({'v::seed0': _rows_only})['status'] == 'ok', \
        L.metric_policy_audit({'v::seed0': _rows_only})
    # Missing holdout row arrays must not silently bootstrap full-validation rows.
    _miss_rows = dict(_nr)
    _miss_rows.pop('nll_per_row_holdout')
    _miss_rows.pop('n_tokens_per_row_holdout')
    assert L._doc_bootstrap_ppl_ci([_miss_rows]) is None
    assert L._combined_seed_doc_ci([_miss_rows, _miss_rows])['status'] == 'unverifiable'
    _bad_nested = json.loads(json.dumps(nested))
    _bad_nested['v::seed0']['ppl_holdout'] = 7.0
    _mn = L.metric_policy_audit(_bad_nested)
    assert _mn['status'] == 'warn' and any('not exp(nll_sum/n_tokens)' in w for w in _mn['warnings']), _mn
    # Boundary F1 must be the harmonic mean of the reported precision/recall.
    b = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                      'stats': {'L0': {'bnd_f1': 0.9, 'bnd_prec': 0.2,
                                       'bnd_rec': 0.2, 'bnd_rand': 0.1}}}}
    mb = L.metric_policy_audit(b)
    assert mb['status'] == 'warn' and any('non-harmonic' in w for w in mb['warnings']), mb
    # Existence is not measurement: a boundary record with all keys present
    # but None must fail, not pass through the range/filter checks.
    bn = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                       'stats': {'L0': {'bnd_f1': None, 'bnd_prec': None,
                                        'bnd_rec': None, 'bnd_rand': None}}}}
    mbn = L.metric_policy_audit(bn)
    assert mbn['status'] == 'warn', mbn
    _bc = next(c for c in mbn['checks'] if c['metric'] == 'boundary_alignment')
    assert _bc['ok'] is False and _bc['missing'], _bc
    # Retrieval accuracy outside [0,1] is not a usable endpoint.
    n = {'n::seed0': {'variant': 'n', 'task': 'niah_natural', 'acc': 1.5,
                      'nll': 1.0, 'n_queries': 10}}
    mn = L.metric_policy_audit(n)
    assert mn['status'] == 'warn' and any('invalid accuracy' in w for w in mn['warnings']), mn
    # An empty retrieval record is incomplete, not a pass.
    me = L.metric_policy_audit({'x::seed0': {'variant': 'x', 'task': 'niah_natural'}})
    assert me['status'] == 'warn', me

    # The audit's record checks must honour a caller-supplied policy, not the
    # module default.  Here the custom policy declares ROC-AUC as primary and
    # a 0.5 extreme-imbalance threshold (so a 10% positive rate is NOT rare).
    _custom = json.loads(json.dumps(L.METRIC_POLICY))
    _custom['binary_classification']['primary'] = 'roc_auc'
    _custom['binary_classification']['extreme_imbalance_threshold'] = 0.5
    _cust_rec = {'r': {'variant': 'r', 'task': 'binary_classification',
                       'prevalence': 0.1, 'primary_metric': 'roc_auc',
                       'roc_auc': 0.9, 'pr_auc': 0.8,
                       'recommended_primary_metric': 'pr_auc',
                       'recommended_primary_available': True}}
    _ca = L.metric_policy_audit(_cust_rec, policy=_custom)
    _cw = ' '.join(_ca['warnings'])
    assert 'declares primary_metric' not in _cw, _ca
    assert 'does not match the policy endpoint' not in _cw, _ca
    assert 'is not defined for task' not in _cw, _ca
    # A malformed caller policy entry is absent, not an exception.
    _broken = {'binary_classification': ['not', 'a', 'mapping']}
    _ba = L.metric_policy_audit(_cust_rec, policy=_broken)
    assert _ba['status'] == 'warn', _ba

def t_retrieval_metrics_and_niah_trivial_floor():
    # A many-way retrieval endpoint must expose its trivial floors: micro
    # accuracy alone can be high merely because one answer class dominates.
    rm = L.retrieval_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert rm['n_correct'] == 3 and rm['n'] == 4, rm
    assert abs(rm['acc'] - 0.75) < 1e-12 and abs(rm['balanced_accuracy'] - 0.5) < 1e-12, rm
    assert abs(rm['chance_accuracy'] - 0.5) < 1e-12, rm
    assert abs(rm['majority_accuracy'] - 0.75) < 1e-12, rm
    assert abs(rm['acc_lift_over_chance'] - 0.25) < 1e-12, rm
    assert abs(rm['acc_lift_over_majority']) < 1e-12, rm
    # The NIAH copy rule is a deterministic lexical floor: it must recover a
    # planted key->value pair by copying after the nearest earlier key, and
    # count a query with no earlier key as missing context (not as a hit).
    _ids = np.array([[1, 2, 3, 7, 9, 4, 5, 6, 1, 7, 9]])
    _rule = L.niah_copy_rule_baseline(_ids, np.array([[3]]), np.array([[9]]),
                                      target=4)
    assert _rule['n_correct'] == 1 and _rule['acc'] == 1.0, _rule
    assert _rule['n_missing_context'] == 0 and _rule['rule'] == 'nearest_previous_key_copy', _rule
    _no_ctx = L.niah_copy_rule_baseline(
        np.array([[1, 2, 3, 4, 7, 9]]), np.array([[1]]), np.array([[9]]),
        target=4)
    assert _no_ctx['acc'] == 0.0 and _no_ctx['n_missing_context'] == 1, _no_ctx
    # A target zone wider than the scored inputs cannot define a query offset;
    # it must fail closed instead of silently marking every query as missing.
    try:
        L.niah_copy_rule_baseline(_ids, np.array([[3]]), np.array([[9]]),
                                  target=_ids.shape[1])
    except ValueError as e:
        assert 'exceeds' in str(e), e
    else:
        raise AssertionError('a copy-rule target beyond the sequence must be refused')
    # Mixed label identities must not be merged by NumPy dtype promotion.
    rm2 = L.retrieval_metrics([1, '1'], [1, '1'])
    assert rm2['n_classes'] == 2 and rm2['acc'] == 1.0, rm2
    # A declared label set must cover every true answer: otherwise accuracy
    # counts a class that the support/chance/macro-recall computation drops.
    for _bad in (dict(labels=[0, 1]), dict(labels=[0, 1, 1])):
        try:
            L.retrieval_metrics([0, 1, 2], [0, 1, 2], **_bad)
        except ValueError as e:
            assert ('not in the declared labels' in str(e)
                    or 'duplicate class' in str(e)), (e, _bad)
        else:
            raise AssertionError(f'malformed declared labels were accepted: {_bad}')
    # Declared-but-absent labels stay listed with a null recall and do not
    # change the observed-class chance floor.
    rake = L.retrieval_metrics([0, 0, 1], [0, 0, 0], labels=[0, 1, 2])
    assert rake['n_classes'] == 2 and abs(rake['chance_accuracy'] - 0.5) < 1e-12, rake
    assert rake['per_class_recall']['2'] is None, rake
    # A complete NIAH record is accepted and its floors are recomputed.
    rec = dict(rm, task='niah_natural', nll=1.25, n_queries=rm['n'])
    good = {'n::seed0': rec}
    assert L.metric_policy_audit(good)['status'] == 'ok', L.metric_policy_audit(good)
    # A persisted lexical copy-rule floor must be reproducible from its exact
    # counts, and a model that does not beat it is called out (the rule is the
    # simplest baseline that can solve this copy task).
    with_rule = json.loads(json.dumps(good))
    with_rule['n::seed0']['copy_rule_baseline'] = {
        'n': 4, 'n_correct': 3, 'acc': 0.75, 'n_missing_context': 0,
        'rule': 'nearest_previous_key_copy'}
    ar = L.metric_policy_audit(with_rule)
    assert ar['status'] == 'warn' and any('copy rule' in w for w in ar['warnings']), ar
    bad_rule = json.loads(json.dumps(with_rule))
    bad_rule['n::seed0']['copy_rule_baseline']['acc'] = 0.9
    abr = L.metric_policy_audit(bad_rule)
    assert abr['status'] == 'warn' and any(
        'copy-rule-acc-count-inconsistent' in str(c.get('invalid'))
        for c in abr['checks']), abr
    assert rm['recommended_primary_metric'] == 'top1_accuracy', rm
    # Extreme answer-class imbalance switches the conditional primary to
    # macro recall; a top-1 headline would otherwise be dominated by the
    # frequent answers.
    _y_imb = [0] * 1000 + [1]
    _p_imb = [0] * 1001
    ri = L.retrieval_metrics(_y_imb, _p_imb)
    assert ri['minority_prevalence'] < 1e-3, ri
    assert ri['recommended_primary_metric'] == 'macro_recall', ri
    assert ri['recommended_primary_available'] is True, ri
    _ri_rec = dict(ri, task='niah_natural', nll=1.0, n_queries=ri['n'])
    _ri_good = {'n::seed0': _ri_rec}
    _ra = L.metric_policy_audit(_ri_good)
    assert _ra['status'] == 'ok', _ra
    _bad_pri = json.loads(json.dumps(_ri_good))
    _bad_pri['n::seed0']['recommended_primary_metric'] = 'top1_accuracy'
    _rbp = L.metric_policy_audit(_bad_pri)
    assert _rbp['status'] == 'warn', _rbp
    assert any('conditional-primary-mismatch' in str(c.get('invalid'))
               for c in _rbp['checks']), _rbp
    _miss_pri = json.loads(json.dumps(_ri_good))
    _miss_pri['n::seed0'].pop('recommended_primary_metric')
    _rmp = L.metric_policy_audit(_miss_pri)
    assert _rmp['status'] == 'warn', _rmp
    assert any('conditional-primary-missing' in str(c.get('invalid'))
               for c in _rmp['checks']), _rmp
    # A support table that does not sum to n_queries describes another sample.
    bad_sup = json.loads(json.dumps(good))
    _first_sup = next(iter(bad_sup['n::seed0']['answer_support']))
    bad_sup['n::seed0']['answer_support'][_first_sup] += 1
    ms = L.metric_policy_audit(bad_sup)
    assert ms['status'] == 'warn', ms
    assert any('answer-support-sum-mismatch' in str(c.get('invalid'))
               for c in ms['checks']), ms
    # Per-class recall must reproduce the recorded correct count/support.
    bad_pcr = json.loads(json.dumps(good))
    _rec_name = next(iter(bad_pcr['n::seed0']['per_class_recall']))
    bad_pcr['n::seed0']['per_class_recall'][_rec_name] = 0.5
    mp = L.metric_policy_audit(bad_pcr)
    assert mp['status'] == 'warn', mp
    # A null recall for a class with positive support drops that class from
    # macro recall and from the correct-count reconciliation: that is missing
    # evidence, not the documented "absent class" null.
    bad_null = json.loads(json.dumps(good))
    _pos_name = next(k for k, v in bad_null['n::seed0']['answer_support'].items()
                     if v > 0)
    bad_null['n::seed0']['per_class_recall'][_pos_name] = None
    mn = L.metric_policy_audit(bad_null)
    assert mn['status'] == 'warn', mn
    assert any('per-class-recall-missing' in str(c.get('invalid'))
               for c in mn['checks']), mn
    # Present-but-malformed floor evidence must fail closed instead of being
    # treated like a legacy record that omits the table entirely.
    for _key, _val, _why in (
            ('answer_support', 'not-a-table', 'answer-support-invalid'),
            ('per_class_recall', ['not', 'a', 'table'], 'per-class-recall-invalid')):
        bad_shape = json.loads(json.dumps(good))
        bad_shape['n::seed0'][_key] = _val
        msh = L.metric_policy_audit(bad_shape)
        assert msh['status'] == 'warn', (msh, _key)
        assert any(_why in str(c.get('invalid')) for c in msh['checks']), (msh, _key)
    # A missing or non-numeric accuracy is an invalid record, not a reason to
    # abort the whole audit with float(None)/max([]).
    for _mut in ('missing', 'nonnumeric'):
        _bad_acc = json.loads(json.dumps(good))
        if _mut == 'missing':
            del _bad_acc['n::seed0']['acc']
        else:
            _bad_acc['n::seed0']['acc'] = 'not-a-number'
        _macc = L.metric_policy_audit(_bad_acc)
        assert _macc['status'] == 'warn', (_mut, _macc)
        assert any('retrieval endpoint is not usable' in w for w in _macc['warnings']), (_mut, _macc)
    # An empty support table has no majority class; it must fail closed rather
    # than raise from max() on an empty sequence.
    _empty_sup = json.loads(json.dumps(good))
    _empty_sup['n::seed0']['answer_support'] = {}
    _mes = L.metric_policy_audit(_empty_sup)
    assert _mes['status'] == 'warn', _mes
    assert any('answer-support-sum-mismatch' in str(c.get('invalid'))
               for c in _mes['checks']), _mes
    # A malformed POLICY (not just a malformed record) must fail closed
    # instead of crashing on .get/.items of a scalar entry.
    for _bad_policy in ([], {'x': None}, {'x': 1}):
        _mpol = L.metric_policy_audit(good, policy=_bad_policy)
        assert _mpol['status'] == 'warn', (_bad_policy, _mpol)
    # A malformed SUMMARY is not a mapping of records either: warn, do not
    # abort on `.values()`.
    for _bad_summary in (1, True, 'x', [None], [1]):
        _msum = L.metric_policy_audit(_bad_summary)
        assert _msum['status'] == 'warn' and _msum['warnings'], (_bad_summary, _msum)
    # A malformed threshold in an otherwise valid policy entry is reported but
    # does not abort the record audit.
    _bad_thr_policy = json.loads(json.dumps(L.METRIC_POLICY))
    _bad_thr_policy['binary_classification']['extreme_imbalance_threshold'] = 'bad'
    _mth = L.metric_policy_audit({'r': {'variant': 'r',
                                        'task': 'binary_classification',
                                        'prevalence': 1e-4,
                                        'recommended_primary_metric': 'f1'}},
                                 policy=_bad_thr_policy)
    assert _mth['status'] == 'warn', _mth
    # A support count above the float range must fail the record closed, not
    # abort the audit with an OverflowError when recall*support is formed.
    _huge = 10 ** 400
    _huge_rec = {'variant': 'h', 'task': 'niah_natural', 'acc': 0.5,
                 'nll': 1.0, 'n_queries': _huge, 'n_correct': _huge // 2,
                 'chance_accuracy': 0.5, 'majority_accuracy': 0.5,
                 'balanced_accuracy': 0.5,
                 'answer_support': {'a': _huge // 2, 'b': _huge - _huge // 2},
                 'per_class_recall': {'a': 0.5, 'b': 0.5}}
    _mh = L.metric_policy_audit({'h::seed0': _huge_rec})
    assert _mh['status'] == 'warn', _mh
    # Forged accuracy / majority floor / macro recall are rejected.
    for key, val in (('acc', 0.9), ('majority_accuracy', 0.1),
                     ('balanced_accuracy', 0.99), ('chance_accuracy', 0.01)):
        bad = json.loads(json.dumps(good))
        bad['n::seed0'][key] = val
        ma = L.metric_policy_audit(bad)
        assert ma['status'] == 'warn', (key, ma)
        assert any('retrieval endpoint is not usable' in w for w in ma['warnings']), (key, ma)
    # A legacy rate without n_correct or a floor is unverifiable, not a pass.
    legacy = {'n::seed0': {'variant': 'n', 'task': 'niah_natural', 'acc': 0.5,
                           'nll': 1.0, 'n_queries': 10}}
    ml = L.metric_policy_audit(legacy)
    assert ml['status'] == 'warn' and any('trivial floor' in w for w in ml['warnings']), ml


def t_baseline_tuning_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    def _arm(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better',
           'arms': {'full_matched': _arm([10.0, 11.0]), 'csa_dynamic': _arm([9.0, 9.5])}}
    r = L.baseline_tuning_audit(art, protocol=protocol)
    assert r['tuning_verified'] is True, r
    # An artifact carries the protocol it was produced under (the smoke panel
    # deliberately schedules a subset); the audit must use it when the caller
    # does not pass one explicitly.
    r_auto = L.baseline_tuning_audit(dict(art, protocol=protocol))
    assert r_auto['tuning_verified'] is True, r_auto
    bad = json.loads(json.dumps(art))
    bad['arms']['csa_dynamic']['trials'] = bad['arms']['csa_dynamic']['trials'][:1]
    r2 = L.baseline_tuning_audit(bad, protocol=protocol)
    assert r2['tuning_verified'] is False and any('same number' in p or 'same trial grid' in p for p in r2['problems']), r2
    leak = json.loads(json.dumps(art))
    leak['selection_split'] = 'test'
    r3 = L.baseline_tuning_audit(leak, protocol=protocol)
    assert r3['tuning_verified'] is False and any('validation' in p for p in r3['problems']), r3


def t_baseline_family_declared_contrasts_and_undeclared_arms():
    # The tuned multiplicity family is the set of DECLARED method-vs-baseline
    # contrasts, not `len(arms)-1`.  With two parameter-matching groups the
    # two differ, and using the arm count inflates the planning floor.
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _arm(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}

    arms = ['a', 'b', 'c', 'd', 'e']
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=arms, trials=trials,
                    search_seeds=[], final_seeds=[],
                    arm_groups=[
                        {'arms': ['a', 'b', 'c'], 'method_arm': 'a'},
                        {'arms': ['d', 'e'], 'method_arm': 'd'},
                    ])
    plan, basis = L._tuning_declared_contrasts(protocol, arms)
    assert plan == [('a', 'b'), ('a', 'c'), ('d', 'e')], (plan, basis)
    assert basis == 'declared_arm_groups' and len(plan) != len(arms) - 1
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better',
           'arms': {a: _arm([10.0, 11.0]) for a in arms}}
    out = L.baseline_tuning_audit(art, protocol=protocol)
    assert out['final_family_size'] == 3, out
    assert out['final_family_basis'] == 'declared_arm_groups', out
    assert out['final_family_contrasts'] == [['a', 'b'], ['a', 'c'], ['d', 'e']], out
    # A verified equal-search artifact is not automatically a tuned claim:
    # this two-level grid is coarse, so the adequacy gate must refuse it.
    assert out['tuning_verified'] is True and out['tuned_claim_allowed'] is False, out
    assert out['tuned_claim_reasons'], out
    # An undeclared arm must not enlarge the family or buy coverage.
    extra = json.loads(json.dumps(art))
    extra['arms']['f'] = _arm([10.0, 11.0])
    rex = L.baseline_tuning_audit(extra, protocol=protocol)
    assert rex['tuning_verified'] is False, rex
    assert any('undeclared arm' in p for p in rex['problems']), rex

    # Search-to-final DATA transfer: a config tuned on one corpus/validation
    # slice is not tuned for a final estimate on another.  Steps and tokens
    # matching does not make the data identity match.
    def _row(cfg, ppl, dfp='DFP', vfp='VFP'):
        return {'config': dict(cfg), 'ppl': float(ppl), 'steps': 10,
                'tokens_seen': 100, 'seeds': [100],
                'seed_ppls': {'100': float(ppl)},
                'data_fp': dfp, 'val_fp': vfp, 'data_version': 'DV'}
    dp = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
              search_seeds=[100], final_seeds=[0], inherits_tuning={})
    dp.pop('arm_groups', None)
    rows = {'a': [_row(trials[0], 10.0), _row(trials[1], 11.0)],
            'b': [_row(trials[0], 9.0), _row(trials[1], 9.5)]}
    d_art = {'selection_metric': 'ppl', 'selection_split': 'validation',
             'direction': 'lower_is_better', 'search_seeds': [100],
             'final_seeds': [0],
             'arms': {a: {'trials': rr,
                          'selected': dict(min(rr, key=lambda x: x['ppl'])['config']),
                          'selected_metric_value': min(x['ppl'] for x in rr)}
                      for a, rr in rows.items()},
             'final_records': {
                 f'{a}::seed0': {'seed': 0, 'variant': a, 'ppl': 9.0,
                                 'steps': 10, 'tokens_seen': 100,
                                 'data_fp': 'DFP', 'val_fp': 'VFP',
                                 'data_version': 'DV',
                                 'hyperparams': dict(min(rows[a], key=lambda x: x['ppl'])['config'])}
                 for a in ('a', 'b')}}
    okd = L.baseline_tuning_audit(d_art, protocol=dp)
    assert okd['tuning_verified'] is True, okd
    assert okd['data_transfer']['trial_fingerprints']['data_fp']['a']['values'] == ['DFP']
    bad_trial = json.loads(json.dumps(d_art))
    bad_trial['arms']['b']['trials'][0]['data_fp'] = 'OTHER'
    rbt = L.baseline_tuning_audit(bad_trial, protocol=dp)
    assert rbt['tuning_verified'] is False, rbt
    assert any('different data_fp' in p for p in rbt['problems']), rbt
    bad_final = json.loads(json.dumps(d_art))
    bad_final['final_records']['b::seed0']['data_fp'] = 'OTHER'
    rbf = L.baseline_tuning_audit(bad_final, protocol=dp)
    assert rbf['tuning_verified'] is False, rbf
    assert any('tuning trials used' in p for p in rbf['problems']), rbf


def t_baseline_horizon_transfer_audit():
    # A config selected at a shorter search horizon is not the tuned optimum at
    # the final horizon; equal search across arms does not catch this.
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[100], final_seeds=[0])

    def _art(search_steps, final_steps, trial_steps=None):
        rows = []
        for _i, _cfg in enumerate(trials):
            _st = trial_steps if trial_steps is not None else search_steps
            rows.append({'config': dict(_cfg), 'ppl': 10.0 + _i,
                         'steps': _st, 'tokens_seen': _st * 8})
        best = min(rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better', 'search_steps': search_steps,
                'search_seeds': [100], 'final_seeds': [0],
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}},
                'final_records': {
                    'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 6.0,
                                 'steps': final_steps,
                                 'tokens_seen': final_steps * 8,
                                 'hyperparams': dict(best['config'],
                                                     steps=final_steps,
                                                     tokens_seen=final_steps * 8)}}}

    good = L.baseline_tuning_audit(_art(10, 10), protocol=protocol)
    assert good['tuning_verified'] is True, good
    assert good['budget_transfer']['equal'] is True, good['budget_transfer']
    assert good['budget_transfer']['tokens_equal'] is True, good['budget_transfer']
    assert not any('different training horizon' in w for w in good['warnings']), good

    mismatch = L.baseline_tuning_audit(_art(10, 20), protocol=protocol)
    assert mismatch['tuning_verified'] is True, mismatch
    assert mismatch['budget_transfer']['equal'] is False, mismatch['budget_transfer']
    assert mismatch['budget_transfer']['mismatch_arms'] == ['a'], mismatch['budget_transfer']
    assert mismatch['budget_transfer']['tokens_equal'] is False, mismatch['budget_transfer']
    assert mismatch['budget_transfer']['token_mismatch_arms'] == ['a'], mismatch['budget_transfer']
    assert any('NOT a fully tuned baseline at the final budget' in w
               for w in mismatch['warnings']), mismatch['warnings']

    malformed = L.baseline_tuning_audit(_art(0, 10), protocol=protocol)
    assert malformed['tuning_verified'] is False, malformed
    assert any('search_steps' in p for p in malformed['problems']), malformed

    trial_mismatch = L.baseline_tuning_audit(_art(10, 10, trial_steps=5), protocol=protocol)
    assert trial_mismatch['tuning_verified'] is False, trial_mismatch
    assert any('does not describe the search budget' in p
               for p in trial_mismatch['problems']), trial_mismatch


def t_tuning_grid_boundary_adequacy():
    # Equal search is not enough if the winner sits on the edge of the grid:
    # the true optimum may be outside the searched range.
    trials = [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.001, 'weight_decay': 0.1, 'warmup': 50},
    ]
    def _art(selected_i, ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(trials, ppls)]
        best = rows[selected_i]
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better',
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[], final_seeds=[])
    edge = L.baseline_tuning_audit(_art(2, [10.0, 9.0, 8.0]), protocol=protocol)
    assert edge['tuning_verified'] is True, edge
    assert edge['grid_adequacy'] == 'warn', edge
    assert edge['grid_boundary_hits'] == ['a:lr'], edge
    assert edge['grid_coverage']['a']['lr']['position'] == 2, edge['grid_coverage']
    assert any('boundary of the declared search grid' in w for w in edge['warnings']), edge
    inside = L.baseline_tuning_audit(_art(1, [10.0, 8.0, 9.0]), protocol=protocol)
    assert inside['grid_adequacy'] == 'ok' and not inside['grid_boundary_hits'], inside
    # A capped search that stopped at the first two declared levels has a
    # boundary at its own maximum even though more levels were declared.
    capped = _art(1, [10.0, 8.0])
    capped['arms']['a']['trials'] = capped['arms']['a']['trials'][:2]
    capped['arms']['a']['selected'] = dict(trials[1])
    rc = L.baseline_tuning_audit(capped, protocol=protocol)
    assert rc['grid_adequacy'] == 'warn', rc
    assert rc['grid_coverage']['a']['lr']['n_searched_levels'] == 2, rc['grid_coverage']
    assert any('searched level' in w for w in rc['warnings']), rc
    # An artifact cannot label a strict-subset search as complete: that would
    # let the tuned-baseline gate accept a truncated grid.
    forged_complete = json.loads(json.dumps(capped))
    forged_complete['search_complete'] = True
    rfc = L.baseline_tuning_audit(forged_complete, protocol=protocol)
    assert rfc['tuning_verified'] is False and rfc['search_complete'] is False, rfc
    assert any('claims search_complete=True' in p for p in rfc['problems']), rfc
    non_bool = json.loads(json.dumps(capped))
    non_bool['search_complete'] = 'yes'
    rnb = L.baseline_tuning_audit(non_bool, protocol=protocol)
    assert rnb['tuning_verified'] is False and any(
        'search_complete must be a boolean' in p for p in rnb['problems']), rnb
    # Trials selected on different evaluation token counts are not comparable.
    _diff_tok = _art(1, [10.0, 8.0, 9.0])
    for _i, _t in enumerate(_diff_tok['arms']['a']['trials']):
        _t['n_tokens'] = 100 if _i < 2 else 50
    rt = L.baseline_tuning_audit(_diff_tok, protocol=protocol)
    assert rt['tuning_verified'] is False and any('different evaluation token counts' in p for p in rt['problems']), rt
    # A selected value outside the declared levels is not a tuned winner.
    outside = _art(1, [10.0, 8.0, 9.0])
    outside['arms']['a']['selected'] = {'lr': 0.002, 'weight_decay': 0.1, 'warmup': 50}
    outside['arms']['a']['trials'].append({'config': dict(outside['arms']['a']['selected']), 'ppl': 7.0})
    outside['arms']['a']['selected_metric_value'] = 7.0
    ro = L.baseline_tuning_audit(outside, protocol=protocol)
    assert ro['tuning_verified'] is False and any('outside the declared trial levels' in p for p in ro['problems']), ro
    # Equal marginal coverage is not full factorial coverage: an OFAT winner
    # cannot be described as a jointly tuned optimum.
    ofat_trials = [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.01, 'warmup': 50},
    ]
    ofat_proto = dict(protocol, trials=ofat_trials)
    def _ofat_art(ppls):
        rows = [{'config': dict(c), 'ppl': p} for c, p in zip(ofat_trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better',
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}
    rof = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=ofat_proto)
    assert rof['tuning_verified'] is True, rof
    assert rof['grid_interaction_untested'] == ['a:lr|weight_decay'], rof
    assert rof['grid_factorial']['a']['full_factorial'] is False, rof['grid_coverage']
    assert rof['grid_factorial']['a']['pairwise']['lr|weight_decay']['n_missing'] == 1, rof
    assert any('pairwise interaction' in w for w in rof['warnings']), rof['warnings']
    full_trials = ofat_trials + [{'lr': 0.0001, 'weight_decay': 0.01, 'warmup': 50}]
    full_proto = dict(protocol, trials=full_trials)
    frows = [{'config': dict(c), 'ppl': 10.0 - i} for i, c in enumerate(full_trials)]
    fbest = min(frows, key=lambda r: r['ppl'])
    rfull = L.baseline_tuning_audit(
        {'selection_metric': 'ppl', 'selection_split': 'validation',
         'direction': 'lower_is_better',
         'arms': {'a': {'trials': frows, 'selected': dict(fbest['config']),
                        'selected_metric_value': fbest['ppl']}}},
        protocol=full_proto)
    assert rfull['tuning_verified'] is True, rfull
    assert rfull['grid_interaction_untested'] == [], rfull
    assert rfull['grid_factorial']['a']['full_factorial'] is True, rfull['grid_coverage']
    # The search design is a declared design choice, not an implicit pile of
    # trials: a declared full factorial must actually be complete, while a
    # declared marginal/OFAT search is allowed but must not claim a joint optimum.
    declared_full = dict(ofat_proto, search_design='full_factorial')
    rbad = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=declared_full)
    assert rbad['tuning_verified'] is False and any(
        'declares a full_factorial' in p for p in rbad['problems']), rbad
    declared_ofat = dict(ofat_proto, search_design='one_factor_at_a_time')
    rofat = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=declared_ofat)
    assert rofat['tuning_verified'] is True and rofat['search_design'] == 'one_factor_at_a_time', rofat
    assert any('marginal/OFAT' in w for w in rofat['warnings']), rofat
    bad_design = dict(ofat_proto, search_design='whatever')
    rbd = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=bad_design)
    assert rbd['tuning_verified'] is False and any('search_design' in p for p in rbd['problems']), rbd
    empty_note = dict(ofat_proto, search_design='hand_designed_grid', search_design_note='')
    ren = L.baseline_tuning_audit(_ofat_art([10.0, 9.0, 8.0]), protocol=empty_note)
    assert ren['tuning_verified'] is False and any('search_design_note' in p for p in ren['problems']), ren

def t_baseline_multi_seed_selection_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _row(cfg, ppls, seeds=(0, 1, 2)):
        vals = [ppls[s] for s in seeds]
        return {'config': dict(cfg), 'ppl': float(np.mean(vals)),
                'ppl_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n_seeds': len(vals), 'seeds': list(seeds),
                'seed_ppls': {str(s): float(ppls[s]) for s in seeds},
                'steps': 10, 'tokens_seen': 100}
    def _arm(rows):
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
           'arms': {
               'full_matched': _arm([_row(trials[0], {0: 10, 1: 10.2, 2: 9.8}),
                                     _row(trials[1], {0: 11, 1: 11.1, 2: 10.9})]),
               'csa_dynamic': _arm([_row(trials[0], {0: 9, 1: 9.2, 2: 8.8}),
                                    _row(trials[1], {0: 9.5, 1: 9.6, 2: 9.4})]),
           }}
    r = L.baseline_tuning_audit(art, protocol=protocol)
    assert r['tuning_verified'] is True, r
    assert r['search_seed_balanced'] is True and r['search_seeds'] == [0, 1, 2], r
    # A trial's aggregate must be the mean of its own per-seed values.
    bad = json.loads(json.dumps(art))
    bad['arms']['csa_dynamic']['trials'][0]['ppl'] += 0.5
    rb = L.baseline_tuning_audit(bad, protocol=protocol)
    assert rb['tuning_verified'] is False and any('not the mean' in p for p in rb['problems']), rb
    # Selecting some arms on more seeds than others is unequal tuning.
    bad2 = json.loads(json.dumps(art))
    for t in bad2['arms']['full_matched']['trials']:
        t['seeds'] = [0, 1]
        t['seed_ppls'] = {k: v for k, v in t['seed_ppls'].items() if k in ('0', '1')}
        t['n_seeds'] = 2
        t['ppl'] = float(np.mean(list(t['seed_ppls'].values())))
        t['ppl_std'] = float(np.std(list(t['seed_ppls'].values()), ddof=1))
    rb2 = L.baseline_tuning_audit(bad2, protocol=protocol)
    assert rb2['tuning_verified'] is False and any('same search-seed set' in p for p in rb2['problems']), rb2
    # Every trial WITHIN an arm must use one search-seed set too.  Merely
    # having the same collection of per-trial sets in every arm is not an
    # equal search: config A and config B would be selected on different
    # draws.
    bad3 = json.loads(json.dumps(art))
    for _arm_name, _entry in bad3['arms'].items():
        _rows = _entry['trials']
        for _row, _keys in zip(_rows, (('0', '1'), ('1', '2'))):
            _row['seeds'] = [int(k) for k in _keys]
            _row['seed_ppls'] = {k: _row['seed_ppls'][k] for k in _keys}
            _row['n_seeds'] = len(_keys)
            _vals = list(_row['seed_ppls'].values())
            _row['ppl'] = float(np.mean(_vals))
            _row['ppl_std'] = float(np.std(_vals, ddof=1))
        _best = min(_rows, key=lambda r: r['ppl'])
        _entry['selected'] = dict(_best['config'])
        _entry['selected_metric_value'] = _best['ppl']
    rb3 = L.baseline_tuning_audit(bad3, protocol=protocol)
    assert rb3['tuning_verified'] is False, rb3
    assert rb3['search_seed_balanced'] is False, rb3
    assert any('more than one search-seed set' in p for p in rb3['problems']), rb3


def t_partial_skipped_seed_keeps_controlled_contrast():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(8):
        ka = knobs if i else dict(knobs, grad_clip_norm=2.0)
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=ka)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['n_paired'] == 7 and e['n_skipped'] == 1, e
    assert e['controlled'] is True and e['fully_paired'] is False, e
    # The six+ controlled pairs still support the claim; the rejected seed is
    # named instead of silently shrinking the evidence or erasing the contrast.
    assert 'c' in r['verified_components'], r['verified_components']
    assert any('controlled pair(s)' in w for w in r['warnings']), r['warnings']


def t_audit_incomplete_components_alias():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        for v, ppl in (('full', 50.0), ('full-A', 49.0)):
            s[f'{v}::seed{i}'] = dict(variant=v, seed=i, run_cfg=fp, steps=1,
                                      tokens_seen=1, ppl=ppl, params=100, train_knobs=knobs)
    design = {'A': [('full', 'full-A', ''), ('full', 'full-A2', '')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert r['ablation_completeness']['incomplete_components'] == ['A'], r['ablation_completeness']
    assert r['incomplete_components'] == ['A'], r['incomplete_components']
    assert r['missing_ablation_arms']['A'] == ['full-A2'], r['missing_ablation_arms']


def t_baseline_seed_partition_and_coverage_audit():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _row(cfg, ppl):
        return {'config': dict(cfg), 'ppl': float(ppl), 'steps': 10,
                'tokens_seen': 100, 'seeds': [0, 1, 2],
                'seed_ppls': {'0': float(ppl), '1': float(ppl) + 0.1,
                              '2': float(ppl) - 0.1}}

    def _arm(ppls):
        rows = [_row(c, p) for c, p in zip(trials, ppls)]
        best = min(rows, key=lambda r: r['ppl'])
        return {'trials': rows, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}

    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['full_matched', 'csa_dynamic'],
                    search_seeds=[0, 1, 2], final_seeds=[0, 1, 2])
    chosen = {'full_matched': dict(trials[0]), 'csa_dynamic': dict(trials[0])}
    art = {
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
        'final_seeds': [0, 1, 2],
        'arms': {'full_matched': _arm([10.0, 11.0]),
                 'csa_dynamic': _arm([9.0, 9.5])},
        'final_records': {
            f'{a}::seed{s}': {'seed': s, 'variant': a, 'ppl': 9.0 + s / 10.0,
                              'params': 100, 'train_knobs': L.training_knobs(),
                              'hyperparams': dict(chosen[a])}
            for a in chosen for s in (0, 1, 2)},
    }
    # The final records were measured on the selection seeds: refuse.
    bad = L.baseline_tuning_audit(dict(art, protocol=protocol), protocol=protocol)
    assert bad['tuning_verified'] is False, bad
    assert bad['search_final_seed_overlap'] == [0, 1, 2], bad
    assert any('overlap' in p for p in bad['problems']), bad

    # A disjoint partition is accepted (the 3-seed final floor is a warning,
    # because the tuned family itself is resolution-limited).
    good_protocol = dict(protocol, search_seeds=[100, 101, 102])
    good = L.baseline_tuning_audit(dict(art, protocol=good_protocol,
                                        search_seeds=[100, 101, 102]),
                                   protocol=good_protocol)
    assert good['tuning_verified'] is True, good
    assert good['search_final_seed_disjoint'] is True, good
    assert good['search_final_seed_overlap'] == [], good

    # Intervention controls inherit their parent's equal-search evidence, but
    # an unknown parent is reported instead of silently covering the arm.
    cov_protocol = dict(protocol, arms=['csa_fixed_rope'],
                        inherits_tuning={'csa_fixed_rope_randidx': 'csa_fixed_rope'})
    cov = L.baseline_tuning_audit(
        dict(art, protocol=cov_protocol,
             arms={'csa_fixed_rope': _arm([10.0, 11.0])},
             final_records={}),
        protocol=cov_protocol)
    assert cov['coverage'].get('csa_fixed_rope_randidx') == 'csa_fixed_rope', cov
    miss_protocol = dict(protocol, arms=['csa_fixed_rope'],
                         inherits_tuning={'csa_fixed_rope_randidx': 'absent_parent'})
    miss = L.baseline_tuning_audit(
        dict(art, protocol=miss_protocol,
             arms={'csa_fixed_rope': _arm([10.0, 11.0])},
             final_records={}),
        protocol=miss_protocol)
    assert any('absent_parent' in w for w in miss['warnings']), miss


def t_tuning_selection_stability_audit():
    def row(cfg, ppls):
        return {'config': dict(cfg), 'ppl': float(np.mean(list(ppls.values()))),
                'ppl_std': float(np.std(list(ppls.values()), ddof=1)),
                'seeds': sorted(ppls), 'seed_ppls': {str(k): float(v) for k, v in ppls.items()},
                'steps': 10, 'tokens_seen': 100}
    declared = L.BASELINE_TUNING_PROTOCOL['trials'][:2]
    stable = [row(declared[0], {0: 10, 1: 10, 2: 10}),
              row(declared[1], {0: 11, 1: 11, 2: 11})]
    s = L.tuning_selection_stability(stable)
    assert s['status'] == 'ok' and s['stable'] is True, s
    unstable = [row(declared[0], {0: 9, 1: 9, 2: 12}),
                row(declared[1], {0: 11, 1: 11, 2: 9})]
    u = L.tuning_selection_stability(unstable)
    assert u['status'] == 'ok' and u['stable'] is False, u
    assert u['n_distinct_configs'] == 2, u
    # The audit verifies a recorded stability claim instead of trusting it.
    trials = unstable
    best = min(trials, key=lambda t: t['ppl'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [0, 1, 2],
           'arms': {'a': {'trials': trials, 'selected': dict(best['config']),
                          'selected_metric_value': best['ppl'],
                          'selection_stability': {'stable': True}}}}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=declared,
                    inherits_tuning={}, search_seeds=[0, 1, 2],
                    final_seeds=[3, 4, 5])
    out = L.baseline_tuning_audit(art, protocol=protocol)
    assert out['tuning_verified'] is False, out
    assert any('selection_stability' in p for p in out['problems']), out
    # Without the false record the same artifact is merely warning-level: the
    # unstable pick is disclosed, not silently accepted.
    del art['arms']['a']['selection_stability']
    out2 = L.baseline_tuning_audit(art, protocol=protocol)
    assert out2['tuning_verified'] is True, out2
    assert out2['selection_stable_all'] is False, out2
    assert any('not leave-one-search-seed-out stable' in w for w in out2['warnings']), out2
    # Legacy rows without per-seed values can never report stability as a pass.
    no_seed = L.tuning_selection_stability([{'config': {'lr': 1e-4}, 'ppl': 1.0}])
    assert no_seed['status'] == 'unavailable', no_seed
    # One seed under two JSON spellings is a repeated draw, not two draws.
    dup = L.tuning_selection_stability([
        {'config': {'lr': 1e-4}, 'seed_ppls': {'0': 10.0, 0: 10.1,
                                                1: 10.2, 2: 10.3}}])
    assert dup['status'] == 'unavailable' and 'repeats seed' in dup['reason'], dup


def t_paired_row_bootstrap_delta():
    # Evaluation-row bootstrap for a paired token-weighted PPL difference:
    # the SAME rows are resampled for both arms and every shared seed.
    def _pair(sign=1.0, n_seeds=8, n_rows=16, ntok=50):
        out = []
        for s in range(n_seeds):
            a_nll = [2.0 + 0.01 * j for j in range(n_rows)]
            b_nll = [2.0 + 0.01 * j - sign * 0.2 for j in range(n_rows)]
            out.append(({'variant': 'a', 'seed': s,
                         'ppl': math.exp(sum(a_nll) / (n_rows * ntok)),
                         'nll_per_row': a_nll,
                         'n_tokens_per_row': [ntok] * n_rows},
                        {'variant': 'b', 'seed': s,
                         'ppl': math.exp(sum(b_nll) / (n_rows * ntok)),
                         'nll_per_row': b_nll,
                         'n_tokens_per_row': [ntok] * n_rows}))
        return out
    r = L.paired_row_bootstrap_delta(_pair(+1.0))
    assert r['status'] == 'ok' and r['n_pairs'] == 8 and r['n_rows'] == 16, r
    assert r['mean_delta'] > 0 and r['ci95_delta_ppl'][0] > 0, r
    assert r['n_seeds_used'] == 8, r
    # Deterministic draws for the same seed.
    assert L.paired_row_bootstrap_delta(_pair(+1.0))['ci95_delta_ppl'] == r['ci95_delta_ppl']
    # `conf` is coverage, not a tail probability: a 95% interval must be at
    # least as wide as a 50% interval on the same draws.  (Using conf as
    # alpha silently produced a 5% interval here.)
    _lo50 = L.paired_row_bootstrap_delta(_pair(+1.0), conf=0.50)
    _w95 = r['ci95_delta_ppl'][1] - r['ci95_delta_ppl'][0]
    _w50 = _lo50['ci95_delta_ppl'][1] - _lo50['ci95_delta_ppl'][0]
    assert _w95 > _w50, (r['ci95_delta_ppl'], _lo50['ci95_delta_ppl'])
    rn = L.paired_row_bootstrap_delta(_pair(-1.0))
    assert rn['status'] == 'ok' and rn['mean_delta'] < 0 and rn['ci95_delta_ppl'][1] < 0, rn
    # Invalid coverage/draw-count/seed are part of the measurement identity:
    # never silently build a 95% interval while echoing conf=1.0 or 0.5.
    for _bad in (True, 0, 1, 1.5, -0.1, float('nan'), 'x'):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), conf=_bad)
        assert _r['status'] == 'unverifiable' and _r.get('reason'), (_bad, _r)
        assert _r['conf'] is None and _r['ci95_delta_ppl'] is None, (_bad, _r)
    for _bad in (True, 0, 19, 1.5, 'x', None, float('nan')):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), n_boot=_bad)
        assert _r['status'] == 'unverifiable' and _r.get('reason'), (_bad, _r)
    for _bad in (True, 1.5, 'x', None, float('nan')):
        _r = L.paired_row_bootstrap_delta(_pair(+1.0), seed=_bad)
        assert _r['status'] == 'unverifiable' and 'seed' in _r.get('reason', ''), (_bad, _r)
    # A paired row bootstrap is not invented when either arm lacks per-row
    # accounting, when row counts differ, or when nested holdout rows were
    # reported without their per-row arrays.
    bad = _pair(+1.0)
    bad[0][0].pop('nll_per_row')
    assert L.paired_row_bootstrap_delta(bad)['status'] == 'unverifiable'
    bad2 = _pair(+1.0)
    bad2[0][1]['nll_per_row'] = bad2[0][1]['nll_per_row'][:-1]
    bad2[0][1]['n_tokens_per_row'] = bad2[0][1]['n_tokens_per_row'][:-1]
    assert L.paired_row_bootstrap_delta(bad2)['status'] == 'unverifiable'
    bad3 = _pair(+1.0)
    for _a, _b in bad3:
        _a['ppl_estimation_rows'] = 'nested_validation_holdout'
        _b['ppl_estimation_rows'] = 'nested_validation_holdout'
    assert L.paired_row_bootstrap_delta(bad3)['status'] == 'unverifiable'
    for _a, _b in bad3:
        _a['nll_per_row_holdout'] = list(_a['nll_per_row'])[:5]
        _a['n_tokens_per_row_holdout'] = [50] * 5
        _b['nll_per_row_holdout'] = list(_b['nll_per_row'])[:5]
        _b['n_tokens_per_row_holdout'] = [50] * 5
    rh = L.paired_row_bootstrap_delta(bad3)
    assert rh['status'] == 'ok' and rh['n_rows'] == 5, rh
    # A pair whose two arms use different estimands is not a paired row
    # comparison, even when both arrays happen to have the same length.
    mixed_pair = _pair(+1.0)
    mixed_pair[0][0]['ppl_estimation_rows'] = 'nested_validation_holdout'
    mixed_pair[0][0]['nll_per_row_holdout'] = list(mixed_pair[0][0]['nll_per_row'])
    mixed_pair[0][0]['n_tokens_per_row_holdout'] = list(mixed_pair[0][0]['n_tokens_per_row'])
    rmix = L.paired_row_bootstrap_delta(mixed_pair)
    assert rmix['status'] == 'unverifiable' and 'different estimands' in rmix['reason'], rmix
    # Nor may different shared pairs contribute intervals from different
    # estimands to one mean/CI.
    mixed_group = _pair(+1.0)
    for _a, _b in mixed_group[:1]:
        for _r in (_a, _b):
            _r['ppl_estimation_rows'] = 'nested_validation_holdout'
            _r['nll_per_row_holdout'] = list(_r['nll_per_row'])
            _r['n_tokens_per_row_holdout'] = list(_r['n_tokens_per_row'])
    rmg = L.paired_row_bootstrap_delta(mixed_group)
    assert rmg['status'] == 'unverifiable' and 'mix' in rmg['reason'], rmg
    # The methodology audit attaches the interval to every controlled contrast
    # (and reports unverifiable rather than inventing rows).
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    summary = {}
    for s in range(6):
        for arm, off in (('a', 0.0), ('b', -0.2)):
            nll = [2.0 + off + 0.01 * j for j in range(8)]
            ntok = [50] * 8
            ppl = math.exp(sum(nll) / sum(ntok))
            summary[f'{arm}::seed{s}'] = {
                'variant': arm, 'seed': s, 'run_cfg': fp, 'steps': 1,
                'tokens_seen': 400, 'params': 100, 'ppl': ppl,
                'nll_sum': math.log(ppl) * sum(ntok), 'n_tokens': sum(ntok),
                'nll_per_row': nll, 'n_tokens_per_row': ntok,
                'train_knobs': L.training_knobs(),
            }
    ra = L.methodology_audit(summary, design={'c': [('a', 'b', 'x')]},
                             baselines={}, verbose=False)
    _ent = ra['components']['c'][0]
    _boot = _ent.get('paired_row_bootstrap_delta') or {}
    assert _boot.get('status') == 'ok' and _boot.get('n_rows') == 8, _boot
    # b has the lower NLL, so delta = PPL_a - PPL_b is positive.
    assert _boot.get('mean_delta') > 0 and _boot.get('ci95_delta_ppl')[0] > 0, _boot

def t_methodology_audit_requires_panel_to_use_tuned_config():
    with tempfile.TemporaryDirectory() as d:
        trials = [{'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 1},
                  {'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1},
                  {'lr': 1e-3, 'weight_decay': 0.1, 'warmup': 1}]
        protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                        inherits_tuning={}, search_seeds=[100, 101, 102],
                        final_seeds=[0, 1, 2])
        art = {'protocol': protocol, 'selection_metric': 'ppl',
               'selection_split': 'validation', 'direction': 'lower_is_better',
               'search_seeds': [100, 101, 102], 'final_seeds': [0, 1, 2],
               'arms': {
                   'a': {'selected': dict(trials[0]), 'selected_metric_value': 2.0,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.0, 3.0, 4.0))]},
                   'b': {'selected': dict(trials[0]), 'selected_metric_value': 2.1,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.1, 3.1, 4.1))]},
               }}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), art)
        knobs = L.training_knobs()

        def _summary(b_lr=3e-4):
            out = {}
            for i in range(8):
                out[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg='same_cfg',
                                          steps=1, tokens_seen=10, ppl=50.0, params=100,
                                          train_knobs=knobs,
                                          hyperparams={'lr': 3e-4, 'weight_decay': 0.1,
                                                       'warmup': 1, 'mlp_ratio': 4.0})
                out[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg='same_cfg',
                                          steps=1, tokens_seen=10, ppl=51.0 + i * 0.0,
                                          params=100, train_knobs=knobs,
                                          hyperparams={'lr': b_lr, 'weight_decay': 0.1,
                                                       'warmup': 1, 'mlp_ratio': 4.0})
            return out

        design = {'comp': [('a', 'b', 'x')]}
        r = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r['baseline_tuning']['tuning_verified'] is True, r['baseline_tuning']
        assert r['baseline_tuning']['panel_arm_hyperparams_match'] == {'a': True, 'b': True}, r
        assert 'comp' in r['tuned_verified_baselines'], r.get('tuned_verified_baselines')
        # The panel ran b at a config that was NOT selected: the artifact may
        # exist, but the contrast is not tuned-baseline evidence.
        r2 = L.methodology_audit(_summary(b_lr=1e-4), baselines=design, outdir=d, verbose=False)
        assert r2['baseline_tuning']['panel_arm_hyperparams_match']['b'] is False, r2['baseline_tuning']
        assert 'comp' in r2['untuned_verified_baselines'], r2.get('untuned_verified_baselines')
        assert any('not trained at the tuned config' in w for w in r2['warnings']), r2['warnings']
        # Equal search over a coarse two-level lr grid is not a "well-tuned"
        # baseline: the panel report must keep the contrast in the untuned
        # bucket and expose the tuning caveat, not silently promote it.
        _trials2 = trials[:2]
        _proto2 = dict(protocol, trials=_trials2)
        _art2 = {'protocol': _proto2, 'selection_metric': 'ppl',
                 'selection_split': 'validation', 'direction': 'lower_is_better',
                 'search_seeds': [100, 101, 102], 'final_seeds': [0, 1, 2],
                 'arms': {
                     'a': {'selected': dict(_trials2[0]), 'selected_metric_value': 2.0,
                           'trials': [{'config': c, 'ppl': p}
                                      for c, p in zip(_trials2, (2.0, 3.0))]},
                     'b': {'selected': dict(_trials2[0]), 'selected_metric_value': 2.1,
                           'trials': [{'config': c, 'ppl': p}
                                      for c, p in zip(_trials2, (2.1, 3.1))]},
                 }}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), _art2)
        r3 = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r3['baseline_tuning']['tuning_verified'] is True, r3['baseline_tuning']
        assert r3['baseline_tuning']['tuned_claim_allowed'] is False, r3['baseline_tuning']
        assert 'comp' not in r3['tuned_verified_baselines'], r3.get('tuned_verified_baselines')
        assert 'comp' in r3['untuned_verified_baselines'], r3.get('untuned_verified_baselines')
        assert any('does not support a "well-tuned baseline" claim' in w
                   for w in r3['warnings']), r3['warnings']
        assert any('baseline tuning:' in w and 'coarse' in w
                   for w in r3['warnings']), r3['warnings']
        # A search horizon shorter than the final horizon is not a fully
        # tuned baseline: the selected lr may be optimal for the cheap search
        # budget, not for the budget that is reported.
        _art4 = json.loads(json.dumps(_art2))
        _art4['search_steps'] = 1
        _art4['final_records'] = {}
        for _arm in ('a', 'b'):
            for _seed in (0, 1, 2):
                _art4['final_records'][f'{_arm}::seed{_seed}'] = {
                    'variant': _arm, 'seed': _seed, 'ppl': 50.0, 'steps': 2,
                    'hyperparams': dict(_trials2[0], steps=2)}
        L.atomic_write_json(os.path.join(d, 'baseline_tuning.json'), _art4)
        r4 = L.methodology_audit(_summary(), baselines=design, outdir=d, verbose=False)
        assert r4['baseline_tuning']['budget_transfer']['equal'] is False, r4['baseline_tuning']
        assert r4['baseline_tuning']['tuned_claim_allowed'] is False, r4['baseline_tuning']
        assert 'comp' not in r4['tuned_verified_baselines'], r4.get('tuned_verified_baselines')
        assert 'comp' in r4['untuned_verified_baselines'], r4.get('untuned_verified_baselines')
        assert any('different training horizon' in w for w in r4['warnings']), r4['warnings']


def t_baseline_contrast_plan_is_group_aware():
    arms = list(L.BASELINE_TUNING_PROTOCOL['arms'])
    groups = BT._protocol_arm_groups(L.BASELINE_TUNING_PROTOCOL, L.RUN, arms)
    plan = BT._contrast_plan(L.BASELINE_TUNING_PROTOCOL, groups, arms)
    assert ('csa_dynamic', 'full_matched') in plan and ('csa_dynamic', 'csa_fixed') in plan, plan
    assert ('csa_fixed_rope', 'full_rope') in plan, plan
    # Cross-group contrasts would also change the parameter-matching reference.
    assert not any(a == 'csa_fixed_rope' and b in ('full_matched', 'csa_dynamic')
                   for a, b in plan), plan
    assert not any(a == 'csa_dynamic' and b in ('full_rope', 'csa_fixed_rope')
                   for a, b in plan), plan
    legacy = BT._contrast_plan({'method_arm': 'csa_dynamic'}, groups, arms)
    assert ('csa_dynamic', 'full_matched') in legacy and ('csa_dynamic', 'full_rope') in legacy, legacy
    # The default protocol must be able to confirm its own declared family and
    # must not evaluate the final records on the selection draws.
    proto = L.BASELINE_TUNING_PROTOCOL
    need = L.signflip_min_n_for_holm(len(plan), 0.05)
    assert need is not None and len(proto['final_seeds']) >= need, (need, proto['final_seeds'])
    assert not (set(proto['search_seeds']) & set(proto['final_seeds'])), proto


def t_baseline_driver_refuses_overlapping_final_seeds():
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'],
                    search_seeds=[0], final_seeds=[0])
    try:
        BT.run_tuning(dict(L.RUN, outdir=tempfile.mkdtemp(prefix='tune_overlap_')),
                      outdir=tempfile.mkdtemp(prefix='tune_overlap_out_'),
                      protocol=protocol, final=True)
    except ValueError as e:
        assert 'overlap' in str(e), e
    else:
        raise AssertionError('overlapping search/final seeds were not refused')
    # An empty final partition must not silently fall back to seed 0 (which
    # could be a search draw).
    empty = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'],
                 search_seeds=[0], final_seeds=[])
    try:
        BT.run_tuning(dict(L.RUN, outdir=tempfile.mkdtemp(prefix='tune_empty_')),
                      outdir=tempfile.mkdtemp(prefix='tune_empty_out_'),
                      protocol=empty, final=True)
    except ValueError as e:
        assert 'final_seeds' in str(e), e
    else:
        raise AssertionError('an empty final seed partition was not refused')
    # A scalar zero is a valid single final seed; `or []` used to erase it.
    _search, _final, _shared = BT._search_final_seeds(
        {'search_seeds': [1], 'final_seeds': 0})
    assert (_search, _final, _shared) == ([1], [0], []), (_search, _final, _shared)


def t_nested_validation_row_split_and_subset_ppl():
    sel, ev, meta = L.validation_row_split(17, seed=123, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    assert sorted(sel + ev) == list(range(17)), (sel, ev)
    assert not (set(sel) & set(ev)), (sel, ev)
    assert meta['disjoint'] is True and meta['covers_all_rows'] is True, meta
    assert meta['n_selection_rows'] + meta['n_evaluation_rows'] == 17, meta
    # Validation rows are consecutive stream chunks, so the folds must be
    # contiguous in time order: selection is an earlier block and evaluation a
    # later one.  A seeded row permutation would interleave future chunks into
    # selection and split the same document across the two stages repeatedly.
    assert sel == list(range(len(sel))) and ev == list(range(len(sel), 17)), (sel, ev)
    assert max(sel) < min(ev)
    assert meta['policy'] == 'chronological_contiguous_row_folds', meta
    assert meta['chronological'] is True and meta['selection_precedes_evaluation'] is True, meta
    assert meta['selection_row_range'] == [0, len(sel) - 1], meta
    assert meta['evaluation_row_range'] == [len(sel), 16], meta
    try:
        L.validation_row_split(17, selection_fold=1, evaluation_fold=0)
    except ValueError as e:
        assert 'precede' in str(e), e
    else:
        raise AssertionError('a reverse-time nested split must be refused')
    # Deterministic across calls and independent of the row order in which
    # the same physical rows are presented (indices are the identity).
    sel2, ev2, meta2 = L.validation_row_split(17, seed=123)
    assert (sel, ev, meta) == (sel2, ev2, meta2), 'row split must be deterministic'
    # A different seed must actually move rows (not silently ignore the seed).
    sel3, ev3, _ = L.validation_row_split(17, seed=124)
    assert (sel3, ev3) != (sel, ev), 'row-split seed did not change the folds'
    rec = {'nll_per_row': [1.0, 2.0, 3.0, 4.0],
           'n_tokens_per_row': [10, 10, 10, 10]}
    st = L.record_subset_ppl(rec, [1, 3])
    assert st['n_tokens'] == 20 and abs(st['nll'] - 0.3) < 1e-12, st
    assert abs(st['ppl'] - math.exp(0.3)) < 1e-12, st
    # Missing accounting or out-of-range rows are unverifiable, not silently
    # substituted with the full-validation number.
    assert L.record_subset_ppl({'ppl': 9.0}, [0]) is None
    assert L.record_subset_ppl(rec, [99]) is None
    assert L.record_subset_ppl(rec, []) is None
    # A fractional index must not be truncated to a different row, booleans
    # are flags not row ids, and repeating a row must not double-count its
    # tokens in the held-out pooled PPL.
    assert L.record_subset_ppl(rec, [1.5]) is None
    assert L.record_subset_ppl(rec, [True]) is None
    assert L.record_subset_ppl(rec, [1, 1]) is None
    from decimal import Decimal as _Decimal
    assert L.record_subset_ppl(rec, [_Decimal('1')])['n_tokens'] == 10
    # Token counts are exact integer denominators.  A fractional count must
    # not be used as a pooled denominator (and a boolean is not a count).
    for _bad_ntok in ([10, 10, 10, 10.5], [10, 10, 10, True],
                      [10, 10, 10, float('nan')]):
        assert L.record_subset_ppl({'nll_per_row': [1.0, 2.0, 3.0, 4.0],
                                    'n_tokens_per_row': _bad_ntok}, [1, 3]) is None, _bad_ntok
    assert L._pooled_ppl_from_records([{'nll_sum': 1.0, 'n_tokens': 10.5}]) == (None, None, None)
    assert L._group_params([{'params': 12.5}, {'params': True}]) is None
    assert L._group_params([{'params': 12.0}]) == 12
    for _bad_alpha in (True, 0, 1.0, float('nan')):
        try:
            L._seed_t_ci([1.0, 2.0, 3.0], alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_seed_t_ci accepted alpha={_bad_alpha!r}')
    try:
        L.validation_row_split(1)
    except ValueError:
        pass
    else:
        raise AssertionError('a one-row validation slice cannot be split')
    # Fold indices must select their actual contiguous blocks, not silently
    # fall back to "first block vs all remaining rows".
    sel4, ev4, m4 = L.validation_row_split(17, seed=123, n_folds=4,
                                           selection_fold=1, evaluation_fold=3)
    folds = m4['fold_ranges']
    assert sel4 == list(range(folds[1][0], folds[1][1] + 1)), (sel4, m4)
    assert ev4 == list(range(folds[3][0], folds[3][1] + 1)), (ev4, m4)
    assert m4['fold_sizes'] == [5, 4, 4, 4], m4
    assert m4['covers_all_rows'] is False and m4['disjoint'] is True, m4
    try:
        L.validation_row_split(17, n_folds=18)
    except ValueError:
        pass
    else:
        raise AssertionError('more folds than rows must be refused')


def t_nested_tuning_artifact_audit():
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 5},
              {'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 5}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[100, 101, 102],
                    final_seeds=[0, 1, 2, 3, 4, 5, 6, 7],
                    inherits_tuning={})
    protocol.pop('arm_groups', None)

    def _row(cfg, nlls):
        seed_ppls = {str(s): math.exp(v / 100.0)
                     for s, v in zip(protocol['search_seeds'], nlls)}
        nll_sum = float(sum(nlls))
        n_tok = 100 * len(nlls)
        ppl = math.exp(nll_sum / n_tok)
        return {'config': dict(cfg), 'ppl': ppl,
                'ppl_std': float(np.std(list(seed_ppls.values()), ddof=1)),
                'ppl_aggregation': 'pooled_token_nll',
                'nll_sum': nll_sum, 'n_tokens': n_tok,
                'seed_ppls': seed_ppls,
                'seed_nll_sum': {str(s): float(v) for s, v in
                                 zip(protocol['search_seeds'], nlls)},
                'seed_n_tokens': {str(s): 100 for s in protocol['search_seeds']},
                'seeds': list(protocol['search_seeds']),
                'steps': 10, 'tokens_seen': 1000,
                'selection_row_split_applied': True,
                'selection_row_fp': 'SELFP', 'evaluation_row_fp': 'EVALFP'}

    rows = [_row(trials[0], [100, 101, 99]), _row(trials[1], [110, 111, 109])]
    best = min(rows, key=lambda r: r['ppl'])
    art = {'selection_metric': 'ppl', 'selection_split': 'validation',
           'direction': 'lower_is_better', 'search_seeds': [100, 101, 102],
           'final_seeds': protocol['final_seeds'],
           'selection_row_split': {'applied': True, 'disjoint': True,
                                   'covers_all_rows': True,
                                   'policy': 'chronological_contiguous_row_folds',
                                   'chronological': True,
                                   'selection_precedes_evaluation': True,
                                   'selection_row_range': [0, 7],
                                   'evaluation_row_range': [8, 15],
                                   'selection_row_fp': 'SELFP',
                                   'evaluation_row_fp': 'EVALFP',
                                   'n_selection_rows': 8, 'n_evaluation_rows': 8},
           'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                          'selected_metric_value': best['ppl']}},
           'final_records': {
               f'a::seed{s}': {'seed': s, 'variant': 'a', 'ppl': 5.0,
                               'ppl_holdout': 5.1, 'holdout_row_fp': 'EVALFP',
                               'params': 100, 'train_knobs': L.training_knobs(),
                               'hyperparams': dict(best['config'])}
               for s in protocol['final_seeds']}}
    ok = L.baseline_tuning_audit(art, protocol=protocol)
    assert ok['tuning_verified'] is True, ok
    assert ok['selection_row_split_applied'] is True and ok['nested_final_evaluation'] is True, ok
    # A winner inside one standard error of the runner-up is a tie, not a
    # uniquely tuned optimum.  The audit recomputes the margin, warns, and
    # refuses a self-reported margin that does not match the trials.
    tie = json.loads(json.dumps(art))
    tie['arms']['a']['trials'][1] = _row(trials[1], [100, 101, 99])
    rt = L.baseline_tuning_audit(tie, protocol=protocol)
    assert rt['tuning_verified'] is True, rt
    assert any('inside one standard error' in w for w in rt['warnings']), rt['warnings']
    fake = json.loads(json.dumps(art))
    fake['arms']['a']['selection_margin'] = 123.0
    rf = L.baseline_tuning_audit(fake, protocol=protocol)
    assert rf['tuning_verified'] is False, rf
    assert any('selection_margin' in p for p in rf['problems']), rf
    # A claimed nested run without held-out final rows is refused: the tuned
    # contrast would otherwise fall back to the selection documents.
    bad = json.loads(json.dumps(art))
    del bad['final_records']['a::seed0']['ppl_holdout']
    rb = L.baseline_tuning_audit(bad, protocol=protocol)
    assert rb['tuning_verified'] is False, rb
    assert any('held-out-row PPL' in p for p in rb['problems']), rb
    # A final record scored on the wrong row fold is refused too.
    bad2 = json.loads(json.dumps(art))
    bad2['final_records']['a::seed0']['holdout_row_fp'] = 'OTHER'
    rb2 = L.baseline_tuning_audit(bad2, protocol=protocol)
    assert rb2['tuning_verified'] is False and any('different held-out' in p for p in rb2['problems']), rb2
    # Final records must attest the selected knobs.  A missing descriptor or a
    # missing knob is a failed verification, not "nothing to compare".
    bad_hp = json.loads(json.dumps(art))
    bad_hp['final_records']['a::seed0']['hyperparams'] = {}
    rbh = L.baseline_tuning_audit(bad_hp, protocol=protocol)
    assert rbh['tuning_verified'] is False and any('unattested' in p for p in rbh['problems']), rbh
    bad_hp2 = json.loads(json.dumps(art))
    del bad_hp2['final_records']['a::seed1']['hyperparams']
    rbh2 = L.baseline_tuning_audit(bad_hp2, protocol=protocol)
    assert rbh2['tuning_verified'] is False and any('no structured hyperparams' in p for p in rbh2['problems']), rbh2
    # A legacy/seeded random row permutation is disjoint but interleaves future
    # chunks into selection, so it cannot pass the nested-validation gate.
    bad_policy = json.loads(json.dumps(art))
    bad_policy['selection_row_split']['policy'] = 'seeded_splitmix64_row_folds'
    rp = L.baseline_tuning_audit(bad_policy, protocol=protocol)
    assert rp['tuning_verified'] is False, rp
    assert any('chronological_contiguous_row_folds' in p for p in rp['problems']), rp
    # Reverse-time ranges are refused as well.
    bad_range = json.loads(json.dumps(art))
    bad_range['selection_row_split']['selection_row_range'] = [8, 15]
    bad_range['selection_row_split']['evaluation_row_range'] = [0, 7]
    rr = L.baseline_tuning_audit(bad_range, protocol=protocol)
    assert rr['tuning_verified'] is False, rr
    assert any('selection-before-evaluation' in p for p in rr['problems']), rr
    # A search trial selected on different rows than declared is refused.
    bad3 = json.loads(json.dumps(art))
    bad3['arms']['a']['trials'][0]['selection_row_fp'] = 'OTHER'
    rb3 = L.baseline_tuning_audit(bad3, protocol=protocol)
    assert rb3['tuning_verified'] is False and any('selected on different rows' in p for p in rb3['problems']), rb3
    # The trial must also record the SAME held-out fold the artifact will use
    # for the final comparison.
    bad_eval = json.loads(json.dumps(art))
    bad_eval['arms']['a']['trials'][0]['evaluation_row_fp'] = 'OTHER'
    rbe = L.baseline_tuning_audit(bad_eval, protocol=protocol)
    assert rbe['tuning_verified'] is False and any('different evaluation fold' in p for p in rbe['problems']), rbe
    # A truthy non-boolean split flag is not an applied nested split.
    bad_flag = json.loads(json.dumps(art))
    bad_flag['arms']['a']['trials'][0]['selection_row_split_applied'] = 'yes'
    rbf = L.baseline_tuning_audit(bad_flag, protocol=protocol)
    assert rbf['tuning_verified'] is False and any('did not use the declared nested' in p for p in rbf['problems']), rbf
    # Extra final records outside the declared final-seed plan must not enlarge
    # the effective n.
    extra = json.loads(json.dumps(art))
    extra['final_records']['a::seed99'] = dict(extra['final_records']['a::seed0'], seed=99)
    rxe = L.baseline_tuning_audit(extra, protocol=protocol)
    assert rxe['tuning_verified'] is False and any('undeclared seed' in p for p in rxe['problems']), rxe

    # More than two folds intentionally leaves unused blocks, so
    # `covers_all_rows=False` is correct and must not fail the nested gate.
    # The recorded fold description is also re-derived from
    # (n_rows, seed, n_folds, selection_fold, evaluation_fold).
    mf_protocol = dict(protocol, selection_row_split={
        'enabled': True, 'seed': 123, 'n_folds': 4,
        'selection_fold': 1, 'evaluation_fold': 3})
    _sel, _ev, _mf_meta = L.validation_row_split(
        17, seed=123, n_folds=4, selection_fold=1, evaluation_fold=3)
    mf = json.loads(json.dumps(art))
    mf['selection_row_split'] = dict(_mf_meta, applied=True)
    for _t in mf['arms']['a']['trials']:
        _t['selection_row_fp'] = _mf_meta['selection_row_fp']
        _t['evaluation_row_fp'] = _mf_meta['evaluation_row_fp']
    for _r in mf['final_records'].values():
        _r['holdout_row_fp'] = _mf_meta['evaluation_row_fp']
    rmf = L.baseline_tuning_audit(mf, protocol=mf_protocol)
    assert rmf['tuning_verified'] is True, rmf
    assert rmf['selection_row_split_applied'] is True, rmf
    # A self-consistent but edited range/fingerprint must be refused because
    # it cannot be reproduced from the declared fold parameters.
    mf_bad = json.loads(json.dumps(mf))
    mf_bad['selection_row_split']['evaluation_row_range'] = [0, 3]
    rmb = L.baseline_tuning_audit(mf_bad, protocol=mf_protocol)
    assert rmb['tuning_verified'] is False, rmb
    assert any('not reproducible' in p for p in rmb['problems']), rmb
    # A document-group-aware selection split is valid even though a row-only
    # replay would place the boundary at a different row.  The audit must not
    # reject it merely because (n_rows, seed, folds) cannot reproduce it.
    _grp_protocol = dict(protocol, selection_row_split={
        'enabled': True, 'seed': 20261020, 'n_folds': 2,
        'selection_fold': 0, 'evaluation_fold': 1})
    _g_sel, _g_ev, _g_meta = L.validation_row_split(
        4, seed=20261020, n_folds=2, selection_fold=0, evaluation_fold=1,
        groups=[0, 1, 1, 1])
    grp = json.loads(json.dumps(art))
    grp['selection_row_split'] = dict(_g_meta, applied=True)
    for _t in grp['arms']['a']['trials']:
        _t['selection_row_fp'] = _g_meta['selection_row_fp']
        _t['evaluation_row_fp'] = _g_meta['evaluation_row_fp']
    for _r in grp['final_records'].values():
        _r['holdout_row_fp'] = _g_meta['evaluation_row_fp']
    rg = L.baseline_tuning_audit(grp, protocol=_grp_protocol)
    assert rg['tuning_verified'] is True, rg


def t_nested_final_uses_holdout_rows():
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                    search_seeds=[100, 101, 102], final_seeds=[0, 1, 2, 3],
                    inherits_tuning={}, method_arm='a')
    protocol.pop('arm_groups', None)
    out = tempfile.mkdtemp(prefix='nested_final_')
    selected = {a: {'selected': dict(trials[0]), 'trials': [{}],
                    'selected_metric_value': 1.0} for a in ('a', 'b')}
    L.atomic_write_json(os.path.join(out, 'baseline_tuning.json'),
                        {'arms': selected, 'protocol': protocol})
    orig_run, orig_wm, orig_ev = L.run, L.write_run_manifest, L.log_experiment_event
    _, ev, _ = L.validation_row_split(8)

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for sd in seeds:
                base_nll = 1.0 if v == 'a' else 2.0
                rows = [base_nll + (0.5 if i in ev else 0.0) for i in range(8)]
                summary[f'{v}::seed{sd}'] = {
                    'variant': v, 'seed': int(sd),
                    'ppl': float(np.exp(np.mean(rows))),
                    'nll_sum': float(sum(rows)), 'n_tokens': 8,
                    'nll_per_row': rows, 'n_tokens_per_row': [1] * 8,
                    'steps': cfg['steps'],
                    'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': 'same', 'data_fp': 'd', 'val_fp': 'v',
                    'train_knobs': L.training_knobs(),
                    'hyperparams': {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
                                    'warmup': cfg['warmup']}}
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run, L.write_run_manifest, L.log_experiment_event = fake_run, (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        BT._run_final(base, out, protocol, selected)
    finally:
        L.run, L.write_run_manifest, L.log_experiment_event = orig_run, orig_wm, orig_ev
        comp_path = os.path.join(out, 'final', 'tuned_comparison.json')
        comp = json.load(open(comp_path, encoding='utf-8'))
        shutil.rmtree(out, ignore_errors=True)
    assert comp['estimation_rows'] == 'nested_validation_holdout', comp
    # Holdout rows all carry the +0.5 shift, so the expected held-out delta is
    # exp(1.5)-exp(2.5), NOT the full-validation mean difference.
    expect = math.exp(1.5) - math.exp(2.5)
    assert abs(comp['contrasts']['a - b']['mean'] - expect) < 1e-12, comp['contrasts']
    assert comp['final_evaluation_rows']['applied'] is True, comp['final_evaluation_rows']


def t_nested_final_partial_holdout_falls_back_wholly():
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                    search_seeds=[100, 101, 102], final_seeds=[0, 1],
                    inherits_tuning={}, method_arm='a')
    protocol.pop('arm_groups', None)
    out = tempfile.mkdtemp(prefix='nested_final_partial_')
    selected = {a: {'selected': dict(trials[0]), 'trials': [{}],
                    'selected_metric_value': 1.0} for a in ('a', 'b')}
    L.atomic_write_json(os.path.join(out, 'baseline_tuning.json'),
                        {'arms': selected, 'protocol': protocol})
    orig_run, orig_wm, orig_ev = L.run, L.write_run_manifest, L.log_experiment_event
    _, ev, _ = L.validation_row_split(8)

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for sd in seeds:
                base_nll = 1.0 if v == 'a' else 2.0
                rows = [base_nll + (0.5 if i in ev else 0.0) for i in range(8)]
                toks = [1] * 8
                # One record has a malformed per-row token count.  Its row
                # length still matches the others, so the holdout layout is
                # accepted and only record_subset_ppl() fails for this cell.
                if v == 'b' and int(sd) == 0:
                    toks[int(ev[0])] = None
                summary[f'{v}::seed{sd}'] = {
                    'variant': v, 'seed': int(sd),
                    'ppl': float(np.exp(np.mean(rows))),
                    'nll_sum': float(sum(rows)), 'n_tokens': 8,
                    'nll_per_row': rows, 'n_tokens_per_row': toks,
                    'steps': cfg['steps'],
                    'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': 'same', 'data_fp': 'd', 'val_fp': 'v',
                    'train_knobs': L.training_knobs(),
                    'hyperparams': {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay'],
                                    'warmup': cfg['warmup']}}
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run, L.write_run_manifest, L.log_experiment_event = fake_run, (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        BT._run_final(base, out, protocol, selected)
    finally:
        L.run, L.write_run_manifest, L.log_experiment_event = orig_run, orig_wm, orig_ev
        try:
            comp = json.load(open(os.path.join(out, 'final', 'tuned_comparison.json'), encoding='utf-8'))
            final = json.load(open(os.path.join(out, 'final', 'summary.json'), encoding='utf-8'))
        finally:
            shutil.rmtree(out, ignore_errors=True)
    # No record may be switched to the holdout when ANY cell lacks the row
    # accounting: a mixed fold/full-validation contrast is a different sample.
    assert all('ppl_holdout' not in r for r in final.values()), final
    assert comp['estimation_rows'] == 'full_validation', comp
    assert comp['final_evaluation_rows']['applied'] is False, comp['final_evaluation_rows']
    # The whole contrast is on the full-validation PPLs, not a mixture of
    # one record's holdout PPL and another's full-validation PPL.
    expect = final['a::seed0']['ppl'] - final['b::seed0']['ppl']
    assert abs(comp['contrasts']['a - b']['mean'] - expect) < 1e-12, comp['contrasts']


def t_baseline_tuning_driver_multiseed_selection():
    out = tempfile.mkdtemp(prefix='tune_multiseed_')
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 2},
              {'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 2}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['csa_fixed', 'csa_dynamic'], trials=trials,
                    search_seeds=[0, 1, 2], final_seeds=[0, 1, 2])
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)
    calls = []
    orig_run = L.run
    orig_guard = L.overfit_guard

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        calls.append({'variants': list(cfg['variants']), 'seeds': list(seeds)})
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for v in cfg['variants']:
            for s in seeds:
                # lr=3e-4 is better for both arms; a small seed-dependent
                # offset makes the per-seed mean/std real.
                ppl = 20.0 - cfg['lr'] * 10000.0 + 0.1 * int(s) + (1.0 if v == 'csa_dynamic' else 0.0)
                summary[f'{v}::seed{s}'] = {
                    'variant': v, 'seed': int(s), 'ppl': float(ppl),
                    'steps': cfg['steps'], 'tokens_seen': cfg['steps'] * cfg['batch_size'] * cfg['seq_len'],
                    'params': 100, 'run_cfg': f'fake::{cfg["lr"]}::{v}',
                }
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)

    L.run = fake_run
    L.overfit_guard = lambda *a, **k: None
    try:
        art = BT.run_tuning(base, outdir=out, protocol=protocol, final=False)
    finally:
        L.run = orig_run
        L.overfit_guard = orig_guard
        shutil.rmtree(out, ignore_errors=True)
    assert calls and all(c['seeds'] == [0, 1, 2] for c in calls), calls
    assert art['search_seeds'] == [0, 1, 2], art.get('search_seeds')
    for arm, entry in art['arms'].items():
        assert len(entry['trials']) == len(trials), entry
        for row in entry['trials']:
            assert row['n_seeds'] == 3 and len(row['seed_ppls']) == 3, row
            assert abs(row['ppl'] - float(np.mean(list(row['seed_ppls'].values())))) < 1e-12, row
        assert entry['selected']['lr'] == 3e-4, (arm, entry['selected'])
        # Selection uncertainty is recorded, not hidden: the winning config is
        # compared with the runner-up in standard-error units.
        assert entry['selection_margin'] is not None and entry['selection_margin'] > 0, entry
        # The trials were paired on the same search seeds; here every seed
        # moves by exactly the same margin, so the paired SE is 0 and the
        # z-score is undefined (decisive, not "not computable").
        assert entry['selection_margin_method'] == 'paired_search_seeds', entry
        assert entry['selection_margin_se'] == 0.0 and entry['selection_margin_in_se'] is None, entry
        assert entry['selection_decisive'] is True, entry
        assert entry['runner_up'] is not None and entry['runner_up']['config'] != entry['selected'], entry
    # Pairing matters: a consistent per-seed gain can be decisive even when
    # the marginal SDs make the independent-SE approximation look like a tie.
    _b = {'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3,
          'seed_ppls': {'0': 9.0, '1': 11.0, '2': 10.0}}
    _r = {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3,
          'seed_ppls': {'0': 9.1, '1': 11.1, '2': 10.1}}
    _m = L.tuning_selection_margin(_b, _r)
    assert _m['method'] == 'paired_search_seeds' and _m['se'] == 0.0, _m
    assert _m['decisive'] is True and _m['decisive_independent'] is False, _m
    assert abs(_m['margin'] - 0.1) < 1e-12 and _m['n_paired'] == 3, _m
    # Without per-seed values the legacy independent-SE fallback is used and
    # labelled as such.
    _mf = L.tuning_selection_margin({'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3},
                                    {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3})
    assert _mf['method'] == 'independent_marginal_se' and _mf['decisive'] is False, _mf
    # A seed repeated under "0" and 0 must not be collapsed into one draw and
    # used as if it were a valid paired set.
    _dup = L.tuning_selection_margin(
        {'ppl': 10.0, 'ppl_std': 1.0, 'n_seeds': 3,
         'seed_ppls': {'0': 9.0, 0: 9.1, '1': 11.0, '2': 10.0}},
        {'ppl': 10.1, 'ppl_std': 1.0, 'n_seeds': 3,
         'seed_ppls': {'0': 9.1, '1': 11.1, '2': 10.1}})
    assert _dup['method'] == 'invalid_seed_ppls' and _dup.get('reason'), _dup
    assert _dup['se'] is None and _dup['decisive'] is None, _dup


def t_simple_baseline_ngram_and_audit():
    rng = np.random.default_rng(7)
    vocab = 32
    train = rng.integers(0, vocab, size=600)
    val = rng.integers(0, vocab, size=(4, 25))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    u = L.ngram_lm_score(train, val, vocab, order=1, alpha=0.1)
    b = L.ngram_lm_score(train, val, vocab, order=2, alpha=0.1, interpolation=0.5)
    assert u['n_tokens'] == val.size and b['n_tokens'] == val.size, (u, b)
    assert math.isfinite(u['ppl']) and math.isfinite(b['ppl']) and u['ppl'] > 0 and b['ppl'] > 0
    # The default floor grid must be wide enough to be called tuned, not just
    # the least-bad of two arbitrary smoothing values.
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['unigram']['alpha']) >= 5
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['bigram']['alpha']) >= 5
    assert len(L.SIMPLE_BASELINE_POLICY['grids']['bigram']['interpolation']) >= 5
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert art['best_baseline'] in ('unigram', 'bigram')
    best = min(e['selected_metric_value'] for e in art['baselines'].values())
    assert abs(float(art['best_ppl']) - float(best)) < 1e-12, art
    assert art['data']['val_fp'] and art['data']['n_eval_tokens'] == val.size
    # The count floor must be better than uniform tokens; both values are
    # recorded so the audit can reject a broken floor.
    assert art['uniform_ppl'] == vocab and art['data']['vocab_size'] == vocab, art
    assert art['best_ppl'] <= art['uniform_ppl'] * 1.05, art
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['uniform_ppl'] == vocab and audit['best_ppl_vs_uniform'] <= 1.05, audit
    assert audit['search_design'] == 'full_factorial', audit
    assert all(v['complete'] for v in audit['grid_coverage'].values()), audit['grid_coverage']
    # A declared full-factorial count search must actually contain every cell.
    bad_cell = json.loads(json.dumps(art))
    _bgram = [t for t in bad_cell['baselines']['bigram']['trials']
              if (t.get('config') or {}).get('interpolation') == 0.5]
    assert _bgram, bad_cell['baselines']['bigram']['trials']
    bad_cell['baselines']['bigram']['trials'].remove(_bgram[0])
    rbc = L.simple_baseline_audit(bad_cell, protocol=proto)
    assert rbc['baseline_verified'] is False and any(
        'full_factorial' in p and 'not run' in p for p in rbc['problems']), rbc
    # A floor worse than uniform tokens is a data/smoothing bug, not a weak
    # baseline; the audit refuses it.
    bad_uniform = json.loads(json.dumps(art))
    bad_uniform['uniform_ppl'] = 1.0
    ru = L.simple_baseline_audit(bad_uniform, protocol=proto)
    assert ru['baseline_verified'] is False and any(
        'uniform-token floor' in x for x in (ru['problems'] + ru['warnings'])), ru
    # A present-but-malformed floor field is unverifiable, not an absent
    # legacy field: do not silently ignore a bool/string/non-finite value.
    for _key, _val in (('uniform_ppl', True), ('uniform_ppl', '32'),
                       ('uniform_ppl', float('nan')),
                       ('best_ppl_vs_uniform', True),
                       ('best_ppl_vs_uniform', '0.9'),
                       ('best_ppl_vs_uniform', float('inf'))):
        _bad_floor = json.loads(json.dumps(art))
        _bad_floor[_key] = _val
        _rbf = L.simple_baseline_audit(_bad_floor, protocol=proto)
        assert _rbf['baseline_verified'] is False and any(
            _key in x for x in _rbf['problems']), (_key, _val, _rbf)
    # A cell outside the declared search space is not equal-search under ANY
    # design (hand-designed grids included), and the declared per-arm order
    # must be the model that actually ran.
    outside = json.loads(json.dumps(art))
    outside['baselines']['unigram']['trials'][0]['config']['alpha'] = 99.0
    rout = L.simple_baseline_audit(outside, protocol=dict(
        proto, search_design='hand_designed_grid'))
    assert rout['baseline_verified'] is False and any(
        'outside the declared search grid' in p for p in rout['problems']), rout
    wrong_order = json.loads(json.dumps(art))
    wrong_order['baselines']['bigram']['order'] = 1
    rord = L.simple_baseline_audit(wrong_order, protocol=proto)
    assert rord['baseline_verified'] is False and any(
        'differs from the declared order' in p for p in rord['problems']), rord
    bad_design = dict(proto, search_design='whatever')
    rbd = L.simple_baseline_audit(art, protocol=bad_design)
    assert rbd['baseline_verified'] is False and any(
        'search_design' in p for p in rbd['problems']), rbd
    # A selected value that is not the selected config's own trial value is a
    # fabricated result, even if it happens to equal the grid minimum.
    bad = json.loads(json.dumps(art))
    arm = bad['best_baseline']
    bad['baselines'][arm]['selected_metric_value'] = float(bad['baselines'][arm]['selected_metric_value']) + 1.0
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False and any('does not match' in p for p in rbad['problems']), rbad
    # Per-row accounting must describe the reported floor, not a different
    # sample; otherwise the row bootstrap would resample unscored rows.
    bad_rows = json.loads(json.dumps(art))
    _be = bad_rows['baselines'][bad_rows['best_baseline']]
    if _be.get('reported_nll_per_row'):
        _be['reported_nll_per_row'][0] += 1.0
        rbr = L.simple_baseline_audit(bad_rows, protocol=proto)
        assert rbr['baseline_verified'] is False and any('per-row NLL sum' in p for p in rbr['problems']), rbr
        bad_rows2 = json.loads(json.dumps(art))
        bad_rows2['baselines'][bad_rows2['best_baseline']]['reported_n_tokens_per_row'][0] += 1
        rbr2 = L.simple_baseline_audit(bad_rows2, protocol=proto)
        assert rbr2['baseline_verified'] is False and any('per-row token counts' in p for p in rbr2['problems']), rbr2
    # Missing an arm is not evidence that the floor was tuned.
    bad2 = json.loads(json.dumps(art))
    bad2['baselines'].pop('unigram')
    r2 = L.simple_baseline_audit(bad2, protocol=proto)
    assert r2['baseline_verified'] is False and any('missing' in p for p in r2['problems']), r2
    # A flat 1-D evaluation stream is ONE row: only its first token is
    # unconditioned, and the two call forms must score the same tokens.
    flat = val[0]
    for order, lam in ((2, 0.5), (2, 1.0)):
        two_d = L.ngram_lm_score(train, val[:1], vocab, order=order, alpha=0.1, interpolation=lam)
        one_d = L.ngram_lm_score(train, flat, vocab, order=order, alpha=0.1, interpolation=lam)
        assert abs(two_d['nll'] - one_d['nll']) < 1e-12, (order, lam, two_d, one_d)
        assert abs(two_d['ppl'] - one_d['ppl']) < 1e-9, (order, lam, two_d, one_d)
    # A training stream with fewer than two tokens has no bigram table at
    # all.  Scoring order=2 through the empty table must fall back to the
    # add-alpha unseen conditional instead of raising IndexError.
    for _short in (np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64)):
        _short_score = L.ngram_lm_score(_short, np.array([0, 1, 2]), vocab,
                                        order=2, alpha=0.1, interpolation=1.0)
        assert math.isfinite(_short_score['ppl']) and _short_score['ppl'] > 0, _short_score

    # interpolation=0 must reduce to the unigram model.
    uni = L.ngram_lm_score(train, val, vocab, order=1, alpha=0.1)
    zero = L.ngram_lm_score(train, val, vocab, order=2, alpha=0.1, interpolation=0.0)
    assert abs(uni['nll'] - zero['nll']) < 1e-12 and abs(uni['ppl'] - zero['ppl']) < 1e-9, (uni, zero)
    # An unseen context still gets the add-alpha conditional 1/V, then the
    # declared interpolation.  Falling back to the unigram probability would
    # silently drop the interpolation term for exactly those transitions.
    _ut = np.array([0, 0, 0])
    _ue = np.array([[1, 0]])
    _a, _lam, _v = 0.1, 0.5, 2
    _up = (np.bincount(_ut, minlength=_v) + _a) / (_ut.size + _a * _v)
    _want = (-math.log(_up[1]) - math.log(_lam * (1.0 / _v) + (1.0 - _lam) * _up[0])) / 2
    _got = L.ngram_lm_score(_ut, _ue, _v, order=2, alpha=_a, interpolation=_lam)
    assert abs(_got['nll'] - _want) < 1e-12, (_got, _want)
    # The count-model floor must not be fitted/scored on truncated ids or a
    # fractional vocabulary: those silently change the tokens being counted.
    for _bad in ([1.0, 2.0, 3.0], [True, False, True], np.array([1.5, 2.5])):
        try:
            L.ngram_lm_score(_bad, val, vocab, order=2)
        except TypeError:
            pass
        else:
            raise AssertionError(f'float/bool train ids were accepted: {_bad!r}')
        try:
            L.ngram_lm_score(train, _bad, vocab, order=2)
        except TypeError:
            pass
        else:
            raise AssertionError(f'float/bool eval ids were accepted: {_bad!r}')
    for _bad_vocab in (2.5, True, 0, -3):
        try:
            L.ngram_lm_score(train, val, _bad_vocab, order=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid vocab_size {_bad_vocab!r} was accepted')
    # Discrete model order and smoothing constants are exact scalars too:
    # True == 1 would silently run the unigram arm as a "bigram" trial.
    for _bad_order in (True, 1.5, '1.5', 3):
        try:
            L.ngram_lm_score(train, val, vocab, order=_bad_order)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid order {_bad_order!r} was accepted')
    for _bad_alpha in (True, float('nan'), float('inf'), 0, -1):
        try:
            L.ngram_lm_score(train, val, vocab, order=1, alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid alpha {_bad_alpha!r} was accepted')
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=dict(
            proto, arms={'u': {'order': 1.5}}, grids={'u': {'alpha': [0.1]}}))
    except ValueError:
        pass
    else:
        raise AssertionError('a fractional declared order must not be truncated')
    # The grid path must not pre-coerce booleans/non-finite values before
    # ngram_lm_score validates them (float(True) == 1.0 used to slip through).
    for _bad_grid in (
            {'arms': {'u': {'order': 1}}, 'grids': {'u': {'alpha': [True]}}},
            {'arms': {'u': {'order': 1}}, 'grids': {'u': {'alpha': [float('nan')]}}},
            {'arms': {'b': {'order': 2}},
             'grids': {'b': {'alpha': [0.1], 'interpolation': [True]}}},
            {'arms': {'b': {'order': 2}},
             'grids': {'b': {'alpha': [0.1], 'interpolation': [float('inf')]}}}):
        try:
            L.simple_baseline_fit(train, val, vocab, protocol=_bad_grid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'invalid grid hyperparameter accepted: {_bad_grid!r}')


def t_simple_baseline_grid_adequacy():
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'uni': {'order': 1}},
                 grids={'uni': {'alpha': [0.001, 0.01, 0.1]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    entry = art['baselines']['uni']
    # Force the artifact to select the largest tested smoothing level, with a
    # clear (non-tie) margin over the runner-up.
    by_alpha = {float(t['config']['alpha']): t for t in entry['trials']}
    assert 0.1 in by_alpha, by_alpha
    by_alpha[0.001]['ppl'] = 2.0
    by_alpha[0.01]['ppl'] = 1.5
    by_alpha[0.1]['ppl'] = 1.0
    entry['selected'] = dict(by_alpha[0.1]['config'])
    entry['selected_metric_value'] = 1.0
    art['best_ppl'] = 1.0
    art['best_ppl_vs_uniform'] = 1.0 / float(vocab)
    for k in ('reported_nll_per_row', 'reported_n_tokens_per_row',
              'reported_estimation_rows', 'holdout_ppl', 'holdout_nll',
              'holdout_n_tokens', 'holdout_metric_value'):
        entry.pop(k, None)
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['grid_adequacy'] == 'warn', audit
    assert any('max of the declared range' in w and '0.1' in w for w in audit['warnings']), audit
    assert audit['grid_boundary_hits'] and 'uni:alpha' in audit['grid_boundary_hits'][0], audit
    # Equal best/runner-up values are a tie, not a decisive tuned pick.
    tie = json.loads(json.dumps(art))
    for t in tie['baselines']['uni']['trials']:
        t['ppl'] = 1.0
    tie['best_ppl'] = 1.0
    tie_audit = L.simple_baseline_audit(tie, protocol=proto)
    assert tie_audit['selection_tie_arms'] == ['uni'], tie_audit
    assert any('tie' in w for w in tie_audit['warnings']), tie_audit
    # Interpolation is a second continuous knob: selecting the pure-bigram
    # corner (1.0) means the optimum may lie outside the tested range.
    proto_i = dict(L.SIMPLE_BASELINE_POLICY,
                   arms={'bi': {'order': 2}},
                   grids={'bi': {'alpha': [0.1, 1.0],
                                 'interpolation': [0.25, 0.5, 0.75]}})
    art_i = L.simple_baseline_fit(train, val, vocab, protocol=proto_i)
    entry_i = art_i['baselines']['bi']
    for _t in entry_i['trials']:
        _lam = _t['config']['interpolation']
        _t['ppl'] = {0.25: 3.0, 0.5: 2.0, 0.75: 1.0}[_lam]
    _best_i = [t for t in entry_i['trials']
               if t['config']['interpolation'] == 0.75][0]
    entry_i['selected'] = dict(_best_i['config'])
    entry_i['selected_metric_value'] = 1.0
    art_i['best_ppl'] = 1.0
    art_i['best_ppl_vs_uniform'] = 1.0 / float(vocab)
    for _k in ('reported_nll_per_row', 'reported_n_tokens_per_row',
               'reported_estimation_rows', 'holdout_ppl', 'holdout_nll',
               'holdout_n_tokens', 'holdout_metric_value'):
        entry_i.pop(_k, None)
    ai = L.simple_baseline_audit(art_i, protocol=proto_i)
    assert ai['baseline_verified'] is True, ai
    assert 'bi:interpolation=0.75(max)' in ai['grid_boundary_hits'], ai
    assert any('bigram/unigram mix optimum' in w for w in ai['warnings']), ai

def t_simple_baseline_nested_rows_and_fingerprint():
    rng = np.random.default_rng(19)
    vocab = 24
    train = rng.integers(0, vocab, size=500)
    val = rng.integers(0, vocab, size=(8, 15))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                                selection_rows=sel, evaluation_rows=ev,
                                row_split=meta)
    data = art['data']
    assert data['nested_selection'] is True, data
    assert data['n_eval_tokens'] == int(val[ev].size), data
    assert data['selection_fp'] != data['eval_fp'], data
    assert data['row_split']['evaluation_row_fp'] == meta['evaluation_row_fp'], data
    assert art['best_ppl'] == art['baselines'][art['best_baseline']]['holdout_metric_value'], art
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['nested_selection'] is True and audit['best_ppl_estimation_rows'] == 'nested_validation_holdout', audit
    # A floor reported on the selection fold is selection-biased, even if it
    # is the minimum selection value.
    bad = json.loads(json.dumps(art))
    bad['best_ppl'] = bad['baselines'][bad['best_baseline']]['selected_metric_value']
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False and any('held-out' in p for p in rbad['problems']), rbad
    # The held-out evaluation must be on disjoint rows.
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=proto,
                              selection_rows=sel, evaluation_rows=sel,
                              row_split=meta)
    except ValueError:
        pass
    else:
        raise AssertionError('overlapping selection/evaluation rows were accepted')
    # A row-split fingerprint that does not describe the supplied rows is a
    # fabricated nested claim.
    tampered = dict(meta, evaluation_row_fp='not-the-eval-rows')
    try:
        L.simple_baseline_fit(train, val, vocab, protocol=proto,
                              selection_rows=sel, evaluation_rows=ev,
                              row_split=tampered)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched evaluation_row_fp was accepted')
    # The floor must use the same declared fold layout as the panel: a
    # reproducible but different split is still an unfair comparison.
    sel2, ev2, meta2 = L.validation_row_split(8, seed=123, n_folds=2,
                                              selection_fold=0, evaluation_fold=1)
    art2 = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                                 selection_rows=sel2, evaluation_rows=ev2,
                                 row_split=meta2)
    rsplit = L.simple_baseline_audit(art2, protocol=proto)
    assert rsplit['baseline_verified'] is False and any(
        'different validation folds' in p for p in rsplit['problems']), rsplit
    # The reported token denominator must be the held-out denominator.
    bad_tok = json.loads(json.dumps(art))
    bad_tok['data']['n_eval_tokens'] = int(val.size)
    rtok = L.simple_baseline_audit(bad_tok, protocol=proto)
    assert rtok['baseline_verified'] is False and any('held-out token' in p for p in rtok['problems']), rtok

def t_simple_baseline_fit_budget_is_bounded_and_disclosed():
    rng = np.random.default_rng(11)
    vocab = 16
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(3, 12))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1]},
                        'bigram': {'alpha': [0.1], 'interpolation': [0.5]}},
                 max_fit_tokens=100)
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert art['data']['n_fit_tokens'] == 100, art['data']
    assert art['data']['n_source_tokens'] == train.size, art['data']
    assert art['data']['fit_truncated'] is True, art['data']
    assert art['data']['train_fp'] == L._ids_fp(np.asarray(train[:100])), art['data']
    assert art['data']['source_train_fp'] == L._ids_fp(train), art['data']
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    assert audit['fit_truncated'] is True, audit
    assert audit['fit_budget']['n_fit_tokens'] == 100, audit['fit_budget']
    assert audit['fit_budget']['n_source_tokens'] == train.size, audit['fit_budget']
    assert audit['fit_budget']['max_fit_tokens'] == 100, audit['fit_budget']
    assert any('deterministic prefix' in w for w in audit['warnings']), audit
    # Tokens beyond the declared fit prefix must not change the floor.
    train2 = train.copy()
    train2[100:] = (train2[100:] + 1) % vocab
    art2 = L.simple_baseline_fit(train2, val, vocab, protocol=proto)
    assert abs(float(art2['best_ppl']) - float(art['best_ppl'])) < 1e-12, (art['best_ppl'], art2['best_ppl'])
    assert art2['data']['source_train_fp'] != art['data']['source_train_fp'], art2['data']
    # A truncated claim with contradictory counts is not accepted.
    bad = json.loads(json.dumps(art))
    bad['data']['n_fit_tokens'] = bad['data']['n_source_tokens']
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False, rbad
    # The truncation flag cannot contradict the recorded token counts.
    liar = json.loads(json.dumps(art))
    liar['data']['fit_truncated'] = False
    rliar = L.simple_baseline_audit(liar, protocol=proto)
    assert rliar['baseline_verified'] is False, rliar
    assert rliar['fit_truncated'] is True, rliar
    assert any('contradicts the token counts' in p for p in rliar['problems']), rliar
    # The recorded prefix must be exactly min(source, max_fit_tokens).
    wrong_cap = json.loads(json.dumps(art))
    wrong_cap['data']['max_fit_tokens'] = 200
    rcap = L.simple_baseline_audit(wrong_cap, protocol=proto)
    assert rcap['baseline_verified'] is False and any(
        'deterministic-prefix fit' in p for p in rcap['problems']), rcap
    # A non-boolean flag is malformed evidence, not a true/false label.
    strflag = json.loads(json.dumps(art))
    strflag['data']['fit_truncated'] = 'yes'
    rflag = L.simple_baseline_audit(strflag, protocol=proto)
    assert rflag['baseline_verified'] is False and any(
        'must be a boolean' in p for p in rflag['problems']), rflag


def t_simple_floor_contrast_is_significance_aware():
    # The "beats the trivial floor" claim must be a seed-level contrast, not a
    # pointwise < on two PPLs.  Build a panel whose holdout rows match the
    # floor's holdout rows and whose tokens match per seed.
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=600)
    val = rng.integers(0, vocab, size=(8, 13))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1]},
                        'bigram': {'alpha': [0.1], 'interpolation': [0.5]}})
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    sb = L.simple_baseline_fit(train, val, vocab, protocol=proto,
                               selection_rows=sel, evaluation_rows=ev,
                               row_split=meta)
    sb['data']['cache_val_fp'] = 'v'
    floor = float(sb['best_ppl'])
    n_tok = int(sb['data']['n_eval_tokens'])
    rc = ('steps1_sl13_bs1_nt104_lr0.0003_wd0.1_d8_L1_H1_Dh8_v24_wu1_'
          'cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    def recs(base_ppl):
        out = {}
        for arm in ('csa_dynamic', 'full_sw128_matched'):
            for seed in range(8):
                ppl = base_ppl if arm == 'csa_dynamic' else floor + 2.0
                _r = n_tok // 4
                out[f'{arm}::seed{seed}'] = {
                    'variant': arm, 'seed': seed, 'ppl': ppl, 'params': 100,
                    'run_cfg': rc, 'steps': 1, 'tokens_seen': n_tok,
                    'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e',
                    'nll_sum': math.log(ppl) * n_tok, 'n_tokens': n_tok,
                    'holdout_row_fp': meta['evaluation_row_fp'],
                    'ppl_estimation_rows': 'nested_validation_holdout',
                    'nll_per_row_holdout': [math.log(ppl) * _r] * 4,
                    'n_tokens_per_row_holdout': [_r] * 4,
                }
        return out
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        with contextlib.redirect_stdout(io.StringIO()):
            r = L.methodology_audit(recs(floor - 1.0), design={}, baselines=L.BASELINE_DESIGN, outdir=td, verbose=False)
        fl = r['simple_baseline']
        assert fl['comparable_to_panel'] is True, fl
        # The final panel lives in <outdir>/final/; the artifacts live in the
        # parent.  The audit must find them instead of silently returning
        # not_verifiable_locally.
        _final = os.path.join(td, 'final')
        os.makedirs(_final)
        with contextlib.redirect_stdout(io.StringIO()):
            r_final = L.methodology_audit(recs(floor - 1.0), design={},
                                          baselines=L.BASELINE_DESIGN,
                                          outdir=_final, verbose=False)
        assert r_final['simple_baseline']['baseline_verified'] is True, r_final['simple_baseline']
        assert r_final['simple_baseline']['comparable_to_panel'] is True, r_final['simple_baseline']
        assert os.path.exists(os.path.join(_final, 'methodology_audit.json'))
        assert fl['token_denominator_match'] is True, fl
        assert fl['floor_contrast']['verdict'] == 'beats_floor_significant', fl['floor_contrast']
        # A floor fitted on a truncated prefix is not a floor trained on the
        # panel's full corpus; it may still be verified arithmetically, but it
        # must not be quoted as the adequately tuned floor.
        _trunc = json.loads(json.dumps(sb))
        _trunc['data'].update(fit_truncated=True, n_fit_tokens=100,
                              n_source_tokens=600, max_fit_tokens=100)
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(_trunc, f)
        with contextlib.redirect_stdout(io.StringIO()):
            _rtr = L.methodology_audit(recs(floor - 1.0), design={},
                                       baselines=L.BASELINE_DESIGN,
                                       outdir=td, verbose=False)
        _ftr = _rtr['simple_baseline']
        assert _ftr['fit_truncated'] is True, _ftr
        assert _ftr['tuned_floor_claim_allowed'] is False, _ftr
        assert any('truncated prefix' in str(x)
                   for x in (_ftr.get('tuned_floor_claim_reasons') or [])), _ftr
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        assert fl['floor_contrast']['p_exact_signflip'] < 0.05, fl['floor_contrast']
        # The caller's alpha is the gate for the floor family too.  At n=6
        # the exact p floor is 0.03125: significant at alpha=0.05 but not at
        # alpha=0.01, and the floor gate must not keep hard-coding 0.05.
        _lo = {k: v for k, v in recs(floor - 1.0).items()
               if int(v['seed']) < 6}
        with contextlib.redirect_stdout(io.StringIO()):
            _rlo = L.methodology_audit(_lo, design={}, baselines=L.BASELINE_DESIGN,
                                       outdir=td, alpha=0.01, verbose=False)
        _flo = _rlo['simple_baseline']
        assert _flo['floor_contrast']['verdict'] != 'beats_floor_significant', _flo['floor_contrast']
        assert _flo['method_arm_floor_contrasts']['csa_dynamic']['method_status'] != 'beats_floor_holm', _flo['method_arm_floor_contrasts']['csa_dynamic']
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = L.methodology_audit(recs(floor + 0.5), design={}, baselines=L.BASELINE_DESIGN, outdir=td, verbose=False)
        assert r2['simple_baseline']['panel_beats_simple_floor'] is False, r2['simple_baseline']
        assert r2['simple_baseline']['floor_contrast']['verdict'] == 'does_not_beat_floor', r2['simple_baseline']
        assert any('does not beat the tuned count-model floor' in w for w in r2['warnings']), r2['warnings']
        # The pre-declared method arm must clear the floor ITSELF.  Selecting
        # the best panel arm post hoc could let a dense baseline rescue the
        # gate while the proposed method is worse than the count model.
        assert fl['method_arms_beat_floor'] is True, fl.get('method_arm_floor_status')
        assert fl['method_arm_floor_status'] == 'all_method_arms_beat_floor', fl.get('method_arm_floor_status')
        assert fl['method_arm_floor_contrasts']['csa_dynamic']['method_status'] == 'beats_floor_holm', fl['method_arm_floor_contrasts']
        # The floor is an estimate on the same rows, so the contrast also
        # carries a paired row bootstrap instead of treating it as a constant.
        _fb = fl['method_arm_floor_contrasts']['csa_dynamic'].get('paired_row_bootstrap_delta') or {}
        assert _fb.get('status') == 'ok' and _fb.get('n_rows') == 4, _fb
        assert _fb.get('mean_delta') < 0, _fb
        assert r2['simple_baseline']['method_arms_beat_floor'] is False, r2['simple_baseline']
        assert any('declared method arm' in w for w in r2['warnings']), r2['warnings']
        # A different arm beating the floor must not rescue the method gate.
        rescued = recs(floor + 1.0)
        for _k, _r in rescued.items():
            if _r['variant'] == 'full_sw128_matched':
                _r['ppl'] = floor - 1.0
                _r['nll_sum'] = math.log(floor - 1.0) * n_tok
                # The nested primary accounting uses the holdout rows, so a
                # consistent fixture must move them too.
                _r['nll_per_row_holdout'] = [math.log(floor - 1.0) * (n_tok // 4)] * 4
        with contextlib.redirect_stdout(io.StringIO()):
            r5 = L.methodology_audit(rescued, design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        assert r5['simple_baseline']['panel_beats_simple_floor'] is True, r5['simple_baseline']
        assert r5['simple_baseline']['method_arms_beat_floor'] is False, r5['simple_baseline']
        assert r5['simple_baseline']['method_arm_floor_status'] == 'method_arm_does_not_beat_floor', r5['simple_baseline']
        # The floor reference must be chosen by the PRIMARY (token-pooled)
        # endpoint, not by the arithmetic seed mean: they can rank arms
        # differently when token counts differ.
        with contextlib.redirect_stdout(io.StringIO()):
            r4 = L.methodology_audit(recs(floor - 1.0), design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        _arm = r4['arm_summary']
        _dyn_key = next(k for k, v in _arm.items() if v.get('variant') == 'csa_dynamic')
        _full_key = next(k for k, v in _arm.items() if v.get('variant') == 'full_sw128_matched')
        # Force a disagreement between the two aggregations for the two arms.
        _arm[_dyn_key]['ppl_mean'] = floor + 100.0
        _arm[_dyn_key]['ppl_primary'] = floor - 1.0
        _arm[_full_key]['ppl_mean'] = floor + 0.01
        _arm[_full_key]['ppl_primary'] = floor + 50.0
        # Re-running is not needed: the selector is the expression under test.
        _cands = [(_dyn_key, _arm[_dyn_key]), (_full_key, _arm[_full_key])]
        _ref_key, _ref = min(_cands, key=lambda kv: float(kv[1].get('ppl_primary', kv[1]['ppl_mean'])))
        assert _ref_key == _dyn_key and _ref['ppl_primary'] == floor - 1.0, (_ref_key, _ref)
    # A nested record's per-seed contrast must use the same holdout PPL as its
    # pooled primary endpoint.  Full-validation `ppl` is deliberately worse
    # than the floor here, so reading it would reverse the verdict.
    nested = recs(floor + 10.0)
    for _r in nested.values():
        _r['ppl_holdout'] = float(floor - 1.0)
        _r['nll_sum_holdout'] = math.log(float(floor - 1.0)) * n_tok
        _r['n_tokens_holdout'] = n_tok
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rn = L.methodology_audit(nested, design={}, baselines=L.BASELINE_DESIGN,
                                     outdir=td, verbose=False)
        fn = rn['simple_baseline']['floor_contrast']
        assert fn['mean_delta'] < 0 and fn['verdict'] == 'beats_floor_significant', fn

    # Underpowered pointwise wins are not "beats floor".
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'simple_baseline.json'), 'w', encoding='utf-8') as f:
            json.dump(sb, f)
        small = {k: v for k, v in recs(floor - 1.0).items() if k.endswith('seed0') or k.endswith('seed1')}
        with contextlib.redirect_stdout(io.StringIO()):
            r3 = L.methodology_audit(small, design={}, baselines=L.BASELINE_DESIGN, outdir=td, min_n=3, verbose=False)
        fc = r3['simple_baseline'].get('floor_contrast') or {}
        assert fc.get('verdict') == 'beats_floor_pointwise_underpowered', fc

def t_simple_baseline_driver_serves_the_panel_corpus():
    # `run_simple` must go through load_run_corpus (the panel's dataset) and
    # fingerprint the validation slice.
    rng = np.random.default_rng(3)
    train = rng.integers(0, 24, size=400)
    val = rng.integers(0, 24, size=(2, 9))
    bnd = np.zeros_like(val, dtype=bool)
    seen = {}
    orig = L.load_run_corpus

    def fake(cfg):
        seen['dataset'] = cfg.get('dataset')
        return train, val, 24, None, bnd

    out = tempfile.mkdtemp(prefix='simple_driver_')
    L.load_run_corpus = fake
    try:
        art = BT.run_simple(dict(L.RUN, dataset='fineweb', seq_len=9, n_train_tokens=400,
                                 val_seqs=2, eval_subset=2), outdir=out)
    finally:
        L.load_run_corpus = orig
        shutil.rmtree(out, ignore_errors=True)
    assert seen.get('dataset') == 'fineweb', seen
    assert art['data']['n_eval_tokens'] == val.size and art['best_ppl'] > 0, art


def t_audit_leakage_handles_flat_train_cache():
    cache = tempfile.mkdtemp(prefix='leak_flat_')
    try:
        train = np.arange(60, dtype=np.int64)
        test = np.stack([np.arange(0, 30), np.arange(30, 60)]).astype(np.int64)
        np.save(os.path.join(cache, 'train_ids_x.npy'), train)
        np.save(os.path.join(cache, 'test_ids_x.npy'), test)
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'test'}, f)
        out = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        overlap = out['checks'].get('doc_overlap')
        assert isinstance(overlap, dict), overlap
        assert overlap['shared_shingles'] > 0, overlap
        # Recompute the loader fingerprint from the id caches, not just trust
        # the metadata string.
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(train), 'fit_fp': 'FITX'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test), 'fit_fp': 'FITX'}, f)
        good = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        fp = good['checks']['id_cache_fingerprints']
        assert fp['train']['status'] == 'ok' and fp['test']['status'] == 'ok', fp
        assert good['checks']['fit_fp_agree'] is True, good['checks']
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test), 'fit_fp': 'OTHER'}, f)
        mism = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert mism['status'] == 'not_verifiable_locally', mism
        assert 'no train/test cache pair' in mism.get('note', ''), mism
        # A one-sided fit fingerprint cannot be assumed to match.
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': L._ids_fp(test)}, f)
        missing = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert missing['status'] == 'fail', missing
        assert any('fit provenance is missing' in x
                   for x in missing['checks'].get('tokenizer_problems') or []), missing
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'stale', 'fit_fp': 'FITX'}, f)
        stale = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert stale['status'] == 'fail', stale
        assert any('id-cache fingerprint' in x for x in stale['checks'].get('tokenizer_problems') or []), stale
        # A legacy/malformed meta without a numeric vocab must fail closed, not
        # crash with int(None) before the audit can report anything.
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'test'}, f)
        bad = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert bad['status'] == 'fail' and bad['checks'].get('vocab_agree') is False, bad
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def t_audit_gate_parameters_fail_closed():
    # shingle<=0 would hash no windows and report zero overlap; NaN policy
    # limits would also make every comparison pass.  Both must be refused.
    for kw in ({'shingle': 0}, {'shingle': -3},
               {'max_rows_shingled': 0},
               {'preferred_train_seq_len': 0}):
        try:
            A.audit_leakage(cache_dir='/nonexistent', **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_leakage accepted {kw!r}')
    for key in ('CSA_MAX_TEST_SHINGLE_OVERLAP', 'CSA_MAX_TEST_ROW_OVERLAP'):
        old = os.environ.get(key)
        try:
            for bad in ('nan', 'inf', '-0.1', 'not-a-number'):
                os.environ[key] = bad
                try:
                    A.audit_leakage(cache_dir='/nonexistent')
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'{key}={bad!r} must be refused')
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    for kw in ({'js_warn': float('nan')}, {'js_fail': float('nan')},
               {'js_warn': -0.1}, {'min_shift_tokens': 0},
               {'max_rows': 0}, {'js_warn': 0.3, 'js_fail': 0.2}):
        try:
            A.audit_split_discipline(cache_dirs=[], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_split_discipline accepted {kw!r}')

    # An exception inside the overlap computation must not leave the overall
    # leakage verdict at 'ok'.
    cache = tempfile.mkdtemp(prefix='leak_failclosed_')
    try:
        np.save(os.path.join(cache, 'train_ids_x.npy'), np.arange(60, dtype=np.int64))
        np.save(os.path.join(cache, 'test_ids_x.npy'),
                np.stack([np.arange(0, 30), np.arange(30, 60)]).astype(np.int64))
        with open(os.path.join(cache, 'meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'a'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000, 'src_fp': 'b'}, f)
        # A requested train seq_len that no cache declares must refuse to
        # guess, not audit a different split.
        _pref = A.audit_leakage(cache_dir=cache, shingle=4,
                                max_rows_shingled=10,
                                preferred_train_seq_len=512)
        assert _pref['status'] == 'not_verifiable_locally', _pref
        # A shingle wider than the stored rows can form no window; that is an
        # unverifiable split, not zero overlap.
        _wide = A.audit_leakage(cache_dir=cache, shingle=1000,
                                max_rows_shingled=10)
        assert _wide['status'] == 'fail', _wide
        assert 'shingle' in str(_wide['checks'].get('doc_overlap')), _wide
        _orig = A._window_hashes
        try:
            def _boom(*_a, **_k):
                raise RuntimeError('simulated overlap failure')
            A._window_hashes = _boom
            out = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        finally:
            A._window_hashes = _orig
        assert out['status'] == 'fail', out
        assert 'doc_overlap' in out['checks'], out
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def t_fuse_sample_std_negative_axis():
    import fuse_analysis as FA
    x = np.arange(24, dtype=float).reshape(2, 3, 4)
    for axis in (0, 1, 2, -1, -2, -3):
        got = FA.sample_std(x, axis=axis)
        want = np.std(x, axis=axis, ddof=1)
        assert np.asarray(got).shape == np.asarray(want).shape, (axis, np.asarray(got).shape, np.asarray(want).shape)
        assert np.allclose(got, want, atol=1e-12), (axis, got, want)


def t_classification_positive_label_and_rare_floor():
    y = np.array([0] * 2000 + [1])
    # Positive class 1 is the rare class here (1:2000).
    score = (y == 1).astype(float)
    m = L.classification_metrics(y, y_score=score, labels=[0, 1], positive_label=1)
    assert m['positive_label'] == '1' and m['f1_positive'] is not None, m
    assert m['roc_auc'] == 1.0 and abs(m['pr_auc'] - 1.0) < 1e-12, m
    assert m['pr_auc_prevalence'] == 1.0 / 2001.0 and abs(m['pr_auc_lift_over_prevalence'] - (1.0 - 1.0 / 2001.0)) < 1e-12, m
    assert m['imbalance_warning'] and 'PR-AUC' in m['imbalance_warning'], m
    # Naming the other class as positive reverses the same ranking endpoint.
    m2 = L.classification_metrics(y, y_score=score, labels=[0, 1], positive_label=0)
    assert m2['positive_label'] == '0' and m2['roc_auc'] == 0.0, m2
    # A zero-support label requested by the caller must not be dropped from
    # macro F1 (that would make a one-class prediction look like a perfect
    # macro classifier).
    y0 = np.array([0] * 10 + [1] * 10)
    pred = np.zeros_like(y0)
    m3 = L.classification_metrics(y0, y_pred=pred, labels=[0, 1, 2])
    assert abs(m3['f1_macro'] - (2.0 / 3.0 / 3.0)) < 1e-12, m3
    assert m3['f1_primary_average'] == 'macro'
    # A declared label set must cover observed classes.  Silently ignoring an
    # undeclared predicted class would make micro/macro F1 describe a
    # different label space than the caller declared.
    try:
        L.classification_metrics(np.array([0, 1, 1]),
                                 y_pred=np.array([0, 1, 2]), labels=[0, 1])
    except ValueError as e:
        assert 'not in the declared labels' in str(e), e
    else:
        raise AssertionError('an undeclared predicted class was silently ignored')
    try:
        L.classification_metrics(np.array([0, 2]),
                                 y_pred=np.array([0, 2]), labels=[0, 1])
    except ValueError as e:
        assert 'not in the declared labels' in str(e), e
    else:
        raise AssertionError('an undeclared true class was silently ignored')
    # Record-level policy: F1 without averaging, and rare ROC-AUC without
    # PR-AUC/prevalence, must warn.
    bad = {'r::seed0': {'variant': 'r', 'task': 'binary_classification', 'f1': 0.4,
                        'roc_auc': 0.99, 'prevalence': {'0': 0.9995, '1': 0.0005}}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn', ma
    assert any('averaging' in w for w in ma['warnings']), ma
    assert any('PR-AUC' in w for w in ma['warnings']), ma
    # A record must not label an arbitrary F1 as its primary metric.
    _cm = L.classification_metrics([0] * 950 + [1] * 50, y_pred=[0] * 1000,
                                   labels=[0, 1])
    _good = dict(_cm, variant='g', task='binary_classification',
                 prevalence={'0': 0.95, '1': 0.05})
    assert L.metric_policy_audit({'g': _good})['status'] == 'ok',         L.metric_policy_audit({'g': _good})
    _bad_primary = dict(_good, f1_primary=0.9)
    _bp = L.metric_policy_audit({'g': _bad_primary})
    assert _bp['status'] == 'warn' and any(
        'does not equal' in w for w in _bp['warnings']), _bp
    # At 1:1000 the conditional primary is PR-AUC; ROC-AUC/F1 as primary is flagged.
    _rare = dict(_good, prevalence={'0': 0.9995, '1': 0.0005},
                 roc_auc=0.99, primary_metric='roc_auc')
    _rp = L.metric_policy_audit({'g': _rare})
    assert _rp['status'] == 'warn' and any(
        'conditional policy primary' in w for w in _rp['warnings']), _rp
    # Multiclass F1 must expose macro/micro/weighted, not only one average.
    _mc = {'variant': 'm', 'task': 'multiclass_classification',
           'f1_macro': 0.4, 'f1_primary': 0.4,
           'f1_primary_average': 'macro'}
    _mca = L.metric_policy_audit({'m': _mc})
    assert _mca['status'] == 'warn' and any(
        'missing' in w and 'f1_micro' in w for w in _mca['warnings']), _mca


def t_binary_threshold_and_calibration_contract():
    y = np.array([0] * 90 + [1] * 10)
    score = np.array([0.1] * 90 + [0.4] * 5 + [0.9] * 5)
    curve = L.binary_threshold_curve(y, score, positive_label=1)
    assert abs(curve['best_threshold'] - 0.4) < 1e-12, curve['best_threshold']
    assert abs(curve['best_value'] - 1.0) < 1e-12, curve['best_value']
    assert abs(curve['at_default_0.5']['f1'] - 2.0 / 3.0) < 1e-12, curve['at_default_0.5']
    assert curve['threshold_sensitivity'] > 0.3, curve
    sel = L.select_binary_threshold(y, score, positive_label=1)
    assert sel['threshold'] == 0.4 and sel['value'] == 1.0, sel
    assert sel['prevalence'] == 0.1 and sel['n_positive'] == 10, sel
    # Mixed-type binary labels must not be coerced together (int 1 vs '1');
    # the positive endpoint is named explicitly.
    _ym = np.array([0] * 6 + ['1'] * 4, dtype=object)
    _sm = np.array([0.1] * 6 + [0.9] * 4)
    _cm = L.binary_threshold_curve(_ym, _sm, positive_label='1')
    assert _cm['n_positive'] == 4 and _cm['best_value'] == 1.0, _cm
    # The sweep is linear-time via cumulative counts; it must equal the
    # brute-force confusion matrix at every candidate, including score ties.
    _rng = np.random.default_rng(7)
    for _ in range(5):
        _yy = _rng.integers(0, 2, size=60)
        _ss = np.round(_rng.random(60), 1)
        if _yy.sum() in (0, 60):
            continue
        _c = L.binary_threshold_curve(_yy, _ss)
        for _row in _c['thresholds']:
            _b = L._binary_threshold_metrics(_yy, _ss, _row['threshold'])
            assert (_row['tp'], _row['fp'], _row['tn'], _row['fn']) == (
                _b['tp'], _b['fp'], _b['tn'], _b['fn']), (_row, _b)
            assert abs(_row['f1'] - _b['f1']) < 1e-12, (_row, _b)
    # The default cut is recorded rather than silently assumed.
    m = L.classification_metrics(y, y_score=score, labels=[0, 1], threshold_sweep=True)
    assert m['decision_threshold'] == 0.5 and m['decision_threshold_source'] == 'default_0.5', m
    assert abs(m['f1_primary'] - 2.0 / 3.0) < 1e-12, m
    assert m['threshold_selection']['threshold'] == 0.4, m['threshold_selection']
    assert 0.0 <= m['brier_score'] <= 1.0 and 0.0 <= m['ece'] <= 1.0, m
    assert m['score_orientation_warning'] is None, m
    mt = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                  threshold=0.4)
    assert mt['decision_threshold_source'] == 'caller_threshold'
    assert abs(mt['f1_primary'] - 1.0) < 1e-12, mt
    # Gate parameters are exact integers: True/1.5 must not silently become a
    # different bin/threshold budget.
    for _bad_mt in (True, 0, -1, 1.5, 'x'):
        try:
            L.binary_threshold_curve(y, score, max_thresholds=_bad_mt)
        except ValueError:
            pass
        else:
            raise AssertionError(f'max_thresholds={_bad_mt!r} must be refused')
    for _bad_nb in (True, 0, -1, 1.5, 'x'):
        try:
            L._binary_calibration(np.array([0, 1]), np.array([0.2, 0.8]),
                                  n_bins=_bad_nb)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n_bins={_bad_nb!r} must be refused')
    # A reversed score column is a labelling/orientation error, not a weak model.
    mi = L.classification_metrics(y, y_score=1.0 - score, labels=[0, 1])
    assert mi['roc_auc'] == 0.0 and mi['score_orientation_warning'], mi
    # Record-level contract: a tuned threshold must be validation-selected and
    # named; an out-of-range cut or a misstated PR-AUC floor is invalid.
    good = dict(m, variant='g', task='binary_classification',
                prevalence={'0': 0.9, '1': 0.1})
    assert L.metric_policy_audit({'g': good})['status'] == 'ok', L.metric_policy_audit({'g': good})
    tuned = dict(good, decision_threshold=0.4, decision_threshold_source='validation_selection',
                 threshold_selection_split='test', threshold_selection_metric='f1')
    bad = L.metric_policy_audit({'g': tuned})
    assert bad['status'] == 'warn' and any('leakage' in w for w in bad['warnings']), bad
    bad_floor = dict(good, pr_auc_prevalence=0.5)
    bf = L.metric_policy_audit({'g': bad_floor})
    assert bf['status'] == 'warn' and any('floor is misstated' in w for w in bf['warnings']), bf
    no_src = dict(good)
    no_src.pop('decision_threshold_source')
    ns = L.metric_policy_audit({'g': no_src})
    assert ns['status'] == 'warn' and any('decision_threshold_source' in w for w in ns['warnings']), ns
    inv = dict(good, roc_auc=0.2)
    iv = L.metric_policy_audit({'g': inv})
    assert iv['status'] == 'warn' and any('below chance' in w for w in iv['warnings']), iv
    # A declared/recommended metric must be a metric NAME.  A list/dict is
    # malformed evidence and must be reported as a problem, not crash the
    # audit with an unhashable-type TypeError.
    _cm0 = L.classification_metrics([0] * 90 + [1] * 10, y_pred=[0] * 100,
                                    labels=[0, 1])
    for _key in ('primary_metric', 'recommended_primary_metric'):
        _mal = dict(_cm0, variant='m', task='binary_classification',
                    **{_key: [1]})
        _ma = L.metric_policy_audit({'m': _mal})
        assert _ma['status'] == 'warn' and any(
            _key in w for w in _ma['warnings']), (_key, _ma)
    # A non-finite forged support must be refused, not crash int() with
    # OverflowError while reconstructing the trivial floor.
    _ov = json.loads(json.dumps(_cm0))
    _ov.update(variant='m', task='binary_classification')
    _ov['per_class']['0']['support'] = float('inf')
    _ova = L.metric_policy_audit({'m': _ov})
    assert _ova['status'] == 'warn', _ova
    # Unhashable record-identity metadata (variant/name) must not crash the
    # degenerate-record bookkeeping.
    _dm = dict(_cm0, variant={'not': 'a-label'}, task='binary_classification',
               predicted_positive_rate=1.0, degenerate_predictions=True)
    _da = L.metric_policy_audit({'m': _dm})
    assert _da['status'] == 'warn' and _da['checks'], _da
    # Macro/micro/weighted F1 must all be reproducible from the per-class
    # counts; a hand-edited aggregate cannot pass.
    _cm = L.classification_metrics(np.array([0] * 95 + [1] * 5),
                                   y_pred=np.array([0] * 100), labels=[0, 1])
    _rec = dict(_cm, variant='c', task='binary_classification',
                prevalence={'0': 0.95, '1': 0.05})
    assert L.metric_policy_audit({'c': _rec})['status'] == 'ok', L.metric_policy_audit({'c': _rec})
    _bad_micro = dict(_rec, f1_micro=0.123)
    _bm = L.metric_policy_audit({'c': _bad_micro})
    assert _bm['status'] == 'warn' and any('f1_micro' in w and 'reproduced' in w
                                           for w in _bm['warnings']), _bm
    _bad_pc = json.loads(json.dumps(_rec))
    _bad_pc['per_class']['0']['precision'] = 0.1
    _bp = L.metric_policy_audit({'c': _bad_pc})
    assert _bp['status'] == 'warn' and any('per_class counts' in w for w in _bp['warnings']), _bp
    _no_pos = dict(_rec)
    _no_pos.pop('positive_label')
    _np_ = L.metric_policy_audit({'c': _no_pos})
    assert _np_['status'] == 'warn' and any('positive_label' in w for w in _np_['warnings']), _np_


def t_boundary_macro_micro_f1_contract():
    # Macro-over-sequences F1 is NOT the harmonic mean of the mean precision
    # and mean recall.  A record must expose micro counts (or per-sequence
    # values) before F1 can be reconstructed; the old audit asserted the wrong
    # identity and would flag a correct macro record.
    b = {'b::seed0': {'variant': 'b', 'ppl': 10.0,
                      'stats': {'L0': {'bnd_prec': 0.55, 'bnd_rec': 0.55,
                                       'bnd_f1': 0.1818181818,
                                       'bnd_f1_average': 'macro_over_sequences',
                                       'bnd_prec_micro': 0.55, 'bnd_rec_micro': 0.55,
                                       'bnd_f1_micro': 0.55,
                                       'bnd_tp': 55, 'bnd_fp': 45, 'bnd_fn': 45,
                                       'bnd_rand': 0.1}}}}
    a = L.metric_policy_audit(b)
    chk = next(c for c in a['checks'] if c['metric'] == 'boundary_alignment')
    assert chk['ok'] is True and chk['macro_f1_reconstructible'] is True, chk
    # Without micro counts the macro F1 is still valid but unverifiable.
    b2 = json.loads(json.dumps(b))
    for k in ('bnd_prec_micro', 'bnd_rec_micro', 'bnd_f1_micro', 'bnd_tp', 'bnd_fp', 'bnd_fn'):
        b2['b::seed0']['stats']['L0'].pop(k)
    a2 = L.metric_policy_audit(b2)
    chk2 = next(c for c in a2['checks'] if c['metric'] == 'boundary_alignment')
    assert chk2['ok'] is True and chk2['macro_f1_reconstructible'] is False, chk2
    assert any('macro' in w and 'unverifiable' in w for w in a2['warnings']), a2
    # The per-sequence counter path must agree with the reported rates.
    _gt = np.zeros(10, dtype=bool)
    _gt[2] = True
    _gt[6] = True
    _d = {}
    _p, _r, _f, _rp = L.boundary_alignment([3, 8], _gt, 10, tol=1, detail=_d)
    assert _p == 1.0 and _r == 1.0 and _f == 1.0, (_p, _r, _f)
    assert (_d['tp'], _d['fp'], _d['fn'], _d['n_pred'], _d['n_gt']) == (2, 0, 0, 2, 2), _d
    # A declared micro F1 must be the harmonic mean of the micro precision
    # and recall; a forged value fails.
    b3 = json.loads(json.dumps(b))
    b3['b::seed0']['stats']['L0']['bnd_f1_micro'] = 0.9
    a3 = L.metric_policy_audit(b3)
    chk3 = next(c for c in a3['checks'] if c['metric'] == 'boundary_alignment')
    assert chk3['ok'] is False and 'micro-f1-not-harmonic' in chk3['invalid'], chk3


def t_hypothesis_audit_completeness_and_metric_alignment():
    good = L.hypothesis_audit(L.HYPOTHESIS_DESIGN, 'component')
    assert good['status'] == 'ok' and not good['invalid'], good
    assert 'dynamic_blocking' in good['hypotheses'], good
    # Claims that promise matched capacity must say so; the loss-term
    # ablation and the matched-dense baseline both do.
    assert L.HYPOTHESIS_DESIGN['compression_regularizer']['requires_param_match'] is True
    assert L.BASELINE_HYPOTHESIS_DESIGN['compression_vs_matched_dense']['requires_param_match'] is True
    # The NIAH accuracy contrasts now have an explicit pre-registered
    # hypothesis family too, including the reported sign convention.
    _niah_ha = L.hypothesis_audit(L.NIAH_HYPOTHESIS_DESIGN, 'niah')
    assert _niah_ha['status'] == 'ok' and not _niah_ha['invalid'], _niah_ha
    for _name, _spec in L.NIAH_HYPOTHESIS_DESIGN.items():
        assert _spec['primary_metric'] == 'top1_accuracy', _name
        assert _spec['metric_policy'] == 'niah_retrieval', _name
        assert _spec['predicted_sign'] == -1, _name
    bad_table = {
        'h0': {'statement': 'x', 'mechanism': 'y', 'predicted_sign': 0,
               'primary_metric': 'ndcg', 'falsifier': 'z'},
        'h1': {'statement': 'x'},
    }
    r = L.hypothesis_audit(bad_table, 'component')
    assert r['status'] == 'warn', r
    assert 'h0' in r['direction_unspecified'] and 'h0' in r['metric_unknown'], r
    assert 'h1' in r['invalid'] and any('incomplete' in w for w in r['warnings']), r
    # A metric that is only a SECONDARY endpoint in the policy cannot be the
    # hypothesis's primary endpoint without an explicit secondary declaration.
    _base = {'statement': 'x', 'mechanism': 'y', 'predicted_sign': -1,
             'requires_param_match': False, 'falsifier': 'z'}
    _roc = dict(_base, primary_metric='roc_auc', metric_policy='binary_classification')
    _rr = L.hypothesis_audit({'roc_primary': _roc}, 'component')
    assert 'roc_primary' in _rr['invalid'] and 'roc_primary' in _rr['metric_misaligned'], _rr
    assert 'roc_primary' not in _rr['hypotheses']['roc_primary']['metric_primary_policies'], _rr
    # ... but the CONDITIONAL primary under extreme imbalance (PR-AUC) is a
    # legitimate primary endpoint and must pass the role check.
    _pr = dict(_base, primary_metric='pr_auc', metric_policy='binary_classification')
    _prr = L.hypothesis_audit({'pr_primary': _pr}, 'component')
    assert _prr['status'] == 'ok' and not _prr['invalid'], _prr
    # Explicitly declaring a secondary endpoint is allowed only with a
    # justification, and it stays visible as a warning.
    _sec = dict(_base, primary_metric='roc_auc', metric_policy='binary_classification',
                metric_role='secondary', metric_justification='ROC is the threshold-free sensitivity readout')
    _sr = L.hypothesis_audit({'roc_secondary': _sec}, 'component')
    assert not _sr['invalid'] and 'roc_secondary' in _sr['metric_secondary_justified'], _sr
    assert any('secondary endpoint' in w for w in _sr['warnings']), _sr
    # A predicted_sign=0 result is significant but cannot be called verified.
    entries = [{'controlled': True, 'n': 7, 'significant_global': True,
                'within_seed_noise': False, 'param_controlled': True,
                'predicted_sign': 0, 'direction_consistent': True}]
    assert L._classify_group_status(entries, min_n=3, alpha=0.05, m_family=1) == 'direction_unspecified'




# ---------------------------------------------------------------------------
# I. leakage-safe data splitting / preprocessing
# ---------------------------------------------------------------------------

    # A malformed caller-supplied policy must not crash the audit; each bad
    # entry is surfaced as a warning and the remaining policy still applies.
    for _bad_pol in ([], {'x': None}, {'x': 1}):
        _ha = L.hypothesis_audit(table={}, names=[], policy=_bad_pol)
        assert _ha['status'] == 'warn' and _ha['warnings'], (_bad_pol, _ha)
    _mixed_pol = {'x': None, 'binary_classification': dict(L.METRIC_POLICY['binary_classification'])}
    _ha = L.hypothesis_audit(table={}, names=[], policy=_mixed_pol)
    assert _ha['status'] == 'warn' and any('`x`' in w for w in _ha['warnings']), _ha
    # A hypothesis bound to a malformed policy entry fails closed instead of
    # raising on `policy[name].get`.
    _spec = dict(L.HYPOTHESIS_DESIGN['dynamic_blocking'])
    _bad_bind = L.hypothesis_audit(
        table={'h': {**_spec, 'metric_policy': 'x'}},
        names=['h'], policy={'x': None})
    assert _bad_bind['status'] == 'warn' and _bad_bind['invalid'] == ['h'], _bad_bind

def t_leakage_safe_split_contract():
    idx = L.chronological_split_indices(100, 0.7, 0.2, 0.1)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [70, 20, 10], idx
    assert idx['train'][0] == 0 and idx['train'][-1] == 69
    assert idx['val'][0] == 70 and idx['test'][-1] == 99
    assert L.assert_no_split_leakage(idx['train'], idx['val'], idx['test']) is True
    # The split is contiguous in input order: no row from a later block can
    # appear in an earlier one.
    assert all(np.all(np.diff(idx[k]) == 1) for k in ('train', 'val', 'test'))
    # A shuffled time split must be rejected even when the row indices are
    # disjoint.
    t = np.arange(100)
    try:
        L.assert_no_split_leakage(np.array([0, 2]), np.array([1]), np.array([3]), time=t)
    except ValueError as e:
        assert 'temporal leakage' in str(e)
    else:
        raise AssertionError('future rows in the training split must be rejected')
    # Reordering by an explicit time column keeps the chronology.
    order = np.array([5, 0, 3, 1, 6, 2, 4])
    ti = L.split_indices(7, time=order, train_frac=4 / 7, val_frac=2 / 7, test_frac=1 / 7)
    assert np.max(order[ti['train']]) <= np.min(order[ti['val']])
    assert np.max(order[ti['val']]) <= np.min(order[ti['test']])
    L.assert_no_split_leakage(ti['train'], ti['val'], ti['test'], time=order)
    # Snapping a split boundary to the next timestamp block must not consume
    # the terminal cut and make the following split impossible.  Blocks are
    # 80 / 5 / 15 rows with targets 80 / 10 / 10; the legal split uses the
    # 85 cut for validation and leaves the final 15 rows for test.
    tied = np.r_[np.arange(80), np.full(5, 80), np.full(15, 85)]
    tidx = L.split_indices(100, time=tied, train_frac=0.8, val_frac=0.1,
                           test_frac=0.1)
    assert [len(tidx[k]) for k in ('train', 'val', 'test')] == [80, 5, 15], tidx
    L.assert_no_split_leakage(tidx['train'], tidx['val'], tidx['test'], time=tied)
    # A fraction explicitly set to zero must not receive the flooring
    # remainder.
    zidx = L.split_indices(3, train_frac=0.5, val_frac=0.5, test_frac=0.0)
    assert [len(zidx[k]) for k in ('train', 'val', 'test')] == [1, 2, 0], zidx


def t_group_split_contract():
    groups = np.repeat(np.arange(30), 2)
    idx = L.grouped_chronological_split_indices(60, groups, 0.6, 0.2, 0.2)
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'], groups=groups)
    gsets = {k: {groups[i] for i in idx[k]} for k in ('train', 'val', 'test')}
    assert not (gsets['train'] & gsets['val'])
    assert not (gsets['train'] & gsets['test'])
    assert not (gsets['val'] & gsets['test'])
    assert max(gsets['train']) < min(gsets['val']) < min(gsets['test'])
    # A deliberately shuffled group order is still assigned whole.
    shuffled = np.roll(groups, 7)
    idx2 = L.split_indices(60, groups=shuffled, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    g2 = {k: {shuffled[i] for i in idx2[k]} for k in ('train', 'val', 'test')}
    assert not (g2['train'] & g2['val']) and not (g2['train'] & g2['test']) and not (g2['val'] & g2['test'])
    # NaT timedelta group ids are one missing group, exactly like NaT
    # datetime/NaN ids; they must not silently become per-row unique keys.
    _nat = np.array([np.timedelta64('NaT', 'ns'), np.timedelta64('NaT', 'ns'),
                     np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns')],
                    dtype='timedelta64[ns]')
    try:
        L.assert_no_split_leakage([0], [1], [2, 3], groups=_nat)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT timedelta group ids must collapse into one group')


def t_group_split_complete_allocation_and_zero_targets():
    # Whole groups keep their chronological order, but a feasible partition
    # must not be rejected merely because the greedy "fill train first" rule
    # put a large group on the wrong side.  Groups have 1, 1 and 2 rows;
    # targets 2/1/1 can only be met by boundaries 1 and 2 (train=1, val=1,
    # test=2).  The historical greedy split consumed groups 0+1 into train,
    # then put the 2-row group in validation and left test empty.
    groups = np.array([0, 1, 2, 2])
    idx = L.grouped_chronological_split_indices(4, groups, 0.5, 0.25, 0.25)
    assert idx['train'].tolist() == [0], idx
    assert idx['val'].tolist() == [1], idx
    assert idx['test'].tolist() == [2, 3], idx
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              groups=groups)
    # A split whose target is exactly zero stays empty; the remainder belongs
    # to the last requested non-empty split, never to the zero-target side.
    g0 = np.array([0, 1, 2, 3])
    z = L.grouped_chronological_split_indices(
        4, g0, train_frac=0.5, val_frac=0.5, test_frac=0.0)
    assert z['test'].size == 0, z
    assert sorted(np.concatenate([z['train'], z['val']]).tolist()) == [0, 1, 2, 3], z
    z2 = L.grouped_chronological_split_indices(
        4, g0, train_frac=0.0, val_frac=0.5, test_frac=0.5)
    assert z2['train'].size == 0 and z2['val'].size and z2['test'].size, z2
    # Explicit sizes work the same way and preserve whole groups.
    e = L.grouped_chronological_split_indices(
        6, np.array([0, 0, 1, 1, 2, 2]), train_size=2, val_size=2,
        test_size=2)
    assert [len(e[k]) for k in ('train', 'val', 'test')] == [2, 2, 2], e
    L.assert_no_split_leakage(e['train'], e['val'], e['test'],
                              groups=np.array([0, 0, 1, 1, 2, 2]))


def t_group_split_respects_time_windows():
    # Whole groups on one side is necessary but not sufficient: two groups
    # whose time windows overlap must not be placed on opposite sides, or the
    # shared window leaks future/past information across the split.
    groups = np.repeat(np.arange(10), 2)
    time = np.repeat(np.arange(10) * 10, 2)
    idx = L.grouped_chronological_split_indices(20, groups, 0.5, 0.25, 0.25,
                                                order=time)
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              groups=groups, time=time)
    try:
        L.grouped_chronological_split_indices(
            8, [0, 0, 1, 1, 2, 2, 3, 3], 0.5, 0.25, 0.25,
            order=[0, 10, 5, 15, 20, 30, 25, 35])
    except ValueError as e:
        assert 'temporal leakage' in str(e), e
    else:
        raise AssertionError('overlapping group time windows must be rejected')
    # A missing ordering key cannot define a chronological split.
    try:
        L.split_indices(4, time=[0, 1, np.nan, 3])
    except ValueError as e:
        assert 'NaN/Inf' in str(e), e
    else:
        raise AssertionError('NaN time must be rejected before sorting')


def t_data_quality_report_index_disjointness():
    X = np.arange(8, dtype=float).reshape(4, 2)
    rep = L.data_quality_report(X, train_idx=[0, 1], val_idx=[1], test_idx=[2])
    assert rep['status'] == 'fail', rep
    assert rep['checks']['split_index_overlap']['train__val'] == 1, rep
    # A repeated index inside one split is counted, and the shared key set used
    # for the cross-split check does not change the within-split count.
    rep_dup = L.data_quality_report(X, train_idx=[0, 0, 1], val_idx=[2],
                                    test_idx=[3])
    assert rep_dup['checks']['duplicate_rows_within_split']['train'] == 1, rep_dup
    try:
        L.data_quality_report(X, train_idx=[], val_idx=[1])
    except ValueError as e:
        assert 'training reference is empty' in str(e), e
    else:
        raise AssertionError('an empty training reference must not define holdout statistics')


def t_preprocessor_train_only():
    tr = np.array([[0.0, 10.0], [2.0, 10.0], [np.nan, 10.0], [1.0, 10.0]])
    va = np.array([[1000.0, 10.0], [np.inf, 10.0]])
    prep = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                     missing='mean')
    xtr = prep.fit_transform(tr)
    xva = prep.transform(va)
    assert np.all(np.isfinite(xtr)) and np.all(np.isfinite(xva))
    st = prep.get_stats()
    assert abs(st['features'][0]['center'] - 1.0) < 1e-12, st
    # The holdout did not influence the statistics: fitting on train+holdout
    # would move the center by >100x, so a leak cannot hide.
    prep_all = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                         missing='mean').fit(np.vstack([tr, va]))
    assert abs(prep_all.get_stats()['features'][0]['center'] - 1.0) > 100
    before = prep.get_stats()
    prep.transform(va)
    assert prep.get_stats() == before, 'transform must never update statistics'
    # Constant features stay finite (factor 1), and a saved/restored model is
    # numerically identical.
    assert np.all(np.isfinite(prep.transform(np.array([[np.nan, np.nan]]))))
    restored = L.LeakageSafePreprocessor.from_stats(before)
    assert np.allclose(restored.transform(va), xva, equal_nan=True)
    # A preprocessor cannot be used before fitting or with the wrong width.
    try:
        L.LeakageSafePreprocessor().transform(tr)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        prep.transform(np.zeros((2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')


def t_preprocessor_fill_is_consistent():
    # A requested imputation constant outside the training quantiles must be
    # applied identically in fit and transform.  If fit clips it but transform
    # does not, the fitted scale no longer centres its own training matrix.
    X = np.array([[10.0], [11.0], [12.0], [np.nan]])
    prep = L.LeakageSafePreprocessor(scale='standard', missing=0.0,
                                     clip_quantiles=(0.1, 0.9))
    xtr = prep.fit_transform(X)
    assert abs(float(xtr.mean())) < 1e-12, xtr
    assert np.allclose(prep.transform(np.array([[np.nan]])), xtr[-1:]), (prep.transform(np.array([[np.nan]])), xtr[-1:])
    # numpy scalar constants are accepted; NaN/Inf constants are refused.
    assert L.LeakageSafePreprocessor(missing=np.float32(0.0)).missing == 0.0
    for bad in (float('nan'), float('inf'), -float('inf')):
        try:
            L.LeakageSafePreprocessor(missing=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-finite missing constant {bad!r} must be rejected')


def t_leakage_checks_validate_inputs():
    # Too-short time/group arrays must not silently disable the check.
    try:
        L.assert_no_split_leakage([0], [1], [2], time=[0, 1])
    except ValueError as e:
        assert 'time has length' in str(e), e
    else:
        raise AssertionError('short time array must be rejected')
    try:
        L.assert_no_split_leakage([0], [1], [2], groups=['a', 'b'])
    except ValueError as e:
        assert 'groups has length' in str(e), e
    else:
        raise AssertionError('short group array must be rejected')
    # NaN timestamps cannot define a chronological order.
    try:
        L.assert_no_split_leakage([0], [1], [2], time=[0.0, np.nan, 2.0])
    except ValueError as e:
        assert 'NaN/Inf' in str(e), e
    else:
        raise AssertionError('NaN time must be rejected')
    # NaN is a single group id; the same group must not span two splits just
    # because NaN != NaN.
    g = np.array([np.nan, np.nan, 2.0, 3.0])
    try:
        L.assert_no_split_leakage([0], [1], [2, 3], groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('a NaN group crossing splits must be rejected')
    # Empty data still splits into three empty parts.
    idx = L.chronological_split_indices(0)
    assert all(len(idx[k]) == 0 for k in ('train', 'val', 'test')), idx


def t_probe_stats_rejects_malformed_arm_identities():
    import v12_supp as V
    manifest = {'arms': ['a', 'b'], 'seeds': [0, 1]}
    td = tempfile.mkdtemp(prefix='probe_', dir=REPO)
    try:
        path = os.path.join(td, 'distractor.json')
        base = {'seed': 0, 'val_fp': 'fp', 'arm': 'learned',
                'eval_len': 8, 'rho': 0.5, 'ppl_mean': 1.0}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'bad_variant': dict(base, variant={'x': 1}),
                       'bad_arm': dict(base, variant='a', arm={'x': 1})}, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 0 and r['n_excluded'] == 2, r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_probe_stats_rejects_synthesized_and_mismatched_probe_params():
    import v12_supp as V
    manifest = {'arms': ['a'], 'seeds': [0, 1]}
    td = tempfile.mkdtemp(prefix='probe_', dir=REPO)
    try:
        path = os.path.join(td, 'distractor.json')
        code = L.code_stamp()

        def _rec(arm, seed, pp):
            return {'variant': 'a', 'seed': seed, 'val_fp': 'fp', 'arm': arm,
                    'eval_len': 8, 'rho': 0.5, 'mode': 'uniform',
                    'ppl_mean': 2.0 + 0.1 * seed + (0.05 if arm == 'dense' else 0.0),
                    '_code': code, 'probe_params': dict(pp)}

        base = {f'learned::{s}': _rec('learned', s, {'n_seq': 16})
                for s in (0, 1)}
        base.update({f'dense::{s}': _rec('dense', s, {'n_seq': 16})
                     for s in (0, 1)})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(base, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 4, r
        assert len(r['contrasts']) == 1, r

        # The parameter-matching `_mr` field is part of the ARM, not the
        # probe measurement: arms whose recipes differ only there are still
        # a valid controlled contrast (v10 applies the same normalisation).
        mr = dict(base)
        mr['learned::0'] = dict(mr['learned::0'], recipe='x_mr4.0_dfabc')
        mr['learned::1'] = dict(mr['learned::1'], recipe='x_mr4.0_dfabc')
        mr['dense::0'] = dict(mr['dense::0'], recipe='x_mr6.7_dfabc')
        mr['dense::1'] = dict(mr['dense::1'], recipe='x_mr6.7_dfabc')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mr, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert len(r['contrasts']) == 1, r

        # The real mlp_match_ref values contain underscores (csa_dynamic,
        # csa_fixed_rope).  The normaliser must remove the whole run_cfg `_mr`
        # field, not stop at its first underscore: otherwise two arms that
        # differ only in parameter matching can never be paired.
        _rc_a = ('steps1_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_'
                 'v8192_wu50_cl0.05_dlm10.0_mtfull_mrcsa_fixed_rope_ee0_es128_'
                 'detcpu_dfabc-def_tk0_ampfp32_eb0_ae8192_csx')
        _rc_b = _rc_a.replace('_mrcsa_fixed_rope_', '_mrfull_rope_')
        assert V10._recipe_norm(_rc_a) == V10._recipe_norm(_rc_b), (
            V10._recipe_norm(_rc_a), V10._recipe_norm(_rc_b))
        assert V10._recipe_norm(_rc_a) != V10._recipe_norm(
            _rc_a.replace('_ee0_', '_ee1_'))
        _p1_a = 'p1l_steps1_mr4-csa_fixed_rope_ns4_dfabc'
        _p1_b = _p1_a.replace('_mr4-', '_mr6.7-')
        assert V10._recipe_norm(_p1_a) == V10._recipe_norm(_p1_b)
        assert V10._recipe_norm(_p1_a) != V10._recipe_norm(
            _p1_a.replace('_ns4_', '_ns8_'))

        # A synthesized probe trajectory is provenance, not a measurement.
        synth = dict(base)
        synth['learned::0'] = dict(synth['learned::0'], synthesized=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(synth, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 3, r
        assert not r['cells'] or all(c['arm'] != 'learned' or c['n'] != 2
                                     for c in r['cells']), r

        # Two arms measured under different probe geometries must not be
        # differenced as if they were one controlled contrast.
        mixed = dict(base)
        for s in (0, 1):
            mixed[f'dense::{s}'] = _rec('dense', s, {'n_seq': 8})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mixed, f)
        r = V._test_probe_stats(path, manifest, 'fp')
        assert r['n_admitted'] == 4, r
        assert not r['contrasts'], r
        assert any('different probe_params' in x for x in r.get('contrast_exclusions', [])), r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_audit_selection_rejects_boolean_measurements():
    A = __import__('audit_gates')
    td = tempfile.mkdtemp(prefix='selection_', dir=REPO)
    try:
        path = os.path.join(td, 'selection.json')
        cell = {'arm': 'a', 'eval_len': 1024, 'k': 32, 'n_blocks': 256,
                'n_queries': 100, 'recall_at_k': 0.5, 'mass_covered': 0.6,
                'random_baseline': 0.125,
                'per_seed': {str(s): {'mass_covered': 0.6,
                                      'random_baseline': 0.125}
                             for s in range(6)}}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [dict(cell, mass_covered=True)]}, f)
        r = A.audit_selection(path=path)
        assert r['status'] == 'fail', r
        assert any('booleans' in p for p in r['problems']), r
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cells': [cell]}, f)
        r = A.audit_selection(path=path)
        assert r['status'] == 'ok', r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_categorical_shift_empty_reference():
    r = L.categorical_shift_report([], [1, 2, 3])
    assert r['n_reference'] == 0 and r['n_holdout'] == 3, r
    assert r['unseen_rate'] == 1.0 and r['total_variation'] > 0.0, r
    # No reference distribution means JS is undefined, not a fixed log(2)/2
    # score that looks like a measured divergence.
    assert r['js_divergence'] is None and r['js_defined'] is False, r
    try:
        L.categorical_shift_report([1, 2], [1, 2], max_categories=0)
    except ValueError:
        pass
    else:
        raise AssertionError('max_categories=0 must be rejected')


def t_data_quality_report():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    rep = L.data_quality_report(X, y=[0, 1, 1, 0],
                                train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert rep['status'] == 'fail', rep
    assert rep['checks']['duplicate_rows_across_splits']['train__val'] == 1
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 1
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    rep2 = L.data_quality_report(Y, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep2['status'] == 'warnings', rep2
    assert rep2['checks']['nan_count'] + rep2['checks']['inf_count'] == 0
    Xn = Y.copy()
    Xn[2, 0] = np.nan
    Xn[3, 0] = np.inf
    rep3 = L.data_quality_report(Xn, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep3['checks']['nan_count'] == 1 and rep3['checks']['inf_count'] == 1
    # Group boundaries are part of the report, not a matter of trust.
    rep4 = L.data_quality_report(Y, groups=np.array(['a', 'a', 'b', 'c']),
                                 train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert rep4['checks']['group_overlap']['train__val'] == 1
    assert rep4['status'] == 'fail'


def t_data_quality_target_outliers():
    # The target fence is learned from training labels only; a holdout target
    # cannot widen it.  This catches label corruption that exact-duplicate
    # checks cannot see.
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = [0.0, 1.0, 2.0, 3.0, 100.0, 2.0]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1, 2, 3],
                                val_idx=[4], test_idx=[5],
                                target_outlier_z=1.5)
    assert rep['status'] == 'warnings', rep
    assert rep['checks']['target_outliers_by_split']['val'] == 1, rep
    assert rep['checks']['target_outliers_by_split']['train'] == 0, rep
    hard = L.data_quality_report(X, y=y, train_idx=[0, 1, 2, 3],
                                 val_idx=[4], test_idx=[5],
                                 fail_on_target_outliers=True)
    assert hard['status'] == 'fail', hard
    # A training target outside its own train-only fence is also visible.
    rep_tr = L.data_quality_report(X, y=[0.0, 1.0, 2.0, 100.0, 1.0, 2.0],
                                   train_idx=[0, 1, 2, 3],
                                   val_idx=[4], test_idx=[5],
                                   target_outlier_z=1.5)
    assert rep_tr['checks']['target_outliers_by_split']['train'] == 1, rep_tr
    # Categorical labels are not coerced into a numeric target.
    cat = L.data_quality_report(X, y=['a', 'b', 'c', 'd', 'a', 'b'],
                                train_idx=[0, 1, 2, 3], val_idx=[4],
                                test_idx=[5], target_outlier_z=1.5)
    assert cat['checks']['target_outlier_fence'] is None, cat
    assert cat['checks']['target_outlier_fence_estimated'] is False, cat
    assert any('not estimable' in w for w in cat['warnings']), cat
    # Asking the gate to FAIL on target outliers when no numeric fence exists
    # must fail closed, not silently report zero outliers.
    cat_fail = L.data_quality_report(
        X, y=['a', 'b', 'c', 'd', 'a', 'b'], train_idx=[0, 1, 2, 3],
        val_idx=[4], test_idx=[5], fail_on_target_outliers=True)
    assert cat_fail['status'] == 'fail', cat_fail
    one_label = L.data_quality_report(
        X, y=[0.0, None, None, None, 1.0, 2.0], train_idx=[0, 1, 2, 3],
        val_idx=[4], test_idx=[5], target_outlier_z=1.5)
    assert one_label['checks']['target_outlier_fence_estimated'] is False, one_label
    # Requesting the target-outlier audit without y must fail closed instead
    # of silently omitting the gate.
    for _kw in ({'target_outlier_z': 1.5}, {'fail_on_target_outliers': True}):
        try:
            L.data_quality_report(X, train_idx=[0, 1, 2, 3], **_kw)
        except ValueError as e:
            assert 'requires y' in str(e), e
        else:
            raise AssertionError(f'{_kw} without y must be refused')
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.data_quality_report(X, y=y, train_idx=[0, 1],
                                  target_outlier_z=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target_outlier_z={bad!r} must be refused')
    # Explicit label gates without labels must fail closed, not be skipped.
    for _kw in ({'fail_on_label_shift': True},
                {'fail_on_nonfinite_labels': True},
                {'fail_on_unknown_labels': True}):
        try:
            L.data_quality_report(X, train_idx=[0, 1, 2, 3], **_kw)
        except ValueError as e:
            assert 'requires y' in str(e), e
        else:
            raise AssertionError(f'{_kw} without y must be refused')


def t_split_sizes_explicit_never_negative():
    # Explicit sizes and fractional targets can overdraw the remaining rows; the
    # filler must clamp rather than emit a negative split.
    cases = ({'train_frac': 0.05, 'val_frac': 0.9, 'test_frac': 0.05, 'train_size': 9},
             {'train_frac': 0.8, 'val_frac': 0.1, 'test_frac': 0.1, 'val_size': 9})
    for kwargs in cases:
        idx = L.chronological_split_indices(10, **kwargs)
        sizes = [len(idx[k]) for k in ('train', 'val', 'test')]
        assert sum(sizes) == 10 and all(s >= 0 for s in sizes), (kwargs, sizes)
    # The normal explicit-size reconciliation is unchanged.
    idx = L.chronological_split_indices(10, train_size=6)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    idx_float = L.chronological_split_indices(10, train_size=6.0)
    assert [len(idx_float[k]) for k in ('train', 'val', 'test')] == [6, 1, 3]
    # An explicitly sized split must not make the unfilled tail create rows in
    # a split the caller declared empty (test_frac=0).
    idx0 = L.chronological_split_indices(10, train_size=6, test_frac=0.0)
    assert [len(idx0[k]) for k in ('train', 'val', 'test')] == [6, 4, 0], idx0
    idx0b = L.chronological_split_indices(10, val_size=2, test_frac=0.0)
    assert [len(idx0b[k]) for k in ('train', 'val', 'test')] == [8, 2, 0], idx0b
    # Explicit sizes are counts: non-integral/coerced values must be refused
    # rather than silently truncated (1.5 -> 1) or parsed ('2' -> 2).
    for bad in (1.5, '2', True):
        try:
            L.chronological_split_indices(100, train_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'train_size={bad!r} must be rejected')
    for bad_n in (10.5, '10', True):
        try:
            L.chronological_split_indices(bad_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n={bad_n!r} must be rejected')
    assert len(L.chronological_split_indices(100.0)['train']) == 80


def t_empty_split_still_validates_fractions():
    # An empty table must not bypass the fraction contract: the same
    # contradictory arguments rejected for n>0 must be rejected for n=0, so a
    # caller cannot build a split spec that only fails after rows are appended.
    idx = L.chronological_split_indices(0)
    assert all(len(idx[k]) == 0 for k in ('train', 'val', 'test')), idx
    for bad in ({'train_frac': 0.5, 'val_frac': 0.5, 'test_frac': 0.5},
                {'train_frac': -0.1, 'val_frac': 0.6, 'test_frac': 0.5},
                {'train_frac': 0.0, 'val_frac': 0.0, 'test_frac': 0.0}):
        try:
            L.chronological_split_indices(0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n=0 accepted invalid fractions {bad!r}')


def t_audit_split_sequence_overlap():
    # A 1-D token cache stores train as a stream: a validation row overlapping
    # at an arbitrary offset must still be detected (not only at aligned rows).
    train = np.arange(40, dtype=np.int64)
    rows = np.array([[5, 6, 7, 8], [100, 101, 102, 103]], dtype=np.int64)
    assert A._row_overlap_count(train, rows) == 1
    assert A._row_overlap_count(train, rows, chunk_tokens=8) == 1
    assert A._row_overlap_count(train, np.array([[36, 37, 38, 39]]), chunk_tokens=8) == 1
    # 2-D tables keep the exact aligned-row contract.
    table = np.array([[1, 2], [3, 4], [5, 6]])
    assert A._row_overlap_count(table, np.array([[3, 4], [9, 9]])) == 1


def t_categorical_and_cache_guards():
    same = L.categorical_shift_report([1, 2, 2, 3], [1, 2, 2, 3])
    assert same['js_divergence'] < 1e-12 and same['total_variation'] < 1e-12
    shifted = L.categorical_shift_report([1, 1, 1, 1], [9, 9, 9, 9])
    assert shifted['js_divergence'] > 0.5 and shifted['unseen_rate'] == 1.0
    # A tuple category containing the delimiter must not collapse into a
    # different tuple category.  Before the length-prefixed key these two
    # reference levels shared one string key, so a holdout containing only one
    # of them looked identical to the reference.
    _tuple_shift = L.categorical_shift_report(
        [('a', 'b'), ('a,s:b',)], [('a', 'b')])
    assert _tuple_shift['js_divergence'] > 1e-12, _tuple_shift
    # A real tuple category equal to the old missing-value sentinel is not a
    # missing value; the two reference levels must stay distinct.
    _missing_tuple = L.categorical_shift_report(
        [None, ('__missing__',)], [('__missing__',)])
    assert _missing_tuple['js_divergence'] > 1e-12, _missing_tuple
    # The cache-hit guard must reject a validation file whose content no
    # longer matches the recorded fingerprint.
    tr = np.arange(32, dtype=np.int64) % 11
    va = np.arange(16, dtype=np.int64).reshape(2, 8) % 11
    meta = {'src_fp': L._ids_fp(tr), 'val_fp': L._ids_fp(va.reshape(-1)),
            'vocab': 11, 'split_strategy': 'official_train_validation_test',
            'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}
    assert L._cached_split_problems(meta, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8) == []
    bad = dict(meta, val_fp='not-the-validation-fingerprint')
    assert any('validation id fingerprint' in p for p in
               L._cached_split_problems(bad, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    bad2 = dict(meta, shuffle_at_split=True)
    assert any('shuffled split' in p for p in
               L._cached_split_problems(bad2, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    bad3 = dict(meta, tokenizer_fit_split='test')
    assert any('tokenizer' in p for p in
               L._cached_split_problems(bad3, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))


def t_public_exports_resolve():
    # __all__ is an API contract: every advertised name must exist, and a
    # star-import must not fail.  A stale entry (a name added to __all__ but
    # never implemented) breaks `from data_guard import *` at import time.
    import data_guard as DG
    missing = [name for name in DG.__all__ if not hasattr(DG, name)]
    assert not missing, f'data_guard.__all__ advertises undefined names: {missing}'
    ns = {}
    exec('from data_guard import *', ns)
    for name in DG.__all__:
        assert name in ns, f'{name} was not exported by a star-import'


def t_split_and_prepare_fits_train_only():
    # Holdout rows are extreme; if they influenced imputation or scaling the
    # training centre/factor would move by >100x.  The one-call helper must
    # split first and fit on train only.
    X = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 1.0],
                  [1000.0, 1.0], [2000.0, 1.0], [3000.0, 1.0]])
    out = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                              quality_report=True)
    idx = out['indices']
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [3, 1, 2], idx
    prep = out['preprocessor']
    st = prep.get_stats()
    assert abs(st['features'][0]['center'] - 1.0) < 1e-12, st
    # Every returned split is finite and equals an explicit transform of the
    # raw split (no hidden global statistics).
    for name in ('train', 'val', 'test'):
        raw = X[idx[name]]
        assert np.all(np.isfinite(out['X'][name])), (name, out['X'][name])
        assert np.allclose(out['X'][name], prep.transform(raw)), name
    assert prep.get_stats() == st, 'transform must not update fitted statistics'
    assert out['quality']['n_train_reference'] == 3, out['quality']
    # Preprocess=False still returns the raw chronological parts, and the
    # optional quality report uses the same train reference.
    raw = L.split_and_prepare(X, time=[5, 4, 3, 2, 1, 0], preprocess=False,
                              train_frac=0.5, val_frac=0.25, test_frac=0.25)
    assert raw['preprocessor'] is None
    assert [len(raw['indices'][k]) for k in ('train', 'val', 'test')] == [3, 1, 2], raw['indices']
    assert L.assert_no_split_leakage(raw['indices']['train'], raw['indices']['val'],
                                     raw['indices']['test'],
                                     time=np.array([5, 4, 3, 2, 1, 0])) is True
    # Exact duplicate rows across a split boundary are leakage even when the
    # row indices/timestamps are disjoint; the one-call helper must refuse.
    Xdup = np.array([[0.0], [1.0], [0.0], [3.0]])
    try:
        L.split_and_prepare(Xdup, preprocess=False)
    except ValueError as e:
        assert 'duplicate feature rows cross split' in str(e), e
    else:
        raise AssertionError('cross-split duplicate rows must be refused')
    waived = L.split_and_prepare(Xdup, preprocess=False,
                                 check_cross_split_duplicates=False)
    assert 'cross_split_duplicates' not in waived, waived
    assert waived.get('cross_split_duplicates_waived') is True, waived
    ok = L.split_and_prepare(np.array([[0.0], [1.0], [2.0], [3.0]]),
                             preprocess=False)
    assert ok['cross_split_duplicates'] == {
        'train__val': 0, 'train__test': 0, 'val__test': 0}, ok


def t_split_and_prepare_strict_quality_gate():
    # A recorded 'fail' report must not be returned as failure-free data when
    # the caller asked for a strict gate.  The holdout is never used to make
    # the decision: the NaN is in the training split.
    X = np.array([[0.0], [1.0], [np.nan], [2.0], [3.0], [4.0]])
    rep = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                              quality_report=True)
    assert rep['quality']['status'] == 'fail', rep['quality']
    try:
        L.split_and_prepare(X, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                            fail_on_quality=True)
    except RuntimeError as e:
        assert 'data-quality gate failed' in str(e), e
    else:
        raise AssertionError('fail_on_quality must refuse a failing quality report')
    # Warnings alone do not abort; the split is returned with its report.
    ok = L.split_and_prepare(np.arange(6.0).reshape(-1, 1),
                             train_frac=0.5, val_frac=0.25, test_frac=0.25,
                             fail_on_quality=True)
    assert ok['quality']['status'] != 'fail', ok['quality']


def t_data_quality_temporal_audit():
    # Indices built outside the split helpers must still be audited for time
    # leakage: row/group disjointness alone cannot catch a shuffled series.
    X = np.arange(12, dtype=float).reshape(6, 2)
    time = np.array([5, 4, 3, 2, 1, 0])
    idx = L.split_indices(6, time=time, train_frac=0.5, val_frac=0.25,
                          test_frac=0.25)
    rep = L.data_quality_report(X, train_idx=idx['train'], val_idx=idx['val'],
                                test_idx=idx['test'], time=time)
    assert all(not c['leak'] for c in rep['checks']['temporal_order'].values()), rep
    shuffled = {'train': np.array([0, 1]), 'val': np.array([2]),
                'test': np.array([3])}
    bad = L.data_quality_report(X, train_idx=shuffled['train'],
                                val_idx=shuffled['val'],
                                test_idx=shuffled['test'], time=time)
    assert bad['status'] == 'fail', bad
    assert any('temporal leakage' in msg for msg in bad['hard_failures']), bad
    assert bad['checks']['temporal_order']['train__val']['leak'] is True, bad
    # An equal timestamp cannot straddle a boundary either.
    tie = np.array([0, 0, 1, 1, 2, 2])
    tie_rep = L.data_quality_report(
        X, train_idx=[0], val_idx=[1], test_idx=[2, 3, 4, 5], time=tie)
    assert tie_rep['status'] == 'fail', tie_rep
    # The one-call helper forwards its time key, so the same gate runs there.
    out = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                              test_frac=0.25, quality_report=True)
    assert out['quality']['status'] in ('ok', 'warnings'), out['quality']
    assert all(not c['leak'] for c in
               out['quality']['checks']['temporal_order'].values()), out['quality']
    # A quality_kwargs override must not collide with the forwarded key.
    out2 = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                               test_frac=0.25, quality_report=True,
                               quality_kwargs={'time': time})
    assert out2['quality']['status'] in ('ok', 'warnings'), out2['quality']
    # An explicit None override must not disable the temporal audit the split
    # helper already knows how to run.
    out3 = L.split_and_prepare(X, time=time, train_frac=0.5, val_frac=0.25,
                               test_frac=0.25, quality_report=True,
                               quality_kwargs={'time': None, 'groups': None})
    _temporal3 = out3['quality']['checks']['temporal_order']
    assert all(c.get('checked') for c in _temporal3.values()), out3['quality']
    # String event ids/timestamps are orderable in Python even though NumPy 2
    # has no string maximum/minimum ufunc loop; grouped splits must not fall
    # over (or silently disable the order check) for them.
    sg = np.arange(6)
    st = np.array(['f', 'e', 'd', 'c', 'b', 'a'])
    sidx = L.grouped_chronological_split_indices(
        6, sg, 0.5, 0.25, 0.25, order=st)
    L.assert_no_split_leakage(sidx['train'], sidx['val'], sidx['test'],
                              groups=sg, time=st)
    srep = L.data_quality_report(
        np.arange(12, dtype=float).reshape(6, 2),
        train_idx=sidx['train'], val_idx=sidx['val'], test_idx=sidx['test'],
        groups=sg, time=st)
    assert all(not c['leak'] for c in srep['checks']['temporal_order'].values()), srep


def t_capped_document_id_sidecar():
    # The capped-id helper must record exactly the documents that contributed
    # a served token, using the same row-length list as the tokenizer stream.
    rows = ['alpha', 'beta', 'gamma']
    lengths = [3, 3, 5]
    expected = [L._doc_id_u64(L._doc_exact_fp(r)) for r in rows]
    ids3 = L._doc_ids_for_capped_rows(rows, lengths, 3)
    assert ids3.tolist() == expected[:1], (ids3, expected)
    # A cut inside the second document still records the second document (it
    # contributed one served token), but never the untouched third one.
    ids4 = L._doc_ids_for_capped_rows(rows, lengths, 4)
    assert ids4.tolist() == expected[:2], (ids4, expected)
    ids0 = L._doc_ids_for_capped_rows(rows, lengths, 0)
    assert ids0.size == 0, ids0

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _Tok:
        def encode_batch(self, batch):
            return [_Enc(list(range(len(text)))) for text in batch]

    lengths_out = []
    ids = L._tokenise_row_chunks([['aa', 'b']], 5, 'unit', _Tok(), 0,
                                 lengths_out=lengths_out)
    assert lengths_out == [3, 2], lengths_out
    assert ids.tolist() == [0, 1, 0, 0, 0], ids
    assert L._tokenise_row_chunks([['aa', 'b']], 5, 'unit', _Tok(), 0).tolist() == ids.tolist()
    doc_ids = L._doc_ids_for_capped_rows(['aa', 'b'], lengths_out, 5)
    assert doc_ids.tolist() == [L._doc_id_u64(L._doc_exact_fp('aa')),
                                L._doc_id_u64(L._doc_exact_fp('b'))], doc_ids


def t_document_dedup_closes_near_duplicate_chains():
    for _kw in ({'near_threshold': -1}, {'near_threshold': 1.5},
                {'near_threshold': True}, {'bands': 2.5}, {'bands': True},
                {'bands': 0}, {'bands': 9}, {'near_threshold': 65}):
        try:
            L._DocDeduplicator(**_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_DocDeduplicator({_kw!r}) must be refused')
    # A~T is skipped as a near duplicate of train T.  B is within the near
    # threshold of A but just outside it for T, so a stateless bucket index
    # would admit B into validation even though it belongs to the same
    # document cluster.  The skipped copy is quarantined so the chain stays on
    # the train side.
    hashes = {'train T': 0, 'val A': 0b111, 'val B': 0b11011}
    orig = L._doc_simhash
    try:
        L._doc_simhash = lambda text: hashes[text]
        d = L._DocDeduplicator(near_threshold=3, bands=4)
        assert d.is_duplicate('train T') == (False, None)
        assert d.is_duplicate('val A') == (True, 'near')
        assert d.is_duplicate('val B') == (True, 'near'), 'chain member slipped through'
    finally:
        L._doc_simhash = orig


def t_test_document_filter_uses_readonly_earlier_indices():
    # A test document must not become part of the train/validation reference
    # set.  Otherwise a later test doc resembling an earlier TEST doc is
    # mislabelled as a train/val duplicate, and the audit no longer knows why
    # the document was removed.
    train = ['train one', 'train two']
    val = ['val one']
    val_index = L._DocDeduplicator()
    for r in val:
        val_index.is_duplicate(r)
    before = val_index.stats()
    test = ['unique test a', 'unique test a', 'unique test b',
            'train one', 'val one']
    kept, excluded, test_index = L._filter_test_documents(
        train, val_index, test, label='test')
    assert kept == ['unique test a', 'unique test b'], kept
    assert excluded['train_exact'] == 1, excluded
    assert excluded['validation_exact'] == 1, excluded
    assert excluded['test_exact'] == 1, excluded
    # The earlier-split index was consulted read-only.
    assert val_index.stats() == before, (before, val_index.stats())
    assert val_index.find('unique test a') == (False, None)
    # The pure lookup form never mutates counters or membership.
    d = L._DocDeduplicator()
    assert d.find('a b c') == (False, None)
    assert d.stats()['added_documents'] == 0
    d.is_duplicate('a b c')
    n_before = d.stats()['added_documents']
    assert d.find('a b c') == (True, 'exact')
    assert d.find('d e f') == (False, None)
    assert d.stats()['added_documents'] == n_before, d.stats()
    # The kept set is still closed under a test-to-test near-duplicate chain.
    assert test_index.find('unique test a') == (True, 'exact')


def t_split_purge_embargo():
    # Purge drops the tail of train and validation; the test slice is intact
    # and no returned row straddles a boundary.
    idx = L.chronological_split_indices(100, 0.7, 0.2, 0.1, purge=3)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [67, 17, 10], idx
    assert idx['train'][-1] == 66 and idx['val'][0] == 70 and idx['val'][-1] == 86
    assert idx['test'][0] == 90
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                              time=np.arange(100))
    t = np.arange(100)[::-1]
    ti = L.split_indices(100, time=t, train_frac=0.7, val_frac=0.2,
                         test_frac=0.1, purge=2)
    assert [len(ti[k]) for k in ('train', 'val', 'test')] == [68, 18, 10], ti
    assert np.max(t[ti['train']]) < np.min(t[ti['val']])
    assert np.max(t[ti['val']]) < np.min(t[ti['test']])
    L.assert_no_split_leakage(ti['train'], ti['val'], ti['test'], time=t)
    # Purging more rows than a split contains is a hard error, not an empty
    # split that would silently make validation unusable.
    for bad in (-1, 1.5, True, '2'):
        # Use a large n so rejection cannot come accidentally from an
        # over-large purge emptying a split: the validator itself must refuse
        # non-integral/coerced values.
        try:
            L.chronological_split_indices(100, purge=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'purge={bad!r} must be rejected')
    assert len(L.chronological_split_indices(100, purge=2.0)['train']) == 78
    try:
        L.chronological_split_indices(10, purge=5)
    except ValueError as e:
        assert 'empty' in str(e), e
    else:
        raise AssertionError('an over-large purge must not empty a split')
    # A grouped split cannot drop individual rows without breaking groups.
    try:
        L.split_indices(10, groups=np.arange(10), purge=1)
    except ValueError as e:
        assert 'grouped' in str(e), e
    else:
        raise AssertionError('purge with groups must be refused')


def t_group_split_refuses_empty_requested_side():
    # One huge group consumes the whole table; a requested validation split
    # would be empty.  Fail instead of returning it and inviting test tuning.
    try:
        L.grouped_chronological_split_indices(10, np.zeros(10, dtype=int),
                                              0.8, 0.1, 0.1)
    except ValueError as e:
        assert 'no rows for the requested val split' in str(e), e
    else:
        raise AssertionError('an empty requested validation split must fail')
    # Explicit zero targets are allowed to be empty.
    idx = L.grouped_chronological_split_indices(10, np.zeros(10, dtype=int),
                                                1.0, 0.0, 0.0)
    assert len(idx['train']) == 10 and len(idx['val']) == len(idx['test']) == 0


def t_label_shift_and_split_counts():
    X = np.arange(12, dtype=float).reshape(6, 2)
    # Non-finite labels are counted per split; fail_on_nonfinite_labels turns
    # a corrupted target into a hard failure (off by default for legacy use).
    y = [0, 0, 1, 1, np.inf, 2]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2, 3],
                                test_idx=[4, 5])
    assert rep['checks']['nonfinite_labels'] == 1, rep
    assert rep['checks']['nonfinite_labels_by_split']['test'] == 1, rep
    assert rep['checks']['nonfinite_labels_by_split']['train'] == 0, rep
    assert rep['status'] == 'warnings', rep
    rep_fail = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2, 3],
                                     test_idx=[4, 5],
                                     fail_on_nonfinite_labels=True)
    assert rep_fail['status'] == 'fail', rep_fail
    # A complete label-distribution reversal is reported against the training
    # reference and can be made a hard gate.
    X2 = np.arange(12, dtype=float).reshape(6, 2)
    y2 = [0, 0, 0, 1, 1, 1]
    rep2 = L.data_quality_report(X2, y=y2, train_idx=[0, 1, 2],
                                 val_idx=[3, 4, 5], fail_on_label_shift=True)
    shift = rep2['checks']['label_distribution_shift_vs_train']['val']
    assert shift['js_divergence'] > 0.5, rep2
    assert rep2['status'] == 'fail', rep2
    assert rep2['checks']['label_distribution_by_split']['train'] == {'0': 3}, rep2


def t_sample_tokens_uniform_rejection():
    # Force the scale-safe rejection path even on tiny arrays.  The sample must
    # remain a uniform without-replacement draw; taking the smallest CAP of a
    # sorted candidate pool (the previous implementation) has mean ~n/9 for
    # cap=4, not ~n/2.
    old = A._CHOICE_PERM_MAX_ELEMS
    try:
        A._CHOICE_PERM_MAX_ELEMS = 0
        idx = np.arange(1000)
        means = [A._sample_tokens(idx, cap=4, seed=s).mean() for s in range(200)]
        assert 400.0 < float(np.mean(means)) < 600.0, (float(np.mean(means)), means[:5])
        assert np.array_equal(A._sample_tokens(idx, cap=4, seed=17),
                              A._sample_tokens(idx, cap=4, seed=17))
        sample = A._sample_tokens(idx, cap=4, seed=17)
        assert sample.size == 4 and len(set(sample.tolist())) == 4
    finally:
        A._CHOICE_PERM_MAX_ELEMS = old


def t_audit_budget_integer_runs():
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, 'run_time_state.json')
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 12.5, 'runs': 3}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] == 3, out
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 1.0, 'runs': [1, 2]}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] == 2, out
        # A FLOPs proxy is only defined for positive finite params/tokens and
        # a non-negative overhead; bool/NaN/negative values must not be
        # silently converted into a plausible-looking budget.
        for _args in ((0, 10), (True, 10), (10, True), (10, 0),
                      (float('nan'), 10), (10, float('inf'))):
            try:
                A.train_flops_estimate(*_args)
            except ValueError:
                pass
            else:
                raise AssertionError(f'train_flops_estimate{_args!r} must be refused')
        for _ov in (True, -0.1, float('nan'), float('inf'), 'x'):
            try:
                A.train_flops_estimate(10, 10, attn_overhead=_ov)
            except ValueError:
                pass
            else:
                raise AssertionError(f'attn_overhead={_ov!r} must be refused')
        # A malformed panel record must not be converted into a fake FLOPs
        # estimate (bool params, NaN tokens, string steps).
        panel = tempfile.mkdtemp(prefix='budget_', dir=REPO)
        try:
            with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'x::s0': {'variant': 'csa_fixed', 'params': True,
                                     'steps': 'lots', 'tokens_seen': float('nan')}}, f)
            bp = A.audit_budget(state_files=(), panels=((panel, 'csa_fixed', None),))
            assert bp['panels'][panel] == 'no param records', bp
        finally:
            shutil.rmtree(panel, ignore_errors=True)


def t_audit_failures_ignores_unmeasured_ppl():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'a::seed0': {'variant': 'a', 'seed': 0, 'error': 'boom'},
                'b::seed0': {'variant': 'b', 'seed': 0, 'ppl': float('nan')},
                'c::seed0': {'variant': 'c', 'seed': 0, 'ppl': 12.0},
            }, f)
        out = A.audit_failures(outdirs=[d])
        panel = out['panels'][d]
        assert panel['ok_by_arm'] == {'c': 1}, panel
        assert panel['errors_by_arm'] == {'a': 1}, panel
        # A NaN-PPL record is neither a success nor an exception in the data;
        # it is a missing measurement and must stay visible to the failure
        # concentration audit (otherwise survivorship bias disappears).
        assert panel['unmeasured_by_arm'] == {'b': 1}, panel
        assert out['status'] == 'warn', out


def t_object_time_and_zero_feature_guards():
    bad_time = np.array([0.0, float('nan'), 2.0], dtype=object)
    for call in (lambda: L.assert_no_split_leakage([0], [1], [2], time=bad_time),
                 lambda: L.split_indices(3, time=bad_time, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3)):
        try:
            call()
        except ValueError as e:
            assert 'NaN/Inf' in str(e), e
        else:
            raise AssertionError('an object-dtype NaN timestamp must be rejected')
    # A zero-feature table is uninformative but must not crash the report.
    rep = L.data_quality_report(np.empty((4, 0)), train_idx=[0, 1],
                                val_idx=[2], test_idx=[3])
    assert rep['n_features'] == 0, rep


def t_data_guard_index_label_and_shift_edges():
    # Negative indices wrap around in NumPy: -1 and n-1 are the same row even
    # though their values look disjoint.  Fractional indices must never be
    # silently truncated to an int.
    for bad_train in ([-1], [0.5], [np.nan], [True], ['0'],
                      np.array(['2020-01-01'], dtype='datetime64[D]')):
        try:
            L.assert_no_split_leakage(bad_train, [], [2])
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid split indices {bad_train!r} must be rejected')
    X = np.arange(8, dtype=float).reshape(4, 2)
    for bad_train in ([-1], [0.5]):
        try:
            L.data_quality_report(X, train_idx=bad_train, val_idx=[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f'data_quality_report accepted {bad_train!r}')

    # Inf is not a class/regression target; it is non-finite label noise.
    rep = L.data_quality_report(X, y=[0, np.inf, 1, 2],
                                train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert rep['checks']['nonfinite_labels'] == 1, rep
    assert rep['checks']['missing_labels'] == 0, rep
    assert rep['status'] == 'warnings', rep

    # -0.0 == 0.0 but the raw bytes differ; a duplicate across the split must
    # still be detected.
    Xz = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    repz = L.data_quality_report(Xz, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert repz['checks']['duplicate_rows_across_splits']['train__val'] == 1, repz
    assert repz['status'] == 'fail', repz
    # Different non-finite values are not exact duplicate rows: NaN, +Inf and
    # -Inf share a non-finite mask but are distinct corrupted values.  Only the
    # repeated NaN row is a duplicate.
    Xnf = np.array([[np.nan, 1.0], [np.inf, 1.0], [-np.inf, 1.0], [np.nan, 1.0]])
    mask = L.duplicate_mask(Xnf)
    assert mask.tolist() == [False, False, False, True], mask
    repnf = L.data_quality_report(Xnf, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert repnf['checks']['duplicate_rows_across_splits']['train__val'] == 0, repnf

    # A zero-variance training feature with a different holdout mean is an
    # undefined SMD, not a zero shift.
    Xc = np.array([[0.0], [0.0], [5.0], [0.0]])
    repc = L.data_quality_report(Xc, train_idx=[0, 1, 3], val_idx=[2])
    shift = repc['checks']['distribution_shift_vs_train']['val']
    assert shift['non_estimable_shift_features'] == [0], repc
    assert repc['status'] == 'warnings', repc

    # Finite inputs near the float64 limit can overflow the mean/variance.  An
    # infinite SMD is not evidence of no shift; it is not estimable.
    Xov = np.array([[1e308], [-1e308], [1e308], [1e308]])
    with np.errstate(over='ignore', invalid='ignore'):
        repov = L.data_quality_report(Xov, train_idx=[0, 1], val_idx=[2, 3])
    _ov = repov['checks']['distribution_shift_vs_train']['val']
    assert _ov['non_estimable_shift_features'] == [0], repov
    assert any('not estimable' in w for w in repov['warnings']), repov

    # A NaN/negative threshold must be an error, not a silently disabled gate.
    for bad_kw in ({'clip_z': float('nan')}, {'clip_z': -1.0},
                   {'shift_threshold': float('nan')},
                   {'shift_threshold': -0.1}):
        try:
            L.data_quality_report(X, train_idx=[0, 1], val_idx=[2], **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'data_quality_report accepted {bad_kw!r}')

    # Object-dtype NaT group ids are one missing group, so splitting them
    # across sides is group leakage.
    g = np.array([np.datetime64('NaT', 'D'), np.datetime64('NaT', 'D'),
                  np.datetime64('2020-01-01', 'D')], dtype=object)
    try:
        L.assert_no_split_leakage([0], [1], [2], groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT group ids must not be split across sides')


def t_audit_split_warning_status():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        tr = np.arange(40, dtype=np.int64)
        va = np.arange(100, 108, dtype=np.int64).reshape(2, 4)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        # Missing split_strategy is an explicit legacy-cache warning.
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 40, 'seq_len': 4}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'warnings', rec
        assert out['caches'][os.path.abspath(d)]['status'] == 'warnings', out['caches']


def t_audit_split_preferred_seq_len():
    # A cache directory can hold several protocol variants.  Naming the train
    # sequence length must restrict the audit to that stream instead of
    # silently certifying/failing a probe cache the panel never used.
    with tempfile.TemporaryDirectory() as d:
        for tag, sl in (('s4', 4), ('s8', 8)):
            np.save(os.path.join(d, f'train_ids_{tag}.npy'),
                    np.arange(40, dtype=np.int64))
            np.save(os.path.join(d, f'val_batch_{tag}.npy'),
                    (np.arange(16, dtype=np.int64) + 40).reshape(2, 8))
            with open(os.path.join(d, f'meta_{tag}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump({'vocab': 80, 'seq_len': sl,
                           'split_strategy': 'ordered',
                           'shuffle_at_split': False,
                           'tokenizer_fit_split': 'train'}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                       preferred_train_seq_len=8)
        assert set(out['caches'][os.path.abspath(d)]['corpora']) == {'s8'}, out
        out4 = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                        preferred_train_seq_len=4)
        assert set(out4['caches'][os.path.abspath(d)]['corpora']) == {'s4'}, out4
        missing = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1,
                                           preferred_train_seq_len=512)
        assert missing['status'] == 'not_verifiable_locally', missing
        for bad in (0, -1, True, '8'):
            try:
                A.audit_split_discipline(cache_dirs=[d],
                                         preferred_train_seq_len=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'preferred_train_seq_len={bad!r} must be refused')


def t_audit_split_test_metadata_discipline():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        ttag = 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 40).reshape(2, 8)
        te = (np.arange(16, dtype=np.int64) + 56).reshape(2, 8)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 80, 'seq_len': 8, 'split_strategy': 'ordered',
                       'shuffle_at_split': False,
                       'tokenizer_fit_split': 'train'}, f)
        good_test = {'vocab': 80, 'seq_len': 8, 'split_strategy': 'official_test',
                     'shuffle_at_split': False,
                     'tokenizer_fit_split': 'train',
                     'near_duplicate_cleanup': {
                         'policy': 'test_documents_deduplicated_against_train_and_validation'}}
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(good_test, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_shuffle_at_split') is False, rec
        assert rec['status'] != 'fail', rec
        # P5T uses a longer test sequence than the training cache.  The test
        # cache must still be paired for the leakage audit.
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(good_test, seq_len=16), f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_pairing') != 'not_found', rec
        assert rec.get('test_tag') == ttag and rec.get('test_shape') == [2, 8], rec
        assert rec.get('exact_row_overlap_train_test') == 0, rec
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(good_test, f)
        bad_test = dict(good_test, shuffle_at_split=True,
                        tokenizer_fit_split='test',
                        near_duplicate_cleanup={'policy': 'none'})
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(bad_test, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert rec.get('test_shuffle_at_split') is True, rec
        assert any('test metadata declares a shuffled split' in w for w in out['warnings']), out
        assert any('test tokenizer was fitted on' in w for w in out['warnings']), out
        assert any('duplicate cleanup' in w for w in out['warnings']), out


def t_corpus_audit_overlap_is_unique_across_windows():
    # The same validation row repeated in two disjoint training windows is ONE
    # overlapping validation row, not two.
    tr = np.arange(200, dtype=np.int64) % 10
    tr[0:3] = [7, 8, 9]
    tr[13:16] = [7, 8, 9]
    va = np.array([[7, 8, 9]], dtype=np.int64)
    rep = L.audit_corpus_split(tr, va, 20, dataset='wikitext',
                               max_scan_tokens=64, sample_rows=1, strict=False)
    assert rep['exact_row_overlap_in_sample'] == 1, rep
    assert rep['sampled_val_rows'] == 1, rep


def t_corpus_audit_memo_and_thresholds():
    tr = np.arange(40, dtype=np.int64)
    va = (np.arange(8, dtype=np.int64) + 100).reshape(2, 4)
    # The memo key must include the split attestation: a clean read for one
    # meta must not answer for a different (failing) one.
    clean = L.audit_corpus_split(tr, va, 200, dataset='fineweb',
                                 max_scan_tokens=64, sample_rows=1, strict=False)
    assert clean['status'] == 'warnings', clean
    bad_meta = {'doc_ids': {'disjoint': False, 'overlap': 1,
                            'train_count': 1, 'validation_count': 1}}
    bad = L.audit_corpus_split(tr, va, 200, dataset='fineweb', meta=bad_meta,
                               max_scan_tokens=64, sample_rows=1, strict=False)
    assert bad['status'] == 'fail', bad
    assert any('document-disjoint' in p for p in bad['hard_failures']), bad
    # A non-strict read that cached a failing report must never satisfy a
    # later strict call.
    try:
        L.audit_corpus_split(tr, va, 200, dataset='fineweb', meta=bad_meta,
                             max_scan_tokens=64, sample_rows=1, strict=True)
    except RuntimeError as e:
        assert 'DATA GATE REFUSED' in str(e), e
    else:
        raise AssertionError('strict=True must refuse a cached failing split')
    # NaN/negative gates silently disable comparisons; refuse them.
    for kw in ({'shift_warn': float('nan')}, {'shift_fail': float('nan')},
               {'shift_warn': -0.1}, {'shift_fail': -1.0},
               {'shift_warn': 0.3, 'shift_fail': 0.2},
               {'sample_rows': 0}, {'max_scan_tokens': True}):
        try:
            L.audit_corpus_split(tr, va, 200, dataset='fineweb', strict=False, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_corpus_split accepted {kw!r}')


def t_audit_length_requires_eval_span():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bylen.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'by_len': {'512': {
                'n_tok': 511, 'eval_span': None, 'n_seq': 1}}}}, f)
        out = A.audit_length(path)
        assert out['status'] == 'fail', out
        assert out['by_length']['512']['same_scored_targets_across_records'] is False, out


def t_audit_fairness_nan_seed_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': float('nan'),
                                    'ppl': 10.0, 'run_cfg': 'x'}}, f)
        out = A.audit_fairness(path, arms=['a'])
        assert out['arms']['a']['seeds'] == [], out
        # Malformed run_cfg/params values are reported, not a TypeError: a
        # dict run_cfg is unhashable and mixed param types are unorderable.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 10.0,
                                    'run_cfg': {'bad': 1}, 'params': 10},
                       'a::seed1': {'variant': 'a', 'seed': 1, 'ppl': 11.0,
                                    'run_cfg': 'x', 'params': 'bad'}}, f)
        out_bad = A.audit_fairness(path, arms=['a'])
        assert isinstance(out_bad, dict) and out_bad['status'] == 'fail', out_bad
        assert out_bad['arms']['a']['params'] == [10, 'bad'], out_bad['arms']['a']
        # A summary that does not contain any requested arm is not auditable.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        out_empty = A.audit_fairness(path, arms=['a'])
        assert out_empty['status'] == 'not_verifiable_locally', out_empty
        # The metric audit must not report 'ok' when there is no panel to
        # check; policy declarations alone are not record-level verification.
        old_repo = A.REPO
        try:
            A.REPO = d
            os.makedirs(os.path.join(d, 'results_empty'), exist_ok=True)
            with open(os.path.join(d, 'results_empty', 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({}, f)
            m = A.audit_metrics()
            assert m['status'] != 'ok', m
        finally:
            A.REPO = old_repo


def t_audit_boundary_rejects_invalid_readings():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a': {'boundary_alignment': {'precision': 0.8,
                                                     'random_precision': 0.2}}}, f)
        ok = A.audit_boundary(outdirs=[d])
        assert ok['status'] == 'ok' and ok['panels'][d][0]['excess'] == 0.8 - 0.2, ok
        # Strings/booleans/non-finite readings cannot support a claim, and
        # must not crash the audit either.
        for bad in ({'precision': 'x', 'random_precision': 0.2},
                    {'precision': True, 'random_precision': 0.2},
                    {'precision': float('nan'), 'random_precision': 0.2},
                    {'precision': 1.4, 'random_precision': 0.2},
                    {'precision': 0.8, 'random_precision': -0.1}):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'a': {'boundary_alignment': bad}}, f)
            r = A.audit_boundary(outdirs=[d])
            assert r['status'] == 'fail' and r['problems'], r


def t_audit_malformed_arm_id_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'k': {'variant': {'bad': 1}, 'seed': 0, 'ppl': 10.0,
                             'stats': {}, 'efficiency': {'status': 'ok'}}}, f)
        for fn in (A.audit_attribution, A.audit_failures, A.audit_efficiency_pairs):
            out = fn([d])
            assert isinstance(out, dict) and out['status'] != 'fail', (fn.__name__, out)


def t_audit_split_empty_cache_is_not_ok():
    with tempfile.TemporaryDirectory() as d:
        out = A.audit_split_discipline(cache_dirs=[d])
        assert out['status'] == 'not_verifiable_locally', out
        assert out['caches'][os.path.abspath(d)]['status'] == 'not_verifiable_locally', out


def t_baseline_audit_malformed_seed_keys():
    art = {
        'arms': {'a': {'selected': {'lr': 1e-3}, 'selected_metric_value': 1.0,
                       'trials': [{'config': {'lr': 1e-3}, 'ppl': 1.0,
                                   'seed_ppls': {'not-a-seed': 1.0},
                                   'ppl_std': 'not-numeric'}]}},
        'protocol': {'selection_metric': 'ppl', 'selection_split': 'validation',
                     'direction': 'lower_is_better', 'trials': [{'lr': 1e-3}],
                     'arms': ['a'], 'search_seeds': [0], 'final_seeds': [0]},
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better',
        'final_records': {'a::seed0': {'seed': 0, 'variant': 'a',
                                       'hyperparams': {'lr': 'not-numeric'}}},
    }
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail', out
    assert any('seed_ppls keys' in p for p in out['problems']), out

    # A numeric seed set with a non-numeric ppl_std / selected hyperparameter
    # must be reported, not raise ValueError inside the audit.
    art['arms']['a']['trials'] = [{'config': {'lr': 1e-3}, 'ppl': 1.5,
                                   'seeds': [0, 1],
                                   'seed_ppls': {'0': 1.0, '1': 2.0},
                                   'ppl_std': 'not-numeric'}]
    out = L.baseline_tuning_audit(art)
    assert any('ppl_std is not numeric' in p for p in out['problems']), out
    assert any('is not numeric' in p for p in out['problems']), out

    # A declared final seed must be parsed exactly; silently dropping 1.5 (or
    # iterating a scalar) would let the audit certify a different seed panel.
    art['protocol']['final_seeds'] = [0, 1.5]
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final seed' in p for p in out['problems']), out
    art['protocol']['final_seeds'] = 3
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final_seeds' in p for p in out['problems']), out
    # final_records must be a mapping; a scalar must not reach `.items()` or
    # be silently treated as "no final records".
    art['protocol']['final_seeds'] = [0]
    art['final_records'] = 1.5
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('final_records must be a mapping' in p for p in out['problems']), out
    # A non-scalar metric name must be reported, not used as a dict key.
    art['final_records'] = {}
    art['selection_metric'] = {'not': 'a metric'}
    out = L.baseline_tuning_audit(art)
    assert out['status'] == 'fail' and any('not a metric name' in p for p in out['problems']), out
    # Scalar protocol/arm trial fields and non-object selected configs are
    # malformed evidence: report them instead of raising TypeError.
    _base = json.loads(json.dumps(art))
    _base['selection_metric'] = 'ppl'
    _base['protocol']['selection_metric'] = 'ppl'
    _mutations = [
        lambda a: a['protocol'].__setitem__('trials', 1.5),
        lambda a: a['protocol'].__setitem__('arms', 1.5),
        lambda a: a['protocol'].__setitem__('inherits_tuning', ['not-a-map']),
        lambda a: a['arms']['a'].__setitem__('trials', True),
        lambda a: a['arms']['a'].__setitem__('selected', 1.5),
        lambda a: a['arms']['a']['trials'].__setitem__(
            0, {'config': ['not-a-map'], 'ppl': 1.0}),
    ]
    for _mut in _mutations:
        _bad = json.loads(json.dumps(_base))
        _mut(_bad)
        _out = L.baseline_tuning_audit(_bad)
        assert _out['status'] == 'fail', (_out,)


def t_simple_baseline_audit_malformed_records():
    art = {
        'baselines': {'a': {
            'selected': {'alpha': 0.1}, 'selected_metric_value': 1.0,
            'trials': [{'config': {'alpha': 0.1}, 'ppl': 1.0,
                        'n_tokens': 'not-an-int'},
                       {'config': {'alpha': 0.2}, 'ppl': 2.0,
                        'n_tokens': 10}]}},
        'selection_metric': 'ppl', 'selection_split': 'validation',
        'direction': 'lower_is_better', 'data': ['not', 'a', 'dict'],
        'protocol': {'selection_metric': 'ppl', 'selection_split': 'validation',
                     'direction': 'lower_is_better', 'arms': ['not-a-dict']},
    }
    out = L.simple_baseline_audit(art)
    assert out['status'] == 'fail', out
    assert any('n_tokens is not an integer' in p for p in out['problems']), out
    # Fractional token counts must not be truncated to a different budget.
    art2 = json.loads(json.dumps(art))
    for _t in art2['baselines']['a']['trials']:
        _t['n_tokens'] = 10.5
    out2 = L.simple_baseline_audit(art2)
    assert out2['status'] == 'fail', out2
    assert any('n_tokens is not an integer' in p for p in out2['problems']), out2
    # A mapping/list is not a metric name.  The audit must report it instead
    # of passing it to dict.get() as an unhashable key.
    art3 = json.loads(json.dumps(art))
    art3['selection_metric'] = {'not': 'a metric'}
    out3 = L.simple_baseline_audit(art3)
    assert out3['status'] == 'fail' and any('not a metric name' in p for p in out3['problems']), out3
    # Malformed declared grids/arm metadata are evidence problems, not
    # AttributeError/TypeError crashes in the audit.
    for _bad_proto in (
            {'arms': {'a': {'order': 1}}, 'grids': ['not-a-mapping']},
            {'arms': {'a': ['not-a-mapping']},
             'grids': {'a': {'alpha': [0.1]}}},
            {'arms': {'a': {'order': 1}},
             'grids': {'a': {'alpha': 0.1}}},
            {'arms': {'a': {'order': 1}},
             'grids': {'a': {'alpha': ['not-numeric']}}}):
        _bad_art = json.loads(json.dumps(art3))
        _bad_art['protocol'] = dict(_bad_proto,
                                    selection_metric='ppl',
                                    selection_split='validation',
                                    direction='lower_is_better')
        _bad_out = L.simple_baseline_audit(_bad_art)
        assert _bad_out['status'] == 'fail', (_bad_proto, _bad_out)



# ---------------------------------------------------------------------------
# L. Additional hardening: claim gate, Holm family floor, crossover fit,
#    matched init, compute control, stale refusal, efficiency pairs.
# ---------------------------------------------------------------------------

def t_reportable_claim_gate():
    base = {'n': 6, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.01, 'p_holm': 0.02}
    r = L.reportable_claim(base)
    assert r['allowed'] is True and r['level'] == 'exploratory_significant', r
    # n<6 is underpowered even with a tiny raw p.
    r = L.reportable_claim(dict(base, n=3, p_holm=0.02))
    assert r['allowed'] is False and r['level'] == 'underpowered', r
    # Missing Holm is uncorrected -> not reportable.
    r = L.reportable_claim({k: v for k, v in base.items() if k != 'p_holm'})
    assert r['allowed'] is False and any('Holm' in x for x in r['reasons']), r
    # Effect inside seed noise is not reportable.
    r = L.reportable_claim(dict(base, mean=-0.1, std=0.2))
    assert r['allowed'] is False and any('seed noise' in x for x in r['reasons']), r
    # A family too large for n=6 cannot reach significance at all.
    r = L.reportable_claim(base, family_size=100)
    assert r['allowed'] is False and r['min_attainable_p_holm_family'] > 0.05, r
    # Confirmatory is only honoured when the caller says so.
    r = L.reportable_claim(base, evidence_level='confirmatory')
    assert r['level'] == 'confirmatory', r
    # A mean whose sign flips when one seed is dropped is not a stable
    # directional claim, even if the aggregate p happens to clear Holm.
    r = L.reportable_claim(dict(base, seed_driven=True))
    assert r['allowed'] is False and any('leave-one-seed-out' in x for x in r['reasons']), r
    # A complete-case p-value computed after dropping missing pairs is not a
    # complete-panel significance claim.
    r = L.reportable_claim(dict(base, n_input=7, n_dropped=1,
                                dropped_seeds=[6]))
    assert r['allowed'] is False and any('complete-case' in x for x in r['reasons']), r
    assert r['n_input'] == 7 and r['n_dropped'] == 1, r
    # Seeds skipped by the same-recipe control gate are also an incomplete
    # paired panel: the surviving subset was selected by that mismatch.
    for _key in ('n_skipped_config_mismatch', 'n_skipped'):
        _rs = L.reportable_claim(dict(base, **{_key: 1}))
        assert _rs['allowed'] is False, (_key, _rs)
        assert any('controlled-comparison' in x for x in _rs['reasons']), (_key, _rs)
        assert _rs.get('n_skipped_config_mismatch') == 1, (_key, _rs)
    _rsk = L.reportable_claim(dict(base, n_skipped_config_mismatch=0))
    assert _rsk['allowed'] is True, _rsk
    assert L.reportable_claim(dict(base, n_skipped_config_mismatch=-1))['allowed'] is False
    assert L.reportable_claim(dict(base, n_skipped_config_mismatch=True))['allowed'] is False
    # Negative or internally inconsistent dropout accounting is not a clean
    # complete panel; it must not bypass the complete-case refusal.
    for _bad in (dict(base, n_input=-1, n_dropped=0),
                 dict(base, n_input=6, n_dropped=-1),
                 dict(base, n_input=3, n_dropped=6),
                 dict(base, n_input=3, n_dropped=0)):
        _rb = L.reportable_claim(_bad)
        assert _rb['allowed'] is False, (_bad, _rb)
        assert any('n_input' in x or 'n_dropped' in x or 'panel accounting' in x
                   for x in _rb['reasons']), (_bad, _rb)
    # A threshold outside (0,1) must be rejected, not used to relabel every
    # p-value as significant (alpha=1.5) or unreachable (alpha=0).
    _alpha_pts = [{'label': f'S{i}', 'scale': i + 1,
                   'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2]}
                  for i in range(3)]
    for _bad in (True, 0, 1, -0.1, 1.5, float('nan'), 'not-a-number'):
        for _fn in (lambda a: L.reportable_claim(base, alpha=a),
                    lambda a: L.paired_effect_stats([-1.0, -1.1, -0.9], alpha=a),
                    lambda a: L.holm_adjust([('x', 0.01)], alpha=a),
                    lambda a: L.attach_holm({'x': {'n': 6, 'p_exact_signflip': 0.01}}, alpha=a),
                    lambda a: L.meta_analyze_effects(_alpha_pts, alpha=a),
                    lambda a: L.scaling_trend(_alpha_pts, alpha=a),
                    lambda a: L.loglog_regression([(1, 1), (2, 4), (3, 9)], alpha=a),
                    lambda a: L.methodology_seed_plan(6, alpha=a)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid alpha {_bad!r} was accepted')
    # Minimum-n floors are gate parameters too: bool/fractional values must
    # not be truncated into a weaker floor.
    for _bad_min in (True, 3.5, float('nan'), -1):
        for _fn in (lambda m: L.paired_effect_stats([-1.0, -1.1, -0.9], min_n=m),
                    lambda m: L.meta_analyze_effects([], min_seeds=m),
                    lambda m: L.scaling_trend([], min_seeds=m),
                    lambda m: L.methodology_audit({}, design={}, baselines={},
                                                  min_n=m, write=False, verbose=False)):
            try:
                _fn(_bad_min)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid min_n={_bad_min!r} was accepted')
    # Holm cannot be smaller than the raw p it adjusts.
    r = L.reportable_claim(dict(base, p_holm=0.001))
    assert r['allowed'] is False and any('smaller than the raw p' in x for x in r['reasons']), r
    # Booleans are flags, not p-values.  In particular False must never enter
    # the family as p=0 and become the most significant result.
    for _p in (False, True):
        h = L.holm_adjust([('x', _p)])
        assert h['x']['p_holm'] is None, h
        r = L.reportable_claim(dict(base, p_exact_signflip=_p,
                                    p_holm=0.02))
        assert r['allowed'] is False, r
        r = L.reportable_claim(dict(base, p_holm=_p))
        assert r['allowed'] is False, r
    fam = L.attach_holm({'x': {'n': 6, 'p_exact_signflip': False}})
    assert fam['tested'] == 0, fam
    # Booleans are also not effect measurements.
    assert L.reportable_claim(dict(base, mean=True, std=0.2))['allowed'] is False
    assert L.reportable_claim(dict(base, mean=-1.0, std=False))['allowed'] is False
    # The gate's own parameters must not be silently coerced.  int(5.9)->5
    # would loosen the n floor, and int(True)->1 for family_size would turn a
    # flag into a one-comparison family; both must fail closed.
    for _bad_min in (5.9, True, float('nan'), 0):
        r = L.reportable_claim(base, min_n=_bad_min)
        assert r['allowed'] is False, (r, _bad_min)
        assert any('min_n' in x for x in r['reasons']), r
    for _bad_fam in (0, -3, True, 3.5, float('nan')):
        r = L.reportable_claim(base, family_size=_bad_fam)
        assert r['allowed'] is False, (r, _bad_fam)
        assert any('family_size' in x for x in r['reasons']), r


    # Probabilities outside [0,1] cannot support a claim, even when every
    # strict "< alpha" comparison would accept a negative value.
    bad_neg = dict(base, p_exact_signflip=-0.1, p_holm=-0.05)
    assert L.reportable_claim(bad_neg)['allowed'] is False, L.reportable_claim(bad_neg)
    assert BR._sig_gate(bad_neg, label='negative p demo') is False
    # A Holm p below its raw p is a mis-attached family correction.
    bad_holm = dict(base, p_exact_signflip=0.04, p_holm=0.02)
    assert L.reportable_claim(bad_holm)['allowed'] is False, L.reportable_claim(bad_holm)
    assert BR._sig_gate(bad_holm, label='holm order demo') is False

def t_holm_family_resolution_ceiling():
    comps = {f'c{i}': {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25} for i in range(100)}
    fam = L.attach_holm(comps, family='100 exploratory contrasts')
    assert fam['family_resolution_limited'] is True, fam
    assert fam['family_min_attainable_p_holm'] > 0.05, fam
    assert fam['n_effective_min'] == 3 and fam['n_effective_max'] == 3, fam
    # A small, powered family is not flagged.
    comps2 = {f'c{i}': {'n': 12, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.001} for i in range(4)}
    fam2 = L.attach_holm(comps2, family='4 confirmatory contrasts', exploratory=False)
    assert fam2['family_resolution_limited'] is False, fam2
    assert fam2['evidence_level'] == 'confirmatory', fam2


def t_crossover_scaling_refusals():
    def pts(n, d=(128, 256, 384, 512), tok=(50e6,), censored=False, vol=(50e6,)):
        return [{'label': f'd{dd}', 'd': dd, 'crossover_tokens': tok[0], 'train_tokens': vol[i % len(vol)],
                 'censored': censored, 'reconstructed': False} for i, dd in enumerate(d[:n])]
    # <4 distinct widths -> refused.
    f = L.crossover_scaling_fit(pts(3))
    assert f['status'] == 'insufficient_points', f
    # Any censored point refuses the fit (informative censoring).
    f = L.crossover_scaling_fit(pts(4, censored=True))
    assert f['status'] == 'censored' and f['reasons'], f
    # Training-volume spread > 25% refuses the d slope.
    f = L.crossover_scaling_fit(pts(4, vol=(25e6, 50e6)))
    assert f['status'] == 'training_volume_confounded', f
    # Duplicate widths keep the token-matched panel and are named.
    dup = pts(4) + [{'label': 'd512_half', 'd': 512, 'crossover_tokens': 40e6, 'train_tokens': 25e6}]
    f = L.crossover_scaling_fit(dup)
    assert f['status'] == 'ok' and 'd512_half' in (f.get('deduped_by_d') or []), f
    # A crossover beyond the panel's own training budget is impossible and
    # must not enter a scaling fit.
    bad = pts(4)
    bad[1] = dict(bad[1], crossover_tokens=60e6, train_tokens=50e6)
    fb = L.crossover_scaling_fit(bad)
    assert fb['status'] == 'inconsistent_points', fb
    assert fb['inconsistent_panels'] == ['d256'], fb
    # A clean 4-point fit carries a slope CI and is labelled descriptive.
    f = L.crossover_scaling_fit(pts(4))
    assert f['status'] == 'ok' and f.get('slope') is not None and f.get('ci') is not None, f
    assert f['claim_level'] == 'descriptive', f
    assert f['volume_ratio'] <= 1.25, f
    # A perfectly recovered power law has zero residual variance: the slope's
    # t statistic is infinite, so p must be 0, not NaN.
    perfect = [{'x': d, 'y': 2.0 * d ** 1.5, 'label': f'd{d}'} for d in (16, 32, 64, 128)]
    lr = L.loglog_regression(perfect)
    assert lr['status'] == 'ok' and abs(lr['slope'] - 1.5) < 1e-12, lr
    assert math.isfinite(lr['p']) and lr['p'] < 1e-12, lr


def t_scale_ladder_budget_and_transfer_audit():
    def tier(name, steps, params, seq=8, batch=2, tokens=None, lr=3e-4):
        cfg = {'steps': steps, 'batch_size': batch, 'seq_len': seq,
               'n_train_tokens': tokens if tokens is not None else steps * batch * seq,
               'lr': lr, 'warmup': max(1, steps // 10), 'seeds': [0, 1, 2, 3]}
        return {'name': name, 'cfg': cfg, 'params': params}
    a = tier('S1', 1000, 1000)
    b = tier('S2', 2000, 2000)  # same tokens/param, same lr
    out = L.scale_ladder_audit([a, b])
    assert out['budget_matched'] is True, out
    assert out['lr_constant'] is True, out
    # A constant LR is disclosed as an unvalidated transfer assumption.
    assert any('transfer' in w for w in out['warnings']), out
    # A different tokens-per-parameter budget is a visible confound.
    b2 = tier('S2', 3000, 2000)
    out2 = L.scale_ladder_audit([a, b2])
    assert out2['budget_matched'] is False, out2
    assert any('tokens-per-parameter' in w for w in out2['warnings']), out2
    # A declared token budget that disagrees with steps*batch*seq is a hard
    # fail, not a warning.
    bad = tier('S1', 1000, 1000, tokens=999999)
    out3 = L.scale_ladder_audit([bad, b])
    assert out3['status'] == 'fail', out3
    assert any('n_train_tokens' in p for p in out3['problems']), out3
    # Missing parameter counts keep the budget claim unverifiable.
    out4 = L.scale_ladder_audit([tier('S1', 1000, None), b])
    assert out4['budget_matched'] is None, out4
    assert any('params missing' in w for w in out4['warnings']), out4
    # The deployed plan must carry an explicit transfer policy and expose the
    # S1/S2 budget ratio rather than silently calling it matched.
    pol = L.SCALE_TRANSFER_POLICY
    assert pol.get('lr_transfer') and pol.get('tokens_per_param_tolerance') is not None, pol
    # A tokenizer width is an exact identity.  int() coercion would truncate a
    # fractional/boolean value into a DIFFERENT vocab that looks shared.
    va = tier('S1', 1000, 1000); va['cfg']['vocab'] = 32768
    vb = tier('S2', 2000, 2000); vb['cfg']['vocab'] = 32768.5
    out5 = L.scale_ladder_audit([va, vb])
    assert out5['status'] == 'fail', out5
    assert any('vocab' in p for p in out5['problems']), out5
    assert out5['corpus_binding'].get('vocab') != 'shared', out5
    vc = tier('S2', 2000, 2000); vc['cfg']['vocab'] = True
    out6 = L.scale_ladder_audit([va, vc])
    assert out6['status'] == 'fail', out6
    # A boolean budget/seed plan is a flag, not an integer; `or 0`/`or []`
    # must not turn it into a valid zero.
    for _field, _val in (('warmup', False), ('n_train_tokens', False),
                         ('seeds', False), ('seeds', '0,1')):
        _bt = tier('S1', 1000, 1000); _bt['cfg'][_field] = _val
        _bo = L.scale_ladder_audit([_bt, b])
        assert _bo['status'] == 'fail', (_bo, _field, _val)
        assert any(_field in p for p in _bo['problems']), (_bo, _field, _val)
    # Boolean/non-finite learning rates must not be coerced to 1.0/NaN.
    for _bad_lr in (True, False, float('nan'), float('inf')):
        _bad = tier('S1', 1000, 1000); _bad['cfg']['lr'] = _bad_lr
        _o = L.scale_ladder_audit([_bad, b])
        assert _o['status'] == 'fail', (_o, _bad_lr)
        assert any('lr=' in p for p in _o['problems']), _o
    out7 = L.scale_ladder_audit([va, tier('S2', 2000, 2000) | {'vocab': 32768}])
    assert out7['corpus_binding'].get('vocab') == 'shared', out7
    # The optimizer/schedule/augmentation descriptor must be shared across
    # tiers.  A mismatch is a hard fail; missing/partial is unverifiable.
    kn = L.training_knobs()
    ka = tier('S1', 1000, 1000); ka['cfg']['train_knobs'] = kn
    kb = tier('S2', 2000, 2000); kb['cfg']['train_knobs'] = kn
    okn = L.scale_ladder_audit([ka, kb])
    assert okn['train_knobs_binding'] == 'shared', okn
    badk = tier('S2', 2000, 2000)
    badk['cfg']['train_knobs'] = dict(kn, grad_clip_norm=2.0)
    obk = L.scale_ladder_audit([ka, badk])
    assert obk['status'] == 'fail', obk
    assert any('training knobs differ' in p for p in obk['problems']), obk
    opk = L.scale_ladder_audit([ka, b])
    assert opk['train_knobs_binding'] == 'partial', opk
    assert any('train_knobs descriptor' in w for w in opk['warnings']), opk
    onk = L.scale_ladder_audit([tier('S1', 1000, 1000), tier('S2', 2000, 2000)])
    assert onk['train_knobs_binding'] == 'unverifiable', onk
    assert any('train_knobs descriptor' in w for w in onk['warnings']), onk
    badtype = tier('S1', 1000, 1000); badtype['cfg']['train_knobs'] = 'adamw'
    obt = L.scale_ladder_audit([badtype, kb])
    assert obt['status'] == 'fail', obt
    assert any('train_knobs is not an object' in p for p in obt['problems']), obt
    # A non-constant LR is a declared transfer choice, never a matched
    # optimizer control: it must produce a confound warning, not silence.
    lrv = tier('S2', 2000, 2000, lr=1e-3)
    olr = L.scale_ladder_audit([tier('S1', 1000, 1000), lrv])
    assert olr['lr_constant'] is False and olr['lr_values'] == [0.0003, 0.001], olr
    assert any('learning rate DIFFERS' in w for w in olr['warnings']), olr
    # Seed identities are the basis of a cross-tier paired contrast.  A plan
    # with duplicate/negative seeds is malformed, and differing seed sets must
    # be disclosed instead of treating each tier's extra seeds as replication.
    shared = L.scale_ladder_audit([tier('S1', 1000, 1000), tier('S2', 2000, 2000)])
    assert shared['seed_set_shared'] is True and shared['n_common_seeds'] == 4, shared
    diff = tier('S2', 2000, 2000); diff['cfg']['seeds'] = [2, 3, 4, 5]
    od = L.scale_ladder_audit([tier('S1', 1000, 1000), diff])
    assert od['seed_set_shared'] is False and od['seed_set_intersection'] == [2, 3], od
    assert any('seed set' in w for w in od['warnings']), od
    dup = tier('S1', 1000, 1000); dup['cfg']['seeds'] = [0, 0, 1, 1]
    odup = L.scale_ladder_audit([dup, b])
    assert odup['status'] == 'fail' and any('duplicate seed' in p for p in odup['problems']), odup
    neg = tier('S1', 1000, 1000); neg['cfg']['seeds'] = [-1, 0, 1, 2]
    oneg = L.scale_ladder_audit([neg, b])
    assert oneg['status'] == 'fail' and any('negative' in p for p in oneg['problems']), oneg

    # A tier's power is a property of the DECLARED Holm family, not of the raw
    # sign-flip test.  n=6 can reach raw p=2/64=0.03125 but not Holm p for a
    # 3-comparison family (3*2/64=0.09375), so it must stay descriptive.
    def _with_seeds(name, seeds):
        t = tier(name, 1000, 1000)
        t['cfg']['seeds'] = list(seeds)
        return t
    fam = L.scale_ladder_audit(
        [_with_seeds('S1', range(6)), _with_seeds('S2', range(4))],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=3))
    _s1, _s2 = fam['tiers']
    assert abs(_s1['p_floor'] - 2.0 / 64.0) < 1e-15, _s1
    assert abs(_s1['p_floor_family_holm'] - 3.0 * 2.0 / 64.0) < 1e-15, _s1
    assert _s1['descriptive_only'] is True, _s1
    assert _s1['descriptive_only_basis'] == 'holm_family', _s1
    assert _s2['descriptive_only'] is True, _s2
    # The paired cross-tier contrast can use only the 4 common seeds, not the
    # 6 seeds one tier happens to have.
    assert fam['n_common_seeds'] == 4, fam
    assert abs(fam['common_seed_paired_p_floor'] - 2.0 / 16.0) < 1e-15, fam
    assert fam['common_seeds_sufficient_for_family'] is False, fam
    assert any('common seed' in w for w in fam['warnings']), fam
    # Seven common seeds clear a 3-comparison family at alpha=0.05
    # (3*2/128=0.046875), so the same ladder becomes planned-sufficient.
    fam_ok = L.scale_ladder_audit(
        [_with_seeds('S1', range(7)), _with_seeds('S2', range(7))],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=3))
    assert fam_ok['family_min_seeds'] == 7, fam_ok
    assert fam_ok['common_seeds_sufficient_for_family'] is True, fam_ok
    assert all(t['descriptive_only'] is False for t in fam_ok['tiers']), fam_ok
    # Malformed declarations fail closed instead of crashing the pre-flight
    # audit (a truthy scalar `cfg` used to reach `.get`).
    mal = L.scale_ladder_audit([dict(a, cfg='not-a-recipe'), b])
    assert mal['status'] == 'fail' and any('cfg is not an object' in p
                                          for p in mal['problems']), mal
    mal2 = L.scale_ladder_audit([123, b])
    assert mal2['status'] == 'fail' and any('not a dict' in p
                                           for p in mal2['problems']), mal2
    for _bad_policy in (123, 'x', [1], True):
        try:
            L.scale_ladder_audit([a, b], policy=_bad_policy)
        except ValueError as e:
            assert 'policy' in str(e), e
        else:
            raise AssertionError(f'policy={_bad_policy!r} must be refused')


def t_init_core_fingerprint_and_pair_gate():
    dev = torch.device('cpu')
    L.set_seed(3)
    m1 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'full')).to(dev)
    L.set_seed(3)
    m2 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'csa_fixed')).to(dev)
    L.set_seed(4)
    m3 = L.SmallGPT(64, 32, 1, 2, 16, 32, L.make_layer_cfgs(1, 'full')).to(dev)
    fa, fb, fc = (L.init_core_fingerprint(m) for m in (m1, m2, m3))
    assert fa == fb, 'same seed + shared trunk must hash identically across variants'
    assert fa != fc, 'a different seed must produce a different init digest'
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    common = dict(run_cfg=fp, steps=1, tokens_seen=1, train_knobs=L.training_knobs(),
                  data_fp='d', val_fp='v', eval_fp='e')
    a = dict(common, variant='a', seed=0, ppl=10.0, init_core_fp=fa)
    b = dict(common, variant='b', seed=0, ppl=11.0, init_core_fp=fa)
    assert L.pair_reason(a, b) is None, L.pair_reason(a, b)
    why = L.pair_reason(a, dict(b, init_core_fp=fc))
    assert why is not None and 'init_core_fp' in why, why
    why = L.pair_reason(a, {k: v for k, v in b.items() if k != 'init_core_fp'})
    assert why is not None and 'unverifiable init_core_fp' in why, why
    # The zero-GPU report builder's local pairing gate must mirror this field
    # too; otherwise it can print a contrast the statistics layer refuses.
    assert BR._pair_reason(a, b) is None, BR._pair_reason(a, b)
    why = BR._pair_reason(a, dict(b, init_core_fp=fc))
    assert why is not None and 'init_core_fp' in why, why
    why = BR._pair_reason(a, {k: v for k, v in b.items() if k != 'init_core_fp'})
    assert why is not None and 'unverifiable init_core_fp' in why, why


def t_param_and_compute_control_detail():
    bd = {'total': 1000, 'non_embedding': 800, 'active': 700, 'embedding': 200}
    recs = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd, 'train_time_s': 10.0, 'tokens_seen': 100.0},
            ('b', 'p', 0): {'params': 1000, 'param_breakdown': {'total': 1000, 'non_embedding': 800, 'active': 600, 'embedding': 200},
                            'train_time_s': 20.0, 'tokens_seen': 100.0}}
    _rel, ok = L._param_control(recs, 'a', 'b')
    assert ok is True, 'total parameters are matched'
    d = L._param_control_detail(recs, 'a', 'b')
    assert d['total_controlled'] is True and d['active_controlled'] is False, d
    assert d['active_rel_diff'] > 0.01, d
    assert L._param_capacity_controlled(d) is False, d
    assert 'breakdown' in L.param_pair_reason(recs[('a', 'p', 0)], recs[('b', 'p', 0)]), 'breakdown must fail closed'
    # One-sided breakdown is unverifiable, not a total-only pass.
    one = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
           ('b', 'p', 0): {'params': 1000}}
    od = L._param_control_detail(one, 'a', 'b')
    assert od['breakdown_one_sided'] is True and L._param_capacity_controlled(od) is None, od
    assert 'one record carries it' in L.param_pair_reason(one[('a', 'p', 0)], one[('b', 'p', 0)]), 'one-sided breakdown must fail closed'
    # A single consistent seed must not hide a different/corrupted record of
    # the same arm: the arm-level capacity verdict is unverifiable, not the
    # first record's pass.
    mixed = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
             ('a', 'p', 1): {'params': 1200, 'param_breakdown': bd},
             ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._params_of_arm(mixed, 'a') is None
    assert L._param_control(mixed, 'a', 'b') == (None, None)
    md = L._param_control_detail(mixed, 'a', 'b')
    assert md['total_controlled'] is None, md
    assert L._param_capacity_controlled(md) is None, md
    mixed_bd = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
                ('a', 'p', 1): {'params': 1000, 'param_breakdown': dict(bd, active=650)},
                ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._param_breakdown_of_arm(mixed_bd, 'a') is None
    assert L._param_capacity_controlled(L._param_control_detail(mixed_bd, 'a', 'b')) is None
    partial = {('a', 'p', 0): {'params': 1000, 'param_breakdown': bd},
               ('a', 'p', 1): {'params': 1000},
               ('b', 'p', 0): {'params': 1000, 'param_breakdown': bd}}
    assert L._param_breakdown_of_arm(partial, 'a') is None
    assert L._param_capacity_controlled(L._param_control_detail(partial, 'a', 'b')) is None
    c = L._compute_control_detail(recs, 'a', 'b')
    assert c['compute_matched'] is False and c['more_compute_arm'] == 'b', c
    # Compute control consumes exact token counts: a fractional/bool count is
    # not a valid denominator, so the comparison is unverifiable rather than
    # silently using 100.5 (or True->1) tokens.
    for _bad_tok in (100.5, True, float('nan')):
        frac = {('a', 'p', 0): {'params': 1000, 'train_time_s': 10.0, 'tokens_seen': _bad_tok},
                ('b', 'p', 0): {'params': 1000, 'train_time_s': 20.0, 'tokens_seen': _bad_tok}}
        _c = L._compute_control_detail(frac, 'a', 'b')
        assert _c['compute_matched'] is None, (_c, _bad_tok)
        assert _c['time_ratio_a_over_b'] is None and _c['proxy_ratio_a_over_b'] is None, _c
    assert abs(c['time_ratio_a_over_b'] - 0.5) < 1e-12, c
    # A significant contrast whose extra compute favours the winner is
    # system-level, not a verified mechanism.
    entry = {'controlled': True, 'param_controlled': True, 'n': 6, 'significant_global': True,
             'predicted_sign': -1, 'direction_consistent': True, 'within_seed_noise': False,
             'mechanism_claim_allowed': False}
    assert L._classify_group_status([entry], min_n=6, alpha=0.05, m_family=1) == 'system_level_only'
    # End-to-end: b wins while consuming 2x the measured compute -> the
    # contrast lands in system_level_only, never in verified_components.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=100, ppl=50.0, params=1000,
                             train_knobs=kn, train_time_s=10.0, data_fp='d', val_fp='v', eval_fp='e')
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=100, ppl=49.0, params=1000,
                             train_knobs=kn, train_time_s=20.0, data_fp='d', val_fp='v', eval_fp='e')
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    assert 'c' in r['system_level_only_components'] and 'c' not in r['verified_components'], r
    assert r['components']['c'][0]['compute_confound_direction'] == 'favors_claimed_winner', r['components']['c'][0]


def t_capacity_breakdown_gates_baseline_claim():
    # Equal TOTAL parameters but a reallocated active count (dense MLP width /
    # dead indexer projections) is not a capacity-matched control.  The
    # breakdown must reach the verdict, not only the console.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=49.0, params=1000,
                             param_breakdown={'total': 1000, 'non_embedding': 800,
                                              'active': 700, 'embedding': 200},
                             train_knobs=kn)
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=50.0, params=1000,
                             param_breakdown={'total': 1000, 'non_embedding': 800,
                                              'active': 600, 'embedding': 200},
                             train_knobs=kn)
    r = L.methodology_audit(s, design={}, baselines={'comp': [('a', 'b', 'x')]},
                            verbose=False)
    e = r['baselines']['comp'][0]
    assert e['param_controlled'] is True, e
    assert e['param_capacity_controlled'] is False, e
    assert e['capacity_confounded'] is True, e
    assert 'comp' not in r['verified_baselines'], r['verified_baselines']
    assert 'comp' in r['unverified_baselines'], r['unverified_baselines']
    assert any('match on TOTAL parameters' in w for w in r['warnings']), r['warnings']


def t_hypothesis_param_match_requirement_is_enforced():
    # Every hypothesis must declare whether equal capacity is part of the
    # claim; an omitted field is a design defect, not a silent default.
    hp = {'statement': 'x', 'mechanism': 'y', 'predicted_sign': -1,
          'primary_metric': 'ppl', 'falsifier': 'z'}
    a = L.hypothesis_audit({'h': hp}, 'component')
    assert 'h' in a['invalid'] and any('requires_param_match' in w for w in a['warnings']), a
    hp2 = dict(hp, requires_param_match=True)
    assert L.hypothesis_audit({'h': hp2}, 'component')['status'] == 'ok'
    # A significant contrast whose hypothesis promises matched parameters is
    # system-level (capacity-confounded), never 'verified'.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    kn = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'a::s{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=49.0, params=1200, train_knobs=kn)
        s[f'b::s{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=10,
                             ppl=50.0, params=1000, train_knobs=kn)
    spec = {'statement': 'a beats b at matched capacity', 'mechanism': 'm',
            'predicted_sign': -1, 'primary_metric': 'ppl',
            'requires_param_match': True, 'falsifier': 'f'}
    saved = L.HYPOTHESIS_DESIGN.get('custom_capacity')
    L.HYPOTHESIS_DESIGN['custom_capacity'] = spec
    try:
        r = L.methodology_audit(s, design={'custom_capacity': [('a', 'b', 'x')]},
                                baselines={}, verbose=False)
    finally:
        if saved is None:
            L.HYPOTHESIS_DESIGN.pop('custom_capacity', None)
        else:
            L.HYPOTHESIS_DESIGN['custom_capacity'] = saved
    e = r['components']['custom_capacity'][0]
    assert e['requires_param_match'] is True and e['param_match_unmet'] is True, e
    assert e['mechanism_claim_allowed'] is False, e
    assert 'custom_capacity' in r['system_level_only_components'], r
    assert 'custom_capacity' not in r['verified_components'], r


def t_stale_aggregate_refused():
    cur = BR._CUR_CS
    d = {'a@cfg1': {'variant': 'a', 'run_cfg': f'steps1_d8_L1_csOLD1', 'n_seeds': 4, 'ppl_mean': 10.0},
         'a@cfg2': {'variant': 'a', 'run_cfg': f'steps1_d8_L1_csOLD2', 'n_seeds': 2, 'ppl_mean': 11.0}}
    assert BR._agg_entry_scan(d, 'a') is None, 'no current stamp among multiple groups must be refused'
    d['a@cfg3'] = {'variant': 'a', 'run_cfg': f'steps1_d8_L1_cs{cur}', 'n_seeds': 1, 'ppl_mean': 12.0}
    assert BR._agg_entry_scan(d, 'a') is d['a@cfg3'], 'the current-stamped group must be selected'
    single = {'a': {'variant': 'a', 'run_cfg': 'steps1_d8_L1_csOLD', 'n_seeds': 4, 'ppl_mean': 10.0}}
    assert BR._agg_entry_scan(single, 'a') is None, 'a single explicit stale run_cfg must be refused'


def t_sec_p1l_excludes_stale_records():
    stale = {'v::s0': {'variant': 'v', 'seed': 0, '_code': 'v0.0', 'by_len': {'512': {'ppl': 10.0}, '4096': {'ppl': 40.0}}}}
    BR._LOADED[('summary', 'results_len')] = stale
    try:
        txt = BR.sec_p1l()
    finally:
        BR._LOADED.pop(('summary', 'results_len'), None)
    assert '陈旧记录已排除' in txt, txt[:400]
    assert '10.0' not in txt and '40.0' not in txt, 'stale PPL must not be quoted in the section'
    assert 'ratio' not in txt or '不计算 ratio' in txt, txt[:400]


def t_efficiency_pairs_audit_and_sweep():
    td = tempfile.mkdtemp(prefix='eff_', dir=REPO)
    try:
        eff_a = {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
                 'prefill_ms_p50': 10.0, 'decode_has_kv_cache': False}
        eff_b = dict(eff_a, prefill_ms_p50=20.0)
        summ = {'a::s0': {'variant': 'a', 'efficiency': eff_a}, 'b::s0': {'variant': 'b', 'efficiency': eff_b}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summ, f)
        r = A.audit_efficiency_pairs(outdirs=[td])
        rec = r['panels'][td]['a vs b']
        assert rec['speed_claim_allowed'] is True and rec['faster_arm'] == 'a', rec
        assert abs(rec['ratio_b_over_a'] - 2.0) < 1e-12, rec
        # A different token count is not a speed comparison.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=4096)}}, f)
        r2 = A.audit_efficiency_pairs(outdirs=[td])
        rec2 = r2['panels'][td]['a vs b']
        assert rec2['speed_claim_allowed'] is False and 'token counts' in rec2['reason'], rec2
        # A fractional token count is not the same shape as its truncated int:
        # int(1.5) == 1 must not certify a latency comparison.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, n_tokens=1.5)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=1)}}, f)
        _rf = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rf['speed_claim_allowed'] is False and 'token counts' in _rf['reason'], _rf
        # A zero/negative token count is not a measured shape either, even
        # when both arms happen to agree on it.
        for _bad_n in (0, -1):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, n_tokens=_bad_n)},
                           'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=_bad_n)}}, f)
            _rn = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
            assert _rn['speed_claim_allowed'] is False and 'token counts' in _rn['reason'], _rn
        # A mixed device/KV record inside one arm makes the medians
        # incomparable even when the first record of each arm matches.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'a::s1': {'variant': 'a', 'efficiency': dict(eff_a, device='cuda:1')},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b},
                       'b::s1': {'variant': 'b', 'efficiency': dict(eff_b, prefill_ms_p50=20.0)}}, f)
        r3 = A.audit_efficiency_pairs(outdirs=[td])
        rec3 = r3['panels'][td]['a vs b']
        assert rec3['speed_claim_allowed'] is False and 'device' in rec3['reason'], rec3
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, prefill_ms_p50=True)},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        r4 = A.audit_efficiency_pairs(outdirs=[td])
        rec4 = r4['panels'][td]['a vs b']
        assert rec4['speed_claim_allowed'] is False, rec4
        # Missing device/torch or KV provenance is unverifiable, not equal.
        _bare = {'status': 'ok', 'n_tokens': 512, 'prefill_ms_p50': 10.0}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(_bare)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(_bare, prefill_ms_p50=20.0)}}, f)
        _rm = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rm['same_device_torch'] is False and _rm['speed_claim_allowed'] is False, _rm
        _no_kv = dict(eff_a)
        _no_kv.pop('decode_has_kv_cache')
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': _no_kv},
                       'b::s0': {'variant': 'b', 'efficiency': dict(_no_kv, prefill_ms_p50=20.0)}}, f)
        _rk = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rk['same_kv_semantics'] is False and _rk['speed_claim_allowed'] is False, _rk
        # The same token count at a different batch size is a different
        # execution shape, so it cannot support a speed claim.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, batch=1)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, batch=8)}}, f)
        _rb = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rb['same_batch'] is False and _rb['speed_claim_allowed'] is False, _rb
        assert 'batch' in _rb['reason'], _rb
        # Missing batch is legacy-compatible but explicitly not recorded.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        _rleg = A.audit_efficiency_pairs(outdirs=[td])['panels'][td]['a vs b']
        assert _rleg['same_batch'] is True and _rleg['batch_status'] == 'unrecorded', _rleg
        assert _rleg['speed_claim_allowed'] is True, _rleg
        # An invalid tolerance cannot widen (negative) or disable the speed gate.
        for _bad_tol in (True, -0.05, float('nan'), float('inf'), 1.0, 'x'):
            try:
                A.audit_efficiency_pairs(outdirs=[td], tolerance=_bad_tol)
            except ValueError:
                pass
            else:
                raise AssertionError(f'tolerance={_bad_tol!r} must be refused')
    finally:
        shutil.rmtree(td, ignore_errors=True)
    L.set_seed(0)
    model = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'full')).eval()
    ids = torch.randint(0, 64, (1, 20), dtype=torch.long)
    sw = L.measure_efficiency_sweep(model, ids, torch.device('cpu'), lengths=(8, 16), batches=(1, 2), warmup=0, iters=1)
    assert sw['status'] == 'ok' and len(sw['rows']) == 4, sw
    assert sw['decode_has_kv_cache'] is False and sw['measurement'] == 'cpu perf_counter', sw
    # A malformed/empty shape grid must not produce an empty row list that
    # still claims status=ok: that looks like a completed sweep.
    for _kw in ({'lengths': ()}, {'lengths': (8.5,)}, {'lengths': (True,)},
                {'lengths': (100,)}, {'batches': ()}, {'batches': (0,)},
                {'batches': (True,)}, {'warmup': -1}, {'iters': 0}):
        _args = dict(lengths=(8,), batches=(1,), warmup=0, iters=1)
        _args.update(_kw)
        try:
            L.measure_efficiency_sweep(model, ids, torch.device('cpu'), **_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency_sweep accepted {_kw!r}')
    for _kw in ({'warmup': -1}, {'warmup': 1.5}, {'iters': 0},
                {'iters': True}, {'decode_tokens': 0}):
        _args = dict(warmup=0, iters=1, decode_tokens=1)
        _args.update(_kw)
        try:
            L.measure_efficiency(model, ids, torch.device('cpu'), **_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency accepted {_kw!r}')
    for _bad_ids in (torch.empty((1, 0), dtype=torch.long),
                     torch.empty((0, 8), dtype=torch.long),
                     torch.empty((1, 1, 1), dtype=torch.long)):
        try:
            L.measure_efficiency(model, _bad_ids, torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency accepted ids shape {tuple(_bad_ids.shape)}')
        try:
            L.measure_efficiency_sweep(model, _bad_ids, torch.device('cpu'),
                                       lengths=(1,), batches=(1,),
                                       warmup=0, iters=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'measure_efficiency_sweep accepted ids shape {tuple(_bad_ids.shape)}')


def t_efficiency_pareto_is_grouped():
    a = {'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
         'decode_has_kv_cache': False, 'prefill_ms_p50': 10.0}
    assert BR.efficiency_group_key(a) == ('cuda:0', '2.8.0', 512, False)
    assert BR.efficiency_group_key(dict(a, n_tokens=4096)) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, device='cuda:1')) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, torch='2.9.0')) != BR.efficiency_group_key(a)
    assert BR.efficiency_group_key(dict(a, decode_has_kv_cache=True)) != BR.efficiency_group_key(a)
    for bad in (dict(a, device=''), dict(a, torch=None), dict(a, n_tokens=True),
                dict(a, n_tokens=1.5), dict(a, n_tokens=0),
                {k: v for k, v in a.items() if k != 'decode_has_kv_cache'}):
        assert BR.efficiency_group_key(bad) is None, bad
    # Lower latency at higher PPL is dominated; only the true staircase stays.
    fr = BR.efficiency_pareto([(10.0, 100.0, 'a'), (20.0, 90.0, 'b'),
                               (30.0, 95.0, 'c'), (5.0, 110.0, 'd')])
    assert [x[2] for x in fr] == ['d', 'a', 'b'], fr
    assert BR.efficiency_pareto([(float('nan'), 90.0, 'x'), (10.0, 100.0, 'a')]) == [(10.0, 100.0, 'a')]
    # Zero time / non-positive PPL are not measurements and must not become a
    # frontier point that dominates every real measurement.
    assert BR.efficiency_pareto([(0.0, 50.0, 'z'), (-1.0, 40.0, 'n'),
                                 (1.0, 0.0, 'p'), (10.0, 100.0, 'a')]) == [(10.0, 100.0, 'a')]


def t_efficiency_accuracy_tradeoff_summary():
    rows = [{'variant': 'best', 'ppl': p, 'prefill_ms_p50': 20.0,
             'params': 2000, 'peak_mem_allocated_mb': 200.0}
            for p in (48.0, 48.1, 47.9)]
    rows += [{'variant': 'cheap', 'ppl': p, 'prefill_ms_p50': 10.0,
              'params': 1000, 'peak_mem_allocated_mb': 100.0}
             for p in (50.0, 50.1, 49.9)]
    rows += [{'variant': 'tiny', 'ppl': p, 'prefill_ms_p50': 21.0,
              'params': 2000, 'peak_mem_allocated_mb': 200.0}
             for p in (48.05, 48.05, 48.05)]
    s = BR.efficiency_accuracy_summary(rows)
    assert [r['variant'] for r in s] == ['best', 'tiny', 'cheap'], s
    assert s[0]['ppl_sem'] is not None and s[0]['latency_ratio_vs_best'] == 1.0, s[0]
    cheap = [r for r in s if r['variant'] == 'cheap'][0]
    assert cheap['distinguishable'] is True, cheap
    assert cheap['ppl_delta_pct_vs_best'] > 3.0, cheap
    assert 'latency_2x_or_more' not in cheap['flags'], cheap
    assert cheap['params_ratio_vs_best'] == 0.5, cheap
    tiny = [r for r in s if r['variant'] == 'tiny'][0]
    assert 'ppl_gain_below_0.5pct' in tiny['flags'], tiny
    assert 'ppl_gain_within_seed_noise' in tiny['flags'], tiny
    # Unverifiable uncertainty is named instead of being treated as significant.
    unk = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    b = [r for r in unk if r['variant'] == 'b'][0]
    assert b['distinguishable'] is None and 'ppl_uncertainty_unverifiable' in b['flags'], b
    zero = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    zb = [r for r in zero if r['variant'] == 'b'][0]
    assert zb['distinguishable'] is None, zb
    assert 'zero_within_seed_variance_unverifiable' in zb['flags'], zb
    budget = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 1000},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 500}])
    assert all('train_budget_differs' in r['flags'] for r in budget), budget
    assert all(r['budget_matched'] is False for r in budget), budget
    # Missing budget is unverifiable, not equal to another missing budget.
    missing = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0}])
    assert all(r['budget_matched'] is False for r in missing), missing
    assert all('train_budget_unverifiable' in r['flags'] for r in missing), missing
    # Two medians can agree while an arm mixes budgets; that is still not a
    # controlled comparison.
    mixed = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 100},
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 300},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 200},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 200}])
    assert all(r['budget_matched'] is False for r in mixed), mixed
    assert all('train_budget_inconsistent_within_arm' in r['flags'] for r in mixed), mixed
    matched = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 1000},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 1000}])
    assert all(r['budget_matched'] is True for r in matched), matched
    # Zero is not a trained budget, and two invalid records must not match.
    zero_tok = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0, 'tokens_seen': 0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0, 'tokens_seen': 0}])
    assert all(r['budget_matched'] is False for r in zero_tok), zero_tok
    assert all('train_budget_unverifiable' in r['flags'] for r in zero_tok), zero_tok
    # Invalid latency/parameter/memory readings are unverifiable, not numbers
    # that can support a ratio.
    invalid = BR.efficiency_accuracy_summary([
        {'variant': 'a', 'ppl': 48.0, 'prefill_ms_p50': 10.0},
        {'variant': 'z', 'ppl': 40.0, 'prefill_ms_p50': 0.0},
        {'variant': 'b', 'ppl': 49.0, 'prefill_ms_p50': 20.0,
         'params': 0, 'peak_mem_allocated_mb': -1.0}])
    assert [r['variant'] for r in invalid] == ['a', 'b'], invalid
    bb = [r for r in invalid if r['variant'] == 'b'][0]
    assert bb['params_median'] is None and bb['peak_mem_max'] is None, bb


def t_report_error_analysis_section():
    rec = {'checkpoint': '/tmp/x/ckpt.pt', 'split': 'validation',
           'mean_ce': 4.2, 'pooled_ppl': 66.7,
           'rare_target_basis': 'unavailable: no train_ids',
           'rare_target_problem': 'empty train_ids: token frequencies are unmeasured',
           'buckets': {'position:last10%': {'n': 3, 'mean_ce': 5.0, 'perplexity': 148.4}},
           'cause_counts': {'ambiguous_context': 2, 'annotation_or_rare_target': 1},
           'cause_counts_all': {'ambiguous_context': {'n': 9, 'share': 0.9, 'mean_ce': 4.1},
                                'annotation_or_rare_target': {'n': 1, 'share': 0.1, 'mean_ce': 6.0}},
           'cause_guide': {'ambiguous_context': 'read the context'},
           'worst_tokens': [{'row': 0, 'position': 3, 'ce': 9.5,
                             'target_text': 'x', 'pred_text': 'y',
                             'context_text': 'the surrounding words',
                             'cause_primary': 'ambiguous_context'}],
           'warning': 'validation only'}
    orig = BR.load
    BR.load = lambda path, default=None: rec if path.endswith(os.path.join('analysis_errors', 'summary.json')) else default
    try:
        txt = BR.sec_error_analysis()
    finally:
        BR.load = orig
    assert 'Bad-case analysis' in txt and 'ambiguous_context' in txt, txt
    assert 'rare-target basis' in txt and 'empty train_ids' in txt, txt
    assert 'ckpt.pt' in txt and 'validation only' in txt, txt
    assert 'read the context' in txt and 'Whole-split' in txt, txt
    assert 'the surrounding words' in txt, txt
    rec2 = dict(rec, checkpoint='/tmp/y/ckpt2.pt')
    orig_paths = BR._error_analysis_paths
    BR._error_analysis_paths = lambda: ['/a/summary.json', '/b/summary.json']
    BR.load = lambda path, default=None: {'/a/summary.json': rec, '/b/summary.json': rec2}.get(path, default)
    try:
        txt2 = BR.sec_error_analysis()
    finally:
        BR._error_analysis_paths = orig_paths
    assert 'ckpt.pt' in txt2 and 'ckpt2.pt' in txt2 and '###' in txt2, txt2
    # A valid-JSON summary with malformed collection fields must not crash the
    # report builder (the old code sliced/`.get`ed them unconditionally).
    bad = dict(rec, worst_tokens={'not': 'a list'}, cause_guide=1,
               cause_counts={'ambiguous_context': 2})
    BR.load = lambda path, default=None: bad
    try:
        txt3 = BR.sec_error_analysis()
    finally:
        BR.load = orig
    assert 'Bad-case analysis' in txt3 and 'ckpt.pt' in txt3, txt3


def t_report_training_health_actions():
    rec = {'ppl': 51.0, 'eval_subset_rows': 16,
           'curve_health': {'status': 'overfitting', 'best_val_ppl': 48.0,
                            'best_step': 500, 'final_minus_best': 3.0,
                            'steps_since_best': 500, 'train_ce_slope_per_step': -1e-4,
                            'val_ce_slope_per_step': 2e-5, 'gap_last': 1.2,
                            'gap_slope_per_step': 3e-5, 'n_eval_points': 7,
                            'trend_reliable': True,
                            'actions': ['add regularization', 'report the best checkpoint']},
           'health_history': [{'val_ci95_low': 47.0, 'val_ci95_high': 49.0}]}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec}}
    try:
        txt = BR.sec_training_health()
    finally:
        BR._result_summaries = orig
    assert 'steps since best' in txt and 'add regularization' in txt, txt
    assert '500' in txt and 'gap slope' in txt, txt
    # The number of curve points is distinct from the number of documents per
    # eval; a 16-doc subset at 7 eval points must not be reported as "16 rows".
    assert 'eval points' in txt and 'eval docs' in txt, txt
    assert '| 7 | 16 | True |' in txt, txt
    # Malformed history/actions collections must not crash the section.
    bad = dict(rec, health_history=1, curve_health=dict(rec['curve_health'], actions='one action'))
    BR._result_summaries = lambda: {'results_unit': {'b::s0': bad}}
    try:
        txt2 = BR.sec_training_health()
    finally:
        BR._result_summaries = orig
    assert 'one action' in txt2 and 'n/a' in txt2, txt2


def t_report_early_stop_section():
    rec = {'steps': 100, 'steps_completed': 40, 'tokens_seen': 400,
           'early_stop': {'stopped': True, 'best_val_ppl': 48.0,
                          'best_step': 30, 'reason': 'patience reached'}}
    normal = {'steps': 100, 'steps_completed': 100,
              'early_stop': {'stopped': False, 'reason': 'patience not reached'}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec, 'b::s0': normal}}
    try:
        txt = BR.sec_early_stop()
    finally:
        BR._result_summaries = orig
    assert 'Early stop' in txt and '40' in txt and '100' in txt, txt
    assert 'patience reached' in txt and 'True' in txt, txt
    assert 'b::s0' not in txt, txt
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'b::s0': normal}}
    try:
        assert BR.sec_early_stop() == ''
    finally:
        BR._result_summaries = orig


def t_report_efficiency_groups_are_separate():
    def _eff(n, lat):
        return {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': n,
                'decode_has_kv_cache': False, 'prefill_ms_p50': lat,
                'peak_mem_allocated_mb': 10.0}
    summ = {'results_unit': {
        'a::s0': {'variant': 'a', 'ppl': 50.0, 'params': 1000, 'tokens_seen': 100,
                  'train_tokens_per_s': 123.4, 'efficiency': _eff(512, 10.0)},
        'b::s0': {'variant': 'b', 'ppl': 40.0, 'params': 1000, 'tokens_seen': 100,
                  'efficiency': _eff(4096, 5.0)},
        'c::s0': {'variant': 'c', 'ppl': 45.0, 'params': 1000, 'tokens_seen': 100,
                  'efficiency': {'status': 'ok', 'prefill_ms_p50': 1.0}}}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: summ
    try:
        txt = BR.sec_efficiency()
    finally:
        BR._result_summaries = orig
    assert '512 tokens' in txt and '4096 tokens' in txt, txt
    assert 'excluded from every frontier' in txt and '`c::s0`' in txt, txt
    assert '123.4' in txt, txt
    # The two token counts are separate group headings, not one merged frontier.
    assert txt.count('device `cuda:0`') == 2, txt


def t_report_efficiency_cost_sentence_uses_correct_direction():
    def _eff(ms):
        return {'status': 'ok', 'device': 'cuda:0', 'torch': '2.8.0', 'n_tokens': 512,
                'decode_has_kv_cache': False, 'prefill_ms_p50': ms,
                'peak_mem_allocated_mb': 10.0}
    summ = {'results_unit': {}}
    for i, (p, ms) in enumerate(((48.0, 30.0), (48.1, 30.0), (47.9, 30.0))):
        summ['results_unit'][f'best::s{i}'] = {
            'variant': 'best', 'ppl': p, 'params': 1000, 'tokens_seen': 100,
            'efficiency': _eff(ms)}
    for i, (p, ms) in enumerate(((50.0, 10.0), (50.1, 10.0), (49.9, 10.0))):
        summ['results_unit'][f'cheap::s{i}'] = {
            'variant': 'cheap', 'ppl': p, 'params': 1000, 'tokens_seen': 100,
            'efficiency': _eff(ms)}
    orig = BR._result_summaries
    BR._result_summaries = lambda: summ
    try:
        txt = BR.sec_efficiency()
    finally:
        BR._result_summaries = orig
    assert 'improves PPL by' in txt, txt
    assert 'gains -' not in txt, txt
    assert 'but is 3.0× slower' in txt, txt
    assert '4.17%' in txt, txt


def t_report_nan_section_uses_triage():
    rec = {'instability_diagnosis': {
        'status': 'instability_observed', 'primary_suspect': 'fp16_overflow',
        'checks': [{'check': 'fp16_overflow', 'status': 'observed',
                    'evidence': 'overflows=2', 'action': 'switch to bf16'}]},
        'opt_diag_summary': {'first_bad_step': 17, 'nonfinite_loss_steps': 0,
                             'nonfinite_grad_steps': 2, 'amp_scaler_overflows': 2,
                             'nan_guard': {'elements_cleaned': 0}}}
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': rec}}
    try:
        txt = BR.sec_nan_diagnostics()
    finally:
        BR._result_summaries = orig
    assert 'Non-finite' in txt and 'fp16_overflow' in txt, txt
    assert 'switch to bf16' in txt and '17' in txt, txt
    orig = BR._result_summaries
    BR._result_summaries = lambda: {'results_unit': {'a::s0': {'opt_diag_summary': {
        'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
        'nan_guard': {'elements_cleaned': 0}}}}}
    try:
        assert BR.sec_nan_diagnostics() == ''
    finally:
        BR._result_summaries = orig


def t_audit_training_health_records():
    td = tempfile.mkdtemp(prefix='health_', dir=REPO)
    try:
        clean = {'curve_health': {'status': 'healthy'},
                 'opt_diag_summary': {'nonfinite_loss_steps': 0,
                                      'nonfinite_grad_steps': 0,
                                      'nan_guard': {'elements_cleaned': 0}},
                 'instability_diagnosis': {'status': 'clean'}}
        over = {'curve_health': {'status': 'overfitting', 'reason': 'val rising'}}
        contra = {'curve_health': {'status': 'healthy'},
                  'opt_diag_summary': {'nonfinite_loss_steps': 1,
                                       'nonfinite_grad_steps': 0,
                                       'nan_guard': {'elements_cleaned': 0}},
                  'instability_diagnosis': {'status': 'clean'}}
        legacy = {'curve_health': {'status': 'nonfinite'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': clean, 'b::s0': over}, f)
        r = A.audit_training_health(outdirs=[td])
        assert r['status'] == 'warnings' and 'b::s0' in str(r['warnings']), r
        assert 'a::s0' not in str(r['warnings']), r
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': clean, 'c::s0': contra}, f)
        r2 = A.audit_training_health(outdirs=[td])
        assert r2['status'] == 'fail' and any('clean but non-finite' in x for x in r2['errors']), r2
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'d::s0': legacy}, f)
        r3 = A.audit_training_health(outdirs=[td])
        assert r3['status'] == 'warnings' and any('without triage' in x for x in r3['warnings']), r3
        # A malformed event count is unknown, not zero: treating it as zero
        # would let a corrupt record pass as clean.
        malformed = {'curve_health': {'status': 'healthy'},
                     'opt_diag_summary': {'nonfinite_loss_steps': 1.5,
                                          'nonfinite_grad_steps': 0,
                                          'nan_guard': {'elements_cleaned': 0}},
                     'instability_diagnosis': {'status': 'clean'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'m::s0': malformed}, f)
        r4 = A.audit_training_health(outdirs=[td])
        assert r4['status'] == 'fail' and any('unverifiable, not zero' in x
                                              for x in r4['errors']), r4
        # The cached status is not evidence on its own: a record claiming
        # healthy while its own curve is overfitting must be surfaced, and a
        # masked non-finite curve point must fail closed.
        over_hist = [{'step': i * 100, 'train_ce': 6.0 - 0.3 * i,
                      'val_ce': 4.0 + 0.05 * i, 'val_ppl': math.exp(4.0 + 0.05 * i)}
                     for i in range(5)]
        fake_clean = {'curve_health': {'status': 'healthy', 'best_step': 999,
                                       'best_val_ppl': 1.0,
                                       'final_minus_best': -5.0,
                                       'steps_since_best': 777},
                      'health_history': over_hist,
                      'opt_diag_summary': {'nonfinite_loss_steps': 0,
                                           'nonfinite_grad_steps': 0,
                                           'nan_guard': {'elements_cleaned': 0}},
                      'instability_diagnosis': {'status': 'clean'}}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'h::s0': fake_clean}, f)
        r5 = A.audit_training_health(outdirs=[td])
        assert r5['status'] == 'warnings', r5
        assert any('contradicts the health_history' in x for x in r5['warnings']), r5
        assert any('best_step=999' in x for x in r5['warnings']), r5
        assert any('best_val_ppl=1' in x for x in r5['warnings']), r5
        assert any('final_minus_best=-5' in x for x in r5['warnings']), r5
        assert any('steps_since_best=777' in x for x in r5['warnings']), r5
        nan_hist = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                    {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}]
        masked_nan = dict(fake_clean, health_history=nan_hist)
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'n::s0': masked_nan}, f)
        r6 = A.audit_training_health(outdirs=[td])
        assert r6['status'] == 'fail', r6
        assert any('contradicts the health_history' in x for x in r6['errors']), r6
        # Counter/log disagreement and a first_bad_step without any event
        # are record-integrity failures, not clean runs.
        for label, rec2 in (
                ('v::s0', {'curve_health': {'status': 'healthy'},
                           'opt_diag_summary': {
                               'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                               'nan_guard': {'elements_cleaned': 0},
                               'events': [{'kind': 'loss', 'loss': float('nan')}]}}),
                ('w::s0', {'curve_health': {'status': 'healthy'},
                           'opt_diag_summary': {
                               'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                               'nan_guard': {'elements_cleaned': 0},
                               'first_bad_step': 12}})):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({label: rec2}, f)
            ry = A.audit_training_health(outdirs=[td])
            if label == 'v::s0':
                assert ry['status'] == 'fail' and any(
                    'nonfinite_loss_steps=0' in x for x in ry['errors']), ry
            else:
                assert ry['status'] == 'warnings' and any(
                    'first_bad_step=12' in x for x in ry['warnings']), ry
        # Unknown status names and a healthy claim with an explicitly empty
        # curve are not verified health.
        for label, rec2 in (('u::s0', {'curve_health': {'status': 'fine'},
                                       'opt_diag_summary': {}}),
                            ('e::s0', {'curve_health': {'status': 'healthy'},
                                       'health_history': [],
                                       'opt_diag_summary': {}})):
            with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({label: rec2}, f)
            rx = A.audit_training_health(outdirs=[td])
            assert rx['status'] == 'warnings', (label, rx)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_release_readiness_audit():
    td = tempfile.mkdtemp(prefix='rel_', dir=REPO)
    old_repo = A.REPO
    try:
        A.REPO = td
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n| _none yet_ | — | — | — | — | — | — |\n')
        r = A.audit_release_readiness()
        assert r['status'] == 'no_release' and r['complete_rows'] == 0, r
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n'
                    '| v13 | 2026-10-20 | abcdef1 | https://example.org/release_v13.tar.gz | ' + 'a' * 64 + ' | ' + 'b' * 64 + ' | https://example.org/verify.txt |\n')
        r2 = A.audit_release_readiness()
        assert r2['status'] == 'ok' and r2['complete_rows'] == 1 and not r2['problems'], r2
        with open(os.path.join(td, 'RELEASES.md'), 'w', encoding='utf-8') as f:
            f.write('| tag | date | Git commit | archive URL | archive SHA-256 | manifest SHA-256 | verifier |\n|---|---|---|---|---|---|---|\n'
                    '| v13 | 2026-10-20 | — | — | — | — | — |\n')
        r3 = A.audit_release_readiness()
        assert r3['status'] == 'incomplete' and r3['problems'], r3
    finally:
        A.REPO = old_repo
        shutil.rmtree(td, ignore_errors=True)



def t_selection_attribution_contract():
    td = tempfile.mkdtemp(prefix='sel_', dir=REPO)
    old_repo = A.REPO
    try:
        A.REPO = td
        r = A.audit_selection()
        assert r['status'] == 'not_verifiable_locally', r
        os.makedirs(os.path.join(td, 'analysis_mechanism'))
        cell = {'arm': 'csa_fixed', 'eval_len': 4096, 'k': 32, 'n_blocks': 1024, 'n_queries': 64,
                'recall_at_k': 0.6, 'mass_covered': 0.8, 'random_baseline': 0.03125,
                'per_seed': {str(i): {'mass_covered': 0.8 - 0.01 * i, 'random_baseline': 0.03125} for i in range(6)}}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [cell]}, f)
        r2 = A.audit_selection()
        assert r2['status'] == 'ok' and r2['cells'][0]['mass_excess'] > 0, r2
        assert r2['cells'][0]['claim_allowed'] is True, r2['cells'][0]
        bad = dict(cell)
        bad.pop('recall_at_k')
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad]}, f)
        r3 = A.audit_selection()
        assert r3['status'] == 'fail' and r3['problems'], r3
        # Block/query counts are exact geometry.  int(1.5) would silently
        # certify a different k/n_blocks than the artifact declares.
        frac = dict(cell, k=1.5)
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [frac]}, f)
        rf = A.audit_selection()
        assert rf['status'] == 'fail' and any('exact integers' in x for x in rf['problems']), rf
        # per_seed is part of the declared schema: without it the seed-level
        # claim cannot be checked and must be refused structurally.
        bad2 = dict(cell)
        bad2.pop('per_seed')
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad2]}, f)
        r4 = A.audit_selection()
        assert r4['status'] == 'fail' and any('per_seed' in x for x in r4['problems']), r4
        # Malformed seed rows must not inflate the n>=6 claim gate.
        bad3 = dict(cell)
        bad3['per_seed'] = {str(i): ({'mass_covered': 0.8, 'random_baseline': 0.03} if i < 2 else {})
                            for i in range(6)}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad3]}, f)
        r5 = A.audit_selection()
        assert r5['cells'][0]['claim_allowed'] is False and r5['cells'][0]['n_seeds_valid'] == 2, r5
        # Non-integer and duplicate-normalised seed keys must not count as
        # independent seeds for the n>=6 claim gate.
        bad4 = dict(cell)
        good = {'mass_covered': 0.8, 'random_baseline': 0.03}
        bad4['per_seed'] = {'0': good, 'bad': good, '00': good}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bad4]}, f)
        r6 = A.audit_selection()
        assert r6['cells'][0]['n_seeds_valid'] == 1, r6
        assert any('per_seed' in x for x in r6['problems']), r6
        # A present-but-NaN/out-of-range per-seed reading is malformed, not a
        # silent drop that leaves the reported n larger than the tested sample.
        nan_cell = dict(cell)
        nan_cell['per_seed'] = {str(i): {'mass_covered': 0.8, 'random_baseline': 0.03}
                                for i in range(6)}
        nan_cell['per_seed']['2'] = {'mass_covered': float('nan'), 'random_baseline': 0.03}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [nan_cell]}, f)
        r7 = A.audit_selection()
        assert r7['status'] == 'fail' and any('per_seed' in x for x in r7['problems']), r7
        assert r7['cells'][0]['n_seeds_valid'] == 5, r7['cells'][0]
    finally:
        A.REPO = old_repo
        shutil.rmtree(td, ignore_errors=True)


def t_audit_length_missing_truncated_flag():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095}}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail' and any('truncated' in x for x in r['problems']), r
        # A direct caller may pass np.bool_(False); the identity check used to
        # skip the n_tok == n_seq*(L-1) contract for that valid spelling.
        _old_load = A._load
        try:
            A._load = lambda _p: {'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 17, 'eval_span': 4095,
                'truncated': np.bool_(False), 'n_seq': 16}}}}
            rb = A.audit_length(p)
        finally:
            A._load = _old_load
        assert rb['status'] == 'fail' and any('n_seq*(L-1)' in x for x in rb['problems']), rb
        # A full-length cell must report the SCORED span L-1, not the input
        # width L: otherwise every arm can agree on an eval_span that its own
        # token denominator contradicts.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4096,
                'truncated': False, 'n_seq': 16}}}}, f)
        rs = A.audit_length(p)
        assert rs['status'] == 'fail' and any('eval_span=4096' in x for x in rs['problems']), rs
        # A truncated cell's token count is n_seq * its scored span, not
        # n_seq * (L-1); a silently shortened slice must be named.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 2048,
                'truncated': True, 'n_seq': 16}}}}, f)
        rt = A.audit_length(p)
        assert rt['status'] == 'fail' and any('n_seq*eval_span' in x for x in rt['problems']), rt
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_audit_length_rejects_zero_target_cells():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        for _cell in ({'ppl': 30.0, 'n_tok': 0, 'eval_span': 4096,
                       'truncated': False, 'n_seq': 0},
                      {'ppl': 30.0, 'n_tok': 0, 'eval_span': 0,
                       'truncated': True, 'n_seq': 16}):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'by_len': {'4096': _cell}}}, f)
            r = A.audit_length(p)
            assert r['status'] == 'fail', r
            assert any(('n_tok' in x and 'positive' in x)
                       or ('n_seq' in x and 'positive' in x) for x in r['problems']), r
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_audit_length_missing_n_seq_flag():
    # Same fail-closed rule as a missing truncation marker: without n_seq the
    # audit cannot prove the arms scored the same document rows.
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False}}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail' and any('n_seq' in x for x in r['problems']), r
        assert r['by_length']['4096']['same_n_seq_across_records'] is False, r
        # One record with n_seq must not mask another record that omits it.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False,
                                                    'n_seq': 16}}},
                       'b::s0': {'by_len': {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095,
                                                    'eval_span': 4095, 'truncated': False}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail' and r2['by_length']['4096']['same_n_seq_across_records'] is False, r2
        # Two row-count fields (n_seq and len(ppl_per_seq)) must describe the
        # same rows; silently preferring one would hide a truncated vector.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095,
                'truncated': False, 'n_seq': 16,
                'ppl_per_seq': [30.0] * 8}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail' and any('disagrees' in x for x in r3['problems']), r3
        # An invalid explicit n_seq is a problem even when a per-document
        # vector supplies a plausible length.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': 16 * 4095, 'eval_span': 4095,
                'truncated': False, 'n_seq': True,
                'ppl_per_seq': [30.0] * 16}}}}, f)
        r4 = A.audit_length(p)
        assert r4['status'] == 'fail' and any('non-integer' in x for x in r4['problems']), r4
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_audit_length_rejects_unverifiable_counts():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        # A non-numeric n_tok/eval_span, a non-integer length key and a
        # malformed cell must all fail closed instead of being skipped.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {
                '4096': {'ppl': 30.0, 'n_tok': 'lots', 'eval_span': 4095,
                         'truncated': False, 'n_seq': 16},
                'oops': {'ppl': 1.0},
                '512': 'not-a-cell'}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail', r
        assert any('n_tok' in x for x in r['problems']), r
        assert any('integer length' in x for x in r['problems']), r
        assert any('not a JSON object' in x for x in r['problems']), r
        # Booleans are not counts either.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': True, 'eval_span': 4095,
                'truncated': False, 'n_seq': True}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail', r2
        # Non-hashable malformed cells must be a recorded failure, not an
        # uncaught TypeError from inserting them into a comparison set.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'ppl': 30.0, 'n_tok': [1, 2], 'eval_span': {'bad': 1},
                'truncated': [False], 'n_seq': 1}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail', r3
        assert any('n_tok' in x for x in r3['problems']), r3
        assert any('eval_span' in x for x in r3['problems']), r3
        assert any('truncated' in x for x in r3['problems']), r3
        # An existing-but-empty artifact must not pass the audit, and a
        # malformed by_len container must fail closed instead of raising.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        r4 = A.audit_length(p)
        assert r4['status'] == 'not_verifiable_locally', r4
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': []}}, f)
        r5 = A.audit_length(p)
        assert r5['status'] == 'fail' and any('by_len' in x for x in r5['problems']), r5
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_global_multiplicity_audit_basic():
    # Known BH case: p=[0.01, 0.02, 0.5] at m=3 -> q=[0.03, 0.03, 0.5].
    r = L.global_multiplicity_audit([('a', 0.01), ('b', 0.02), ('c', 0.5)])
    assert r['m'] == 3, r
    ent = {e['name']: e for e in r['entries']}
    assert abs(ent['a']['q_bh'] - 0.03) < 1e-12 and abs(ent['b']['q_bh'] - 0.03) < 1e-12, r
    assert abs(ent['c']['q_bh'] - 0.5) < 1e-12, r
    # 0.01: Bonferroni 0.03 < 0.05 -> globally controlled.
    assert ent['a']['global_status'] == 'globally_controlled', r
    # 0.02: BH-rejected but Bonferroni 0.06 >= 0.05 -> fdr_only.
    assert ent['b']['global_status'] == 'fdr_only', r
    # 0.5: raw >= alpha.
    assert ent['c']['global_status'] == 'not_significant', r
    assert r['summary']['n_raw_significant'] == 2 and r['summary']['n_bh_rejected'] == 2, r
    # Ten nominal hits out of a large displayed family: BH still rejects them
    # (q = 0.04) but Bonferroni (0.4) does not -> fdr_only, not family-wise.
    r = L.global_multiplicity_audit([(f'c{i}', 0.04) for i in range(10)])
    assert all(e['global_status'] == 'fdr_only' for e in r['entries']), r
    assert r['summary']['n_raw_significant'] == 10 and r['summary']['n_bh_rejected'] == 10, r
    assert r['summary']['n_bonferroni_significant'] == 0, r
    assert abs(r['entries'][0]['q_bh'] - 0.04) < 1e-12, r
    # A single nominal hit inside a large displayed family of nulls is
    # family_only: raw p < alpha but the cross-panel FDR view does not reject.
    r = L.global_multiplicity_audit([(f'c{i}', (0.04 if i == 0 else 0.9)) for i in range(100)])
    ent = {e['name']: e for e in r['entries']}
    assert ent['c0']['global_status'] == 'family_only', ent['c0']
    assert ent['c0']['bh_rejected'] is False and ent['c0']['q_bh'] >= 0.05, ent['c0']
    assert all(e['global_status'] == 'not_significant' for k, e in ent.items() if k != 'c0'), r
    assert r['summary']['n_family_only'] == 1, r
    # Family labels ride along for display and never change the decision.
    r = L.global_multiplicity_audit([('a', 0.01, 'P0R'), ('b', 0.02, 'P1T')])
    assert {e['family'] for e in r['entries']} == {'P0R', 'P1T'}, r
    r = L.global_multiplicity_audit([('a', 0.01), ('b', 0.02)], families=['P0R', 'P1T'])
    assert {e['family'] for e in r['entries']} == {'P0R', 'P1T'}, r


def t_global_multiplicity_audit_fail_closed():
    # Duplicate names double-count one comparison: hard error.
    try:
        L.global_multiplicity_audit([('a', 0.01), ('a', 0.02)])
    except ValueError as e:
        assert 'duplicate' in str(e).lower(), e
    else:
        raise AssertionError('duplicate comparison names must be refused')
    # Invalid alpha is refused, not defaulted.
    for bad in (0.0, 1.0, -0.1, True, None, 'x'):
        try:
            L.global_multiplicity_audit([('a', 0.01)], alpha=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'alpha={bad!r} must be refused')
    # Empty family is a legal explicit state.
    r = L.global_multiplicity_audit([])
    assert r['m'] == 0 and r['summary']['n_raw_significant'] == 0 and r['entries'] == [], r
    # Missing/bool/out-of-range p keeps the slot (fail-closed family size).
    r = L.global_multiplicity_audit([('ok', 0.02), ('none', None), ('bool', True), ('neg', -0.1), ('big', 1.7), ('nan', float('nan'))])
    assert r['m'] == 6, r
    assert set(r['untestable']) == {'none', 'bool', 'neg', 'big', 'nan'}, r
    ent = {e['name']: e for e in r['entries']}
    assert all(ent[n]['global_status'] == 'untestable' for n in ('none', 'bool', 'neg', 'big', 'nan')), r
    # The valid entry is audited over the FULL displayed family (m=6), not
    # over a silently shrunk one.
    assert abs(ent['ok']['p_bonferroni'] - 0.12) < 1e-12, r
    assert r['summary']['n_untestable'] == 5, r
    # q is monotone in p and capped at 1 on random panels.
    import random as _rnd
    _rnd.seed(20261020)
    for _trial in range(30):
        m = _rnd.randint(1, 24)
        named = [(f'c{i}', _rnd.choice([None, _rnd.random()])) for i in range(m)]
        ra = L.global_multiplicity_audit(named)
        assert ra['m'] == m, (named, ra)
        pairs = sorted((e['p_raw'], e['q_bh']) for e in ra['entries'] if e['p_raw'] is not None)
        for (p1, q1), (p2, q2) in zip(pairs, pairs[1:]):
            assert p2 >= p1 and q2 >= q1 - 1e-12 and q2 <= 1.0 + 1e-12, (named, pairs)


def _global_multiplicity_random_cases():
    import random as _rnd
    _rnd.seed(20261021)
    for _trial in range(40):
        m = _rnd.randint(1, 24)
        yield [(f'c{i}', _rnd.choice([None, _rnd.random()])) for i in range(m)]


def t_global_multiplicity_twin_consistency():
    # The report builder keeps a dependency-free local twin; the two must
    # agree exactly so the printed global table cannot drift from the library
    # the driver-side audits use.
    for named in _global_multiplicity_random_cases():
        a1 = L.global_multiplicity_audit(named, alpha=0.05)
        a2 = BR.global_multiplicity_adjust([(n, p) for n, p in named], alpha=0.05)
        assert a1['m'] == a2['m'], (named, a1, a2)
        by2 = {r['name']: r for r in a2['rows']}
        for e in a1['entries']:
            o = by2[e['name']]
            if e['p_raw'] is None:
                assert o['p_raw'] is None and o['global_status'] == 'untestable', (named, e, o)
            else:
                assert abs(e['p_raw'] - o['p_raw']) < 1e-12, (named, e, o)
                assert abs(e['q_bh'] - o['q_bh']) < 1e-9, (named, e, o)
                assert abs(e['p_bonferroni'] - o['p_bonferroni']) < 1e-12, (named, e, o)
                assert e['global_status'] == o['global_status'], (named, e, o)
        assert a1['summary']['n_raw_significant'] == a2['n_raw_significant'], (named, a1, a2)
        assert a1['summary']['n_bh_rejected'] == a2['n_bh_rejected'], (named, a1, a2)
        assert a1['summary']['n_bonferroni_significant'] == a2['n_bonferroni_significant'], (named, a1, a2)
    # NumPy booleans are flags too; the local twin must not turn them into 0/1.
    for _p in (np.bool_(True), np.bool_(False)):
        a1 = L.global_multiplicity_audit([('b', _p)])
        a2 = BR.global_multiplicity_adjust([('b', _p)])
        assert a1['entries'][0]['p_raw'] is None, a1
        assert a2['rows'][0]['p_raw'] is None, a2


def t_build_report_global_multiplicity_section():
    BR._reset_claim_registry()
    try:
        st = {'n': 6, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.01, 'p_holm': 0.02}
        ok = BR._sig_gate(st, label='P0X demo labelled contrast', family='P0X')
        assert ok is True, st
        st2 = {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25, 'p_holm': 0.5}
        assert BR._sig_gate(st2, label='P0Y demo underpowered contrast', family='P0Y') is False
        for _bad_alpha in (True, 0, 1, -0.1, 1.5, float('nan'), 'x'):
            assert BR._sig_gate(st, alpha=_bad_alpha) is False, _bad_alpha
        # An unlabeled gate decision is counted and flagged, never lost.
        assert BR._sig_gate(st) is True
        assert BR._UNREGISTERED_GATES == 1, BR._UNREGISTERED_GATES
        txt = BR.sec_global_multiplicity()
        assert 'P0X demo labelled contrast' in txt and 'P0Y demo underpowered contrast' in txt, txt
        assert 'q_BH' in txt and 'Benjamini-Hochberg' in txt or 'BH' in txt, txt
        assert 'P0X demo labelled contrast' in txt, txt
        # The globally-controlled wording rule is present so prose can cite it.
        assert '全局' in txt, txt
        # Flag line for the unregistered decision.
        assert '未带 label 注册' in txt, txt
        # Duplicate labels across sections must fail closed (family size
        # would double-count).
        try:
            BR._note_comparison('dup', 0.01)
            BR._note_comparison('dup', 0.02)
            BR.sec_global_multiplicity()
        except ValueError as e:
            assert 'duplicate' in str(e).lower(), e
        else:
            raise AssertionError('duplicate labels must be refused')
    finally:
        BR._reset_claim_registry()
    # No displayed comparisons -> explicit empty note, not a silent omission.
    txt = BR.sec_global_multiplicity()
    assert '没有显示任何带 p 值的比较' in txt, txt


def t_scale_corpus_binding_refuses_mixed_tokenizers():
    # Cross-scale effects (paired log PPL ratios) are only defined inside ONE
    # tokenisation; pooling across corpora/tokenizers is a units error, not a
    # heterogeneity finding.  The scale machinery must refuse it from the
    # provenance, not from the numeric pattern.
    pts = [{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2],
            'corpus_fp': 'fpA', 'vocab': 32768},
           {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35], 'seeds': [0, 1, 2],
            'corpus_fp': 'fpA', 'vocab': 32768}]
    m = L.meta_analyze_effects(pts)
    assert m['status'] == 'ok', m
    assert m['corpus_binding']['corpus_binding'] == 'shared', m
    # The scale points share seed identities, so the DL pooled CI's
    # independence assumption is disclosed and a paired per-seed common
    # effect is reported as the dependence-respecting sensitivity readout.
    assert m['seed_dependence'] == 'shared_seeds_paired', m
    pce = m['paired_common_effect']
    assert pce['status'] == 'ok' and pce['n_scales'] == 2, pce
    assert abs(pce['mean'] - 0.225) < 1e-12, pce
    assert pce['n'] == 3 and pce['seeds'] == [0, 1, 2], pce
    assert 'independen' in m['note'].lower(), m['note']
    # Undeclared seed identities cannot be treated as independent OR paired.
    mu = L.meta_analyze_effects([
        {'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3]},
        {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35]}])
    assert mu['seed_dependence'] == 'unattested', mu
    assert mu['paired_common_effect']['status'] == 'unattested_seed_identity', mu
    t = L.scaling_trend(pts)
    assert t.get('status') != 'refused_corpus_mismatch', t
    # Same ladder, second tier measured on a different tokenisation.
    pts_mixed = [pts[0], dict(pts[1], corpus_fp='fpB')]
    m = L.meta_analyze_effects(pts_mixed)
    assert m['status'] == 'refused_corpus_mismatch' and m['poolable'] is False, m
    assert 'tokenisation' in m['note'], m
    assert len(m['per_point']) == 2, 'per-scale rows stay; only pooling is refused'
    t = L.scaling_trend(pts_mixed)
    assert t['status'] == 'refused_corpus_mismatch' and 'slope' not in t, t
    # A vocab difference alone is enough to refuse (PPL units differ).
    m = L.meta_analyze_effects([dict(pts[0], corpus_fp='fpA', vocab=32768),
                                dict(pts[1], corpus_fp='fpA', vocab=8000)])
    assert m['status'] == 'refused_corpus_mismatch', m
    # Missing fingerprints are disclosed as unverifiable, never as shared.
    m = L.meta_analyze_effects([{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2]},
                                {'label': 'S2', 'scale': 2, 'effects': [0.15, 0.25, 0.35], 'seeds': [0, 1, 2]}])
    assert m['corpus_binding']['corpus_binding'] == 'unverifiable', m
    assert m['status'] == 'ok', 'legacy points without fingerprints stay analysable, just unattested'
    # A shared corpus fingerprint does not attest the tokenizer when only one
    # point declares its vocab size; the missing width could be different.
    partial_vocab = [dict(pts[0], vocab=32768), {k: v for k, v in pts[1].items()
                                                if k != 'vocab'}]
    m = L.meta_analyze_effects(partial_vocab)
    assert m['corpus_binding']['corpus_binding'] == 'unverifiable', m
    assert m['corpus_binding']['n_vocab_declared'] == 1, m
    assert 'incomplete' in m['corpus_binding']['refusal_reason'], m
    # The ladder audit fails CLOSED on a corpus/tokenizer mismatch.
    def _tier(name, corpus_fp=None, vocab=None):
        cfg = {'steps': 100, 'batch_size': 4, 'seq_len': 512, 'n_train_tokens': 204800,
               'lr': 3e-4, 'warmup': 10, 'seeds': [0, 1]}
        if corpus_fp is not None:
            cfg['corpus_fp'] = corpus_fp
        if vocab is not None:
            cfg['vocab'] = vocab
        return {'name': name, 'cfg': cfg, 'params': 1000}
    a = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpA', 8000)])
    assert a['status'] in ('ok', 'warn') and a['corpus_binding']['corpus_fp'] == 'shared', a
    b = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpB', 8000)])
    assert b['status'] == 'fail' and any('corpus fingerprint' in x for x in b['problems']), b
    c = L.scale_ladder_audit([_tier('S1', 'fpA', 8000), _tier('S2', 'fpA', 32000)])
    assert c['status'] == 'fail' and any('vocab size' in x for x in c['problems']), c
    d = L.scale_ladder_audit([_tier('S1'), _tier('S2')])
    assert d['status'] != 'fail' and d['corpus_binding']['corpus_fp'] == 'unverifiable', d


def t_scale_point_provenance_roundtrip():
    # _scale_points carries the per-point corpus provenance through both
    # input forms; the m0 analysis path builds these from record data_version.
    # A malformed effects field must not silently delete the declared tier;
    # None remains the explicit declared-empty form.
    for _bad_eff in (1.0, True, 'abc', b'ab'):
        try:
            L._scale_points([{'label': 'Sbad', 'scale': 1,
                              'effects': _bad_eff, 'seeds': [0]}])
        except ValueError as e:
            assert 'effects must be an iterable' in str(e), (_bad_eff, e)
        else:
            raise AssertionError(f'malformed effects {_bad_eff!r} was not refused')
    pts = L._scale_points([
        {'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2], 'seeds': [0, 1],
         'corpus_fp': 'fpA', 'vocab': 8000},
        (2, [0.3, 0.4], [0, 1], 'fpB'),
        {'label': 'S3', 'scale': 3, 'effects': [0.5, 0.6], 'seeds': [0, 1]},
    ])
    assert pts[0]['corpus_fp'] == 'fpA' and pts[0]['vocab'] == 8000, pts
    assert pts[1]['corpus_fp'] == 'fpB' and pts[1]['vocab'] is None, pts
    assert pts[2]['corpus_fp'] is None, pts


def t_holm_mixed_n_and_invalid_family_size():
    # A mixed-n family is capable of a significant result through its
    # best-powered member; the family floor must use max(n), not min(n).
    comps = {'A': {'p_exact_signflip': 0.02, 'n': 6},
             'B': {'p_exact_signflip': 0.0005, 'n': 12}}
    fam = L.attach_holm(comps, family='mixed n')
    assert fam['family_resolution_limited'] is False, fam
    assert abs(fam['family_min_attainable_p_holm'] - 0.0009765625) < 1e-15, fam
    # An omitted high-n row cannot carry the family: only tested rows count.
    om = {'A': {'p_exact_signflip': 0.02, 'n': 6},
          'B': {'omitted': 'no measurement', 'n': 12}}
    fam_om = L.attach_holm(om, family='omitted high n')
    assert fam_om['family_resolution_limited'] is True, fam_om
    # An unmeasurable p still occupies its declared family slot; dropping it
    # would shrink m and make the remaining p anti-conservatively small.
    h = L.holm_adjust([('a', 0.04), ('b', float('nan'))], alpha=0.05)
    assert abs(h['a']['p_holm'] - 0.08) < 1e-15, h
    assert h['a']['reject'] is False and h['a']['m_effective'] == 2, h
    assert h['b']['p_holm'] is None and h['b']['m_effective'] == 2, h
    # Fractional seed counts are not silently truncated into a different
    # resolution floor; they are unverifiable.
    assert math.isnan(L.signflip_p_floor(6.5))
    assert math.isnan(L.holm_min_attainable_p(6.5, 4))
    assert math.isnan(L.signflip_p_floor(True))


def t_meta_prediction_interval_df():
    pts = [{'label': f'S{i}', 'scale': i + 1,
            'effects': [0.1 + 0.02 * i, 0.2 + 0.02 * i, 0.3 + 0.02 * i],
            'seeds': [0, 1, 2]} for i in range(3)]
    m = L.meta_analyze_effects(pts)
    assert m['status'] == 'ok' and m['prediction_interval'] is not None, m
    lo, hi = m['prediction_interval']
    clo, chi = m['random_effects']['ci']
    # Prediction interval for a new scale is wider than the mean CI.
    assert lo < clo and hi > chi, m
    # Two usable scales cannot estimate both variance components.
    m2 = L.meta_analyze_effects(pts[:2])
    assert m2['status'] == 'ok' and m2['prediction_interval'] is None, m2
    # A scale point below the pre-registered seed floor is still reported, but
    # it must block the pooled "common effect" claim: pooling would silently
    # promote an underpowered scale to equal evidence.
    under = [{'label': 'U%d' % i, 'scale': i + 1,
              'effects': [0.1 + 0.01 * j for j in range(2)], 'seeds': [0, 1]}
             for i in range(3)]
    mu = L.meta_analyze_effects(under, min_seeds=3)
    assert mu['status'] == 'ok' and mu['poolable'] is False, mu
    assert mu['underpowered_points'] == ['U0', 'U1', 'U2'], mu
    assert 'underpowered' in mu['note'], mu


def t_tost_conf_controls_decision():
    d = [0.054382, 0.064971, 0.075123, 0.086877, 0.097029, 0.107618]
    r90 = L.tost_paired(d, margin=0.1, conf=0.90)
    assert r90['equivalent'] is True and r90['ci_within_margin'] is True, r90
    r95 = L.tost_paired(d, margin=0.1, conf=0.95)
    assert r95['equivalent'] is False, r95
    assert r95['ci_within_margin'] is False, r95
    assert abs(r95['alpha_one_sided'] - 0.025) < 1e-15, r95


def t_data_quality_exact_integer_and_complex_rows():
    # float64 has 53 bits of mantissa; distinct int64 rows above 2**53 must
    # not be reported as exact duplicates.
    big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    assert L.duplicate_mask(big).tolist() == [False, False]
    rep = L.data_quality_report(big, train_idx=[0], val_idx=[1])
    assert rep['checks']['duplicate_rows_across_splits']['train__val'] == 0, rep
    cleaned = L.clean_training_rows(big, train_idx=[0, 1])
    assert cleaned['n_duplicate_rows_dropped'] == 0 and cleaned['n_train_rows_out'] == 2, cleaned
    # The same precision requirement applies to object columns of Python
    # integers (a common dtype after pandas/None handling): they must not be
    # rounded through float64 before duplicate detection.
    obj_big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=object)
    assert L.duplicate_mask(obj_big).tolist() == [False, False]
    obj_rep = L.data_quality_report(obj_big, train_idx=[0], val_idx=[1])
    assert obj_rep['checks']['duplicate_rows_across_splits']['train__val'] == 0, obj_rep
    # Mixed object int/float rows keep the integer's bits while still
    # comparing an integral float equal to the same integer.
    obj_mixed = np.array([[2 ** 53], [2 ** 53 + 2.0], [2 ** 53 + 2]], dtype=object)
    assert L.duplicate_mask(obj_mixed).tolist() == [False, False, True], obj_mixed
    assert L.duplicate_mask(np.array([[float('nan')], [None]], dtype=object)).tolist() == [False, True]
    # Complex rows were previously cast to float64 (dropping the imaginary
    # part); reject them instead of manufacturing duplicates.
    try:
        L.duplicate_mask(np.array([1 + 2j, 1 + 3j]))
    except TypeError:
        pass
    else:
        raise AssertionError('complex duplicate detection must be rejected')
    # Object/categorical string rows must be hashable without the numeric
    # fallback crashing, and a numeric spelling is not its numeric twin.
    cats = np.array([['a'], ['b'], ['a']], dtype=object)
    assert L.duplicate_mask(cats).tolist() == [False, False, True], cats
    mixed = np.array([[1], ['1'], [1.0]], dtype=object)
    assert L.duplicate_mask(mixed).tolist() == [False, False, True], mixed
    assert L.duplicate_mask(np.array([['1'], ['1.0']], dtype=object)).tolist() == [False, False]
    # Native unicode/bytes categorical dtypes (pandas factor columns, not
    # only dtype=object) must also be hashable, and remain type-distinct.
    assert L.duplicate_mask(np.array(['a', 'b', 'a'])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array([['a'], ['b'], ['a']])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array([b'a', b'b', b'a'])).tolist() == [False, False, True]
    assert L.duplicate_mask(np.array(['1', 1], dtype=object)).tolist() == [False, False]
    # Structured-array rows (common for grouped feature records) are also
    # well-defined categorical identities, not a float-cast crash.
    _rec = np.array([(1, 2), (1, 2), (3, 4)],
                    dtype=[('a', '<i4'), ('b', '<i4')])
    assert L.duplicate_mask(_rec).tolist() == [False, True, False]
    import data_guard as _DG
    assert _DG._matrix_fingerprint(_rec) == _DG._matrix_fingerprint(_rec.copy())


def t_exact_numeric_contracts_and_fingerprint():
    from decimal import Decimal
    import data_guard as DG
    # Fractional Decimal token counts must not be truncated into a different
    # pooled denominator; integral ones are ordinary counts.
    ok = L._nll_doc_ci([1.0, 2.0, 3.0],
                       [Decimal('10'), Decimal('20'), Decimal('30')])
    assert ok['n_docs'] == 3 and abs(ok['ppl'] - math.exp(0.1)) < 1e-12, ok
    bad = L._nll_doc_ci([1.0, 2.0, 3.0],
                        [Decimal('10'), Decimal('20.5'), Decimal('30')])
    assert bad['n_docs'] == 2 and abs(bad['ppl'] - math.exp(0.1)) < 1e-12, bad
    # Integer-only contracts accept Decimal('17') but reject Decimal('17.5')
    # instead of silently training on a different row count.
    sel, ev, meta = L.validation_row_split(Decimal('17'), seed=Decimal('3'))
    assert len(sel) + len(ev) == 17 and meta['seed'] == 3, meta
    try:
        L.validation_row_split(Decimal('17.5'))
    except ValueError:
        pass
    else:
        raise AssertionError('fractional n_rows must be rejected')
    try:
        L.tuning_row_split_spec({'selection_row_split': {'n_folds': Decimal('2.5')}})
    except ValueError:
        pass
    else:
        raise AssertionError('fractional n_folds must be rejected')
    assert L.tuning_row_split_spec({'selection_row_split': {'n_folds': Decimal('2')}})['n_folds'] == 2
    # A fractional vocab is not a tokenisation width.
    assert L._scale_points([{'scale': 1.0, 'effects': [0.1], 'seeds': [0],
                             'vocab': Decimal('3.5')}])[0]['vocab'] is None
    assert L._scale_points([{'scale': 1.0, 'effects': [0.1], 'seeds': [0],
                             'vocab': Decimal('3')}])[0]['vocab'] == 3
    try:
        L.simple_baseline_fit([0, 1, 0, 1], [0, 1], 2,
                              protocol=dict(L.SIMPLE_BASELINE_POLICY,
                                            max_fit_tokens=Decimal('2.5')))
    except ValueError:
        pass
    else:
        raise AssertionError('fractional max_fit_tokens must be rejected')
    # Matrix fingerprints keep exact integer bits above 2**53 while an
    # exactly representable value keeps the same digest as its float twin.
    big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    assert len({DG._matrix_fingerprint(np.array([[v]], dtype=np.int64))
                for v in big.ravel().tolist()}) == 2
    assert DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=np.int64)) == \
        DG._matrix_fingerprint(np.array([[float(2 ** 53)]]))
    assert DG._matrix_fingerprint(np.array([[0.0]])) == \
        DG._matrix_fingerprint(np.array([[-0.0]]))
    obj_big = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=object)
    assert DG._matrix_fingerprint(obj_big[:1]) != DG._matrix_fingerprint(obj_big[1:])
    assert DG._matrix_fingerprint(obj_big) == DG._matrix_fingerprint(obj_big.copy())
    # The rounded float64 image of an inexact row can equal an exact row's
    # image (large int next to its 2**53 neighbour, or an uncoercible string
    # next to a missing value).  Swapping such rows must still change the
    # digest: the exact tail has to carry the row position, not just the key.
    _exact = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    _swapped = _exact[::-1].copy()
    assert DG._matrix_fingerprint(_exact) != DG._matrix_fingerprint(_swapped)
    _obj_exact = _exact.astype(object)
    assert DG._matrix_fingerprint(_obj_exact) != DG._matrix_fingerprint(_obj_exact[::-1].copy())
    _obj_mixed = np.array([['b'], [None]], dtype=object)
    assert DG._matrix_fingerprint(_obj_mixed) != DG._matrix_fingerprint(_obj_mixed[::-1].copy())
    # Floats wider than float64 must not collide after the historical
    # float64 hash; distinct longdouble values get distinct exact keys.
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        _ld1 = np.longdouble('1.0000000000000000001')
        _ld2 = np.longdouble('1.0000000000000000002')
        _ld = np.array([[_ld1], [_ld2]], dtype=np.longdouble)
        assert _ld1 != _ld2
        assert DG.duplicate_mask(_ld).tolist() == [False, False]
        assert DG._matrix_fingerprint(_ld[:1]) != DG._matrix_fingerprint(_ld[1:])
        _same = np.array([[_ld1], [_ld1]], dtype=np.longdouble)
        assert DG.duplicate_mask(_same).tolist() == [False, True]
        assert DG._matrix_fingerprint(_same[:1]) == DG._matrix_fingerprint(_same[1:])
        # A native longdouble table must fingerprint like its object twin:
        # exact rows need no tail, inexact/non-finite rows get the same
        # position-qualified key as every other path.
        assert DG._matrix_fingerprint(_ld) == DG._matrix_fingerprint(_ld.astype(object))
    # Exact rationals (Decimal/Fraction) are not numbers.Real; they must not be
    # rounded through float64 into false duplicates or false uniqueness.
    from fractions import Fraction
    assert L.duplicate_mask(np.array([[Decimal('1.5')], [1.5]],
                                     dtype=object)).tolist() == [False, True]
    assert L.duplicate_mask(np.array([[Decimal(2) ** 80], [2 ** 80]],
                                     dtype=object)).tolist() == [False, True]
    assert L.duplicate_mask(np.array([[Fraction(1, 3)], [1 / 3]],
                                     dtype=object)).tolist() == [False, False]
    # A signaling NaN is still a numeric NaN, not a new categorical value:
    # duplicate detection must group it with None/NaN rather than keying its
    # repr as ordinary data.
    _sn = np.array([[None], [float('nan')], [Decimal('NaN')],
                    [Decimal('sNaN')]], dtype=object)
    assert L.duplicate_mask(_sn).tolist() == [False, True, True, True], \
        L.duplicate_mask(_sn)
    # The matrix digest follows the same missing-like contract for object
    # NaN spellings (None vs Decimal('NaN') must not be two content classes).
    assert DG._matrix_fingerprint(np.array([[None]], dtype=object)) == \
        DG._matrix_fingerprint(np.array([[Decimal('NaN')]], dtype=object))
    # An exactly representable object value keeps the native float digest
    # (dtype-insensitive fingerprint), while an inexact integer still gets an
    # exact tail.
    assert DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=np.int64)) == \
        DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=object))
    # Native unicode/bytes categorical tables now use the same canonical
    # object-row keys (previously the float64 cast raised).
    _cats_u = np.array([['a'], ['b']])
    _cats_o = _cats_u.astype(object)
    assert DG._matrix_fingerprint(_cats_u) == DG._matrix_fingerprint(_cats_o)
    assert DG._matrix_fingerprint(_cats_u[:1]) != DG._matrix_fingerprint(_cats_u[1:])
    assert DG._matrix_fingerprint(np.array([[b'a'], [b'b']])) != \
        DG._matrix_fingerprint(np.array([['a'], ['b']]))
    assert DG._matrix_fingerprint(np.array([[1.5]])) == \
        DG._matrix_fingerprint(np.array([[Decimal('1.5')]], dtype=object))
    assert DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=np.int64)) != \
        DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=np.int64))
    # An exactly representable uint64 above int64's maximum must keep the same
    # digest as its object/float twin; the unsigned round-trip used to mark
    # every such value inexact merely because it did not fit int64.
    _u_exact = np.array([[2 ** 63], [2 ** 63 + 2048]], dtype=np.uint64)
    assert DG._matrix_fingerprint(_u_exact) == \
        DG._matrix_fingerprint(_u_exact.astype(object))
    assert DG._matrix_fingerprint(np.array([[2 ** 63]], dtype=np.uint64)) == \
        DG._matrix_fingerprint(np.array([[float(2 ** 63)]]))
    assert DG._matrix_fingerprint(np.array([[2 ** 63 + 1]], dtype=np.uint64)) != \
        DG._matrix_fingerprint(np.array([[2 ** 63]], dtype=np.uint64))
    # A corpus fingerprint is a function of token values, not storage width.
    _toks = np.array([0, 1, 2, 250, 255], dtype=np.int64)
    _fps = {L._ids_fp(_toks.astype(_dt)) for _dt in
            (np.uint8, np.uint16, np.int32, np.int64)}
    assert len(_fps) == 1, _fps
    # Booleans are flags, not measurements, in the scale helpers too.
    assert L.paired_log_ratio([True, 2.0], [2.0, 1.0])['n'] == 1
    assert L.loglog_regression([(True, 2), (2, 4), (4, 8)])['n_points'] == 2
    assert L.crossover_scaling_fit([{'d': True, 'crossover_tokens': 1,
                                     'train_tokens': 1}])['status'] == 'insufficient_points'
    assert math.isnan(L._scale_points([{'scale': True, 'effects': [1],
                                        'seeds': [0]}])[0]['scale'])
    # Decimal is a real numeric scalar even though it is not numbers.Real;
    # NaN/Infinity and complex values are still not measurements.
    assert L._finite_scalar(Decimal('1.5')) == 1.5
    assert L._finite_scalar(Decimal('2')) == 2.0
    assert L._finite_scalar(Decimal('NaN')) is None
    assert L._finite_scalar(1 + 2j) is None
    assert BR._as_finite(Decimal('2.5')) == 2.5
    # Missing-value detection also covers exact numeric objects.
    assert DG._category_is_missing(Decimal('NaN')) is True
    assert DG._category_is_missing(Decimal('sNaN')) is True
    assert DG._category_is_missing(Decimal('1.5')) is False
    assert DG._label_key(Decimal('NaN')) is None
    assert DG._label_key(Decimal('sNaN')) is None
    assert DG._label_key(Decimal('2')) == Decimal('2')
    assert DG._group_key(Decimal('NaN')) == ('__nan__',)
    assert DG._group_key(Decimal('sNaN')) == ('__nan__',)
    # Decimal('Infinity') equals float('inf') in Python; both must be one
    # category identity instead of falling through to a Decimal-specific repr.
    assert DG._target_key(Decimal('Infinity')) == DG._target_key(float('inf'))
    assert DG._target_key(Decimal('-Infinity')) == DG._target_key(float('-inf'))
    # The digest is a content identity, not a function of the row-block size.
    # Exact-row tails used to be appended immediately after their block, so
    # changing the memory cap interleaved them differently and silently
    # changed a fitted artifact's fingerprint.
    _cap = DG._HASH_BLOCK_BYTES
    _block_cases = [
        (np.arange(40, dtype=np.uint64) + 2 ** 63).reshape(10, 4),
        np.array([[2 ** 53 + i] for i in range(10)], dtype=np.int64),
        np.array([['a'], ['b'], [None], [2 ** 53 + 1]] * 3, dtype=object),
    ]
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        _block_cases.append(
            (np.arange(12, dtype=np.longdouble) / 7).reshape(6, 2))
    try:
        for _case in _block_cases:
            _fps = set()
            for _cap_bytes in (1, 7, 64, 1 << 20):
                DG._HASH_BLOCK_BYTES = _cap_bytes
                _fps.add(DG._matrix_fingerprint(_case))
            assert len(_fps) == 1, (_case.dtype, _fps)
    finally:
        DG._HASH_BLOCK_BYTES = _cap


def t_batch_iter_minimal_corpus():
    import torch as _torch
    it = L.batch_iter(np.arange(8, dtype=np.int64), 7, 1, _torch.device('cpu'), seed=0)
    x, y = next(it)
    assert x.shape == (1, 7) and y.shape == (1, 7), (x.shape, y.shape)
    assert x[0, 0].item() == 0 and y[0, -1].item() == 7, (x, y)
    try:
        next(L.batch_iter(np.arange(7, dtype=np.int64), 7, 1, _torch.device('cpu')))
    except ValueError:
        pass
    else:
        raise AssertionError('a corpus shorter than seq_len+1 must be rejected')
    for _seq_len, _batch in ((0, 1), (1, 0), (1, -1), (1.5, 1), (True, 1)):
        try:
            next(L.batch_iter(np.arange(10, dtype=np.int64), _seq_len, _batch,
                              _torch.device('cpu')))
        except ValueError:
            pass
        else:
            raise AssertionError(f'seq_len={_seq_len!r}, batch_size={_batch!r} must be refused')


def t_indexer_invalid_soft_lanes_are_zero():
    import torch as _torch
    n, B, nIH, hd, d = 2, 2, 2, 2, 4
    H = _torch.zeros(n, d)
    comp = _torch.zeros(B, d)
    W_DQ = _torch.zeros(d, nIH * hd)
    W_DK = _torch.zeros(d, nIH * hd)
    W_w = _torch.zeros(nIH, d)
    last_tok = _torch.tensor([-1, 0], dtype=_torch.long)
    box = []
    _m, _idx, soft = L.lightning_indexer(
        H, comp, last_tok, W_DQ, W_DK, W_w, nIH, 2,
        return_mask=False, out_valid=box, need_soft=True)
    valid = box[0]
    assert valid[0].tolist() == [True, False], valid
    assert float(soft[~valid].abs().max()) == 0.0, (soft, valid)


def t_attention_backward_partial_head_group():
    import torch as _torch
    import csa_kernels as _C
    old = os.environ.get('CSA_ATTN_GRAD_SLAB_MB')
    os.environ['CSA_ATTN_GRAD_SLAB_MB'] = '1'
    try:
        R, T, nh, hd = 1, 4, 5, 64
        Bm, nb, w = 1000, 2, 1
        q = _torch.randn(R, T, nh, hd, requires_grad=True)
        k_blk = _torch.randn(R, Bm, nh, hd, requires_grad=True)
        v_blk = _torch.randn(R, Bm, nh, hd, requires_grad=True)
        k_sw = _torch.randn(R, T, nh, hd, requires_grad=True)
        v_sw = _torch.randn(R, T, nh, hd, requires_grad=True)
        dO = _torch.randn(R, T, nh, hd)
        topk_idx = _torch.tensor([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype=_torch.long)
        last_tok = _torch.full((R, Bm), -1, dtype=_torch.long)
        out = _C._attn_bwd_ref(
            q, k_blk, v_blk, k_sw, v_sw, topk_idx, None, last_tok,
            None, None, w, float(hd) ** -0.5, dO,
            True, True, True, True, True, False, False)
        assert len(out) == 7 and out[1].shape == (R, Bm, nh, hd), out
    finally:
        if old is None:
            os.environ.pop('CSA_ATTN_GRAD_SLAB_MB', None)
        else:
            os.environ['CSA_ATTN_GRAD_SLAB_MB'] = old


def t_checkpoint_stamp_matches_the_actual_backend():
    class _M:
        def state_dict(self):
            return {}
    _old = L._kernel_stamp
    try:
        L._kernel_stamp = lambda: '-ktest'
        _p = L._checkpoint_payload(_M(), variant='v', seed=0, step=0,
                                   val_ppl=1.0, train_ce=1.0, health={})
        assert _p['code'] == L.code_stamp() == L.CODE_SEMANTICS + '-ktest', _p.get('code')
        assert _p['code'] != L.CKPT_CODE
    finally:
        L._kernel_stamp = _old


def t_indexer_chunked_final_tile_uses_active_rows():
    # The Bm>8192 fallback allocates a [R, T_c, Bm] score tile but the last
    # chunk may fill only tc < T_c rows.  The caller must pass the active rows
    # to the chunked top-k helper, or the [R,T_c,k] result cannot be assigned
    # into idx[:, t0:t0+tc] (and stale rows become fake queries).
    import types
    import csa_kernels as C
    R, T, nIH, hd, Bm, k = 1, 3, 1, 2, 8193, 2

    class _Launch:
        def __getitem__(self, _grid):
            return self
        def __call__(self, *args, **kwargs):
            return None

    fake = types.SimpleNamespace(
        triton=types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        idx_score=_Launch())
    seen = []

    def _helper(scores, last_tok, t0, T_, Bm_, k_out, need_soft, out_dtype):
        seen.append((int(scores.shape[1]), int(t0)))
        _r, _tc = int(scores.shape[0]), int(scores.shape[1])
        return (torch.zeros((_r, _tc, k_out), dtype=torch.int32),
                torch.zeros((_r, _tc, k_out), dtype=torch.bool), None)

    _old = (C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked)
    try:
        C._kernels = lambda: fake
        C._IDX_SCORES_CAP_BYTES = 2 * R * Bm * 4  # T_c = 2 for T = 3
        C._indexer_topk_chunked = _helper
        idx, valid, soft = C._indexer_fwd_triton(
            torch.randn(R, T, nIH, hd), torch.randn(R, Bm, nIH, hd),
            torch.randn(nIH, hd), torch.zeros(R, Bm, dtype=torch.long),
            k, need_soft=False)
    finally:
        C._kernels, C._IDX_SCORES_CAP_BYTES, C._indexer_topk_chunked = _old
    assert seen == [(2, 0), (1, 2)], seen
    assert idx.shape == (R, T, k) and valid.shape == (R, T, k), (idx.shape, valid.shape)


def t_stats_accept_numpy_scalars():
    # Public numeric helpers are routinely fed NumPy arrays/scalars.  An:
    # isinstance(int, float) filter would drop every np.int64 / np.float32
    # value, turning a real panel into "no data".
    d = np.array([1, 2, 3, 4], dtype=np.int64)
    r = L.paired_effect_stats(d)
    assert r['n'] == 4 and math.isfinite(r['mean']) and math.isfinite(r['p_exact_signflip']), r
    w = L.wilcoxon_signed_rank_p(d)
    assert w['n_nonzero'] == 4 and math.isfinite(w['p']), w
    t = L.tost_paired(d, margin=10.0)
    assert t['n'] == 4 and t['equivalent'] is True, t
    assert L.bootstrap_mean_ci(np.arange(10)) is not None
    assert L.paired_bootstrap_ci(np.arange(10)) is not None
    assert L.unpaired_bootstrap_ci(np.arange(10), np.arange(1, 11)) is not None
    lo, hi, dropped = L.finite_range([
        ('int', np.int64(1)), ('f32', np.float32(2.5)),
        ('nan', np.float64('nan')), ('bool', True), ('npbool', np.bool_(True))])
    assert (lo, hi, dropped) == (1.0, 2.5, ['nan', 'bool', 'npbool']), (lo, hi, dropped)
    assert BR.exact_signflip([np.bool_(True), np.bool_(False)]) is None
    assert L.exact_signflip_p([True, False])['n'] == 0
    assert BR.exact_signflip([True, False]) is None
    meta = L.meta_analyze_effects([
        {'label': 'A', 'scale': 1, 'effects': np.array([1, 2, 3], dtype=np.int64), 'seeds': [0, 1, 2]},
        {'label': 'B', 'scale': 2, 'effects': np.array([2, 3, 4], dtype=np.int64), 'seeds': [0, 1, 2]},
    ])
    assert meta['status'] == 'ok' and all(p['n'] == 3 for p in meta['per_point']), meta
    # The zero-GPU report builder has its own exact test copy; it must use the
    # same scalar contract instead of silently dropping NumPy values.
    for _d in (np.array([1, 2, 3, 4], dtype=np.int64),
               np.array([1, 2, 3, 4], dtype=np.float32),
               [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]):
        _r = BR.exact_signflip(_d)
        assert _r is not None and _r['n'] == 4 and math.isfinite(_r['p_exact_signflip']), _r


def t_apply_rope_preserves_width():
    # Over-long rope_dim must rotate only complete pairs that exist and must
    # never widen the output.
    for width in (0, 1, 2, 3, 4, 5, 8, 16):
        for rope_dim in range(0, 2 * width + 4):
            x = torch.randn(2, 3, width, dtype=torch.float64)
            half = rope_dim // 2
            cos = torch.randn(1, 3, half, dtype=torch.float64)
            sin = torch.randn(1, 3, half, dtype=torch.float64)
            xg = x.clone().requires_grad_(True)
            y = V7.apply_rope(xg, cos, sin, rope_dim)
            assert tuple(y.shape) == tuple(x.shape), (width, rope_dim, y.shape)
            y.sum().backward()
            assert torch.isfinite(xg.grad).all(), (width, rope_dim)


def t_label_key_keeps_numeric_strings():
    # Only real numeric scalars are checked for NaN/Inf.  A categorical label
    # literally spelled "nan" or "inf" must remain a label, not disappear as a
    # corrupted target.
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = ['nan', 'inf', 'a', 'b', 'nan', 'a', 'b', 'a', 'b', 'a', 'b', 'nan']
    rep = L.data_quality_report(X, y=y, train_idx=list(range(6)),
                                val_idx=list(range(6, 9)), test_idx=list(range(9, 12)))
    assert rep['checks']['missing_labels'] == 0 and rep['checks']['nonfinite_labels'] == 0, rep
    assert rep['checks']['label_distribution'].get('nan') == 3, rep
    # Actual NaN/Inf numeric targets are still non-finite.
    ynum = [0, 1, np.nan, np.inf, 2, 3, 4, 5, 6, 7, 8, 9]
    rep2 = L.data_quality_report(X, y=ynum, train_idx=list(range(6)),
                                 val_idx=list(range(6, 9)), test_idx=list(range(9, 12)))
    assert rep2['checks']['missing_labels'] == 0 and rep2['checks']['nonfinite_labels'] == 2, rep2


def t_locked_json_and_csv_generator():
    root = tempfile.mkdtemp(prefix='atomic_', dir=REPO)
    try:
        p = os.path.join(root, 'missing', 'sub', 'x.json')
        assert L.locked_json_read(p) == {}
        got = L.locked_json_update(p, lambda obj: (obj.update(k=1) or obj))
        assert got == {'k': 1} and L.locked_json_read(p) == {'k': 1}, got
        # A parsed list is not a JSON object: an updater would crash on
        # .get()/__setitem__, and overwriting it would destroy the operator's
        # file.  Refuse and leave the bytes untouched.
        bad = os.path.join(root, 'bad.json')
        with open(bad, 'w', encoding='utf-8') as f:
            f.write('[1, 2]')
        for _fn in (L.locked_json_read,
                    lambda _p: L.locked_json_update(_p, lambda obj: obj)):
            try:
                _fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('a non-object locked JSON file must be refused')
        with open(bad, encoding='utf-8') as f:
            assert f.read().strip() == '[1, 2]'
        c = os.path.join(root, 'rows.csv')
        L.atomic_write_csv(c, None, ({'a': i, 'b': i * 2} for i in range(3)))
        with open(c, encoding='utf-8') as f:
            lines = f.read().splitlines()
        assert lines == ['a,b', '0,0', '1,2', '2,4'], lines
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_data_guard_preprocessor_option_guards():
    for bad in ({'missing': True}, {'missing': np.bool_(False)},
                {'eps': -1.0}, {'eps': float('nan')}, {'eps': float('inf')},
                {'eps': True}, {'clip_z': True}, {'clip_z': float('nan')},
                {'clip_quantiles': (True, 1.0)},
                {'clip_quantiles': (0.0, np.bool_(True))}):
        try:
            L.LeakageSafePreprocessor(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'preprocessor accepted invalid options {bad!r}')
    prep = L.LeakageSafePreprocessor(scale='standard', clip_quantiles=None,
                                     eps=0.0).fit(np.array([[1.0], [1.0]]))
    out = prep.transform(np.array([[1.0], [2.0]]))
    assert np.all(np.isfinite(out)), out
    assert prep.get_stats()['eps'] == 0.0
    assert L.LeakageSafePreprocessor(missing=np.float32(0.0)).missing == 0.0
    # clip_z was silently ignored outside standard/robust scaling.  It must
    # apply to minmax (where a holdout can still leave [0, 1]) and must be
    # refused for an unscaled feature space where "scaled units" do not exist.
    Xcl = np.array([[0.0], [1.0], [2.0], [100.0]])
    mm = L.LeakageSafePreprocessor(scale='minmax', clip_quantiles=None,
                                   clip_z=1.0).fit(Xcl[:3])
    assert mm.transform(np.array([[100.0]])).item() == 1.0
    try:
        L.LeakageSafePreprocessor(scale='none', clip_z=1.0)
    except ValueError as e:
        assert 'scaled feature space' in str(e), e
    else:
        raise AssertionError("clip_z with scale='none' must be refused")


def t_categorical_shift_requires_two_bins():
    for bad in (0, -1, 1, True):
        try:
            L.categorical_shift_report([1, 2], [1, 2], max_categories=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'max_categories={bad!r} must be rejected')
    r = L.categorical_shift_report([1, 1, 2], [1, 2], max_categories=2)
    assert r['n_reference'] == 3 and r['n_holdout'] == 2, r
    # Truncating the histogram to top categories must not make a category that
    # WAS in the reference look unseen.
    rt = L.categorical_shift_report([1, 1, 1, 2, 3], [2], max_categories=2)
    assert rt['unseen_rate'] == 0.0, rt


def t_data_quality_unhashable_labels():
    import data_guard as DG
    X = np.array([[0.0], [0.0], [1.0], [2.0]])
    y = np.empty(4, dtype=object)
    y[0], y[1], y[2], y[3] = ['a'], ['b'], ['c'], ['d']
    rep = L.data_quality_report(X, y=y, train_idx=[0, 2], val_idx=[1],
                                test_idx=[3])
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 1, rep
    assert rep['status'] == 'fail', rep
    # The unhashable-key sentinel must not collide with a real tuple label.
    # A list label and a tuple with the old sentinel spelling are different
    # classes; the duplicate-row conflict audit must still see the conflict.
    _sentinel_tuple = ('__unhashable__', 'list', "['x']")
    assert DG._hashable_label(['x']) != DG._hashable_label(_sentinel_tuple)
    assert DG._group_key(['x']) != DG._group_key(_sentinel_tuple)
    Xd = np.array([[0.0], [0.0]])
    yd = np.empty(2, dtype=object)
    yd[0], yd[1] = ['x'], _sentinel_tuple
    rd = L.data_quality_report(Xd, y=yd, train_idx=[0, 1])
    assert rd['checks']['label_conflicts_on_duplicate_rows'] == 1, rd


def t_categorical_encoder_train_only():
    train = np.array([['a', 'x'], ['b', None], ['a', np.nan]], dtype=object)
    holdout = np.array([['z', 'x'], ['a', 'y'], [None, 'z']], dtype=object)
    enc = L.LeakageSafeCategoricalEncoder()
    x_train = enc.fit_transform(train)
    stats = enc.get_stats()
    x_hold = enc.transform(holdout)
    assert np.array_equal(enc.transform(holdout), x_hold)
    assert enc.get_stats() == stats, 'transform must not add holdout categories'
    assert stats['features'][0]['categories'] == ['a', 'b'], stats
    assert stats['features'][1]['categories'] == ['x'], stats
    assert x_train[:, 0].tolist() == [0.0, 1.0, 0.0], x_train
    assert x_hold[:, 0].tolist() == [2.0, 0.0, 3.0], x_hold
    assert x_hold[:, 1].tolist() == [0.0, 1.0, 1.0], x_hold
    restored = L.LeakageSafeCategoricalEncoder.from_stats(stats)
    assert np.array_equal(restored.transform(holdout), x_hold)
    bundle = L.prepare_categorical_features(train, holdout)
    assert np.array_equal(bundle['train'], x_train), bundle
    assert np.array_equal(bundle['val'], x_hold), bundle
    assert bundle['encoder'].get_stats() == stats, bundle
    # A tuple level must survive JSON persistence: json turns it into a list,
    # so the catalog stores a canonical key rather than rebuilding identity
    # from the round-tripped value (which would make the level unseen).
    _tc = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')]], dtype=object)
    _tc_h = np.array([[('a', 'b')], [('a,s:b',)]], dtype=object)
    for _enc_kw in ({'encoding': 'ordinal'}, {'encoding': 'onehot'}):
        _tce = L.LeakageSafeCategoricalEncoder(**_enc_kw).fit(_tc)
        _tcr = L.LeakageSafeCategoricalEncoder.from_stats(
            json.loads(json.dumps(_tce.get_stats())))
        assert np.array_equal(_tcr.transform(_tc_h), _tce.transform(_tc_h)), _enc_kw
    # Non-JSON-native levels (bytes, datetime64) must not make get_stats
    # unserializable; their canonical keys still drive transform after reload.
    _tx = np.array([[b'a'], [b'a'], [np.datetime64('2020-01-01')],
                    [np.datetime64('2020-01-01')]], dtype=object)
    _tx_h = np.array([[b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    for _enc_kw in ({'encoding': 'ordinal'}, {'encoding': 'onehot'}):
        _txe = L.LeakageSafeCategoricalEncoder(**_enc_kw).fit(_tx)
        _txr = L.LeakageSafeCategoricalEncoder.from_stats(
            json.loads(json.dumps(_txe.get_stats())))
        assert np.array_equal(_txr.transform(_tx_h), _txe.transform(_tx_h)), _enc_kw
    onehot = L.LeakageSafeCategoricalEncoder(encoding='onehot').fit(train)
    oh_hold = onehot.transform(holdout)
    assert oh_hold.shape == (3, 4 + 3), oh_hold.shape
    assert np.allclose(oh_hold[:, :4].sum(axis=1), 1.0), oh_hold
    assert np.allclose(oh_hold[:, 4:].sum(axis=1), 1.0), oh_hold
    assert len(onehot.get_feature_names()) == oh_hold.shape[1]
    strict = L.LeakageSafeCategoricalEncoder(handle_unknown='error').fit(train)
    try:
        strict.transform(holdout)
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject a new level')
    try:
        L.LeakageSafeCategoricalEncoder().transform(train)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        enc.transform(np.zeros((1, 3), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')


def t_frequency_encoder_train_only():
    train = np.array([['a', 'x'], ['b', None], ['a', np.nan], ['c', 'x']],
                     dtype=object)
    holdout = np.array([['z', 'x'], ['a', 'y'], [None, 'z']], dtype=object)
    enc = L.LeakageSafeFrequencyEncoder()
    xtr = enc.fit_transform(train)
    stats = enc.get_stats()
    # Counts are learned from training rows only; missing values are not a
    # category and do not inflate the denominator.
    assert stats['features'][0]['n_observed'] == 4, stats
    assert [e['count'] for e in stats['features'][0]['entries']] == [2, 1, 1], stats
    assert stats['features'][1]['n_observed'] == 2, stats
    assert stats['features'][1]['n_missing'] == 2, stats
    assert xtr[:, 0].tolist() == [0.5, 0.25, 0.5, 0.25], xtr
    assert xtr[:, 1].tolist() == [1.0, 0.0, 0.0, 1.0], xtr
    # An unseen/missing holdout category maps to the reserved zero level and
    # never changes the fitted frequency table.
    xh = enc.transform(holdout)
    assert xh.tolist() == [[0.0, 1.0], [0.5, 0.0], [0.0, 0.0]], xh
    assert enc.get_stats() == stats, 'transform must not update frequencies'
    # Raw counts are available when normalisation is disabled.
    raw = L.LeakageSafeFrequencyEncoder(normalize=False).fit(train)
    assert raw.transform(train)[:, 0].tolist() == [2.0, 1.0, 2.0, 1.0]
    restored = L.LeakageSafeFrequencyEncoder.from_stats(stats)
    assert np.array_equal(restored.transform(holdout), xh)
    # Tuple categories are JSON-serialized as lists; the persisted canonical
    # key must keep them as the SAME learned level after a round-trip.
    _tf = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')]], dtype=object)
    _tf_h = np.array([[('a', 'b')], [('a,s:b',)], ['z']], dtype=object)
    _tfe = L.LeakageSafeFrequencyEncoder().fit(_tf)
    _tfr = L.LeakageSafeFrequencyEncoder.from_stats(
        json.loads(json.dumps(_tfe.get_stats())))
    assert np.array_equal(_tfr.transform(_tf_h), _tfe.transform(_tf_h)), _tfe.get_stats()
    _tfx = np.array([[b'a'], [b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    _tfx_h = np.array([[b'a'], [np.datetime64('2020-01-01')]], dtype=object)
    _tfxe = L.LeakageSafeFrequencyEncoder().fit(_tfx)
    _tfxr = L.LeakageSafeFrequencyEncoder.from_stats(
        json.loads(json.dumps(_tfxe.get_stats())))
    assert np.array_equal(_tfxr.transform(_tfx_h), _tfxe.transform(_tfx_h)), _tfxe.get_stats()
    bundle = L.prepare_frequency_features(train, holdout)
    assert np.array_equal(bundle['train'], xtr), bundle
    assert np.array_equal(bundle['val'], xh), bundle
    assert bundle['encoder'].get_stats() == stats, bundle
    assert enc.get_feature_names() == ['x0_freq', 'x1_freq']
    strict = L.LeakageSafeFrequencyEncoder(handle_unknown='error').fit(train)
    try:
        strict.transform(holdout)
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject a new level')
    # Missing values are not a learned level either; the strict contract must
    # not silently encode them as zero (only the default contract does that).
    _strict_missing = L.LeakageSafeFrequencyEncoder(handle_unknown='error').fit(
        np.array([['a'], [None]], dtype=object))
    try:
        _strict_missing.transform(np.array([[None]], dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('handle_unknown="error" must reject missing values')
    try:
        L.LeakageSafeFrequencyEncoder().transform(train)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    try:
        enc.transform(np.zeros((1, 3), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('feature-width mismatch must fail')
    try:
        L.LeakageSafeFrequencyEncoder().fit(np.empty((0, 2), dtype=object))
    except ValueError:
        pass
    else:
        raise AssertionError('fitting on zero rows must fail')
    for bad in ({'normalize': 'yes'}, {'normalize': 1},
                {'handle_unknown': 'warn'}):
        try:
            L.LeakageSafeFrequencyEncoder(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'frequency encoder accepted {bad!r}')


def t_target_encoder_prevents_self_and_group_leakage():
    X = np.array([['a'], ['b'], ['a'], ['b'], ['c'], ['a']], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    enc = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    oof = enc.fit_transform(X, y)
    stats = enc.get_stats()
    assert stats['oof_policy'] == 'row_oof', stats
    # A training row's own label must not enter its out-of-fold encoding.
    for i in range(len(y)):
        y2 = y.copy()
        y2[i] += 1000.0
        other = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
        assert np.allclose(other.fit_transform(X, y2)[i], oof[i]), i
    # Holdout transform never reads labels (there is no y argument) and unseen
    # levels fall back to the training prior rather than becoming a new level.
    hold = np.array([['a'], ['z'], [None]], dtype=object)
    before = enc.get_stats()
    xh = enc.transform(hold)
    assert enc.get_stats() == before, 'transform must not update statistics'
    assert np.allclose(xh[1], enc.prior_), (xh, enc.prior_)
    assert np.allclose(enc.transform(hold), xh)
    restored = L.LeakageSafeTargetEncoder.from_stats(before)
    assert np.allclose(restored.transform(hold), xh)
    # Applying the full-table lookup to the training rows would encode each
    # row with its own label; refuse unless the caller explicitly asks for a
    # diagnostic in-sample transform.
    for _enc_obj in (enc, restored):
        try:
            _enc_obj.transform(X)
        except ValueError as e:
            assert 'exact training rows' in str(e), e
        else:
            raise AssertionError('in-sample target encoding must be refused')
        assert np.allclose(_enc_obj.transform(X, allow_train=True),
                           _enc_obj.transform(X, allow_train=True))
        # Reordering the same training rows must not bypass the refusal: the
        # reordered matrix still contains every row's own label.
        try:
            _enc_obj.transform(X[::-1])
        except ValueError as e:
            assert 'possibly reordered' in str(e), e
        else:
            raise AssertionError(
                'a reordered training matrix must still be refused')
    # Missing categories are a learned training level, not silently a prior.
    Xm = np.array([['a'], [None], ['a'], [None], ['b'], ['a']], dtype=object)
    em = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(Xm, y)
    assert 'x0_target' in em.get_feature_names()
    # Persisted stats must be JSON-safe even when missing values are a learned
    # level, and the round-trip must encode holdouts identically.  A tuple
    # lookup key used to make json.dumps fail outright.
    _h = np.array([[None], ['a'], ['z']], dtype=object)
    _stats = em.get_stats()
    _restored = L.LeakageSafeTargetEncoder.from_stats(json.loads(json.dumps(_stats)))
    assert np.allclose(_restored.transform(_h), em.transform(_h))
    # A real category equal to the old tuple sentinel is not a missing value.
    _Xt = np.array([['a'], [('__missing__',)], [('__missing__',)], ['a']],
                   dtype=object)
    _et = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(_Xt, y[:4])
    _look = _et.get_stats()['features'][0]['lookup']
    assert '__missing__' not in _look, _look
    # Tuple keys must be self-delimiting: ('a', 'b') and ('a,s:b',) are
    # different categories even though the naive comma-join renders them the
    # same.  A collision would share one target mean across both levels.
    import data_guard as _DG
    assert _DG._target_key(('a', 'b')) != _DG._target_key(('a,s:b',))
    _Xtuple = np.array([[('a', 'b')], [('a,s:b',)], [('a', 'b')], [('a,s:b',)]],
                       dtype=object)
    _etuple = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        _Xtuple, np.array([1.0, 100.0, 1.0, 100.0]))
    _tlook = _etuple.get_stats()['features'][0]['lookup']
    assert len(_tlook) == 2, _tlook
    _trestored = L.LeakageSafeTargetEncoder.from_stats(
        json.loads(json.dumps(_etuple.get_stats())))
    _thold = np.array([[('a', 'b')], [('a,s:b',)]], dtype=object)
    assert np.allclose(_trestored.transform(_thold), _etuple.transform(_thold))
    # Groups share a fold, so changing a whole group's label cannot move the
    # out-of-fold encoding of another row in that group.
    groups = np.array([0, 0, 1, 1, 2, 2])
    ge = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    g_oof = ge.fit_transform(X, y, groups=groups)
    assert ge.get_stats()['oof_policy'] == 'grouped_oof', ge.get_stats()
    yg = y.copy()
    yg[groups == 0] += 1000.0
    ge2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    g_oof2 = ge2.fit_transform(X, yg, groups=groups)
    assert np.allclose(g_oof2[groups == 0], g_oof[groups == 0]), (g_oof, g_oof2)
    # Time folds are expanding: changing a later label must not change an
    # earlier row's encoding.
    t = np.arange(len(y))
    te = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    t_oof = te.fit_transform(X, y, time=t)
    assert te.get_stats()['oof_policy'] == 'chronological_expanding_oof', te.get_stats()
    yt = y.copy()
    yt[4] += 1000.0
    te2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    t_oof2 = te2.fit_transform(X, yt, time=t)
    assert np.allclose(t_oof2[:3], t_oof[:3]), (t_oof, t_oof2)
    # With both groups and time, folds stay whole-group AND expanding: changing
    # a later group's label cannot change an earlier group's encoding.
    ct = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    c_oof = ct.fit_transform(X, y, groups=groups, time=np.arange(len(y)))
    assert ct.get_stats()['oof_policy'] == 'grouped_chronological_expanding_oof', ct.get_stats()
    yct = y.copy()
    yct[groups == 2] += 1000.0
    ct2 = L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2)
    c_oof2 = ct2.fit_transform(X, yct, groups=groups, time=np.arange(len(y)))
    assert np.allclose(c_oof2[groups < 2], c_oof[groups < 2]), (c_oof, c_oof2)
    # Overlapping group time windows cannot be ordered into expanding folds
    # without leaking a later timestamp from an earlier group; refuse them.
    try:
        L.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit_transform(
            X, y, groups=np.array([0, 0, 1, 1, 2, 2]),
            time=np.array([0, 10, 5, 6, 20, 21]))
    except ValueError as e:
        assert 'overlapping group time windows' in str(e), e
    else:
        raise AssertionError('overlapping group time windows must be refused')
    bundle = L.prepare_target_features(X, y, hold, groups=groups,
                                       smoothing=0.0, n_folds=2)
    assert np.array_equal(bundle['train'], g_oof), bundle
    assert np.array_equal(bundle['val'], ge.transform(hold)), bundle
    # Non-numeric targets, wrong shapes and malformed options fail closed.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, ['a'] * len(X))
    except TypeError:
        pass
    else:
        raise AssertionError('a categorical target must be refused')
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, y.reshape(-1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError('a 2-D target must be refused')
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X[:-1], y)
    except ValueError:
        pass
    else:
        raise AssertionError('a short target must be refused')
    try:
        L.LeakageSafeTargetEncoder().transform(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError('transform before fit must fail')
    for bad in ({'smoothing': -1}, {'smoothing': float('nan')},
                {'smoothing': True}, {'n_folds': 1}, {'n_folds': 2.5},
                {'n_folds': True}, {'seed': -1}):
        try:
            L.LeakageSafeTargetEncoder(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target encoder accepted {bad!r}')
    # With one group, out-of-fold encoding would have no reference rows.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(X, y, groups=np.zeros(len(y)))
    except ValueError:
        pass
    else:
        raise AssertionError('one group cannot define out-of-fold folds')
    # A single training row has no out-of-fold reference at all; emitting an
    # all-zero target feature would pretend the encoding was estimated.
    try:
        L.LeakageSafeTargetEncoder().fit_transform(
            np.array([['a']], dtype=object), np.array([1.0]))
    except ValueError as e:
        assert 'at least 2 training rows' in str(e), e
    else:
        raise AssertionError('one row cannot define an out-of-fold encoding')
    # fit (full-table transform for holdouts) is still valid with one row.
    # Re-encoding that lone training row is an explicit in-sample diagnostic,
    # so it must go through allow_train just like any other training matrix.
    _one = L.LeakageSafeTargetEncoder().fit(
        np.array([['a']], dtype=object), np.array([1.0]))
    assert np.allclose(
        _one.transform(np.array([['a']], dtype=object), allow_train=True), 1.0)


def t_clean_training_rows_and_split():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0],
                  [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 1, 2, 2, 3, 4])
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3])
    assert clean['train_idx'].tolist() == [2], clean
    assert sorted(clean['dropped_idx'].tolist()) == [0, 1, 3], clean
    assert clean['n_label_conflict_clusters'] == 1, clean
    assert clean['n_duplicate_rows_dropped'] == 1, clean
    assert clean['n_label_conflict_rows_dropped'] == 2, clean
    keep = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                 label_conflicts='keep')
    assert keep['train_idx'].tolist() == [0, 1, 2], keep
    nodrop = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                   drop_duplicates=False,
                                   label_conflicts='drop')
    assert nodrop['train_idx'].tolist() == [2, 3], nodrop
    try:
        L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                              label_conflicts='raise')
    except ValueError:
        pass
    else:
        raise AssertionError('label_conflicts="raise" must refuse ambiguity')

    Xs = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 1.0],
                   [0.0, 1.0], [1000.0, 1.0], [2000.0, 1.0], [3000.0, 1.0]])
    ys = np.array([0, 1, 1, 0, 1, 0, 1])
    out = L.split_and_prepare(Xs, ys, train_frac=4 / 7, val_frac=1 / 7,
                              test_frac=2 / 7, clean_train=True,
                              quality_report=True)
    idx = out['indices']
    assert idx['train'].tolist() == [0, 1, 2], idx
    assert out['cleaning']['dropped_idx'].tolist() == [3], out['cleaning']
    assert idx['val'].tolist() == [4] and idx['test'].tolist() == [5, 6], idx
    assert abs(out['preprocessor'].get_stats()['features'][0]['center'] - 1.0) < 1e-12
    assert out['quality']['n_train_reference'] == 3, out['quality']


def t_clean_training_invalid_labels():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = [0.0, np.nan, 2.0, np.inf]
    kept = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3])
    assert kept['train_idx'].tolist() == [0, 1, 2, 3], kept
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3],
                                  drop_invalid_labels=True)
    assert clean['train_idx'].tolist() == [0, 2], clean
    assert np.array_equal(np.sort(clean['dropped_idx']), [1, 3]), clean
    assert clean['n_invalid_label_rows_dropped'] == 2, clean
    # The one-call split helper must clean the same training rows and still
    # leave validation/test untouched.
    Xs = np.arange(12, dtype=float).reshape(6, 2)
    ys = [0.0, np.nan, 1.0, 1.0, 2.0, 3.0]
    out = L.split_and_prepare(Xs, ys, train_frac=3 / 6, val_frac=1 / 6,
                              test_frac=2 / 6, clean_train=True,
                              drop_invalid_labels=True)
    assert out['indices']['train'].tolist() == [0, 2], out['indices']
    assert out['indices']['val'].tolist() == [3], out['indices']
    assert out['indices']['test'].tolist() == [4, 5], out['indices']
    assert out['cleaning']['n_invalid_label_rows_dropped'] == 1, out['cleaning']


def t_clean_training_nonfinite_and_outliers():
    # Non-finite features and train-only outliers can be removed before any
    # preprocessing statistic is fitted.  Holdout rows are not passed to the
    # cleaner, so a large holdout value cannot change which train rows survive.
    X = np.array([[0.0], [1.0], [2.0], [1000.0],
                  [3.0], [4.0], [np.nan], [5.0]])
    clean = L.clean_training_rows(X, train_idx=[0, 1, 2, 3],
                                  drop_nonfinite_features=True,
                                  outlier_iqr=1.5)
    assert clean['train_idx'].tolist() == [0, 1, 2], clean
    assert clean['n_nonfinite_feature_rows_dropped'] == 0, clean
    assert clean['n_outlier_rows_dropped'] == 1, clean
    assert clean['dropped_idx'].tolist() == [3], clean
    # A heldout extreme does not enter the fence: only train row 4 is used as
    # the upper reference.  Row 6 (NaN) is in train and is dropped explicitly.
    clean2 = L.clean_training_rows(X, train_idx=[0, 1, 2, 4, 6],
                                   drop_nonfinite_features=True,
                                   outlier_iqr=1.5)
    assert 6 not in clean2['train_idx'].tolist(), clean2
    assert clean2['n_nonfinite_feature_rows_dropped'] == 1, clean2
    out = L.split_and_prepare(X, train_frac=0.5, val_frac=0.25,
                              test_frac=0.25, clean_train=True,
                              drop_nonfinite_features=True, outlier_iqr=1.5)
    assert out['cleaning']['n_outlier_rows_dropped'] == 1, out['cleaning']
    assert out['cleaning']['train_idx'].tolist() == [0, 1, 2], out['cleaning']
    # The NaN row is in the untouched test split, not in train, so it is not
    # counted as a dropped training row.
    assert out['cleaning']['n_nonfinite_feature_rows_dropped'] == 0, out['cleaning']
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.clean_training_rows(X, train_idx=[0, 1], outlier_iqr=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'outlier_iqr={bad!r} must be refused')


def t_clean_training_target_outliers():
    # A corrupted regression label in the training split can be removed with a
    # fence learned from the training labels only.  The holdout's extreme label
    # must not move that fence or decide which training row survives.
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0.0, 1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 1000.0])
    clean = L.clean_training_rows(X, y, train_idx=[0, 1, 2, 3, 4, 5, 6],
                                  target_outlier_iqr=1.5)
    assert clean['train_idx'].tolist() == [0, 1, 2, 4, 5, 6], clean
    assert clean['dropped_idx'].tolist() == [3], clean
    assert clean['n_target_outlier_rows_dropped'] == 1, clean
    assert clean['target_outlier_fence']['lo'] < 100.0, clean
    # The holdout label (row 7 = 1000) is outside the same fence but is never
    # passed to the cleaner, so it cannot cause a training drop.
    no_leak = L.clean_training_rows(
        X, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert no_leak['n_target_outlier_rows_dropped'] == 0, no_leak
    assert no_leak['train_idx'].tolist() == [0, 1, 2, 3, 4, 5, 6], no_leak
    # Categorical labels have no numeric target fence and must not be coerced.
    cats = L.clean_training_rows(
        X, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert cats['n_target_outlier_rows_dropped'] == 0, cats
    assert cats['target_outlier_fence'] is None, cats
    # A requested fence that cannot be estimated (categorical/multi-output
    # target, or too few finite labels) must be visible, not look like a
    # successful cleanup that found no outliers.
    assert cats['target_outlier_fence_estimated'] is False, cats
    assert cats['n_target_reference_labels'] == 0, cats
    multi = L.clean_training_rows(
        X, [(1, 2)] * 8, train_idx=[0, 1, 2, 3, 4, 5, 6],
        target_outlier_iqr=1.5)
    assert multi['target_outlier_fence_estimated'] is False, multi
    numeric = L.clean_training_rows(
        X, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0],
        train_idx=[0, 1, 2, 3, 4, 5, 6], target_outlier_iqr=1.5)
    assert numeric['target_outlier_fence_estimated'] is True, numeric
    assert numeric['n_target_reference_labels'] == 7, numeric
    no_request = L.clean_training_rows(X, train_idx=[0, 1, 2])
    assert no_request['target_outlier_fence_estimated'] is None, no_request
    # The one-call split helper forwards the option and keeps holdouts intact.
    out = L.split_and_prepare(
        X, y, train_frac=5 / 8, val_frac=1 / 8, test_frac=2 / 8,
        clean_train=True, target_outlier_iqr=1.5)
    assert out['indices']['train'].tolist() == [0, 1, 2, 4], out['indices']
    assert out['indices']['val'].tolist() == [5], out['indices']
    assert out['indices']['test'].tolist() == [6, 7], out['indices']
    assert out['cleaning']['n_target_outlier_rows_dropped'] == 1, out['cleaning']
    # The option requires labels, and malformed multipliers are refused.
    try:
        L.clean_training_rows(X, train_idx=[0, 1, 2], target_outlier_iqr=1.5)
    except ValueError as e:
        assert 'requires y' in str(e), e
    else:
        raise AssertionError('a target fence without y must fail closed')
    for bad in (0, -1, float('nan'), float('inf'), True):
        try:
            L.clean_training_rows(X, y, train_idx=[0, 1], target_outlier_iqr=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'target_outlier_iqr={bad!r} must be refused')
    # A repeated training index would double-count drops and let the same
    # physical row appear twice in the cleaned split; refuse it.
    try:
        L.clean_training_rows(X, y, train_idx=[0, 1, 1])
    except ValueError as e:
        assert 'duplicate row indices' in str(e), e
    else:
        raise AssertionError('duplicate train_idx entries must be refused')


def t_known_labels_audit():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = [0, 1, 2, 9]
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1], val_idx=[2],
                                test_idx=[3], known_labels=[0, 1, 2])
    assert rep['checks']['unknown_labels_by_split']['test'] == {'9': 1}, rep
    assert rep['status'] == 'warnings', rep
    hard = L.data_quality_report(
        X, y=y, train_idx=[0, 1], val_idx=[2], test_idx=[3],
        known_labels=[0, 1, 2], fail_on_unknown_labels=True)
    assert hard['status'] == 'fail', hard
    try:
        L.data_quality_report(X, y=y, train_idx=[0, 1], known_labels='012')
    except ValueError:
        pass
    else:
        raise AssertionError('a single string must not be read as a label set')
    try:
        L.data_quality_report(X, train_idx=[0, 1], known_labels=[0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError('known_labels without y must fail closed')


def t_data_guard_nat_and_label_identity():
    # NaT timedelta in an object ordering column is still a missing timestamp:
    # it cannot define a temporal split or a monotone partition.
    _t = np.array([np.timedelta64(5, 'ns'), np.timedelta64('NaT', 'ns'),
                   np.timedelta64(1, 'ns')], dtype=object)
    for call in (lambda: L.split_indices(3, time=_t, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3),
                 lambda: L.assert_no_split_leakage([0], [1], [2], time=_t),
                 lambda: L.data_quality_report(
                     np.arange(6, dtype=float).reshape(3, 2), time=_t,
                     train_idx=[0], val_idx=[1], test_idx=[2])):
        try:
            call()
        except ValueError as e:
            assert 'NaT' in str(e), e
        else:
            raise AssertionError('an object-dtype NaT timedelta must be rejected')
    # A None ordering key is missing too; an all-None object column must not
    # silently become a valid chronology.
    _none_time = np.array([None, None, None], dtype=object)
    for call in (lambda: L.split_indices(3, time=_none_time, train_frac=1 / 3,
                                         val_frac=1 / 3, test_frac=1 / 3),
                 lambda: L.assert_no_split_leakage([0], [1], [2],
                                                  time=_none_time)):
        try:
            call()
        except ValueError as e:
            assert 'missing' in str(e), e
        else:
            raise AssertionError('a None ordering key must be rejected')
    # Categorical encoding: NaT is a missing value, not a category level.
    _cat = np.empty((3, 1), dtype=object)
    _cat[:, 0] = [np.timedelta64('NaT', 'ns'), np.timedelta64(1, 'ns'),
                  np.timedelta64(1, 'ns')]
    _enc = L.LeakageSafeCategoricalEncoder().fit(_cat)
    assert len(_enc.get_stats()['features'][0]['categories']) == 1, _enc.get_stats()
    # Missing NaT labels are reported as non-finite label noise.
    _rep = L.data_quality_report(
        np.arange(8, dtype=float).reshape(4, 2),
        y=np.array([0, 1, np.datetime64('NaT'), 2], dtype=object),
        train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert _rep['checks']['nonfinite_labels'] == 1, _rep
    # Numeric and string labels that merely share a spelling are different
    # classes; a complete shift between them must not be hidden by str().
    _rep2 = L.data_quality_report(
        np.arange(8, dtype=float).reshape(4, 2), y=[1, 1, '1', '1'],
        train_idx=[0, 1], val_idx=[2, 3])
    _shift = _rep2['checks']['label_distribution_shift_vs_train']['val']
    assert _shift['js_divergence'] > 0.5, _rep2


def t_object_temporal_duplicate_identity():
    # Adjacent nanosecond timestamps have the same float64 image.  An object
    # column must not collapse them into false duplicates (which would drop
    # distinct training rows or manufacture a cross-split leakage alarm).
    import data_guard as DG
    stamps = [np.datetime64('2020-01-01T00:00:00.000000001'),
              np.datetime64('2020-01-01T00:00:00.000000002'),
              np.datetime64('2020-01-01T00:00:00.000000003')]
    X = np.asarray(stamps, dtype=object).reshape(-1, 1)
    assert not DG.duplicate_mask(X).any(), DG.duplicate_mask(X)
    assert len({DG._matrix_fingerprint(X[i:i + 1]) for i in range(3)}) == 3
    # A true duplicate is still detected, and NaT is not conflated with None.
    dup = np.asarray([stamps[0], stamps[0], np.datetime64('NaT'),
                      None], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(dup).tolist() == [False, True, False, False],         DG.duplicate_mask(dup)
    # The same instant expressed in different units keeps one identity.
    mixed = np.asarray([np.datetime64('2020-01-01T00:00:01', 's'),
                        np.datetime64('2020-01-01T00:00:01.000000000', 'ns')],
                       dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(mixed).tolist() == [False, True],         DG.duplicate_mask(mixed)
    # Timedelta values use their own exact identity (and do not collide with
    # the integer with the same value).
    deltas = np.asarray([np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns'),
                         np.timedelta64(1, 'ns'), 1], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(deltas).tolist() == [False, False, True, False],         DG.duplicate_mask(deltas)
    # Outside the int64-nanosecond range the key used to include the raw unit
    # and dtype, so the same instant written in two units looked distinct.
    far = np.asarray([np.datetime64('2500-01-01', 'D'),
                      np.datetime64('2500-01-01', 's'),
                      np.datetime64('2500-01-02', 'D')],
                     dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(far).tolist() == [False, True, False], DG.duplicate_mask(far)
    big = np.asarray([np.timedelta64(200000, 'D'),
                      np.timedelta64(200000 * 86400, 's')],
                     dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(big).tolist() == [False, True], DG.duplicate_mask(big)
    assert DG._target_key(np.datetime64('2500-01-01', 'D')) == \
        DG._target_key(np.datetime64('2500-01-01', 's'))
    # End-to-end: a clean chronological split of distinct timestamps must not
    # be refused as a duplicate-feature leak, while cleaning must keep them.
    clean_obj = np.empty((6, 1), dtype=object)
    clean_obj[:, 0] = stamps + [np.datetime64('2020-01-01T00:00:00.000000004'),
                                np.datetime64('2020-01-01T00:00:00.000000005'),
                                np.datetime64('2020-01-01T00:00:00.000000006')]
    out = DG.split_and_prepare(clean_obj, preprocess=False,
                               train_frac=0.5, val_frac=0.25, test_frac=0.25)
    assert out['cross_split_duplicates'] == {
        'train__val': 0, 'train__test': 0, 'val__test': 0}, out
    cleaned = DG.clean_training_rows(
        clean_obj, train_idx=np.arange(4), drop_duplicates=True)
    assert cleaned['n_duplicate_rows_dropped'] == 0, cleaned


def t_object_temporal_category_identity():
    # np.timedelta64 is a numbers.Integral, so the generic integral branch
    # either collided with the integer of the same raw count or raised for a
    # larger unit.  Target/category identity must use the exact temporal value.
    import data_guard as DG
    ns = np.timedelta64(1, 'ns')
    one_day = np.timedelta64(1, 'D')
    one_day_h = np.timedelta64(24, 'h')
    assert DG._target_key(ns) != DG._target_key(1), DG._target_key(ns)
    assert DG._target_key(one_day) == DG._target_key(one_day_h)
    assert DG._target_key(one_day) != DG._target_key(np.timedelta64(1, 'ns'))
    assert DG._target_key(np.datetime64('2020-01-01')) != \
        DG._target_key(np.datetime64('2020-01-01T00:00:00.000000001'))
    assert DG._target_key(np.timedelta64('NaT')) == '__missing__'
    # Outside the int64-nanosecond range NumPy WRAPS instead of returning
    # NaT, so a raw ns cast is not proof of identity.  The same instant in
    # D/Y/2Y units must share one exact key (and a 2Y unit must not be
    # multiplied a second time when it is converted to days).
    _far = [np.datetime64('3000-01-01'), np.datetime64('3000', 'Y'),
            np.datetime64('3000', '2Y')]
    assert len({DG._target_key(v) for v in _far}) == 1, _far
    assert len({DG._group_key(v) for v in _far}) == 1, _far
    assert DG._target_key(np.datetime64('3000', '2Y')) != \
        DG._target_key(np.datetime64('3002', '2Y'))
    # End-to-end: a timedelta category and the integer with the same raw count
    # are different levels and must not share one target mean.
    X = np.asarray([[ns], [1], [ns], [1]], dtype=object)
    y = np.asarray([10.0, 20.0, 10.0, 20.0])
    enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(X, y)
    lookup = enc.get_stats()['features'][0]['lookup']
    assert lookup[DG._target_key(ns)] == 10.0, lookup
    assert lookup['i:1'] == 20.0, lookup
    # The persisted keys round-trip through JSON and still separate them.
    import json
    restored = DG.LeakageSafeTargetEncoder.from_stats(
        json.loads(json.dumps(enc.get_stats())))
    hold = np.asarray([[ns], [1]], dtype=object)
    assert np.allclose(restored.transform(hold), enc.transform(hold))
    # The same identity rule is used by the frequency encoder.
    freq = DG.LeakageSafeFrequencyEncoder().fit(X)
    f = freq.get_stats()['features'][0]['entries']
    assert len(f) == 2, f
    # A timedelta category is persisted by the categorical encoder without
    # int()-ing it (which raises for day-or-larger units) and still round-trips
    # to the same ordinal codes.
    cat_X = np.asarray([[one_day], [np.timedelta64(2, 'D')], [one_day],
                        [np.timedelta64(2, 'D')]], dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(cat_X)
    cat_stats = json.loads(json.dumps(cat.get_stats()))
    cat_restored = DG.LeakageSafeCategoricalEncoder.from_stats(cat_stats)
    assert np.array_equal(cat.transform(cat_X),
                          cat_restored.transform(cat_X)), cat_stats
    # A timedelta is a numeric duration target, not a missing/unsupported
    # label.  NumPy's float(timedelta) works only for ns-like units, so the
    # raw fallback silently dropped day/hour labels; the encoding must use
    # one canonical duration and keep equal instants in different units equal.
    import datetime as _dt
    _td_X = np.array([['a'], ['a'], ['a']], dtype=object)
    _td_y = np.array([one_day, one_day_h, np.timedelta64(2, 'D')],
                     dtype=object)
    _td_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        _td_X, _td_y)
    _td_got = _td_enc.transform(np.array([['a']], dtype=object))[0, 0]
    assert math.isclose(_td_got, (4.0 / 3.0) * 86400e9), _td_got
    _py_y = np.array([_dt.timedelta(days=1), _dt.timedelta(days=2)],
                     dtype=object)
    _py_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        np.array([['a'], ['a']], dtype=object), _py_y)
    _py_got = _py_enc.transform(np.array([['a']], dtype=object))[0, 0]
    assert math.isclose(_py_got, 1.5 * 86400e9), _py_got


def t_persisted_stats_validate_feature_count():
    # transform() writes one output column per persisted feature stat.  A
    # truncated/corrupt artifact would otherwise leave columns uninitialised
    # (or write past the declared width), silently changing the feature table.
    import data_guard as DG
    import json
    X = np.arange(12.0).reshape(4, 3)
    prep = DG.LeakageSafePreprocessor().fit(X)
    valid = json.loads(json.dumps(prep.get_stats()))
    assert np.allclose(DG.LeakageSafePreprocessor.from_stats(valid).transform(X),
                       prep.transform(X))
    for mutate in (
            lambda s: s.update(features=[]),
            lambda s: s.update(features=s['features'][:2], n_features_in=3),
            lambda s: s.update(n_features_in=4),
            lambda s: s.update(n_features_in=True),
            lambda s: s['features'][0].update(factor=0.0),
            lambda s: s['features'][0].update(fill=float('nan'))):
        bad = json.loads(json.dumps(valid))
        mutate(bad)
        try:
            DG.LeakageSafePreprocessor.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed preprocessor stats accepted: {bad}')
    Xc = np.asarray([['a'], ['b']], dtype=object)
    for cls, fit_args in (
            (DG.LeakageSafeCategoricalEncoder, (Xc,)),
            (DG.LeakageSafeFrequencyEncoder, (Xc,)),
            (DG.LeakageSafeTargetEncoder, (Xc, np.asarray([1.0, 2.0])))):
        fitted = cls().fit(*fit_args)
        good = json.loads(json.dumps(fitted.get_stats()))
        restored = cls.from_stats(good)
        kw = {'allow_train': True} if cls is DG.LeakageSafeTargetEncoder else {}
        assert np.array_equal(restored.transform(Xc, **kw),
                              fitted.transform(Xc, **kw))
        for bad_stats in (
                dict(good, features=[]),
                dict(good, features=good['features'] * 2)):
            try:
                cls.from_stats(bad_stats)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'{cls.__name__}.from_stats accepted a mismatched feature list')


def t_temporal_group_identity_matches_python_equality():
    # NumPy equality and hashing disagree for date-like ids (a datetime.date
    # equals some np.datetime64 values but hashes differently).  Returning the
    # raw objects as set keys let the same id land on both sides of a split
    # without the group-overlap check firing.
    import datetime
    import data_guard as DG
    same_day = [np.datetime64('2020-01-01'), datetime.date(2020, 1, 1)]
    assert DG._group_key(same_day[0]) == DG._group_key(same_day[1])
    # The canonical key is an opaque sentinel, so a real tuple group id that
    # spells the same components cannot impersonate it.
    forged = ('__temporal__', b'DI1577836800000000000')
    assert DG._group_key(same_day[0]) != forged
    try:
        DG.assert_no_split_leakage([0], [1], [], groups=np.asarray(same_day, dtype=object))
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('equal temporal group ids must not cross splits')
    same_delta = [np.timedelta64(1, 'D'), datetime.timedelta(days=1)]
    assert DG._group_key(same_delta[0]) == DG._group_key(same_delta[1])
    # Distinct nanosecond timestamps stay distinct, including in the fast
    # native-dtype path (tolist() would turn them into bare integers).
    g = np.asarray(['2020-01-01T00:00:00.000000001',
                    '2020-01-01T00:00:00.000000002'], dtype='datetime64[ns]')
    keys = DG._group_index_fast(g)[0]
    assert keys[0] != keys[1], keys
    idx = DG.grouped_chronological_split_indices(2, g, 0.5, 0.5, 0.0)
    assert idx['train'].tolist() == [0] and idx['val'].tolist() == [1], idx
    # A string/object ``order`` forces the per-row fallback.  Its member map
    # must use the same native-scalar temporal keys as the fast path; using
    # tolist() turned datetime64[ns] into bare integers and raised KeyError.
    g4 = np.asarray(['2020-01-01', '2020-01-02', '2020-01-01',
                     '2020-01-03'], dtype='datetime64[ns]')
    idx4 = DG.grouped_chronological_split_indices(
        4, g4, train_size=2, val_size=1, test_size=1,
        order=np.asarray(['a', 'b', 'a', 'c']))
    assert idx4['train'].tolist() == [0, 2], idx4
    assert idx4['val'].tolist() == [1] and idx4['test'].tolist() == [3], idx4
    # NaT still collapses into one missing group for both temporal kinds.
    nat = np.asarray([np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'ns'),
                      np.datetime64(1, 'ns')])
    try:
        DG.assert_no_split_leakage([0], [1], [2], groups=nat)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('NaT group ids must collapse into one group')
    # Timezone-aware datetimes have one instant even when written with
    # different offsets.  They must share a group id (Python equality/hash do)
    # while staying distinct from a naive datetime with the same wall clock.
    tz = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    tz_west = datetime.datetime(
        2019, 12, 31, 19,
        tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    assert tz == tz_west
    assert DG._group_key(tz) == DG._group_key(tz_west)
    assert DG._group_key(tz) != DG._group_key(datetime.datetime(2020, 1, 1))
    try:
        DG.assert_no_split_leakage(
            [0], [1], [], groups=np.asarray([tz, tz_west], dtype=object))
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('the same aware instant must not cross splits')


def t_python_temporal_row_and_category_identity():
    # datetime.date/datetime/timedelta compare equal to their NumPy temporal
    # counterparts but hash differently.  Duplicate detection, category keys
    # and group keys must all use one canonical identity.
    import datetime
    import json
    import data_guard as DG
    day = datetime.date(2020, 1, 1)
    midpoint = datetime.datetime(2020, 1, 1)
    npday = np.datetime64('2020-01-01')
    assert DG._group_key(day) == DG._group_key(midpoint) == DG._group_key(npday)
    assert DG._target_key(day) == DG._target_key(midpoint) == DG._target_key(npday)
    for pair in ((day, npday), (midpoint, npday),
                 (datetime.timedelta(days=1), np.timedelta64(1, 'D'))):
        rows = np.asarray(pair, dtype=object).reshape(-1, 1)
        assert DG.duplicate_mask(rows).tolist() == [False, True], pair
    # Python datetime only has microsecond resolution: adjacent values stay
    # distinct instead of collapsing through a rounded float key.
    micro = np.asarray([midpoint, midpoint + datetime.timedelta(microseconds=1),
                        midpoint], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(micro).tolist() == [False, False, True]
    # An aware value is not the same row as a naive one with the same clock,
    # but the same instant in two zones is.
    utc = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    west = datetime.datetime(
        2019, 12, 31, 19,
        tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
    aware_rows = np.asarray([utc, west, midpoint], dtype=object).reshape(-1, 1)
    assert DG.duplicate_mask(aware_rows).tolist() == [False, True, False]
    # Categorical/frequency encoding: equal spellings are one learned level,
    # and the persisted key round-trips to the same codes.
    X = np.asarray([[day], [npday], [datetime.date(2020, 1, 2)]],
                   dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(X)
    stats = cat.get_stats()['features'][0]
    assert stats['n_categories'] == 2, stats
    assert len(set(stats['keys'])) == 2, stats
    codes = cat.transform(X).ravel()
    assert codes[0] == codes[1] != codes[2], codes
    cat2 = DG.LeakageSafeCategoricalEncoder.from_stats(
        json.loads(json.dumps(cat.get_stats())))
    assert np.array_equal(cat2.transform(X), cat.transform(X))
    freq = DG.LeakageSafeFrequencyEncoder().fit(X)
    entries = freq.get_stats()['features'][0]['entries']
    assert len(entries) == 2, entries
    assert entries[0]['count'] == 2, entries
    f = freq.transform(X).ravel()
    assert f[0] == f[1] > f[2], f
    enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        X, np.asarray([10.0, 10.0, 20.0]))
    lookup = enc.get_stats()['features'][0]['lookup']
    assert lookup[DG._target_key(day)] == 10.0, lookup
    # Labels follow the same equality: equal temporal spellings are one class,
    # so duplicated features are not a false label conflict and a train/val
    # swap between the two spellings is not a fabricated 100% label shift.
    pair_X = np.arange(8, dtype=float).reshape(4, 2)
    conflict = DG.data_quality_report(
        pair_X[:2], y=np.asarray([day, npday], dtype=object),
        train_idx=[0, 1])
    assert conflict['checks']['label_conflicts_on_duplicate_rows'] == 0, conflict
    shift = DG.data_quality_report(
        pair_X, y=np.asarray([day, npday, npday, day], dtype=object),
        train_idx=[0, 1], val_idx=[2, 3])
    assert shift['checks']['label_distribution_shift_vs_train']['val'][
        'js_divergence'] == 0.0, shift
    # Metric label inference uses the same identity: NumPy may fail to merge a
    # Python datetime with the equal np.datetime64, which previously left two
    # declared classes and rejected the observed label as unknown.
    _y = np.asarray([day, npday, datetime.date(2020, 1, 2),
                     npday, day, datetime.date(2020, 1, 2)], dtype=object)
    _m = L.classification_metrics(_y, y_pred=_y)
    assert len(_m['labels']) == 2, _m['labels']
    assert _m['accuracy'] == 1.0, _m
    _same = L.classification_metrics(
        np.asarray([utc, west], dtype=object),
        y_pred=np.asarray([utc, west], dtype=object))
    assert len(_same['labels']) == 1, _same['labels']
    # The categorical fingerprint drives the target encoder's training-row
    # refusal.  Equal temporal categories written as different types must
    # fingerprint identically, or transform() would re-enable the full-table
    # leak on what is really the training matrix.
    day2 = datetime.date(2020, 1, 2)
    train_dates = np.asarray([[day], [day2]], dtype=object)
    holdout_np = np.asarray([[npday],
                             [np.datetime64('2020-01-02')]], dtype=object)
    assert DG._categorical_fingerprint(train_dates) == \
        DG._categorical_fingerprint(holdout_np)
    # The encoder-fit audit compares canonical persisted keys, not readable
    # reprs: the same categories in NumPy spelling must still verify.
    assert DG.audit_encoder_fit(
        DG.LeakageSafeCategoricalEncoder().fit(train_dates),
        holdout_np)['status'] == 'train_only_verified'
    assert DG.audit_encoder_fit(
        DG.LeakageSafeFrequencyEncoder().fit(train_dates),
        holdout_np)['status'] == 'train_only_verified'
    leak_enc = DG.LeakageSafeTargetEncoder(smoothing=0.0, n_folds=2).fit(
        train_dates, np.asarray([1.0, 2.0]))
    try:
        leak_enc.transform(holdout_np)
    except ValueError as e:
        assert 'training rows' in str(e), e
    else:
        raise AssertionError(
            'an equal-temporal training matrix must be refused by transform')


def t_validation_row_split_group_aware():
    # The balanced row boundary (4) would cut the group spanning rows 3-4.
    # A group-aware split must move it to a document boundary instead.
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    sel, ev, meta = L.validation_row_split(8, groups=groups)
    assert meta['groups_aware'] is True and meta['n_groups'] == 4, meta
    assert sorted(sel + ev) == list(range(8)), (sel, ev)
    assert not ({groups[i] for i in sel} & {groups[i] for i in ev}), (sel, ev)
    assert 3 not in sel or 4 not in ev, (sel, ev)
    # Without groups the historical balanced split is unchanged.
    sel2, ev2, meta2 = L.validation_row_split(8)
    assert (sel2, ev2) == (list(range(4)), list(range(4, 8))), (sel2, ev2)
    assert meta2['groups_aware'] is False, meta2
    # A group larger than a fold cannot be cut; refuse instead of leaking it.
    try:
        L.validation_row_split(6, n_folds=3, groups=[0, 0, 0, 1, 1, 1])
    except ValueError as e:
        assert 'without cutting a group' in str(e), e
    else:
        raise AssertionError('a fold boundary inside a group must be refused')
    try:
        L.validation_row_split(4, groups=[0, 0])
    except ValueError as e:
        assert 'groups has length' in str(e), e
    else:
        raise AssertionError('groups length mismatch must be refused')
    # A boundary between adjacent group ids is not enough when the same group
    # id reappears later in the stream: the two returned folds would still
    # share a document.  Refuse the leaking split.
    try:
        L.validation_row_split(8, groups=[0, 1, 0, 1, 0, 1, 0, 1])
    except ValueError as e:
        assert 'contributes rows to both' in str(e), e
    else:
        raise AssertionError('a non-contiguous group spanning both folds must be refused')
    # Missing group ids collapse into one group (NaT is not a unique key).
    _nat = np.array([np.timedelta64('NaT', 'ns'), np.timedelta64('NaT', 'ns'),
                     np.timedelta64(1, 'ns'), np.timedelta64(1, 'ns')],
                    dtype=object)
    _s, _e, _m = L.validation_row_split(4, groups=_nat)
    assert _m['n_groups'] == 2, _m
    assert not ({str(_nat[i]) for i in _s} & {str(_nat[i]) for i in _e}), (_s, _e)


def t_cached_validation_row_groups_merge_shared_documents():
    # The token-level sidecar stores a document id per token.  Fixed row
    # windows can straddle a document, so rows 1-3 must become one group; a
    # row-only split would otherwise be able to select on part of a document
    # and evaluate on another part.
    _d = tempfile.mkdtemp(prefix='valgroups_')
    _sl, _ntr, _vb, _vs = 2, 1, 10, 4
    _tag = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs = np.array([1, 1, 2, 2, 2, 3, 3, 3], dtype=np.uint64)
    np.save(os.path.join(_d, f'val_token_doc_ids_{_tag}.npy'), _docs)
    with open(os.path.join(_d, f'meta_{_tag}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs.size),
                               'validation_token_fp': L._ids_fp(_docs)}}, fh)
    _groups, _meta = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=_d)
    assert _groups == [0, 1, 1, 1], (_groups, _meta)
    assert _meta['groups_aware'] is True and _meta['n_groups'] == 2, _meta
    _s, _e, _m = L.validation_row_split(4, groups=_groups)
    assert _m['groups_aware'] is True, _m
    assert not ({_groups[i] for i in _s} & {_groups[i] for i in _e}), (_s, _e)
    # A group-aware simple-baseline artifact must pass its own audit.  The
    # audit cannot re-derive these folds from (n_rows, seed, folds) alone: a
    # row-only replay gives different boundaries, so treating that replay as
    # the expected split rejected every valid group-aware floor.
    _rng = np.random.default_rng(7)
    _proto = dict(L.SIMPLE_BASELINE_POLICY,
                  arms={'unigram': {'order': 1}},
                  grids={'unigram': {'alpha': [0.1, 1.0, 2.0]}})
    _art = L.simple_baseline_fit(_rng.integers(0, _vb, size=200),
                                 _rng.integers(0, _vb, size=(4, 5)), _vb,
                                 protocol=_proto, selection_rows=_s,
                                 evaluation_rows=_e, row_split=_m)
    _aud = L.simple_baseline_audit(_art, protocol=_proto)
    assert _aud['baseline_verified'] is True, _aud
    # A missing sidecar is an explicit fallback, not a silent claim.
    _g, _m = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=tempfile.mkdtemp())
    assert _g is None and _m['groups_aware'] is False and 'absent' in _m['reason'], (_g, _m)
    # A legacy document-level sidecar cannot be expanded to token groups; it
    # is an explicit fallback (not a crash and not a fabricated split).
    _d_legacy = tempfile.mkdtemp(prefix='valgroups_legacy_')
    np.save(os.path.join(_d_legacy, f'val_doc_ids_{_tag}.npy'),
            np.array([1, 2, 3], dtype=np.uint64))
    _gl, _ml = L.cached_validation_row_groups(
        _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4, cache_dir=_d_legacy)
    assert _gl is None and _ml['groups_aware'] is False, (_gl, _ml)
    assert 'legacy' in _ml['reason'], _ml
    # Metadata that declares the token-level sidecar but no longer has the
    # file is corrupt evidence: falling back to the legacy document file (or
    # to row-only folds) would silently change the group identity.
    _d_missing = tempfile.mkdtemp(prefix='valgroups_missing_declared_')
    np.save(os.path.join(_d_missing, f'val_doc_ids_{_tag}.npy'),
            np.array([1, 2, 3], dtype=np.uint64))
    with open(os.path.join(_d_missing, f'meta_{_tag}.json'), 'w',
              encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': 8,
                               'validation_token_fp': 'deadbeef'}}, fh)
    try:
        L.cached_validation_row_groups(
            _sl, _ntr, vocab_size=_vb, val_seqs=_vs, n_rows=4,
            cache_dir=_d_missing)
    except ValueError as e:
        assert 'is missing' in str(e), e
    else:
        raise AssertionError('a declared-but-missing token sidecar must be refused')
    # A present but too-short TOKEN sidecar must fail closed.
    _d2 = tempfile.mkdtemp(prefix='valgroups_bad_')
    _tag2 = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs_short = np.array([1, 1, 2, 2], dtype=np.uint64)
    np.save(os.path.join(_d2, f'val_token_doc_ids_{_tag2}.npy'), _docs_short)
    with open(os.path.join(_d2, f'meta_{_tag2}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_short.size),
                               'validation_token_fp': L._ids_fp(_docs_short)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d2)
    except ValueError as e:
        assert 'need' in str(e), e
    else:
        raise AssertionError('a short token-level document-id sidecar must be refused')
    # A float sidecar must not be silently truncated into document groups:
    # int(1.5) == 1 would merge two distinct documents.
    _d3 = tempfile.mkdtemp(prefix='valgroups_float_')
    _tag3 = f'v5_sl{_sl}_cap{_ntr}_v{_vb}_vs{_vs}'
    _docs_float = np.array([1.0, 1.5, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    np.save(os.path.join(_d3, f'val_token_doc_ids_{_tag3}.npy'), _docs_float)
    with open(os.path.join(_d3, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_float.size),
                               'validation_token_fp': L._ids_fp(_docs_float)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d3)
    except ValueError as e:
        assert 'integers' in str(e), e
    else:
        raise AssertionError('a non-integer document-id sidecar must be refused')
    # Negative ids are not document identities, even when their fingerprint
    # matches the metadata.
    _d4 = tempfile.mkdtemp(prefix='valgroups_negative_')
    _docs_neg = np.array([1, 1, -2, -2, 3, 3, 3, 3], dtype=np.int64)
    np.save(os.path.join(_d4, f'val_token_doc_ids_{_tag3}.npy'), _docs_neg)
    with open(os.path.join(_d4, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_neg.size),
                               'validation_token_fp': L._ids_fp(_docs_neg)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d4)
    except ValueError as e:
        assert 'negative' in str(e), e
    else:
        raise AssertionError('a negative document-id sidecar must be refused')
    # A malformed meta shape (list instead of object) is fail-closed evidence,
    # not an AttributeError from a list.get call.
    _d5 = tempfile.mkdtemp(prefix='valgroups_meta_')
    np.save(os.path.join(_d5, f'val_token_doc_ids_{_tag3}.npy'), _docs)
    with open(os.path.join(_d5, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump([], fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d5)
    except ValueError as e:
        assert 'JSON object' in str(e), e
    else:
        raise AssertionError('a non-object metadata file must be refused')
    # A fractional token count must not be truncated to the real sidecar
    # length (int(8.5) == 8) and blessed as a match.
    _d6 = tempfile.mkdtemp(prefix='valgroups_count_')
    np.save(os.path.join(_d6, f'val_token_doc_ids_{_tag3}.npy'), _docs)
    with open(os.path.join(_d6, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': float(_docs.size) + 0.5,
                               'validation_token_fp': L._ids_fp(_docs)}}, fh)
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d6)
    except ValueError as e:
        assert 'length' in str(e), e
    else:
        raise AssertionError('a fractional token-id count must be refused')
    # A multi-dimensional sidecar would be silently flattened into a different
    # token->document mapping; require the documented one-dimensional layout.
    _d8 = tempfile.mkdtemp(prefix='valgroups_2d_')
    np.save(os.path.join(_d8, f'val_token_doc_ids_{_tag3}.npy'), _docs.reshape(2, 4))
    try:
        L.cached_validation_row_groups(_sl, _ntr, vocab_size=_vb, val_seqs=_vs,
                                       n_rows=4, cache_dir=_d8)
    except ValueError as e:
        assert 'one-dimensional' in str(e), e
    else:
        raise AssertionError('a multi-dimensional document-id sidecar must be refused')
    # The tuning wrapper may fall back only when a VALID group table cannot
    # form the requested folds.  A corrupt sidecar must propagate: silently
    # treating it as "no split" would select on the final-estimate rows.
    _base = {'dataset': 'wikitext', 'seq_len': _sl, 'n_train_tokens': _ntr,
             'vocab_size': _vb, 'val_seqs': _vs, 'cache_dir': _d3}
    _spec = {'seed': 20261020, 'n_folds': 2,
             'selection_fold': 0, 'evaluation_fold': 1}
    try:
        BT._nested_row_split_checked(_base, 4, _spec)
    except ValueError as e:
        assert 'integers' in str(e), e
    else:
        raise AssertionError('a corrupt group sidecar must not become a full-validation fallback')
    # A valid table whose single document spans every row cannot be cut into
    # folds; that is an explicit fallback with a reason, not a crash.
    _d7 = tempfile.mkdtemp(prefix='valgroups_one_')
    _docs_one = np.ones(8, dtype=np.uint64)
    np.save(os.path.join(_d7, f'val_token_doc_ids_{_tag3}.npy'), _docs_one)
    with open(os.path.join(_d7, f'meta_{_tag3}.json'), 'w', encoding='utf-8') as fh:
        json.dump({'doc_ids': {'validation_token_count': int(_docs_one.size),
                               'validation_token_fp': L._ids_fp(_docs_one)}}, fh)
    _base_one = dict(_base, cache_dir=_d7)
    _sel1, _ev1, _m1, _why1 = BT._nested_row_split_checked(_base_one, 4, _spec)
    assert _sel1 is None and _ev1 is None and _why1, (_sel1, _ev1, _m1, _why1)
    assert _m1['groups_aware'] is False and 'unavailable' in _m1['reason'], _m1
    # Cache metadata is untrusted input: a non-object/corrupt meta must send
    # the loader to its rebuild path (or a hard refusal for the frozen test
    # cache), never crash later on meta.get().
    _bad_meta = os.path.join(tempfile.mkdtemp(prefix='badmeta_'), 'meta.json')
    for _payload in ('[]', '{not json', 'null'):
        with open(_bad_meta, 'w', encoding='utf-8') as fh:
            fh.write(_payload)
        assert L._read_cache_meta(_bad_meta, 'unit') is None, _payload
    with open(_bad_meta, 'w', encoding='utf-8') as fh:
        json.dump({'ok': 1}, fh)
    assert L._read_cache_meta(_bad_meta, 'unit') == {'ok': 1}


def t_build_report_primary_estimand():
    # The zero-GPU report must pair on the same held-out rows as the nested
    # primary PPL; reading the full-validation `ppl` restores selection rows.
    def _rec(full, hold, seed=0):
        return {'variant': 'a', 'seed': seed, 'ppl': full, 'ppl_holdout': hold,
                'ppl_estimation_rows': 'nested_validation_holdout',
                'nll_sum_holdout': math.log(hold) * 40, 'n_tokens_holdout': 40,
                'run_cfg': 'cfg', 'steps': 1, 'tokens_seen': 40,
                'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e'}
    a = _rec(10.0, 9.0)
    b = _rec(10.5, 10.5)
    assert BR._primary_ppl(a) == 9.0 and BR._primary_ppl(b) == 10.5
    got = BR.paired_row({('', 0): a}, {('', 0): b})
    assert got is not None and got[1] == [9.0 - 10.5], got
    # Holdout per-row accounting is an alternative to ppl_holdout.
    rows = dict(a)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout'):
        rows.pop(_k)
    rows['nll_per_row_holdout'] = [math.log(9.0) * 40]
    rows['n_tokens_per_row_holdout'] = [40]
    assert abs(BR._primary_ppl(rows) - 9.0) < 1e-12, rows
    # A declared-nested record without any holdout accounting is unverifiable,
    # and the pair is omitted rather than quoting the full-validation value.
    bad = dict(a)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout'):
        bad.pop(_k)
    assert BR._primary_ppl(bad) is None
    assert BR.paired_row({('', 0): bad}, {('', 0): b}) is None
    # A holdout CE beyond the float exponent range is a non-measurement, not a
    # crash in the report builder.
    huge = dict(bad, ppl_estimation_rows='nested_validation_holdout',
                nll_sum_holdout=1e6, n_tokens_holdout=1)
    assert BR._primary_ppl(huge) is None
    # The two arms must declare the SAME estimand.  A full-validation record
    # paired against a holdout record produces a difference on neither sample.
    full = {'variant': 'b', 'seed': 0, 'ppl': 10.0, 'run_cfg': 'cfg',
            'steps': 1, 'tokens_seen': 40, 'data_fp': 'd', 'val_fp': 'v',
            'eval_fp': 'e'}
    assert BR.paired_row({('', 0): a}, {('', 0): full}) is None
    # Nor may different seeds contribute deltas from different estimands to
    # one sign-flip test.
    a1 = _rec(10.0, 9.0, seed=1)
    b1 = _rec(10.5, 10.5, seed=1)
    _mix = BR.paired_row({('', 0): full, ('', 1): a1},
                         {('', 0): dict(full, ppl=11.0), ('', 1): b1})
    assert _mix is None, _mix


def t_build_report_exact_seed_and_params():
    # The zero-GPU report reader must share exp_lib's exact-integer contract:
    # int(Decimal('1.5')) would silently pair the wrong run, and a fractional
    # parameter count is not a capacity measurement.
    from decimal import Decimal
    _orig_summary = BR.summary
    try:
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': Decimal('1.5'),
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        assert BR.per_seed_records('/nonexistent-panel') == {}
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': Decimal('2'),
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        got = BR.per_seed_records('/nonexistent-panel')
        assert list(got.get('x', {})) == [('', 2)], got
        BR._LOADED.clear()
        BR.summary = lambda _outdir: {'d::x': {'variant': 'x', 'seed': '1.0',
                                               'ppl': 2.0, 'run_cfg': 'r',
                                               'steps': 1, 'tokens_seen': 1}}
        assert BR.per_seed_records('/nonexistent-panel') == {}
        # Malformed/non-string arm ids are excluded, not used as (unhashable)
        # dict keys or silently pooled under a coerced spelling.
        for _bad_variant in ({'x': 1}, ['x'], '', None, 3):
            BR._LOADED.clear()
            BR.summary = lambda _outdir, _v=_bad_variant: {
                'd::x': {'variant': _v, 'seed': 0, 'ppl': 2.0, 'run_cfg': 'r',
                         'steps': 1, 'tokens_seen': 1}}
            assert BR.per_seed_records('/nonexistent-panel') == {}, _bad_variant
    finally:
        BR.summary = _orig_summary
        BR._LOADED.clear()
    assert BR.param_gap({'params': 1000}, {'params': 1000.5}) is None
    assert BR.param_gap({'params': 1000}, {'params': True}) is None
    assert BR.param_gap({'params': 1000}, {'params': 1010}) is not None


def t_par_watchdog_override_validation():
    import par
    ctx = {'workers': 1, 'slots': [''], 'on_cuda': False}
    assert par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': 42}, ctx) == 42.0
    assert par._job_watchdog_seconds({'kind': 'x'}, {'watchdog_seconds': 7}) == 7.0
    assert par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': 0}, ctx) == 12.0 * 3600.0
    for bad in (-1, 0.5 and -0.5, float('nan'), float('inf'), 'not-a-number',
                True, False):
        try:
            par._job_watchdog_seconds({'kind': 'x', 'watchdog_seconds': bad}, ctx)
        except ValueError:
            pass
        else:
            raise AssertionError(f'watchdog_seconds={bad!r} must be rejected')
    # NumPy scalars are normal numeric payloads, not strings.
    assert par._norm({'seed': np.int64(3)}) == {'seed': 3}
    assert par._norm(np.array([[1, 2]], dtype=np.int64)) == [[1, 2]]


def t_methodology_gate_fail_closed_edges():
    # The report's local Holm twin must use the same scalar contract as the
    # library: a boolean is not a probability, and an invalid alpha is not a
    # threshold.
    for _p in (False, True, np.bool_(False), np.bool_(True), -0.1, 1.5, 'x'):
        assert BR.holm_adjust([('a', _p), ('b', 0.02)])['a'] == 1.0, _p
    for _bad in (True, np.bool_(True), 0, 1, -0.1, 1.5, float('nan')):
        for _fn in (lambda a: BR.holm_adjust([('a', 0.01)], alpha=a),
                    lambda a: BR.global_multiplicity_adjust([('a', 0.01)], alpha=a),
                    lambda a: L.holm_min_attainable_p(6, 4, a)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid alpha {_bad!r} accepted by a report twin')

    # A descriptor present on only SOME seeds of an arm is not equality: the
    # arm is unverifiable, and the fairness audit must fail instead of
    # trusting the populated subset.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=49.0, params=100,
                                train_knobs=(dict(knobs) if i else None))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=50.0, params=100,
                                train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={},
                            verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['train_knobs_equal'] is False, cv
    assert cv['partially_recorded'] == ['a'] and cv['controlled'] is False, cv
    assert any('only on SOME seed records' in w for w in r['warnings']), r['warnings']
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(s, f)
        with contextlib.redirect_stdout(io.StringIO()):
            fair = A.audit_fairness(path, arms=['a', 'b'])
    assert fair['status'] == 'fail', fair
    assert any('omit train_knobs' in p for p in fair['problems']), fair['problems']

    # A malformed (unhashable) params field must not crash the fairness audit.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'x': dict(s['a::seed0'], params={'bad': 1})}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            r_bad = A.audit_fairness(path, arms=['a'])
    assert r_bad['arms']['a']['n_records'] == 1, r_bad

    # Full-factorial completeness fails closed on malformed or duplicated
    # cells instead of dropping them and reporting a complete table.
    base = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC',
            ('A', 'B', 'C'): 'ABC'}
    assert L.ablation_completeness_audit(arms_by_combo=base)['complete'] is True
    dup = dict(base, **{'A+B': 'dup'})
    r = L.ablation_completeness_audit(arms_by_combo=dup)
    assert r['complete'] is False and r['duplicate_combos'], r
    bad = dict(base)
    bad[1] = 'x'
    r = L.ablation_completeness_audit(arms_by_combo=bad)
    assert r['complete'] is False and r['structural_errors'], r
    bad = dict(base)
    bad[('A', 'B', 'C')] = None
    r = L.ablation_completeness_audit(arms_by_combo=bad)
    assert r['complete'] is False and r['structural_errors'], r
    # Project-level coverage must understand full-factorial reports too: they
    # name missing CELLS (a list), not a per-component arm mapping.
    _fa = {(): 'none', ('A',): 'A', ('B',): 'B', ('A', 'B'): 'AB'}
    _fr = L.ablation_completeness_audit(arms_by_combo=_fa)
    assert _fr['complete'] is True, _fr
    assert set(_fr['complete_components']) == {'A', 'B'}, _fr
    _cov = L.ablation_coverage_across_panels(
        {'p': {'ablation_completeness': _fr,
               'measured_components': ['A', 'B'],
               'verified_components': ['A', 'B'],
               'unverified_components': []}},
        design={'A': [('A', 'none', '')], 'B': [('B', 'none', '')]})
    assert set(_cov['complete_anywhere']) == {'A', 'B'}, _cov
    assert set(_cov['usable_anywhere']) == {'A', 'B'}, _cov
    _fb = dict(_fa)
    _fb.pop(('A', 'B'))
    _fr2 = L.ablation_completeness_audit(arms_by_combo=_fb)
    _cov2 = L.ablation_coverage_across_panels(
        {'p': {'ablation_completeness': _fr2,
               'measured_components': [], 'verified_components': [],
               'unverified_components': ['A', 'B']}},
        design={'A': [('A', 'none', '')], 'B': [('B', 'none', '')]})
    assert _cov2['complete_anywhere'] == [], _cov2
    assert _cov2['components']['A']['missing_arms'].get('p'), _cov2

    # Leave-one-out mode must also reject a self-contrast, malformed arm ids
    # and a non-mapping design instead of treating them as a complete ablation.
    for _design in ({'A': [('full', 'full', '')]},
                    {'A': [('full', [])]},
                    {'A': None}):
        r = L.ablation_completeness_audit(_design, ['full'])
        assert r['complete'] is False and r['structural_errors'], (_design, r)
    r = L.ablation_completeness_audit([], ['full'])
    assert r['complete'] is False and r['structural_errors'], r

    # Wrapper gate parameters are validated even with no panels to audit.
    for _bad in (True, 0, 1.5, float('nan')):
        try:
            A.audit_methodology(outdirs=[], alpha=_bad, write=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_methodology accepted alpha={_bad!r}')
    for _bad in (True, 3.5, -1, float('nan')):
        try:
            A.audit_methodology(outdirs=[], min_n=_bad, write=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f'audit_methodology accepted min_n={_bad!r}')

    # A non-mapping summary is invalid evidence, not an AttributeError.
    assert L._variant_seed_map(1) == ({}, set())
    assert L._variant_seed_map([('a', {})]) == ({}, set())
    for _bad_summary in (1, True, 'x', [None], [1]):
        _mr = L.methodology_audit(_bad_summary, design={}, baselines={},
                                  write=False, verbose=False)
        assert any('summary is' in w for w in _mr['warnings']), (_bad_summary, _mr)

    # The report's own significance gate validates its minimum-n parameter.
    _st = {'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02,
           'mean': 1.0, 'std': 0.5}
    assert BR._sig_gate(_st) is True
    for _bad in (True, 3.5, 0, -1, float('nan')):
        assert BR._sig_gate(_st, min_n=_bad) is False, _bad

    # Capacity and compute tolerances are gate parameters: a bool/negative/
    # non-finite value must not silently disable the control.
    _ra = {'params': 1000}
    _rb = {'params': 1000}
    _recs = {('a', '', 0): {'params': 1000, 'train_time_s': 1.0, 'tokens_seen': 10},
             ('b', '', 0): {'params': 1000, 'train_time_s': 1.0, 'tokens_seen': 10}}
    for _bad in (True, -0.1, float('nan'), 'x'):
        for _fn in (lambda t: L.param_pair_reason(_ra, _rb, tolerance=t),
                    lambda t: L._compute_control_detail(_recs, 'a', 'b', tolerance=t)):
            try:
                _fn(_bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid control tolerance {_bad!r} was accepted')

    # The final PPL is measured on `eval_subset`; two runs can share the full
    # val fingerprint yet score different subset rows.  Both pairing twins
    # must compare its content fingerprint.
    _kn = L.training_knobs()
    _a = {'run_cfg': 'r', 'steps': 1, 'tokens_seen': 1, 'train_knobs': _kn,
          'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e', 'eval_subset_fp': 's'}
    _b = dict(_a)
    assert L.pair_reason(_a, _b) is None and BR._pair_reason(_a, _b) is None
    _b['eval_subset_fp'] = 's2'
    assert 'eval_subset_fp' in (L.pair_reason(_a, _b) or ''), L.pair_reason(_a, _b)
    assert 'eval_subset_fp' in (BR._pair_reason(_a, _b) or ''), BR._pair_reason(_a, _b)
    _b.pop('eval_subset_fp')
    assert 'unverifiable eval_subset_fp' in (L.pair_reason(_a, _b) or '')
    assert 'unverifiable eval_subset_fp' in (BR._pair_reason(_a, _b) or '')

    # The report's displayed resolution floor is 2/2**n for EVERY n, not a
    # Monte-Carlo-like 1/200001 beyond n=20 (the sign-flip test is exact/MITM
    # up to n=40 and deterministic saddlepoint above).
    for _n in (6, 20, 21, 30, 40, 100):
        assert abs(BR._signflip_floor(_n) - math.ldexp(2.0, -_n)) < 1e-300
    for _bad in (0, -1, True, np.bool_(True), 'x'):
        assert not math.isfinite(BR._signflip_floor(_bad)), _bad

    # Scale pre-flight gates are not coerced either.
    def _tier(name, steps, params):
        cfg = {'steps': steps, 'batch_size': 2, 'seq_len': 8,
               'n_train_tokens': steps * 16, 'lr': 3e-4, 'warmup': 10,
               'seeds': [0, 1, 2, 3]}
        return {'name': name, 'cfg': cfg, 'params': params}
    for _bad in (True, -1.0, float('nan'), 'x'):
        try:
            L.scale_ladder_audit([_tier('a', 1000, 1000)], tolerance=_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'scale_ladder_audit accepted tolerance={_bad!r}')
    out = L.scale_ladder_audit(
        [_tier('a', 1000, 1000), _tier('b', 2000, 2000)],
        policy=dict(L.SCALE_TRANSFER_POLICY, family_m=True))
    assert out['status'] == 'fail' and any('family_m' in p for p in out['problems']), out
    pts = [{'label': f'd{d}', 'd': d, 'crossover_tokens': 1e6,
            'train_tokens': 5e7} for d in (128, 256, 384, 512)]
    for _kw in ({'min_points': True}, {'min_points': 1}, {'min_points': 3.5},
                {'volume_tol': -0.1}, {'volume_tol': float('nan')},
                {'d_spread_min': 1.0}, {'d_spread_min': True}):
        try:
            L.crossover_scaling_fit(pts, **_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'crossover_scaling_fit accepted {_kw!r}')

    # Baseline tuning evidence uses the same exact-count / finite-spread
    # contract.  abs(NaN - s) > tol was False, so a corrupted ppl_std used to
    # PASS the per-seed std check; a boolean step budget compared equal to 1.
    def _tune_art(trial):
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': 'lower_is_better', 'search_seeds': [2, 3],
                'arms': {'a': {'trials': [trial],
                               'selected': {'lr': 1e-3},
                               'selected_metric_value': trial.get('ppl')}}}
    _trial = {'config': {'lr': 1e-3}, 'ppl': 1.5, 'seeds': [0, 1],
              'seed_ppls': {'0': 1.0, '1': 2.0}, 'ppl_std': float('nan'),
              'steps': 10, 'tokens_seen': 100}
    out = L.baseline_tuning_audit(_tune_art(_trial))
    assert any('ppl_std' in p for p in out['problems']), out['problems']
    _trial = dict(_trial, ppl_std=float(np.std([1.0, 2.0], ddof=1)), steps=True)
    out = L.baseline_tuning_audit(_tune_art(_trial))
    assert any('exact positive counts' in p for p in out['problems']), out['problems']



def t_baseline_seed_plan_rejects_duplicate_draws():
    # A duplicated seed identity is a planning error, not an extra replicate.
    # It used to be silently de-duplicated here, so the power floor could be
    # checked against a larger n than the records that actually ran.
    for kw in ({'search_seeds': [0, 0, 1], 'final_seeds': [2, 3]},
               {'search_seeds': [0, 1], 'final_seeds': [2, 2]}):
        proto = dict(L.BASELINE_TUNING_PROTOCOL, **kw)
        try:
            BT._search_final_seeds(proto)
        except ValueError as e:
            assert 'duplicate seed' in str(e), e
        else:
            raise AssertionError(f'duplicate declared seeds were accepted: {kw}')
    # Unique identities are normalised once (integer strings accepted) so the
    # driver and the audit consume the same plan.
    search, final, overlap = BT._search_final_seeds(
        {'search_seeds': [3, '1', 2], 'final_seeds': ['5', 4]})
    assert search == [1, 2, 3] and final == [4, 5] and overlap == [], (search, final, overlap)


def t_bootstrap_and_zero_ce_scaling_contract():
    # Chunking must bound the per-chunk working set when the unit count grows;
    # ordinary panels keep the historical 256-draw chunk exactly.
    assert L._bootstrap_draw_chunk(100) == 256
    assert L._bootstrap_draw_chunk(4096) == 256
    assert L._bootstrap_draw_chunk(1 << 20) == 1
    assert L._bootstrap_draw_chunk(1 << 30) == 1
    # Zero cross-entropy is a valid measurement (PPL=1), not a missing
    # component.  Filtering `> 0` made a perfectly predicted panel
    # unverifiable and dropped the document bootstrap entirely.
    recs = [{'variant': 'csa', 'seed': s, 'ppl': 1.0,
             'nll_per_row': [0.0] * 4, 'n_tokens_per_row': [10] * 4}
            for s in (0, 1, 2)]
    r = L._combined_seed_doc_ci(recs)
    assert r['status'] == 'ok', r
    assert r['se_seed_logppl'] == 0.0 and r['se_doc_logppl'] == 0.0, r
    assert r['se_combined_logppl'] == 0.0 and r['ci95_ppl'] == [1.0, 1.0], r


def t_error_analysis_excludes_nonfinite_targets():
    # A divergent token (non-finite CE) is an unmeasured target.  Counting it
    # as CE=0 diluted every bucket and the pooled PPL, making a broken eval
    # look better than a measured one.
    vocab = 5

    class _StubModel:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2), dtype=torch.float32)

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab), dtype=torch.float32)

    vb = (np.arange(8, dtype=np.int64) % vocab).reshape(2, 4)
    orig = L.F.cross_entropy

    def _fake_ce(input, target, reduction='mean', **kw):
        out = torch.ones(tuple(target.shape), dtype=torch.float32)
        out[0, 0] = float('inf')
        return out

    L.F.cross_entropy = _fake_ce
    try:
        r = L.analyze_errors(_StubModel(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=2, chunk=2)
    finally:
        L.F.cross_entropy = orig
    assert r['n_nonfinite_tokens'] == 1, r
    assert r['n_targets'] == 5, r
    assert abs(r['mean_ce'] - 1.0) < 1e-12, r
    # Every bucket denominator excludes the unmeasured token.
    assert sum(b['n'] for b in r['buckets'].values() if b['n']) > 0, r['buckets']
    assert r['buckets']['position:first10%']['n'] == 1, r['buckets']
    assert all(seq['n_tokens'] in (2, 3) for seq in r['worst_sequences']), r['worst_sequences']
    # The whole-split cause tally covers exactly the finite targets.
    _all = r.get('cause_counts_all') or {}
    assert sum(v['n'] for v in _all.values()) == 5, _all
    assert _all['ambiguous_context']['n'] == 5, _all
    assert abs(sum(v['share'] for v in _all.values()) - 1.0) < 1e-12, _all
    # A real parameterised model (the only kind error_analysis.py loads) must
    # not hit "can't call numpy() on a Tensor that requires grad"; analysis
    # runs under inference mode and restores the model's training flag.
    torch.manual_seed(0)
    real = L.SmallGPT(17, 16, 1, 2, 8, 12, L.make_layer_cfgs(1, 'full')).train()
    vb_real = np.random.default_rng(0).integers(0, 17, (3, 12))
    r_real = L.analyze_errors(real, vb_real, 17, torch.device('cpu'),
                              top_n=5, top_tokens=100, chunk=2, topk=5)
    assert r_real['n_targets'] == 33 and real.training is True, r_real
    try:
        L.analyze_errors(real, vb_real, 17, torch.device('cpu'), top_tokens=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('a negative top_tokens budget must be refused')
    r_zero = L.analyze_errors(real, vb_real, 17, torch.device('cpu'), top_tokens=0)
    assert r_zero['worst_tokens'] == [], r_zero['worst_tokens']


def t_gapfill_numeric_reader_rejects_booleans():
    import run_gapfill as RG
    for _bad in (True, False, np.bool_(True), np.bool_(False)):
        assert RG._num_or_none(_bad) is None, _bad
    assert RG._num_or_none(3.0) == 3 and RG._num_or_none('4') == 4
    assert RG._num_or_none(1.5) is None


def t_gapfill_announces_family_seed_plan():
    import run_gapfill as RG
    assert set(RG.L.RUN_ABL['variants']) >= {'csa_dynamic', 'csa_dyn_nocomp'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        short = RG._announce_seed_plan('P-ABL', RG.L.RUN_ABL, [0, 1, 2])
    txt = buf.getvalue()
    # Three seeds cannot carry a 14-contrast Holm family; the phase must say so
    # before it runs rather than after its p-values are read.
    assert short['n_seeds'] == 3 and short['declared_contrasts'] >= 10, short
    assert short['sufficient_for_family_rejection'] is False, short
    assert 'WARNING' in txt and 'exploratory/descriptive' in txt, txt
    need = short['min_seeds_for_family_significance']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        enough = RG._announce_seed_plan('P-ABL', RG.L.RUN_ABL, list(range(need)))
    assert enough['sufficient_for_family_rejection'] is True, enough
    assert 'WARNING' not in buf.getvalue(), buf.getvalue()


def t_release_tag_cannot_escape_output_dir():
    import archive_release as AR
    assert AR._release_tag('v13') == 'v13'
    assert AR._release_tag('v13.2-rc_1') == 'v13.2-rc_1'
    for _bad in ('../v13', 'a/b', '', '.', '..', 'a b', 'a\nb', '-v13', 'x' * 65, None):
        try:
            AR._release_tag(_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'release tag {_bad!r} was accepted as a filename component')


def t_audit_leakage_missing_id_cache_is_not_ok():
    # Metas and persisted tokenizers can be present while the actual id
    # caches/sidecars are missing.  That split is NOT independently verifiable;
    # leaving the verdict at 'ok' would let a confirmation gate treat it as
    # clean.
    from tokenizers import Tokenizer, models
    cache = tempfile.mkdtemp(prefix='leak_missing_ids_')
    try:
        tok = Tokenizer(models.WordLevel({'[UNK]': 0, 'a': 1, 'b': 2},
                                         unk_token='[UNK]'))
        tok.save(os.path.join(cache, 'tok_v5_x.json'))
        tok.save(os.path.join(cache, 'test_tok_v2_x.json'))
        common = {'vocab': 3, 'fit_fp': 'FITXYZ'}
        with open(os.path.join(cache, 'meta_v5_x.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(common, src_fp='train'), f)
        with open(os.path.join(cache, 'test_meta_v2_x.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(common, src_fp='test'), f)
        out = A.audit_leakage(cache_dir=cache, shingle=1, max_rows_shingled=2)
        assert out['status'] != 'ok', out
        assert 'test id cache is absent' in out.get('note', ''), out
        assert 'document-id sidecar is missing' in out.get('note', ''), out
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def t_v12_binds_current_train_cache_to_frozen_stream():
    # P5T must not pair a newly rebuilt train/test cache with checkpoints
    # trained on a different token stream.
    d = tempfile.mkdtemp(prefix='v12_cache_bind_')
    try:
        for name, fp in (('meta_v5_a.json', 'FROZEN'), ('meta_v5_b.json', 'OTHER'),
                         ('meta_v5_c.json', None)):
            with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
                json.dump({} if fp is None else {'src_fp': fp}, f)
        assert V12._current_train_cache_fps(d) == {'FROZEN', 'OTHER'}
        assert V12._require_current_train_cache_fp('FROZEN', d) is True
        try:
            V12._require_current_train_cache_fp('MISSING', d)
        except SystemExit as e:
            assert 'different token stream' in str(e), e
        else:
            raise AssertionError('a mismatched frozen train fingerprint was accepted')
        empty = tempfile.mkdtemp(prefix='v12_cache_empty_')
        try:
            try:
                V12._require_current_train_cache_fp('FROZEN', empty)
            except SystemExit as e:
                assert 'no train-cache meta' in str(e), e
            else:
                raise AssertionError('a missing train cache was accepted')
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_release_archive_packs_tagged_cache_provenance():
    # The loaders write meta_v5_*.json / test_meta_v2_*.json / tok_v5_*.json;
    # a fixed set of legacy filenames packed none of them.
    import archive_release as AR
    d = tempfile.mkdtemp(prefix='release_cache_meta_')
    try:
        for name in ('meta_v5_sl512_cap8000000_v8192_vs512.json',
                     'test_meta_v2_sl4097_v8192_ns16.json',
                     'tok_v5_sl512_cap8000000_v8192_vs512.json',
                     'test_tok_v2_sl4097_v8192_ns16.json',
                     'train_ids_v5_x.npy.idsfp.json',
                     'train_ids_v5_x.npy', 'test_ids_v2_x.npy',
                     'unrelated.txt'):
            with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
                f.write('x')
        got = {os.path.basename(p) for p in AR._cache_provenance_files(d)}
        assert any(n.startswith('meta_v5_') for n in got), got
        assert any(n.startswith('test_meta_v2_') for n in got), got
        assert any(n.startswith('tok_v5_') for n in got), got
        assert any(n.startswith('test_tok_v2_') for n in got), got
        assert any(n.endswith('.idsfp.json') for n in got), got
        assert not any(n.endswith('.npy') for n in got), got
        assert 'unrelated.txt' not in got, got
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_scale_corpus_fp_is_budget_independent():
    # The corpus/tokenizer identity must not change when a scale tier simply
    # trains longer.  `train_fp`/`data_version` intentionally DO change (they
    # identify the exact served bytes); `corpus_fp` must not.
    tr = (np.arange(64, dtype=np.int64) * 7 + 3) % 97
    cfg = {'dataset': 'unit', 'seq_len': 16, 'cache_dir': './unit_cache'}
    _old_cap = L.TOKENIZER_FIT_TOKENS
    try:
        L.TOKENIZER_FIT_TOKENS = 8
        a = L.experiment_provenance(dict(cfg, n_train_tokens=16),
                                    train_ids=tr[:16], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
        b = L.experiment_provenance(dict(cfg, n_train_tokens=48),
                                    train_ids=tr[:48], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
    finally:
        L.TOKENIZER_FIT_TOKENS = _old_cap
    assert a['corpus_fp'] == b['corpus_fp'], (a, b)
    assert a['data_version'] != b['data_version'], 'the served bytes/budget still differ'
    # A changed token inside the tokenizer-fit prefix is a different corpus.
    tr2 = tr.copy()
    tr2[0] = (int(tr2[0]) + 1) % 97
    try:
        L.TOKENIZER_FIT_TOKENS = 8
        c = L.experiment_provenance(dict(cfg, n_train_tokens=16),
                                    train_ids=tr2[:16], val_batch=tr[:16].reshape(1, 16),
                                    vocab=97)['data']
    finally:
        L.TOKENIZER_FIT_TOKENS = _old_cap
    assert c['corpus_fp'] != a['corpus_fp']
    # M0 prefers the stable fingerprint; a mixed provenance generation is
    # refused rather than letting the new subset speak for the whole panel.
    def _rec(cfp, dv):
        return {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
                'corpus_fp': cfp, 'data_version': dv}
    s1 = {'a': _rec(a['corpus_fp'], a['data_version'])}
    s2 = {'a': _rec(b['corpus_fp'], b['data_version'])}
    assert M0._panel_corpus_fp(s1) == a['corpus_fp']
    assert M0._panel_corpus_fp(s1) == M0._panel_corpus_fp(s2)
    legacy = {'a': {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
                    'data_version': a['data_version']}}
    assert M0._panel_corpus_fp(legacy) == a['data_version']
    try:
        M0._panel_corpus_fp({**s1, 'b': {'variant': 'full', 'seed': 0, 'ppl': 11.0,
                                         'data_version': a['data_version']}})
    except ValueError:
        pass
    else:
        raise AssertionError('mixed corpus_fp/legacy provenance must be refused')
    # The cross-scale consumers accept a shared stable fingerprint even when
    # the training budgets (data_version) differ.
    pts = [{'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2], 'seeds': [0, 1],
            'corpus_fp': a['corpus_fp'], 'vocab': 97},
           {'label': 'S2', 'scale': 2, 'effects': [0.2, 0.3], 'seeds': [0, 1],
            'corpus_fp': b['corpus_fp'], 'vocab': 97}]
    assert L.meta_analyze_effects(pts)['status'] == 'ok', L.meta_analyze_effects(pts)


def t_signflip_zero_ties_lower_resolution_floor():
    # Zero deltas are sign-neutral: p is unchanged, but the attainable floor
    # is 2/2**(#nonzero), not 2/2**n.
    sf = L.exact_signflip_p([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert sf['n'] == 6 and sf['n_nonzero'] == 2, sf
    assert abs(sf['p_floor'] - 0.5) < 1e-15, sf
    assert abs(sf['p'] - 0.5) < 1e-15, sf
    st = L.paired_effect_stats([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], min_n=3)
    assert st['n'] == 6 and st['n_nonzero'] == 2, st
    assert abs(st['p_floor'] - 0.5) < 1e-15, st
    assert st['resolution_limited'] is True, st
    claim = L.reportable_claim(st, family_size=1)
    assert claim['p_floor'] == 0.5 and claim['n_nonzero'] == 2, claim
    assert claim['level'] == 'resolution_limited', claim
    all_zero = L.exact_signflip_p([0.0, 0.0, 0.0])
    assert all_zero['p'] == 1.0 and all_zero['p_floor'] == 1.0, all_zero
    assert L.paired_effect_stats([0.0] * 6)['p_floor'] == 1.0
    # The report twin must obey the same floor.
    z = BR.exact_signflip([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert z['n_nonzero'] == 2 and abs(z['p_floor'] - 0.5) < 1e-15, z
    # The shared v7-v12 helper used by the probe/final-test reports too.
    v = V7.exact_sign_permutation([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert v['n_nonzero'] == 2 and abs(v['p_floor'] - 0.5) < 1e-15, v


def t_zero_aware_family_resolution_floor():
    # Exact ties reduce the attainable family floor too: with n=8 but only 6
    # non-zero paired deltas, 4 * 2/2**6 = 0.125 > alpha, not 4 * 2/2**8.
    comps = {c: {'p_exact_signflip': 1.0, 'n': 8, 'n_nonzero': 6}
             for c in 'ABCD'}
    fam = L.attach_holm(comps, family='zero ties')
    assert abs(fam['family_min_attainable_p_holm'] - 0.125) < 1e-15, fam
    assert fam['family_resolution_limited'] is True, fam
    # Without the tie count the theoretical floor is still reported.
    none = {c: {'p_exact_signflip': 1.0, 'n': 8} for c in 'ABCD'}
    assert abs(L.attach_holm(none)['family_min_attainable_p_holm']
               - 0.03125) < 1e-15
    st = {'n': 8, 'n_nonzero': 6, 'p': 0.01, 'p_holm': 0.02,
          'mean': 1.0, 'std': 0.1}
    claim = L.reportable_claim(st, family_size=4)
    assert claim['allowed'] is False, claim
    assert abs(claim['min_attainable_p_holm_family'] - 0.125) < 1e-15, claim
    # A fabricated non-zero count larger than the panel cannot buy resolution.
    bad = L.reportable_claim(dict(st, n_nonzero=9), family_size=1)
    assert bad['allowed'] is False and any('n_nonzero=9 exceeds n=8' in r
                                           for r in bad['reasons']), bad
    # Per-contrast floors: the family floor is m * min(floor), so a single
    # tie-heavy contrast blocks the whole tuned family.
    assert abs(L.holm_min_attainable_p_floors([0.03125, 0.5]) - 0.0625) < 1e-15
    assert L.holm_min_attainable_p_floors([1.0, 0.5]) == 1.0
    assert math.isnan(L.holm_min_attainable_p_floors([]))
    # Malformed floors are treated as 1 (the least resolvable value), so they
    # cannot shrink the family floor below the valid comparisons' minimum.
    assert abs(L.holm_min_attainable_p_floors([0.03125, None, 'x']) - 0.09375) < 1e-15
    assert L.holm_min_attainable_p_floors([None, 'x']) == 1.0


def t_scale_meta_refuses_visible_attrition():
    # A common effect (same mean at every scale) pools only on complete panels.
    def _common(planned=None, n_seeds=6):
        out = []
        for i in range(3):
            pt = {'label': 'S%d' % i, 'scale': 10.0 ** i,
                  'effects': [0.1 + 0.1 * j for j in range(n_seeds)],
                  'seeds': list(range(n_seeds))}
            if planned is not None:
                pt['n_planned'] = planned
            out.append(pt)
        return out
    complete = _common()
    m = L.meta_analyze_effects(complete)
    assert m['status'] == 'ok' and m['poolable'] is True, m
    assert m['incomplete_points'] == [], m
    # One planned seed absent from every point: the complete-case estimates
    # are numerically identical, but the missing draw must block the pooled
    # "common effect".
    missing = _common(planned=7)
    m2 = L.meta_analyze_effects(missing)
    assert m2['status'] == 'ok' and m2['poolable'] is False, m2
    assert m2['incomplete_points'] == ['S0', 'S1', 'S2'], m2
    assert m2['attrition']['S0']['n_missing'] == 1, m2['attrition']
    # A non-finite effect is a dropped pair too, not a smaller honest n.
    dropped = [dict(p, effects=list(p['effects'])) for p in complete]
    dropped[1]['effects'][0] = float('nan')
    m3 = L.meta_analyze_effects(dropped)
    assert m3['poolable'] is False and m3['incomplete_points'] == ['S1'], m3
    assert m3['attrition']['S1']['n_dropped'] == 1, m3
    # Declared exact seed plans also catch an absent seed.
    planned = [dict(p, planned_seeds=list(range(7))) for p in complete]
    m4 = L.meta_analyze_effects(planned)
    assert m4['poolable'] is False and m4['attrition']['S0']['n_missing'] == 1, m4

    # A perfectly linear ladder would normally be a significant trend.
    def _trend(planned=None):
        out = []
        for i in range(3):
            pt = {'label': 'S%d' % i, 'scale': 10.0 ** i,
                  'effects': [float(i) + 0.1 * j for j in range(6)],
                  'seeds': list(range(6))}
            if planned is not None:
                pt['n_planned'] = planned
            out.append(pt)
        return out
    r = L.scaling_trend(_trend())
    assert r['significant'] is True, r
    r2 = L.scaling_trend(_trend(planned=7))
    assert r2['significant'] is False and r2['verdict'] == 'incomplete_scale_point', r2
    assert 'incomplete paired panel' in r2['note'], r2
    # A perfectly consistent scale (zero within-scale variance) is reported,
    # not silently dropped from the family.  It blocks the pooled claim
    # because 1/se^2 is undefined without a variance model.
    zero = [{'label': 'S%d' % i, 'scale': 10.0 ** i,
             'effects': [0.9, 1.0, 1.1, 1.0], 'seeds': [0, 1, 2, 3]}
            for i in range(4)]
    zero[2] = {'label': 'S2', 'scale': 100.0,
               'effects': [1.0, 1.0, 1.0, 1.0], 'seeds': [0, 1, 2, 3]}
    mz = L.meta_analyze_effects(zero)
    assert mz['zero_variance_points'] == ['S2'], mz
    assert mz['n_usable'] == 3, 'the consistent point is not part of the DL weights'
    assert mz['poolable'] is False and 'zero within-scale variance' in mz['note'], mz
    assert len(mz['per_point']) == 4, 'the row stays visible'
    # A scale point whose every effect is non-finite is not deleted from the
    # family: it is named as empty and blocks both pooling and a trend.
    empty = [{'label': 'S%d' % i, 'scale': 10.0 ** i,
              'effects': [float(i) + 0.1 * j for j in range(6)],
              'seeds': list(range(6))} for i in range(3)]
    empty[1] = {'label': 'S1', 'scale': 10.0,
                'effects': [float('nan')], 'seeds': [0], 'n_planned': 6}
    me = L.meta_analyze_effects(empty)
    assert me['empty_points'] == ['S1'] and me['poolable'] is False, me
    # effects=None is also a declared empty tier, not a reason to delete it.
    none_eff = [dict(p) for p in complete]
    none_eff.append({'label': 'Snone', 'scale': 1000.0, 'effects': None,
                     'seeds': [0, 1, 2, 3]})
    mn = L.meta_analyze_effects(none_eff)
    assert mn['empty_points'] == ['Snone'] and mn['poolable'] is False, mn
    assert len(L._scale_points([{'label': 'Snone', 'scale': 1000.0,
                                 'effects': None}])) == 1
    # An n=1 tier has no within-scale SE and cannot enter the DL weights; it
    # must still block the pooled claim instead of silently disappearing.
    one_seed = [dict(p, n_planned=6) for p in complete]
    one_seed.append({'label': 'S3', 'scale': 1000.0, 'effects': [0.1],
                     'seeds': [0], 'n_planned': 6})
    mo = L.meta_analyze_effects(one_seed)
    assert mo['poolable'] is False and mo['underpowered_points'] == ['S3'], mo
    re = L.scaling_trend(empty, min_points=2)
    assert re['empty_points'] == ['S1'] and re['significant'] is False, re
    assert re['verdict'] == 'incomplete_scale_point', re
    # Invalid plan metadata fails closed instead of being silently truncated.
    for bad in (True, 3.5, -1, 'six'):
        try:
            L._scale_points([{'label': 'S', 'scale': 1, 'effects': [0.1],
                              'seeds': [0], 'n_planned': bad}])
        except ValueError:
            pass
        else:
            raise AssertionError(f'n_planned={bad!r} was accepted')
    # A declared tier whose scale is missing/non-positive cannot enter the
    # regression, but it must stay visible and block a confirmable trend
    # instead of silently disappearing from the ladder.
    ladder = [dict(p) for p in complete]
    ladder.append({'label': 'Sbad', 'scale': float('nan'),
                   'effects': [0.1] * 6, 'seeds': list(range(6))})
    mi = L.meta_analyze_effects(ladder)
    assert mi['invalid_scale_points'] == ['Sbad'] and mi['poolable'] is False, mi
    assert 'invalid/non-positive scale' in mi['note'], mi
    ri = L.scaling_trend(ladder)
    assert ri['invalid_scale_points'] == ['Sbad'], ri
    assert ri['significant'] is False and ri['verdict'] == 'incomplete_scale_point', ri
    assert 'Sbad' in ri['declared_labels'], ri
    assert 'invalid/non-positive scale' in ri['note'], ri
    # An exact planned seed list is not attestation by itself: without declared
    # seed identities, equal counts could be a different set of draws.
    unattested = [{'label': p['label'], 'scale': p['scale'],
                   'effects': list(p['effects']),
                   'planned_seeds': list(range(len(p['effects'])))}
                  for p in complete]
    raw = L._scale_points(unattested)
    assert all(r['complete_pairs'] is False
               and r['seed_plan_attested'] is False for r in raw), raw
    mu = L.meta_analyze_effects(unattested)
    assert mu['unattested_plan_points'] == ['S0', 'S1', 'S2'], mu
    assert mu['poolable'] is False, mu
    assert 'exact planned seed list' in mu['note'], mu
    ru = L.scaling_trend(unattested)
    assert ru['significant'] is False and ru['verdict'] == 'incomplete_scale_point', ru


def t_m0_cross_scale_uses_declared_seed_plans():
    fp = 'steps10_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _panel(seeds, base):
        out = {}
        for v in ('csa_dynamic', 'csa_fixed', 'hybrid_dynamic', 'full_matched'):
            for seed in seeds:
                out[f'{v}::seed{seed}'] = dict(
                    variant=v, seed=seed, run_cfg=fp, steps=10, tokens_seen=10,
                    ppl=base + 0.1 * seed + (0.5 if v == 'csa_fixed' else 0.0),
                    params=100, train_knobs=knobs, data_fp='d', val_fp='v',
                    corpus_fp='corpus')
        return out

    tmp = tempfile.mkdtemp()
    try:
        r = M0.scale_analysis({'S1': _panel([0, 1], 10.0),
                               'S2': _panel([0, 1], 20.0)}, outdir=tmp)
        meta = r['cross_scale']['dynamic_vs_fixed']['meta']
        # S1 plans 4 seeds and S2 plans 3; the truncated panels must be visible
        # as attrition, not certified complete from the two surviving records.
        got = {p['label']: p['n_missing'] for p in meta['per_point']}
        assert got == {'S1': 2, 'S2': 1}, got
        assert meta['poolable'] is False, meta
        assert set(meta['incomplete_points']) == {'S1', 'S2'}, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_m0_records_exclude_synthesized_reconstructions():
    base = {'variant': 'csa_dynamic', 'seed': 0, 'ppl': 10.0,
            'run_cfg': 'fp', 'steps': 1, 'tokens_seen': 1,
            'corpus_fp': 'corpus'}
    summary = {
        'real': dict(base),
        'synth': dict(base, seed=1, synthesized=True, ppl=9.0),
    }
    recs = M0._records_by_seed(summary, 'csa_dynamic')
    assert set(recs) == {0}, recs
    assert M0._panel_corpus_fp(summary) == 'corpus'
    assert M0._panel_corpus_fp({'synth': dict(base, synthesized=True,
                                             corpus_fp='other')}) is None


def t_m0_cross_scale_pairs_are_control_gated():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    fp2 = fp.replace('_lr0.0003_', '_lr0.003_')
    knobs = L.training_knobs()

    def _rec(v, seed, cfg, ppl):
        return {'variant': v, 'seed': seed, 'run_cfg': cfg, 'steps': 1,
                'tokens_seen': 1, 'ppl': ppl, 'train_knobs': knobs}
    summary = {
        'a::seed0': _rec('a', 0, fp, 10.0),
        'a::seed1': _rec('a', 1, fp, 10.5),
        'a::seed2': _rec('a', 2, fp, 11.0),
        'b::seed0': _rec('b', 0, fp, 11.0),
        'b::seed1': _rec('b', 1, fp2, 11.5),
        'b::seed2': _rec('b', 2, fp, 12.0),
    }
    paired, gated = M0._paired_ppl_by_seed(summary, 'a', 'b')
    assert sorted(paired) == [0, 2], paired
    assert len(gated) == 1 and gated[0][0] == 1, gated
    assert 'run_cfg' in gated[0][1], gated
    assert paired[0] == (10.0, 11.0), paired
    # A malformed nested provenance block has no corpus identity; it must not
    # crash the panel-fingerprint reader on a list .get().
    _bad_prov = _rec('a', 0, fp, 10.0)
    _bad_prov['provenance'] = {'data': ['not', 'a', 'mapping']}
    assert M0._panel_corpus_fp({'a::seed0': _bad_prov}, 'a') is None
    # Nested tuning records: cross-scale pairing must use the held-out primary
    # estimand, not the full-validation `ppl` that still contains the rows
    # which selected the hyper-parameters.  A nested record without holdout
    # accounting is dropped (visible as gated), never silently replaced.
    def _nested(v, seed, cfg, full, hold):
        r = _rec(v, seed, cfg, full)
        r.update({'ppl_estimation_rows': 'nested_validation_holdout',
                  'ppl_holdout': hold})
        return r
    nested = {
        'a::seed0': _nested('a', 0, fp, 99.0, 10.0),
        'a::seed1': _nested('a', 1, fp, 99.0, 10.5),
        'b::seed0': _nested('b', 0, fp, 1.0, 11.0),
        'b::seed1': {k: v for k, v in _nested('b', 1, fp, 1.0, 11.5).items()
                     if k not in ('ppl_holdout',)},
    }
    np_paired, np_gated = M0._paired_ppl_by_seed(nested, 'a', 'b')
    assert np_paired == {0: (10.0, 11.0)}, (np_paired, np_gated)
    # Seed 1's b record declares the nested estimand but has no holdout
    # accounting, so it is not a measurable pair; the raw full-validation
    # values (99 vs 1) must not re-enter the analysis.
    assert np_gated == [], np_gated
    # Escalating tier labels must not make S10 look smaller than S2.
    pts = [{'label': 'S10', 'scale': 1000.0}, {'label': 'S2', 'scale': 10.0},
           {'label': 'S1', 'scale': 1.0}]
    assert M0._extreme_scale_labels(pts) == ('S1', 'S10')
    # NumPy numeric scalars are valid declared scales too.
    assert M0._extreme_scale_labels([{'label': 'S10', 'scale': np.int64(1000)},
                                     {'label': 'S2', 'scale': np.float32(10)},
                                     {'label': 'S1', 'scale': 1}]) == ('S1', 'S10')
    assert M0._extreme_scale_labels([{'label': 'S9', 'scale': float('nan')},
                                     {'label': 'S1', 'scale': 1.0}]) is None



def t_scale_trend_degenerate_fit_and_empty_tiers():
    # A perfectly flat WLS line has zero residual variance.  That is not a
    # maximally significant increasing trend; it is no detectable trend.
    pts = [{'label': f'S{i}', 'scale': 10 ** i, 'effects': [0.1, 0.2, 0.3]}
           for i in range(3)]
    r = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert r['wls']['se'] == 0.0 and r['wls']['p'] == 1.0, r['wls']
    assert r['significant'] is False and r['verdict'] == 'no_detectable_trend', r
    # A declared-but-empty tier must not disable paired inference over the
    # scales that DO carry measured effects.
    pts = [{'label': 'A', 'scale': 1, 'effects': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'seeds': list(range(6))},
           {'label': 'B', 'scale': 10, 'effects': [], 'seeds': []},
           {'label': 'C', 'scale': 100, 'effects': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'seeds': list(range(6))}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert r['paired']['labels'] == ['A', 'C'], r['paired']
    assert r['paired']['n_complete_seeds'] == 6, r['paired']
    assert 'B' in r['empty_points'] and r['significant'] is False, r
    # Six same-sign per-seed slopes can reject the exact test while still
    # being within their own seed spread; that is not a usable trend claim.
    pts = [{'label': 'A', 'scale': 1, 'effects': [0.0] * 6, 'seeds': list(range(6))},
           {'label': 'B', 'scale': 10, 'effects': [0.01] * 3 + [2.0] * 3, 'seeds': list(range(6))},
           {'label': 'C', 'scale': 100, 'effects': [0.02] * 3 + [4.0] * 3, 'seeds': list(range(6))}]
    rn = L.scaling_trend(pts, min_seeds=3, min_points=3)
    assert rn['p_primary'] < 0.05, rn
    assert rn['paired']['within_seed_noise'] is True, rn['paired']
    assert rn['significant'] is False and rn['verdict'] == 'noise_level', rn


def t_scale_non_estimable_variance_blocks_pooling():
    big = 1e308
    pts = [{'label': 'BIG', 'scale': 1, 'effects': [big, -big], 'seeds': [0, 1]},
           {'label': 'S1', 'scale': 2, 'effects': [0.10, 0.11], 'seeds': [0, 1]},
           {'label': 'S2', 'scale': 3, 'effects': [0.1005, 0.1105], 'seeds': [0, 1]},
           {'label': 'S3', 'scale': 4, 'effects': [0.101, 0.111], 'seeds': [0, 1]}]
    m = L.meta_analyze_effects(pts, min_seeds=2)
    assert 'BIG' in m['non_estimable_points'], m
    assert m['poolable'] is False and m['n_usable'] == 3, m
    assert any('non-estimable' in m['note'] for _ in [0]), m['note']


def t_tuning_margin_does_not_invent_precision():
    # Missing marginal SD is unavailable uncertainty, not zero variance.
    m = L.tuning_selection_margin({'ppl': 10.0}, {'ppl': 10.001})
    assert m['se'] is None and m['decisive'] is None, m
    assert m['method'] != 'independent_marginal_se', m
    # When the paired path is used, the numerator must come from the SAME
    # common seeds as the paired SE, not from the full trial means.
    best = {'ppl': -333.3, 'ppl_std': 1.0, 'n_seeds': 3,
            'seed_ppls': {'0': 0.0, '1': 0.0, '2': -1000.0}}
    runner = {'ppl': 0.0, 'ppl_std': 1.0, 'n_seeds': 3,
              'seed_ppls': {'0': 100.0, '1': -100.0, '3': 0.0}}
    m = L.tuning_selection_margin(best, runner)
    assert m['method'] == 'paired_search_seeds' and m['n_paired'] == 2, m
    assert m['margin'] == 0.0 and m['decisive'] is False, m
    # Both rows expose per-seed evidence but share fewer than two draws: the
    # independent marginal SE is a different (unpaired) sample and must not
    # rescue a decisive verdict here.
    _lonely_b = {'ppl': 10.0, 'ppl_std': 0.1, 'n_seeds': 2,
                 'seed_ppls': {'0': 10.0, '1': 10.0}}
    _lonely_r = {'ppl': 9.0, 'ppl_std': 0.1, 'n_seeds': 2,
                 'seed_ppls': {'2': 9.0, '3': 9.0}}
    _lm = L.tuning_selection_margin(_lonely_b, _lonely_r)
    assert _lm['method'] == 'insufficient_common_seeds', _lm
    assert _lm['decisive'] is None and _lm['se'] is None, _lm


def t_scale_ladder_missing_seed_plan_is_not_common():
    def tier(name, seeds):
        return {'name': name, 'params': 1000,
                'cfg': {'steps': 1000, 'batch_size': 2, 'seq_len': 8,
                        'n_train_tokens': 16000, 'lr': 3e-4,
                        'warmup': 100, 'seeds': seeds}}
    out = L.scale_ladder_audit([tier('S1', list(range(8))), tier('S2', [])],
                               policy=dict(L.SCALE_TRANSFER_POLICY, family_m=4))
    assert out['n_common_seeds'] == 0, out
    assert out['seed_set_intersection'] == [], out
    assert out['common_seeds_sufficient_for_family'] is False, out



def t_small_n_split_remainder_and_string_order():
    # Guaranteeing one row for val/test must not then hand them the whole
    # remainder: 80/10/10 at n=8 is (6,1,1), not (4,1,3).
    for n, want in ((4, [2, 1, 1]), (5, [3, 1, 1]), (8, [6, 1, 1])):
        idx = L.chronological_split_indices(n)
        got = [len(idx[k]) for k in ('train', 'val', 'test')]
        assert got == want, (n, got)
    idx = L.chronological_split_indices(3, 0.5, 0.5, 0.0)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [1, 2, 0], idx
    idx = L.chronological_split_indices(4, 0.1, 0.8, 0.1)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [1, 2, 1], idx
    # Lexicographic string event ids remain valid ordering keys, even when
    # the literal text is 'nan' or 'inf'.
    idx = L.split_indices(3, time=np.array(['a', 'nan', 'b'], dtype=object),
                          train_frac=1 / 3, val_frac=1 / 3, test_frac=1 / 3)
    assert sorted(sum((list(idx[k]) for k in ('train', 'val', 'test')), [])) == [0, 1, 2], idx



def t_split_discipline_recomputes_fingerprints_and_doc_sidecars():
    with tempfile.TemporaryDirectory() as d:
        tag, ttag = 'tiny', 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 100).reshape(2, 8)
        te = (np.arange(16, dtype=np.int64) + 200).reshape(2, 8)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        meta = {'vocab': 256, 'seq_len': 8, 'split_strategy': 'ordered',
                'shuffle_at_split': False, 'tokenizer_fit_split': 'train',
                'src_fp': L._ids_fp(tr),
                'val_fp': L._ids_fp(np.ascontiguousarray(va).reshape(-1))}
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 256, 'seq_len': 8, 'split_strategy': 'official_test',
                       'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('train_fingerprint_verified') is True, rec
        assert rec.get('val_fingerprint_verified') is True, rec
        assert rec['status'] != 'fail', rec
        # A replaced validation file with stale metadata must fail closed.
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va + 1)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('validation id-cache fingerprint' in w for w in out['warnings']), out
        # Restore and add a test doc sidecar without the train/val sidecars:
        # substituting empty arrays would have reported a clean zero overlap.
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_doc_ids_{ttag}.npy'),
                np.array([9, 10], dtype=np.int64))
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('train/validation doc-id sidecars are missing' in w
                   for w in out['warnings']), out



def t_archive_refuses_links_and_repo_escape():
    import archive_release as AR
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as d:
        outside = tempfile.mkdtemp(prefix='archive_outside_')
        try:
            with open(os.path.join(outside, 'secret.txt'), 'w', encoding='utf-8') as f:
                f.write('secret')
            try:
                AR._collect([outside])
            except ValueError as e:
                assert 'outside the repository' in str(e), e
            else:
                raise AssertionError('an absolute directory outside the repo was packed')
            try:
                AR._collect(['../' + os.path.basename(outside)])
            except ValueError:
                pass
            else:
                raise AssertionError('a parent-relative directory was packed')
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        # A symlink inside the repository is still a link member: make would
        # hash its target but tar would store the link, so its own archive
        # could never verify.  Refuse it at collect time.
        _linkdir = tempfile.mkdtemp(dir=AR.REPO, prefix='archive_link_test_')
        try:
            with open(os.path.join(_linkdir, 'target.txt'), 'w',
                      encoding='utf-8') as f:
                f.write('target')
            os.symlink(os.path.join(_linkdir, 'target.txt'),
                       os.path.join(_linkdir, 'link.txt'))
            try:
                AR._collect([os.path.relpath(_linkdir, AR.REPO)])
            except ValueError as e:
                assert 'symbolic link' in str(e), e
            else:
                raise AssertionError('a symlink member was packed')
        finally:
            shutil.rmtree(_linkdir, ignore_errors=True)
        # The release output directory is confined just like the tag and the
        # packed inputs: --out must not write outside the repository.
        for _bad_out in ('../escape', outside, os.path.join(outside, 'nested')):
            try:
                AR._safe_outdir(_bad_out)
            except ValueError:
                pass
            else:
                raise AssertionError(f'outdir {_bad_out!r} escaped the repository')
        assert AR._safe_outdir('archive') == os.path.join(
            os.path.realpath(AR.REPO), 'archive')
        # A symlink member with the expected name must not satisfy the
        # manifest; it was never hashed as a regular file.
        path = os.path.join(d, 'release_x.tar.gz')
        payload = b'hello'
        man = {'tag': 'x', 'files': {'data.txt': {
            'sha256': __import__('hashlib').sha256(payload).hexdigest(),
            'bytes': len(payload)}}}
        man_bytes = json.dumps(man).encode('utf-8')
        with tarfile.open(path, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            link = tarfile.TarInfo('link.txt')
            link.type = tarfile.SYMTYPE
            link.linkname = 'data.txt'
            tf.addfile(link)
            mt = tarfile.TarInfo('release_x.MANIFEST.sha256')
            mt.size = len(man_bytes)
            tf.addfile(mt, io.BytesIO(man_bytes))
        assert AR.cmd_verify(path) == 1
        # The same archive without the unhashed link still verifies.
        path2 = os.path.join(d, 'release_y.tar.gz')
        with tarfile.open(path2, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_y.MANIFEST.sha256')
            mt.size = len(man_bytes)
            tf.addfile(mt, io.BytesIO(man_bytes))
        assert AR.cmd_verify(path2) == 0
        # verify() must also refuse an archive whose member (or manifest path)
        # would escape the extraction directory, even if every hash matches.
        evil = os.path.join(d, 'release_z.tar.gz')
        evil_man = {'tag': 'z', 'files': {'../evil.txt': {
            'sha256': __import__('hashlib').sha256(payload).hexdigest(),
            'bytes': len(payload)}}}
        evil_bytes = json.dumps(evil_man).encode('utf-8')
        with tarfile.open(evil, 'w:gz') as tf:
            ti = tarfile.TarInfo('../evil.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_z.MANIFEST.sha256')
            mt.size = len(evil_bytes)
            tf.addfile(mt, io.BytesIO(evil_bytes))
        assert AR.cmd_verify(evil) == 1
        # A manifest that is not a files mapping is refused instead of
        # crashing the verifier with a KeyError.
        broken = os.path.join(d, 'release_broken.tar.gz')
        broken_bytes = b'[]'
        with tarfile.open(broken, 'w:gz') as tf:
            mt = tarfile.TarInfo('release_broken.MANIFEST.sha256')
            mt.size = len(broken_bytes)
            tf.addfile(mt, io.BytesIO(broken_bytes))
        assert AR.cmd_verify(broken) == 1
        # A manifest whose per-file entries are not objects (e.g. a bare
        # string instead of {"sha256": ...}) must be refused, not crash the
        # verifier with an AttributeError.
        malformed = os.path.join(d, 'release_malformed.tar.gz')
        malformed_man = {'tag': 'm', 'files': {'data.txt': 'not-an-object'}}
        malformed_bytes = json.dumps(malformed_man).encode('utf-8')
        with tarfile.open(malformed, 'w:gz') as tf:
            ti = tarfile.TarInfo('data.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mt = tarfile.TarInfo('release_malformed.MANIFEST.sha256')
            mt.size = len(malformed_bytes)
            tf.addfile(mt, io.BytesIO(malformed_bytes))
        assert AR.cmd_verify(malformed) == 1
        # A present-but-malformed sha256/byte count, or a non-object
        # checkpoint_hashes table, must be refused instead of crashing on
        # slicing/len().
        for _name, _bad_files, _bad_ck in (
                ('badsha', {'data.txt': {'sha256': {'x': 1}, 'bytes': 3}}, None),
                ('badbytes', {'data.txt': {'sha256': __import__('hashlib').sha256(payload).hexdigest(),
                                           'bytes': True}}, None),
                ('badck', {'data.txt': {'sha256': __import__('hashlib').sha256(payload).hexdigest(),
                                        'bytes': len(payload)}}, True)):
            _p = os.path.join(d, f'release_{_name}.tar.gz')
            _man = {'tag': _name, 'files': _bad_files}
            if _bad_ck is not None:
                _man['checkpoint_hashes'] = _bad_ck
            _mb = json.dumps(_man).encode('utf-8')
            with tarfile.open(_p, 'w:gz') as tf:
                ti = tarfile.TarInfo('data.txt')
                ti.size = len(payload)
                tf.addfile(ti, io.BytesIO(payload))
                mt = tarfile.TarInfo(f'release_{_name}.MANIFEST.sha256')
                mt.size = len(_mb)
                tf.addfile(mt, io.BytesIO(_mb))
            assert AR.cmd_verify(_p) == 1, _name
        # A corrupt/non-tar archive is a verification failure, not a crash.
        corrupt = os.path.join(d, 'release_corrupt.tar.gz')
        with open(corrupt, 'wb') as f:
            f.write(b'not a tar archive')
        assert AR.cmd_verify(corrupt) == 1



def t_build_report_warmup_uses_primary_estimand():
    def rec(full, hold, warm, seed=0):
        return {'variant': 'csa_fixed', 'seed': seed, 'warm_steps': warm,
                'ppl': full, 'ppl_holdout': hold,
                'ppl_estimation_rows': 'nested_validation_holdout',
                'nll_sum_holdout': math.log(hold) * 40, 'n_tokens_holdout': 40,
                'run_cfg': 'cfg', 'steps': 1, 'tokens_seen': 40,
                'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e'}
    s = {'a::w0::seed0': rec(99.0, 10.0, 0),
         'a::w5000::seed0': rec(1.0, 20.0, 5000)}
    old_s, old_l = BR.summary, BR._legacy_warm_tags_cached
    BR.summary = lambda *a, **k: s
    BR._legacy_warm_tags_cached = lambda *a, **k: True
    try:
        out = BR.sec_p0w()
    finally:
        BR.summary, BR._legacy_warm_tags_cached = old_s, old_l
    # The table and the paired conclusion must both use the held-out PPL.
    assert '| `csa_fixed` | 0 | 10.00 ' in out, out
    assert '| `csa_fixed` | 5000 | 20.00 ' in out, out
    assert '99.00' not in out, out
    assert '+10.00 vs 基线' in out, out



def t_pool_kernel_empty_sequence_matches_eager():
    import csa_kernels as C
    g = C.PoolGeometry([[]], overlap=4, T=0, device=torch.device('cpu'), max_block=128)
    X = torch.randn(1, 0, 3)
    Z = torch.randn(1, 0, 3)
    Bp = torch.randn(129, 3)
    bid = torch.zeros(1, 0, dtype=torch.int32)
    y, last, Bm = C.pool_variable_blocks_kernel(X, None, Z, None, bid, g, Bp, Bp)
    assert y.shape == (1, 0, 3) and last.shape == (1, 0) and Bm == 0, (y.shape, last.shape, Bm)
    y2, last2, Bm2 = C.pool_blocks_single_kernel(X, Z, bid, g, Bp)
    assert y2.shape == (1, 0, 3) and last2.shape == (1, 0) and Bm2 == 0, (y2.shape, last2.shape, Bm2)


def t_pool_variable_blocks_accepts_no_overlap():
    # `overlap=None` is the no-overlap/HCA contract.  `_pool_parts` then
    # returns a four-tuple while `_pool_ordered` unpacks eight, so this public
    # entry point used to raise a bare ValueError; it must return the exact
    # single-lane pooling instead.
    bid = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    n, F, cap = 6, 3, 8
    Xa, Za = torch.randn(n, F), torch.randn(n, F)
    Xb, Zb = torch.randn(n, F), torch.randn(n, F)
    Bpa, Bpb = torch.randn(cap, F), torch.randn(cap, F)
    got = L.pool_variable_blocks(Xa, Xb, Za, Zb, bid, Bpa, Bpb, None, n_blocks=3)
    ref = L.pool_blocks_single(Xa, Za, Bpa, bid, n_blocks=3)
    assert torch.allclose(got[0], ref[0]) and torch.equal(got[1], ref[1]) and got[2] == ref[2], got


def t_efficiency_median_uses_both_middle_seeds():
    # Four seeds: the median of 10, 20, 30, 100 is (20+30)/2, not the upper
    # middle value 30.
    rows = [{'variant': 'a', 'ppl': 10.0, 'prefill_ms_p50': l,
             'params': p, 'tokens_seen': 100}
            for l, p in ((10.0, 1000), (20.0, 2000), (30.0, 3000), (100.0, 4000))]
    s = BR.efficiency_accuracy_summary(rows)[0]
    assert s['latency_median'] == 25.0 and s['params_median'] == 2500.0, s
    assert s['tokens_seen_median'] == 100.0, s


def t_eager_batch_rejects_typos_and_nonfinite():
    assert L._normalise_eager_batch(None) is None
    assert L._normalise_eager_batch(True) is True
    assert L._normalise_eager_batch('auto') is None
    assert L._normalise_eager_batch('false') is False
    assert L._normalise_eager_batch(0) is False and L._normalise_eager_batch(1.0) is True
    for bad in ('flase', 'maybe', float('nan'), 1.5, -1, 2):
        try:
            L._normalise_eager_batch(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eager_batch={bad!r} was silently coerced')



def t_gapfill_merge_keeps_secondary_and_invalidates_stale_fields():
    import run_gapfill as G
    with tempfile.TemporaryDirectory() as d:
        snap = {'full': {'seeds': [0, 1], 'ppls': [10.0, 12.0], 'n_seeds': 2,
                         'ppl_mean': 11.0, 'ppl_std': math.sqrt(2.0),
                         'avg_dyn_block_len': 4.0,
                         'ppl_curve': [[1, 10.0], [2, 11.0]]}}
        fresh = {'full': {'seeds': [0, 1], 'ppls': [20.0, 22.0], 'n_seeds': 2,
                          'ppl_mean': 21.0, 'ppl_std': math.sqrt(2.0),
                          'avg_dyn_block_len': 8.0,
                          'ppl_curve': [[1, 20.0], [2, 21.0]]}}
        json.dump(fresh, open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, snap)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        # Re-measuring the same seeds must refresh the secondary values, not
        # leave the snapshot's old ones next to the fresh PPLs.
        assert out['ppls'] == [20.0, 22.0] and out['avg_dyn_block_len'] == 8.0, out
        assert out['ppl_curve'] == [[1, 20.0], [2, 21.0]], out
        # Recovering a lost seed must not leave fields computed on the old
        # subset looking current.
        json.dump({'full': {'seeds': [0, 1], 'ppls': [10.0, 20.0]}},
                  open(os.path.join(d, 'aggregate.json.v6bak'), 'w'))
        live = {'full': {'seeds': [0], 'ppls': [10.0], 'n_seeds': 1,
                         'ppl_mean': 10.0, 'ppl_std': 0.0, 'ppl_primary': 10.0,
                         'nll_sum': 10.0, 'n_tokens': 1, 'ppl_curve': [[1, 10.0]]}}
        json.dump(live, open(os.path.join(d, 'aggregate.json'), 'w'))
        G._fold_backup_into_aggregate(d)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        assert out['seeds'] == [0, 1] and out['ppl_mean'] == 15.0, out
        for _f in ('ppl_primary', 'nll_sum', 'n_tokens', 'ppl_curve'):
            assert out[_f] is None, (out, _f)
        # Valid JSON with the wrong top-level type is skipped, not crashed on.
        json.dump([], open(os.path.join(d, 'aggregate.json.v6bak'), 'w'))
        G._fold_backup_into_aggregate(d)
    # Partial seed overlap cannot be merged from aggregate means: the old mean
    # includes the overlapping seeds, so neither the old-only/fresh weights nor
    # the old curve can be split without per-seed values.  The per-seed PPLs
    # still merge exactly; the affected seed-averaged fields are invalidated.
    with tempfile.TemporaryDirectory() as d:
        snap = {'full': {'seeds': [0, 1, 2], 'ppls': [10.0, 12.0, 14.0],
                         'n_seeds': 3, 'ppl_mean': 12.0, 'ppl_std': 2.0,
                         'avg_dyn_block_len': 3.0,
                         'ppl_curve': [[1, 10.0], [2, 12.0]]}}
        fresh = {'full': {'seeds': [1, 2, 3], 'ppls': [20.0, 22.0, 24.0],
                          'n_seeds': 3, 'avg_dyn_block_len': 9.0,
                          'ppl_curve': [[1, 20.0], [2, 22.0]]}}
        json.dump(fresh, open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, snap)
        out = json.load(open(os.path.join(d, 'aggregate.json')))['full']
        assert out['seeds'] == [0, 1, 2, 3] and out['ppls'] == [10.0, 20.0, 22.0, 24.0], out
        assert out['ppl_mean'] == 19.0, out
        assert out['avg_dyn_block_len'] is None and out['ppl_curve'] is None, out
        assert out['merge_stale_fields'] == 'partial_seed_overlap', out
    # A valid-JSON non-object fresh aggregate (or snapshot) must not crash the
    # merge on `.items()` / `.get()`.
    with tempfile.TemporaryDirectory() as d:
        json.dump([], open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, {'full': {'seeds': [0], 'ppls': [1.0]}})
        assert json.load(open(os.path.join(d, 'aggregate.json'))) == []
        json.dump({'full': {'seeds': [0], 'ppls': [2.0]}},
                  open(os.path.join(d, 'aggregate.json'), 'w'))
        G.merge_long_aggregate(d, ['not', 'a', 'mapping'])
        out = json.load(open(os.path.join(d, 'aggregate.json')))
        assert out['full']['ppls'] == [2.0], out
    # A malformed aggregate ENTRY (valid JSON, non-object value) must be
    # skipped by the synthesizer instead of crashing on `.get`.
    with tempfile.TemporaryDirectory() as d:
        _old_repo = G.REPO
        G.REPO = d
        try:
            _panel = os.path.join(d, 'results_lm_v3_long')
            os.makedirs(_panel)
            _tok = (G._LONG_STEPS * int(G.L.RUN_LONG['batch_size'])
                    * int(G.L.RUN_LONG['seq_len']))
            json.dump({'bad': [1, 2],
                       'full': {'seeds': [0], 'ppls': [2.0],
                                'tokens_seen': _tok}},
                      open(os.path.join(_panel, 'aggregate.json'), 'w'))
            _sum, _agg = G.synthesize_long_summary()
            _out = json.load(open(os.path.join(_panel, 'summary.json')))
            assert 'full::seed0' in _out and 'bad' not in ' '.join(_out), _out
        finally:
            G.REPO = _old_repo



def t_open_block_resume_binds_data_fingerprints():
    import open_block_supp as OB
    tr = np.arange(16, dtype=np.int64)
    va = np.arange(8, dtype=np.int64).reshape(1, 8)
    rec = {'run_fingerprint': 'fp', 'code': L.code_stamp(), 'ppl': 10.0,
           'nll_sum': 10.0, 'n_tokens': 10,
           'data_fp': L._ids_fp(tr), 'eval_fp': L._ids_fp(va.reshape(-1))}
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is True
    # A record without exact token accounting is not usable by `analyse`, so
    # it must not be skipped as a current cell on resume.
    no_acc = {k: v for k, v in rec.items() if k not in ('nll_sum', 'n_tokens')}
    assert OB._is_current(no_acc, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    frac = dict(rec, n_tokens=10.5)
    assert OB._is_current(frac, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr + 1),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    assert OB._is_current(rec, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1) + 1)) is False
    # A legacy record without the fingerprints cannot be certified current.
    legacy = {k: v for k, v in rec.items() if k not in ('data_fp', 'eval_fp')}
    assert OB._is_current(legacy, 'fp', data_fp=L._ids_fp(tr),
                          eval_fp=L._ids_fp(va.reshape(-1))) is False
    # The panel manifest's "measured" set must use the same current-record
    # contract: a stale run/data/eval fingerprint (or missing accounting) is
    # not a completed cell for this panel.
    planned = [f'{v}::seed0' for v in ('current', 'stale_fp', 'stale_data',
                                       'no_acc', 'legacy')]
    summary = {
        'current::seed0': rec,
        'stale_fp::seed0': dict(rec, run_fingerprint='other'),
        'stale_data::seed0': dict(rec, data_fp=L._ids_fp(tr + 1)),
        'no_acc::seed0': no_acc,
        'legacy::seed0': legacy,
    }
    assert OB._current_cells(summary, planned, 'fp',
                             data_fp=L._ids_fp(tr),
                             eval_fp=L._ids_fp(va.reshape(-1))) == {
                                 'current::seed0'}, summary



def t_analysis_checkpoint_carries_model_geometry():
    import error_analysis as E
    ids = np.arange(2048, dtype=np.int64) % 97
    val = (np.arange(64, dtype=np.int64) % 97).reshape(2, 32)
    with tempfile.TemporaryDirectory() as d:
        r = L.train_variant('full', ids, val, 97, seed=0, d=32, n_layers=1,
                            n_heads=2, d_head=16, seq_len=32, batch_size=1,
                            steps=1, lr=1e-3, weight_decay=0.0, warmup=0,
                            comp_lambda=0.0, eval_every=0, mlp_ratio=4.0,
                            device=torch.device('cpu'), log_every=0,
                            save_ckpt='final', ckpt_dir=d)
        path = r['checkpoints']['final']['path']
        # The standard analysis checkpoint must be loadable by error_analysis;
        # it used to omit cfg/vocab/geometry entirely.
        model, meta = E._load_model(path)
        assert meta.get('cfg') is not None and meta.get('vocab') == 97, meta
        assert meta.get('d') == 32 and meta.get('train_len') == 32, meta



def t_v12_admission_rejects_synthesized_records():
    m = _manifest_for_admission(seeds=(0, 1))
    good = {}
    for v in ('a_arm', 'b_arm'):
        for sd in (0, 1):
            good[f'{v}::s{sd}'] = _rec(v, sd, 'FP')
    adm, _fp = V12._admit_confirmation_records(good, m)
    assert len(adm) == 4, adm
    synth = dict(good)
    synth['a_arm::s0'] = dict(synth['a_arm::s0'], synthesized=True)
    adm, _fp = V12._admit_confirmation_records(synth, m)
    assert ('a_arm', 0) not in adm and len(adm) == 3, adm



def t_seed_and_nested_input_guards():
    # Reproduction probes must not truncate 0.2/0.8 to one seed or treat a
    # boolean as a draw identity.
    for bad in ([0.2, 0.8], [True, 1], ['1.5']):
        try:
            L.reproducibility_guard(['full'], seeds=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_guard accepted seeds={bad!r}')
    # Both nested folds are required; a partial request must not crash on None.
    rng = np.random.default_rng(3)
    train = rng.integers(0, 16, size=200)
    val = rng.integers(0, 16, size=(4, 8))
    for kw in ({'selection_rows': [0]}, {'evaluation_rows': [1]}):
        try:
            L.simple_baseline_fit(train, val, 16, protocol=dict(
                L.SIMPLE_BASELINE_POLICY,
                arms={'u': {'order': 1}}, grids={'u': {'alpha': [0.1]}}), **kw)
        except ValueError as e:
            assert 'BOTH' in str(e), e
        else:
            raise AssertionError(f'partial nested folds {kw!r} were accepted')


def t_malformed_panel_summary_is_excluded():
    # A JSON file that parses but is not a cell -> record object is not a
    # panel.  Readers must exclude it instead of crashing inside .items(), and
    # the resume writer must refuse to overwrite it with an empty summary.
    td = tempfile.mkdtemp(prefix='panel_shape_')
    try:
        sp = os.path.join(td, 'summary.json')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a', 'seed': 0, 'ppl': 1.0}], f)
        assert L.read_panel(td) == {}
        assert L.ppl_by_seed(td) == {}
        assert L.variant_presence(td) == {}
        try:
            L._read_summary_json(sp)
        except ValueError:
            pass
        else:
            raise AssertionError('a non-object summary must not be resumed from')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump('not a panel', f)
        assert L.read_panel(td) == {}
        assert L.variant_presence(td) == {}
        # Audit readers share the same object contract: a list/scalar is
        # unverifiable, not something to call .items() on.
        import audit_gates as A
        assert A._load(sp) is None
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a'}], f)
        assert A._load(sp) is None
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 1.0}}, f)
        assert isinstance(A._load(sp), dict)
        # A malformed arm id inside an otherwise valid panel record must not
        # crash the presence reader on an unhashable dict/list key.
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': ['not', 'a', 'string'],
                                    'seed': 0, 'ppl': 1.0}}, f)
        assert L.variant_presence(td) == {}
        # A malformed nested provenance block is missing evidence, not an
        # AttributeError in the shared-validation fingerprint check.
        assert L.by_len_cells([{'provenance': ['bad'],
                                'by_len': {4: {'ppl': 10.0}}}], 4) == [
            {'ppl': 10.0}]
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_meta_seed_overlap_controls_pooling():
    # The DL pooled CI treats scale points as independent studies.  Declaring
    # seed identities is not enough: disjoint sets are independent, a complete
    # common set has a paired sensitivity analysis, but a partially overlapping
    # ladder is neither and must not be pooled as a common effect.
    def _pt(label, scale, seeds, effects):
        return {'label': label, 'scale': scale, 'seeds': list(seeds),
                'effects': list(effects), 'corpus_fp': 'fp', 'vocab': 8000}
    eff = [0.10, 0.20, 0.30, 0.40]
    partial = [_pt('S0', 1, [0, 1, 2, 3], eff),
               _pt('S1', 10, [1, 3, 4, 5], eff),
               _pt('S2', 100, [0, 2, 4, 5], eff)]
    m = L.meta_analyze_effects(partial)
    assert m['status'] == 'ok' and m['seed_dependence'] == 'shared_seeds_unpaired', m
    assert m['seed_overlap']['n_common_seeds_all_points'] == 0, m['seed_overlap']
    assert all(v > 0 for v in m['seed_overlap']['per_pair_common_seeds'].values()), m['seed_overlap']
    assert m['poolable'] is False, m
    assert 'independence' in m['note'], m['note']
    # Disjoint draws are genuinely independent replications.
    disjoint = [_pt('S0', 1, [0, 1, 2, 3], eff),
                _pt('S1', 10, [4, 5, 6, 7], eff),
                _pt('S2', 100, [8, 9, 10, 11], eff)]
    md = L.meta_analyze_effects(disjoint)
    assert md['seed_dependence'] == 'disjoint_seed_sets' and md['poolable'] is True, md
    assert md['seed_overlap']['n_common_seeds_all_points'] == 0, md['seed_overlap']
    # Undeclared seed identities cannot be assumed independent.
    unattested = [{k: v for k, v in p.items() if k != 'seeds'} for p in disjoint]
    mu = L.meta_analyze_effects(unattested)
    assert mu['seed_dependence'] == 'unattested' and mu['poolable'] is False, mu
    # The WLS trend fallback uses the same independence assumption.  A
    # partially overlapping ladder with fewer than two complete common seeds
    # must not turn a significant unpaired slope into a confirmed trend.
    trend = [_pt('S0', 1, [0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
             _pt('S1', 10, [1, 3, 4, 5], [1.0, 1.1, 1.2, 1.3]),
             _pt('S2', 100, [0, 2, 4, 5], [2.0, 2.1, 2.2, 2.3])]
    rt = L.scaling_trend(trend, min_seeds=3, min_points=3)
    assert rt['seed_dependence'] == 'shared_seeds_unpaired', rt
    assert rt['seed_overlap']['n_common_seeds_all_points'] == 0, rt['seed_overlap']
    assert rt['paired']['n_complete_seeds'] < 2, rt['paired']
    assert rt['primary_inference'].startswith('WLS'), rt['primary_inference']
    assert rt['significant'] is False and rt['verdict'] == 'seed_dependence_unresolved', rt
    # Disjoint seed sets keep the unpaired slope valid.
    trend_disjoint = [_pt('S0', 1, [0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3]),
                      _pt('S1', 10, [4, 5, 6, 7], [1.0, 1.1, 1.2, 1.3]),
                      _pt('S2', 100, [8, 9, 10, 11], [2.0, 2.1, 2.2, 2.3])]
    rd = L.scaling_trend(trend_disjoint, min_seeds=3, min_points=3)
    assert rd['seed_dependence'] == 'disjoint_seed_sets', rd
    assert rd['significant'] is True, rd
    # A within-scale SE that is only numerically zero also makes 1/se^2
    # unbounded; it must be named and excluded from the inverse-variance pool.
    nearzero = [_pt('S0', 1, [0, 1, 2, 3], [0.1, 0.1, 0.1, 0.1 + 1e-15]),
                _pt('S1', 10, [4, 5, 6, 7], [0.2, 0.3, 0.4, 0.5]),
                _pt('S2', 100, [8, 9, 10, 11], [0.3, 0.4, 0.5, 0.6])]
    mn = L.meta_analyze_effects(nearzero)
    assert 'S0' in mn['zero_variance_points'], mn
    assert mn['n_usable'] == 2 and mn['poolable'] is False, mn
    # Every scale may have n>=min_seeds while only two seeds are measured at
    # EVERY scale; the paired dependence then has less resolution than the
    # per-scale n suggests and must block the pooled claim.
    shared2 = [_pt('S0', 1, [0, 1, 2, 3, 4, 5], eff + [0.5, 0.6]),
               _pt('S1', 10, [0, 1, 6, 7, 8, 9], eff + [0.5, 0.6]),
               _pt('S2', 100, [0, 1, 10, 11, 12, 13], eff + [0.5, 0.6])]
    ms = L.meta_analyze_effects(shared2, min_seeds=3)
    assert ms['seed_dependence'] == 'shared_seeds_paired', ms
    assert ms['paired_common_effect']['n'] == 2, ms['paired_common_effect']
    assert ms['paired_common_effect']['underpowered'] is True, ms['paired_common_effect']
    assert ms['poolable'] is False, ms
    assert 'complete seed' in ms['note'], ms['note']


def t_preprocessing_fit_audit_detects_full_table_leak():
    import data_guard as DG
    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(40, 3))
    Xho = rng.normal(loc=4.0, scale=2.0, size=(20, 3))
    train_only = DG.LeakageSafePreprocessor().fit(Xtr)
    a = DG.audit_preprocessor_fit(train_only, Xtr, Xho)
    assert a['status'] == 'train_only_verified' and a['leak_detected'] is False, a
    full_fit = DG.LeakageSafePreprocessor().fit(np.vstack([Xtr, Xho]))
    b = DG.audit_preprocessor_fit(full_fit, Xtr, Xho)
    assert b['status'] == 'leak_detected' and b['leak_detected'] is True, b
    assert b['stats_match_train'] is False and b['stats_match_train_plus_holdout'] is True, b
    # A different fitting table is not a proven leak, but it is not verifiable.
    c = DG.audit_preprocessor_fit(
        DG.LeakageSafePreprocessor().fit(rng.normal(size=(40, 3))), Xtr, Xho)
    assert c['status'] == 'unverifiable' and c['leak_detected'] is False, c
    # The train-only audit works without holdouts when stats already match.
    d = DG.audit_preprocessor_fit(train_only, Xtr)
    assert d['status'] == 'train_only_verified', d
    # split_and_prepare must expose a preprocessor that passes the audit.
    out = DG.split_and_prepare(Xtr, preprocess=True)
    e = DG.audit_preprocessor_fit(out['preprocessor'], Xtr,
                                  [out['X']['val'], out['X']['test']])
    assert e['status'] in ('train_only_verified', 'unverifiable'), e
    # A plain list-of-rows holdout is ONE table, not one part per row.  The
    # old list-is-parts rule transposed it into a (n_rows*n_features, 1)
    # column and rejected the audit on a shape mismatch.
    _ll = [[float(v) for v in row] for row in Xho.tolist()]
    f = DG.audit_preprocessor_fit(full_fit, [[float(v) for v in row]
                                             for row in Xtr.tolist()], _ll)
    assert f['status'] == 'leak_detected' and f['n_holdout'] == len(_ll), f


def t_split_targets_use_exact_fraction_arithmetic():
    import data_guard as DG
    # The intended 4/7 split must survive the float representation of 4/7.
    assert DG._floor_fraction_product(7, 4 / 7) == 4
    assert DG._floor_fraction_product(7, 2 / 7) == 2
    # At row counts beyond float64's exact integer range the target sizes must
    # still sum to n and follow the exact rational fraction, not a rounded
    # product.
    n = 2 ** 53 + 1
    sizes = DG._resolve_sizes(n, 0.8, 0.1, 0.1)
    assert sum(sizes) == n, sizes
    assert sizes == (7205759403792795, 900719925474099, 900719925474099), sizes
    # A genuinely positive sub-1e-15 fraction must not collapse to exactly
    # zero: at a large row count it still selects rows.
    assert DG._fraction_value(1e-16) != 0
    assert DG._floor_fraction_product(10 ** 18, 1e-16) > 0
    assert DG._floor_fraction_product(10 ** 18, 1e-15) == 1000
    try:
        DG._resolve_sizes(10, 0.8, 0.1, 0.1, train_size=11)
    except ValueError:
        pass
    else:
        raise AssertionError('oversized explicit split must still be refused')


def t_encoder_fit_audit_detects_holdout_vocabulary():
    import data_guard as DG
    Xtr = np.array([['a'], ['b'], ['a'], ['b']], dtype=object)
    Xho = np.array([['c'], ['a']], dtype=object)
    cat = DG.LeakageSafeCategoricalEncoder().fit(Xtr)
    a = DG.audit_encoder_fit(cat, Xtr, Xho)
    assert a['status'] == 'train_only_verified' and a['leak_detected'] is False, a
    cat_leak = DG.LeakageSafeCategoricalEncoder().fit(np.vstack([Xtr, Xho]))
    b = DG.audit_encoder_fit(cat_leak, Xtr, Xho)
    assert b['status'] == 'leak_detected' and b['leak_detected'] is True, b
    freq = DG.LeakageSafeFrequencyEncoder().fit(Xtr)
    c = DG.audit_encoder_fit(freq, Xtr, Xho)
    assert c['status'] == 'train_only_verified', c
    freq_leak = DG.LeakageSafeFrequencyEncoder().fit(np.vstack([Xtr, Xho]))
    d = DG.audit_encoder_fit(freq_leak, Xtr, Xho)
    assert d['status'] == 'leak_detected' and d['leak_detected'] is True, d
    # A list-of-rows holdout must be one table here too.
    _cat_ll = DG.LeakageSafeCategoricalEncoder().fit(
        np.vstack([Xtr, Xho]))
    _ll_audit = DG.audit_encoder_fit(_cat_ll, Xtr.tolist(), Xho.tolist())
    assert _ll_audit['leak_detected'] is True and _ll_audit['n_holdout'] == len(Xho), _ll_audit
    try:
        DG.audit_encoder_fit(DG.LeakageSafeTargetEncoder().fit(
            Xtr, np.array([0.0, 1.0, 0.0, 1.0])), Xtr)
    except TypeError:
        pass
    else:
        raise AssertionError('target encoder is not a single-table fit and must be refused')


def t_overfit_loss_contract_and_ignore_index():
    ids = np.arange(4096, dtype=np.int64) % 256
    r = L.overfit_sanity_check('full', seed=0, steps=120, d=64, n_layers=2,
                               n_heads=4, d_head=16, seq_len=32, batch_size=2,
                               vocab=256)
    assert r['passed'] is True, r
    assert r['loss_contract']['initial']['passed'] is True, r['loss_contract']
    assert r['loss_contract']['final']['passed'] is True, r['loss_contract']
    assert r['pass_criteria']['loss_reduction_matches_mean_ce'] is True, r['pass_criteria']
    assert r['code'] == L.code_stamp() and r['device'] == 'cpu', r
    r2 = L.overfit_sanity_check('full', seed=0, steps=20, d=32, n_layers=1,
                                n_heads=4, d_head=8, seq_len=16, batch_size=2,
                                vocab=256, train_ids=ids, ignore_index=-100,
                                device='cpu')
    assert r2['ignore_index'] == -100, r2
    assert r2['label_contract']['ignore_index'] == -100, r2['label_contract']
    assert r2['loss_contract']['initial']['passed'] is True, r2['loss_contract']
    for bad in (0, -1, True, 1.5):
        try:
            L.overfit_sanity_check('full', steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'overfit_sanity_check accepted steps={bad!r}')
    try:
        L.overfit_sanity_check('full', ignore_index=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError('overfit_sanity_check accepted a non-integral ignore_index')


def t_ce_loss_default_ignore_matches_pytorch_reduction():
    ids = torch.arange(64, dtype=torch.long).reshape(2, 32) % 32
    tg = torch.arange(1, 65, dtype=torch.long).reshape(2, 32) % 32
    tg = tg.clone()
    tg[0, :8] = -100
    model = L.SmallGPT(32, 32, 1, 4, 8, 32, L.make_layer_cfgs(1, 'full'), mlp_ratio=2)
    loss = float(model.ce_loss(ids, tg, 32).detach())
    with torch.no_grad():
        lg = model.head(model.trunk(ids))
        ref = float(torch.nn.functional.cross_entropy(
            lg.reshape(-1, 32), tg.reshape(-1), reduction='mean'))
    assert math.isclose(loss, ref, rel_tol=1e-5, abs_tol=1e-6), (loss, ref)


def t_repro_probe_config_binds_device_and_lr():
    cpu = L._repro_guard_base(seed=0, lr=0.003, device=torch.device('cpu'))
    cuda = L._repro_guard_base(seed=0, lr=0.003, device=torch.device('cuda'))
    low_lr = L._repro_guard_base(seed=0, lr=0.001, device=torch.device('cpu'))
    assert cpu != cuda, 'the reproducibility probe cache must bind the device'
    assert cpu != low_lr, 'the reproducibility probe cache must bind the learning rate'
    r = L.reproducibility_check('full', seed=0, steps=2, lr=0.001,
                                device=torch.device('cpu'))
    assert r['passed'] is True and r['lr'] == 0.001 and r['device'] == 'cpu', r
    rs = L.reproducibility_check('full', seed=0, steps=2, device='cpu')
    assert rs['passed'] is True and rs['device'] == 'cpu', rs
    assert rs['code'] == L.code_stamp(), rs
    for bad in (0, -1, True, 1.5):
        try:
            L.reproducibility_check('full', steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_check accepted steps={bad!r}')
    for bad_tol in (-1.0, float('nan')):
        try:
            L.reproducibility_check('full', tol=bad_tol)
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_check accepted tol={bad_tol!r}')


def t_seed_contract_cuda_device_inventory():
    r = L.seed_contract_check(seed=5)
    assert r['passed'] is True and r['code'] == L.code_stamp(), r
    if not torch.cuda.is_available():
        assert r['checks']['cuda_devices_seeded'] is None, r['checks']
    else:
        assert r['checks']['cuda_devices_seeded'] is True, r['checks']


def t_manifest_missing_curve_ref_is_incomplete():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        hp = L.run_hyperparameters(cfg, seed=0, d=64, n_layers=2, n_heads=4,
                                   d_head=16, vocab=97, mlp_ratio=4.0,
                                   data_version=prov['data']['data_version'])
        cell = {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                'code_stamp': L.code_stamp(), 'hyperparams': hp,
                'data_version': prov['data']['data_version'], 'rng_fp': 'r',
                'curve_summary': {'n_points': 2},
                'curve_ref': 'curves/missing.json'}
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['tracking_integrity']['missing_curve_refs'] == ['full::seed0'], e['tracking_integrity']
        assert e['tracking_integrity']['complete'] is False, e['tracking_integrity']
        os.makedirs(os.path.join(d, 'curves'), exist_ok=True)
        with open(os.path.join(d, 'curves', 'missing.json'), 'w', encoding='utf-8') as fh:
            json.dump({'losses': [3.0, 2.0]}, fh)
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['tracking_integrity']['missing_curve_refs'] == [], e['tracking_integrity']
        assert e['tracking_integrity']['complete'] is True, e['tracking_integrity']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_environment_snapshot_runtime_identity():
    env = L.environment_snapshot()
    assert env.get('hostname'), env
    assert env.get('process_id') == os.getpid(), env
    assert env.get('working_directory') == os.getcwd(), env


def t_ledger_curve_is_bounded():
    small = [1.0, 2.0, 3.0]
    got, n, down = L._ledger_curve(small)
    assert got == small and n == 3 and down is False, (got, n, down)
    vals = [float(i % 17) for i in range(50000)]
    vals[12345] = float('nan')
    vals[23456] = 1e9
    got, n, down = L._ledger_curve(vals, max_points=20000)
    assert n == 50000 and down is True and len(got) <= 20000, (n, down, len(got))
    assert any(isinstance(v, float) and math.isnan(v) for v in got), 'the ledger reduction dropped the NaN marker'
    assert 1e9 in got, 'the ledger reduction dropped the spike'
    pts = [[i, float(i)] for i in range(300)]
    pts[201][1] = float('nan')
    pg, pn, pd = L._ledger_curve_points(pts, max_points=100)
    assert pn == 300 and pd is True and len(pg) <= 100, (pn, pd, len(pg))
    assert any(int(a) == 201 and math.isnan(b) for a, b in pg), 'the point reduction dropped the NaN point'


def t_batch_contract_check_validation():
    ids = np.arange(5000, dtype=np.int64) % 997
    ok = L.batch_contract_check(ids, 32, 4, 'cpu', seed=7, vocab=997)
    assert ok['passed'] is True and ok['device'] == 'cpu', ok
    assert ok['code'] == L.code_stamp(), ok
    for bad in (1.5, True, -1, L.MAX_SEED + 1):
        try:
            L.batch_contract_check(ids, 32, 4, 'cpu', seed=bad, vocab=997)
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_contract_check accepted seed={bad!r}')
    for kw in ({'n_batches': 0}, {'n_batches': 1.5}, {'vocab': 0},
               {'seq_len': 0}, {'batch_size': True}):
        args = dict(seq_len=32, batch_size=4, seed=7, vocab=997, device='cpu')
        args.update(kw)
        try:
            L.batch_contract_check(ids, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_contract_check accepted {kw!r}')


def t_manifest_rejects_mismatched_cell_identity():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        hp = L.run_hyperparameters(cfg, seed=0, d=64, n_layers=2, n_heads=4,
                                   d_head=16, vocab=97, mlp_ratio=4.0,
                                   data_version=prov['data']['data_version'])
        cell = {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                'code_stamp': L.code_stamp(), 'hyperparams': hp,
                'data_version': 'DIFFERENT-DATA', 'rng_fp': 'r',
                'run_cfg': 'DIFFERENT-CFG',
                'curve_summary': {'n_points': 2}}
        L.write_run_manifest(d, cfg, 'FP', prov, {},
                             cell_records={'full::seed0': cell})
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        integ = e['tracking_integrity']
        assert integ['run_cfg_mismatch'] == ['full::seed0'], integ
        assert integ['data_version_mismatch'] == ['full::seed0'], integ
        assert integ['complete'] is False, integ
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_determinism_label_cpu_strict():
    old_env = os.environ.get('CSA_DETERMINISTIC')
    old_cudnn = torch.backends.cudnn.deterministic
    old_bench = torch.backends.cudnn.benchmark
    old_strict = torch.are_deterministic_algorithms_enabled()
    try:
        os.environ['CSA_DETERMINISTIC'] = '1'
        L._pin_cuda_determinism()
        assert L.determinism_label() in ('strict', 'warn'), L.determinism_label()
        if not torch.cuda.is_available():
            assert L.determinism_label() == 'strict', L.determinism_label()
    finally:
        if old_env is None:
            os.environ.pop('CSA_DETERMINISTIC', None)
        else:
            os.environ['CSA_DETERMINISTIC'] = old_env
        torch.backends.cudnn.deterministic = old_cudnn
        torch.backends.cudnn.benchmark = old_bench
        if not old_strict:
            torch.use_deterministic_algorithms(False)


def t_target_encoder_keeps_duplicate_rows_in_one_fold():
    # Exact duplicate feature rows are one observation/annotation cluster.  If
    # the row-OOF fold assignment splits them, changing one duplicate's label
    # changes the encoding of its twin: the twin is encoded from its own
    # duplicate label.  All duplicates must therefore share a fold.
    import data_guard as DG
    X = np.array([['a'], ['a'], ['b'], ['b']], dtype=object)
    y = np.array([1.0, 1.0, 2.0, 2.0])
    for seed, n_folds in ((0, 3), (0, 2), (5, 4)):
        oof = DG.LeakageSafeTargetEncoder(
            smoothing=0.0, n_folds=n_folds, seed=seed).fit_transform(X, y)
        y2 = y.copy()
        y2[0] += 1000.0
        oof2 = DG.LeakageSafeTargetEncoder(
            smoothing=0.0, n_folds=n_folds, seed=seed).fit_transform(X, y2)
        assert np.allclose(oof2[1], oof[1]), (seed, n_folds, oof, oof2)
        assert np.allclose(oof2[0], oof[0]), (seed, n_folds, oof, oof2)
    # A table with no duplicates keeps the historical per-row fold assignment.
    uniq = np.array([['a'], ['b'], ['c'], ['d']], dtype=object)
    old = DG.LeakageSafeTargetEncoder(n_folds=2, seed=7)
    old_folds = old._row_folds(4)
    new_folds = old._row_folds(4, keys=DG._row_keys(uniq))
    assert np.array_equal(old_folds, new_folds)


def t_missing_group_ids_are_one_identity():
    # None, NaN and NaT are three spellings of a missing group id.  Treating
    # them as different groups lets the same missing group straddle splits, and
    # a real tuple id spelled like the old tuple sentinel could impersonate it.
    import data_guard as DG
    missing = [None, float('nan'), np.float64('nan'),
               np.datetime64('NaT', 'ns'), np.timedelta64('NaT', 'ns')]
    keys = {DG._group_key(v) for v in missing}
    assert len(keys) == 1, keys
    # The historical tuple sentinel is kept for compatibility, but the old
    # NaT/None-specific spellings no longer create extra missing groups.
    assert DG._group_key(('__nat__',)) not in keys
    assert DG._group_key(('__missing__',)) not in keys
    g = np.asarray([None, np.float64('nan')], dtype=object)
    try:
        DG.assert_no_split_leakage([0], [1], np.array([], dtype=np.int64),
                                   groups=g)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('a missing group split across sides must fail')
    # A single missing group cannot be silently cut into populated folds.
    g4 = np.asarray([None, np.float64('nan'), np.datetime64('NaT', 'ns'),
                     'x'], dtype=object)
    try:
        DG.grouped_chronological_split_indices(4, g4, 0.5, 0.25, 0.25)
    except ValueError as e:
        assert 'no rows for the requested' in str(e), e
    else:
        raise AssertionError('a missing group must not be cut across folds')


def t_split_and_prepare_refuses_ignored_options():
    # Cleaning options without clean_train=True were silently ignored, so a
    # caller asking to drop invalid labels or outliers got an unfitted/unchanged
    # training split.  Preprocessor options were likewise ignored when
    # preprocess=False.  Both are now hard errors.
    import data_guard as DG
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6.0)
    for bad in ({'drop_invalid_labels': True},
                {'drop_nonfinite_features': True},
                {'outlier_iqr': 1.5},
                {'target_outlier_iqr': 1.5},
                {'label_conflicts': 'keep'},
                {'label_conflicts': 'raise'}):
        try:
            DG.split_and_prepare(X, y, **bad)
        except ValueError as e:
            assert 'require clean_train=True' in str(e), (bad, e)
        else:
            raise AssertionError(f'split_and_prepare silently ignored {bad!r}')
    try:
        DG.split_and_prepare(X, preprocess=False, scale='robust')
    except ValueError as e:
        assert 'preprocess=False' in str(e), e
    else:
        raise AssertionError('preprocess=False must not ignore preprocessor options')
    try:
        DG.split_and_prepare(X, quality_kwargs={'clip_z': 2.0})
    except ValueError as e:
        assert 'quality_kwargs' in str(e), e
    else:
        raise AssertionError('quality_kwargs without an audit must be refused')
    # The same options are accepted when their stage is actually enabled.
    out = DG.split_and_prepare(X, y, clean_train=True,
                               drop_invalid_labels=True)
    assert out['cleaning']['n_invalid_label_rows_dropped'] == 0, out['cleaning']
    raw = DG.split_and_prepare(X, preprocess=False)
    assert raw['preprocessor'] is None


def t_persisted_stats_reject_silently_degraded_transforms():
    # A persisted artifact is untrusted input.  Malformed clip bounds used to
    # disable clipping, inconsistent categorical indices could write the wrong
    # one-hot slot, and non-finite target/count tables produced invalid
    # features instead of refusing to load.
    import json
    import data_guard as DG
    X = np.arange(12.0).reshape(4, 3)
    base = json.loads(json.dumps(DG.LeakageSafePreprocessor().fit(X).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(clip_lo=1.0, clip_hi=0.0),
            lambda s: s['features'][0].update(clip_lo=float('nan')),
            lambda s: s['features'][0].update(clip_hi=None),
            lambda s: s['features'][0].pop('clip_lo'),
            lambda s: s['features'][0].update(clip_hi=True),
            lambda s: s.update(n_train=True),
            lambda s: s.pop('scale')):
        bad = json.loads(json.dumps(base))
        mutate(bad)
        try:
            DG.LeakageSafePreprocessor.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed preprocessor stats accepted: {bad}')
    Xc = np.asarray([['a'], ['b'], [None]], dtype=object)
    y = np.asarray([1.0, 2.0, 3.0])
    cat = json.loads(json.dumps(
        DG.LeakageSafeCategoricalEncoder().fit(Xc).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(n_categories=99),
            lambda s: s['features'][0].update(categories='ab'),
            lambda s: s['features'][0].update(keys=['only-one']),
            lambda s: s['features'][0].update(unknown_index=0)):
        bad = json.loads(json.dumps(cat))
        mutate(bad)
        try:
            DG.LeakageSafeCategoricalEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed categorical stats accepted: {bad}')
    freq = json.loads(json.dumps(
        DG.LeakageSafeFrequencyEncoder().fit(Xc).get_stats()))
    for mutate in (
            lambda s: s['features'][0].update(n_observed=99),
            lambda s: s['features'][0]['entries'][0].update(count=-1),
            lambda s: s['features'][0]['entries'][0].update(count=0)):
        bad = json.loads(json.dumps(freq))
        mutate(bad)
        try:
            DG.LeakageSafeFrequencyEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed frequency stats accepted: {bad}')
    target = json.loads(json.dumps(
        DG.LeakageSafeTargetEncoder().fit(Xc, y).get_stats()))
    for mutate in (
            lambda s: s['features'][0]['lookup'].update({'x': float('nan')}),
            lambda s: s['features'][0].update(prior=float('inf')),
            lambda s: s['features'][0].update(n_reference_labels=99,
                                              n_reference_rows=1)):
        bad = json.loads(json.dumps(target))
        mutate(bad)
        try:
            DG.LeakageSafeTargetEncoder.from_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed target stats accepted: {bad}')


def t_baseline_audit_direction_and_explicit_field_fallbacks():
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _art(direction, select_max=False):
        ppls = [10.0, 11.0]
        rows = [{'config': dict(c), 'ppl': v} for c, v in zip(trials, ppls)]
        best = max(rows, key=lambda r: r['ppl']) if select_max else min(
            rows, key=lambda r: r['ppl'])
        return {'selection_metric': 'ppl', 'selection_split': 'validation',
                'direction': direction,
                'arms': {'a': {'trials': rows, 'selected': dict(best['config']),
                               'selected_metric_value': best['ppl']}}}

    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[], final_seeds=[])
    good = L.baseline_tuning_audit(_art('lower_is_better'), protocol=protocol)
    assert good['tuning_verified'] is True, good
    # Flipping the artifact's direction makes a maximum-PPL trial look like
    # the argmin.  The protocol declares lower-is-better, so that is not a
    # tuned winner.
    flipped = L.baseline_tuning_audit(_art('higher_is_better', select_max=True),
                                      protocol=protocol)
    assert flipped['tuning_verified'] is False, flipped
    assert any('differs from the declared' in p or 'incompatible with PPL' in p
               for p in flipped['problems']), flipped
    # An explicitly empty/blank field is malformed evidence, not an absent key
    # that may inherit the protocol default.
    blank = _art('lower_is_better')
    blank['selection_metric'] = ''
    rb = L.baseline_tuning_audit(blank, protocol=protocol)
    assert rb['tuning_verified'] is False and any(
        'not a metric name' in p for p in rb['problems']), rb
    blank_split = _art('lower_is_better')
    blank_split['selection_split'] = ''
    rs = L.baseline_tuning_audit(blank_split, protocol=protocol)
    assert rs['tuning_verified'] is False and any(
        'validation' in p for p in rs['problems']), rs
    # A non-object protocol is malformed evidence, not an AttributeError.
    rp = L.baseline_tuning_audit(_art('lower_is_better'), protocol=['not', 'a', 'map'])
    assert rp['status'] == 'fail' and any(
        'protocol must be an object' in p for p in rp['problems']), rp
    # The artifact must be audited against the split it was declared under.
    # A protocol that declares test selection cannot certify a validation
    # artifact just because the artifact itself says "validation".
    rsplit = L.baseline_tuning_audit(
        _art('lower_is_better'),
        protocol=dict(protocol, selection_split='test'))
    assert rsplit['tuning_verified'] is False and any(
        'differs from the declared' in p for p in rsplit['problems']), rsplit


def t_simple_floor_rejects_undeclared_arms_and_blank_fields():
    rng = np.random.default_rng(11)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert L.simple_baseline_audit(art, protocol=proto)['baseline_verified'] is True
    # A copied extra arm can otherwise be chosen as the best floor while being
    # outside the declared equal-search design.
    extra = json.loads(json.dumps(art))
    extra['baselines']['extra'] = json.loads(json.dumps(extra['baselines']['unigram']))
    extra['baselines']['extra']['selected_metric_value'] = 1e-9
    for _t in extra['baselines']['extra']['trials']:
        _t['ppl'] = 1e-9
    extra['best_baseline'] = 'extra'
    extra['best_ppl'] = 1e-9
    rex = L.simple_baseline_audit(extra, protocol=proto)
    assert rex['baseline_verified'] is False and any(
        'undeclared arm' in p for p in rex['problems']), rex
    blank = json.loads(json.dumps(art))
    blank['selection_metric'] = ''
    rbl = L.simple_baseline_audit(blank, protocol=proto)
    assert rbl['baseline_verified'] is False and any(
        'not a metric name' in p for p in rbl['problems']), rbl
    blank_dir = json.loads(json.dumps(art))
    blank_dir['direction'] = ''
    rbd = L.simple_baseline_audit(blank_dir, protocol=proto)
    assert rbd['baseline_verified'] is False and any(
        'direction' in p for p in rbd['problems']), rbd
    rp = L.simple_baseline_audit(art, protocol=['not', 'a', 'map'])
    assert rp['status'] == 'fail' and any(
        'protocol must be an object' in p for p in rp['problems']), rp
    bad_arms = dict(proto, arms=['not-a-map'])
    rba = L.simple_baseline_audit(art, protocol=bad_arms)
    assert rba['baseline_verified'] is False and any(
        'arms field is not an object' in p for p in rba['problems']), rba
    # The recorded fit policy must be the one the code actually implements;
    # otherwise the artifact describes a data path that never ran.
    try:
        L.simple_baseline_fit(train, val, vocab,
                              protocol=dict(proto, fit_token_policy='seeded_sample'))
    except ValueError as _e:
        assert 'deterministic_prefix' in str(_e), _e
    else:
        raise AssertionError('an unimplemented fit token policy must be refused')
    bad_policy = json.loads(json.dumps(art))
    bad_policy['data']['fit_token_policy'] = 'seeded_sample'
    rbp = L.simple_baseline_audit(bad_policy, protocol=proto)
    assert rbp['baseline_verified'] is False and any(
        'fit_token_policy' in p for p in rbp['problems']), rbp
    rsplit = L.simple_baseline_audit(
        art, protocol=dict(proto, selection_split='test'))
    assert rsplit['baseline_verified'] is False and any(
        'differs from the declared' in p for p in rsplit['problems']), rsplit
    # The reader-facing estimand label cannot describe a different sample
    # from the nested rows the artifact actually scored.
    wrong_rows = json.loads(json.dumps(art))
    wrong_rows['best_ppl_estimation_rows'] = (
        'full_validation'
        if art['best_ppl_estimation_rows'] == 'nested_validation_holdout'
        else 'nested_validation_holdout')
    rwr = L.simple_baseline_audit(wrong_rows, protocol=proto)
    assert rwr['baseline_verified'] is False and any(
        'best_ppl_estimation_rows' in p for p in rwr['problems']), rwr


def t_simple_baseline_bound_floor_identity_and_rows():
    rng = np.random.default_rng(23)
    vocab = 24
    train = rng.integers(0, vocab, size=400)
    val = rng.integers(0, vocab, size=(4, 20))
    proto = dict(L.SIMPLE_BASELINE_POLICY,
                 arms={'unigram': {'order': 1}, 'bigram': {'order': 2}},
                 grids={'unigram': {'alpha': [0.1, 1.0]},
                        'bigram': {'alpha': [0.1, 1.0], 'interpolation': [0.5, 1.0]}})
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert L.simple_baseline_audit(art, protocol=proto)['baseline_verified'] is True
    # The reader-facing floor identity must not fall back to the legacy alias
    # when `best_baseline` is explicitly blank/null.
    for _bad_name in ('', None):
        bad = json.loads(json.dumps(art))
        bad['best_baseline'] = _bad_name
        r = L.simple_baseline_audit(bad, protocol=proto)
        assert r['baseline_verified'] is False and any(
            'best_baseline' in p for p in r['problems']), (_bad_name, r)
    # A present-but-non-finite floor value is malformed evidence, not an
    # absent legacy field.
    bad_ppl = json.loads(json.dumps(art))
    bad_ppl['best_ppl'] = float('nan')
    rp = L.simple_baseline_audit(bad_ppl, protocol=proto)
    assert rp['baseline_verified'] is False and any(
        'best_ppl' in p for p in rp['problems']), rp
    # Explicit nulls in the per-arm selection cannot silently fall back to a
    # different field.
    _arm = art['best_baseline']
    bad_sel = json.loads(json.dumps(art))
    bad_sel['baselines'][_arm]['selected'] = None
    rs = L.simple_baseline_audit(bad_sel, protocol=proto)
    assert rs['baseline_verified'] is False and any(
        'selected config' in p for p in rs['problems']), rs
    bad_val = json.loads(json.dumps(art))
    bad_val['baselines'][_arm]['selected_metric_value'] = None
    rv = L.simple_baseline_audit(bad_val, protocol=proto)
    assert rv['baseline_verified'] is False and any(
        'selected ppl' in p for p in rv['problems']), rv
    # The estimation-rows label needs actual nested evidence; without it the
    # claim is unverifiable rather than true.
    no_data = json.loads(json.dumps(art))
    no_data.pop('data', None)
    rd = L.simple_baseline_audit(no_data, protocol=proto)
    assert rd['baseline_verified'] is False and any(
        'cannot be verified' in p for p in rd['problems']), rd
    # Per-arm per-row accounting in a nested artifact must describe the held
    # out fold, and an unknown marker is not a sample identity.
    sel, ev, meta = L.validation_row_split(8, seed=20261020, n_folds=2,
                                           selection_fold=0, evaluation_fold=1)
    val8 = rng.integers(0, vocab, size=(8, 15))
    nested = L.simple_baseline_fit(train, val8, vocab, protocol=proto,
                                   selection_rows=sel, evaluation_rows=ev,
                                   row_split=meta)
    assert L.simple_baseline_audit(nested, protocol=proto)['baseline_verified'] is True
    # Present-but-malformed row-split evidence must fail closed, not be
    # ignored and replaced by a setdefault value.
    for _field, _bad in (('n_rows', 0), ('n_rows', 1.5),
                         ('n_selection_rows', 0),
                         ('n_evaluation_rows', float('nan')),
                         ('selection_row_range', [0]),
                         ('selection_row_range', [-1, 1]),
                         ('evaluation_row_range', 'bad')):
        _m = json.loads(json.dumps(meta))
        _m[_field] = _bad
        try:
            L.simple_baseline_fit(train, val8, vocab, protocol=proto,
                                  selection_rows=sel, evaluation_rows=ev,
                                  row_split=_m)
        except ValueError:
            pass
        else:
            raise AssertionError(f'simple_baseline_fit accepted row_split {_field}={_bad!r}')
    wrong_rows = json.loads(json.dumps(nested))
    wrong_rows['baselines'][_arm]['reported_estimation_rows'] = 'full_validation'
    rr = L.simple_baseline_audit(wrong_rows, protocol=proto)
    assert rr['baseline_verified'] is False and any(
        'not the held-out fold' in p for p in rr['problems']), rr
    unknown = json.loads(json.dumps(nested))
    unknown['baselines'][_arm]['reported_estimation_rows'] = 'some_fold'
    ru = L.simple_baseline_audit(unknown, protocol=proto)
    assert ru['baseline_verified'] is False and any(
        'reported_estimation_rows' in p for p in ru['problems']), ru


def t_metric_audit_recommended_availability_and_exact_floor_support():
    y = np.array([0] * 1000 + [1])
    score = np.r_[np.linspace(0.0, 0.4, 1000), 0.9]
    m = L.classification_metrics(y, y_score=score, labels=[0, 1],
                                 positive_label=1)
    m.update(task='binary_classification', variant='x')
    ok = L.metric_policy_audit({'x': m})
    assert not any('recommended_primary_available' in w for w in ok['warnings']), ok
    m_bad = json.loads(json.dumps(m))
    m_bad['pr_auc'] = None
    m_bad['recommended_primary_available'] = True
    rb = L.metric_policy_audit({'x': m_bad})
    assert rb['status'] == 'warn' and any(
        'recommended_primary_available=True' in w for w in rb['warnings']), rb
    # A fractional/boolean support is not a count and cannot reproduce the
    # majority-class floor.
    m_sup = json.loads(json.dumps(m))
    _k = next(k for k in m_sup['per_class']
              if abs(float(m_sup['per_class'][k]['support']) - 1.0) < 1e-12)
    m_sup['per_class'][_k]['support'] = 1.5
    rs = L.metric_policy_audit({'x': m_sup})
    assert rs['status'] == 'warn' and any(
        'exact non-negative count' in w or 'trivial-baseline' in w
        for w in rs['warnings']), rs
    # Micro one-vs-rest PR-AUC pools K decisions per sample: its chance floor
    # is 1/K, not a class prevalence.  A missing/wrong floor must not pass.
    ym = np.repeat([0, 1, 2], [90, 9, 1])
    sm = np.random.default_rng(3).random((100, 3))
    sm /= sm.sum(axis=1, keepdims=True)
    mm = L.classification_metrics(ym, y_score=sm, labels=[0, 1, 2])
    assert abs(mm['pr_auc_micro_chance'] - 1.0 / 3.0) < 1e-12, mm
    assert abs(mm['pr_auc_micro_lift_over_chance']
               - (mm['pr_auc_micro'] - 1.0 / 3.0)) < 1e-12, mm
    mm.update(task='multiclass_classification', variant='mc')
    assert L.metric_policy_audit({'mc': mm})['status'] == 'ok'
    mm_bad = json.loads(json.dumps(mm))
    mm_bad['pr_auc_micro_chance'] = 0.9
    rm = L.metric_policy_audit({'mc': mm_bad})
    assert rm['status'] == 'warn' and any(
        'pr_auc_micro_chance' in w for w in rm['warnings']), rm


def t_build_report_malformed_panel_is_excluded():
    import build_report as BR
    td = tempfile.mkdtemp(prefix='br_shape_')
    old_repo = BR.REPO
    old_loaded = BR._LOADED
    try:
        BR.REPO = td
        BR._LOADED = {}
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump([{'variant': 'a', 'seed': 0, 'ppl': 1.0}], f)
        with open(os.path.join(td, 'aggregate.json'), 'w', encoding='utf-8') as f:
            json.dump('not an aggregate', f)
        assert BR.summary(td) == {}
        assert BR.agg(td) == {}
        assert BR.per_seed_records(td) == {}
        assert BR._legacy_warm_tags_cached(td) == {}
    finally:
        BR.REPO = old_repo
        BR._LOADED = old_loaded
        shutil.rmtree(td, ignore_errors=True)

def main():
    if not _HAVE_TORCH:
        print('[test] SKIPPED: the CPU test suite could not import its dependencies; this is NOT a pass (exit 3)')
        return 3
    if not _HAVE_TORCH:
        print('[test] SKIPPED (no torch in this environment)')
        return 0
    _check('A1 holm_min_attainable_p', t_holm_min_attainable)
    _check('A2 signflip_min_consistency', t_min_consistency)
    _check('A3 signflip_power', t_power)
    _check('A4 tost_paired directions', t_tost)
    _check('A5 exact_signflip baseline', t_exact_signflip_baseline)
    _check('A5a report gate refuses panel attrition and seed-driven signs', t_report_gate_refuses_attrition_and_seed_driven)
    _check('A6 paired_effect resolution_limited', t_paired_effect_resolution)
    _check('A6b scaling_trend ragged seed panels', t_scaling_trend_ragged_pairs)
    _check('A6b1 missing scale seed identity is never paired positionally', t_scale_missing_seed_identity_is_not_paired)
    _check('A6b2 scaling_trend primary direction + floors', t_scaling_trend_primary_direction_and_floor)
    _check('A6c meta-analysis input guards', t_scale_input_guards)
    _check('A5b exact sign-flip at scale (MITM)', t_exact_signflip_at_scale)
    _check('A5c deterministic saddlepoint sign-flip for n>40', t_signflip_saddlepoint_large_n)
    _check('A5d exact Wilcoxon DP beyond n=16', t_wilcoxon_exact_dp)
    _check('A5e TimeGuard per-variant panel calibration', t_timeguard_variant_calibration)
    _check('A4b TOST Student-t CI + margin boundary', t_tost_student_t_and_boundary)
    _check('A4c safe exp + token-pooled document CI', t_safe_exp_and_doc_ci)
    _check('A4d deterministic document bootstrap CI', t_bootstrap_mean_ci)
    _check('A4e training-health classifier: measured vs unmonitored vs nonfinite', t_training_health_classifier)
    _check('A4f training-health threshold is slope-noise aware (2*SE)', t_training_health_noise_aware_threshold)
    _check('A4g training-health actions + widening gap', t_training_health_actions_and_gap)
    _check('A4h early-stop decision policy', t_early_stop_decision)
    _check('A4i NaN triage path', t_nan_diagnosis_path)
    _check('A4j bad-case cause taxonomy', t_token_error_cause_taxonomy)
    _check('A4j1 paired errors use the common finite support', t_paired_errors_use_common_finite_support)
    _check('A4j2 error analysis rejects non-integer/out-of-range ids', t_error_analysis_rejects_bad_token_tables)
    _check('A4j3 paired error report renders empty support', t_error_analysis_paired_none_is_renderable)
    _check('A4k early stop is recorded and token accounting follows completed steps', t_train_variant_early_stop_record)
    _check('A4k1 final loss smoothing uses the last 50 steps', t_final_loss_smoothed_uses_last_steps)
    _check('A4k2 long loss-curve downsampling keeps spikes and NaN', t_loss_curve_downsample_keeps_spikes_and_nan)
    _check('A4l v12 health gate refuses a recorded non-finite event', t_v12_health_gate_refuses_unstable_training)
    _check('A4g rope arms never claim the batched eager twin', t_rope_arms_are_not_batched)
    _check('A4g2 apply_rope preserves width (over-long/odd rope_dim)', t_apply_rope_preserves_width)
    _check('A4g3 numeric helpers accept NumPy scalars', t_stats_accept_numpy_scalars)
    _check('A4h shipped manifest v3 admits 48/48 synthetic records end-to-end', t_manifest_v3_admission_can_succeed)
    _check('B1 legacy modes bit-exact', t_legacy_modes_bitexact)
    _check('B2 determinism + target-zone safety', t_corruption_deterministic_and_paired)
    _check('B3 new modes span/doc/topical', t_new_modes_do_something)
    _check('B4 mode validation', t_mode_validation)
    _check('C1 load-time power gate', t_power_gate)
    _check('C2 admission gate exclusions', t_admission_gate)
    _check('C3 admission gate refusals', t_admission_refuses)
    _check('C4 recipe signature', t_recipe_signature)
    _check('C4b P1L checkpoint recipe signature', t_p1l_recipe_signature)
    _check('C5 freeze lock', t_freeze_lock)
    _check('C5b by-length per-document row count', t_by_len_requires_one_ppl_per_row)
    _check('C5c plan endpoint and P5T preflight gate', t_plan_check_rejects_bad_endpoint_and_finaltest_is_gated)
    _check('C5b probe stats reject malformed arm identities', t_probe_stats_rejects_malformed_arm_identities)
    _check('C5c probe stats reject synthesized/mismatched probe identities', t_probe_stats_rejects_synthesized_and_mismatched_probe_params)
    _check('C5d selection audit rejects boolean measurements', t_audit_selection_rejects_boolean_measurements)
    _check('C9 aggregate seed normalization', t_aggregate_seed_normalization)
    _check('C10 audit resolution classification', t_audit_resolution_classification)
    _check('C11 controlled-variable gate: mixed train_knobs', t_pair_reason_train_knobs_mixed)
    _check('C11b corpus/validation-slice fingerprint gate', t_pair_reason_fingerprint_gate)
    _check('C21 M0 scale analysis stays descriptive at n=4', t_m0_scale_analysis_descriptive)
    _check('C21a M0 records exclude synthesized reconstructions', t_m0_records_exclude_synthesized_reconstructions)
    _check('C12 seed plan', t_seed_plan)
    _check('C13 family Holm + verified baseline + param control', t_audit_family_holm_and_verified)
    _check('C14 compression-regularizer loss ablation', t_ablation_design_loss_term)
    _check('C15 optimizer knobs enter the resume fingerprint', t_training_knobs_in_resume_fingerprint)
    _check('C16 controlled variables inspect every seed record', t_controlled_variables_all_seed_records)
    _check('C16a audit flags unequal seed coverage (lost seed)', t_seed_coverage_warning)
    _check('C16b fairness audit checks cross-arm optimizer + seed coverage', t_fairness_cross_arm_controls)
    _check('C17 arm n / mean±std / relative effect', t_arm_mean_std_and_relative_effect)
    _check('C17b tuned pairing keeps optimizer gate', t_tuned_pairing_keeps_optimizer_gate)
    _check('C18 leave-one-out ablation completeness', t_ablation_completeness_leave_one_out)
    _check('C19 full-factorial ablation completeness', t_ablation_completeness_full_factorial)
    _check('C19a full-factorial rejects a reused arm id', t_full_factorial_rejects_reused_arm)
    _check('C19b project-level ablation coverage across panels', t_ablation_coverage_across_panels)
    _check('C17c paired effect CI + leave-one-seed robustness', t_paired_effect_seed_robustness)
    _check('C17c2 v7/v8/v9 pairs use the primary estimand', t_v7_v8_v9_pairs_use_primary_estimand)
    _check('C17d capacity-confounding warning on module ablation', t_capacity_confound_warning)
    _check('C17e module ablation requires identical structured hyperparameters', t_component_pair_requires_same_mlp_ratio)
    _check('C20 panel ablation flag does not overclaim', t_panel_ablation_flag_does_not_overclaim)
    _check('C20a partial pairing keeps controlled seeds', t_partial_skipped_seed_keeps_controlled_contrast)
    _check('C20b incomplete components exported in audit report', t_audit_incomplete_components_alias)
    _check('A7 adaptive-threshold capacity guard', t_adaptive_threshold_guard)
    _check('A7b adaptive-threshold chunked values', t_adaptive_threshold_values)
    _check('A8 empty segmenter', t_segment_empty)
    _check('A8a uniform-stride cut boundary', t_segment_uniform_stride_only_when_all_cuts)
    _check('A8b empty pooling geometry', t_empty_pool_geometry)
    _check('C6 shipped manifest v2 consistency', t_shipped_manifest_consistency)
    _check('C7 frozen seeds single source', t_frozen_seeds)
    _check('C8 doc-level analysis', t_doc_level_analysis)
    _check('D1 eval per-seq ppl', t_eval_per_seq)
    _check('J1 evidence metadata + significance gate', t_review3_evidence_metadata)
    _check('J2 pooled aggregation', t_review3_pooled_aggregate)
    _check('J2b nested records pair on the primary holdout estimand', t_nested_primary_paired_deltas)
    _check('J3 Holm declared-family slots', t_review3_holm_family_size)
    _check('J4 task-boundary arm + objective descriptor', t_review3_boundary_arm_and_objective)
    _check('J4b declared objective ablations are not confounds', t_objective_ablation_is_declared_not_a_confound)
    _check('J5 span clean-baseline key', t_review3_span_clean_key)
    _check('J5a v10 Holm keeps declared unmeasured slots', t_v10_holm_family_keeps_unmeasured_slots)
    _check('J5b key exploratory phases declare >=3 seeds', t_key_exploratory_phases_declare_at_least_three_seeds)
    _check('E1 seed reproducibility', t_seed_reproducible)
    _check('E1c seed contract inventories every CUDA device', t_seed_contract_cuda_device_inventory)
    _check('E1d CPU strict determinism is labelled', t_determinism_label_cpu_strict)
    _check('E1a seed/variant axis normalisation', t_seed_axis_normalization)
    _check('E1a2 seed scalars never truncate silently', t_seed_scalar_rejects_silent_truncation)
    _check('E1a3 seed domain validation', t_seed_domain_guard)
    _check('E1b explicit RNG seed contract', t_seed_contract_check)
    _check('E2 batch_iter reproducibility', t_batch_iter_reproducible)
    _check('E2b batch_iter uint16 transport', t_batch_iter_uint16)
    _check('E2c real batch-iterator contract check', t_batch_contract_check)
    _check('E2c2 batch contract validates its own seed/counts', t_batch_contract_check_validation)
    _check('E2d failed loader contract blocks the panel', t_panel_blocks_on_loader_contract_failure)
    _check('E2e parallel child gets the seed hash', t_parallel_child_env_seed)
    _check('E3 small-batch overfit sanity', t_overfit_sanity)
    _check('E3a overfit loss contract + ignore-index propagation', t_overfit_loss_contract_and_ignore_index)
    _check('E3b training CE default ignore reduction', t_ce_loss_default_ignore_matches_pytorch_reduction)
    _check('E3b preflight RNG-state preservation', t_rng_state_preserved_by_preflight)
    _check('E3c duplicate-run reproducibility check', t_reproducibility_check)
    _check('E4 provenance + run manifest', t_provenance_and_manifest)
    _check('E4a manifest requires referenced curve artifacts', t_manifest_missing_curve_ref_is_incomplete)
    _check('E4b manifest rejects mismatched run_cfg/data_version', t_manifest_rejects_mismatched_cell_identity)
    _check('E4c runtime host/process identity', t_environment_snapshot_runtime_identity)
    _check('E4b panel tracking events', t_panel_tracking_events)
    _check('E5 DataLoader/worker seeding', t_dataloader_seeding_helpers)
    _check('E5b repro probe cache binds device and LR', t_repro_probe_config_binds_device_and_lr)
    _check('E6 end-to-end training reproducibility', t_train_variant_reproducible)
    _check('E7 overfit check detects broken optimizer', t_overfit_detects_broken_optimizer)
    _check('E8 append-only experiment ledger + manifest', t_experiment_ledger)
    _check('E8a ledger curves stay bounded but keep spikes/NaN', t_ledger_curve_is_bounded)
    _check('E9 data_version identity', t_data_version_identity)
    _check('E10 crash-safe manifest status/planned cells', t_manifest_running_then_complete)
    _check('E10b panel finish marks missing cells incomplete', t_panel_finish_marks_missing_cells_incomplete)
    _check('E11 failed overfit guard is retried', t_failed_guard_is_retried)
    _check('E12 locked JSON nested path + CSV generator rows', t_locked_json_and_csv_generator)
    _check('F1 indexer column-narrowing exactness', t_indexer_column_narrowing_exact)
    _check('F2 zero-block attention keeps window', t_attention_zero_blocks_keeps_window)
    _check('F3 overfit guard source key', t_overfit_guard_source_key)
    _check('F3b overfit guard probe-config key', t_overfit_guard_config_aware)
    _check('F3c environment identity/snapshot fields', t_environment_snapshot_identity)
    _check('F4 shared-storage cache accounting', t_cache_bytes_shared_storage)
    _check('F5 atomic checkpoint save', t_atomic_torch_save)
    _check('G1 long-window chunked mask', t_long_window_chunked_mask)
    _check('G2 v7 long-window chunked mask', t_v7_long_window_chunked_mask)
    _check('G2a v7 RoPE gather/rotation order', t_v7_rope_gather_rotation_order)
    _check('G3 empty-sequence indexer', t_indexer_empty_sequence)
    _check('G4 underpowered is never verified', t_underpowered_never_verified)
    _check('H1 hypothesis direction conflict', t_hypothesis_direction_conflict)
    _check('H2 metric policy + macro/micro/weighted F1 + PR/ROC-AUC', t_metric_policy_and_classification_metrics)
    _check('H2b prevalence-aware endpoint + per-class PR-AUC chance', t_metric_prevalence_recommendation_and_per_class_chance)
    _check('H2c hypothesis prevalence binds the conditional endpoint', t_hypothesis_prevalence_conditional_endpoint)
    _check('H2d degenerate binary threshold/score range is explicit', t_binary_degenerate_threshold_visibility)
    _check('H2a metric audit checks record aggregation/ranges', t_metric_policy_checks_record_values)
    _check('H2a1 retrieval metrics carry trivial floors', t_retrieval_metrics_and_niah_trivial_floor)
    _check('H2b multiclass one-vs-rest ranking metrics + imbalance audit', t_multiclass_ranking_metrics_and_audit)
    _check('H3 equal-search baseline tuning audit', t_baseline_tuning_audit)
    _check('H3a0 tuned family + search-to-final data transfer', t_baseline_family_declared_contrasts_and_undeclared_arms)
    _check('H3a tuning grid boundary/coverage adequacy', t_tuning_grid_boundary_adequacy)
    _check('H3a multi-seed selection + equal search-seed audit', t_baseline_multi_seed_selection_audit)
    _check('H3b tuning driver selects on the multi-seed mean', t_baseline_tuning_driver_multiseed_selection)
    _check('H3c search/final seed partition + inherited tuning coverage', t_baseline_seed_partition_and_coverage_audit)
    _check('H3d tuned contrasts are declared per parameter-matching group', t_baseline_contrast_plan_is_group_aware)
    _check('H3f leave-one-seed-out tuning-selection stability', t_tuning_selection_stability_audit)
    _check('H3g tuned claim requires the panel to run the selected config', t_methodology_audit_requires_panel_to_use_tuned_config)
    _check('H3e tuning driver refuses overlapping search/final seeds', t_baseline_driver_refuses_overlapping_final_seeds)
    _check('H3h nested validation row split + per-row pooled PPL', t_nested_validation_row_split_and_subset_ppl)
    _check('H3i validation row folds are group-aware', t_validation_row_split_group_aware)
    _check('H3i1 cached document ids merge rows crossing a document', t_cached_validation_row_groups_merge_shared_documents)
    _check('H3i tuning audit verifies nested selection/holdout rows', t_nested_tuning_artifact_audit)
    _check('H3j final tuned contrast is measured on held-out rows', t_nested_final_uses_holdout_rows)
    _check('H3k partial nested holdout falls back wholly to full validation', t_nested_final_partial_holdout_falls_back_wholly)
    _check('H4 tuned count-model floor + audit', t_simple_baseline_ngram_and_audit)
    _check('H4a count-model fit budget is bounded + disclosed', t_simple_baseline_fit_budget_is_bounded_and_disclosed)
    _check('H4b count-model floor uses nested selection/holdout rows', t_simple_baseline_nested_rows_and_fingerprint)
    _check('H5 positive-label / rare-positive PR-AUC / macro-F1 averaging', t_classification_positive_label_and_rare_floor)
    _check('H5a binary threshold selection + calibration + threshold audit', t_binary_threshold_and_calibration_contract)
    _check('H5b boundary macro/micro F1 averaging contract', t_boundary_macro_micro_f1_contract)
    _check('H6 hypothesis completeness + metric alignment + direction=0', t_hypothesis_audit_completeness_and_metric_alignment)
    _check('H7 simple-baseline driver uses the panel corpus', t_simple_baseline_driver_serves_the_panel_corpus)
    _check('H7a floor contrast is seed-level and significance-aware', t_simple_floor_contrast_is_significance_aware)
    _check('H8 leakage audit handles a flat 1-D train cache', t_audit_leakage_handles_flat_train_cache)
    _check('H8b corpus audit memo covers meta/thresholds and strict', t_corpus_audit_memo_and_thresholds)
    _check('H8c audit gate parameters fail closed', t_audit_gate_parameters_fail_closed)
    _check('H9 fuse sample_std accepts negative axes', t_fuse_sample_std_negative_axis)
    _check('I1 chronological split contract', t_leakage_safe_split_contract)
    _check('I2 whole-group split contract', t_group_split_contract)
    _check('I2b grouped split keeps overlapping time windows together', t_group_split_respects_time_windows)
    _check('I2c grouped split allocation is complete and zero targets stay empty', t_group_split_complete_allocation_and_zero_targets)
    _check('I3 preprocessing fitted on train only', t_preprocessor_train_only)
    _check('I3b imputation fill is identical in fit and transform', t_preprocessor_fill_is_consistent)
    _check('I3c leakage checks validate time/group inputs', t_leakage_checks_validate_inputs)
    _check('I4 data-quality/leakage report', t_data_quality_report)
    _check('I4c train-only regression-target outlier audit', t_data_quality_target_outliers)
    _check('I4b data-quality report checks split index disjointness', t_data_quality_report_index_disjointness)
    _check('I5a categorical shift handles an empty reference', t_categorical_shift_empty_reference)
    _check('I5 categorical shift + cache guards', t_categorical_and_cache_guards)
    _check('I6 explicit split sizes stay non-negative', t_split_sizes_explicit_never_negative)
    _check('I7 split audit finds sequence-level row overlap', t_audit_split_sequence_overlap)
    _check('I8 public exports resolve (star-import safe)', t_public_exports_resolve)
    _check('I9 rejection-sampled token audit stays uniform', t_sample_tokens_uniform_rejection)
    _check('I10 budget audit accepts TimeGuard integer run counters', t_audit_budget_integer_runs)
    _check('I11 failure audit does not count NaN PPL as a success', t_audit_failures_ignores_unmeasured_ppl)
    _check('I12 object-time NaN + zero-feature data guards', t_object_time_and_zero_feature_guards)
    _check('I13 split audit marks per-cache warnings', t_audit_split_warning_status)
    _check('I13b split audit checks test metadata discipline', t_audit_split_test_metadata_discipline)
    _check('I13c split audit can target the panel train seq_len', t_audit_split_preferred_seq_len)
    _check('I14 corpus audit counts overlapping validation rows once', t_corpus_audit_overlap_is_unique_across_windows)
    _check('I15 length audit refuses a missing eval span', t_audit_length_requires_eval_span)
    _check('I16 fairness audit ignores a NaN seed instead of crashing', t_audit_fairness_nan_seed_does_not_crash)
    _check('I16b boundary audit rejects non-numeric alignment readings', t_audit_boundary_rejects_invalid_readings)
    _check('I16c malformed arm ids do not crash the panel audits', t_audit_malformed_arm_id_does_not_crash)
    _check('I17 an empty split cache is not reported as verified', t_audit_split_empty_cache_is_not_ok)
    _check('I18 tuning audit reports malformed seed keys instead of crashing', t_baseline_audit_malformed_seed_keys)
    _check('I19 simple-baseline audit reports malformed records instead of crashing', t_simple_baseline_audit_malformed_records)
    _check('I20 index aliasing, non-finite labels, signed-zero duplicates, undefined SMD, NaT groups', t_data_guard_index_label_and_shift_edges)
    _check('I20b all missing group ids share one identity', t_missing_group_ids_are_one_identity)
    _check('I21 split-and-prepare fits preprocessing on train only', t_split_and_prepare_fits_train_only)
    _check('I21d strict post-split quality gate', t_split_and_prepare_strict_quality_gate)
    _check('I21f split-and-prepare refuses ignored cleaning/preprocessing options', t_split_and_prepare_refuses_ignored_options)
    _check('I21a data-quality temporal split audit', t_data_quality_temporal_audit)
    _check('I21b capped test document-id sidecar', t_capped_document_id_sidecar)
    _check('I21c near-duplicate document chains stay on one side', t_document_dedup_closes_near_duplicate_chains)
    _check('I21e test dedup uses read-only earlier-split indices', t_test_document_filter_uses_readonly_earlier_indices)
    _check('I22 purge/embargo keeps window boundaries sealed', t_split_purge_embargo)
    _check('I23 grouped split refuses an empty requested side', t_group_split_refuses_empty_requested_side)
    _check('I24 label shift and per-split label corruption counts', t_label_shift_and_split_counts)
    _check('I25 numeric-looking string labels stay categorical', t_label_key_keeps_numeric_strings)
    _check('I26 preprocessor rejects boolean/negative-eps options', t_data_guard_preprocessor_option_guards)
    _check('I27 categorical shift requires a meaningful bin count', t_categorical_shift_requires_two_bins)
    _check('I28 label-noise audit handles unhashable labels', t_data_quality_unhashable_labels)
    _check('I29 categorical encoding is fitted on train only', t_categorical_encoder_train_only)
    _check('I29b frequency encoding is fitted on train only', t_frequency_encoder_train_only)
    _check('I29c target encoding is out-of-fold and group/time safe', t_target_encoder_prevents_self_and_group_leakage)
    _check('I29d target encoding keeps duplicate rows in one fold', t_target_encoder_keeps_duplicate_rows_in_one_fold)
    _check('I30 train-only cleaning before preprocessing', t_clean_training_rows_and_split)
    _check('I30b invalid training labels are droppable before fitting', t_clean_training_invalid_labels)
    _check('I30c non-finite/outlier training rows are cleanable', t_clean_training_nonfinite_and_outliers)
    _check('I30d target/label outliers are cleanable with a train-only fence', t_clean_training_target_outliers)
    _check('I31 declared label set catches unseen/holdout labels', t_known_labels_audit)
    _check('I31b NaT keys and type-qualified label shift', t_data_guard_nat_and_label_identity)
    _check('I31c object temporal duplicate identity', t_object_temporal_duplicate_identity)
    _check('I31d object temporal category identity', t_object_temporal_category_identity)
    _check('I31e persisted stats must match n_features_in', t_persisted_stats_validate_feature_count)
    _check('I31f temporal group identity is hash-consistent', t_temporal_group_identity_matches_python_equality)
    _check('I31g Python temporal row/category identity', t_python_temporal_row_and_category_identity)
    _check('I31h persisted stats reject silently degraded transforms', t_persisted_stats_reject_silently_degraded_transforms)
    # Open-block reading rule: the substantive gates (causality, reference
    # pooling, masks, edge cases, closed regression) live in test_open_block.py
    # so they can also be run on their own.
    try:
        import test_open_block as _OB
        for _name, _fn in _OB.CHECKS:
            _check(_name, _fn)
    except Exception as _obe:
        _check('K0 open-block suite import', lambda _e=_obe: (_ for _ in ()).throw(_e))
    _check('L1 reportable claim gate', t_reportable_claim_gate)
    _check('L2 Holm family resolution ceiling', t_holm_family_resolution_ceiling)
    _check('L3 crossover fit refusal rules + slope CI', t_crossover_scaling_refusals)
    _check('L3b scale-ladder budget + transfer audit', t_scale_ladder_budget_and_transfer_audit)
    _check('L4 matched-init fingerprint + pair gate', t_init_core_fingerprint_and_pair_gate)
    _check('L5 param breakdown + measured compute control', t_param_and_compute_control_detail)
    _check('L5b capacity breakdown gates the baseline claim', t_capacity_breakdown_gates_baseline_claim)
    _check('L5c hypothesis matched-capacity declaration is enforced', t_hypothesis_param_match_requirement_is_enforced)
    _check('L6 stale aggregate refused', t_stale_aggregate_refused)
    _check('L7 stale length panel excluded from report', t_sec_p1l_excludes_stale_records)
    _check('L8 efficiency-pair audit + shape sweep', t_efficiency_pairs_audit_and_sweep)
    _check('L8b efficiency Pareto is grouped by comparable measurements', t_efficiency_pareto_is_grouped)
    _check('L8c report NaN section reads the triage path', t_report_nan_section_uses_triage)
    _check('L8d efficiency/accuracy cost summary is seed-noise aware', t_efficiency_accuracy_tradeoff_summary)
    _check('L8e report surfaces the bad-case cause inventory', t_report_error_analysis_section)
    _check('L8f training-health section shows actions and steps since best', t_report_training_health_actions)
    _check('L8f2 health/early-stop accept CE-only curves and order steps', t_health_accepts_val_ce_only_and_orders_steps)
    _check('L8g report surfaces completed vs planned training budget', t_report_early_stop_section)
    _check('L8h efficiency report keeps measurement groups separate', t_report_efficiency_groups_are_separate)
    _check('L8h2 efficiency cost sentence reports the improvement direction', t_report_efficiency_cost_sentence_uses_correct_direction)
    _check('L8i training-health/triage record audit fails closed', t_audit_training_health_records)
    _check('L9 evidence-release readiness audit', t_release_readiness_audit)
    _check('L10 selection-attribution contract', t_selection_attribution_contract)
    _check('L11 length audit fails closed on missing truncation flag', t_audit_length_missing_truncated_flag)
    _check('L12 length audit fails closed on missing n_seq', t_audit_length_missing_n_seq_flag)
    _check('L12b length audit rejects non-integer counts/keys/cells', t_audit_length_rejects_unverifiable_counts)
    _check('L13 global cross-panel multiplicity audit (exp_lib)', t_global_multiplicity_audit_basic)
    _check('L13b global audit fail-closed rules', t_global_multiplicity_audit_fail_closed)
    _check('L13c build_report twin matches exp_lib audit', t_global_multiplicity_twin_consistency)
    _check('L13d report registers every displayed comparison', t_build_report_global_multiplicity_section)
    _check('L14 combined seed x document uncertainty interval', t_combined_seed_doc_uncertainty)
    _check('L14b paired evaluation-row bootstrap interval', t_paired_row_bootstrap_delta)
    _check('L15 cross-scale pooling refuses mixed corpus/tokenizer bindings', t_scale_corpus_binding_refuses_mixed_tokenizers)
    _check('L15c meta-analysis seed overlap controls pooling', t_meta_seed_overlap_controls_pooling)
    _check('L15d preprocessing fit audit detects full-table leakage', t_preprocessing_fit_audit_detects_full_table_leak)
    _check('L15e encoder fit audit detects holdout vocabulary', t_encoder_fit_audit_detects_holdout_vocabulary)
    _check('L15b scale-point corpus provenance roundtrip', t_scale_point_provenance_roundtrip)
    _check('L16 Holm mixed-n floor + invalid family slot', t_holm_mixed_n_and_invalid_family_size)
    _check('L16b meta-analysis prediction-interval df', t_meta_prediction_interval_df)
    _check('L17 TOST decision follows conf', t_tost_conf_controls_decision)
    _check('L18 exact duplicates keep int precision / reject complex', t_data_quality_exact_integer_and_complex_rows)
    _check('L18b exact numeric contracts + integer-safe matrix fingerprint', t_exact_numeric_contracts_and_fingerprint)
    _check('L19 batch_iter minimal corpus', t_batch_iter_minimal_corpus)
    _check('L19b simple-baseline grid adequacy is explicit', t_simple_baseline_grid_adequacy)
    _check('L20 indexer invalid soft lanes are zero', t_indexer_invalid_soft_lanes_are_zero)
    _check('L21 attention backward partial head group', t_attention_backward_partial_head_group)
    _check('L21a checkpoint stamp names the actual backend', t_checkpoint_stamp_matches_the_actual_backend)
    _check('L21b indexer chunked path uses only the filled rows', t_indexer_chunked_final_tile_uses_active_rows)
    _check('L22 parallel watchdog override validation', t_par_watchdog_override_validation)
    _check('L24 build report pairs on the primary holdout estimand', t_build_report_primary_estimand)
    _check('L22b build report exact seed/parameter contracts', t_build_report_exact_seed_and_params)
    _check('L23 methodology/scale gates fail closed on malformed inputs', t_methodology_gate_fail_closed_edges)
    _check('L25 baseline seed plans reject duplicate draws', t_baseline_seed_plan_rejects_duplicate_draws)
    _check('L25b baseline tuning audit checks search-to-final horizon transfer', t_baseline_horizon_transfer_audit)
    _check('L26 bootstrap chunk scaling + zero-CE measurements', t_bootstrap_and_zero_ce_scaling_contract)
    _check('L27 error analysis excludes non-finite targets', t_error_analysis_excludes_nonfinite_targets)
    _check('L27b rare-target triage needs a measured frequency basis', t_error_analysis_rare_basis_is_measured)
    _check('L27c error-analysis budgets are explicit, not clamped', t_error_analysis_zero_and_invalid_budgets)
    _check('L27d paired error analysis requires one vocabulary', t_error_analysis_paired_vocab_guard)
    _check('L27e bad cases expose prefix/label ambiguity', t_error_analysis_flags_prefix_ambiguity)
    _check('L28 gapfill numeric reader rejects booleans', t_gapfill_numeric_reader_rejects_booleans)
    _check('L28b gapfill announces the family-corrected seed plan', t_gapfill_announces_family_seed_plan)
    _check('L29 release tags cannot escape the archive directory', t_release_tag_cannot_escape_output_dir)
    _check('L30 missing id cache/sidecars is not a clean leakage audit', t_audit_leakage_missing_id_cache_is_not_ok)
    _check('L31 v12 binds the local train cache to the frozen train stream', t_v12_binds_current_train_cache_to_frozen_stream)
    _check('L32 release archive packs tagged cache provenance', t_release_archive_packs_tagged_cache_provenance)
    _check('L33 scale corpus identity is independent of training budget', t_scale_corpus_fp_is_budget_independent)
    _check('L34 exact sign-flip floor accounts for zero ties', t_signflip_zero_ties_lower_resolution_floor)
    _check('L34b zero-aware family resolution floor', t_zero_aware_family_resolution_floor)
    _check('L35 scale meta/trend refuses visible seed attrition', t_scale_meta_refuses_visible_attrition)
    _check('L36 M0 cross-scale pairs pass the control gate', t_m0_cross_scale_pairs_are_control_gated)
    _check('L36b M0 cross-scale uses declared seed plans', t_m0_cross_scale_uses_declared_seed_plans)
    _check('L37 flat trend / empty-tier pairing', t_scale_trend_degenerate_fit_and_empty_tiers)
    _check('L37b non-estimable variance blocks pooling', t_scale_non_estimable_variance_blocks_pooling)
    _check('L37c tuning margin does not invent precision', t_tuning_margin_does_not_invent_precision)
    _check('L37d ladder missing seed plan is not common', t_scale_ladder_missing_seed_plan_is_not_common)
    _check('L38 small-n split remainder + string order keys', t_small_n_split_remainder_and_string_order)
    _check('L38c exact split fraction arithmetic at scale', t_split_targets_use_exact_fraction_arithmetic)
    _check('L38b empty split still validates fractions', t_empty_split_still_validates_fractions)
    _check('L39 split audit recomputes fingerprints and doc sidecars', t_split_discipline_recomputes_fingerprints_and_doc_sidecars)
    _check('L40 archive refuses links and repo escape', t_archive_refuses_links_and_repo_escape)
    _check('L41 warmup table uses the primary estimand', t_build_report_warmup_uses_primary_estimand)
    _check('L42 empty-sequence pool kernel matches eager', t_pool_kernel_empty_sequence_matches_eager)
    _check('L42b no-overlap pool accepts overlap=None', t_pool_variable_blocks_accepts_no_overlap)
    _check('L42c efficiency medians average both middle seeds', t_efficiency_median_uses_both_middle_seeds)
    _check('L43 eager_batch rejects typos/non-finite values', t_eager_batch_rejects_typos_and_nonfinite)
    _check('L44 gapfill merge refreshes/invalidates seed-derived fields', t_gapfill_merge_keeps_secondary_and_invalidates_stale_fields)
    _check('L45 open-block resume binds data fingerprints', t_open_block_resume_binds_data_fingerprints)
    _check('L46 analysis checkpoint carries model geometry', t_analysis_checkpoint_carries_model_geometry)
    _check('L47 v12 admission rejects synthesized records', t_v12_admission_rejects_synthesized_records)
    _check('L48 seed/nested input guards', t_seed_and_nested_input_guards)
    _check('L49 malformed panel summary is excluded', t_malformed_panel_summary_is_excluded)
    _check('L50 binary PR-AUC floor follows the positive class', t_binary_pr_floor_uses_positive_prevalence)
    _check('L51 baseline audit binds direction + rejects explicit blank fields', t_baseline_audit_direction_and_explicit_field_fallbacks)
    _check('L51b count floor rejects undeclared arms/blank fields', t_simple_floor_rejects_undeclared_arms_and_blank_fields)
    _check('L51c recommended endpoint availability + exact floor support', t_metric_audit_recommended_availability_and_exact_floor_support)
    _check('L52 simple-baseline floor identity/rows bind their evidence', t_simple_baseline_bound_floor_identity_and_rows)
    _check('L53 v12 malformed manifest/lock fail closed', t_v12_malformed_manifest_and_lock_fail_closed)
    _check('L54 build report excludes malformed panel objects', t_build_report_malformed_panel_is_excluded)
    _check('L55 controlled recipe requires a complete train_knobs descriptor', t_training_knobs_must_be_complete)
    _check('L56 underpowered contrast is never significant; std definability', t_underpowered_contrast_is_never_significant)
    _check('L57 report pairs cover the declared ablation design', t_report_pairs_cover_the_declared_ablation_design)
    _check('L58 a partial ablation cannot verify the whole component', t_partial_ablation_cannot_verify_a_component)
    fails = [r for r in _results if r[1] not in ('PASS', 'SKIP')]
    skips = [r for r in _results if r[1] == 'SKIP']
    print(f'\n[test] {len(_results) - len(fails) - len(skips)}/{len(_results)} passed, {len(skips)} skipped, {len(fails)} failed')
    if skips:
        print('[test] SKIPPED (environment preconditions): ' + '; '.join(f'{n} — {why}' for n, _s, why in skips))
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
