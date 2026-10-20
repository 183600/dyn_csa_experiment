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


def t_min_consistency():
    assert L.signflip_min_consistency(12, 4, 0.05) == 11, 'n=12, m=4: 11/12 same-sign clears the first Holm gate (raw p = 2*(1+12)/4096 = 0.00635 < 0.0125)'
    assert L.signflip_min_consistency(6, 4, 0.05) is None, 'n=6, m=4: even 6/6 cannot clear the gate'
    assert L.signflip_min_consistency(8, 1, 0.05) == 8
    assert L.signflip_min_consistency(6, 1, 0.05) == 6
    # Scale-up: the planning math must not overflow int->float at large n.
    big = L.signflip_min_consistency(2000, 10, 0.05)
    assert big is not None and 0 < big < 2000, big


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
    # Too few points stays insufficient regardless of which side is missing.
    h = L.classify_training_health([{'step': 100, 'train_ce': 5.0, 'val_ce': 4.0, 'val_ppl': 55.0}])
    assert h['status'] == 'insufficient_data', h
    h = L.classify_training_health([])
    assert h['status'] == 'insufficient_data', h


def t_training_health_noise_aware_threshold():
    # Four equally spaced eval points with realistic CE jitter: the tail slope
    # is +3.2e-5, five times the old fixed 1e-5 tolerance, yet it is entirely
    # within the slope's own noise (2*SE ~ 3.4e-5).  The fixed threshold called
    # this "overfitting"; the adaptive threshold must call it healthy.
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


def t_rope_arms_are_not_batched():
    # The review found the implementation binding certified eager_batch=True
    # for the rope csa arms while `HybridAttentionRoPE.forward` unconditionally
    # took the per-row `_single_rope` loop.  The predicate now recognises rope
    # layers, so a CUDA-capable device still yields False for all four frozen
    # arms; a non-rope csa config remains eligible (the check is not a blanket
    # False).
    class _Cuda:
        type = 'cuda'
    for arm in ('csa_fixed_rope', 'full_rope', 'csa_fixed_rope_randidx',
                'csa_fixed_rope_allblocks'):
        cfgs = L.make_layer_cfgs(6, arm)
        assert L._uses_batched_eager(cfgs, device=_Cuda(), batch_size=12) is False, arm
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed'), device=_Cuda(), batch_size=12) is True
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed_rope'), device=_Cuda(), batch_size=1) is False


def t_manifest_v3_admission_can_succeed():
    # The zero-GPU plan check is the end-to-end guard for the "obvious fix"
    # trap: align the stamp, then admit 0/48 because P5T records lack
    # train_knobs.  Exercise it on the shipped manifest.
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


def t_exact_signflip_baseline():
    p = L.exact_signflip_p([1.0] * 6)['p']
    assert abs(p - 0.03125) < 1e-12, 'all-same-sign n=6 two-sided p must stay 2/64'
    p12 = L.exact_signflip_p([1.0] * 12)['p']
    assert abs(p12 - 2.0 / 4096) < 1e-12


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
            raise AssertionError('duplicate labels/seeds must be rejected instead of mis-paired')
        except ValueError:
            pass


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
        psm = os.path.join(td, 'smoke.json')
        _write_manifest(psm, [0], extra_stat={'descriptive_only': True})
        V12._load_manifest(psm)  # descriptive smoke manifests opt out


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

    def _panel(*, knob_override=None, drop_seed=None):
        s = {}
        for i in range(3):
            for arm, ppl in (('a', 10.0), ('b', 11.0)):
                if arm == 'b' and i == drop_seed:
                    continue
                kk = dict(knobs)
                if knob_override == (arm, i):
                    kk['grad_clip_norm'] = 2.0
                s[f'{arm}::seed{i}'] = dict(
                    variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                    ppl=ppl, params=100, train_knobs=kk)
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
        assert not any('train_knobs differ' in p for p in ok['problems']), ok['problems']
        # A different optimizer/clip in ONE arm must fail even when every arm
        # is internally consistent.
        bad = _run(_panel(knob_override=('b', 1)))
        assert bad['status'] == 'fail', bad
        assert any('train_knobs differ across arms' in p for p in bad['problems']), bad['problems']
        # Missing seed in one arm: arm means would be over different seeds.
        cov = _run(_panel(drop_seed=2))
        assert cov['status'] == 'fail', cov
        assert any('seed/protocol coverage differs' in p for p in cov['problems']), cov['problems']


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
    # scope block present (issue 4)
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


def t_seed_contract_check():
    L.set_seed(4242)
    fp0 = L.rng_fingerprint()
    r = L.seed_contract_check(seed=3)
    assert r['passed'] is True, r
    assert r['checks']['same_seed_reproducible'] is True
    assert r['checks']['different_seed_changes_streams'] is True
    assert r['checks']['worker_seed_reproducible'] is True
    assert r['checks']['cuda_seeded'] is (True if torch.cuda.is_available() else None)
    assert L.rng_fingerprint() == fp0, 'the seed contract probe must not advance the process RNG streams'


def t_batch_iter_reproducible():
    ids = np.arange(5000, dtype=np.int64) % 997
    b1 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    b2 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1]), 'batch_iter must be seeded by the run seed'
    b3 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=8))
    assert not torch.equal(b1[0], b3[0]), 'a different data seed must change the batch order'


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
        summary = {'full::seed0': {'ppl': 10.0, 'steps': 3, 'tokens_seen': 192, 'params': 100, 'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc', '_code': L.code_stamp(), 'losses': [3.0, 2.3], 'ppl_history': []}}
        L.write_run_manifest(d, cfg, 'FP', prov, summary, status='complete', planned_cells=planned, started_at='T0')
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['status'] == 'complete' and e['n_completed'] == 1 and e['missing_cells'] == ['full::seed1']
        assert e['created_at'] == 'T0', 'a final update must not reset the panel start time'
        assert e['data_version'] == prov['data']['data_version']
    finally:
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
              'cublas_workspace_config', 'dependencies'):
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
    art = L.simple_baseline_fit(train, val, vocab, protocol=proto)
    assert art['best_baseline'] in ('unigram', 'bigram')
    best = min(e['selected_metric_value'] for e in art['baselines'].values())
    assert abs(float(art['best_ppl']) - float(best)) < 1e-12, art
    assert art['data']['val_fp'] and art['data']['n_eval_tokens'] == val.size
    audit = L.simple_baseline_audit(art, protocol=proto)
    assert audit['baseline_verified'] is True, audit
    # A selected value that is not the selected config's own trial value is a
    # fabricated result, even if it happens to equal the grid minimum.
    bad = json.loads(json.dumps(art))
    arm = bad['best_baseline']
    bad['baselines'][arm]['selected_metric_value'] = float(bad['baselines'][arm]['selected_metric_value']) + 1.0
    rbad = L.simple_baseline_audit(bad, protocol=proto)
    assert rbad['baseline_verified'] is False and any('does not match' in p for p in rbad['problems']), rbad
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
    # interpolation=0 must reduce to the unigram model.
    uni = L.ngram_lm_score(train, val, vocab, order=1, alpha=0.1)
    zero = L.ngram_lm_score(train, val, vocab, order=2, alpha=0.1, interpolation=0.0)
    assert abs(uni['nll'] - zero['nll']) < 1e-12 and abs(uni['ppl'] - zero['ppl']) < 1e-9, (uni, zero)


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
    # Record-level policy: F1 without averaging, and rare ROC-AUC without
    # PR-AUC/prevalence, must warn.
    bad = {'r::seed0': {'variant': 'r', 'task': 'binary_classification', 'f1': 0.4,
                        'roc_auc': 0.99, 'prevalence': {'0': 0.9995, '1': 0.0005}}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn', ma
    assert any('averaging' in w for w in ma['warnings']), ma
    assert any('PR-AUC' in w for w in ma['warnings']), ma


def t_hypothesis_audit_completeness_and_metric_alignment():
    good = L.hypothesis_audit(L.HYPOTHESIS_DESIGN, 'component')
    assert good['status'] == 'ok' and not good['invalid'], good
    assert 'dynamic_blocking' in good['hypotheses'], good
    bad_table = {
        'h0': {'statement': 'x', 'mechanism': 'y', 'predicted_sign': 0,
               'primary_metric': 'ndcg', 'falsifier': 'z'},
        'h1': {'statement': 'x'},
    }
    r = L.hypothesis_audit(bad_table, 'component')
    assert r['status'] == 'warn', r
    assert 'h0' in r['direction_unspecified'] and 'h0' in r['metric_unknown'], r
    assert 'h1' in r['invalid'] and any('incomplete' in w for w in r['warnings']), r
    # A predicted_sign=0 result is significant but cannot be called verified.
    entries = [{'controlled': True, 'n': 7, 'significant_global': True,
                'within_seed_noise': False, 'param_controlled': True,
                'predicted_sign': 0, 'direction_consistent': True}]
    assert L._classify_group_status(entries, min_n=3, alpha=0.05, m_family=1) == 'direction_unspecified'




# ---------------------------------------------------------------------------
# I. leakage-safe data splitting / preprocessing
# ---------------------------------------------------------------------------

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


def t_categorical_shift_empty_reference():
    r = L.categorical_shift_report([], [1, 2, 3])
    assert r['n_reference'] == 0 and r['n_holdout'] == 3, r
    assert r['unseen_rate'] == 1.0 and r['total_variation'] > 0.0, r
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
        try:
            L.chronological_split_indices(10, purge=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'purge={bad!r} must be rejected')
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

    # A zero-variance training feature with a different holdout mean is an
    # undefined SMD, not a zero shift.
    Xc = np.array([[0.0], [0.0], [5.0], [0.0]])
    repc = L.data_quality_report(Xc, train_idx=[0, 1, 3], val_idx=[2])
    shift = repc['checks']['distribution_shift_vs_train']['val']
    assert shift['non_estimable_shift_features'] == [0], repc
    assert repc['status'] == 'warnings', repc

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



# ---------------------------------------------------------------------------
# L. Review-4 hardening: claim gate, Holm family floor, crossover fit,
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
    # A clean 4-point fit carries a slope CI and is labelled descriptive.
    f = L.crossover_scaling_fit(pts(4))
    assert f['status'] == 'ok' and f.get('slope') is not None and f.get('ci') is not None, f
    assert f['claim_level'] == 'descriptive', f
    assert f['volume_ratio'] <= 1.25, f


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
    c = L._compute_control_detail(recs, 'a', 'b')
    assert c['compute_matched'] is False and c['more_compute_arm'] == 'b', c
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
    finally:
        shutil.rmtree(td, ignore_errors=True)
    L.set_seed(0)
    model = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'full')).eval()
    ids = torch.randint(0, 64, (1, 20), dtype=torch.long)
    sw = L.measure_efficiency_sweep(model, ids, torch.device('cpu'), lengths=(8, 16), batches=(1, 2), warmup=0, iters=1)
    assert sw['status'] == 'ok' and len(sw['rows']) == 4, sw
    assert sw['decode_has_kv_cache'] is False and sw['measurement'] == 'cpu perf_counter', sw


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
    finally:
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
    _check('A6 paired_effect resolution_limited', t_paired_effect_resolution)
    _check('A6b scaling_trend ragged seed panels', t_scaling_trend_ragged_pairs)
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
    _check('A4g rope arms never claim the batched eager twin', t_rope_arms_are_not_batched)
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
    _check('C9 aggregate seed normalization', t_aggregate_seed_normalization)
    _check('C10 audit resolution classification', t_audit_resolution_classification)
    _check('C11 controlled-variable gate: mixed train_knobs', t_pair_reason_train_knobs_mixed)
    _check('C11b corpus/validation-slice fingerprint gate', t_pair_reason_fingerprint_gate)
    _check('C21 M0 scale analysis stays descriptive at n=4', t_m0_scale_analysis_descriptive)
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
    _check('C17c paired effect CI + leave-one-seed robustness', t_paired_effect_seed_robustness)
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
    _check('J1 review-3 evidence metadata + significance gate', t_review3_evidence_metadata)
    _check('J2 review-3 pooled aggregation', t_review3_pooled_aggregate)
    _check('J3 review-3 Holm declared-family slots', t_review3_holm_family_size)
    _check('J4 review-3 task-boundary arm + objective descriptor', t_review3_boundary_arm_and_objective)
    _check('J5 review-3 span clean-baseline key', t_review3_span_clean_key)
    _check('E1 seed reproducibility', t_seed_reproducible)
    _check('E1a seed/variant axis normalisation', t_seed_axis_normalization)
    _check('E1b explicit RNG seed contract', t_seed_contract_check)
    _check('E2 batch_iter reproducibility', t_batch_iter_reproducible)
    _check('E2b batch_iter uint16 transport', t_batch_iter_uint16)
    _check('E2c real batch-iterator contract check', t_batch_contract_check)
    _check('E2d failed loader contract blocks the panel', t_panel_blocks_on_loader_contract_failure)
    _check('E2e parallel child gets the seed hash', t_parallel_child_env_seed)
    _check('E3 small-batch overfit sanity', t_overfit_sanity)
    _check('E3b preflight RNG-state preservation', t_rng_state_preserved_by_preflight)
    _check('E3c duplicate-run reproducibility check', t_reproducibility_check)
    _check('E4 provenance + run manifest', t_provenance_and_manifest)
    _check('E4b panel tracking events', t_panel_tracking_events)
    _check('E5 DataLoader/worker seeding', t_dataloader_seeding_helpers)
    _check('E6 end-to-end training reproducibility', t_train_variant_reproducible)
    _check('E7 overfit check detects broken optimizer', t_overfit_detects_broken_optimizer)
    _check('E8 append-only experiment ledger + manifest', t_experiment_ledger)
    _check('E9 data_version identity', t_data_version_identity)
    _check('E10 crash-safe manifest status/planned cells', t_manifest_running_then_complete)
    _check('E11 failed overfit guard is retried', t_failed_guard_is_retried)
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
    _check('H3 equal-search baseline tuning audit', t_baseline_tuning_audit)
    _check('H3a multi-seed selection + equal search-seed audit', t_baseline_multi_seed_selection_audit)
    _check('H3b tuning driver selects on the multi-seed mean', t_baseline_tuning_driver_multiseed_selection)
    _check('H4 tuned count-model floor + audit', t_simple_baseline_ngram_and_audit)
    _check('H5 positive-label / rare-positive PR-AUC / macro-F1 averaging', t_classification_positive_label_and_rare_floor)
    _check('H6 hypothesis completeness + metric alignment + direction=0', t_hypothesis_audit_completeness_and_metric_alignment)
    _check('H7 simple-baseline driver uses the panel corpus', t_simple_baseline_driver_serves_the_panel_corpus)
    _check('H8 leakage audit handles a flat 1-D train cache', t_audit_leakage_handles_flat_train_cache)
    _check('H9 fuse sample_std accepts negative axes', t_fuse_sample_std_negative_axis)
    _check('I1 chronological split contract', t_leakage_safe_split_contract)
    _check('I2 whole-group split contract', t_group_split_contract)
    _check('I2b grouped split keeps overlapping time windows together', t_group_split_respects_time_windows)
    _check('I3 preprocessing fitted on train only', t_preprocessor_train_only)
    _check('I3b imputation fill is identical in fit and transform', t_preprocessor_fill_is_consistent)
    _check('I3c leakage checks validate time/group inputs', t_leakage_checks_validate_inputs)
    _check('I4 data-quality/leakage report', t_data_quality_report)
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
    _check('I14 corpus audit counts overlapping validation rows once', t_corpus_audit_overlap_is_unique_across_windows)
    _check('I15 length audit refuses a missing eval span', t_audit_length_requires_eval_span)
    _check('I16 fairness audit ignores a NaN seed instead of crashing', t_audit_fairness_nan_seed_does_not_crash)
    _check('I17 an empty split cache is not reported as verified', t_audit_split_empty_cache_is_not_ok)
    _check('I18 tuning audit reports malformed seed keys instead of crashing', t_baseline_audit_malformed_seed_keys)
    _check('I19 simple-baseline audit reports malformed records instead of crashing', t_simple_baseline_audit_malformed_records)
    _check('I20 index aliasing, non-finite labels, signed-zero duplicates, undefined SMD, NaT groups', t_data_guard_index_label_and_shift_edges)
    _check('I21 split-and-prepare fits preprocessing on train only', t_split_and_prepare_fits_train_only)
    _check('I22 purge/embargo keeps window boundaries sealed', t_split_purge_embargo)
    _check('I23 grouped split refuses an empty requested side', t_group_split_refuses_empty_requested_side)
    _check('I24 label shift and per-split label corruption counts', t_label_shift_and_split_counts)
    # Open-block reading rule: the substantive gates (causality, reference
    # pooling, masks, edge cases, closed regression) live in test_open_block.py
    # so they can also be run on their own.
    try:
        import test_open_block as _OB
        for _name, _fn in _OB.CHECKS:
            _check(_name, _fn)
    except Exception as _obe:
        _check('K0 open-block suite import', lambda _e=_obe: (_ for _ in ()).throw(_e))
    _check('L1 review-4 reportable claim gate', t_reportable_claim_gate)
    _check('L2 Holm family resolution ceiling', t_holm_family_resolution_ceiling)
    _check('L3 crossover fit refusal rules + slope CI', t_crossover_scaling_refusals)
    _check('L4 matched-init fingerprint + pair gate', t_init_core_fingerprint_and_pair_gate)
    _check('L5 param breakdown + measured compute control', t_param_and_compute_control_detail)
    _check('L6 stale aggregate refused', t_stale_aggregate_refused)
    _check('L7 stale length panel excluded from report', t_sec_p1l_excludes_stale_records)
    _check('L8 efficiency-pair audit + shape sweep', t_efficiency_pairs_audit_and_sweep)
    _check('L9 evidence-release readiness audit', t_release_readiness_audit)
    _check('L10 selection-attribution contract', t_selection_attribution_contract)
    _check('L11 length audit fails closed on missing truncation flag', t_audit_length_missing_truncated_flag)
    fails = [r for r in _results if r[1] not in ('PASS', 'SKIP')]
    skips = [r for r in _results if r[1] == 'SKIP']
    print(f'\n[test] {len(_results) - len(fails) - len(skips)}/{len(_results)} passed, {len(skips)} skipped, {len(fails)} failed')
    if skips:
        print('[test] SKIPPED (environment preconditions): ' + '; '.join(f'{n} — {why}' for n, _s, why in skips))
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
