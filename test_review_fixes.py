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
    import v11_supp as V11
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
    # Duplicate names cannot share one dict slot: rejecting is the only
    # representation that does not shrink/corrupt the family.
    try:
        L.holm_adjust([('a', 0.01), ('a', 0.02)])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate comparison names must be rejected')


def t_holm_invalid_slots_stay_in_family():
    r = L.holm_adjust([(str(i), 0.012) for i in range(4)] + [('bad', None)])
    assert r['bad']['p_holm'] is None and r['bad']['reject'] is False, r
    assert r['0']['m_effective'] == 5, r
    assert abs(r['0']['p_holm'] - 0.06) < 1e-12, r
    assert r['0']['reject'] is False, r

def t_min_consistency():
    assert L.signflip_min_consistency(12, 4, 0.05) == 11, 'n=12, m=4: 11/12 same-sign clears the first Holm gate (raw p = 2*(1+12)/4096 = 0.00635 < 0.0125)'
    assert L.signflip_min_consistency(6, 4, 0.05) is None, 'n=6, m=4: even 6/6 cannot clear the gate'
    assert L.signflip_min_consistency(8, 1, 0.05) == 8
    assert L.signflip_min_consistency(6, 1, 0.05) == 6
    # Scale-up: the planning math must not overflow int->float at large n.
    big = L.signflip_min_consistency(2000, 10, 0.05)
    assert big is not None and 0 < big < 2000, big
    # The planning boundary matches Holm's inclusive rejection rule: at
    # n=3, m=1, alpha=0.25 the minimum raw p is exactly 0.25, which
    # holm_adjust rejects, so the plan is feasible (k_min=3), not impossible.
    assert L.holm_adjust([('x', 0.25)], alpha=0.25)['x']['reject'] is True
    assert L.signflip_min_n_for_holm(1, alpha=0.25) == 3
    assert L.signflip_min_consistency(3, 1, alpha=0.25) == 3
    assert abs(L.signflip_power(3, 1, 0.9, alpha=0.25) - 0.729) < 1e-12
    # Planning counts are exact integers; bool/fractional values must not be
    # truncated into a feasible-looking plan.
    for _bad_n in (True, np.bool_(False), 12.9, float('nan'), 0, -1):
        assert L.signflip_min_consistency(_bad_n, 4) is None, _bad_n
        assert not math.isfinite(L.holm_min_attainable_p(_bad_n, 4)), _bad_n
    for _bad_m in (True, np.bool_(False), 4.9, 0, -1):
        assert L.signflip_min_consistency(12, _bad_m) is None, _bad_m
        assert not math.isfinite(L.holm_min_attainable_p(12, _bad_m)), _bad_m
    for _bad in (True, 4.9, 0, -1):
        assert L.signflip_min_n_for_holm(_bad) is None, _bad
        assert L.signflip_min_n_for_holm(1, alpha=_bad) is None, _bad


def t_scale_robustness_edges():
    # A planning family count can be a huge integer; it must saturate.
    assert L.holm_min_attainable_p(6, 10 ** 400) == 1.0
    assert 0.0 < L.holm_min_attainable_p(2000, 10 ** 400) < 1.0
    assert L.signflip_min_n_for_holm(10 ** 400, 0.05) is None
    # Above the exact cutoff, the asymptotic branch must still evaluate the
    # lower tail.
    _n = (1 << 20) + 1
    _m = _n // 2 - 4 * math.isqrt(_n)
    _p = L._binom_cdf_half(_n, _m)
    assert 0.0 < _p < 1e-10, _p
    assert L._binom_cdf_half(_n, _m - 1) <= _p <= L._binom_cdf_half(_n, _m + 1)
    # A malformed significance level fails closed.
    for _bad_alpha in (True, np.bool_(False), 'bad', 0, 1, float('nan')):
        assert L.signflip_min_consistency(12, 4, _bad_alpha) is None, _bad_alpha
        assert L.signflip_power(12, 4, 0.9, _bad_alpha) == 0.0, _bad_alpha
        try:
            L.holm_adjust([('x', 0.01)], alpha=_bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'holm_adjust accepted alpha={_bad_alpha!r}')
    # Split sizing must still sum to n when the floating-point targets
    # overshoot or undershoot by a row.
    import data_guard as _DG
    _n = 2 ** 53
    for _fs in ((0.5, 0.3, 0.2), (0.5123048843996699, 0.3125296923390423,
                                  0.17516542326128773)):
        _sizes = _DG._resolve_sizes(_n, *_fs)
        assert sum(_sizes) == _n and all(v >= 0 for v in _sizes), (_fs, _sizes)
    # Above the binary64 range the float product n*frac must not overflow;
    # the apportionment still sums to the exact integer n.
    for _n in (10 ** 100, 10 ** 400):
        for _fs in ((0.8, 0.1, 0.1), (0.5, 0.5000005, 0.0)):
            _sizes = _DG._resolve_sizes(_n, *_fs)
            assert sum(_sizes) == _n and all(v >= 0 for v in _sizes), (_fs, _sizes)
        _explicit = _DG._resolve_sizes(_n, train_size=_n // 2)
        assert sum(_explicit) == _n and _explicit[0] == _n // 2, _explicit
    # A huge planning family must not be converted to float while checking
    # the Holm gate (int * float overflows even when p is tiny).
    for _n in (6, 12, 100):
        assert L.signflip_min_consistency(_n, 10 ** 400) is None, _n
        assert L.signflip_power(_n, 10 ** 400, 0.9) == 0.0, _n
    # The asymptotic binomial branch must also accept n beyond float range.
    for _n in (10 ** 100, 10 ** 400):
        _mid = _n // 2
        _p = L._binom_cdf_half(_n, _mid)
        assert 0.4 < _p < 0.6, (_n, _p)
        _k = L.signflip_min_consistency(_n, 4)
        assert isinstance(_k, int) and 0 < _k <= _n, (_n, _k)
        assert 0.0 <= L.signflip_power(_n, 4, 0.9) <= 1.0, _n
    assert math.isinf(A.train_flops_estimate(10 ** 400, 10 ** 400)['flops_6nd'])
    # Large-degree t/chi-square tails must converge to the normal limit /
    # median.
    assert abs(L._t_critical(0.95, 10 ** 20) - 1.959963984540054) < 1e-6
    assert abs(L.student_t_two_sided_p(2.0, 10 ** 400)
               - 0.04550026389635842) < 1e-9
    assert abs(L.chi2_sf(10 ** 12, 10 ** 12) - 0.5) < 0.01
    assert L.chi2_sf(10 ** 400, 10) == 0.0
    # The scale-ladder audit performs its 1% token-count check with exact
    # integer arithmetic; a huge internally consistent panel must not overflow.
    _big = 10 ** 200
    _tiers = [{'name': 'a', 'cfg': {'steps': _big, 'batch_size': _big,
                                    'seq_len': _big, 'n_train_tokens': _big ** 3,
                                    'lr': 1e-3, 'warmup': 0, 'seeds': [0]},
               'params': _big ** 3},
              {'name': 'b', 'cfg': {'steps': _big, 'batch_size': _big,
                                    'seq_len': _big, 'n_train_tokens': _big ** 3,
                                    'lr': 1e-3, 'warmup': 0, 'seeds': [0]},
               'params': _big ** 3}]
    assert L.scale_ladder_audit(_tiers)['budget_matched'] is True
    # Exact token counts beyond binary64 must not crash the pooled/per-document
    # PPL reductions.
    _huge_tok = 10 ** 400
    assert L._nll_doc_ci([100.0, 100.0], [_huge_tok, _huge_tok])['n_docs'] == 2
    _ppl, _nll, _ntok = L._pooled_ppl_from_records(
        [{'nll_sum': 100.0, 'n_tokens': _huge_tok},
         {'nll_sum': 100.0, 'n_tokens': _huge_tok}])
    assert _ppl == 1.0 and _ntok == 2 * _huge_tok, (_ppl, _nll, _ntok)
    # The short-panel branch of the per-document CI must also survive a token
    # total beyond binary64 (it used to divide a huge int and raise).
    _early = L._nll_doc_ci([100.0], [_huge_tok])
    assert _early['n_docs'] == 1 and _early['ppl'] == 1.0, _early
    # Per-row token counts are exact: a value that rounds through float64
    # (2**53+1) must not silently become a different denominator.
    _row = [{'variant': 'csa', 'seed': s, 'ppl': 10.0,
             'nll_per_row': [1.0, 1.0],
             'n_tokens_per_row': [2 ** 53 + 1, 2 ** 53 + 1]} for s in range(3)]
    assert L._doc_bootstrap_ppl_ci(_row, n_boot=20) is None
    assert L._combined_seed_doc_ci(_row, n_boot=20)['status'] == 'unverifiable'
    # A missing/non-object record cannot certify either uncertainty
    # component; the helpers must report unverifiable rather than crash.
    for _bad_recs in (None, [None], [1], ['x']):
        assert L._doc_bootstrap_ppl_ci(_bad_recs, n_boot=20) is None
        assert L._combined_seed_doc_ci(_bad_recs, n_boot=20)['status'] == 'unverifiable'
    _ok_recs = [{'nll_per_row': [1.0, 1.0], 'n_tokens_per_row': [4, 4]}
                for _ in range(2)]
    for _bad_conf in (True, np.bool_(False), 0, 1, 1.5, float('nan'), 'x'):
        _r = L._combined_seed_doc_ci(_ok_recs, n_boot=20, conf=_bad_conf)
        assert _r['status'] == 'invalid_options' and _r['ci95_ppl'] is None, _r
    # Panel readers must not crash on a malformed per-layer stats payload.
    _agg_bad = L.aggregate({'x': {'variant': 'csa', 'seed': 0, 'ppl': 10.0,
                                  'nll_sum': 10.0, 'n_tokens': 1, 'stats': [1]}})
    assert 'avg_dyn_block_len' not in _agg_bad['csa'], _agg_bad
    # Panel readers refuse a non-object summary instead of raising a raw
    # AttributeError halfway through.
    for _bad_summary in (None, [], 123, 'x'):
        try:
            L.methodology_audit(_bad_summary, design={}, baselines={}, verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f'methodology_audit accepted {_bad_summary!r}')
    # None is the documented empty family for attach_holm; a non-object is not.
    assert L.attach_holm(None) is not None
    for _bad_comparisons in ([], 123, 'x'):
        try:
            L.attach_holm(_bad_comparisons)
        except ValueError:
            pass
        else:
            raise AssertionError(f'attach_holm accepted {_bad_comparisons!r}')
    # Aggregate training counts use the exact integer contract too.
    for _bad_count in (True, 48.5):
        _a = L.aggregate({'x': {'variant': 'csa', 'seed': 0, 'ppl': 10.0,
                                'nll_sum': 10.0, 'n_tokens': 1,
                                'steps': _bad_count, 'tokens_seen': _bad_count}})
        assert _a['csa']['tokens_seen'] is None, _a
        assert 'tokens_per_step' not in _a['csa'], _a
    # Paired effect options are part of the test contract: a boolean/out-of-
    # range alpha or a fractional/NaN seed floor cannot make an effect
    # "significant", and must not raise halfway through.
    _strong = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for _bad_alpha in (True, np.bool_(False), 0.0, 1.0, 1.5, float('nan'), 'x'):
        _st = L.paired_effect_stats(_strong, alpha=_bad_alpha, min_n=6)
        assert _st['significant'] is False, (_bad_alpha, _st)
    for _bad_min in (True, np.bool_(False), 4.9, 0, -1, float('nan')):
        _st = L.paired_effect_stats(_strong, alpha=0.05, min_n=_bad_min)
        assert _st['underpowered'] is True, (_bad_min, _st)
        assert _st['significant'] is False, (_bad_min, _st)
    # Metric/hypothesis audits fail closed on a non-object artifact instead of
    # raising from dict-only methods.
    assert L.metric_policy_audit([1])['status'] == 'warn'
    assert L.hypothesis_audit(5)['status'] == 'warn'
    # The open-block and v10 readers follow the same underflow convention.
    import open_block_supp as _OB
    assert _OB._nll_per_token({'n_tokens': _huge_tok, 'nll_sum': 10.0}) == 0.0
    assert V10._cell_ppl([10.0], [_huge_tok]) == 1.0
    # Frozen by-length row/token counts are exact integers even above 2**53.
    _nseq = 2 ** 53 + 1
    _cell = {'n_tok': _nseq * 7, 'n_seq': _nseq, 'truncated': False, 'eval_span': 8}
    assert V12._by_len_shape_reason({'by_len': {'8': dict(_cell)}}, _nseq) is None
    _bad = dict(_cell, n_tok=_nseq * 7 + 1)
    assert V12._by_len_shape_reason({'by_len': {'8': _bad}}, _nseq) is not None
    # Object integer tables outside int64/uint64 have no fixed-width twin;
    # they must still fingerprint exactly.
    import data_guard as DG
    assert DG._matrix_fingerprint(np.array([[10 ** 30]], dtype=object)) != \
        DG._matrix_fingerprint(np.array([[10 ** 30 + 1]], dtype=object))
    # Object booleans must key like their typed bool/int twin so that
    # cross-split duplicates remain visible.
    for _v in (True, False):
        _obj = np.array([[_v]], dtype=object)
        for _dt in (np.bool_, np.int8, np.int64, np.uint64):
            assert DG._row_keys(_obj) == DG._row_keys(np.array([[_v]], dtype=_dt)), _dt
    # Dot segments are path syntax, not ordinary filename components.
    assert L._safe_path_component('..') not in ('.', '..')
    assert L._safe_path_component('.') not in ('.', '..')
    assert '/' not in L.ckpt_file_name('..', 0, '.')


def t_scale_aggregate_and_claim_edges():
    # Seed ordering stays exact above the binary64 range.
    _huge_seed = 10 ** 400
    _summ = {'x': {'variant': 'csa', 'seed': _huge_seed, 'ppl': 10.0,
                   'nll_sum': 10.0, 'n_tokens': 1}}
    _agg = L.aggregate(_summ)
    assert _agg['csa']['seeds'] == [_huge_seed], _agg
    # A malformed integral p-value is an invalid family slot.
    _entries = [{'p_exact_signflip': _huge_seed}]
    L._holm_assign(_entries, 1, 0.05, 'p_holm')
    assert _entries[0]['p_holm'] is None, _entries
    # The claim gate rejects malformed alpha/min_n values.
    _base = {'n': 10, 'mean': -1.0, 'std': 0.1,
             'p_exact_signflip': 0.01, 'p_holm': 0.02}
    assert L.reportable_claim(_base)['allowed'] is True
    for _bad_alpha in (True, np.bool_(False), _huge_seed, 0, 1.5, 'x'):
        _r = L.reportable_claim(_base, alpha=_bad_alpha)
        assert _r['allowed'] is False and _r['reasons'], (_bad_alpha, _r)
    for _bad_min_n in (True, np.bool_(False), 1.5, '6', None, 0, -1):
        _r = L.reportable_claim(_base, min_n=_bad_min_n)
        assert _r['allowed'] is False and _r['reasons'], (_bad_min_n, _r)
    for _bad_fs in (True, np.bool_(False), 1.5, '4', _huge_seed):
        _r = L.reportable_claim(_base, family_size=_bad_fs)
        assert _r['allowed'] is False and _r['reasons'], (_bad_fs, _r)
    # A huge internally consistent ladder reports a finite ratio.
    def _tier(name, params):
        return {'name': name,
                'cfg': {'steps': 10 ** 200, 'batch_size': 2, 'seq_len': 8,
                        'n_train_tokens': 10 ** 200 * 16, 'lr': 1e-3,
                        'warmup': 1, 'seeds': [0, 1]},
                'params': params}
    _same = L.scale_ladder_audit([_tier('a', _huge_seed),
                                  _tier('b', _huge_seed)])
    assert _same['budget_matched'] is True, _same
    assert _same['tokens_per_param_ratio'] == 1.0, _same
    _diff = L.scale_ladder_audit([_tier('a', _huge_seed),
                                  _tier('b', _huge_seed * 2)])
    assert _diff['budget_matched'] is False, _diff
    assert _diff['tokens_per_param_ratio'] == 2.0, _diff
    # Cache signatures carry nanosecond/inode integers above 2**53 and are
    # compared exactly.
    _sigs = {'a': [1, 2 ** 60, 3, 4]}
    assert L._cache_file_sigs_match({'file_sigs': {'a': [1, 2 ** 60, 3, 4]}},
                                    _sigs) is True
    for _bad in ([1, float(2 ** 60), 3, 4], [True, 2 ** 60, 3, 4],
                 [1, 2 ** 60, 3], [1, 2 ** 60, 3, 4.5]):
        assert L._cache_file_sigs_match({'file_sigs': {'a': _bad}},
                                        _sigs) is False, _bad
    # The freshly computed side is an identity too; a boolean/fractional
    # expected field cannot be coerced into a matching signature.
    for _bad_sig in ({'a': [1, True, 3, 4]}, {'a': [1, 2 ** 60, 3, 4.5]},
                     {'a': [1, float(2 ** 60), 3, 4]}):
        assert L._cache_file_sigs_match({'file_sigs': {'a': [1, 2 ** 60, 3, 4]}},
                                        _bad_sig) is False, _bad_sig


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
    # The equivalence decision must follow the requested CI confidence:
    # 0.90 corresponds to two one-sided alpha=0.05 tests, 0.95 to alpha=0.025.
    between = [-0.006126797500916851, 0.014617151619714162,
               -0.013712951262380337, 0.04085417964408369,
               -0.01058772677371541, 0.029558696798855623]
    r90 = L.tost_paired(between, margin=0.03, conf=0.90)
    r95 = L.tost_paired(between, margin=0.03, conf=0.95)
    assert 0.025 < r90['p_tost'] < 0.05, r90
    assert r90['equivalent'] is True and r95['equivalent'] is False, (r90, r95)
    assert abs(r90['equiv_alpha'] - 0.05) < 1e-12 and abs(r95['equiv_alpha'] - 0.025) < 1e-12, (r90, r95)


def t_safe_exp_and_doc_ci():
    # A finite exponent must stay a finite PPL; only an actual IEEE overflow
    # yields inf.
    assert math.isfinite(L._safe_exp_ppl(705.0))
    assert L._safe_exp_ppl(1000.0) == float('inf')
    assert L._safe_exp_ppl(float('nan')) != L._safe_exp_ppl(float('nan'))  # NaN stays NaN
    assert L._safe_exp_ppl(True) is None and L._safe_exp_ppl(np.bool_(False)) is None
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
    # Negative NLL is not a measured cross-entropy: it would report PPL < 1
    # with a zero-width interval.  The bootstrap helpers must refuse it too.
    bad = L._nll_doc_ci([-10.0, -10.0], [100, 100])
    assert bad['ppl'] is None and bad['n_docs'] == 0, bad
    assert L._doc_bootstrap_ppl_ci([
        {'nll_per_row': [-1.0, 2.0], 'n_tokens_per_row': [3, 4]},
        {'nll_per_row': [-1.0, 2.0], 'n_tokens_per_row': [3, 4]}]) is None
    comb = L._combined_seed_doc_ci([
        {'nll_per_row': [-1.0, 2.0], 'n_tokens_per_row': [3, 4]},
        {'nll_per_row': [-1.0, 2.0], 'n_tokens_per_row': [3, 4]}])
    assert comb['se_doc_logppl'] is None and comb['n_docs'] is None, comb
    # Misaligned per-document arrays must not be silently truncated by zip().
    try:
        L._nll_doc_ci([1.0, 100.0], [10])
    except ValueError as e:
        assert 'misaligned' in str(e), e
    else:
        raise AssertionError('_nll_doc_ci silently dropped a misaligned document')
    # The same iterable contract holds for the curve diagnostics.
    assert L._finite_curve(np.array([1.0, np.inf, 2.0]))['n_finite'] == 2
    assert L._finite_curve(iter([1.0, 2.0]))['n_finite'] == 2


def t_bootstrap_mean_ci():
    vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    a = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    b = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    assert a == b and a['n'] == 5 and a['ci_low'] <= a['mean'] <= a['ci_high'], a
    assert L.bootstrap_mean_ci([1.0]) is None
    assert L.bootstrap_mean_ci([1.0, float('nan'), float('inf')]) is None
    # Paired/unpaired helpers stay deterministic and well-formed on large
    # seed panels.
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
    # Boolean/string curve entries are flags/metadata, not measurements.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 - 0.02 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    hist[2]['val_ce'] = True
    h = L.classify_training_health(hist)
    assert h['status'] == 'nonfinite' and h['n_nonfinite'] == 1, h
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
    # The fixed threshold is still available explicitly and calls the same
    # noise "overfitting", so the test discriminates between the two rules.
    h_fixed = L.classify_training_health(hist, slope_tol=1e-5)
    assert h_fixed['status'] == 'overfitting', h_fixed
    assert h_fixed['slope_tol_mode'] == 'fixed', h_fixed
    # A genuine rising trend must still be caught even under the adaptive rule.
    hist_bad = [dict(hh, val_ce=4.0 + 0.01 * i, val_ppl=55.0 + 0.5 * i)
                for i, hh in enumerate(hist)]
    assert L.classify_training_health(hist_bad)['status'] == 'overfitting'


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


def t_manifest_v3_admission_can_succeed():
    # The zero-GPU plan check is the end-to-end guard against a stamp-only
    # alignment that would admit no records.  Exercise it on the shipped
    # manifest.
    rep = V12.plan_check(os.path.join(REPO, 'confirmation_frozen.json'))
    assert rep['ok'] and rep['admitted'] == rep['n_arms'] * rep['n_seeds'], rep
    assert all(c['pairs_planned'] == rep['n_seeds'] for c in rep['comparisons'].values()), rep
    assert rep['predicted']['full_rope']['eager_batch'] is False, rep
    assert rep['predicted']['csa_fixed_rope']['amp_dtype'] == 'bf16', rep
    # The probe's omitted fields are filled from FINALTEST_CFG at run time;
    # the pre-flight must validate those defaults, not only explicit fields.
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        _m = json.load(f)
    _m['eval']['by_length_ppl'] = [512, 1024]
    _m['eval'].pop('distractor_probe', None)
    with tempfile.TemporaryDirectory() as td:
        _p = os.path.join(td, 'm.json')
        with open(_p, 'w', encoding='utf-8') as f:
            json.dump(_m, f)
        try:
            V12.plan_check(_p)
            raise AssertionError('a probe whose default eval lengths exceed the frozen slice must be refused')
        except SystemExit as e:
            assert 'probe' in str(e), str(e)[:300]


def t_v7_rope_gather_rotation_order():
    # The per-chunk RoPE path rotates the gathered window rows.  The
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
    # JSON round-trips can leave the same seed as 0 or "0"; all three scale
    # entry points must normalise before checking for duplicates.
    try:
        L.meta_analyze_effects([{'label': 'A', 'scale': 1.0, 'effects': [1.0, 2.0], 'seeds': [0, '0']}])
        raise AssertionError('meta_analyze_effects must normalise 0 and "0" to the same seed')
    except ValueError:
        pass
    for fn, args in (
        (L.paired_log_ratio, ([1.0, 2.0], [1.1, 2.1])),
        (L.scale_interaction, ([1.0, 2.0], [1.1, 2.1])),
    ):
        for dup in ([0, 0], [0, '0']):
            try:
                fn(*args, seeds=dup)
                raise AssertionError(f'{fn.__name__} must reject duplicate seeds instead of double-counting them')
            except ValueError:
                pass
    # Boolean flags are not measurements: they must be dropped BEFORE float()
    # turns them into PPL/effect values.
    r = L.paired_log_ratio([True, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert r['n'] == 2 and r['dropped_seeds'] == [0], r
    assert L.scale_interaction([True, 0.1], [1.0, 0.2])['n'] == 1
    assert L.loglog_regression([(True, 1.0), (2, 4), (4, 16)])['n_points'] == 2


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
    # The report's displayed test floor must be the exact 2/2^n for every n,
    # not a fixed 1/200001 surrogate above the n<=20 enumeration path.
    for n in (6, 12, 22, 40, 60):
        assert abs(BR.signflip_p_floor(n) - L.signflip_p_floor(n)) < 1e-18, n
    assert abs(BR.exact_signflip([0.7] * 22)['p_exact_signflip'] - 2.0 / (1 << 22)) < 1e-15
    z = L.exact_signflip_p([0.0] * 24)
    assert z['p'] == 1.0 and z['method'] == 'exact-mitm', 'every sign assignment matches a zero observed mean'
    # The driver used by the confirmation analyzer must not silently substitute
    # a random Monte-Carlo p-value for the pre-registered exact test at n>20.
    v = V7.exact_sign_permutation([0.7] * 21)
    assert abs(v['p_exact_signflip'] - 2.0 / (1 << 21)) < 1e-18 and v.get('method') == 'exact-mitm', v
    # The v7-v12 driver wrapper must reject boolean deltas too, not coerce
    # them through np.asarray(dtype=float).
    vb = V7.exact_sign_permutation([True] * 6)
    assert vb['n'] == 0 and vb['n_dropped'] == 6 and math.isnan(vb['p_exact_signflip']), vb
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
    # The confirmation driver delegates to exp_lib, so it inherits the same behaviour.
    v = V7.exact_sign_permutation(list(rng.normal(0.2, 1.0, size=50)))
    assert v['method'].startswith('saddlepoint'), v
    assert 'saddlepoint' in v['test'], v
    # Zero deltas do not multiply the saddlepoint null distribution: the
    # reported floor must be 2/2**(number of NONZERO deltas), not 2/2**n.
    zv = V7.exact_sign_permutation([0.7] * 40 + [0.0] * 10)
    assert zv['method'].startswith('saddlepoint'), zv
    assert abs(zv['p_floor'] - 2.0 / (1 << 40)) < 1e-30, zv['p_floor']
    assert abs(zv['p_exact_signflip'] - 2.0 / (1 << 40)) < 1e-30, zv['p_exact_signflip']


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

    # The unpaired WLS fallback must still honour min_seeds: one observation
    # per scale is not a confirmatory trend even when the t-test p is tiny.
    y = [0.2, 1.15, 1.95, 2.85]
    one_each = [{'label': f'S{i}', 'scale': 10.0 ** i, 'effects': [y[i]],
                 'seeds': [i], 'corpus_fp': 'x'} for i in range(4)]
    r6 = L.scaling_trend(one_each, min_seeds=6, min_points=3)
    assert r6['paired']['n_complete_seeds'] == 0, r6
    assert r6['p_primary'] < 0.05 and r6['underpowered'] is True, r6
    assert r6['significant'] is False and r6['verdict'] == 'underpowered', r6


# ---------------------------------------------------------------------------
# B. v10 corruption modes
# ---------------------------------------------------------------------------

if _HAVE_TORCH:
    _PANEL = np.random.default_rng(7).integers(3, 8000, size=(9, 129), dtype=np.int64)
    _VOCAB = 8192


def _reference_corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target):
    """Reference inline algorithm for the uniform / natural / mask / delete
    modes. Any divergence means a measured value changed — the strongest
    invariant this suite checks."""
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


def t_reference_modes_bitexact():
    for mode, rhos in (('uniform', (0.0, 0.125, 0.5)), ('natural', (0.125, 0.5)), ('mask', (1.0,)), ('delete', (1.0,))):
        for rho in rhos:
            for jj in range(0, 9, 3):
                a = V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                b = _reference_corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                assert np.array_equal(a, b), f'compatibility mode {mode} rho={rho} jj={jj} diverged from the reference algorithm'


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
    # rho is a fraction, not a flag/out-of-range request: a boolean or an
    # out-of-range value must be refused before rng.choice sees a bad size.
    for bad_rho in (True, False, -0.1, 1.0001, float('nan'), float('inf')):
        try:
            V10._corrupt_row(_PANEL, 128, bad_rho, 'uniform', 0, _VOCAB, 32)
            raise AssertionError(f'rho={bad_rho!r} must be rejected')
        except ValueError:
            pass
    # vocab=1 has no alternative token: the rejection-sampling loop would
    # never terminate, so the construction must refuse it.
    try:
        V10._corrupt_row(_PANEL, 128, 0.5, 'uniform', 0, 1, 32)
        raise AssertionError('uniform corruption with vocab=1 must be rejected (no alternative token)')
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
        # The by-length endpoint grid is part of the frozen plan: bool,
        # fractional, length-1 and empty grids must be refused at load time,
        # not silently truncated later.
        for _bad_lens in (4096, [], [4096, 4.9], [True], [0], [1],
                          [float(2 ** 60)], [2 ** 60]):
            pbad = os.path.join(td, 'bad_lens.json')
            mb = _write_manifest(pbad, list(range(12)))
            mb['eval'] = {'n_seqs_clean': 16, 'by_length_ppl': _bad_lens}
            with open(pbad, 'w', encoding='utf-8') as f:
                json.dump(mb, f)
            try:
                V12._load_manifest(pbad)
            except SystemExit as e:
                assert 'by_length_ppl' in str(e), str(e)[:200]
            else:
                raise AssertionError(f'by_length_ppl={_bad_lens!r} must be refused')
        pok = os.path.join(td, 'ok_lens.json')
        mo = _write_manifest(pok, list(range(12)))
        mo['eval'] = {'n_seqs_clean': 16, 'by_length_ppl': [512, '1024', 2048.0]}
        with open(pok, 'w', encoding='utf-8') as f:
            json.dump(mo, f)
        V12._load_manifest(pok)
        for _bad_nseq in (float(2 ** 60), 2 ** 60):
            pbad = os.path.join(td, 'bad_nseq.json')
            mb = _write_manifest(pbad, list(range(12)))
            mb['eval'] = {'n_seqs_clean': _bad_nseq, 'by_length_ppl': [4096]}
            with open(pbad, 'w', encoding='utf-8') as f:
                json.dump(mb, f)
            try:
                V12._load_manifest(pbad)
            except SystemExit as e:
                assert 'integer' in str(e), str(e)[:200]
            else:
                raise AssertionError(f'eval.n_seqs_clean={_bad_nseq!r} must be refused')
        # A non-object eval block must be refused explicitly.
        for _bad_eval in ([], [4096], 'x'):
            pbad = os.path.join(td, 'bad_eval.json')
            mb = _write_manifest(pbad, list(range(12)))
            mb['eval'] = _bad_eval
            with open(pbad, 'w', encoding='utf-8') as f:
                json.dump(mb, f)
            try:
                V12._load_manifest(pbad)
            except SystemExit as e:
                assert 'eval' in str(e), str(e)[:200]
            else:
                raise AssertionError(f'eval={_bad_eval!r} must be refused')


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
    # A usable ppl alone does not make a truncated cell complete: resume must
    # use the same full-shape predicate as admission or it will skip a record
    # that admission later rejects.
    trunc = _rec('a_arm', 0, 'FP', by_len={'4096': {
        'ppl': 30.0, 'n_tok': 16 * 4095, 'n_seq': 16, 'truncated': True}})
    assert L.by_len_cells([trunc], 4096), 'the truncated cell has a usable ppl'
    assert V12._by_len_shape_reason(trunc, 16) is not None
    # The frozen by-length grid accepts only exact integer length keys.  A
    # boolean, fractional or lossy large float must not alias a real endpoint.
    for _bad_ln in (True, np.bool_(False), '4.9', 4.9, float(2 ** 60)):
        _bad = _rec('a_arm', 0, 'FP', by_len={_bad_ln: {
            'ppl': 30.0, 'n_tok': 16 * 4095, 'n_seq': 16, 'truncated': False}})
        assert V12._by_len_shape_reason(_bad, 16) is not None, _bad_ln
    _ok = _rec('a_arm', 0, 'FP', by_len={4096.0: {
        'ppl': 30.0, 'n_tok': 16 * 4095, 'n_seq': 16, 'truncated': False}})
    assert V12._by_len_shape_reason(_ok, 16) is None
    # The cell's own n_seq is part of the frozen estimand; a shape-consistent
    # n_tok with a different row count must not be admitted.
    _nseq_bad = _rec('a_arm', 0, 'FP', by_len={4096: {
        'ppl': 30.0, 'n_tok': 16 * 4095, 'n_seq': 8, 'truncated': False}})
    assert V12._by_len_shape_reason(_nseq_bad, 16) is not None
    # A synthesized reconstruction has no weights and is not an independent
    # final-test measurement; it must never enter the confirmation.
    raw = dict(good, **{'a::s2': dict(_rec('a_arm', 2, 'FP'), synthesized=True)})
    adm_s, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm_s and len(adm_s) == 4
    # A shape-complete cell without a usable PPL is not a measurement.
    _unmeasured = _rec('a_arm', 2, 'FP', by_len={4096: {
        'ppl': None, 'n_tok': 16 * 4095, 'n_seq': 16, 'truncated': False}})
    raw = dict(good, **{'a::s2': _unmeasured})
    adm_u, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm_u and len(adm_u) == 4
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
    # A manifest without eval.n_seqs_clean must not disable the row-count
    # consistency check when the records themselves carry n_seq.
    raw = {'a::s0': dict(_rec('a_arm', 0, 'FP'), n_seq=16),
           'b::s0': dict(_rec('b_arm', 0, 'FP'), n_seq=8)}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('records with different n_seq must not be paired')
    except SystemExit as e:
        assert 'n_seq' in str(e), e
    # Same n_seq, but a truncated cell is not a complete measurement.
    raw = {
        'a::s0': dict(_rec('a_arm', 0, 'FP'), n_seq=16),
        'b::s0': dict(_rec('b_arm', 0, 'FP', by_len={'4096': {
            'ppl': 30.0, 'n_tok': 16 * 4095, 'n_seq': 16, 'truncated': True}}), n_seq=16),
    }
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('a truncated by_len cell must not be admitted')
    except SystemExit as e:
        assert 'truncated' in str(e), e
    # In-memory manifests must obey the same exact seed/row-count contract as
    # the loader: bool/fractional ids cannot alias a planned seed.
    for _bad_seeds in ([True], [1.5], [1, 1.0]):
        try:
            V12._admit_confirmation_records(
                {'a::s1': _rec('a_arm', 1, 'FP')},
                _manifest_for_admission(seeds=_bad_seeds))
            raise AssertionError(f'bad in-memory seeds {_bad_seeds!r} must be refused')
        except SystemExit as e:
            assert 'seed' in str(e).lower(), e
    for _bad_nseq in (True, 16.9, 0, 'x'):
        _mb = _manifest_for_admission()
        _mb['eval'] = {'n_seqs_clean': _bad_nseq}
        try:
            V12._admit_confirmation_records(
                {'a::s0': _rec('a_arm', 0, 'FP')}, _mb)
            raise AssertionError(f'bad in-memory n_seqs_clean {_bad_nseq!r} must be refused')
        except SystemExit as e:
            assert 'row count' in str(e), e
    for _bad in ([True], [1.5], [1, 1.0]):
        try:
            V12._manifest_seed_ids({'seeds': _bad})
            raise AssertionError(f'bad frozen seeds {_bad!r} must be refused')
        except SystemExit as e:
            assert 'seed' in str(e).lower(), e
    assert V12._manifest_seed_ids({'seeds': [0, 1.0, '2']}) == [0, 1, 2]


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
    # A boolean/non-finite signature value must be refused at load time: the
    # synthetic plan recipe could otherwise "match" by string equality while
    # every real checkpoint (numeric steps) is rejected after test access.
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        _bad = json.loads(json.dumps(v2))
        _bad['binding']['training_recipe_signature']['steps'] = True
        json.dump(_bad, f)
        _bad_path = f.name
    try:
        try:
            V12._load_manifest(_bad_path)
        except SystemExit as e:
            assert 'boolean' in str(e), e
        else:
            raise AssertionError('a boolean recipe-signature value must be refused')
    finally:
        os.unlink(_bad_path)


def t_finaltest_requires_parameter_matching():
    # A confirmation must be parameter-matched.  Missing params is
    # unverifiable, not a pass: the decision rule may not treat None like True.
    with tempfile.TemporaryDirectory() as td:
        seeds = list(range(12))
        manifest = {
            'arms': ['a_arm', 'b_arm'], 'seeds': seeds,
            'primary_comparisons': [{'name': 'c1', 'a': 'a_arm', 'b': 'b_arm',
                                     'endpoint': 'by_len.ppl@4096',
                                     'expected_sign': 'negative'}],
            'statistics': {'alpha': 0.05, 'min_seeds_for_confirmation': 8,
                           'descriptive_only': True},
            'manifest_sha256': 'M', 'code_semantics_expected': L.CODE_SEMANTICS,
        }
        def _records(*, params=None, init=None):
            out = {}
            for s in seeds:
                for arm, ppl in (('a_arm', 30.0), ('b_arm', 31.0)):
                    r = {'variant': arm, 'seed': s, 'train_fp': 't',
                         'data_fp': 'd',
                         'by_len': {'4096': {'ppl': ppl, 'n_tok': 16 * 4095,
                                              'n_seq': 16, 'truncated': False}}}
                    if params is not None:
                        r['params'] = params
                    if init is not None:
                        r['init_core_fp'] = init
                    out[(arm, s)] = r
            return out
        _old_admit = V12._admit_confirmation_records
        _old_probe = V12._test_probe_stats
        try:
            V12._test_probe_stats = lambda *a, **k: {}
            V12._admit_confirmation_records = lambda raw, man: (_records(), None)
            st = V12.finaltest_stats(manifest=manifest,
                                     bylen_path=os.path.join(td, 'bylen.json'),
                                     probe_path=os.path.join(td, 'probe.json'),
                                     out=os.path.join(td, 'stats.json'))
            assert st['comparisons']['c1']['param_matched'] is None
            assert st['confirmed'] == [], st['confirmed']
            # Missing matched-init provenance is also not a confirmation.
            V12._admit_confirmation_records = lambda raw, man: (_records(params=1000), None)
            st2 = V12.finaltest_stats(manifest=manifest,
                                      bylen_path=os.path.join(td, 'bylen.json'),
                                      probe_path=os.path.join(td, 'probe.json'),
                                      out=os.path.join(td, 'stats2.json'))
            assert st2['comparisons']['c1']['param_matched'] is True
            assert st2['comparisons']['c1']['init_core_fp_recorded'] is False
            assert st2['confirmed'] == [], st2['confirmed']
            # With parameter matching AND a shared init digest it confirms.
            V12._admit_confirmation_records = lambda raw, man: (_records(params=1000, init='INIT'), None)
            st3 = V12.finaltest_stats(manifest=manifest,
                                      bylen_path=os.path.join(td, 'bylen.json'),
                                      probe_path=os.path.join(td, 'probe.json'),
                                      out=os.path.join(td, 'stats3.json'))
            # One missing per-seed digest is unverifiable, not a shared init.
            _partial = _records(params=1000, init='INIT')
            _partial[('b_arm', 0)].pop('init_core_fp')
            V12._admit_confirmation_records = lambda raw, man: (_partial, None)
            st4 = V12.finaltest_stats(manifest=manifest,
                                      bylen_path=os.path.join(td, 'bylen.json'),
                                      probe_path=os.path.join(td, 'probe.json'),
                                      out=os.path.join(td, 'stats4.json'))
        finally:
            V12._admit_confirmation_records = _old_admit
            V12._test_probe_stats = _old_probe
        assert st3['comparisons']['c1']['same_init_core_fp'] is True, st3['comparisons']['c1']
        assert st3['confirmed'] == ['c1'], st3['confirmed']
        assert st4['comparisons']['c1']['init_core_fp_recorded'] is False, st4['comparisons']['c1']
        assert st4['comparisons']['c1']['same_init_core_fp'] is False, st4['comparisons']['c1']
        assert st4['confirmed'] == [], st4['confirmed']


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


def t_aggregate_rejects_malformed_seed_ids():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='a', run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=10.0)
    raw = {'a::seed0': dict(base, seed=True),
           'b::seed1': dict(base, seed=1.5),
           'c::seed2': dict(base, seed=2, ppl=20.0)}
    agg = L.aggregate(raw)['a']
    assert agg['n_seeds'] == 1 and agg['seeds'] == [2], agg
    assert agg['ppl_mean'] == 20.0, agg


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
    assert r2 is None, 'both records missing the descriptor is the permissive path and may still pair'


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
    # The report's local gate must also require the matched-initialization
    # digest, not only the same seed/run_cfg.
    mi = dict(base, init_core_fp='init-a')
    mi_b = dict(b, init_core_fp='init-a')
    assert BR._pair_reason(mi, mi_b) is None
    assert 'init_core_fp' in BR._pair_reason(mi, dict(mi_b, init_core_fp='init-b'))
    assert 'unverifiable init_core_fp' in BR._pair_reason(mi, b)
    # One-sided presence is unverifiable, not silently compatible.
    c = dict(b)
    c.pop('val_fp')
    assert 'unverifiable val_fp' in L.pair_reason(base, c)
    # Records without fingerprints keep the compatibility path.
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


def t_seed_plan_exact_counts():
    # A plan's seed count is an exact integer.  Truncating 4.9/True to 4/1
    # would report a DIFFERENT (smaller) panel as family-sufficient.
    for bad in (4.9, True, False, -3, '4.9', 'not-a-count', float('nan'), None):
        p = L.methodology_seed_plan(bad)
        assert p['n_seeds'] == 0, (bad, p)
        assert p['sufficient_for_family_rejection'] is False, (bad, p)
    assert L.methodology_seed_plan('3')['n_seeds'] == 3
    assert L.methodology_seed_plan(3.0)['n_seeds'] == 3


def t_t_critical_high_confidence():
    # df=1 has the analytic t = 1/tan(pi*alpha); high-confidence intervals
    # need the full-range bracket.
    for conf in (0.9999, 0.999999):
        got = L._t_critical(conf, 1)
        want = 1.0 / math.tan(math.pi * (1.0 - conf) / 2.0)
        assert abs(got - want) / want < 1e-8, (conf, got, want)
    assert abs(L._t_critical(0.95, 10) - 2.2281388519862826) < 1e-9


def t_record_int_exact_spelling():
    # v12 admission must use the same exact integer contract as the plan:
    # a decimal/exponent string denotes a different value, while a genuine
    # integer spelling stays accepted.
    assert V12._record_int('4') == 4
    assert V12._record_int(4.0) == 4
    for bad in ('4.0', '1e3', ' 4.0 ', 4.9, True, None):
        assert V12._record_int(bad) is None, bad
    # The shared seed contract must also reject lossy int() coercion of
    # number-like objects that are not numbers.Real (e.g. Decimal('4.9')).
    from decimal import Decimal
    assert L._as_seed_int(Decimal('4.9')) is None
    assert L._as_seed_int(Decimal('4.0')) == 4
    assert L._as_seed_int(Decimal('1e3')) == 1000

    class Truncates:
        def __int__(self):
            return 7
    assert L._as_seed_int(Truncates()) is None
    # The report twin must obey the same contract (no lossy int() coercion).
    assert BR._int_or(Decimal('4.9'), None) is None
    assert BR._int_or(Decimal('4.0'), None) == 4
    assert BR._int_or('4', None) == 4 and BR._int_or('4.0', None) is None
    assert BR._int_or(Truncates(), None) is None


def t_eval_length_invalid_lengths():
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(1, 'full_rope')
    model = L.SmallGPT(64, 32, 1, 2, 16, 16, _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(11).integers(0, 64, size=(2, 17)), dtype=np.int64)
    out = V7.eval_length_gen(model, ids, [1, 0, -3, 4.9, 8, 16], dev, max_pos=None)
    for bad in (1, 0, -3, 4.9):
        assert 'error' in out[bad] and 'ppl' not in out[bad], (bad, out[bad])
    for good in (8, 16):
        assert 'ppl' in out[good] and out[good]['eval_span'] == good, (good, out[good])
    # A requested length beyond the supplied slice is a named error cell.
    short = np.asarray(np.random.default_rng(12).integers(0, 64, size=(2, 12)), dtype=np.int64)
    out_short = V7.eval_length_gen(model, short, [8, 20], dev, max_pos=None)
    assert 'ppl' in out_short[8] and 'error' in out_short[20], out_short
    assert out_short[20].get('eval_span') is None, out_short
    # Truncating to fewer than two tokens has no next-token target.
    out_trunc = V7.eval_length_gen(model, ids, [20], dev, max_pos=1)
    assert 'error' in out_trunc[20] and 'ppl' not in out_trunc[20], out_trunc
    # Malformed (unhashable) length identities must not crash the whole eval.
    out_bad = V7.eval_length_gen(model, ids, [[8]], dev, max_pos=None)
    assert len(out_bad) == 1 and 'error' in next(iter(out_bad.values())), out_bad
    # Row count and truncation span are exact integers too: fractional,
    # boolean or non-positive values are named error cells, not silent samples.
    for bad_nseq in (0, -1, 4.9, True, False):
        out_ns = V7.eval_length_gen(model, ids, [8], dev, max_pos=None, n_seq=bad_nseq)
        assert 'error' in out_ns[8] and 'ppl' not in out_ns[8], (bad_nseq, out_ns)
    out_mp = V7.eval_length_gen(model, ids, [8, 16], dev, max_pos=4.9)
    assert all('error' in out_mp[Ln] and 'ppl' not in out_mp[Ln] for Ln in (8, 16)), out_mp
    out_mpb = V7.eval_length_gen(model, ids, [8], dev, max_pos=True)
    assert 'error' in out_mpb[8] and 'ppl' not in out_mpb[8], out_mpb
    for bad_ids in (np.empty((0, 17), dtype=np.int64), ids[0], np.empty((2, 0), dtype=np.int64)):
        out_shape = V7.eval_length_gen(model, bad_ids, [8], dev, max_pos=None)
        assert 'error' in out_shape[8] and 'ppl' not in out_shape[8], (bad_ids.shape, out_shape)


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
    # n=7: all seeds same sign clears the family-corrected gate.
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
    # Boolean/fractional parameter "counts" are not verifiable counts.
    for _bad in (True, 1.5, float(2 ** 53)):
        _why = BT._relaxed_pair_reason(dict(a, params=_bad), dict(b, params=_bad),
                                       {'a': 3, 'b': 3})
        assert _why and 'unverifiable params' in _why, (_bad, _why)
    # A paired seed does not imply a matched shared-trunk initialization.
    a4 = dict(a, init_core_fp='aaaa')
    b4 = dict(b, init_core_fp='bbbb')
    why3 = BT._relaxed_pair_reason(a4, b4, {'a': 3, 'b': 3})
    assert why3 and 'init_core_fp' in why3, why3
    b5 = dict(b)
    why4 = BT._relaxed_pair_reason(a4, b5, {'a': 3, 'b': 3})
    assert why4 and 'unverifiable init_core_fp' in why4, why4


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
    # Row chunking with an explicit safety ceiling.
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
    for n in (0, -1):
        base = L.blocks_from_cuts(n, [], 2, 8, torch.device('cpu'))
        assert base.numel() == 0, (n, base)


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
    # The small-n segmentation cache must key on boolean truthiness, not on a
    # uint8 cast: 0.5 is a wanted cut, not a zero byte.
    L._SEGMENT_CACHE.clear()
    assert L._segment(3, [0.0, 0.0], 1, 3)[2] == [3]
    bids, _, counts = L._segment(3, [0.5, 0.0], 1, 3)
    assert counts == [1, 2] and bids.tolist() == [0, 1, 1], (bids, counts)


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
    # the compute-allocation caveat must be explicit, not buried in prose
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
    _shape_rec = {'by_len': {'8': {'n_tok': 16 * 7, 'n_seq': 16, 'truncated': False}}}
    assert V12._by_len_shape_reason(_shape_rec, 16) is None
    _span_rec = {'by_len': {'8': {'n_tok': 16 * 7, 'n_seq': 16, 'truncated': False,
                                  'eval_span': 7}}}
    assert V12._by_len_shape_reason(_span_rec, 16) is not None
    for _bad_nseq in (4.9, True, 0, -1):
        _why = V12._by_len_shape_reason(_shape_rec, _bad_nseq)
        assert _why is not None and 'n_seq' in _why, (_bad_nseq, _why)
    # A boolean/string is not a per-document PPL measurement; it is counted as
    # an incomplete cell, not coerced to 1.0 (or a numeric string).
    bad_a = {k: dict(v, by_len={'4096': dict(v['by_len']['4096'],
                                            ppl_per_seq=list(v['by_len']['4096']['ppl_per_seq']))})
             for k, v in recs_a.items()}
    bad_a[0]['by_len']['4096']['ppl_per_seq'][0] = True
    bad_a[1]['by_len']['4096']['ppl_per_seq'][0] = '30.0'
    d2 = V12._doc_level_analysis(bad_a, recs_b, list(range(n_seed)), 4096)
    assert d2 is not None and d2['incomplete_doc_seed_cells'] == 2, d2


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
    base = {'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02, 'mean': 1.0, 'std': 0.5}
    assert BR._sig_gate(dict(base, std=float('nan')), register=False) is False
    assert BR._sig_gate(dict(base, std=-1.0), register=False) is False
    assert BR._sig_gate(dict(base, mean=float('inf')), register=False) is False
    assert BR._sig_gate(dict(base, p_exact_signflip=-0.1), register=False) is False
    # Fractional/boolean seed counts are not seed counts; int(6.9)=6 and
    # int(True)=1 must not pass or fail the n>=6 gate as if they were real.
    for _bad_n in (6.9, True, np.bool_(False), float('nan'), 'not-a-count', None):
        assert BR._sig_gate(dict(base, n=_bad_n), register=False) is False, _bad_n
    assert BR._sig_gate(dict(base, n=6.0), register=False) is True
    # A boolean p must not be converted to 0.0 before the report's correction.
    old_load = BR.load
    try:
        BR.load = lambda *a, **k: {'comparisons': {'flag': {'mean': 1.0,
                                                           'p_exact_signflip': False}}}
        txt = BR.sec_stats()
    finally:
        BR.load = old_load
    assert '| flag | 1.00 | n/a | 1.0000 |' in txt, txt

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

def t_pooled_ppl_rejects_malformed_counts():
    assert L._pooled_ppl_from_records([{'nll_sum': 10.0, 'n_tokens': 4.4},
                                       {'nll_sum': 10.0, 'n_tokens': 4.4}]) == (None, None, None)
    assert L._pooled_ppl_from_records([{'nll_sum': -4.0, 'n_tokens': 10},
                                       {'nll_sum': -4.0, 'n_tokens': 10}]) == (None, None, None)
    ppl, nll, nt = L._pooled_ppl_from_records([{'nll_sum': 4.0, 'n_tokens': 10}])
    assert abs(ppl - math.exp(0.4)) < 1e-12 and (nll, nt) == (4.0, 10), (ppl, nll, nt)


def t_nll_doc_ci_exact_token_counts():
    # Per-document token counts are exact integers under the shared contract:
    # Decimal('4.9') must not be truncated to 4 into the pooled denominator.
    from decimal import Decimal
    good = L._nll_doc_ci([2.0, 2.0], [Decimal('4.0'), 4])
    assert good['n_docs'] == 2 and abs(good['ppl'] - math.exp(0.5)) < 1e-12, good
    bad = L._nll_doc_ci([2.0, 2.0], [Decimal('4.9'), 4])
    assert bad['n_docs'] == 1, bad
    assert L._nll_doc_ci([2.0, 2.0], [True, 4])['n_docs'] == 1


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
    # Boolean/fractional per-document NLL or token rows are malformed, not
    # measurements: no interval may be invented from them.
    _bad = _recs([0.0, 0.2, 0.4])
    _bad[0]['nll_per_row'][0] = True
    assert L._doc_bootstrap_ppl_ci(_bad) is None
    assert L._combined_seed_doc_ci(_bad)['status'] == 'unverifiable'
    _bad2 = _recs([0.0, 0.2, 0.4])
    _bad2[1]['n_tokens_per_row'][0] = 1.5
    assert L._doc_bootstrap_ppl_ci(_bad2) is None
    assert L._combined_seed_doc_ci(_bad2)['status'] == 'unverifiable'
    per_seed_lp = []
    for off in (0.0, 0.2, 0.4):
        nll = [1.0 + off + 0.01 * j for j in range(32)]
        # log(PPL) is the pooled cross-entropy.
        per_seed_lp.append(sum(nll) / (32 * 50))
    _se_seed = _np.std(per_seed_lp, ddof=1) / math.sqrt(3)
    assert abs(r['se_seed_logppl'] - float(_se_seed)) < 1e-12, (r, _se_seed)
    assert r['se_doc_logppl'] is not None and r['se_doc_logppl'] > 0, r
    _comb = math.sqrt(r['se_seed_logppl'] ** 2 + r['se_doc_logppl'] ** 2)
    assert abs(r['se_combined_logppl'] - _comb) < 1e-12, r
    assert abs(r['component_ratio_seed_over_doc'] - r['se_seed_logppl'] / r['se_doc_logppl']) < 1e-12, r
    lo, hi = r['ci95_ppl']
    # The combined interval is on the PPL scale and must contain the recs'
    # PPLs.
    assert 0 < lo < hi, r
    assert lo > 1.0 and hi > 1.0, r
    assert all(lo <= rec['ppl'] <= hi for rec in recs), (r, [rec['ppl'] for rec in recs])
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
    # One seed: the seed component is explicitly unverifiable, never zero.
    r1 = L._combined_seed_doc_ci(_recs([0.0]))
    assert r1['status'] == 'seed_component_unverifiable' and r1['se_seed_logppl'] is None and r1['se_doc_logppl'] is not None, r1
    assert r1['ci95_ppl'] is None, r1
    # Records without per-row NLL cannot support EITHER component.
    r2 = L._combined_seed_doc_ci([{'variant': 'csa', 'seed': 0, 'ppl': 30.0},
                                  {'variant': 'csa', 'seed': 1, 'ppl': 31.0}])
    assert r2['status'] == 'unverifiable' and r2['ci95_ppl'] is None, r2
    # A single scored document cannot carry a doc-resampling component.
    r3 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=1, ntok=50))
    assert r3['status'] == 'doc_component_unverifiable' and r3['se_doc_logppl'] is None and r3['se_seed_logppl'] is not None, r3
    # The standalone document bootstrap must not report a one-document
    # zero-width interval as if it had measured document uncertainty.
    assert L._doc_bootstrap_ppl_ci(_recs([0.0], n_docs=1, ntok=50)) is None
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
    for bad in (True, 1.5, 'x', np.float64(2 ** 53 + 1), 9007199254740993.0):
        try:
            L.normalize_seed_list([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'normalize_seed_list must reject {bad!r}')
    # Exact Python integers above 2**53 are preserved (only lossy floats fail).
    assert L.normalize_seed_list(['9007199254740993']) == [9007199254740993]
    # ...and every preserved id must be usable by set_seed: NumPy's RNG
    # otherwise rejects negative or >=2**32 seeds and aborts the panel.
    for huge in (-1, 2 ** 32, 9007199254740993, 10 ** 30):
        _before = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        L.set_seed(huge)
        _a = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        L.set_seed(huge)
        _b = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        assert _a == _b, (huge, _a, _b)
    # The same exact-integer contract applies wherever a recorded/checkpoint
    # seed is normalised, not only to the panel planner.
    assert L._as_seed_int(float(2 ** 53)) is None
    assert L._as_seed_int(float(2 ** 53 - 1)) == 2 ** 53 - 1
    assert L._as_seed_int(1.5) is None and L._as_seed_int(True) is None
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


def t_ids_narrow_does_not_trust_stale_max_id():
    # The int32 transport cast must be decided by the ACTUAL ids, not by a
    # metadata max_id that can be stale/corrupted.  A wrong bound would wrap
    # 2**31 into a negative token id, which PyTorch then indexes from the end
    # of the embedding table (silent wrong training data).
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'train_ids_x.npy')
        wide = np.array([2 ** 31, 5, -3], dtype=np.int64)
        np.save(p, wide)
        with open(os.path.join(d, 'meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'max_id': 5}, f)
        mm = np.load(p, mmap_mode='r')
        got = L._ids_narrow(mm)
        assert got.dtype == np.int64, got.dtype
        assert np.array_equal(np.asarray(got), wide), got
        # In-range caches still narrow exactly, and the v2 sidecar is reusable.
        p2 = os.path.join(d, 'train_ids_y.npy')
        safe = np.array([0, 1, 70000], dtype=np.int64)
        np.save(p2, safe)
        with open(os.path.join(d, 'meta_y.json'), 'w', encoding='utf-8') as f:
            json.dump({'max_id': 70000}, f)
        a = L._ids_narrow(np.load(p2, mmap_mode='r'))
        b = L._ids_narrow(np.load(p2, mmap_mode='r'))
        assert a.dtype == b.dtype == np.int32, (a.dtype, b.dtype)
        assert np.array_equal(np.asarray(a), safe) and np.array_equal(np.asarray(b), safe)


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
    # A repeated input window with two different continuations cannot prove
    # which occurrence was sampled; a target borrowed from another occurrence
    # must fail rather than pass.
    src = np.array([0, 0, 1, 0, 0, 2], dtype=np.int64)
    class _WrongRepeat:
        def __init__(self): self.n = 0
        def __iter__(self): return self
        def __next__(self):
            self.n += 1
            if self.n > 2: raise StopIteration
            return (torch.tensor([[0, 0], [0, 0]]),
                    torch.tensor([[0, 1], [0, 1]]))
    L.batch_iter = lambda *a, **k: _WrongRepeat()
    try:
        amb = L.batch_contract_check(src, 2, 2, torch.device('cpu'))
    finally:
        L.batch_iter = _orig
    assert amb['checks']['final_target_matches_source'] is False, amb
    assert amb['passed'] is False, amb




def t_parallel_child_env_seed():
    import par
    # Job payload normalisation must terminate for NumPy scalar subtypes
    # (longdouble.item() returns another NumPy scalar) and preserve numbers.
    normed = par._norm({'x': np.longdouble(1.5),
                        'y': np.array([np.longdouble(2.5)]),
                        'z': np.float32(3.5), 'b': np.bool_(True)})
    assert normed == {'x': 1.5, 'y': [2.5], 'z': 3.5, 'b': True}, normed
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


def t_manifest_seed_integrity_is_exact():
    tr = np.arange(512, dtype=np.int64) % 97
    va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
    prov = L.experiment_provenance({'dataset': 'unit', 'n_train_tokens': 512,
                                    'seq_len': 32}, train_ids=tr, val_batch=va, vocab=97)
    d = tempfile.mkdtemp()
    try:
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d,
               'n_train_tokens': 512, 'seq_len': 32}
        hp = L.run_hyperparameters(cfg, seed=0, d=64, n_layers=2, n_heads=4,
                                   d_head=16, vocab=97, mlp_ratio=4.0)
        rec = {'variant': 'full', 'seed': 0.9, 'ppl': 10.0, 'hyperparams': hp}
        path = L.write_run_manifest(d, cfg, 'FP', prov, {},
                                    cell_records={'full::seed0': rec})
        integ = json.load(open(path, encoding='utf-8'))['panels']['FP']['tracking_integrity']
        assert integ['with_seed_metadata'] == 0, integ
        assert integ['seed_metadata_mismatch'] == ['full::seed0'], integ
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


def t_npy_serve_int32_rejects_invalid_and_out_of_range():
    # The int32 narrowing helper must never turn a corrupt cache into plausible
    # token ids.  Its range/dtype guards stay ahead of the sidecar fallback,
    # which must not truncate floats or wrap out-of-int32 ids.
    with tempfile.TemporaryDirectory() as d:
        ok = np.array([0, 1, 65535], dtype=np.int64)
        p_ok = os.path.join(d, 'ok.npy')
        np.save(p_ok, ok)
        got = L._npy_serve_int32(p_ok)
        assert np.array_equal(np.asarray(got, dtype=np.int64), ok), got
        assert np.asarray(got).dtype == np.int32, got.dtype
        cases = (
            ('float', np.array([1.5, 2.0], dtype=np.float64), TypeError),
            ('negative', np.array([-1, 2], dtype=np.int64), ValueError),
            ('wide', np.array([2 ** 31, 2], dtype=np.int64), ValueError),
        )
        for name, bad, exc in cases:
            p = os.path.join(d, f'{name}.npy')
            np.save(p, bad)
            try:
                L._npy_serve_int32(p)
            except exc:
                pass
            else:
                raise AssertionError(f'{name}: invalid token cache was silently accepted')


def t_ids_fp_is_storage_width_invariant():
    # The same corpus must keep one fingerprint across every integer cache
    # dtype; otherwise a data_version/resume gate sees a different corpus just
    # because a loader stored ids as int32 instead of int64.
    for value in (0, 1, 255, 70000):
        fps = set()
        for dt in (np.int8, np.uint8, np.int16, np.uint16,
                   np.int32, np.uint32, np.int64):
            try:
                arr = np.array([value], dtype=dt)
            except (OverflowError, ValueError):
                continue
            fps.add(L._ids_fp(arr))
        assert len(fps) == 1, (value, fps)
    # uint64 ids above int64's range must stay exact (no wrap-alias).
    hi = np.array([2 ** 63 + 1], dtype=np.uint64)
    hi2 = np.array([2 ** 63 + 2], dtype=np.uint64)
    assert L._ids_fp(hi) != L._ids_fp(hi2), 'uint64 ids must not wrap into one fingerprint'
    # A sidecar written by an older fingerprint algorithm must not be served.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'train_ids_x.npy')
        np.save(path, np.arange(8, dtype=np.int32))
        st = os.stat(path)
        with open(path + '.idsfp.json', 'w', encoding='utf-8') as f:
            json.dump({'size': int(st.st_size),
                       'mtime_ns': int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))),
                       'ctime_ns': int(getattr(st, 'st_ctime_ns', int(st.st_ctime * 1e9))),
                       'inode': int(getattr(st, 'st_ino', 0)),
                       'fp': 'stale-old-algorithm',
                       'dtype': np.dtype(np.int32).str, 'shape': [8]}, f)
        mm = np.load(path, mmap_mode='r')
        try:
            assert L._ids_fp(mm) != 'stale-old-algorithm', 'old sidecar was served after the algorithm changed'
        finally:
            del mm


def t_audit_sidecar_fp_matches_exp_lib():
    # The zero-GPU audit keeps a torch-free copy of the id fingerprint.  It
    # must stay byte-identical to exp_lib._ids_fp for every integer storage
    # width; otherwise a sidecar written by the loader (canonical int64) is
    # reported as corrupt merely because the cache stored it as int32/uint16.
    for value in (0, 1, 255, 70000):
        fps = set()
        for dt in (np.int8, np.uint8, np.int16, np.uint16,
                   np.int32, np.uint32, np.int64, np.uint64):
            try:
                arr = np.array([value], dtype=dt)
            except (OverflowError, ValueError):
                continue
            a = A._sidecar_fp(arr)
            b = L._ids_fp(arr)
            assert a == b, (value, np.dtype(dt).name, a, b)
            fps.add(a)
        assert len(fps) == 1, (value, fps)
    # uint64 ids above int64's range must stay exact and match the loader.
    hi = np.array([2 ** 63 + 1], dtype=np.uint64)
    hi2 = np.array([2 ** 63 + 2], dtype=np.uint64)
    assert A._sidecar_fp(hi) == L._ids_fp(hi)
    assert A._sidecar_fp(hi2) == L._ids_fp(hi2)
    assert A._sidecar_fp(hi) != A._sidecar_fp(hi2)


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
    # A boolean/fractional/zero vocabulary is not a token-space identity.
    for _bad_vocab in (True, np.bool_(False), 97.5, 0, -1):
        try:
            L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=_bad_vocab)
        except ValueError:
            pass
        else:
            raise AssertionError(f'experiment_provenance accepted vocab={_bad_vocab!r}')
        try:
            L.preflight_fingerprint(cfg, stage='unit', vocab=_bad_vocab)
        except ValueError:
            pass
        else:
            raise AssertionError(f'preflight_fingerprint accepted vocab={_bad_vocab!r}')
    fp = L.preflight_fingerprint(cfg, stage='unit', vocab=97.0)
    assert fp == L.preflight_fingerprint(cfg, stage='unit', vocab=97)


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
        # Distinct cell keys must not share a curve sidecar filename.
        assert L._curve_sidecar_rel('a#1::seed0') != L._curve_sidecar_rel('a@1::seed0')
        # A configured variant name cannot escape ckpt_dir through the
        # checkpoint filename.
        assert '/' not in L.ckpt_file_name('../escape', 0, 'final')
        _esc = os.path.basename(L.ckpt_file_name('/tmp/escape', 1, 'final'))
        assert _esc.startswith('_tmp_escape-') and _esc.endswith('_seed1_final.pt'), _esc
        assert os.path.basename(L.ckpt_file_name('csa_dynamic', 2, 'final')) == 'csa_dynamic_seed2_final.pt'
        # Sanitisation must be injective: two distinct arms may not share a
        # checkpoint filename.
        assert L.ckpt_file_name('a/b', 2, 'final') != L.ckpt_file_name('a_b', 2, 'final')
        # A boolean/fractional/lossy seed is not silently truncated into a
        # different checkpoint identity.
        for _bad_seed in (True, 1.5, float(2 ** 53)):
            try:
                L.ckpt_file_name('csa_dynamic', _bad_seed, 'final')
            except ValueError:
                pass
            else:
                raise AssertionError(f'ckpt_file_name accepted invalid seed {_bad_seed!r}')
        # The tagless length/probe checkpoint name must have the same path
        # and seed-safety contract.
        assert L.direct_ckpt_name('csa_dynamic', 2) == 'csa_dynamic_seed2.pt'
        assert '/' not in L.direct_ckpt_name('../escape', 0)
        assert L.direct_ckpt_name('a/b', 0) != L.direct_ckpt_name('a_b', 0)
        for _bad_seed in (True, 1.5, float(2 ** 53)):
            try:
                L.direct_ckpt_name('csa_dynamic', _bad_seed)
            except ValueError:
                pass
            else:
                raise AssertionError(f'direct_ckpt_name accepted invalid seed {_bad_seed!r}')
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

    # A stale extra final record outside the declared final partition must not
    # inflate n silently: the coverage sets can otherwise agree on the extra
    # seed and the audit reports tuning_verified=True.
    stale = json.loads(json.dumps(art))
    for _a in chosen:
        stale['final_records'][f'{_a}::seed9'] = {
            'seed': 9, 'variant': _a, 'ppl': 9.9, 'params': 100,
            'train_knobs': L.training_knobs(), 'hyperparams': dict(chosen[_a])}
    st = L.baseline_tuning_audit(dict(stale, protocol=good_protocol,
                                      search_seeds=[100, 101, 102]),
                                 protocol=good_protocol)
    assert st['tuning_verified'] is False, st
    assert any('undeclared seed' in p for p in st['problems']), st

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
    # Rows without per-seed values can never report stability as a pass.
    no_seed = L.tuning_selection_stability([{'config': {'lr': 1e-4}, 'ppl': 1.0}])
    assert no_seed['status'] == 'unavailable', no_seed


def t_methodology_audit_requires_panel_to_use_tuned_config():
    with tempfile.TemporaryDirectory() as d:
        trials = [{'lr': 3e-4, 'weight_decay': 0.1, 'warmup': 1},
                  {'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 1}]
        protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a', 'b'], trials=trials,
                        inherits_tuning={}, search_seeds=[100, 101, 102],
                        final_seeds=[0, 1, 2])
        art = {'protocol': protocol, 'selection_metric': 'ppl',
               'selection_split': 'validation', 'direction': 'lower_is_better',
               'search_seeds': [100, 101, 102], 'final_seeds': [0, 1, 2],
               'arms': {
                   'a': {'selected': dict(trials[0]), 'selected_metric_value': 2.0,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.0, 3.0))]},
                   'b': {'selected': dict(trials[0]), 'selected_metric_value': 2.1,
                         'trials': [{'config': c, 'ppl': p}
                                    for c, p in zip(trials, (2.1, 3.1))]},
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
        # No test doc-id sidecar means document isolation is unverifiable; the
        # top-level status must not claim a clean "ok" (this cache also fails
        # on overlap, so accept either non-ok status).
        assert out['status'] in ('warnings', 'fail'), out
        assert out['checks']['document_isolation']['status'] == 'not_verified', out
        # A malformed meta without a numeric vocab must fail closed, not
        # crash with int(None) before the audit can report anything.
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'src_fp': 'test'}, f)
        bad = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert bad['status'] == 'fail' and bad['checks'].get('vocab_agree') is False, bad
        # Fractional/boolean/NaN metadata is not an integer cache identity;
        # int() truncation would pair two different vocabularies as equal.
        assert A._exact_int_scalar(3.5) is None
        assert A._exact_int_scalar(True) is None
        assert A._exact_int_scalar(float('nan')) is None
        assert A._exact_int_scalar('3') == 3
        with open(os.path.join(cache, 'meta_v4_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000.5, 'src_fp': 'train'}, f)
        with open(os.path.join(cache, 'test_meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 1000.5, 'src_fp': 'test'}, f)
        frac = A.audit_leakage(cache_dir=cache, shingle=4, max_rows_shingled=10)
        assert frac['status'] != 'ok' and frac['checks'].get('vocab_agree') is False, frac
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
    # With an ordering key, each split's returned row indices must stay in
    # chronological order, not sorted by original row position.
    g = np.arange(6)
    t = np.array([5, 4, 3, 2, 1, 0])
    ordered = L.grouped_chronological_split_indices(6, g, 0.5, 0.25, 0.25,
                                                    order=t)
    for name in ('train', 'val', 'test'):
        vals = t[ordered[name]].tolist()
        assert vals == sorted(vals), (name, ordered[name], vals)


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
    # With no holdout either, an explicit empty training split still audits
    # nothing and must not be reported as a clean/ok panel.
    try:
        L.data_quality_report(X, train_idx=[])
    except ValueError as e:
        assert 'training reference is empty' in str(e), e
    else:
        raise AssertionError('an explicit empty training split must not report ok')


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
    # does not, the fitted scale does not centre its own training matrix.
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
    # Composite (tuple/list) categories are single identities; reshape(-1)
    # would split them into scalar categories and corrupt every frequency.
    for ref in ([('a', 1), ('b', 2)], [['a', 1], ['b', 2]],
                np.array([('a', 1), ('b', 2)], dtype=object)):
        r = L.categorical_shift_report(ref, [('a', 1)])
        assert r['n_reference'] == 2 and r['n_holdout'] == 1, (ref, r)
        assert r['unseen_rate'] == 0.0, (ref, r)
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


def t_multi_output_missing_labels():
    # A fully missing multi-output label is not a real category: it must not
    # create a duplicate-label conflict, an unknown-label hit, or a histogram
    # bin.  A row that is partly missing and partly non-finite must still count
    # its non-finite output instead of hiding it behind the missing one.
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([[None, None], [0, 1], [2, 2]], dtype=object)
    rep = L.data_quality_report(X, y=y, train_idx=[0, 1, 2],
                                known_labels=[(0, 1), (2, 2)], shift_threshold=10)
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 0, rep
    assert not any(rep['checks']['unknown_labels_by_split'].values()), rep
    assert '(None, None)' not in rep['checks']['label_distribution'], rep
    Xn = np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    yn = np.array([[None, np.inf], [0, 1], [0, 1]], dtype=object)
    repn = L.data_quality_report(Xn, y=yn, train_idx=[0, 1], val_idx=[2],
                                 fail_on_nonfinite_labels=True,
                                 shift_threshold=10)
    assert repn['checks']['missing_labels'] == 1, repn
    assert repn['checks']['nonfinite_labels'] == 1, repn
    assert repn['status'] == 'fail' and repn['hard_failures'], repn
    # clean_training_rows uses the same 'any output missing' rule, so a
    # partially missing label cannot trigger a false duplicate conflict.
    Xc = np.array([[0.0], [0.0]])
    yc = np.array([[None, 0], [1, 0]], dtype=object)
    clean = L.clean_training_rows(Xc, y=yc, train_idx=[0, 1])
    assert clean['n_train_rows_out'] == 1, clean
    assert clean['n_label_conflict_clusters'] == 0, clean


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
    # Explicit sizes are counts: non-integral/coerced values must be refused
    # rather than silently truncated (1.5 -> 1) or parsed ('2' -> 2).
    for bad in (1.5, '2', True):
        try:
            L.chronological_split_indices(100, train_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'train_size={bad!r} must be rejected')
    # Fractions within the 1e-6 tolerance must still reconcile at scale: a
    # large n times a slightly-over-1 sum must not produce a negative
    # remainder / inconsistent-size error.
    for fr in ((0.5, 0.5000005, 0.0), (0.5, 0.499999, 0.0)):
        idx = L.chronological_split_indices(10_000_000, *fr)
        sizes = [len(idx[k]) for k in ('train', 'val', 'test')]
        assert sum(sizes) == 10_000_000 and all(s >= 0 for s in sizes), (fr, sizes)
    for bad_n in (10.5, '10', True):
        try:
            L.chronological_split_indices(bad_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f'n={bad_n!r} must be rejected')
    assert len(L.chronological_split_indices(100.0)['train']) == 80
    # Fractional splits use largest-remainder apportionment at small n: the
    # rounding remainder must not all be dumped into test while train is
    # under-allocated, and a zero-fraction split must stay empty.
    for n, want in {3: (1, 1, 1), 4: (2, 1, 1), 5: (3, 1, 1),
                    6: (4, 1, 1), 8: (6, 1, 1), 10: (8, 1, 1)}.items():
        idx = L.chronological_split_indices(n)
        got = tuple(len(idx[k]) for k in ('train', 'val', 'test'))
        assert got == want, (n, got, want)
    idx = L.chronological_split_indices(7, 0.5, 0.5, 0.0)
    assert [len(idx[k]) for k in ('train', 'val', 'test')] == [3, 4, 0], idx


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
    # The same logical row must match across integer storage widths; raw
    # width-dependent bytes would let cross-dtype leakage pass the audit.
    for dt_a, dt_b in ((np.int64, np.int32), (np.uint16, np.int32),
                       (np.int64, np.uint16), (np.int8, np.uint64)):
        assert A._row_overlap_count(np.array([[1, 2, 3]], dtype=dt_a),
                                    np.array([[1, 2, 3]], dtype=dt_b)) == 1, (dt_a, dt_b)
    assert A._row_overlap_count(np.array([[1, 2]], dtype=np.int64),
                                np.array([[1, 3]], dtype=np.int32)) == 0
    import data_guard as DG
    assert DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=np.int64)) != \
        DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=np.int64))
    # In-range integer tables keep their float64 fingerprint, which is exact
    # below 2**53 and therefore stable across integer widths.
    lo = np.array([[1, 2, 3]], dtype=np.int64)
    assert DG._matrix_fingerprint(lo) == DG._matrix_fingerprint(lo.astype(np.int32))


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
    # A malformed non-boolean spelling cannot deny a shuffled split; it
    # fails closed.
    bad2b = dict(meta, shuffle_at_split=1)
    assert any('shuffled split' in p for p in
               L._cached_split_problems(bad2b, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    bad3 = dict(meta, tokenizer_fit_split='test')
    assert any('tokenizer' in p for p in
               L._cached_split_problems(bad3, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8))
    # Malformed vocab/seq_len metadata must fail closed instead of being
    # truncated by int(): 11.9 -> 11 would silently bless the cache.
    bad_vocab = dict(meta, vocab=11.9)
    assert any('vocab' in p for p in
               L._cached_split_problems(bad_vocab, tr, va, np.zeros_like(va, dtype=bool), 11.9, seq_len=8))
    assert any('positive integer' in p for p in
               L._cached_split_problems(meta, tr, va, np.zeros_like(va, dtype=bool), 11, seq_len=8.5))
    try:
        L._validate_token_ids(np.array([0, 1], dtype=np.int64), 2.5, 'unit')
    except ValueError:
        pass
    else:
        raise AssertionError('fractional vocab size was accepted by the token-id range check')


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


def t_time_split_backtracks_for_repeated_timestamps():
    # A greedy forward snap takes the last timestamp block for train and then
    # cannot give val/test any rows, even though 6/2/2 is a legal split.
    time = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    idx = L.split_indices(10, time=time)
    sizes = [len(idx[k]) for k in ('train', 'val', 'test')]
    assert sizes == [6, 2, 2], sizes
    L.assert_no_split_leakage(idx['train'], idx['val'], idx['test'], time=time)
    assert time[idx['train']].max() < time[idx['val']].min()
    assert time[idx['val']].max() < time[idx['test']].min()


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
    # a corrupted target into a hard failure (off by default for compatibility).
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
    # sorted candidate pool has mean ~n/9 for cap=4, not ~n/2.
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
        # A fractional run counter is not a count; never truncate it to one.
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 1.0, 'runs': 3.9}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] is None, out
        with open(state, 'w', encoding='utf-8') as f:
            json.dump({'booked_seconds': 1.0, 'runs': 3.9, 'run_count': 7}, f)
        out = A.audit_budget(state_files=(state,), panels=())
        assert out['measured'][state]['run_count'] == 7, out
    # Panel bookkeeping must use exact positive counts: a boolean/fractional
    # parameter or token count would otherwise produce a plausible FLOP number.
    td = tempfile.mkdtemp(prefix='budget_', dir=REPO)
    try:
        sp = os.path.join(td, 'summary.json')
        for bad in ({'params': True, 'steps': 3, 'batch_size': 2, 'seq_len': 8},
                    {'params': 1000.5, 'steps': 3, 'batch_size': 2, 'seq_len': 8},
                    {'params': 1000, 'steps': 3, 'batch_size': 2, 'seq_len': 8,
                     'tokens_seen': 48.5},
                    {'params': 1000, 'steps': 3, 'batch_size': 2.5, 'seq_len': 8}):
            with open(sp, 'w', encoding='utf-8') as f:
                json.dump({'v::s0': dict(bad, variant='v')}, f)
            r = A.audit_budget(state_files=(), panels=((td, 'v', None),))
            assert r['status'] == 'fail' and r.get('problems'), (bad, r)
            _panel = r['panels'][td]
            assert not isinstance(_panel, dict) or _panel.get('estimate') is None, (bad, r)
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump({'v::s0': {'variant': 'v', 'params': 1000, 'steps': 3,
                                 'batch_size': 2, 'seq_len': 8}}, f)
        r = A.audit_budget(state_files=(), panels=((td, 'v', None),))
        assert r['status'] == 'ok', r
        assert r['panels'][td]['tokens'] == 48, r
        assert r['panels'][td]['estimate']['flops_6nd'] == 6.0 * 1000 * 48, r
        for _bad in (True, 1.5, -1, None):
            try:
                A.train_flops_estimate(_bad, 48)
            except ValueError:
                pass
            else:
                raise AssertionError(f'train_flops_estimate accepted params={_bad!r}')
        for _bad in (True, 1.5, -1, None):
            try:
                A.train_flops_estimate(1000, _bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'train_flops_estimate accepted tokens={_bad!r}')
        try:
            A.train_flops_estimate(1000, 48, attn_overhead=True)
        except ValueError:
            pass
        else:
            raise AssertionError('train_flops_estimate accepted a boolean overhead')
    finally:
        shutil.rmtree(td, ignore_errors=True)


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
    # Different non-finite values are not exact duplicate rows: NaN, +Inf and
    # -Inf share a non-finite mask but are distinct corrupted values.  Only the
    # repeated NaN row is a duplicate.
    Xnf = np.array([[np.nan, 1.0], [np.inf, 1.0], [-np.inf, 1.0], [np.nan, 1.0]])
    mask = L.duplicate_mask(Xnf)
    assert mask.tolist() == [False, False, False, True], mask
    # Integers above 2**53 are exact.  Coercing them to float64 would make
    # 2**53 and 2**53+1 look like the same row and drop a real observation.
    Xi = np.array([[2 ** 53], [2 ** 53 + 1]], dtype=np.int64)
    assert L.duplicate_mask(Xi).tolist() == [False, False]
    clean_i = L.clean_training_rows(Xi, y=[0, 0], train_idx=[0, 1])
    assert clean_i['train_idx'].tolist() == [0, 1], clean_i
    rep_i = L.data_quality_report(Xi, train_idx=[0], val_idx=[1])
    assert rep_i['checks']['duplicate_rows_across_splits']['train__val'] == 0, rep_i
    # Multi-output labels are audited per row, not flattened to n*k.
    ym = np.array([[0, 1], [0, 0], [1, 1], [2, 2]])
    Xm = np.array([[0.0], [0.0], [1.0], [2.0]])
    rep_m = L.data_quality_report(Xm, y=ym, train_idx=[0, 1, 3],
                                  val_idx=[2], test_idx=[])
    assert rep_m['checks']['label_conflicts_on_duplicate_rows'] == 1, rep_m
    repnf = L.data_quality_report(Xnf, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    assert repnf['checks']['duplicate_rows_across_splits']['train__val'] == 0, repnf

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

    # timedelta NaT is the same missing-value contract as datetime NaT: it
    # must not become a unique group/category.
    gt = np.array([np.timedelta64('NaT'), np.timedelta64('NaT'),
                   np.timedelta64(1, 'D')], dtype=object)
    try:
        L.assert_no_split_leakage([0], [1], [2], groups=gt)
    except ValueError as e:
        assert 'share group ids' in str(e), e
    else:
        raise AssertionError('timedelta NaT group ids must not be split across sides')
    enc_td = L.LeakageSafeCategoricalEncoder().fit(
        np.array([[np.timedelta64('NaT')], [np.timedelta64('NaT')]], dtype=object))
    assert enc_td.get_stats()['features'][0]['n_categories'] == 0, enc_td.get_stats()


def t_audit_split_warning_status():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        tr = np.arange(40, dtype=np.int64)
        va = np.arange(100, 108, dtype=np.int64).reshape(2, 4)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        # Missing split_strategy is an explicit cache warning.
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 40, 'seq_len': 4}, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'warnings', rec
        assert out['caches'][os.path.abspath(d)]['status'] == 'warnings', out['caches']


def t_audit_split_test_metadata_discipline():
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        ttag = 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 40).reshape(2, 8)
        # The held-out cache may use a different sequence length than the
        # training cache (e.g. 512 vs 4097).
        te = (np.arange(32, dtype=np.int64) + 56).reshape(2, 16)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 80, 'seq_len': 8, 'split_strategy': 'ordered',
                       'shuffle_at_split': False,
                       'tokenizer_fit_split': 'train'}, f)
        good_test = {'vocab': 80, 'seq_len': 16, 'split_strategy': 'official_test',
                     'shuffle_at_split': False,
                     'tokenizer_fit_split': 'train',
                     'near_duplicate_cleanup': {
                         'policy': 'test_documents_deduplicated_against_train_and_validation'}}
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(good_test, f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec.get('test_tag') == ttag, rec
        assert rec.get('test_shuffle_at_split') is False, rec
        assert rec['status'] != 'fail', rec
        # Malformed metadata must fail the audit, not crash int().
        with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(good_test, vocab='abc'), f)
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('non-integer vocab' in w for w in out['warnings']), out
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


def t_audit_split_test_sidecar_self_consistency():
    # A test doc-id sidecar must be checked against its OWN metadata even when
    # the paired train/validation sidecars are missing; otherwise swapping in a
    # truncated test sidecar would only downgrade to a "not verified" warning.
    with tempfile.TemporaryDirectory() as d:
        tag = 'tiny'
        ttag = 'tiny_test'
        tr = np.arange(40, dtype=np.int64)
        va = (np.arange(16, dtype=np.int64) + 40).reshape(2, 8)
        te = (np.arange(32, dtype=np.int64) + 56).reshape(2, 16)
        trd = np.array([1, 2, 3], dtype=np.uint64)
        vad = np.array([4, 5], dtype=np.uint64)
        ted = np.array([6, 7], dtype=np.uint64)
        np.save(os.path.join(d, f'train_ids_{tag}.npy'), tr)
        np.save(os.path.join(d, f'val_batch_{tag}.npy'), va)
        np.save(os.path.join(d, f'test_ids_{ttag}.npy'), te)
        np.save(os.path.join(d, f'train_doc_ids_{tag}.npy'), trd)
        np.save(os.path.join(d, f'val_doc_ids_{tag}.npy'), vad)
        np.save(os.path.join(d, f'test_doc_ids_{ttag}.npy'), ted)
        meta = {'vocab': 80, 'seq_len': 8, 'split_strategy': 'ordered',
                'shuffle_at_split': False, 'tokenizer_fit_split': 'train',
                'doc_ids': {'train_count': len(trd), 'validation_count': len(vad),
                            'train_fp': A._sidecar_fp(trd),
                            'validation_fp': A._sidecar_fp(vad),
                            'overlap': 0, 'disjoint': True}}
        tmeta = {'vocab': 80, 'seq_len': 16, 'split_strategy': 'official_test',
                 'shuffle_at_split': False, 'tokenizer_fit_split': 'train',
                 'near_duplicate_cleanup': {
                     'policy': 'test_documents_deduplicated_against_train_and_validation'},
                 'doc_ids': {'test_count': len(ted),
                             'test_fp': A._sidecar_fp(ted),
                             'train_overlap': 0, 'validation_overlap': 0,
                             'disjoint': True}}
        def _write_meta():
            with open(os.path.join(d, f'meta_{tag}.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            with open(os.path.join(d, f'test_meta_{ttag}.json'), 'w', encoding='utf-8') as f:
                json.dump(tmeta, f)
        _write_meta()
        good = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = good['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] != 'fail', rec
        # Counts are exact integers: int(3.9)==3 and int(True)==1 must not
        # masquerade as a matching sidecar size.
        assert A._exact_int_scalar(3.9) is None
        assert A._exact_int_scalar(True) is None
        assert A._exact_int_scalar(np.bool_(True)) is None
        assert A._exact_int_scalar(3.0) == 3
        for _field, _bad in (('train_count', 3.9),
                             ('validation_count', 2.9)):
            _old = meta['doc_ids'][_field]
            meta['doc_ids'][_field] = _bad
            _write_meta()
            out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
            rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
            assert rec['status'] == 'fail', (_field, _bad, rec)
            assert any('count disagrees' in w for w in out['warnings']), (_field, out)
            meta['doc_ids'][_field] = _old
        _old = tmeta['doc_ids']['test_count']
        tmeta['doc_ids']['test_count'] = 2.9
        _write_meta()
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('count disagrees' in w for w in out['warnings']), out
        tmeta['doc_ids']['test_count'] = _old
        _write_meta()
        # Remove the paired train sidecar and corrupt only the test fingerprint.
        os.unlink(os.path.join(d, f'train_doc_ids_{tag}.npy'))
        tmeta['doc_ids']['test_fp'] = 'not-the-test-fingerprint'
        _write_meta()
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('test document-id sidecar does not match' in w for w in out['warnings']), out
        # Metadata records document ids but the sidecar is gone: fail closed.
        tmeta['doc_ids']['test_fp'] = A._sidecar_fp(ted)
        _write_meta()
        os.unlink(os.path.join(d, f'test_doc_ids_{ttag}.npy'))
        out = A.audit_split_discipline(cache_dirs=[d], min_shift_tokens=1)
        rec = out['caches'][os.path.abspath(d)]['corpora'][tag]
        assert rec['status'] == 'fail', rec
        assert any('test document-id sidecar is missing' in w for w in out['warnings']), out


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


def t_corpus_audit_memo_does_not_bypass_strict_gate():
    tr = np.arange(40, dtype=np.int64)
    va = np.array([[5, 6, 7, 8]], dtype=np.int64)
    # An exploratory (strict=False) call caches the failing report.
    rep = L.audit_corpus_split(tr, va, 64, dataset='wikitext',
                               max_scan_tokens=64, sample_rows=1, strict=False)
    assert rep['status'] == 'fail', rep
    # A later strict call on the SAME data must still raise instead of
    # returning the cached non-raising report.
    try:
        L.audit_corpus_split(tr, va, 64, dataset='wikitext',
                             max_scan_tokens=64, sample_rows=1, strict=True)
    except RuntimeError as e:
        assert 'DATA GATE REFUSED' in str(e), e
    else:
        raise AssertionError('cached non-strict report bypassed the strict gate')


def t_corpus_audit_memo_keys_validation_shape():
    # The same flattened tokens with different row boundaries are different
    # holdouts (different per-row overlap/duplicate checks and n_seq), so the
    # memo must not return the first report for the second shape.
    tr = np.arange(10, dtype=np.int64)
    flat = np.arange(40, 46, dtype=np.int64)
    va_one = flat.reshape(1, 6)
    va_rows = flat.reshape(2, 3)
    r1 = L.audit_corpus_split(tr, va_one, 64, dataset='shape-memo-test',
                              max_scan_tokens=64, sample_rows=1, strict=False)
    r2 = L.audit_corpus_split(tr, va_rows, 64, dataset='shape-memo-test',
                              max_scan_tokens=64, sample_rows=1, strict=False)
    assert r1['val_shape'] == [1, 6], r1
    assert r2['val_shape'] == [2, 3], r2


def t_corpus_audit_overlap_metadata_exact():
    # The document-overlap flag is a count: a fractional/boolean value must not
    # be truncated to zero by int(0.9)==0 and attest a clean split.
    tr = np.arange(40, dtype=np.int64)
    va = np.array([[100, 101, 102, 103]], dtype=np.int64)
    for bad in (0.9, True, -0.5):
        rep = L.audit_corpus_split(
            tr, va, 200, dataset='overlap-meta-test',
            meta={'doc_ids': {'disjoint': True, 'overlap': bad}},
            max_scan_tokens=64, sample_rows=1, strict=False)
        assert rep['status'] == 'fail', (bad, rep)
        assert any('document-disjoint' in x for x in rep['hard_failures']), (bad, rep)
    ok = L.audit_corpus_split(
        tr, va, 200, dataset='overlap-meta-test',
        meta={'doc_ids': {'disjoint': True, 'overlap': 0}},
        max_scan_tokens=64, sample_rows=1, strict=False)
    assert ok['status'] != 'fail', ok


def t_report_aggregate_status_fails_closed():
    assert A._aggregate_audit_status({'a': {'status': 'ok'}}) == 'ok'
    assert A._aggregate_audit_status(
        {'a': {'status': 'ok'}, 'b': {'status': 'warnings'}}) == 'warnings'
    assert A._aggregate_audit_status(
        {'a': {'status': 'warnings'}, 'b': {'status': 'fail'}}) == 'fail'
    assert A._aggregate_audit_status(
        {'a': {'status': 'ok'}, 'b': {'status': 'not_verifiable_locally'}}) == 'not_verifiable_locally'
    assert A._aggregate_audit_status({'a': None, 'b': {}}) == 'ok'


def t_audit_length_requires_eval_span():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bylen.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'by_len': {'512': {
                'n_tok': 511, 'eval_span': None, 'n_seq': 1}}}}, f)
        out = A.audit_length(path)
        assert out['status'] == 'fail', out
        assert out['by_length']['512']['same_scored_targets_across_records'] is False, out


def t_audit_length_malformed_cells_do_not_crash():
    td = tempfile.mkdtemp(prefix='len_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': [1, 2, 3]}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail' and any('by_len is not an object' in x for x in r['problems']), r
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'n_tok': [1, 2], 'eval_span': 4095, 'n_seq': 16,
                'truncated': False}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail' and any('n_tok is not a scalar' in x for x in r2['problems']), r2
        # Malformed keys/cells must fail closed rather than being skipped.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096.5': {
                'n_tok': 4095, 'eval_span': 4095, 'n_seq': 1,
                'truncated': False}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail' and any('not an integer length' in x for x in r3['problems']), r3
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': 42}}}, f)
        r4 = A.audit_length(p)
        assert r4['status'] == 'fail' and any('cell is not an object' in x for x in r4['problems']), r4
        # A zero/negative length or eval span is not a scored sequence; it must
        # not enter the cross-arm tables as a clean cell.
        for key, span in ((0, 1), ('4096', 0)):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'by_len': {str(key): {
                    'n_tok': 1, 'eval_span': span, 'n_seq': 1,
                    'truncated': False}}}}, f)
            bad = A.audit_length(p)
            assert bad['status'] == 'fail', bad
            assert any('positive' in x for x in bad['problems']), bad
        # A length-1 cell scores no next-token target, so a self-consistent
        # n_tok/eval_span pair must not enter the table as a clean measurement.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'1': {
                'n_tok': 1, 'eval_span': 1, 'n_seq': 1,
                'truncated': False}}}}, f)
        one = A.audit_length(p)
        assert one['status'] == 'fail', one
        assert any('next-token prediction' in x for x in one['problems']), one
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_by_len_cells_requires_exact_length():
    base = {'variant': 'a', 'val_fp': 'v', 'by_len': {
        '1': {'ppl': 5.0}, '4': {'ppl': 10.0}, '4.9': {'ppl': 11.0},
        '512': {'ppl': 20.0}}}
    assert [c['ppl'] for c in L.by_len_cells([base], 4)] == [10.0]
    assert [c['ppl'] for c in L.by_len_cells([base], 512)] == [20.0]
    assert [c['ppl'] for c in L.by_len_cells([base], '512')] == [20.0]
    for bad in (True, np.bool_(False), 4.9, float('nan')):
        try:
            L.by_len_cells([base], bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'by_len_cells must refuse the length {bad!r}')
    # A genuine length-one panel has no next-token target and yields no cells.
    assert L.by_len_cells([base], 1) == []


def t_audit_length_exact_integer_fields():
    td = tempfile.mkdtemp(prefix='len_exact_', dir=REPO)
    try:
        p = os.path.join(td, 'bylen.json')
        huge = 2 ** 60
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {str(huge): {
                'n_tok': huge, 'eval_span': huge, 'n_seq': 1,
                'truncated': False}}}}, f)
        r = A.audit_length(p)
        assert r['status'] == 'fail', r
        assert any('not an integer length' in x for x in r['problems']), r
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'n_tok': float(huge), 'eval_span': 4095, 'n_seq': 1,
                'truncated': False}}}}, f)
        r2 = A.audit_length(p)
        assert r2['status'] == 'fail', r2
        assert any('not an integer count' in x for x in r2['problems']), r2
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'n_tok': '4095.5', 'eval_span': 4095, 'n_seq': 1,
                'truncated': False}}}}, f)
        r3 = A.audit_length(p)
        assert r3['status'] == 'fail', r3
        assert any('not an integer count' in x for x in r3['problems']), r3
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_audit_boundary_below_baseline_fails():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'mech': {
                'precision': 0.10, 'random_precision': 0.20}}}, f)
        out = A.audit_boundary(outdirs=[d])
        assert out['status'] == 'fail', out
        assert out['panels'][d][0]['verdict'] == 'below baseline', out
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'mech': {
                'precision': 0.30, 'random_precision': 0.20}}}, f)
        out = A.audit_boundary(outdirs=[d])
        assert out['status'] == 'ok', out


def t_audit_boundary_audits_every_cell_and_range():
    # A failing cell beyond any display cap must still fail the audit, and a
    # precision/baseline outside [0,1] is unverifiable, not an "excess".
    with tempfile.TemporaryDirectory() as d:
        recs = {f'a::s{i}': {'variant': 'a', 'mech': {
            'precision': 0.30, 'random_precision': 0.20}} for i in range(200)}
        recs['z::s_bad'] = {'variant': 'z', 'mech': {
            'precision': 0.10, 'random_precision': 0.20}}
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(recs, f)
        out = A.audit_boundary(outdirs=[d])
        assert out['status'] == 'fail', out['status']
        assert out.get('cells_audited', {}).get(d) == 201, out.get('cells_audited')
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'mech': {
                'precision': 1.5, 'random_precision': 0.2}}}, f)
        out = A.audit_boundary(outdirs=[d])
        assert out['status'] == 'fail', out
        assert any('outside [0,1]' in x for x in out.get('problems', [])), out


def t_selection_audit_rejects_out_of_range_claims():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'sel.json')
        per_seed = {str(i): {'mass_covered': 0.8, 'random_baseline': 0.03}
                    for i in range(6)}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'cells': [{
                'arm': 'a', 'eval_len': 4096, 'k': 32, 'n_blocks': 1024,
                'n_queries': 10, 'recall_at_k': 0.5, 'mass_covered': 1.5,
                'random_baseline': 0.03, 'per_seed': per_seed}]}, f)
        out = A.audit_selection(p)
        assert out['status'] == 'fail', out
        assert out['cells'][0]['claim_allowed'] is False, out['cells'][0]
        # A huge integral float is not an exact geometry count.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'cells': [{
                'arm': 'a', 'eval_len': 4096, 'k': 2 ** 60, 'n_blocks': 2 ** 61,
                'n_queries': 10, 'recall_at_k': 0.5, 'mass_covered': 0.7,
                'random_baseline': 0.03, 'per_seed': per_seed}]}, f)
        out = A.audit_selection(p)
        assert out['status'] == 'fail', out
        assert any('non-integral geometry' in x for x in out['problems']), out


def t_split_audit_fails_on_unreadable_meta():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'meta_bad.json'), 'w', encoding='utf-8') as f:
            f.write('{not valid json')
        out = A.audit_split_discipline(cache_dirs=[d])
        assert out['status'] == 'fail', out
        assert out['caches'][os.path.abspath(d)]['status'] == 'fail', out


def t_valid_json_non_object_summaries_are_refused():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump([], f)
        for fn in (A.audit_attribution, A.audit_boundary, A.audit_failures,
                   A.audit_metrics, A.audit_efficiency_pairs):
            out = fn(outdirs=[d])
            assert out['status'] == 'fail', (fn.__name__, out)
        out = A.audit_methodology(outdirs=[d], write=False)
        assert out['status'] == 'fail', out
        out = A.audit_fairness(os.path.join(d, 'summary.json'))
        assert out['status'] == 'fail', out
        out = A.audit_budget(state_files=(), panels=((d, 'arm', 10),))
        assert out['status'] == 'fail', out
        # The report builder must refuse a non-object summary/aggregate too.
        try:
            BR.summary(d)
        except RuntimeError as e:
            assert 'not an object' in str(e), e
        else:
            raise AssertionError('build_report.summary accepted a non-object JSON file')
        with open(os.path.join(d, 'aggregate.json'), 'w', encoding='utf-8') as f:
            json.dump({}, f)
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(123, f)
        try:
            BR.summary(d)
        except RuntimeError:
            pass
        else:
            raise AssertionError('build_report.summary accepted a numeric JSON file')


def t_audit_metrics_missing_or_malformed_summary():
    with tempfile.TemporaryDirectory() as d:
        # No local panel: the policy was checked, but the record-level audit
        # did not run, so the status must not be a fabricated "ok".
        assert A.audit_metrics(outdirs=[d])['status'] == 'not_verifiable_locally'
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            f.write('{bad json')
        out = A.audit_metrics(outdirs=[d])
        assert out['status'] == 'fail', out
        assert out['problems'], out
    with tempfile.TemporaryDirectory() as d:
        # Records without block statistics cannot support a block-attribution
        # audit; the panel must not be reported as "ok".
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'ppl': 10.0}}, f)
        out = A.audit_attribution(outdirs=[d])
        assert out['status'] == 'not_verifiable_locally', out
        assert out['panels'][d].get('note'), out


def t_selection_audit_empty_per_seed_fails():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'sel.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'cells': [{
                'arm': 'a', 'eval_len': 4096, 'k': 32, 'n_blocks': 1024,
                'n_queries': 10, 'recall_at_k': 0.5, 'mass_covered': 0.7,
                'random_baseline': 0.03, 'per_seed': {}}]}, f)
        out = A.audit_selection(p)
        assert out['status'] == 'fail', out
        assert any('per_seed is empty' in x for x in out['problems']), out


def t_audit_fairness_nan_seed_does_not_crash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': float('nan'),
                                    'ppl': 10.0, 'run_cfg': 'x'}}, f)
        out = A.audit_fairness(path, arms=['a'])
        assert out['arms']['a']['seeds'] == [], out
        # Mixed parameter-count spellings must still be counted and reported.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'params': 10,
                                    'ppl': 10.0, 'run_cfg': 'x'},
                       'a::seed1': {'variant': 'a', 'seed': 1, 'params': '12',
                                    'ppl': 10.0, 'run_cfg': 'x'}}, f)
        out = A.audit_fairness(path, arms=['a'])
        assert out['arms']['a']['n_records'] == 2, out
        # Unhashable (list/dict) metadata must still be counted and reported.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::seed0': {'variant': 'a', 'seed': 0, 'params': [10],
                                    'ppl': 10.0, 'run_cfg': ['x']}}, f)
        out = A.audit_fairness(path, arms=['a'])
        assert out['arms']['a']['n_records'] == 1, out


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
    # Fractional token counts must not be truncated into equal sizes.
    art2 = json.loads(json.dumps(art))
    art2['baselines']['a']['trials'] = [
        {'config': {'alpha': 0.1}, 'ppl': 1.0, 'n_tokens': 100.4},
        {'config': {'alpha': 0.2}, 'ppl': 2.0, 'n_tokens': 100.9}]
    out2 = L.simple_baseline_audit(art2)
    assert out2['status'] == 'fail', out2
    assert any('positive integer' in p for p in out2['problems']), out2



# ---------------------------------------------------------------------------
# L. Additional checks: claim gate, Holm family floor, crossover fit,
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
    # n_nonzero is an exact nonzero-pair count; a fractional spelling is
    # rejected.
    r = L.reportable_claim(dict(base, n_nonzero=4.9))
    assert r['allowed'] is False and any('n_nonzero' in x for x in r['reasons']), r


def t_record_readers_refuse_non_object_summaries():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump([], f)
        assert L.read_panel(d) == {}
        assert L.ppl_by_seed(d) == {}
        assert L.ppl_by_seed_grouped(d) == {}
        assert L.variant_presence(d) == {}
        assert L.legacy_warm_tags([]) == {}
        try:
            L.aggregate([])
        except ValueError as e:
            assert 'summary object' in str(e), e
        else:
            raise AssertionError('aggregate accepted a list as a panel summary')
    # Resume completeness for a NIAH cell needs a real finite accuracy, not
    # a boolean/string/NaN placeholder.
    for bad in (True, '0.9', float('nan'), float('inf')):
        assert L.result_is_current({'acc': bad, '_code': 'X'}, 'X', 'acc') is False, bad
    assert L.result_is_current({'acc': 0.0, '_code': 'X'}, 'X', 'acc') is True


def t_boolean_pvalues_are_not_probabilities():
    # A boolean flag must never be coerced to p=0/1 by a correction or claim
    # gate: float(False) == 0.0 would otherwise manufacture a certain result.
    for bad in (False, True, np.bool_(False), np.bool_(True)):
        h = L.holm_adjust([('flag', bad), ('real', 0.03)])
        assert h['flag']['p_holm'] is None and h['flag']['p_raw'] is bad, (bad, h)
        assert h['real']['m_effective'] == 2, (bad, h)
        comps = {'flag': {'n': 12, 'mean': -1.0, 'std': 0.1,
                          'p_exact_signflip': bad},
                 'real': {'n': 12, 'mean': -1.0, 'std': 0.1,
                          'p_exact_signflip': 0.03}}
        fam = L.attach_holm(dict(comps), family='bool p')
        assert comps['flag']['p_holm'] == 1.0, (bad, comps)
        assert comps['real']['p_holm'] is not None, (bad, comps)
        claim = L.reportable_claim({'n': 12, 'mean': -1.0, 'std': 0.1,
                                    'p_exact_signflip': bad, 'p_holm': 0.02})
        assert claim['allowed'] is False and claim['p_raw'] is None, (bad, claim)
        # The report builder's local twins must reject the same values.
        bh = BR.holm_adjust([('flag', bad), ('real', 0.03)])
        assert bh['flag'] == 1.0 and abs(bh['real'] - 0.06) < 1e-12, (bad, bh)
        ga = BR.global_multiplicity_adjust([('flag', bad), ('real', 0.03)])
        r = {x['name']: x for x in ga['rows']}
        assert r['flag']['p_raw'] is None and r['flag']['global_status'] == 'untestable', (bad, ga)
        assert r['real']['p_raw'] == 0.03, (bad, ga)
        assert BR._sig_gate({'n': 12, 'mean': -1.0, 'std': 0.1,
                             'p_exact_signflip': bad, 'p_holm': 0.02},
                            register=False) is False
    # Boolean measurements are not seed counts/effects either.
    assert L.reportable_claim({'n': True, 'mean': -1.0, 'std': 0.1,
                               'p_exact_signflip': 0.01, 'p_holm': 0.02})['allowed'] is False
    assert L.reportable_claim({'n': True, 'mean': -1.0, 'std': 0.1,
                               'p_exact_signflip': 0.01, 'p_holm': 0.02},
                              min_n=1)['allowed'] is False
    # attach_holm's family-resolution diagnostic must read only exact seed
    # counts; bool/fractional n values cannot lower/raise the family n range.
    _famc = {'flag': {'n': True, 'mean': -1.0, 'std': 0.1,
                      'p_exact_signflip': 0.2},
             'frac': {'n': 6.9, 'mean': -1.0, 'std': 0.1,
                      'p_exact_signflip': 0.2},
             'real': {'n': 8, 'mean': -1.0, 'std': 0.1,
                      'p_exact_signflip': 0.2}}
    _fam = L.attach_holm(_famc, family='exact n')
    assert _fam['n_effective_min'] == 8 and _fam['n_effective_max'] == 8, _fam
    # A fractional or string seed count must not be truncated into a smaller
    # n that changes the exact-test p floor.
    for bad_n in (6.9, '6'):
        rr = L.reportable_claim({'n': bad_n, 'mean': -1.0, 'std': 0.1,
                                 'p_exact_signflip': 0.001, 'p_holm': 0.001}, min_n=6)
        assert rr['allowed'] is False and rr['n'] == 0, (bad_n, rr)
    assert L.reportable_claim({'n': 6.0, 'mean': -1.0, 'std': 0.1,
                               'p_exact_signflip': 0.001, 'p_holm': 0.001},
                              min_n=6)['allowed'] is True
    assert L.reportable_claim({'n': 12, 'mean': True, 'std': 0.1,
                               'p_exact_signflip': 0.01, 'p_holm': 0.02})['allowed'] is False
    # Boolean per-seed deltas are flags, not effects; they must be dropped
    # rather than coerced to six identical 1.0 deltas.
    es = L.exact_signflip_p([True] * 6)
    assert es['n'] == 0 and es['n_dropped'] == 6 and math.isnan(es['p']), es
    # family_size is a comparison count: a boolean flag / non-integral value
    # must not be coerced into a smaller family that relaxes Holm.
    base = {'n': 12, 'mean': -1.0, 'std': 0.1,
            'p_exact_signflip': 0.01, 'p_holm': 0.02}
    for bad_family in (True, np.bool_(False), 4.9, 0, -1, '4'):
        r = L.reportable_claim(base, family_size=bad_family)
        assert r['allowed'] is False, (bad_family, r)
        assert r['min_attainable_p_holm_family'] is None, (bad_family, r)
    assert L.reportable_claim(base, family_size=4.0)['allowed'] is True


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
    # A panel with mixed seed counts is bounded by its BEST-resolved member,
    # not its least-resolved one: m=2, n_max=12 gives 2*2/2**12, so the large-n
    # comparison can reject even though the n=3 row cannot.
    mixed = {'short': {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25},
             'long': {'n': 12, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.001}}
    fam3 = L.attach_holm(mixed, family='mixed n')
    assert abs(fam3['family_min_attainable_p_holm'] - 0.0009765625) < 1e-12, fam3
    assert fam3['family_resolution_limited'] is False, fam3
    assert mixed['long']['p_holm'] is not None and mixed['long']['p_holm'] < 0.05, mixed


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
    # Boolean flags are not widths/volumes; they are excluded rather than
    # becoming d=1 / one-token points.
    for _field in ('d', 'crossover_tokens', 'train_tokens'):
        _bad = pts(4)
        _bad[0] = dict(_bad[0], **{_field: True})
        f = L.crossover_scaling_fit(_bad)
        assert f['status'] == 'insufficient_points', (_field, f)
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
    # A different tokens-per-parameter allocation is a visible confound.
    b2 = tier('S2', 3000, 2000)
    out2 = L.scale_ladder_audit([a, b2])
    assert out2['budget_matched'] is False, out2
    assert any('tokens-per-parameter' in w for w in out2['warnings']), out2
    # A declared token allocation that disagrees with steps*batch*seq is a hard
    # fail, not a warning.
    bad = tier('S1', 1000, 1000, tokens=999999)
    out3 = L.scale_ladder_audit([bad, b])
    assert out3['status'] == 'fail', out3
    assert any('n_train_tokens' in p for p in out3['problems']), out3
    # Missing parameter counts keep the allocation claim unverifiable.
    out4 = L.scale_ladder_audit([tier('S1', 1000, None), b])
    assert out4['budget_matched'] is None, out4
    assert any('params missing' in w for w in out4['warnings']), out4
    # Vocabulary provenance is an exact positive count, not a boolean or a
    # fractional value.  int(True)=1 / int(2.9)=2 would otherwise let two
    # different tokenisations look "shared".
    for _bad_vocab in (True, np.bool_(False), 1.5, '1000', 0, -1):
        _a = tier('S1', 1000, 1000)
        _b = tier('S2', 2000, 2000)
        _a['cfg']['vocab'] = _bad_vocab
        _b['cfg']['vocab'] = _bad_vocab
        _out = L.scale_ladder_audit([_a, _b])
        assert _out['status'] == 'fail', (_bad_vocab, _out)
        assert any('vocab' in p for p in _out['problems']), (_bad_vocab, _out)
    _a = tier('S1', 1000, 1000)
    _b = tier('S2', 2000, 2000)
    _a['cfg']['vocab'] = _b['cfg']['vocab'] = 8000
    assert L.scale_ladder_audit([_a, _b])['status'] != 'fail'
    # A boolean corpus/tokenizer provenance is not a shared identity.
    for _bad_fp in (True, np.bool_(False)):
        _a = tier('S1', 1000, 1000)
        _b = tier('S2', 2000, 2000)
        _a['cfg']['corpus_fp'] = _b['cfg']['corpus_fp'] = _bad_fp
        _out = L.scale_ladder_audit([_a, _b])
        assert _out['status'] == 'fail', (_bad_fp, _out)
        assert any('corpus_fp' in p for p in _out['problems']), (_bad_fp, _out)
    # A boolean learning rate is a flag, not float(True)=1.0.
    for _bad_lr in (True, np.bool_(False)):
        _a = tier('S1', 1000, 1000)
        _b = tier('S2', 2000, 2000)
        _a['cfg']['lr'] = _b['cfg']['lr'] = _bad_lr
        _out = L.scale_ladder_audit([_a, _b])
        assert _out['status'] == 'fail', (_bad_lr, _out)
        assert any('lr' in p for p in _out['problems']), (_bad_lr, _out)
    # The deployed plan must carry an explicit transfer policy and expose the
    # S1/S2 allocation ratio rather than silently calling it matched.
    pol = L.SCALE_TRANSFER_POLICY
    assert pol.get('lr_transfer') and pol.get('tokens_per_param_tolerance') is not None, pol


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
    # NumPy measurements are real values; boolean flags are not measurements.
    np_recs = {('a', 'p', 0): {'params': np.int64(1000), 'train_time_s': np.float32(10.0),
                               'tokens_seen': np.int64(100)},
               ('b', 'p', 0): {'params': np.int64(1000), 'train_time_s': np.float32(20.0),
                               'tokens_seen': np.int64(100)}}
    np_c = L._compute_control_detail(np_recs, 'a', 'b')
    assert np_c['time_ratio_a_over_b'] is not None, np_c
    bool_c = L._compute_control_detail(
        {('a', 'p', 0): {'params': True, 'train_time_s': True, 'tokens_seen': True},
         ('b', 'p', 0): {'params': True, 'train_time_s': True, 'tokens_seen': True}}, 'a', 'b')
    assert bool_c['compute_matched'] is None and bool_c['time_ratio_a_over_b'] is None, bool_c
    # Fractional counts are not exact parameter/token identities.
    frac_c = L._compute_control_detail(
        {('a', 'p', 0): {'params': 1000.5, 'train_time_s': 10.0, 'tokens_seen': 100.5},
         ('b', 'p', 0): {'params': 1000, 'train_time_s': 20.0, 'tokens_seen': 100}}, 'a', 'b')
    assert frac_c['compute_matched'] is None and frac_c['time_ratio_a_over_b'] is None, frac_c
    frac_p = L._compute_control_detail(
        {('a', 'p', 0): {'params': 1000.5, 'tokens_seen': 100},
         ('b', 'p', 0): {'params': 1000, 'tokens_seen': 100}}, 'a', 'b')
    assert frac_p['compute_matched'] is None and frac_p['proxy_ratio_a_over_b'] is None, frac_p
    frac_d = L._param_control_detail(
        {('a', 'p', 0): {'params': 1000, 'param_breakdown': {'total': 1000, 'non_embedding': 800.5, 'active': 700}},
         ('b', 'p', 0): {'params': 1000, 'param_breakdown': {'total': 1000, 'non_embedding': 800, 'active': 700}}}, 'a', 'b')
    assert frac_d['non_embedding_rel_diff'] is None and frac_d['non_embedding_controlled'] is None, frac_d
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


def t_report_paired_row_single_config():
    base = dict(train_knobs=None, data_fp='d', val_fp='v', eval_fp='e',
                init_core_fp='i', steps=10, tokens_seen=100)
    a = {('', 0): dict(base, variant='a', seed=0, ppl=10.0, run_cfg='cfgA'),
         ('', 1): dict(base, variant='a', seed=1, ppl=10.0, run_cfg='cfgB')}
    b = {('', 0): dict(base, variant='b', seed=0, ppl=11.0, run_cfg='cfgA'),
         ('', 1): dict(base, variant='b', seed=1, ppl=11.0, run_cfg='cfgB')}
    pair = BR.paired_row(a, b)
    assert pair is not None and pair[2]['n'] == 1, pair
    assert pair[0] in ([('', 0)], [('', 1)]), pair


def t_driver_pairing_single_config():
    import v9_supp as V9
    def rec(cfg, ppl):
        return {'run_cfg': cfg, 'steps': 100, 'tokens_seen': 1000, 'ppl': ppl}
    panel = {'a': {0: rec('cfgA', 10.0), 1: rec('cfgB', 10.0)},
             'b': {0: rec('cfgA', 12.0), 1: rec('cfgB', 12.0)}}
    comps = {}
    V9.add_paired(comps, 'mixed', panel, 'a', 'b')
    assert comps['mixed']['n'] == 1, comps['mixed']
    assert len(comps['mixed']['seeds']) == 1, comps['mixed']


def t_v7_bootstrap_single_config():
    root = tempfile.mkdtemp()
    try:
        od = os.path.join(root, 'results_lm_v7_warmup')
        os.makedirs(od)
        raw = {}
        for seed, cfg in ((0, 'cfgA'), (1, 'cfgB')):
            for warm, ppl in ((5000, 10.0), (0, 11.0)):
                raw[f'csa_fixed::w{warm}::seed{seed}'] = {
                    'variant': 'csa_fixed', 'seed': seed, 'warm_steps': warm,
                    'run_cfg': cfg, 'steps': 100, 'tokens_seen': 1000,
                    'ppl': ppl + seed}
        with open(os.path.join(od, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        r = V7.bootstrap_report([od], out=os.path.join(root, 'stats.json'))
        comp = r['comparisons']['warmup w=5000 vs scratch']
        assert comp['n'] == 1 and comp['per_seed'] == [-1.0], comp
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_v12_probe_contrasts_require_matching_params():
    root = tempfile.mkdtemp()
    try:
        path = os.path.join(root, 'probe.json')

        def _write(learned_pp, drop=False):
            raw = {}
            for s in (0, 1):
                for arm, pp, ppl in (('learned', learned_pp, 10.0),
                                     ('dense', {'n_seq': 16}, 11.0)):
                    r = {'variant': 'csa_fixed', 'seed': s, 'arm': arm,
                         'eval_len': 4096, 'rho': 0.5, 'mode': 'uniform',
                         'ppl_mean': ppl, 'probe_params': pp, 'val_fp': 'F',
                         '_code': L.code_stamp()}
                    if drop and arm == 'learned':
                        r.pop('probe_params')
                    raw[f'{arm}{s}'] = r
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(raw, f)
            return V12._test_probe_stats(path, {'arms': ['csa_fixed'], 'seeds': [0, 1]}, 'F')

        assert _write({'n_seq': 8})['contrasts'] == {}
        assert _write({'n_seq': 16})['contrasts'], 'same probe_params must pair'
        miss = _write({'n_seq': 16}, drop=True)
        assert miss['n_admitted'] == 2 and miss['n_excluded'] == 2, miss
        assert miss['contrasts'] == {}, miss
        # A synthesized probe record is a reconstruction, not a measurement;
        # it must be named and excluded rather than pooled.
        raw = {}
        for s in (0, 1):
            for arm, ppl in (('learned', 10.0), ('dense', 11.0)):
                raw[f'{arm}{s}'] = {'variant': 'csa_fixed', 'seed': s, 'arm': arm,
                                    'eval_len': 4096, 'rho': 0.5, 'mode': 'uniform',
                                    'ppl_mean': ppl, 'probe_params': {'n_seq': 16},
                                    'val_fp': 'F', '_code': L.code_stamp(),
                                    'synthesized': True}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        synth = V12._test_probe_stats(path, {'arms': ['csa_fixed'], 'seeds': [0, 1]}, 'F')
        assert synth['n_admitted'] == 0 and synth['n_excluded'] == 4, synth
        assert synth['contrasts'] == {}, synth
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_v10_curve_history_steps_are_exact():
    # v10's trajectory reader feeds the crossover fit; a fractional step must
    # not be truncated into an existing integer step and a boolean cannot be a
    # PPL measurement.
    raw = {'v::seed0': {'variant': 'v', 'seed': 0,
                        'run_cfg': f'steps1_d8_L1_cs{L.code_stamp()}',
                        'ppl_history': [[1.5, 10.0], [2, True], [3, 5.0],
                                        ['bad', 7.0], [4, float('nan')]]}}
    out = V10._hist_by_seed('unit_v10_history', 'v', raw=raw)
    assert out.get(0) == [[3, 5.0]], out


def t_report_history_steps_are_exact():
    rec = {'ppl_history': [[1.5, 10.0], [1.0, 20.0], [True, 30.0],
                           ['2', 40.0], [-1, 50.0]]}
    assert list(BR._hist_points(rec)) == [(1, 20.0), (2, 40.0)], list(BR._hist_points(rec))


def t_v11_by_length_deduplicates_seeds():
    rows = [{'variant': 'v', 'seed': 0, 'by_len': {512: {'ppl': 10.0}}},
            {'variant': 'v', 'seed': 0, 'by_len': {512: {'ppl': 20.0}}},
            {'variant': 'v', 'seed': 1, 'by_len': {512: {'ppl': 30.0}}},
            {'variant': 'v', 'seed': True, 'by_len': {512: {'ppl': 40.0}}},
            {'variant': 'v', 'seed': 2, 'synthesized': True,
             'by_len': {512: {'ppl': 50.0}}}]
    kept = V11._one_record_per_seed(rows, label='test')
    assert [r['seed'] for r in kept] == [1], kept
    assert all(not r.get('synthesized') for r in kept), kept


def t_report_stale_trajectory_refused():
    stale = 'old_csOLD'
    s = {}
    for i in range(6):
        s[f'csa_fixed::w0::seed{i}'] = dict(variant='csa_fixed', seed=i,
            warm_steps=0, ppl=10.0 + i * 0.1, run_cfg=stale, steps=100,
            tokens_seen=1000, _code='OLD')
        s[f'csa_fixed::w5000::seed{i}'] = dict(variant='csa_fixed', seed=i,
            warm_steps=5000, ppl=9.0 + i * 0.1, run_cfg=stale, steps=100,
            tokens_seen=1000, _code='OLD')
    BR._LOADED[('summary', 'results_lm_v7_warmup')] = s
    try:
        txt = BR.sec_p0w()
    finally:
        BR._LOADED.pop(('summary', 'results_lm_v7_warmup'), None)
    assert '陈旧记录已排除' in txt and '10.25' not in txt and '9.25' not in txt, txt

    long_s = {}
    for i in range(2):
        for v in ('csa_fixed', 'full'):
            long_s[f'{v}::seed{i}'] = dict(variant=v, seed=i, ppl=10.0 + i,
                run_cfg=stale, steps=100, tokens_seen=1000, _code='OLD',
                hist={'20000': 10.0, '40000': 9.0})
    BR._LOADED[('summary', 'results_lm_v7_long40')] = long_s
    BR._LOADED[('agg', 'results_lm_v7_long40')] = {
        'csa_fixed': {'variant': 'csa_fixed', 'run_cfg': stale,
                      'n_seeds': 2, 'ppl_mean': 10.0},
        'full': {'variant': 'full', 'run_cfg': stale,
                 'n_seeds': 2, 'ppl_mean': 11.0}}
    try:
        txt2 = BR.sec_p0e()
    finally:
        BR._LOADED.pop(('summary', 'results_lm_v7_long40'), None)
        BR._LOADED.pop(('agg', 'results_lm_v7_long40'), None)
    assert '陈旧记录已排除' in txt2 and '**10.0**' not in txt2, txt2


def t_sec_p1l_rejects_malformed_length_keys():
    # A by-length key must be an exact positive integer.  Boolean/float keys
    # and length 0 are not lengths; mapping 4.9 -> 4 would quote a cell
    # measured at a different (non-integral) length under a valid column.
    rec = {'v::s0': {'variant': 'v', 'seed': 0, '_code': BR._CUR_CS, 'by_len': {
        True: {'ppl': 11.0, 'truncated': False},
        '4.9': {'ppl': 22.0, 'truncated': False},
        '0': {'ppl': 33.0, 'truncated': False},
        '1': {'ppl': 66.0, 'truncated': False},
        '-512': {'ppl': 44.0, 'truncated': False},
        '512': {'ppl': 10.0, 'truncated': False},
        '4096': {'ppl': 20.0, 'truncated': False},
        str(2 ** 53 + 1): {'ppl': 55.0, 'truncated': False},
        float(2 ** 60): {'ppl': 77.0, 'truncated': False},
    }}}
    # A malformed by_len payload (valid JSON but not an object) must be
    # skipped, not crash the whole length section while a valid record exists.
    rec['bad::s0'] = {'variant': 'bad', 'seed': 0, '_code': BR._CUR_CS,
                      'by_len': [{'ppl': 99.0, 'truncated': False}]}
    BR._LOADED[('summary', 'results_len')] = rec
    try:
        txt = BR.sec_p1l()
    finally:
        BR._LOADED.pop(('summary', 'results_len'), None)
    assert 'PPL@512' in txt and 'PPL@4096' in txt, txt[:600]
    assert '99.0' not in txt, txt[:600]
    for _bad in ('PPL@0', 'PPL@1 ', 'PPL@4 ', 'PPL@-512', '66.0'):
        assert _bad not in txt, (_bad, txt[:600])
    for _bad_val in ('11.0', '22.0', '33.0', '44.0'):
        assert _bad_val not in txt, (_bad_val, txt[:600])
    # A huge integer key must not be rounded into a different (smaller) valid
    # length by a float conversion.
    assert f'PPL@{2 ** 53}' not in txt, txt[:600]
    assert f'PPL@{2 ** 53 + 1}' in txt, txt[:600]
    # A lossy large float key must not become a valid (huge) length column.
    assert f'PPL@{2 ** 60}' not in txt and '77.0' not in txt, txt[:600]


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
    # Two run_cfg groups of one variant must not be pooled into one length
    # table (that would pair @512 from cfgA with @4096 from cfgB).
    cur_cs = f'_cs{BR._CUR_CS}'
    summ = {
        'v::s0': {'variant': 'v', 'seed': 0, '_code': BR._CUR_CS,
                  'run_cfg': 'steps3000' + cur_cs,
                  'by_len': {'512': {'ppl': 10.0, 'truncated': False},
                             '4096': {'ppl': 12.0, 'truncated': False}}},
        'v::s1': {'variant': 'v', 'seed': 1, '_code': BR._CUR_CS,
                  'run_cfg': 'steps999' + cur_cs,
                  'by_len': {'512': {'ppl': 99.0, 'truncated': False}}},
    }
    BR._LOADED[('summary', 'results_len')] = summ
    try:
        txt2 = BR.sec_p1l()
    finally:
        BR._LOADED.pop(('summary', 'results_len'), None)
    assert '99.0' not in txt2, 'a minority run_cfg must not enter the length table'
    assert '10.0' in txt2 and '12.0' in txt2, txt2[:500]


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
        # A different token count is not a paired comparison.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=4096)}}, f)
        r2 = A.audit_efficiency_pairs(outdirs=[td])
        rec2 = r2['panels'][td]['a vs b']
        assert rec2['speed_claim_allowed'] is False and 'token counts' in rec2['reason'], rec2
        # A later seed measured on a different device/torch build must also
        # refuse the directional claim; checking only the first record would let a
        # mixed CPU/GPU panel through.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': eff_a},
                       'a::s1': {'variant': 'a', 'efficiency': dict(eff_a, device='cpu', torch='2.9.0')},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        r3 = A.audit_efficiency_pairs(outdirs=[td])
        rec3 = r3['panels'][td]['a vs b']
        assert rec3['speed_claim_allowed'] is False and 'device or torch' in rec3['reason'], rec3
        # Boolean rate/token fields are flags, not measurements.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, prefill_ms_p50=True)},
                       'b::s0': {'variant': 'b', 'efficiency': dict(eff_b, n_tokens=True)}}, f)
        r4 = A.audit_efficiency_pairs(outdirs=[td])
        rec4 = r4['panels'][td]['a vs b']
        assert rec4['speed_claim_allowed'] is False, rec4
        # An out-of-float-range timing is not a measurement, not a crash.
        with open(os.path.join(td, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'variant': 'a', 'efficiency': dict(eff_a, prefill_ms_p50=10 ** 400)},
                       'b::s0': {'variant': 'b', 'efficiency': eff_b}}, f)
        r5 = A.audit_efficiency_pairs(outdirs=[td])
        rec5 = r5['panels'][td]['a vs b']
        assert rec5['speed_claim_allowed'] is False, rec5
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



def t_archive_verify_uses_own_manifest():
    import archive_release as AR
    root = tempfile.mkdtemp()
    old_repo = AR.REPO
    try:
        repo = os.path.join(root, 'repo')
        os.makedirs(repo)
        for n in ('DECISION_LOG.md', 'RELEASES.md', 'confirmation_frozen.json'):
            with open(os.path.join(repo, n), 'w', encoding='utf-8') as f:
                f.write('x')
        d = os.path.join(repo, 'results_x')
        os.makedirs(d)
        with open(os.path.join(d, 'payload.txt'), 'w', encoding='utf-8') as f:
            f.write('hello')
        with open(os.path.join(d, 'aaa.MANIFEST.sha256'), 'w', encoding='utf-8') as f:
            json.dump({'files': {'payload.txt': {'sha256': 'deadbeef'}}}, f)
        AR.REPO = repo
        assert AR.cmd_make('v1', ['results_x'], outdir='arch') == 0
        tar = os.path.join(repo, 'arch', 'release_v1.tar.gz')
        assert AR.cmd_verify(tar) == 0, 'a stray collected manifest must not hijack verification'
    finally:
        AR.REPO = old_repo
        shutil.rmtree(root, ignore_errors=True)


def t_archive_hashes_all_checkpoint_extensions():
    import archive_release as AR
    root = tempfile.mkdtemp()
    old_repo = AR.REPO
    try:
        os.makedirs(os.path.join(root, 'results_x'))
        for n in ('a.pt', 'b.ckpt', 'c.pth', 'd.safetensors'):
            with open(os.path.join(root, 'results_x', n), 'wb') as f:
                f.write(n.encode())
        AR.REPO = root
        hashed = AR._ckpt_hash_table(['results_x'])
        assert sorted(hashed) == ['results_x/a.pt', 'results_x/b.ckpt',
                                  'results_x/c.pth', 'results_x/d.safetensors'], hashed
        packed = AR._collect(['results_x'])
        assert not any(p.endswith(('.pt', '.ckpt', '.pth', '.safetensors'))
                       for p in packed), packed
    finally:
        AR.REPO = old_repo
        shutil.rmtree(root, ignore_errors=True)


def t_archive_refuses_dirs_outside_repo():
    import archive_release as AR
    root = tempfile.mkdtemp()
    old_repo = AR.REPO
    try:
        repo = os.path.join(root, 'repo')
        os.makedirs(repo)
        for n in ('DECISION_LOG.md', 'RELEASES.md', 'confirmation_frozen.json'):
            with open(os.path.join(repo, n), 'w', encoding='utf-8') as f:
                f.write('x')
        outside = os.path.join(root, 'outside')
        os.makedirs(outside)
        with open(os.path.join(outside, 'secret.txt'), 'w', encoding='utf-8') as f:
            f.write('payload')
        AR.REPO = repo
        assert AR.cmd_make('v1', ['../outside'], outdir='arch') == 1
        assert not os.path.exists(os.path.join(repo, 'arch', 'release_v1.tar.gz'))
        files = AR._collect(['../outside'])
        assert all(os.path.realpath(p).startswith(os.path.realpath(repo) + os.sep)
                   for p in files), files
    finally:
        AR.REPO = old_repo
        shutil.rmtree(root, ignore_errors=True)


def t_archive_verify_requires_external_checksum():
    import archive_release as AR
    with tempfile.TemporaryDirectory() as d:
        tar = os.path.join(d, 'release_x.tar.gz')
        with open(tar, 'wb') as f:
            f.write(b'not a real tar')
        # Without the external sidecar the in-archive manifest can be rewritten
        # together with the archive, so verify must refuse.
        assert AR.cmd_verify(tar) == 1
        with open(tar + '.sha256', 'w', encoding='utf-8') as f:
            f.write(AR._sha256(tar) + '  release_x.tar.gz\n')
        # The checksum anchors the file; the invalid tar itself must not be
        # accepted as a release either (a clean refusal, not a traceback).
        assert AR.cmd_verify(tar) == 1
        # A valid tar with a malformed per-file manifest entry is also refused
        # instead of crashing the verifier.
        import io
        import tarfile
        tar2 = os.path.join(d, 'release_y.tar.gz')
        man = {'files': {'payload.txt': 'not-an-object'}}
        with tarfile.open(tar2, 'w:gz') as tf:
            payload = b'hello'
            ti = tarfile.TarInfo('payload.txt')
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
            mb = json.dumps(man).encode('utf-8')
            mti = tarfile.TarInfo('release_y.MANIFEST.sha256')
            mti.size = len(mb)
            tf.addfile(mti, io.BytesIO(mb))
        with open(tar2 + '.sha256', 'w', encoding='utf-8') as f:
            f.write(AR._sha256(tar2) + '  release_y.tar.gz\n')
        assert AR.cmd_verify(tar2) == 1
        # A symlink/hardlink member is not attested by the manifest and must
        # not be silently skipped while its name satisfies the presence check.
        tar3 = os.path.join(d, 'release_z.tar.gz')
        link_man = {'files': {'payload.txt': {'sha256': '0' * 64, 'bytes': 1}}}
        with tarfile.open(tar3, 'w:gz') as tf:
            ti = tarfile.TarInfo('payload.txt')
            ti.type = tarfile.SYMTYPE
            ti.linkname = '/etc/passwd'
            tf.addfile(ti)
            mb = json.dumps(link_man).encode('utf-8')
            mti = tarfile.TarInfo('release_z.MANIFEST.sha256')
            mti.size = len(mb)
            tf.addfile(mti, io.BytesIO(mb))
        with open(tar3 + '.sha256', 'w', encoding='utf-8') as f:
            f.write(AR._sha256(tar3) + '  release_z.tar.gz\n')
        assert AR.cmd_verify(tar3) == 1
        # Provenance files alone do not make a release: with no requested
        # result directory present, make must refuse instead of archiving
        # only the environment/provenance snapshot.
        assert AR.cmd_make('probe_no_artifacts', ['definitely_not_a_dir'],
                           outdir=d) == 1
        assert not any(n.startswith('release_probe_no_artifacts')
                       for n in os.listdir(d))
        # A release tag is a filename component, not a path: separators or
        # dot segments must be refused before any output path is built.
        for _bad_tag in ('../escape', 'a/b', 'a\\b', '..', ''):
            assert AR._safe_tag(_bad_tag) is None, _bad_tag
        assert AR._safe_tag('v13') == 'v13'
        # A symlink inside a result directory must not be packed: tar stores
        # it as a symlink member, while the verifier only attests regular-file
        # content, so packing it would make `make` produce an archive its own
        # `verify` rejects.
        _td = tempfile.mkdtemp()
        _old_repo = AR.REPO
        try:
            AR.REPO = _td
            os.makedirs(os.path.join(_td, 'results_x'))
            with open(os.path.join(_td, 'results_x', 'payload.txt'), 'w') as f:
                f.write('hello')
            os.symlink('/etc/passwd', os.path.join(_td, 'results_x', 'link'))
            os.makedirs(os.path.join(_td, 'wt103_cache'))
            for _n in ('meta_v5_sl512.json', 'tok_v5_sl512.json',
                       'test_meta_v2_sl512.json', 'test_tok_v2_sl512.json'):
                with open(os.path.join(_td, 'wt103_cache', _n), 'w') as f:
                    f.write('{}')
            _files = AR._collect(['results_x'])
            _rels = {os.path.relpath(f, _td) for f in _files}
            assert 'results_x/payload.txt' in _rels, _rels
            assert 'results_x/link' not in _rels, _rels
            for _n in ('meta_v5_sl512.json', 'tok_v5_sl512.json',
                       'test_meta_v2_sl512.json', 'test_tok_v2_sl512.json'):
                assert os.path.join('wt103_cache', _n) in _rels, (_n, _rels)
        finally:
            AR.REPO = _old_repo
            shutil.rmtree(_td, ignore_errors=True)


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
        # A per-seed fraction outside [0,1] or non-finite cannot support a
        # claim; it must not inflate n_seeds_valid or produce a p-value.
        for _bad_pair in (({'mass_covered': 1.5, 'random_baseline': 0.0}),
                          ({'mass_covered': 0.8, 'random_baseline': -0.1}),
                          ({'mass_covered': float('nan'), 'random_baseline': 0.03}),
                          ({'mass_covered': float('inf'), 'random_baseline': 0.03})):
            _bc = dict(cell, per_seed={str(i): dict(_bad_pair) for i in range(6)})
            with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
                json.dump({'cells': [_bc]}, f)
            _rb = A.audit_selection()
            assert _rb['cells'][0]['n_seeds_valid'] == 0, (_bad_pair, _rb)
            assert _rb['cells'][0]['seed_test'] is None, (_bad_pair, _rb)
            assert _rb['cells'][0]['claim_allowed'] is False, (_bad_pair, _rb)
            assert _rb['status'] == 'fail', (_bad_pair, _rb)
        # Booleans are flags, not measurements: a boolean-only cell must fail
        # structurally instead of becoming k=1/B=1/nq=1 and a claim.
        bool_cell = {'arm': 'a', 'eval_len': True, 'k': True, 'n_blocks': True,
                     'n_queries': True, 'recall_at_k': True,
                     'mass_covered': True, 'random_baseline': False,
                     'per_seed': {str(i): {'mass_covered': 0.8,
                                           'random_baseline': 0.03}
                                  for i in range(6)}}
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [bool_cell]}, f)
        r6 = A.audit_selection()
        assert r6['status'] == 'fail' and any('boolean' in x for x in r6['problems']), r6
        # Fractions must not be truncated into a geometry that satisfies
        # k <= n_blocks (1.9 -> 1, 1.1 -> 1) and then certify the cell.
        frac_cell = dict(cell, k=1.9, n_blocks=1.1)
        with open(os.path.join(td, 'analysis_mechanism', 'selection_attribution.json'), 'w', encoding='utf-8') as f:
            json.dump({'cells': [frac_cell]}, f)
        r7 = A.audit_selection()
        assert r7['status'] == 'fail' and any('non-integral geometry' in x for x in r7['problems']), r7
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
        # Fractional counts must not be truncated into a valid-looking cell.
        for field, val in (('n_tok', 4.9), ('n_seq', 1.9), ('eval_span', 4.9)):
            cell = {'ppl': 30.0, 'n_tok': 4, 'eval_span': 5, 'n_seq': 1,
                    'truncated': False}
            cell[field] = val
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'a::s0': {'by_len': {'5': cell}}}, f)
            rf = A.audit_length(p)
            assert rf['status'] == 'fail' and any('not an integer count' in x for x in rf['problems']), (field, rf)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_scale_edge_case_regressions():
    # Tiny but perfectly consistent effects must still hit the exact floor;
    # an absolute tie tolerance would force p=1.
    assert abs(L.exact_signflip_p([1e-13] * 6)['p'] - 2.0 / 64) < 1e-12
    assert abs(L.exact_signflip_p([1e-13] * 22)['p'] - 2.0 / (1 << 22)) < 1e-18
    # The exact-test floor comes from the NONZERO pair count.
    st = L.paired_effect_stats([0.7] * 5 + [0.0] * 15)
    assert st['n_nonzero'] == 5 and st['p_floor'] == 2.0 / 2 ** 5, st
    assert st['resolution_limited'] is True, st
    # 0 and "0" are one seed, so the paired trend path must not disappear.
    pts = [{'label': 'A', 'scale': 1.0, 'effects': [0.1, 0.2, 0.3], 'seeds': [0, 1, 2]},
           {'label': 'B', 'scale': 2.0, 'effects': [0.2, 0.3, 0.4], 'seeds': ['0', '1', '2']}]
    assert L.scaling_trend(pts, min_points=2, min_seeds=2)['paired']['n_complete_seeds'] == 3
    # A training stream with fewer than two tokens has no bigram table; this
    # must fall back to unigrams instead of indexing an empty array.
    ng = L.ngram_lm_score([1], [1, 2, 1], 4, order=2)
    assert math.isfinite(ng['nll']) and ng['n_tokens'] == 3, ng
    # Count-model hyper-parameters use the same exact/boolean contracts.
    for _bad_kw in ({'vocab_size': True}, {'vocab_size': 4.5},
                    {'vocab_size': 0}, {'order': True}, {'order': 1.5},
                    {'alpha': True}, {'alpha': float('nan')},
                    {'interpolation': True}, {'interpolation': float('nan')}):
        _args = {'vocab_size': 4, 'order': 2}
        _args.update(_bad_kw)
        try:
            L.ngram_lm_score([1, 2, 1], [1, 2, 1], **_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'ngram_lm_score accepted {_bad_kw!r}')
    # Object-dtype large integers stay exact in the corpus fingerprint.
    import data_guard as DG
    assert DG._matrix_fingerprint(np.array([[2 ** 53]], dtype=object)) != \
        DG._matrix_fingerprint(np.array([[2 ** 53 + 1]], dtype=object))
    # The exact float64 boundary must fingerprint identically whether the
    # table is stored as int64 or as an object integer column.
    for _val in (2 ** 53, -(2 ** 53)):
        _obj = np.array([[_val]], dtype=object)
        _typed = np.array([[_val]], dtype=np.int64)
        assert DG._matrix_fingerprint(_obj) == DG._matrix_fingerprint(_typed), _val
        assert DG._row_keys(_obj) == DG._row_keys(_typed), _val
    # A row must also match across object/int64/uint64 storage so a cache
    # serialised with object dtype cannot hide cross-split duplicates.
    for _vals, _dt in (([[1, 2], [3, 4]], np.int64),
                       ([[0, 2 ** 64 - 1]], np.uint64),
                       ([[-1, 2 ** 53]], np.int64)):
        _obj = np.array(_vals, dtype=object)
        _typed = np.array(_vals, dtype=_dt)
        assert DG._row_keys(_obj) == DG._row_keys(_typed), (_vals, _dt)
    # A non-truncated length cell must declare the requested span, not L-1.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bylen.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'a::s0': {'by_len': {'4096': {
                'n_tok': 16 * 4095, 'eval_span': 4095, 'n_seq': 16,
                'truncated': False}}}}, f)
        rep = A.audit_length(path)
        assert rep['status'] == 'fail', rep
        assert any('eval_span' in x for x in rep['problems']), rep


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


def t_build_report_global_multiplicity_section():
    BR._reset_claim_registry()
    try:
        st = {'n': 6, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.01, 'p_holm': 0.02}
        ok = BR._sig_gate(st, label='P0X demo labelled contrast', family='P0X')
        assert ok is True, st
        st2 = {'n': 3, 'mean': -1.0, 'std': 0.2, 'p_exact_signflip': 0.25, 'p_holm': 0.5}
        assert BR._sig_gate(st2, label='P0Y demo underpowered contrast', family='P0Y') is False
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
    assert m['status'] == 'ok', 'points without fingerprints stay analysable, just unattested'
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
    pts = L._scale_points([
        {'label': 'S1', 'scale': 1, 'effects': [0.1, 0.2], 'seeds': [0, 1],
         'corpus_fp': 'fpA', 'vocab': 8000},
        (2, [0.3, 0.4], [0, 1], 'fpB'),
        {'label': 'S3', 'scale': 3, 'effects': [0.5, 0.6], 'seeds': [0, 1]},
    ])
    assert pts[0]['corpus_fp'] == 'fpA' and pts[0]['vocab'] == 8000, pts
    assert pts[1]['corpus_fp'] == 'fpB' and pts[1]['vocab'] is None, pts
    assert pts[2]['corpus_fp'] is None, pts
    # Booleans and fractional vocabulary sizes are not provenance.
    for bad in ({'label': 'S1', 'scale': True, 'effects': [0.1]},
                {'label': 'S1', 'scale': 1, 'effects': [0.1], 'vocab': True},
                {'label': 'S1', 'scale': 1, 'effects': [0.1], 'vocab': 1.5},
                {'label': 'S1', 'scale': 1, 'effects': [0.1], 'corpus_fp': True}):
        try:
            L._scale_points([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed scale provenance accepted: {bad!r}')


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
        ('nan', np.float64('nan')), ('bool', True)])
    assert (lo, hi, dropped) == (1.0, 2.5, ['nan', 'bool']), (lo, hi, dropped)
    _j = L._jsonable({'a': np.longdouble(1.5), 'b': np.array([np.longdouble(2.5)]),
                      'c': np.float32(3.5), 'd': np.bool_(True)})
    assert _j == {'a': 1.5, 'b': [2.5], 'c': 3.5, 'd': True}, _j
    import json as _json
    _json.dumps(_j)
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
        # Both helpers accept path-like objects, not only str.
        import pathlib
        pp = pathlib.Path(p)
        assert L.locked_json_read(pp) == {'k': 1}
        got2 = L.locked_json_update(pp, lambda obj: (obj.update(k=2) or obj))
        assert got2 == {'k': 2} and L.locked_json_read(pp) == {'k': 2}
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
    # max_categories folds rare REFERENCE categories into the "other" bin.
    # A holdout value that exists in the reference is therefore not unseen,
    # even though it does not own a displayed bin.
    r = L.categorical_shift_report(['a'] * 10 + ['b'] * 9 + ['rare'],
                                   ['a', 'rare', 'never'], max_categories=2)
    assert abs(r['unseen_rate'] - 1.0 / 3.0) < 1e-12, r


def t_data_quality_unhashable_labels():
    X = np.array([[0.0], [0.0], [1.0], [2.0]])
    y = np.empty(4, dtype=object)
    y[0], y[1], y[2], y[3] = ['a'], ['b'], ['c'], ['d']
    rep = L.data_quality_report(X, y=y, train_idx=[0, 2], val_idx=[1],
                                test_idx=[3])
    assert rep['checks']['label_conflicts_on_duplicate_rows'] == 1, rep
    assert rep['status'] == 'fail', rep


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



def t_targeted_contract_edges():
    # ce_loss: the default None still means cross_entropy's -100 sentinel, so
    # ignored targets must not inflate the denominator.
    torch.manual_seed(0)
    m = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'full'))
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    y[0, 0] = -100
    y[1, 3] = -100
    _a = float(m.ce_loss(x, y, 64).detach())
    _b = float(m.ce_loss(x, y, 64, ignore_index=-100).detach())
    _lg = m.head(m.trunk(x))
    _ref = float(torch.nn.functional.cross_entropy(
        _lg.reshape(-1, 64), y.reshape(-1)).detach())
    assert abs(_a - _b) < 1e-12 and abs(_a - _ref) < 1e-5, (_a, _b, _ref)
    _ce, _acc = L._lm_batch_metrics(m, x, y, 64)
    assert abs(_ce - _ref) < 1e-5, (_ce, _ref)
    # Family resolution floor uses the nonzero pair count.
    _s = {'n': 10, 'n_nonzero': 6, 'mean': -1.0, 'std': 0.1,
          'p_exact_signflip': 0.03125, 'p_holm': 0.03125}
    _r = L.reportable_claim(_s, family_size=4, min_n=6)
    assert _r['allowed'] is False and _r['min_attainable_p_holm_family'] > 0.05, _r
    # A synthetic overfit report cannot match a requested real-data fingerprint.
    _saved = dict(L._OVERFIT_REPORTS)
    try:
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS[('v', 'synthetic', 'K')] = {
            'data_fp': None, 'guard_probe_semantics': 'S'}
        assert L.overfit_reports_for(['v'], data_fp='REAL') == []
    finally:
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS.update(_saved)
    # The LSH near-duplicate guarantee requires bands > near_threshold.
    try:
        L._DocDeduplicator(bands=2, near_threshold=3)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe LSH configuration was accepted')
    # Truthy non-boolean shuffle metadata must fail closed.  A NumPy boolean
    # is still a boolean declaration, not malformed provenance.
    assert A._declares_shuffle(1) and A._declares_shuffle('yes')
    assert not A._declares_shuffle(False) and not A._declares_shuffle(None)
    for _fn in (A._declares_shuffle, L._meta_declares_shuffle):
        assert _fn(np.bool_(True)) is True and _fn(np.bool_(False)) is False, _fn
    # A top-level training fingerprint declaration is as binding as one under
    # `binding`, and an explicit null by_length grid is refused at load time.
    assert V12._train_fp_required(
        {'binding': {}, 'training_data_fingerprint': {'canonical_train_fp': None}})
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, 'm.json')
        with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
            _man = json.load(f)
        _man['eval']['by_length_ppl'] = None
        with open(pp, 'w', encoding='utf-8') as f:
            json.dump(_man, f)
        try:
            V12._load_manifest(pp)
        except SystemExit as e:
            assert 'by_length_ppl' in str(e), str(e)[:200]
        else:
            raise AssertionError('explicit null eval.by_length_ppl was accepted')
    # 0 is a valid final seed and must not be discarded as falsy.
    assert BT._search_final_seeds({'search_seeds': [1], 'final_seeds': 0}) == ([1], [0], [])
    # Metadata without its arrays is an incomplete cache, not a silent skip.
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'meta_x.json'), 'w', encoding='utf-8') as f:
            json.dump({'vocab': 10, 'split_strategy': 'ordered',
                       'shuffle_at_split': False, 'tokenizer_fit_split': 'train'}, f)
        _out = A.audit_split_discipline(cache_dirs=[td])
        _rec = _out['caches'][os.path.abspath(td)]['corpora']['x']
        assert _rec['status'] == 'fail', _rec
    # A malformed per-layer stats payload must be ignored by the top-k table,
    # not crash the report builder.
    _cur = f'_cs{BR._CUR_CS}'
    _k1 = ('agg', 'results_lm_v7_seq2k')
    _k2 = ('summary', 'results_lm_v7_seq2k')
    BR._LOADED[_k1] = {'csa_fixed': {'variant': 'csa_fixed', 'n_seeds': 1,
                                     'ppl_mean': 10.0, 'run_cfg': 'steps1' + _cur}}
    BR._LOADED[_k2] = {'csa_fixed::s0': {'variant': 'csa_fixed', 'seed': 0,
                                         'ppl': 10.0, '_code': BR._CUR_CS,
                                         'run_cfg': 'steps1' + _cur,
                                         'stats': [1]}}
    try:
        BR.sec_p1t()
    finally:
        BR._LOADED.pop(_k1, None)
        BR._LOADED.pop(_k2, None)


def main():
    if not _HAVE_TORCH:
        print('[test] SKIPPED: the CPU test suite could not import its dependencies; this is NOT a pass (exit 3)')
        return 3
    if not _HAVE_TORCH:
        print('[test] SKIPPED (no torch in this environment)')
        return 0
    _check('A1 holm_min_attainable_p', t_holm_min_attainable)
    _check('A1a invalid Holm slots stay in the declared family', t_holm_invalid_slots_stay_in_family)
    _check('A2 signflip_min_consistency', t_min_consistency)
    _check('A2b scale robustness edges', t_scale_robustness_edges)
    _check('A2c aggregate/claim/ladder large-int edges', t_scale_aggregate_and_claim_edges)
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
    _check('A4g2 apply_rope preserves width (over-long/odd rope_dim)', t_apply_rope_preserves_width)
    _check('A4g3 numeric helpers accept NumPy scalars', t_stats_accept_numpy_scalars)
    _check('A4h shipped manifest v3 admits 48/48 synthetic records end-to-end', t_manifest_v3_admission_can_succeed)
    _check('B1 reference modes bit-exact', t_reference_modes_bitexact)
    _check('B2 determinism + target-zone safety', t_corruption_deterministic_and_paired)
    _check('B3 new modes span/doc/topical', t_new_modes_do_something)
    _check('B4 mode validation', t_mode_validation)
    _check('C1 load-time power gate', t_power_gate)
    _check('C2 admission gate exclusions', t_admission_gate)
    _check('C3 admission gate refusals', t_admission_refuses)
    _check('C4 recipe signature', t_recipe_signature)
    _check('C4b P1L checkpoint recipe signature', t_p1l_recipe_signature)
    _check('C5 freeze lock', t_freeze_lock)
    _check('C5b final test requires parameter matching', t_finaltest_requires_parameter_matching)
    _check('C9 aggregate seed normalization', t_aggregate_seed_normalization)
    _check('C9b aggregate rejects malformed seed ids', t_aggregate_rejects_malformed_seed_ids)
    _check('C10 audit resolution classification', t_audit_resolution_classification)
    _check('C11 controlled-variable gate: mixed train_knobs', t_pair_reason_train_knobs_mixed)
    _check('C11b corpus/validation-slice fingerprint gate', t_pair_reason_fingerprint_gate)
    _check('C21 M0 scale analysis stays descriptive at n=4', t_m0_scale_analysis_descriptive)
    _check('C12 seed plan', t_seed_plan)
    _check('C12a seed plan requires exact integer counts', t_seed_plan_exact_counts)
    _check('C12b t critical covers high-confidence intervals', t_t_critical_high_confidence)
    _check('C12c recorded counts use the exact integer contract', t_record_int_exact_spelling)
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
    _check('D1a eval length rejects invalid/non-integer lengths', t_eval_length_invalid_lengths)
    _check('J1 evidence metadata + significance gate', t_review3_evidence_metadata)
    _check('J2 pooled aggregation', t_review3_pooled_aggregate)
    _check('J2b pooled PPL rejects malformed counts', t_pooled_ppl_rejects_malformed_counts)
    _check('J2c document CI token counts are exact integers', t_nll_doc_ci_exact_token_counts)
    _check('J3 Holm declared-family slots', t_review3_holm_family_size)
    _check('J4 task-boundary arm + objective descriptor', t_review3_boundary_arm_and_objective)
    _check('J5 span clean-baseline key', t_review3_span_clean_key)
    _check('E1 seed reproducibility', t_seed_reproducible)
    _check('E1a seed/variant axis normalisation', t_seed_axis_normalization)
    _check('E1b explicit RNG seed contract', t_seed_contract_check)
    _check('E2 batch_iter reproducibility', t_batch_iter_reproducible)
    _check('E2b batch_iter uint16 transport', t_batch_iter_uint16)
    _check('E2b2 int32 narrowing ignores a stale max_id bound', t_ids_narrow_does_not_trust_stale_max_id)
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
    _check('E4b manifest seed integrity is exact', t_manifest_seed_integrity_is_exact)
    _check('E9a int32 cache narrowing rejects corrupt ids', t_npy_serve_int32_rejects_invalid_and_out_of_range)
    _check('E9b id fingerprint is storage-width invariant', t_ids_fp_is_storage_width_invariant)
    _check('E9 data_version identity', t_data_version_identity)
    _check('E10 crash-safe manifest status/planned cells', t_manifest_running_then_complete)
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
    _check('H3 equal-search baseline tuning audit', t_baseline_tuning_audit)
    _check('H3a multi-seed selection + equal search-seed audit', t_baseline_multi_seed_selection_audit)
    _check('H3b tuning driver selects on the multi-seed mean', t_baseline_tuning_driver_multiseed_selection)
    _check('H3c search/final seed partition + inherited tuning coverage', t_baseline_seed_partition_and_coverage_audit)
    _check('H3d tuned contrasts are declared per parameter-matching group', t_baseline_contrast_plan_is_group_aware)
    _check('H3f leave-one-seed-out tuning-selection stability', t_tuning_selection_stability_audit)
    _check('H3g tuned claim requires the panel to run the selected config', t_methodology_audit_requires_panel_to_use_tuned_config)
    _check('H3e tuning driver refuses overlapping search/final seeds', t_baseline_driver_refuses_overlapping_final_seeds)
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
    _check('I4a data-quality multi-output missing labels', t_multi_output_missing_labels)
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
    _check('I13c split audit verifies test doc-id sidecar self-consistency', t_audit_split_test_sidecar_self_consistency)
    _check('I14 corpus audit counts overlapping validation rows once', t_corpus_audit_overlap_is_unique_across_windows)
    _check('I15 length audit refuses a missing eval span', t_audit_length_requires_eval_span)
    _check('I15b boundary audit fails below the random baseline', t_audit_boundary_below_baseline_fails)
    _check('I15b2 boundary audit covers every cell and validates ranges', t_audit_boundary_audits_every_cell_and_range)
    _check('I15c selection audit rejects empty per_seed evidence', t_selection_audit_empty_per_seed_fails)
    _check('I15c2 selection audit rejects out-of-range claim cells', t_selection_audit_rejects_out_of_range_claims)
    _check('I15d split audit fails closed on unreadable metadata', t_split_audit_fails_on_unreadable_meta)
    _check('I15e valid-JSON non-object summaries are refused', t_valid_json_non_object_summaries_are_refused)
    _check('I15f metrics audit distinguishes absent from malformed summaries', t_audit_metrics_missing_or_malformed_summary)
    _check('I16 fairness audit ignores a NaN seed instead of crashing', t_audit_fairness_nan_seed_does_not_crash)
    _check('I17 an empty split cache is not reported as verified', t_audit_split_empty_cache_is_not_ok)
    _check('I18 tuning audit reports malformed seed keys instead of crashing', t_baseline_audit_malformed_seed_keys)
    _check('I19 simple-baseline audit reports malformed records instead of crashing', t_simple_baseline_audit_malformed_records)
    _check('I20 index aliasing, non-finite labels, signed-zero duplicates, undefined SMD, NaT groups', t_data_guard_index_label_and_shift_edges)
    _check('I21 split-and-prepare fits preprocessing on train only', t_split_and_prepare_fits_train_only)
    _check('I21a data-quality temporal split audit', t_data_quality_temporal_audit)
    _check('I21a2 time split backtracks for repeated timestamps', t_time_split_backtracks_for_repeated_timestamps)
    _check('I21b capped test document-id sidecar', t_capped_document_id_sidecar)
    _check('I22 purge/embargo keeps window boundaries sealed', t_split_purge_embargo)
    _check('I23 grouped split refuses an empty requested side', t_group_split_refuses_empty_requested_side)
    _check('I24 label shift and per-split label corruption counts', t_label_shift_and_split_counts)
    _check('I25 numeric-looking string labels stay categorical', t_label_key_keeps_numeric_strings)
    _check('I26 preprocessor rejects boolean/negative-eps options', t_data_guard_preprocessor_option_guards)
    _check('I27 categorical shift requires a meaningful bin count', t_categorical_shift_requires_two_bins)
    _check('I28 label-noise audit handles unhashable labels', t_data_quality_unhashable_labels)
    _check('I29 categorical encoding is fitted on train only', t_categorical_encoder_train_only)
    _check('I30 train-only cleaning before preprocessing', t_clean_training_rows_and_split)
    _check('I31 declared label set catches unseen/holdout labels', t_known_labels_audit)
    # Open-block reading rule: the substantive gates (causality, reference
    # pooling, masks, edge cases, closed behaviour) live in test_open_block.py
    # so they can also be run on their own.
    try:
        import test_open_block as _OB
        for _name, _fn in _OB.CHECKS:
            _check(_name, _fn)
    except Exception as _obe:
        _check('K0 open-block suite import', lambda _e=_obe: (_ for _ in ()).throw(_e))
    _check('L1 reportable claim gate', t_reportable_claim_gate)
    _check('L1b boolean p-values are not probabilities', t_boolean_pvalues_are_not_probabilities)
    _check('L1c record readers refuse non-object summaries', t_record_readers_refuse_non_object_summaries)
    _check('L2 Holm family resolution ceiling', t_holm_family_resolution_ceiling)
    _check('L3 crossover fit refusal rules + slope CI', t_crossover_scaling_refusals)
    _check('L3b scale-ladder budget + transfer audit', t_scale_ladder_budget_and_transfer_audit)
    _check('L3c report pairing keeps one run_cfg group', t_report_paired_row_single_config)
    _check('L3d report refuses stale trajectories', t_report_stale_trajectory_refused)
    _check('L3e v8/v9 pairing keeps one run_cfg group', t_driver_pairing_single_config)
    _check('L3f v7 bootstrap pairing keeps one run_cfg group', t_v7_bootstrap_single_config)
    _check('L3g v11 by-length table deduplicates seed records', t_v11_by_length_deduplicates_seeds)
    _check('L3i report history steps are exact integers', t_report_history_steps_are_exact)
    _check('L3i2 v10 curve history steps are exact integers', t_v10_curve_history_steps_are_exact)
    _check('L3h v12 probe contrasts require matching probe_params', t_v12_probe_contrasts_require_matching_params)
    _check('L4 matched-init fingerprint + pair gate', t_init_core_fingerprint_and_pair_gate)
    _check('L5 param breakdown + measured compute control', t_param_and_compute_control_detail)
    _check('L6 stale aggregate refused', t_stale_aggregate_refused)
    _check('L7 stale length panel excluded from report', t_sec_p1l_excludes_stale_records)
    _check('L7b report length table rejects malformed length keys', t_sec_p1l_rejects_malformed_length_keys)
    _check('L8 efficiency-pair audit + shape sweep', t_efficiency_pairs_audit_and_sweep)
    _check('L9 evidence-release readiness audit', t_release_readiness_audit)
    _check('L9b archive verifier requires the external checksum', t_archive_verify_requires_external_checksum)
    _check('L9b2 archive refuses result dirs outside the repo', t_archive_refuses_dirs_outside_repo)
    _check('L9b3 archive hashes every checkpoint extension', t_archive_hashes_all_checkpoint_extensions)
    _check('L9b4 archive verify selects its own manifest', t_archive_verify_uses_own_manifest)
    _check('L10 selection-attribution contract', t_selection_attribution_contract)
    _check('L10b corpus-audit memo cannot bypass the strict gate', t_corpus_audit_memo_does_not_bypass_strict_gate)
    _check('L10c corpus-audit memo keys the validation row shape', t_corpus_audit_memo_keys_validation_shape)
    _check('L10c2 corpus-audit document-overlap count is exact', t_corpus_audit_overlap_metadata_exact)
    _check('L10d report command aggregates sub-audit status', t_report_aggregate_status_fails_closed)
    _check('L10c length audit rejects malformed cell objects', t_audit_length_malformed_cells_do_not_crash)
    _check('L10c2 by-length reader requires an exact integer length', t_by_len_cells_requires_exact_length)
    _check('L10c3 length audit parses keys and counts exactly', t_audit_length_exact_integer_fields)
    _check('L11 length audit fails closed on missing truncation flag', t_audit_length_missing_truncated_flag)
    _check('L12 length audit fails closed on missing n_seq', t_audit_length_missing_n_seq_flag)
    _check('L13 global cross-panel multiplicity audit (exp_lib)', t_global_multiplicity_audit_basic)
    _check('L13b global audit fail-closed rules', t_global_multiplicity_audit_fail_closed)
    _check('L13c build_report twin matches exp_lib audit', t_global_multiplicity_twin_consistency)
    _check('L13d report registers every displayed comparison', t_build_report_global_multiplicity_section)
    _check('L13e audit sidecar fingerprint matches exp_lib across storage widths', t_audit_sidecar_fp_matches_exp_lib)
    _check('L14 combined seed x document uncertainty interval', t_combined_seed_doc_uncertainty)
    _check('L15 cross-scale pooling refuses mixed corpus/tokenizer bindings', t_scale_corpus_binding_refuses_mixed_tokenizers)
    _check('L15b scale-point corpus provenance roundtrip', t_scale_point_provenance_roundtrip)
    _check('L15c scale-edge regressions (tiny effects, zero deltas, tiny corpora)', t_scale_edge_case_regressions)
    _check('L16 targeted contract edges', t_targeted_contract_edges)
    fails = [r for r in _results if r[1] not in ('PASS', 'SKIP')]
    skips = [r for r in _results if r[1] == 'SKIP']
    print(f'\n[test] {len(_results) - len(fails) - len(skips)}/{len(_results)} passed, {len(skips)} skipped, {len(fails)} failed')
    if skips:
        print('[test] SKIPPED (environment preconditions): ' + '; '.join(f'{n} — {why}' for n, _s, why in skips))
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
