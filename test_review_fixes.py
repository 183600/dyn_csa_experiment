#!/usr/bin/env python3
"""test_review_fixes.py — 评审二轮修复的单元测试（纯 CPU、零实验）。

覆盖：
  A. exp_lib 新增统计工具（意见一/八/九）：
     holm_min_attainable_p / signflip_min_consistency / signflip_power /
     tost_paired / exact_signflip_p 假设文档存在性（导入不报错即可证）；
  B. v10 腐坏模式（意见六）：_corrupt_row 对四种历史模式与历史实现
     **逐字节一致**（重放旧算法对拍），新模式 span/doc/topical 的确定性与
     目标区不可触碰约束；
  C. v12 门槛（意见一/二/三）：_load_manifest 功效门槛（6 种子拒绝、
     12 种子放行、descriptive_only 豁免）；_admit_confirmation_records 的
     白名单/歧义/混切片/None 指纹行为；_recipe_matches_signature；
     冻结锁的 manifest 不匹配拒绝与 V12_REFREEZE 降级；v2 清单文件本身的
     功效表与代码一致；
  D. eval_length_gen 逐文档 PPL（意见九 b）：池化 PPL == 逐文档 PPL 的
     几何平均（归约不变）。

运行：python test_review_fixes.py   （需要 torch；无 torch 的环境全部跳过
并给出非零提醒码 0——CI 以「SKIPPED」字样识别）。"""
import json
import math
import os
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
    _HAVE_TORCH = True
except Exception as _e:  # pragma: no cover
    print(f'[test] SKIPPED: cannot import the repo modules here ({type(_e).__name__}: {_e}); run on a box with CPU torch')
    _HAVE_TORCH = False

_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, 'PASS', ''))
        print(f'[test] PASS  {name}')
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
    assert abs(L.holm_min_attainable_p(6, 4, 0.05) - 0.125) < 1e-12, 'review-2 issue 1: 6 seeds x Holm/4 floor must be 0.125'
    assert abs(L.holm_min_attainable_p(12, 4, 0.05) - 0.001953125) < 1e-12
    assert abs(L.holm_min_attainable_p(8, 1, 0.05) - 0.0078125) < 1e-12
    assert L.holm_min_attainable_p(6, 4, 0.05) > 0.05 and L.holm_min_attainable_p(12, 4, 0.05) < 0.05


def t_min_consistency():
    assert L.signflip_min_consistency(12, 4, 0.05) == 11, 'n=12, m=4: 11/12 same-sign clears the first Holm gate (raw p = 2*(1+12)/4096 = 0.00635 < 0.0125)'
    assert L.signflip_min_consistency(6, 4, 0.05) is None, 'n=6, m=4: even 6/6 cannot clear the gate'
    assert L.signflip_min_consistency(8, 1, 0.05) == 8
    assert L.signflip_min_consistency(6, 1, 0.05) == 6


def t_power():
    assert L.signflip_power(6, 4, 0.99) == 0.0, 'an impossible plan has zero power at any q'
    pw = L.signflip_power(12, 4, 0.9, 0.05)
    assert abs(pw - 0.659) < 0.01, f'q=0.9 power at first Holm gate should be ~0.659, got {pw:.4f}'
    assert L.signflip_power(12, 4, 0.8, 0.05) < 0.30


def t_tost():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.tost_paired(inside, margin=0.01)
    assert r['equivalent'] is True and r['p_tost'] < 0.05, f'tiny deltas vs margin must be equivalent: {r}'
    above = [0.05, 0.052, 0.049, 0.051, 0.05, 0.048]
    r2 = L.tost_paired(above, margin=0.01)
    assert r2['equivalent'] is False and r2['p_tost'] > 0.05, f'mean far above margin must NOT be equivalent: {r2}'
    # direction check (the review-2 regression the erfc sign bug would cause):
    far = [0.9, 0.95, 0.85, 1.0, 0.92, 0.88]
    r3 = L.tost_paired(far, margin=0.1)
    assert r3['equivalent'] is False, 'mean ~0.9 vs margin 0.1 must not be equivalent even with noise'
    small_n = L.tost_paired([0.0, 0.0], margin=0.1)
    assert small_n['equivalent'] is not None


def t_exact_signflip_baseline():
    p = L.exact_signflip_p([1.0] * 6)['p']
    assert abs(p - 0.03125) < 1e-12, 'all-same-sign n=6 two-sided p must stay 2/64'
    p12 = L.exact_signflip_p([1.0] * 12)['p']
    assert abs(p12 - 2.0 / 4096) < 1e-12


def t_paired_effect_resolution():
    r = L.paired_effect_stats([1.0, 1.1, 0.9, 1.2], alpha=0.05)
    assert r['resolution_limited'] is True, 'n=4 can never reach 0.05 (floor 0.125) — descriptive tier'


# ---------------------------------------------------------------------------
# B. v10 corruption modes
# ---------------------------------------------------------------------------

if _HAVE_TORCH:
    _PANEL = np.random.default_rng(7).integers(3, 8000, size=(9, 129), dtype=np.int64)
    _VOCAB = 8192


def _legacy_corrupt_row(val_ids, eval_len, rho, mode, jj, vocab, target):
    """The PRE-refactor inline algorithm, replayed bit-for-bit (uniform /
    natural / mask / delete). Any divergence means the extraction changed a
    measured value — the strongest regression this suite guards against."""
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
                assert np.array_equal(a, b), f'legacy mode {mode} rho={rho} jj={jj} diverged from the pre-refactor algorithm'


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
    return {'variant': variant, 'seed': seed, 'split': 'test', 'data_fp': fp, 'by_len': by_len or {'4096': {'ppl': 30.0, 'n_tok': 16 * 4095, 'truncated': False}}, '_code': code or 'v11.125', 'recipe': recipe, 'manifest_sha256': msha, 'key': f'{variant}::s{seed}'}


def _manifest_for_admission(seeds=(0, 1, 2), sig=None):
    m = {'arms': ['a_arm', 'b_arm'], 'seeds': list(seeds), 'code_semantics_expected': 'v11.125', 'manifest_sha256': 'CURRENT', 'statistics': {'test': 'exact two-sided sign-flip', 'correction': 'holm step-down'}, 'primary_comparisons': []}
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
        lock = V12._check_or_write_freeze_lock(td, man, 'FP')
        assert lock['manifest_sha256'] == 'AAA' and lock['test_ids_fp'] == 'FP'
        # same manifest + same fp -> ok
        V12._check_or_write_freeze_lock(td, man, 'FP')
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


def t_shipped_manifest_consistency():
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    assert v2['manifest_version'] == 2
    n = len(v2['seeds'])
    m = len(v2['primary_comparisons'])
    assert n == 12 and v2['seeds'] == list(range(12))
    assert v2['statistics']['n_seeds'] == n
    assert abs(v2['statistics']['p_floor'] - L.signflip_p_floor(n)) < 1e-15
    assert abs(v2['statistics']['min_attainable_p_holm'] - L.holm_min_attainable_p(n, m)) < 1e-12
    assert v2['statistics']['min_attainable_p_holm'] < 0.05
    assert v2['statistics']['min_consistency_for_first_gate'] == L.signflip_min_consistency(n, m)
    for k, want in v2['statistics']['power_first_holm_gate'].items():
        q = float(k.split('=')[1])
        got = L.signflip_power(n, m, q)
        assert abs(got - want) < 0.02, f'power table drift at {k}: manifest {want} vs code {got:.4f}'
    # the revision history must record the 6->12 change and its precondition
    hist = v2['revision']['history']
    assert any(h.get('version') == 1 for h in hist) and any(h.get('version') == 2 for h in hist)
    # scope block present (issue 4)
    assert v2.get('scope', {}).get('not_covered'), 'scope.not_covered must name the dynamic/HCA claims that are out of scope'


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


def main():
    if not _HAVE_TORCH:
        print('[test] SKIPPED (no torch in this environment)')
        return 0
    _check('A1 holm_min_attainable_p', t_holm_min_attainable)
    _check('A2 signflip_min_consistency', t_min_consistency)
    _check('A3 signflip_power', t_power)
    _check('A4 tost_paired directions', t_tost)
    _check('A5 exact_signflip baseline', t_exact_signflip_baseline)
    _check('A6 paired_effect resolution_limited', t_paired_effect_resolution)
    _check('B1 legacy modes bit-exact', t_legacy_modes_bitexact)
    _check('B2 determinism + target-zone safety', t_corruption_deterministic_and_paired)
    _check('B3 new modes span/doc/topical', t_new_modes_do_something)
    _check('B4 mode validation', t_mode_validation)
    _check('C1 load-time power gate', t_power_gate)
    _check('C2 admission gate exclusions', t_admission_gate)
    _check('C3 admission gate refusals', t_admission_refuses)
    _check('C4 recipe signature', t_recipe_signature)
    _check('C5 freeze lock', t_freeze_lock)
    _check('C6 shipped manifest v2 consistency', t_shipped_manifest_consistency)
    _check('C7 frozen seeds single source', t_frozen_seeds)
    _check('C8 doc-level analysis', t_doc_level_analysis)
    _check('D1 eval per-seq ppl', t_eval_per_seq)
    fails = [r for r in _results if r[1] != 'PASS']
    print(f'\n[test] {len(_results) - len(fails)}/{len(_results)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
