#!/usr/bin/env python3
"""v12 confirmation suite — 探索/确认分离的确认性实验（评审点 4 + 5）。

前序版本（v7→v11）的全部面板都围绕同一个 validation split 展开，且多轮
追加种子/比较属于**探索性**证据。本驱动把「确认」独立出来：

  P5C   —— 同预算从头训练的选择器对照（csa_fixed_rope_randidx /
           csa_fixed_rope_allblocks）补到 seeds 0-5（0-2 由 v10 P3MTC 覆盖，
           同 recipe 下 resume 自动跳过）；
  P5MP  —— 扩展探针在 seeds 0-5 重跑（uniform/natural/mask/delete 腐坏 +
           NIAH-natural + 从头训练对照臂）；
  P5S2  —— 等训练 token 的大模型规模臂（P3SL_TOK，8000 步）；
  P5S3  —— 等计算预算的大模型规模臂（P3SL_FLOP，参数量比例缩步）；
  P5T   —— **冻结后的独立最终测试**：按仓库根的冻结清单
           `confirmation_frozen.json`（预注册主比较、终点、种子数、检验与
           校正规则），在 WikiText-103 的 held-out test split（wiki.test.raw，
           此前任何 phase 都未读过）上一次性评估冻结臂，输出确认性统计。

模式：
  python v12_supp.py smoke            # 端到端自检（微型 ckpt + 冻结清单副本）
  python v12_supp.py full             # P5C -> P5MP -> P5S2 -> P5S3 -> P5T -> report
  python v12_supp.py phase P5C        # 单阶段（可断点续跑）
  python v12_supp.py analysis         # 零 GPU：确认性统计 -> analysis_v12/
  python v12_supp.py report           # 从落盘产物重建 REPORT_v12.md（零 GPU）

P5T 的防呆：代码语义戳与冻结清单不一致时拒绝运行（必须显式重冻结）；
manifest 不存在时拒绝运行（不许「先看数据再补注册」）。
"""
import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
import numpy as np
import torch
import torch.nn.functional as F
import v10_supp as V10
V9 = V10.V9
V8 = V10.V8
V = V10.V
L = V10.L
REPO = V10.REPO
DEVICE = L.DEVICE
RUN_CAP_V12 = dict(L.RUN_CAP)
RUN_CAP_V12.update(total_hours=float(os.environ.get('V12_CAP_HOURS', 30.0)), state_path='run_time_state_v12.json', already_hours=0.0)

def make_guard():
    g = L.TimeGuard(RUN_CAP_V12)
    if not g.state.get('sps_by_class'):
        for src in ('run_time_state_v11.json', 'run_time_state_v10.json', 'run_time_state_v9.json', 'run_time_state_v8.json', 'run_time_state_v7.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v12] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v12] TimeGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v12] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g

# ---- 阶段配置 -------------------------------------------------------------
P5C_CFG = dict(V10.P3MT_PAYLOAD, seeds=[0, 1, 2, 3, 4, 5], variants=list(V10.SCRATCH_CONTROLS))
P5MP_CFG = dict(V10.PROBE, seeds=[0, 1, 2, 3, 4, 5], variants=['csa_fixed_rope', 'full_rope'] + V10.SCRATCH_CONTROLS, arm_overrides={v: ['learned'] for v in V10.SCRATCH_CONTROLS})
P5S2_CFG = dict(V10.P3SL_TOK_CFG, seeds=[0, 1])
P5S3_CFG = dict(V10.P3SL_FLOP_CFG, seeds=[0, 1])
TEST_OUTDIR = 'results_lm_test'
FREEZE_MANIFEST = 'confirmation_frozen.json'
PHASES = [('P5C', 'lenphase', P5C_CFG, None), ('P5MP', 'probe', P5MP_CFG, None), ('P5S2', 'run', P5S2_CFG, None), ('P5S3', 'run', P5S3_CFG, None), ('P5T', 'finaltest', None, None)]

def run_phase(name, guard, par=None):
    import par as _par
    _ctx = _par.resolve_ctx(par, label=f'v12 {name}')
    for pname, kind, payload, seeds in PHASES:
        if pname != name:
            continue
        cfg = payload() if callable(payload) else payload
        if isinstance(_ctx, dict):
            _ctx = dict(_ctx, module='v12_supp', ckpt_install=True, ckpt_flag=False)
        if kind == 'lenphase':
            return V.run_lenphase(cfg, guard=guard, label=f'v12 {pname}', par=_ctx)
        if kind == 'probe':
            return V10.run_probe(cfg, guard=guard, label=f'v12 {pname}', par=_ctx)
        if kind == 'finaltest':
            print('[v12] NOTE: the final-test phase has no task-level parallel path, running it serially (evaluation only, ckpt per cell)')
            return run_finaltest(dict(FINALTEST_CFG), guard=guard)
        if cfg.get('steps') is None and pname == 'P5S3':
            cfg = dict(cfg, steps=V10._flop_matched_steps(V10.P3SS_CFG, cfg))
        if seeds:
            cfg = dict(cfg, seeds=list(seeds))
        s, _a = L.run(cfg, seeds=cfg.get('seeds'), guard=guard, label=f'v12 {pname}', par=_ctx)
        return s
    raise SystemExit(f'unknown phase {name}')

# ---- P5T：冻结后的独立最终测试 ----------------------------------------------
FINALTEST_CFG = dict(manifest=FREEZE_MANIFEST, outdir=TEST_OUTDIR, ckpt_dir='results_v10_mech/ckpt', mech_summary='results_v10_mech/summary.json', bylen_summary=os.path.join(TEST_OUTDIR, 'bylen.json'), probe_summary=os.path.join(TEST_OUTDIR, 'distractor.json'), eval_lens=[512, 1024, 2048, 4096], n_seq=16, target=512, chunk=4)

def _load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} does not exist — a confirmation run requires a PRE-REGISTERED plan (arms, seeds, endpoints, statistics); write the manifest first, then run P5T')
    try:
        m = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} cannot be parsed ({type(e).__name__}: {e})')
    for field in ('arms', 'seeds', 'primary_comparisons', 'statistics'):
        if field not in m:
            raise SystemExit(f'[v12] REFUSE: freeze manifest {path} lacks the required field {field!r} — refusing to guess the pre-registration')
    _exp = m.get('code_semantics_expected')
    if _exp is not None and str(_exp) != str(L.CODE_SEMANTICS):
        raise SystemExit(f'[v12] REFUSE: freeze manifest expects code semantics {_exp!r} but the current code is {L.CODE_SEMANTICS!r} — the code changed after freezing; re-freeze consciously (update {path} with a dated note) instead of silently testing new code under the old pre-registration')
    return m

@torch.inference_mode()
def _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids):
    """One (arm, seed) clean by-length evaluation on the TEST split."""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['bylen_summary']
    out = {'status': 'ok', 'key': f'{v}::s{seed}', 'variant': v, 'seed': seed}
    ck = os.path.join(cfg['ckpt_dir'], f'{v}_seed{seed}.pt')
    if not os.path.exists(ck):
        print(f'[p5t] SKIP {v} s{seed}: no checkpoint {ck} (run P3MT/P4MT/P5C first)')
        out['status'] = 'skipped_no_ckpt'
        return out
    try:
        d = V._load_ckpt(ck)
    except Exception as _cle:
        print(f'[p5t] SKIP {v} s{seed}: checkpoint unreadable ({type(_cle).__name__}: {_cle})')
        out['status'] = 'skipped_bad_ckpt'
        return out
    if d.get('code') != V.CKPT_CODE:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint predates the current code stamp (code={d.get("code")!r})')
        del d
        out['status'] = 'skipped_stale_ckpt'
        return out
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    _tr = mech_summ.get(f'{v}::seed{seed}') or {}
    if not _tr or not L.result_is_current(_tr, V.CKPT_CODE, 'by_len') or _tr.get('synthesized'):
        print(f'[p5t] REFUSE {v} s{seed}: no current training record backs the checkpoint — the weights have no traceable provenance')
        del d
        out['status'] = 'skipped_no_provenance'
        return out
    _ck_recipe = d.get('recipe')
    eval_lens = [int(x) for x in manifest.get('eval', {}).get('by_length_ppl', cfg['eval_lens'])]
    _cur = L.locked_json_read(spath).get(f'{v}::s{seed}')

    def _stale(rec):
        if not isinstance(rec, dict):
            return True
        if rec.get('_code') != V.CKPT_CODE or rec.get('recipe') != _ck_recipe:
            return True
        bl = rec.get('by_len') or {}
        return not all((L.by_len_cells([rec], Ln) for Ln in eval_lens))
    if not _stale(_cur):
        del d
        out['status'] = 'skipped_resume'
        return out
    _arch = (int(d.get('d', 256)), int(d.get('n_layers', 6)), int(d.get('n_heads', 8)), int(d.get('d_head', 32)))
    model = L.SmallGPT(int(d.get('vocab', 8192)), *_arch, d.get('train_len', 512), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
    model.load_state_dict(d.pop('sd'))
    model.eval()
    r = V.eval_length_gen(model, test_ids, eval_lens, DEVICE, max_pos=None)
    del model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    def _merge(summary):
        summary[f'{v}::s{seed}'] = {'variant': v, 'seed': seed, 'params': _tr.get('params'), 'by_len': r, 'split': 'test', 'data_fp': L._ids_fp(np.ascontiguousarray(test_ids).reshape(-1)), '_code': V.CKPT_CODE, 'recipe': _ck_recipe}
        print(f'  [p5t] {v:32s} s{seed} ' + '  '.join((f'L{k}={vv.get("ppl", float("nan")):.2f}' for k, vv in sorted(r.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    del d
    return out

def run_finaltest(cfg=None, guard=None):
    """Evaluate the FROZEN confirmation set on the held-out test split.

    只读冻结清单列出的臂/种子/终点；评估数据来自 `L.load_wikitest`（test
    split，此前任何 phase 未读）。产物：`results_lm_test/bylen.json`（干净
    by-length）与 `results_lm_test/distractor.json`（uniform 腐坏探针）；
    确认性统计由 `v12_supp.py analysis` 从这些产物现算。"""
    cfg = dict(FINALTEST_CFG) if cfg is None else dict(cfg)
    manifest = _load_manifest(cfg['manifest'])
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    need = max([int(x) for x in manifest.get('eval', {}).get('by_length_ppl', cfg['eval_lens'])]) + 1
    test_ids = cfg.get('test_ids')
    if test_ids is None:
        _t0 = time.time()
        test_ids, vocab_t, _, _ = L.load_wikitest(need, n_seqs=int(cfg.get('n_seq', 16)))
        if guard is not None:
            guard.record_run(time.time() - _t0, 0, 0, 0, 0, 0)
    else:
        print('[p5t] NOTE: test_ids supplied by the caller (smoke only — the real P5T always reads the held-out wiki.test.raw via load_wikitest)')
    print(f'[p5t] manifest {cfg["manifest"]} frozen_at={manifest.get("frozen_at", "?")} arms={manifest["arms"]} seeds={manifest["seeds"]}')
    print(f'[p5t] TEST split rows {test_ids.shape} (held out; no earlier phase read this split)')
    # 1) clean by-length per (arm, seed)
    for v in manifest['arms']:
        for seed in manifest['seeds']:
            _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids)
    # 2) frozen uniform-corruption probe on the SAME test rows (secondary,
    #    descriptive-only per the manifest)
    _probe_eval = manifest.get('eval', {}).get('distractor_probe') or {}
    pcfg = dict(V10.PROBE, outdir=outdir, ckpt_dir=cfg['ckpt_dir'], summary=cfg['probe_summary'], variants=list(manifest['arms']), seeds=[int(s) for s in manifest['seeds']], eval_lens=[int(x) for x in (_probe_eval.get('eval_lens') or [2048, 4096])], rhos=[float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])], modes={'uniform': [float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])]}, niah=None, arm_overrides={v: ['learned'] for v in V10.SCRATCH_CONTROLS}, target=int(_probe_eval.get('target', cfg.get('target', 512))), n_seq=int(_probe_eval.get('n_seq', cfg.get('n_seq', 16))), chunk=int(cfg.get('chunk', 4)))
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    for v in pcfg['variants']:
        for seed in pcfg['seeds']:
            V10.run_probe_cell(pcfg, v, seed, guard=guard, val_ids=test_ids, mech_summ=mech_summ)
    # 3) confirmation statistics (zero GPU, from disk)
    return finaltest_stats(manifest=manifest, bylen_path=cfg['bylen_summary'], probe_path=cfg['probe_summary'], out=cfg.get('stats_out', 'analysis_v12/test_stats.json'))

def finaltest_stats(manifest=None, bylen_path=None, probe_path=None, out='analysis_v12/test_stats.json'):
    """Zero-GPU: paired confirmation statistics under the frozen plan."""
    if manifest is None:
        manifest = _load_manifest(FREEZE_MANIFEST)
    bylen_path = bylen_path or os.path.join(TEST_OUTDIR, 'bylen.json')
    probe_path = probe_path or os.path.join(TEST_OUTDIR, 'distractor.json')
    raw = {}
    if os.path.exists(bylen_path):
        raw = json.load(open(bylen_path, encoding='utf-8'))
    stats = {'manifest': {'frozen_at': manifest.get('frozen_at'), 'arms': manifest['arms'], 'seeds': manifest['seeds'], 'code_semantics_expected': manifest.get('code_semantics_expected'), 'statistics': manifest.get('statistics')}, 'comparisons': {}, 'holm': {}, 'by_length': {}}
    for v in manifest['arms']:
        rows = [r for r in raw.values() if isinstance(r, dict) and r.get('variant') == v and r.get('split') == 'test']
        _lens = sorted({int(k) for r in rows for k in (r.get('by_len') or {})})
        for Ln in _lens:
            _by_seed = {}
            for r in rows:
                bl = r.get('by_len') or {}
                c = bl.get(Ln, bl.get(str(Ln)))
                if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and (not c.get('truncated')):
                    _by_seed[r.get('seed')] = float(c['ppl'])
            if _by_seed:
                _dl = list(_by_seed.values())
                stats['by_length'].setdefault(v, {})[str(Ln)] = {'mean': float(np.mean(_dl)), 'n': len(_dl), 'ppl_by_seed': {str(s): p for s, p in _by_seed.items()}}
    endpoint_of = {}
    for comp in manifest['primary_comparisons']:
        name = comp['name']
        a, b = (comp['a'], comp['b'])
        ep = comp.get('endpoint', 'by_len.ppl@4096')
        Ln = int(str(ep).rsplit('@', 1)[-1])
        da, db = ({}, {})
        for r in raw.values():
            if not (isinstance(r, dict) and r.get('split') == 'test'):
                continue
            bl = r.get('by_len') or {}
            c = bl.get(Ln, bl.get(str(Ln)))
            if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and (not c.get('truncated')):
                if r.get('variant') == a:
                    da[r.get('seed')] = float(c['ppl'])
                if r.get('variant') == b:
                    db[r.get('seed')] = float(c['ppl'])
        common = sorted(set(da) & set(db))
        if len(common) < 2:
            stats['comparisons'][name] = {'omitted': f'no seed pair survives at the frozen endpoint (a: {sorted(da)}, b: {sorted(db)})', 'n': 0}
            continue
        dl = [da[s] - db[s] for s in common]
        res = V.exact_sign_permutation(dl)
        res['seeds'] = common
        res['delta_definition'] = f'{a} - {b} (paired per seed, test split)'
        stats['comparisons'][name] = res
    _fam = [(k, v.get('p_exact_signflip')) for k, v in stats['comparisons'].items() if isinstance(v.get('p_exact_signflip'), (int, float)) and math.isfinite(v.get('p_exact_signflip'))]
    if _fam:
        stats['holm'] = L.holm_adjust(_fam)
    _rej = [k for k, v in stats['holm'].items() if v.get('reject')]
    print(f'[p5t stats] {len(_fam)} frozen primary comparison(s); {len(_rej)} survive Holm at alpha=0.05' + (f': {sorted(_rej)}' if _rej else ''))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    L.atomic_write_json(out, stats, indent=2)
    print(f'[p5t stats] wrote {out}')
    return stats

def v12_analysis(out='analysis_v12/stats.json'):
    """Zero-GPU aggregate: the shared v10 analysis over the extended probe +
    the frozen final-test statistics (when they exist on disk)."""
    V10.v10_analysis(out=out)
    st = json.load(open(out, encoding='utf-8'))
    if os.path.exists(os.path.join(TEST_OUTDIR, 'bylen.json')):
        st['finaltest'] = finaltest_stats()
        L.atomic_write_json(out, st, indent=2)
    return st

def _fmt_pm(cell):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        return f'{cell["mean"]:.2f} ±{cell.get("std", 0.0):.2f}'
    return f'{cell:.2f}'

def build_report(out='REPORT_v12.md', stats_p='analysis_v12/stats.json'):
    v12_analysis(out=stats_p)
    st = json.load(open(stats_p, encoding='utf-8'))
    ft = st.get('finaltest') or {}
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v12 确认报告（冻结清单 + 独立 test split）')
    A('')
    A(f'> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    A('> 参考论文：arXiv:2606.19348（DeepSeek-V4 稀疏注意力的受控复现与机制剖析）')
    A('> 说明：本报告全部数字由 `v12_supp.py report` 从 `results_lm_test/`、`results_v10_mech/`、`analysis_v12/` 的落盘产物计算得到，无手填数值。')
    A('')
    A('本驱动落实评审点 4/5 的确认性协议：v7→v11 的全部面板与 v10/v12 的机制扩展属于**探索性**证据；本报告只报告**冻结后**在 held-out test split（wiki.test.raw，此前任何 phase 未读）上的一次性确认评估。探索性面板的完整数字见 REPORT_v10.md / REPORT_v11.md。')
    A('')
    A('---')
    A('')
    A('## 冻结清单（预注册）')
    A('')
    m = ft.get('manifest') or {}
    if not m and os.path.exists(FREEZE_MANIFEST):
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            m = {'frozen_at': _mc.get('frozen_at'), 'arms': _mc.get('arms'), 'seeds': _mc.get('seeds'), 'code_semantics_expected': _mc.get('code_semantics_expected'), 'statistics': _mc.get('statistics')}
        except Exception:
            pass
    if m:
        A(f'- 冻结时间：{m.get("frozen_at", "?")}；代码语义戳：`{m.get("code_semantics_expected", "?")}`（运行期校验一致才放行）。')
        A(f"- 冻结臂：{', '.join((f'`{v}`' for v in m.get('arms', [])))}；种子：{m.get('seeds')}。")
        _stat = m.get('statistics') or {}
        A(f"- 检验：{_stat.get('test', 'exact two-sided sign-flip')}；n={_stat.get('n_seeds', '?')}（p 值分辨率下限 {_stat.get('p_floor', float('nan'))}）；族 = primary_comparisons；校正 = {_stat.get('correction', 'holm step-down')}。")
        A(f"- 决策规则：{_stat.get('decision_rule', '—')}")
        A('')
        A('| 冻结的主比较 |')
        A('|---|')
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            for c in _mc.get('primary_comparisons', []):
                A(f"| {c.get('name', '?')} — {c.get('a')} vs {c.get('b')} @ {c.get('endpoint', '?')} |")
        except Exception:
            A('| （冻结清单不可读——见运行日志） |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。先跑 `python v12_supp.py phase P5C`（及 P3MT/P4MT 保证两主臂 ckpt 齐全），再跑 `python v12_supp.py phase P5T`。')
        A('')
    A('---')
    A('')
    A('## P5T 独立最终测试（test split，冻结终点）')
    A('')
    bl = ft.get('by_length') or {}
    if bl:
        _lens = sorted({int(k) for v in bl.values() for k in v})
        A('| 臂 | ' + ' | '.join((f'PPL@{Ln}' for Ln in _lens)) + ' |')
        A('|---|' + '---|' * len(_lens))
        for v in sorted(bl):
            row = []
            for Ln in _lens:
                e = bl[v].get(str(Ln))
                row.append(f'{e["mean"]:.2f} (n={e["n"]})' if e else '—')
            A(f'| `{v}` | ' + ' | '.join(row) + ' |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python v12_supp.py phase P5C`（及 P3MT/P4MT 保证两主臂 ckpt 到 6 种子），然后 `python v12_supp.py phase P5T`。冻结清单已入库（见上），随时可以执行；执行前请勿据任何 validation 读数调整冻结内容。')
        A('')
    comps = ft.get('comparisons') or {}
    if comps:
        A('| 冻结主比较（test split，配对） | n | Δ mean±std | d_z | p (exact) | p 下限 | Holm p | Holm 拒绝 |')
        A('|---|---|---|---|---|---|---|---|')
        holm = ft.get('holm') or {}
        for k, v in comps.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | —（{v["omitted"]}） | — | — | — | — | — |')
                continue
            _h = holm.get(k) or {}
            _hp = f'{_h.get("p_holm", float("nan")):.3f}' if _h.get('p_holm') is not None else '—'
            _hr = '✔' if _h.get('reject') else '✘'
            A(f'| {k} | {v["n"]} | {v["mean"]:+.2f} ± {v.get("std", 0):.2f} | {v.get("cohens_dz", float("nan")):+.2f} | {v["p_exact_signflip"]:.3f} | {v.get("p_floor", float("nan")):.3f} | {_hp} | {_hr} |')
        A('')
        _n_sig = [k for k, v in holm.items() if v.get('reject')]
        if _n_sig:
            A(f'**确认结果**：{len(_n_sig)}/{len(comps)} 个冻结主比较在 Holm 校正后于 test split 上成立（{sorted(_n_sig)}）——这是预注册规则下的确认性证据。')
        else:
            A('**确认结果**：没有冻结主比较在 Holm 校正后于 test split 上成立。按预注册的决策规则，相应主张**不获确认**；探索性面板的正向读数保持「探索性」标签，不得写成结论。')
        A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A(f'| `{TEST_OUTDIR}/bylen.json` | P5T：冻结臂在 test split 的干净 by-length 评估 |')
    A(f'| `{TEST_OUTDIR}/distractor.json` | P5T：test split 上的 uniform 腐坏探针（描述性） |')
    A('| `analysis_v12/stats.json` | v10 扩展探针统计 + 冻结确认性统计 |')
    A('| `analysis_v12/test_stats.json` | 冻结主比较 + Holm 校正 |')
    A('| `confirmation_frozen.json` | 预注册冻结清单（仓库内，含日期与决策规则） |')
    A('| `v12_supp.py` | 本驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v12 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V12_NO_PUSH'):
        print(f'[git] push skipped (V12_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V12_NO_SHUTDOWN'):
        print('[v12] shutdown suppressed (V12_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v12] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v12] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state["booked_seconds"] / 3600:.2f} h)')
    git_push('v12: confirmation driver (P5C selector controls to n=6 + P5MP extended probe + P5S2/P5S3 matched scale arms + P5T frozen final test)')
    import par
    _ctx = par.resolve_ctx(label='v12')
    all_ok = True
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v12] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            break
        if not V.cuda_healthy():
            print(f'[v12] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v12 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except SystemExit as _se:
            # a REFUSE (e.g. freeze-manifest gate) is a controlled stop, not a crash
            print(f'[v12] phase {pname} refused: {_se}')
            all_ok = False
            break
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v12: phase {pname} results')
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v12: stats + REPORT_v12.md (analysis_v12)')
    guard.report()
    print('\n[v12] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    import copy
    import shutil
    print('[smoke] 1) tiny frozen-confirmation roundtrip on a synthetic local test split')
    L.set_seed(0)
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    payload = dict(V10.P3MT_PAYLOAD, outdir='results_smoke_v12', seeds=[0], steps=40, eval_lens=[512, 1024])
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope', 'csa_fixed_rope_allblocks']), guard=None, label='v12 smoke', par=False)
    # a smoke freeze manifest: same schema, tiny arms/seeds, current code stamp
    _smoke_manifest = {'frozen_at': time.strftime('%Y-%m-%d'), 'code_semantics_expected': L.CODE_SEMANTICS, 'arms': ['csa_fixed_rope', 'csa_fixed_rope_allblocks'], 'seeds': [0], 'primary_comparisons': [{'name': 'smoke: allblocks_scratch - learned @ L1024', 'a': 'csa_fixed_rope_allblocks', 'b': 'csa_fixed_rope', 'endpoint': 'by_len.ppl@1024'}], 'statistics': {'test': 'exact two-sided sign-flip', 'n_seeds': 1, 'p_floor': 2.0, 'family': 'primary_comparisons', 'correction': 'holm step-down', 'decision_rule': 'smoke only'}, 'eval': {'by_length_ppl': [512, 1024], 'distractor_probe': {'mode': 'uniform', 'rhos': [0.0, 0.5], 'eval_lens': [1024], 'target': 512, 'n_seq': 4}}}
    _mpath = 'results_smoke_v12/frozen.json'
    L.atomic_write_json(_mpath, _smoke_manifest, indent=2)
    # smoke test data: rows from the (tiny local dump or HF) validation split,
    # standing in for the test split so the smoke never depends on wiki.test.raw
    _, val_ids, _, _, _ = L.load_wikitext(1024, 4000000)
    cfg = dict(FINALTEST_CFG, manifest=_mpath, outdir='results_smoke_v12', ckpt_dir='results_smoke_v12/ckpt', mech_summary='results_smoke_v12/summary.json', bylen_summary='results_smoke_v12/bylen.json', probe_summary='results_smoke_v12/distractor.json', n_seq=4, chunk=2, stats_out='results_smoke_v12/test_stats.json', test_ids=np.asarray(val_ids[:4, :1025], dtype=np.int64))
    stats = run_finaltest(cfg)
    got = {r['variant'] for r in json.load(open('results_smoke_v12/bylen.json', encoding='utf-8')).values() if isinstance(r, dict)}
    assert {'csa_fixed_rope', 'csa_fixed_rope_allblocks'} <= got, f'finaltest bylen cells missing: {got}'
    _probe = {}
    if os.path.exists('results_smoke_v12/distractor.json'):
        _probe = json.load(open('results_smoke_v12/distractor.json', encoding='utf-8'))
    _pgot = {(r.get('mode', 'uniform'), r['arm'], float(r['rho'])) for r in _probe.values() if isinstance(r, dict)}
    _pwant = {('uniform', 'learned', 0.0), ('uniform', 'learned', 0.5), ('uniform', 'dense', 0.0), ('uniform', 'dense', 0.5)}
    _pwant = {w for w in _pwant if w[1] != 'dense'}  # full_rope arm is not in the smoke manifest
    assert _pwant <= _pgot, f'finaltest probe cells missing: {_pwant - _pgot}'
    assert 'comparisons' in stats and stats['comparisons'], 'finaltest stats produced no comparison record'
    print(f'  finaltest roundtrip OK ({len(got)} bylen arms, {len(_probe)} probe cells, {len(stats["comparisons"])} frozen comparison(s))')
    print('[smoke] 2) report builder on the smoke artifacts')
    build_report(out='results_smoke_v12/REPORT_v12.md', stats_p='results_smoke_v12/stats.json')
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    print('\n[smoke] PASSED')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'analysis', 'report', 'phase', 'finaltest')
    if mode in ('-h', '--help', 'help'):
        print(__doc__)
        raise SystemExit(0)
    if mode not in _MODES:
        print(f'[v12] unknown mode {mode!r}; expected one of {_MODES} (no argument means "full").  Refusing to start a run.')
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f'[v12] mode "phase" needs a phase name, e.g. `python v12_supp.py phase P5C`.  Available: {[p[0] for p in PHASES]}')
        raise SystemExit(2)
    print(f'[v12] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v12_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v12: phase {sys.argv[2]} results')
        elif mode == 'finaltest':
            run_finaltest(dict(FINALTEST_CFG))
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v12: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
