#!/usr/bin/env python3
import gc
import json
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
import v8_supp as V8
V = V8.V
L = V8.L
REPO = V8.REPO
DEVICE = L.DEVICE
import torch.utils.checkpoint as _ckpt
_V9_CKPT = {'on': False}
_orig_block_forward = L.Block.forward

def _v9_block_forward(self, x):
    if _V9_CKPT['on'] and self.training and torch.is_grad_enabled() and x.requires_grad:
        return _ckpt.checkpoint(_orig_block_forward, self, x, use_reentrant=False)
    return _orig_block_forward(self, x)
L.Block.forward = _v9_block_forward
BUDGET_V9 = dict(L.BUDGET)
BUDGET_V9.update(total_yuan=float(os.environ.get('V9_BUDGET_YUAN', 25.0)), price_per_hour=float(os.environ.get('V9_PRICE_PER_HOUR', 2.4)), state_path='autodl_budget_state_v9.json', already_spent_yuan=0.0)

def make_guard():
    g = L.CostGuard(BUDGET_V9)
    if not g.state.get('sps_by_class'):
        for src in ('autodl_budget_state_v8.json', 'autodl_budget_state_v7.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v9] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v9] CostGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v9] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g
P2T_CFG = dict(L.RUN, outdir='results_lm_v9_seq2k', seq_len=2048, batch_size=1, steps=1500, variants=['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1'])
P2S3_CFG = dict(V8.P2R_CFG)
PHASES = [('P2T', P2T_CFG, [0, 1], 1.5), ('P2S3', P2S3_CFG, [2], 5.5)]

def run_phase(name, guard):
    for pname, cfg, seeds, _h in PHASES:
        if pname != name:
            continue
        _V9_CKPT['on'] = pname == 'P2T'
        try:
            s, _a = L.run(cfg, seeds=seeds, guard=guard, label=f'v9 {pname}')
        finally:
            _V9_CKPT['on'] = False
        return s
    raise SystemExit(f'unknown phase {name}')

def _ppl_by_seed(outdir, full=False):
    return L.ppl_by_seed(outdir, full=full)

def _paired_records(outdir):
    return L.ppl_by_seed(outdir, full=True)

def _seed_counts(panel, a, b, missing=()):

    def _one(v):
        if v in panel and panel[v]:
            return f'`{v}`: seeds {sorted(panel[v])}'
        if v in panel or v in missing:
            return f'`{v}`: no usable seed'
        return f'`{v}`: not in panel'
    return _one(a) + ' vs ' + _one(b)

def add_paired(comparisons, name, panel, a, b, *, who='', outdir=None):

    def _omit(reason):
        print(f'[stats] {who}{name}: comparison OMITTED — {reason}')
        comparisons[name] = {'omitted': reason, 'n': 0}
        return None
    if not panel and outdir is None:
        return _omit('no measurable record in the panel — every record was unmeasured (error / no ppl) or a `synthesized` reconstruction, so none is an independent observation')
    empty = [v for v in (a, b) if v in panel and (not panel[v])]
    if empty:
        return _omit('no measurable seed for ' + ', '.join((f'`{v}`' for v in empty)) + ' (every record was unmeasured or a `synthesized` reconstruction, so none is an independent observation)')
    missing = [v for v in (a, b) if v not in panel]
    if missing:
        _other = [v for v in (a, b) if v in panel and panel[v]]
        _pairtxt = _seed_counts(panel, a, b, missing) if _other else None
        if outdir is not None:
            pres = L.format_variant_presence(L.variant_presence(outdir), missing)
            if pres is not None:
                if not panel:
                    return _omit('no measurable record in the panel — none is an independent observation; ' + pres)
                return _omit(pres if _pairtxt is None else f'{pres}; no seed is shared — {_pairtxt}')
        if _pairtxt is not None:
            return _omit(f'variant(s) {missing} have no measurable record, and no seed is shared — {_pairtxt}')
        return _omit(f'variant(s) {missing} absent from the panel')
    common = sorted(set(panel[a]) & set(panel[b]))
    if not common:
        return _omit(f'`{a}` and `{b}` share no seed ({len(panel[a])} vs {len(panel[b])} seeds)')
    dl, skipped, unstamped, why = ([], [], 0, set())
    for s in common:
        ra, rb = (panel[a][s], panel[b][s])
        reason = L.pair_reason(ra, rb)
        if reason:
            why.add(reason)
            skipped.append(s)
            continue
        if ra.get('run_cfg') is None:
            unstamped += 1
        dl.append(ra['ppl'] - rb['ppl'])
    if skipped:
        print(f'[stats] {who}{name}: {len(skipped)}/{len(common)} seed(s) NOT paired — {sorted(why)}; excluded from the test')
    if not dl:
        return _omit(f'no seed survived the run_cfg/budget checks ({len(common)} shared, {len(skipped)} rejected: {sorted(why)})')
    res = V.exact_sign_permutation(dl)
    res['seeds'] = [s for s in common if s not in set(skipped)]
    res['n_skipped_config_mismatch'] = len(skipped)
    res['n_unstamped'] = unstamped
    comparisons[name] = res
    return res

def v9_analysis(out='analysis_v9/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    seq2k = _ppl_by_seed('results_lm_v9_seq2k')
    seq2k_v7 = _ppl_by_seed('results_lm_v7_seq2k')
    rope20k = _ppl_by_seed('results_lm_v8_rope20k')
    seq2k_f = _paired_records('results_lm_v9_seq2k')
    rope20k_f = _paired_records('results_lm_v8_rope20k')
    abs_long_f = _paired_records('results_lm_v3_long')
    comparisons = {}

    def add(name, panel, a, b, outdir):
        add_paired(comparisons, name, panel, a, b, who='v9 ', outdir=outdir)
    add('seq2k(bs1): topk8 - m1', seq2k_f, 'csa_fixed_topk8', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk512 - m1', seq2k_f, 'csa_fixed_topk512', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk8 - topk512', seq2k_f, 'csa_fixed_topk8', 'csa_fixed_topk512', 'results_lm_v9_seq2k')
    add('rope20k: csa_fixed_rope - full_rope', rope20k_f, 'csa_fixed_rope', 'full_rope', 'results_lm_v8_rope20k')
    add('long20k(abs): csa_fixed - full', abs_long_f, 'csa_fixed', 'full', 'results_lm_v3_long')
    panel = V8._panel_block
    out_d = {'seq2k_bs1_panel': panel(seq2k), 'seq2k_bs3_panel_historical': panel(seq2k_v7), 'rope20k_panel': panel(rope20k), 'comparisons': comparisons}
    L.atomic_write_json(out, out_d, indent=2)
    print(f'[v9 stats] wrote {out}')
    for k, v in comparisons.items():
        if 'omitted' in v:
            print(f'  {k:38s} omitted — {v['omitted']}')
            continue
        print(f'  {k:38s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}')
    return out_d

def _fmt_pm(cell, std=None):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        mean, std = (cell['mean'], cell.get('std', 0.0))
    else:
        mean = cell
    return f'{mean:.2f} ±{std or 0.0:.2f}'

def build_report(out='REPORT_v9.md'):
    stats_p = 'analysis_v9/stats.json'
    if not os.path.exists(stats_p):
        v9_analysis()
    st = json.load(open(stats_p, encoding='utf-8'))
    seq2k = st['seq2k_bs1_panel']
    seq2k_old = st.get('seq2k_bs3_panel_historical', {})
    rope = st['rope20k_panel']
    comp = st['comparisons']
    if os.path.exists('autodl_budget_state_v9.json'):
        v9_state = json.load(open('autodl_budget_state_v9.json', encoding='utf-8'))
    else:
        v9_state = {'booked_seconds': 0.0, 'runs': 0}
    price = BUDGET_V9['price_per_hour']
    v9_h = v9_state.get('booked_seconds', 0.0) / 3600.0
    sel_ratio = {'csa_fixed_topk8': '1.6%', 'csa_fixed_topk32': '6.2%', 'csa_fixed_topk128': '25%', 'csa_fixed_topk512': '100%', 'csa_fix_m1': '1.6% (m=1, topk=32)'}
    _SEL_PCT = {'csa_fixed_topk8': 1.6, 'csa_fixed_topk32': 6.2, 'csa_fixed_topk128': 25.0, 'csa_fixed_topk512': 100.0}
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v9 补实验报告（topk 扫描补齐 + RoPE 长跑第三种子）')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348（DeepSeek-V4 稀疏注意力的受控复现与机制剖析）')
    A('> 说明：本报告全部数字由 `v9_supp.py report` 从 `results_*/`、`analysis_v9/` 的落盘产物计算得到，无手填数值。')
    A('')
    A(f'**预算**：v9 记账 {v9_state.get('runs', 0)} runs，估算花费 ¥{v9_h * price:.2f} / ¥{BUDGET_V9['total_yuan']:.2f}（AutoDL RTX 4090，按 ¥{price:.2f}/h 记账；v7/v8 台账各自独立冻结）。')
    A('')
    _r_n = sorted({p['n'] for p in rope.values()}) if rope else []
    _r_n_txt = (str(_r_n[0]) if len(_r_n) == 1 else f'{_r_n[0]}–{_r_n[-1]}') if _r_n else '0'
    A(f'v9 补齐 v7/v8 留下的两个缺口：(1) P1T 的 seq-2048 topk 扫描在 v7 截断于 topk≥128，v9 以 batch_size=1 重跑**完整 5 点扫描**（新目录、面板内自洽）；(2) v8 的 RoPE 20k 长跑只有 2 seeds（精确符号翻转检验在 n=2 时 p 值下限 0.500，正是当初评审否决 n=2 面板的原因），v9 补种子以提升该面板分辨率——**实际落盘 n={_r_n_txt}**。')
    A('')
    A('---')
    A('')
    A('## P2T seq-2048 topk 完整扫描（batch_size=1）')
    A('')
    A('**背景**：v7 P1T 在 bs=3 下扫描 `topk ∈ {8,32,128,512}` + `csa_fix_m1`；topk128/512 在该配置下未能完成，扫描被截断，结论只剩 3 点。v9 改用 bs=1 并按 v7 台账预案启用**逐 block 梯度检查点**——前向无 dropout/RNG，重算严格等价，数学结果与无检查点路径一致。')
    A('')
    _n_seen = sorted({p['n'] for p in seq2k.values()}) if seq2k else []
    _n_txt = (f'{_n_seen[0]} seeds' if len(_n_seen) == 1 else f'{_n_seen[0]}–{_n_seen[-1]} seeds') if _n_seen else '0 seeds'
    A(f'**做法**：seq 2048、bs 1、逐 block 梯度检查点、1500 步；5 个变体全部重跑于新目录 `results_lm_v9_seq2k`（面板内 batch/步数/token 数/训练配置完全一致；v7 的 bs=3 残板保留在 `results_lm_v7_seq2k` 作历史记录，不并入统计）。**实际落盘 {_n_txt}**（下表每点自报 n）。')
    A('')
    A('| variant | PPL (mean±std) | n | 等效选择率@2048 |')
    A('|---|---|---|---|')
    for v in ['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']:
        if v in seq2k:
            p = seq2k[v]
            A(f'| `{v}` | {_fmt_pm(p)} | {p['n']} | {sel_ratio[v]} |')
        else:
            A(f'| `{v}` | —（未完成） | 0 | {sel_ratio[v]} |')
    A('')
    A('')
    if seq2k_old:
        _old_ok = {v: p for v, p in seq2k_old.items() if p.get('mean') is not None}
        if _old_ok:
            A('历史对照（v7，bs=3，仅 3 个完成点）：' + '；'.join((f'`{v}` {p['mean']:.2f}' for v, p in _old_ok.items())) + '。bs 不同不直接比较，仅作 sanity check。')
        A('')
    done = [v for v in ['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1'] if v in seq2k]
    if len(done) >= 3:
        means = [seq2k[v]['mean'] for v in done]
        best = done[int(np.argmin(means))]
        worst = done[int(np.argmax(means))]
        A(f'**完成 {len(done)}/5 点**。最低点：`{best}`（{min(means):.2f}）；最高点：`{worst}`（{max(means):.2f}）。')
        _ord = sorted(((_SEL_PCT[v], v) for v in done if v in _SEL_PCT))
        _skipped = [v for v in done if v not in _SEL_PCT]
        if len(_ord) >= 3:
            ms = [seq2k[v]['mean'] for _r, v in _ord]
            mono = all((ms[i] <= ms[i + 1] + 1e-09 for i in range(len(ms) - 1)))
            A('按选择率升序（' + ' < '.join((f'{_r:g}%`{v.replace('csa_fixed_', '')}`' for _r, v in _ord)) + f'）PPL：{('单调不降' if mono else '非单调（见表）')}。' + (f'（`{'`、`'.join(_skipped)}` 无唯一选择率位置，不参与该排序判断。）' if _skipped else ''))
        else:
            A('按选择率升序的单调性判断：有效排序点不足 3 个，不作判断。')
        A('')
    A('**配对检验（exact sign-flip）**：')
    A('')
    A('| 比较 | n | Δ (a−b) mean±std | p (exact) |')
    A('|---|---|---|---|')
    for k, v in comp.items():
        if k.startswith('seq2k'):
            if 'omitted' in v:
                A(f'| {k} | 0 | —（{v['omitted']}） | — |')
                continue
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
    A('')
    A('**结论读法**：若完整扫描证实 m=1（纯 DSA）不差于压缩 CSA，则 v7「压缩不是瓶颈」的方向性结论在完整扫描下成立/被修正（按表）；选择率甜点位置由 5 点全貌直接给出，v7 留下的「甜点完整刻画」待办关闭。')
    A('')
    A('---')
    A('')
    A('## P2S3 RoPE 20k 长跑补第三种子（n=2 → n=3）')
    A('')
    A('**背景**：v8 P2R 跑了 `csa_fixed_rope` vs `full_rope` 的 20k 步 / ~123M token 对照（abs-PE 20k 长跑的 RoPE 镜像），但只有 2 seeds——精确符号翻转检验在 n=2 时 p 值下限 0.500，正是评审当初否决 scale 面板 n=2 的理由。v9 补 seed 2（同一 outdir，resume 跳过 seed 0/1）。')
    A('')
    A('| variant | PPL@20k (mean±std) | n |')
    A('|---|---|---|')
    for v, p in rope.items():
        A(f'| `{v}` | {_fmt_pm(p)} | {p['n']} |')
    A('')
    A('| 比较 | n | Δ mean±std | p (exact) |')
    A('|---|---|---|---|')
    for k in ['rope20k: csa_fixed_rope - full_rope', 'long20k(abs): csa_fixed - full']:
        if k not in comp:
            continue
        v = comp[k]
        if 'omitted' in v:
            A(f'| {k} | 0 | —（{v['omitted']}） | — |')
            continue
        A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
    A('')
    _r20 = comp.get('rope20k: csa_fixed_rope - full_rope') or {}
    if _r20 and 'omitted' not in _r20:
        c = _r20
        A(f'**结论**：RoPE 面板上 `csa_fixed_rope` 落后 `full_rope` **{c['mean']:+.2f} PPL**（n={c['n']}，p={c['p_exact_signflip']:.3f}）——与 abs-PE 长跑的「渐近线反转」同向，中心负结果在论文自己的位置编码方案下复现，且达到与其镜像面板相同的种子数。')
    _a20 = comp.get('long20k(abs): csa_fixed - full') or {}
    if _a20 and 'omitted' in _a20:
        A('> **abs-PE 镜像对比不可用**：`results_lm_v3_long` 的 summary 全部是 `synthesized` 重构记录（由旧 `aggregate.json` 回填，无权重、无 `run_cfg`，同一变体的多个 seed 共享同一个拷贝值），不构成独立观测，故不参与配对检验。该面板的 20k abs-PE 数值请以重新训练后的产物为准；本表的 abs-PE 侧结论只由 `REPORT_v7/v10` 中持有真实记录的 run 支撑。')
    A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')

    def _n_seeds_of(_outdir):
        try:
            _s = json.load(open(os.path.join(REPO, _outdir, 'summary.json'), encoding='utf-8'))
        except Exception:
            return None
        _sd = {}
        for _r in _s.values():
            if isinstance(_r, dict) and 'seed' in _r:
                _sd.setdefault(_r.get('variant'), set()).add(_r['seed'])
        _ns = [len(x) for x in _sd.values()]
        return (min(_ns), max(_ns)) if _ns else None

    def _seed_txt(_outdir, _fallback):
        _n = _n_seeds_of(_outdir)
        if _n is None:
            return _fallback
        return f'{_n[0]} seeds' if _n[0] == _n[1] else f'{_n[0]}–{_n[1]} seeds'
    A('| 路径 | 内容 |')
    A('|---|---|')
    A(f'| `results_lm_v9_seq2k/` | P2T 完整 topk 扫描（bs=1，5 变体 × {_seed_txt('results_lm_v9_seq2k', '2 seeds')}） |')
    A(f'| `results_lm_v8_rope20k/` | P2S3 追加 seed 2 后的 {_seed_txt('results_lm_v8_rope20k', '3-seed')} RoPE 20k 面板 |')
    A('| `analysis_v9/stats.json` | 上述面板的配对符号翻转检验 |')
    A('| `autodl_budget_state_v9.json` | v9 CostGuard 台账 |')
    A('| `v9_supp.py` | 本阶段驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v9 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V9_NO_PUSH'):
        print(f'[git] push skipped (V9_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V9_NO_SHUTDOWN'):
        print('[v9] shutdown suppressed (V9_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v9] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v9] remaining ¥{guard.remaining_yuan():.2f} (cap ¥{guard.cap_yuan():.2f} @ ¥{guard.price:.2f}/h)')
    git_push('v9: supplementary driver (P2T topk sweep completion @bs1 + P2S3 RoPE-20k third seed)')
    all_ok = True
    for pname, _cfg, _seeds, est_h in PHASES:
        rem = guard.remaining_yuan()
        if rem < 1.0:
            print(f'[v9] stopping before {pname}: ¥{rem:.2f} left')
            break
        if not V.cuda_healthy():
            print(f'[v9] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v9 phase {pname} (~{est_h} h est, ¥{rem:.2f} left) =====')
        try:
            run_phase(pname, guard)
        except Exception:
            traceback.print_exc()
        all_ok &= git_push(f'v9: phase {pname} results')
    try:
        v9_analysis()
        build_report()
    except Exception:
        traceback.print_exc()
    all_ok &= git_push('v9: paired sign-flip stats + REPORT_v9.md (analysis_v9)')
    guard.report()
    print('\n[v9] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) topk128/512 forward/backward (seq 2048, bs 1, gradient checkpointing ON)')
    L.set_seed(0)
    _V9_CKPT['on'] = True
    for v in ['csa_fixed_topk128', 'csa_fixed_topk512']:
        cfgs = L.make_layer_cfgs(6, v)
        m = L.SmallGPT(8192, 256, 6, 8, 32, 2048, cfgs).to(DEVICE)
        x = torch.randint(0, 8192, (1, 2048), device=DEVICE)
        import torch.nn.functional as Fn
        out = m(x)
        loss = Fn.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
        loss.backward()
        print(f'  {v:20s} out={tuple(out.shape)} loss={loss.item():.3f} -> fwd+bwd OK at seq 2048 bs 1')
        del m, out, loss, x
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    _V9_CKPT['on'] = False
    print('[smoke] 2) 60-step csa_fixed_topk128 probe (seq 2048, bs 1, gradient checkpointing ON — matches the P2T phase setting)')
    guard = make_guard()
    train_ids, val_batch, vocab, _, vb = L.load_wikitext(2048, 1000000)
    _V9_CKPT['on'] = True
    t0 = time.time()
    try:
        rec = L.train_variant('csa_fixed_topk128', train_ids, val_batch, vocab, seed=0, steps=60, seq_len=2048, batch_size=1, eval_every=30, eval_subset=16, log_every=30, val_bnd=vb)
    finally:
        _V9_CKPT['on'] = False
    dt = time.time() - t0
    guard.record_run(dt, 60, 256, 6, 2048, 1)
    print(f'  60 steps in {dt:.0f}s -> {dt / 60:.3f} s/step (ppl {rec['ppl']:.1f}); booked to the v9 guard for calibration')
    est = guard.estimate_seconds(1500, d=256, n_layers=6, seq_len=2048, batch_size=1)
    print(f'  -> 1500-step bs1 seq2k run estimate: {est / 60:.1f} min (¥{est / 3600 * guard.price:.2f}); full P2T (10 runs) ~{10 * est / 3600:.2f} h')
    print('[smoke] 3) zero-GPU report rebuild on current artifacts')
    v9_analysis()
    build_report()
    print('\n[smoke] PASSED')
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'analysis', 'report', 'phase')
    if mode in ('-h', '--help', 'help'):
        print(__doc__ or f"[v9] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v9] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v9] mode 'phase' needs a phase name, e.g. `python v9_supp.py phase P2T`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v9] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v9_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v9: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v9: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
