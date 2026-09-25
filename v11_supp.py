#!/usr/bin/env python3
import glob
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
import v10_supp as V10
V9 = V10.V9
V8 = V10.V8
V = V10.V
L = V10.L
REPO = V10.REPO
DEVICE = L.DEVICE
BUDGET_V11 = dict(L.BUDGET)
BUDGET_V11.update(total_yuan=float(os.environ.get('V11_BUDGET_YUAN', 30.0)), price_per_hour=float(os.environ.get('V11_PRICE_PER_HOUR', 2.4)), state_path='autodl_budget_state_v11.json', already_spent_yuan=0.0)

def make_guard():
    g = L.CostGuard(BUDGET_V11)
    if not g.state.get('sps_by_class'):
        for src in ('autodl_budget_state_v10.json', 'autodl_budget_state_v9.json', 'autodl_budget_state_v8.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v11] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v11] CostGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v11] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g

def P4MT_CFG(seeds=(3, 4, 5)):
    return dict(V10.P3MT_PAYLOAD, seeds=list(seeds))

def run_p4mt(payload=None, guard=None, label='v11 P4MT'):
    if payload is None:
        payload = P4MT_CFG()
    return V.run_lenphase(payload, guard=guard, label=label)

def P4MP_CFG(seeds=(3, 4, 5)):
    return dict(V10.PROBE, seeds=list(seeds))
P4SS_CFG = dict(V10.P3SS_CFG)
P4SL_CFG = dict(V10.P3SL_CFG)
P4F_CFG = dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full'])
PHASES = [('P4MT', 'p4mt', P4MT_CFG, None, 1.0), ('P4MP', 'probe', P4MP_CFG, None, 0.3), ('P4SS', 'run', P4SS_CFG, [2, 3], 1.0), ('P4SL', 'run', P4SL_CFG, [2, 3], 1.3), ('P4F', 'run', P4F_CFG, [2, 3], 1.0)]

def run_phase(name, guard):
    for pname, kind, payload, seeds, _h in PHASES:
        if pname != name:
            continue
        cfg = payload() if callable(payload) else payload
        if kind == 'p4mt':
            return run_p4mt(cfg, guard=guard, label=f'v11 {pname}')
        if kind == 'probe':
            return V10.run_probe(cfg, guard=guard, label=f'v11 {pname}')
        s, _a = L.run(cfg, seeds=seeds, guard=guard, label=f'v11 {pname}')
        return s
    raise SystemExit(f'unknown phase {name}')

def v11_analysis(out='analysis_v11/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    probe = {'cells': [], 'contrasts': {}}
    sp = V10.PROBE['summary']
    if os.path.exists(sp):
        raw = json.load(open(sp, encoding='utf-8'))
        cells, _dup_cells, _bad_cells = ({}, [], [])
        for _k, r in raw.items():
            if not isinstance(r, dict):
                _bad_cells.append((_k, 'not a dict'))
                continue
            if r.get('variant') is None or r.get('seed') is None:
                _bad_cells.append((_k, 'no variant/seed'))
                continue
            if not L.ppl_is_usable(r.get('ppl_mean')):
                _bad_cells.append((_k, 'ppl_mean is not a usable measurement'))
                continue
            if r.get('arm') is None or r.get('eval_len') is None or r.get('rho') is None:
                _bad_cells.append((_k, 'no arm/eval_len/rho'))
                continue
            if r.get('probe_params') is None:
                _bad_cells.append((_k, 'no probe_params'))
                continue
            pk = (r['variant'], r.get('arm'), r.get('eval_len'), r.get('rho'), json.dumps(r.get('probe_params'), sort_keys=True), str(r.get('_code')))
            grp = cells.setdefault(pk, {})
            if r['seed'] in grp:
                _dup_cells.append((_k, pk[:4], r['seed']))
                continue
            grp[r['seed']] = r
        if _bad_cells:
            print(f'[v11 stats] {len(_bad_cells)} record(s) in {sp} carry no variant/seed/ppl_mean and are skipped (they are error or truncation stubs, not measurements): ' + ', '.join((f'{k}({why})' for k, why in _bad_cells[:4])))
        if _dup_cells:
            print(f'[v11 stats] {len(_dup_cells)} probe record(s) share a (cell, probe_params, seed) identity — their PPL is ambiguous and they are dropped from the pooling: ' + ', '.join((f'{k}' for k, _c, _s in _dup_cells[:4])))
        probe_cells = []
        for (v, arm, Ln, rho, _pp, _cd), by_seed in sorted(cells.items(), key=lambda kv: kv[0]):
            means = {s: r['ppl_mean'] for s, r in by_seed.items()}
            assert means, (v, arm, Ln, rho)
            probe_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'seeds': sorted(means), 'ppl_by_seed': means, 'probe_params': json.loads(_pp), 'mean': float(np.mean(list(means.values()))), 'std': float(np.std(list(means.values()), ddof=1)) if len(means) > 1 else 0.0, 'n': len(means)})
        contrasts = {}

        def cell_mean(v, arm, Ln, rho):
            hits = [c for c in probe_cells if (c['variant'], c['arm'], c['eval_len'], c['rho']) == (v, arm, Ln, rho)]
            if not hits:
                return {}
            if len(hits) > 1:
                print(f'[v11 stats] ({v}/{arm}/L{Ln}/r{rho}) holds {len(hits)} probe parameterisations — no unambiguous cell, so the contrast is omitted rather than pooled across them')
                return {}
            return hits[0]
        for Ln in V10.PROBE['eval_lens']:
            for rho in V10.PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho)
                for arm, tag in (('dense', 'full_rope(dense)'), ('randidx', 'csa+randidx'), ('allblocks', 'csa+allblocks')):
                    other = cell_mean('full_rope', 'dense', Ln, rho) if arm == 'dense' else cell_mean('csa_fixed_rope', arm, Ln, rho)
                    if not learned or not other:
                        continue
                    if learned.get('probe_params') != other.get('probe_params'):
                        print(f'[v11 stats] L{Ln} r{rho} {tag}: the two arms were probed under DIFFERENT probe_params — pairing them would difference two different measurements, so the contrast is omitted')
                        continue
                    common = sorted(set(learned['ppl_by_seed']) & set(other['ppl_by_seed']))
                    if len(common) >= 2:
                        dl = [other['ppl_by_seed'][s] - learned['ppl_by_seed'][s] for s in common]
                        contrasts[f'L{Ln} r{rho}: {tag} - csa+learned'] = V.exact_sign_permutation(dl)
                        contrasts[f'L{Ln} r{rho}: {tag} - csa+learned']['seeds'] = common
        probe = {'cells': probe_cells, 'contrasts': contrasts}
        _n_cell_max = max((c['n'] for c in probe_cells), default=0)
        _n_pair_max = max((v['n'] for v in contrasts.values()), default=0)
        _n_pair_min = min((v['n'] for v in contrasts.values()), default=0)
        _n_pair_cells = max([v['n'] * 2 for v in contrasts.values()], default=0)
        probe['n_seeds'] = {'cell_max': _n_cell_max, 'contrast_paired_max': _n_pair_max, 'contrast_paired_min': _n_pair_min}
        if _n_pair_cells and _n_pair_min < _n_cell_max:
            lowered = sorted((k for k, v in contrasts.items() if v['n'] < _n_cell_max))
            print(f'[v11 stats] NOTE: the fullest probe cell reaches n={_n_cell_max} but the WEAKEST contrast pairs only {_n_pair_min} seeds — the arms are not seed-complete with each other, so every p value below is computed at its own PAIRED count, i.e. the worst floor for this panel is {2.0 / 2 ** _n_pair_min:.3f}, not {2.0 / 2 ** _n_cell_max:.3f}. {len(lowered)}/{len(contrasts)} contrast(s) are limited by this: {lowered[:3]}')
    xo = V10._scale_crossovers()
    pts, _no_tok, _recon = ([], [], [])
    for _lab, v in xo.items():
        if v.get('status') != 'ok' or not v.get('mean_crossed'):
            continue
        _fl = v.get('per_seed_synth', {})
        if any((_fl.get(str(s)) for s in v.get('seeds', []))):
            _recon.append(_lab)
            continue
        if 'mean_crossover_tokens' in v:
            pts.append((v['d'], v['mean_crossover_tokens']))
        else:
            _no_tok.append(_lab)
    fit = {}
    for _lab in _no_tok:
        print(f'[v11 stats] crossover fit: dropped {_lab!r} — its panel publishes no tokens/step rate, so its token axis is unknown')
    for _lab in _recon:
        print(f'[v11 stats] crossover fit: dropped {_lab!r} — its trajectory is a RECONSTRUCTION, so the point is not an independent run')
    if len(pts) >= 2:
        xs = np.log10([p[0] for p in pts])
        ys = np.log10([max(p[1], 1.0) for p in pts])
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
        pred = A @ np.array([slope, intercept])
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        fit = {'n_points': len(pts), 'slope': float(slope), 'intercept': float(intercept), 'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'), 'note': 'log10(crossover_tokens) ~ log10(d_model), untruncated crossovers only', 'excluded_no_rate': sorted(_no_tok), 'excluded_reconstructed': sorted(_recon)}
    scale384 = V10._ppl_by_seed('results_lm_v5_scale', full=True)
    scale384_f = V9._paired_records('results_lm_v5_scale')
    panel = V8._panel_block
    comparisons = {}

    def add(name, panel_d, a, b, outdir):
        V9.add_paired(comparisons, name, panel_d, a, b, who='v11 ', outdir=outdir)
    add('scale384: csa_fixed - full', scale384_f, 'csa_fixed', 'full', 'results_lm_v5_scale')
    add('scale384: csa_fixed - full_sw128_matched', scale384_f, 'csa_fixed', 'full_sw128_matched', 'results_lm_v5_scale')
    out_d = {'probe': probe, 'crossover_panels': xo, 'crossover_fit': fit, 'scale384_panel': panel(scale384), 'comparisons': comparisons}
    L.atomic_write_json(out, out_d, indent=2)
    print(f'[v11 stats] wrote {out}')
    for k, v in contrasts_headline(probe).items():
        print(f'  {k:44s} Δ={v['mean']:+7.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}')
    for label, v in xo.items():
        if v.get('status') == 'ok':
            n = len(v['seeds'])
            if not v['mean_crossed']:
                _stps = v['mean_traj']['steps']
                if _stps:
                    print(f'  crossover {label:10s} n={n} mean-traj: CENSORED (> {_stps[-1]:g} steps)')
                else:
                    print(f'  crossover {label:10s} n={n} mean-traj: no eval step shared by every seed — no censored bound')
            elif 'mean_crossover_tokens' in v:
                print(f'  crossover {label:10s} n={n} mean-traj: step {v['mean_crossover_step']:g} ({v['mean_crossover_tokens']:g} tokens)')
            else:
                print(f'  crossover {label:10s} n={n} mean-traj: step {v['mean_crossover_step']:g} (tokens/step unknown)')
    return out_d

def contrasts_headline(probe):
    return {k: v for k, v in probe.get('contrasts', {}).items() if k.startswith('L4096') and ('r0.25' in k or 'r0.5' in k)}

def _fmt_pm(cell, std=None):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        mean, std = (cell['mean'], cell.get('std', 0.0))
    else:
        mean = cell
    return f'{mean:.2f} ±{std or 0.0:.2f}'

def build_report(out='REPORT_v11.md'):
    stats_p = 'analysis_v11/stats.json'
    if not os.path.exists(stats_p):
        v11_analysis()
    st = json.load(open(stats_p, encoding='utf-8'))
    probe = st['probe']
    xo = st['crossover_panels']
    fit = st['crossover_fit']
    scale384 = st['scale384_panel']
    comp = st['comparisons']
    if os.path.exists('autodl_budget_state_v11.json'):
        v11_state = json.load(open('autodl_budget_state_v11.json', encoding='utf-8'))
    else:
        v11_state = {'booked_seconds': 0.0, 'runs': 0}
    price = BUDGET_V11['price_per_hour']
    v11_h = v11_state.get('booked_seconds', 0.0) / 3600.0
    mech_sum = {}
    mp = os.path.join(V10.P3MT_PAYLOAD['outdir'], 'summary.json')
    if os.path.exists(mp):
        mech_sum = json.load(open(mp, encoding='utf-8'))
    probe_cells_ = probe.get('cells', [])
    contr_ = probe.get('contrasts', {})
    n_probe_min = min((c['n'] for c in probe_cells_), default=0)
    n_contr_max = max((v['n'] for v in contr_.values()), default=0)
    _n_seeds = probe.get('n_seeds') or {}
    n_cells_max = max(_n_seeds.get('cell_max', 0), max((c['n'] for c in probe_cells_), default=0))
    _n_pair = sorted({v['n'] for v in contr_.values()})
    n_pair_min = _n_pair[0] if _n_pair else n_contr_max
    _pair_ragged = bool(contr_) and n_pair_min < n_contr_max
    _xo_ok = {k: len(v['seeds']) for k, v in xo.items() if v.get('status') == 'ok'}
    _xo_recon = [k for k in _xo_ok if any((xo[k].get('per_seed_synth', {}).get(str(s)) for s in xo[k].get('seeds', [])))]
    _xo_real = {k: n for k, n in _xo_ok.items() if k not in _xo_recon}
    n_xo = max(_xo_real.values(), default=0)
    n_xo_min = min(_xo_real.values(), default=0)
    _xo_all_max = all((n == n_xo for n in _xo_real.values())) if _xo_real else False
    n384 = max((p['n'] for p in scale384.values()), default=0)
    _n384_set = sorted({p['n'] for p in scale384.values()})
    _n384_full = (scale384.get('full') or {}).get('n')
    n384_ok = _n384_full is not None and _n384_full == n384 and (len(_n384_set) == 1)
    _xo_hit = sum((1 for n in _xo_real.values() if n == n_xo))
    n_tree_min = n_pair_min
    floor_p = 2.0 / 2 ** n_contr_max if n_contr_max else float('nan')
    floor_p_tree = 2.0 / 2 ** n_tree_min if n_tree_min else float('nan')
    probe_ok = n_contr_max >= 6
    probe_tag = f'@ n={n_contr_max}' if n_contr_max else '@ n/a'
    _floor_txt = f'{floor_p:.3f}' if n_tree_min == n_contr_max else f'{floor_p_tree:.3f}（最弱对比 n={n_tree_min}；最全对比 n={n_contr_max} 为 {floor_p:.3f}）'
    _prov_total = 0
    _prov_stamped = 0
    _prov_synth = 0
    _prov_measured = 0
    _prov_panels = []
    _prov_unreadable = []
    for _p in sorted(glob.glob('results_*/summary.json')):
        try:
            _d = json.load(open(_p, encoding='utf-8'))
        except Exception as _pe:
            _prov_unreadable.append((_p.split('/')[0], f'{type(_pe).__name__}: {_pe}'))
            print(f'[provenance] WARNING: {_p} cannot be parsed ({type(_pe).__name__}: {_pe}) — reported as unreadable, NOT silently omitted')
            continue
        _tot = _st = _sy = _me = 0
        for _k, _v in _d.items():
            if not isinstance(_v, dict):
                continue
            _tot += 1
            if _v.get('synthesized'):
                _sy += 1
                continue
            _me += 1
            _rc = _v.get('run_cfg')
            if isinstance(_rc, str) and _rc.endswith(f'_cs{L.CODE_SEMANTICS}'):
                _st += 1
        if _tot:
            _prov_panels.append((_p.split('/')[0], _tot, _me, _st, _sy))
            _prov_total += _tot
            _prov_stamped += _st
            _prov_synth += _sy
            _prov_measured += _me
    _prov_unstamped = _prov_measured - _prov_stamped
    _prov_pct = 100.0 * _prov_stamped / _prov_measured if _prov_measured else 0.0
    lines = []
    A = lines.append
    _xo_txt = (f'n={n_xo}' if _xo_all_max else f'n={n_xo_min}–{n_xo}（{_xo_hit}/{len(_xo_real)} 个已测量面板达 {n_xo}）') if _xo_real else 'n=0'
    _n384_txt = f'n={n384}' if n384_ok else f'n={_n384_set}（`full` 臂 n={_n384_full}）'
    _head_n = f'n={n_contr_max}' if n_tree_min == n_contr_max else f'n={n_tree_min}–{n_contr_max}'
    A(f'# CSA / HCA 受控机制研究 — v11 补实验报告（统计分辨率收尾：主结果 {_head_n}、规模面板 {_xo_txt}）')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348（DeepSeek-V4 稀疏注意力的受控复现与机制剖析）')
    A('> 说明：本报告全部数字由 `v11_supp.py report` 从 `results_*/`、`analysis_v11/` 的落盘产物计算得到，无手填数值。')
    A('')
    A(f'**预算**：v11 记账 {v11_state.get('runs', 0)} runs，估算花费 ¥{v11_h * price:.2f} / ¥{BUDGET_V11['total_yuan']:.2f}（AutoDL RTX 4090，按 ¥{price:.2f}/h 记账；v7–v10 台账各自独立冻结）。')
    A('')
    if probe_ok:
        A(f'v11 不提出新科学问题，只做 v10 留下的三处统计收尾：(1) 主正面结果（远距干扰注入探针）从 n=3 补到 **n={n_contr_max}**——全仓库唯一一个低成本即可让精确符号翻转 p 值下限跨过 0.05 的地方（0.250 → {_floor_txt}）；(**注意**：补满的是该面板的**最强**对比。各臂独立续训，配对取两臂交集，因此最弱对比仍停在 n={n_tree_min}、下限 {floor_p_tree:.3f}，凡引用它的结论不作显著性主张。)；(2) v10 新建的两个交叉定位面板（d=128/4L、d=512/10L）从 n=2 补到 {_xo_txt}——各面板的完成度不同，标题给的是分布而不是上限；(3) d=384 面板的 `full` 臂补到 n={n384}，消除 4 点标度读数中唯一离群点的配对不平衡。')
    else:
        A('v11 计划做三处统计收尾：主正面结果（远距干扰注入探针）补到 n=6、两个交叉定位面板补到 n=4、d=384 面板 `full` 臂补到 n=4。')
        A('')
        A(f'> **⚠ 实际落盘状态（本报告从 disk 实时计算）**：探针对比的最大种子数为 **n={n_contr_max}**（精确符号翻转 p 值下限 {_floor_txt}，**未**跨过 0.05），最小配对 n={n_pair_min}；交叉面板 {_xo_txt}；d=384 面板 {_n384_txt}。**P4MT/P4MP 的 seeds 3–5 尚未产出**，因此本节的 n 仍停留在 v10 的水平，**「n=6」的统计升级未发生**；相关结论须待补跑后方可作显著性主张。')
    if _pair_ragged:
        A('')
        A(f'> **⚠ 探针各臂的种子数不一致**：最全的 cell 达 n={n_cells_max}，但最强的对比也只配上 n={n_contr_max} 对，最弱的只有 n={n_tree_min} 对——`full_rope` 与 `csa_fixed_rope` 是分别续训的，任一臂未补满，跨臂对比就只能用两臂的交集。因此表中每个 p 值的实际下限是 {_floor_txt}，**不是**按 cell 数算出的 {2.0 / 2 ** n_cells_max:.3f}；未配满的对比在下面按「未配满」标注，其 Δ 仅作方向性表述。')
    A('')
    A('---')
    A('')
    A(f'## P4M 干扰注入探针 {probe_tag}（主结果统计升级）')
    A('')
    if not probe_ok:
        A(f'> 计划 n=6；**实际 n={n_contr_max}**（`distractor.json` 仅含 seeds ' + '/'.join((str(s) for s in sorted({s for c in probe_cells_ for s in c['seeds']}))) + '）。下表的 Δ 与 p 值均基于该实际种子数。')
        A('')
    A('**做法**：P4MT 按 v7 P1L / v10 P3MT 原配方（seq 512、3000 步、AdamW lr 3e-4、bs 12）续训 seeds 3/4/5（v10 权重不入库但 seeds 0–2 的探针 cell 已提交在 `distractor.json`，故只需补 3 个种子的权重）；P4MP 对新种子续跑探针（纯 eval，逐 cell 断点续跑）。种子间训练/评测代码路径与 v10 完全一致。')
    A('')
    if mech_sum:
        n_by_v = {}
        for _k, r in mech_sum.items():
            if not isinstance(r, dict) or r.get('variant') is None or r.get('seed') is None:
                continue
            if 'by_len' in r:
                n_by_v.setdefault(r['variant'], set()).add(r['seed'])
        A(f'**双臂 by-length 面板**（n={(min((len(s) for s in n_by_v.values())) if n_by_v else 0)}，ratio = PPL@L / PPL@512，seed 平均）：')
        A('')
        A('| variant | PPL@512 | PPL@2048 | PPL@4096 | ratio@4096 |')
        A('|---|---|---|---|---|')
        for v in V10.P3MT_PAYLOAD['variants']:
            rows = [r for r in mech_sum.values() if r.get('variant') == v and 'by_len' in r]
            if not rows:
                continue

            def _at(Ln, rows=rows):
                cells = L.by_len_cells(rows, Ln)
                ok = [c for c in cells if not c.get('truncated')]
                if ok:
                    return (float(np.mean([c['ppl'] for c in ok])), False)
                return (float(np.mean([c['ppl'] for c in cells])), True) if cells else (float('nan'), False)

            def _fmt_cell(Ln):
                val, tr = _at(Ln)
                return f'{val:.1f}~' if tr else f'{val:.1f}'
            _p512, _512tr = _at(512)
            _p4096, _4096tr = _at(4096)
            if _512tr or _4096tr:
                ratio = '—（含截断 cell，不可比）'
            elif np.isnan(_p512) or np.isnan(_p4096) or (not _p512):
                ratio = '—'
            else:
                ratio = f'×{_p4096 / _p512:.2f}'
            A(f'| `{v}` | {_fmt_cell(512)} | {_fmt_cell(2048)} | {_fmt_cell(4096)} | {ratio} |')
        A('')
        A('> 后缀 `~` 表示该列**只有位置受限（abs-PE）的截断 cell**：它是在比列名更短的 span 上评出来的分数，**不是**长上下文测量值。任何 `~` 行不可用于「外推是否稳健」的结论；RoPE 臂（无位置上限）在同一表里是未被截断的真实长上下文分数，两类不可直接相比。')
    cells = probe.get('cells', [])
    if cells:
        lens = sorted({c['eval_len'] for c in cells})
        rhos = V10.PROBE['rhos']
        arm_order = [('full_rope', 'dense'), ('csa_fixed_rope', 'learned'), ('csa_fixed_rope', 'randidx'), ('csa_fixed_rope', 'allblocks')]
        for Ln in lens:
            n_max = max((c['n'] for c in cells if c['eval_len'] == Ln), default=0)
            A(f'**目标区 PPL（eval_len={Ln}，至多 {n_max} seeds 平均）**：')
            A('')
            A('| 臂 | ' + ' | '.join((f'ρ={r:g}' for r in rhos)) + ' |')
            A('|---|' + '---|' * len(rhos))
            for v, arm in arm_order:
                row = []
                for rho in rhos:
                    m = [c for c in cells if (c['variant'], c['arm'], c['eval_len'], c['rho']) == (v, arm, Ln, rho)]
                    row.append(f'{m[0]['mean']:.2f}' if m else '—')
                A(f'| `{v}`+{arm} | ' + ' | '.join(row) + ' |')
            A('')
    contr = probe.get('contrasts', {})
    if contr:
        A('**配对检验（exact sign-flip，Δ = 对照臂 − 学习选择臂）：**')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) |')
        A('|---|---|---|---|')
        for k, v in contr.items():
            _tag_n = f'{v['n']}' if v['n'] >= n_cells_max else f'{v['n']}（未配满，cell n={n_cells_max}）'
            A(f'| {k} | {_tag_n} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
        A('')
        n6 = [v for v in contr.values() if v['n'] >= 6]
        if n6:
            sig = [v for v in n6 if v['p_exact_signflip'] < 0.05]
            _strongest = max((v['n'] for v in n6))
            A(f'**统计读法**：n>={_strongest} 的格子共 {len(n6)} 个，其中 p<0.05 的 {len(sig)} 个（精确符号翻转在 n={_strongest} 的下限为 {2.0 / 2 ** _strongest:.3f}）——注意这是**这些格子自己**的分辨率，不是整个面板的：本面板最弱的对比仍有 n={n_tree_min}，其下限为 {floor_p_tree:.3f}，**跨不过 0.05**。因此本报告的一切显著性主张只对上面的 n>={_strongest} 行成立，凡引用最弱对比（n={n_tree_min}）的段落一律按效应量 + 方向同向性表述。p≥0.05 的格子同样不作显著性主张。')
            A('')
        else:
            A(f'**统计读法**：本面板实际最大 n={n_contr_max}，精确符号翻转的 p 值下限为 {_floor_txt}，**无法**在该种子数下跨过 0.05。因此下文所有 Δ 一律按效应量 + 方向同向性表述，不作显著性主张；p 值仅作分辨率下限的记录。')
            A('')

    def _rng(tag):
        _lo, _hi, _nonf = L.finite_range(((k, v.get('mean')) for k, v in contr.items() if tag in k))
        if _nonf:
            print(f'[v11 report] WARNING: {len(_nonf)} contrast(s) in {tag!r} carry a non-finite mean and are excluded from the reported range: {sorted(_nonf)[:3]}')
        return None if _lo is None else (_lo, _hi)

    def _fmt(r, nd=1):
        return 'n/a' if not r else f'{r[0]:+.{nd}f}~{r[1]:+.{nd}f}'
    dn, ab, ri = (_rng('dense'), _rng('allblocks'), _rng('randidx'))
    learned_helps = bool(ri) and ri[0] > 0.0
    if contr:
        A(f'**机制结论（区间实时算自上方配对检验，n={n_contr_max}，Δ = 对照臂 − 学习选择臂）：**dense−learned Δ {_fmt(dn)} PPL，allblocks−learned Δ {_fmt(ab)} PPL，randidx−learned Δ {_fmt(ri)} PPL——' + ('randidx 全线更差：学习到的检索对抗噪性有**独立贡献**。' if learned_helps else 'randidx 至少在一格与 learned 相当或更优：抗噪性主要来自稀疏归纳偏置本身，「学习检索滤噪」的强版本不成立。') + (f'（统计分辨率：本段引用的对比中，最全的为 n={n_contr_max}、最弱的仅 n={n_tree_min}，精确符号翻转 p 值下限因此是 {_floor_txt}，**未**跨过 0.05——上述读法只作方向性表述。）' if not probe_ok else '') + (f'（统计分辨率：本段引用的是 n={n_tree_min}–{n_contr_max} 的对比，最弱者下限 {floor_p_tree:.3f}；`dense − learned` 这一格是n={n_tree_min}，其分辨率仍不足以支撑显著性主张。）' if probe_ok and _pair_ragged else '') + '（另注意：v11 之前的探针单元所用的验证文本、indexer 与 RoPE 检查点已不适用于当前代码（见 docs/design_notes.md），本段结论须待 P4MT/P4MP 重跑后方可解读。）')
        A('')
    else:
        A('**机制结论**：`analysis_v11/stats.json` 的 `probe.contrasts` 为空，因此 dense/allblocks/randidx 相对 learned 的 Δ 区间**无数据**，本轮不给出机制读法。这不是「无差异」——是**未测量**。')
        A('')
    A('---')
    A('')
    A(f'## P4S/P4F 交叉点标度读数 {_xo_txt}（d=384 full 臂对齐）')
    A('')
    A('**判据**（与 v10 预注册一致）：seed 均值差距轨迹 ±1 点平滑后首次转正且不再回穿的 eval 步；窗内未交叉记删失。')
    A('')
    A('| 规模 | 面板 | seeds | 交叉步（seed 均值轨迹） | 交叉 token | 状态 |')
    A('|---|---|---|---|---|---|')
    for label, v in xo.items():
        if v.get('status') != 'ok':
            A(f'| {label} | `{v['outdir']}` | — | — | — | 数据缺失 |')
            continue
        n = len(v['seeds'])
        _ntag = f'{n}（重构，非独立种子）' if label in _xo_recon else f'{n}'
        if v['mean_crossed']:
            _tok = f'{v['mean_crossover_tokens'] / 1000000.0:.1f}M' if 'mean_crossover_tokens' in v else 'n/a'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | {v['mean_crossover_step']:g} | {_tok} | 已交叉 |')
        else:
            _stps = v['mean_traj']['steps']
            if not _stps:
                A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | — | — | 无公共 eval 步 |')
                continue
            last = _stps[-1]
            _cbeg = f'> {last * v['tokens_per_step'] / 1000000.0:.1f}M' if v.get('tokens_per_step') else '(tokens/step unknown)'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | > {last:g} | {_cbeg} | 删失 |')
    A('')
    ok = [v for v in xo.values() if v.get('status') == 'ok']
    if ok:
        A('逐种子交叉步：')
        A('')
        for label, v in xo.items():
            if v.get('status') != 'ok':
                continue
            parts = []
            for s, ps in sorted(v['per_seed'].items()):
                parts.append(f's{s}: {ps['crossover_step']:g}{('' if ps['crossed'] else '（删失）')}')
            A(f'- d={v['d']}: ' + '；'.join(parts))
        A('')
    A('**注意**：d=256 的轨迹取自 v6 重构 summary（README 记账说明 #1），逐种子交叉步互为副本，只有均值轨迹有效。')
    A('')
    if fit:
        A(f'**趋势拟合（仅供参考）**：{fit['n_points']} 个已观测交叉点，log10(交叉token) ~ log10(d) 斜率 {fit['slope']:.2f}（R²={fit['r2']:.2f}）。方向性证据，不作精确幂律主张。')
        A('')
    if scale384:
        A(f'**d=384 面板（P4F 后，实际 {_n384_txt}）：**')
        A('')
        A('| variant | PPL (mean±std) | n |')
        A('|---|---|---|')
        for v, p in scale384.items():
            A(f'| `{v}` | {_fmt_pm(p)} | {p['n']} |')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) |')
        A('|---|---|---|---|')
        for k, v in comp.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | —（{v['omitted']}） | — |')
                continue
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
        A('')
    A('---')
    A('')
    A('## 落盘产物的代码语义溯源')
    A('')
    A('上表每一个 PPL 都是**测量值**，但它们不是在同一天由同一版代码测出来的。训练记录把代码语义连同超参写进 `run_cfg` 指纹（尾部 `_cs{CODE_SEMANTICS}`）；`result_is_current` 只认指纹与当前戳一致的记录，缺戳的记录一律按**陈旧**处理并重算——这是保守方向，宁可重跑也不静默复用。')
    A('')
    A(f'**当前戳：`{L.CODE_SEMANTICS}`。** 全仓库 `results_*/summary.json` 共 {_prov_total} 条记录，其中 {_prov_synth} 条是 `synthesized`（从旧 `aggregate.json` 重建，无权重、无 PPL 之外的信息），剩 {_prov_measured} 条实测记录中带代码戳的为 **{_prov_stamped}** （{_prov_pct:.1f}%），无戳 {_prov_unstamped} 条。')
    A('')
    if _prov_stamped == 0 and _prov_measured:
        A('> **⚠ 本仓库全部实测记录均无代码戳。** 这些面板产生于戳建立之前，因此**无法仅凭落盘产物证明**它们由与当前代码语义一致的代码测得。这不等于数字有误——各面板与其 `analysis_*/`、`REPORT_*.md` 内部自洽，且本报告是由 `results_*/` 实时重算得到——但任何「结果由当前代码复现」的主张都**没有工件层面的依据**。要让某个面板获得这一保证，必须让它重跑一次（续跑逻辑会自动重跑缺戳的 cell）。')
    else:
        A(f'> **⚠ {_prov_unstamped} 条实测记录无代码戳**，续跑逻辑会把它们判为陈旧并重算。下表逐面板列出覆盖率；未达 100% 的面板在重跑完成前不能主张「与当前代码语义一致」。')
    A('')
    A('| 面板 | 记录数 | 实测 | 带戳 | 重建 | 溯源状态 |')
    A('|---|---|---|---|---|---|')
    for _name, _t, _m, _s, _sy in _prov_panels:
        _st_txt = '已认证' if _m and _s == _m else '全部无戳' if _s == 0 else f'{_s}/{_m}'
        A(f'| `{_name}/` | {_t} | {_m} | {_s} | {_sy} | {_st_txt} |')
    for _name, _err in _prov_unreadable:
        A(f'| `{_name}/` | — | — | — | — | **无法解析**（{_err}） |')
    A('')
    if _prov_unreadable:
        A(f'> **⚠ 上表有 {len(_prov_unreadable)} 个面板无法解析**，其记录数未计入下方汇总。这些面板的溯源状态**未知**（不是「无戳」），在重新解析成功前不能假定它们与任何代码语义一致。')
        A('')
    A('重建记录（`synthesized`）的 PPL 是从更早的表里抄来的，没有权重支撑，本报告的配对与聚合都已在读取时把它们剔出——它们不会进入任何Δ、n 或 p 值。上面把它们单独计一列，是为了让「这条记录存在」与「这条记录是一次测量」不再被混为一谈。')
    A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')

    def _n_seeds_of(_outdir):
        try:
            _s = json.load(open(os.path.join(REPO, _outdir, 'summary.json'), encoding='utf-8'))
        except Exception:
            return None
        _sd = {}
        for _r in _s.values():
            if isinstance(_r, dict) and 'seed' in _r and _r.get('variant') and L.ppl_is_usable(_r.get('ppl')) and (not _r.get('synthesized')):
                _sd.setdefault(_r['variant'], set()).add(_r['seed'])
        _ns = [len(x) for x in _sd.values()]
        return (min(_ns), max(_ns)) if _ns else None

    def _n_txt(_outdir, _target):
        _n = _n_seeds_of(_outdir)
        if _n is None:
            return f'n=?→{_target}'
        if _n[0] == _n[1]:
            return f'n={_n[0]}' if _n[0] == _target else f'n={_n[0]}（目标 {_target}，未完成）'
        return f'n={_n[0]}–{_n[1]}（目标 {_target}）'
    A(f'| `results_v10_mech/` | P4MT 追加 seeds 3–5 权重与 by-length 行 + P4MP 探针 cell（`distractor.json`，实际 n={n_contr_max}） |')
    A(f'| `results_lm_v10_scale_s/` | P4S：d=128/4L 面板 {_n_txt('results_lm_v10_scale_s', 4)} |')
    A(f'| `results_lm_v10_scale_l/` | P4S：d=512/10L 面板 {_n_txt('results_lm_v10_scale_l', 4)} |')
    A(f'| `results_lm_v5_scale/` | P4F：d=384 `full` 臂 {_n_txt('results_lm_v5_scale', 4)} |')
    A(f'| `analysis_v11/stats.json` | n={n_contr_max} 探针对比、交叉定位（{_xo_txt}）、配对检验 |')
    A('| `autodl_budget_state_v11.json` | v11 CostGuard 台账 |')
    A('| `v11_supp.py` | 本阶段驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v11 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V11_NO_PUSH'):
        print(f'[git] push skipped (V11_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V11_NO_SHUTDOWN'):
        print('[v11] shutdown suppressed (V11_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v11] instance shuts down in {delay_s}s.')

def run_full():
    import atexit
    atexit.register(lambda: sys.stdout.flush())
    guard = make_guard()
    guard.report()
    print(f'[v11] remaining ¥{guard.remaining_yuan():.2f} (cap ¥{guard.cap_yuan():.2f} @ ¥{guard.price:.2f}/h)')
    git_push('v11: supplementary driver (probe n=6 + crossover panels n=4 + d384 full-arm seed completion)')
    all_ok = True
    for pname, _kind, _payload, _seeds, est_h in PHASES:
        rem = guard.remaining_yuan()
        if rem < 1.0:
            print(f'[v11] stopping before {pname}: ¥{rem:.2f} left')
            break
        if not V.cuda_healthy():
            print(f'[v11] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v11 phase {pname} (~{est_h} h est, ¥{rem:.2f} left) =====')
        try:
            run_phase(pname, guard)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v11: phase {pname} results')
    try:
        v11_analysis()
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v11: stats + REPORT_v11.md (analysis_v11)')
    guard.report()
    print('\n[v11] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) tiny P4MT train + probe roundtrip (seed 9, 30 steps)')
    import shutil
    shutil.rmtree('results_smoke_v11', ignore_errors=True)
    payload = dict(P4MT_CFG(), outdir='results_smoke_v11', variants=['csa_fixed_rope'], seeds=[9], steps=30, eval_lens=[512, 1024])
    run_p4mt(payload, guard=None, label='v11 smoke')
    pcfg = dict(V10.PROBE, outdir='results_smoke_v11', ckpt_dir='results_smoke_v11/ckpt', summary='results_smoke_v11/distractor.json', variants=['csa_fixed_rope'], seeds=[9], eval_lens=[1024], rhos=[0.0, 0.5], n_seq=4, chunk=2)
    s = V10.run_probe(pcfg, guard=None, label='v11 smoke')
    got = {(r['arm'], r['rho']) for r in s.values()}
    want = {('learned', 0.0), ('learned', 0.5), ('randidx', 0.0), ('randidx', 0.5), ('allblocks', 0.0), ('allblocks', 0.5)}
    assert want <= got, f'probe cells missing: {want - got}'
    print(f'  probe roundtrip OK ({len(s)} cells)')
    shutil.rmtree('results_smoke_v11', ignore_errors=True)
    print('[smoke] 2) zero-GPU analysis/report on current artifacts')
    v11_analysis()
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
        print(__doc__ or f"[v11] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v11] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v11] mode 'phase' needs a phase name, e.g. `python v11_supp.py phase P4MT`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v11] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v11_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v11: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v11: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
