#!/usr/bin/env python3
import json, itertools, math, os, sys, functools, warnings
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'analysis_fuse')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import exp_lib as L

def _save_fig(fig, name, **kw):
    path = os.path.join(OUT, name)
    tmp = path + '.tmp.png'
    fig.savefig(tmp, format='png', **kw)
    os.replace(tmp, path)

def load(panel):
    path = os.path.join(ROOT, panel, 'summary.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'fuse_analysis: results panel `{panel}` is not present ({path}).\n  The repository tracks SOURCE only — `results_*/` directories, `REPORT*.md` and the budget ledgers are all build products and are not committed.\n  Regenerate the panel before running this zero-GPU analysis, e.g.:  python run_gapfill.py full   (or the driver that owns `{panel}`), then re-run `python fuse_analysis.py`.')
    with open(path, encoding='utf-8') as f:
        return json.load(f)
VARIANTS = {'hybrid_csa_dyn': ('results_lm_v4_abl', 'hybrid dyn (no-fuse)', False), 'hybrid_csa_dyn_fuse': ('results_lm_v4_abl', 'hybrid dyn + fuse', True), 'csa_dyn_fuse': ('results_lm_v4_abl', 'csa dyn + fuse', True), 'csa_dynamic': ('results_lm_v3_1500', 'csa dyn (no-fuse)', False)}
SEEDS = []

_RESOLVED = {}
_PANEL_CACHE = {}

def _read_panel(panel_name):
    hit = _PANEL_CACHE.get(panel_name)
    if hit is None:
        hit = L.read_panel(os.path.join(ROOT, panel_name))
        _PANEL_CACHE[panel_name] = hit
    return hit

def _variant_records(variant):
    panel_name = VARIANTS[variant][0]
    rkey = (panel_name, variant)
    if rkey not in _RESOLVED:
        groups = {}
        for (_v, _tag, _s), (_k, _r) in _read_panel(panel_name).items():
            if _v != variant:
                continue
            groups.setdefault(_tag, {})[_s] = _r
        if not groups:
            _RESOLVED[rkey] = {}
        else:
            if len(groups) > 1:
                _cur_sfx = f'_cs{L.CODE_SEMANTICS}'

                def _stamped(recs):
                    return any((isinstance(r, dict) and isinstance(r.get('run_cfg'), str) and r['run_cfg'].endswith(_cur_sfx) for r in recs.values()))
                _ranked = sorted(groups.items(), key=lambda kv: (not _stamped(kv[1]), -len(kv[1])))
                _keep_tag, _keep = _ranked[0]
                _dropped = {t: sorted(g) for t, g in _ranked[1:]}
                print(f'[fuse_analysis] WARNING: {panel_name}: `{variant}` spans {len(_ranked)} protocol groups — pooling across protocols is not allowed (design §3.3), so the analysis keeps only the `{_keep_tag or '(untagged)'}` group ({len(_keep)} seeds) and drops {_dropped}')
                _RESOLVED[rkey] = _keep
            else:
                _RESOLVED[rkey] = next(iter(groups.values()))
    return _RESOLVED[rkey]

def _panel_seeds():
    for _v in VARIANTS:
        _p = VARIANTS[_v][0]
        if not os.path.exists(os.path.join(ROOT, _p, 'summary.json')):
            load(_p)  # raises FileNotFoundError with the panel-specific message
    sets = [set(_variant_records(_v)) for _v in VARIANTS]
    common = set.intersection(*sets) if sets else set()
    if not common:
        raise ValueError('fuse_analysis: every required panel file exists but no seed is measured in EVERY arm, so no honest paired analysis can be produced; fill the panels first')
    return sorted(common)
BND_KEYS = ['bnd_prec', 'bnd_rec', 'bnd_f1', 'bnd_rand', 'bnd_excess']
LEN_KEYS = ['blocks', 'len_mean', 'len_std', 'len_max', 'frac_at_min', 'delta']

def layer_idx(layers):
    return [int(lk.split('_')[0][1:]) for lk in layers]

def rec(variant, seed):
    r = _variant_records(variant).get(seed)
    if not isinstance(r, dict):
        raise KeyError(f'fuse_analysis: {variant}::seed{seed} is absent from {VARIANTS[variant][0]}/summary.json — the panel is incomplete; run that cell first (or restrict SEEDS to the ones present)')
    if r.get('synthesized') or not L.ppl_is_usable(r.get('ppl')):
        raise ValueError(f'fuse_analysis: {variant}::seed{seed} has no usable `ppl` ({r.get('ppl')!r}) — it is not a measurement, so it cannot be paired. Re-run that cell or exclude the variant.')
    return r

def layers_of(variant, dyn_only=True):
    r = rec(variant, SEEDS[0])
    st = r.get('stats')
    if not isinstance(st, dict):
        raise ValueError(f'fuse_analysis: {variant}::seed{SEEDS[0]} carries no usable `stats` table (got {type(st).__name__}) — the per-layer boundary statistics cannot be read')
    ks = [k for k in st if k.startswith('L')]
    if dyn_only:
        ks = [k for k in ks if 'dyn' in k]
    return ks

@functools.lru_cache(maxsize=None)
def per_seed_layer(variant, key):
    layers = layers_of(variant)
    M = np.full((len(SEEDS), len(layers)), np.nan)
    n_sampled = 0
    for i, s in enumerate(SEEDS):
        st = rec(variant, s).get('stats')
        if not isinstance(st, dict):
            raise ValueError(f'fuse_analysis: {variant}::seed{s} carries no usable `stats` table (got {type(st).__name__}) — the per-layer boundary statistics cannot be read')
        for j, lk in enumerate(layers):
            _cell = st.get(lk)
            if isinstance(_cell, dict):
                v = _cell.get(key, np.nan)
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    v = np.nan
                if key in ('bnd_rand', 'bnd_excess') and _cell.get('bnd_rand_exact') is False:
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        n_sampled += 1
                    v = np.nan
                M[i, j] = v
    if n_sampled:
        print(f'[fuse_analysis] NOTE: `{variant}` {key}: {n_sampled} cell(s) carry `bnd_rand_exact: False` (Monte-Carlo estimate, not the exact baseline) and are masked out rather than pooled with the exact readings')
    return (layers, M)

def _axis_steps(histories):
    if not histories:
        return []
    common = set(histories[0])
    for h in histories[1:]:
        common &= set(h)
        if not common:
            return []
    return sorted(common)

def signflip(deltas):
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    n = len(d)
    obs = abs(d.mean())
    if n > 20:
        rng = np.random.default_rng(0)
        B = 200000
        signs = np.where(rng.random((B, n)) < 0.5, 1.0, -1.0)
        cnt = 1 + int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
        return cnt / (B + 1)
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n

def sample_std(deltas, axis=None):
    d = np.asarray(deltas, dtype=float)
    if axis is None:
        d = d[np.isfinite(d)]
        return float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    if d.shape[axis] < 2:
        return np.zeros(d.shape[:axis] + d.shape[axis + 1:])
    srt = np.moveaxis(d, axis, 0).reshape(d.shape[axis], -1)
    out = np.zeros(srt.shape[1])
    for j in range(srt.shape[1]):
        col = srt[:, j][np.isfinite(srt[:, j])]
        out[j] = float(np.std(col, ddof=1)) if col.size > 1 else 0.0
    return out.reshape(d.shape[:axis] + d.shape[axis + 1:])

def _row_nanmean(M):
    out = np.full(M.shape[0], np.nan)
    for i in range(M.shape[0]):
        r = M[i][np.isfinite(M[i])]
        if r.size:
            out[i] = r.mean()
    return out

def _lay_nanmean(M):
    out = np.full(M.shape[1], np.nan)
    for j in range(M.shape[1]):
        c = M[:, j][np.isfinite(M[:, j])]
        if c.size:
            out[j] = c.mean()
    return out

def _finite_mean(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    for a in argv:
        if a in ('-h', '--help'):
            print(__doc__ or '')
            print('Usage: python fuse_analysis.py\n  Zero GPU.  Reads results_*/summary.json panels (build products, not tracked)\n  and writes analysis_fuse/{fuse_report.md,stats.json}.')
            return 0
        raise SystemExit(f'fuse_analysis: unknown argument {a!r} (this driver takes none; try -h)')
    os.makedirs(OUT, exist_ok=True)
    global SEEDS
    SEEDS = _panel_seeds()
    report = {}
    stats_out = {}
    probe = {}
    lines = []
    lines.append('# fuse vs no-fuse — 零 GPU 边界统计对比（全部数字由落盘产物计算）\n')
    lines.append(f'> 数据来源：`results_lm_v4_abl/summary.json`（1500 步 × {len(SEEDS)} seeds，seq 512）与\n> `results_lm_v3_1500/summary.json`（同配置核心面板）。无权重、无 GPU，\n> 仅使用每条 run 记录的 `stats`（逐层边界对齐 + 块长统计）与 `ppl_history`。\n')
    lines.append('**容差说明（诚实记录）**：落盘的边界对齐只在 `tol=1` 下计算（`boundary_alignment(..., tol=1)`），逐 token 的切点位置未保存、模型权重未保存，因此**无法离线重算其它容差**。下文改用 precision / recall / 相对随机基线的超额（excess）分解来回答「F1 提升从哪来、是否真实」——这一分解对容差选择不敏感。\n')
    def _pair_cells_ok(a, b):
        try:
            for v in (a, b):
                for s in SEEDS:
                    rec(v, s)
            _la, _lb = (layers_of(a), layers_of(b))
            if _la != _lb:
                return f'`{a}` and `{b}` are not on the same layer list ({_la} vs {_lb}); the per-layer deltas would be a misaligned subtraction'
            return None
        except (KeyError, ValueError, TypeError) as e:
            return str(e)
    for pair, tag in [(('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn'), 'A. 同面板配对（hybrid，干净对照）'), (('csa_dyn_fuse', 'csa_dynamic'), 'B. 跨面板同配置（纯 CSA 栈；仅作参考，非严格配对）')]:
        a, b = pair
        lines.append(f'\n## {tag}\n')
        lines.append(f'`{a}` (fuse) vs `{b}` (no-fuse)\n')
        _unavail = _pair_cells_ok(a, b)
        if _unavail:
            lines.append(f'\n**该对照不可用**（{_unavail}）；其余部分照常输出。\n')
            stats_out[f'{a}__vs__{b}'] = dict(unavailable=_unavail)
            report[tag] = False
            continue
        layers = layers_of(a)
        _reasons = {s: L.pair_reason(rec(a, s), rec(b, s)) for s in SEEDS}
        _why = sorted({r for r in _reasons.values() if r})
        _ok_seeds = [s for s in SEEDS if _reasons[s] is None]
        _gate_ok = not _why
        _keep = np.array([_reasons[s] is None for s in SEEDS], dtype=bool)
        header = '| 层 | 指标 | no-fuse (mean±std) | fuse (mean±std) | Δ(fuse−no-fuse) |'
        lines.append(header)
        lines.append('|---|---|---|---|---|')
        pair_stats = {}
        if not _ok_seeds:
            lines.append(f'| — | — | — | — | —（没有通过配置/预算配对门禁的 seed（{_why}），描述性均值与 Δ 不计算——跨配置的种子差不进入任何数字） |')
        for key, name in [('bnd_f1', 'F1'), ('bnd_prec', 'precision'), ('bnd_rec', 'recall'), ('bnd_excess', 'excess(超过随机)')]:
            if not _ok_seeds:
                break
            Mb_layers, Mb = per_seed_layer(b, key)
            Ma_layers, Ma = per_seed_layer(a, key)
            if Ma_layers != Mb_layers:
                raise ValueError(f'fuse_analysis: `{a}` and `{b}` are not on the same layer list ({Ma_layers} vs {Mb_layers}); the per-layer deltas would be a misaligned subtraction.')
            _both = np.isfinite(Ma) & np.isfinite(Mb) & _keep[:, None]
            Ma, Mb = (np.where(_both, Ma, np.nan), np.where(_both, Mb, np.nan))
            db = _lay_nanmean(Ma) - _lay_nanmean(Mb)
            _sb = _row_nanmean(Mb)
            _sa = _row_nanmean(Ma)
            sd_b = sample_std(_sb) if Mb.shape[0] > 1 else 0.0
            sd_a = sample_std(_sa) if Ma.shape[0] > 1 else 0.0
            mean_b = _finite_mean(_sb)
            mean_a = _finite_mean(_sa)
            lines.append(f'| — | **{name}** | {mean_b:.4f}±{sd_b:.4f} | {mean_a:.4f}±{sd_a:.4f} | **{mean_a - mean_b:+.4f}** |')
            pair_stats[name] = dict(layers=list(Ma_layers), layer_idx=layer_idx(Ma_layers), nofuse_perlayer_mean=np.round(_lay_nanmean(Mb), 4).tolist(), fuse_perlayer_mean=np.round(_lay_nanmean(Ma), 4).tolist(), delta_perlayer=np.round(db, 4).tolist(), nofuse_seed_std=float(sd_b), fuse_seed_std=float(sd_a))
        if _ok_seeds and not _gate_ok:
            lines.append(f'\n> 注：上表的均值与 Δ 与下方检验同口径——只在通过配置/预算配对门禁的 {_ok_seeds} 上计算，{len(SEEDS) - len(_ok_seeds)} 个未过门禁的 seed（{_why}）不进入任何数字。')
        _, fb = per_seed_layer(b, 'bnd_f1')
        _, fa = per_seed_layer(a, 'bnd_f1')
        _jf = np.isfinite(fa) & np.isfinite(fb) & _keep[:, None]
        fa = np.where(_jf, fa, np.nan)
        fb = np.where(_jf, fb, np.nan)
        d_f1 = _row_nanmean(fa) - _row_nanmean(fb)
        d_f1 = d_f1[np.isfinite(d_f1)]
        pa = np.array([rec(a, s)['ppl'] for s in SEEDS])
        pb = np.array([rec(b, s)['ppl'] for s in SEEDS])
        d_ppl = (pa - pb)[_keep]
        d_ppl = d_ppl[np.isfinite(d_ppl)]
        if len(d_f1) and len(d_ppl):
            p_f1 = signflip(d_f1)
            p_ppl = signflip(d_ppl)
            _n_f1 = len(d_f1)
            lines.append(f'\n**配对精确符号翻转检验（n={_n_f1}，按 seed 配对）**：')
            if not _gate_ok:
                lines.append(f'> 注：{len(SEEDS) - len(_ok_seeds)} 个 seed 未通过配置/预算配对门禁（{_why}），已按本仓库惯例排除——Δ 与 p 值只在可配对的 {_ok_seeds} 上计算，跨配置的种子差不进入检验。')
            lines.append(f'- 层均边界 F1：Δ = {d_f1.mean():+.4f} ± {sample_std(d_f1):.4f}，{int(max((d_f1 > 0).sum(), (d_f1 < 0).sum()))}/{_n_f1} 同向，p(exact) = {p_f1:.3f}')
            lines.append(f'- 最终 PPL：Δ = {d_ppl.mean():+.2f} ± {sample_std(d_ppl):.2f}，p(exact) = {p_ppl:.3f}（n={len(d_ppl)} 时 p 分辨率下限 {2.0 / 2 ** max(len(d_ppl), 1):.3f}）\n')
        elif not _ok_seeds:
            p_f1 = p_ppl = None
            lines.append(f'\n**配对检验：未执行。** 上述两臂不满足本仓库的可配对条件（{_why}）——它们的训练配置无法证明相同，**不给出 p 值**、不进入任何显著性主张。\n')
        else:
            p_f1 = p_ppl = None
            lines.append('\n**配对检验：未执行。** 可用于配对的有限样本不足，**不给出 p 值**、不进入任何显著性主张。\n')
        unstamped = sum((1 for v in pair for s in SEEDS if (_variant_records(v).get(s) or {}).get('run_cfg') is None))
        stats_out[f'{a}__vs__{b}'] = dict(boundary=pair_stats, f1_delta_per_seed=d_f1.round(4).tolist(), f1_p_exact=p_f1, ppl_fuse=pa[_keep].round(2).tolist(), ppl_nofuse=pb[_keep].round(2).tolist(), ppl_delta_per_seed=d_ppl.round(3).tolist(), ppl_p_exact=p_ppl, n_unstamped=unstamped, paired_test='run' if p_f1 is not None else 'skipped', pair_refusals=_why)
        _is_clean = a == 'hybrid_csa_dyn_fuse'
        if _is_clean:
            probe['d_f1'] = float(d_f1.mean()) if len(d_f1) else None
            probe['sd_f1'] = float(sample_std(d_f1)) if len(d_f1) > 1 else None
            probe['p_f1'] = float(p_f1) if p_f1 is not None else None
            probe['d_ppl'] = float(d_ppl.mean()) if len(d_ppl) else None
            probe['sd_ppl'] = float(sample_std(d_ppl)) if len(d_ppl) > 1 else None
            probe['p_ppl'] = float(p_ppl) if p_ppl is not None else None
            probe['n'] = int(len(d_f1))
            probe['n_ppl'] = int(len(d_ppl))
        else:
            probe['x_d_f1'] = float(d_f1.mean()) if len(d_f1) else None
            probe['x_sd_f1'] = float(sample_std(d_f1)) if len(d_f1) > 1 else None
            probe['x_p_f1'] = float(p_f1) if p_f1 is not None else None
            probe['x_ppl_hybrid'] = None
            probe['x_d_ppl_csa'] = float(d_ppl.mean()) if len(d_ppl) else None
            probe['x_sd_ppl_csa'] = float(sample_std(d_ppl)) if len(d_ppl) > 1 else None
            probe['x_p_ppl_csa'] = float(p_ppl) if p_ppl is not None else None
            probe['x_d_f1_guarded'] = p_f1 is not None
        report[tag] = True
    lines.append(f'\n## 块长分布形状对比（对全部动态层与 {len(SEEDS)} seeds 合并平均）\n')
    lines.append('| 变体 | len_mean | len_std | len_max | frac_at_min(贴下限块占比) | blocks/seq | δ(gate) |')
    lines.append('|---|---|---|---|---|---|---|')
    _blk, _pgate_drop = ({}, [])
    _pair_missing = False
    try:
        _pair_ok_seeds = [s for s in SEEDS if L.pair_reason(rec('hybrid_csa_dyn', s), rec('hybrid_csa_dyn_fuse', s)) is None]
    except (KeyError, ValueError, TypeError):
        _pair_ok_seeds = []
        _pair_missing = True
    _pgate_drop = [s for s in SEEDS if s not in _pair_ok_seeds]
    for _key, _name, _fmt in [('len_mean', 'len_mean', '{:+.3f}'), ('len_std', 'len_std', '{:+.3f}'), ('frac_at_min', 'frac_at_min', '{:+.4f}'), ('blocks', 'blocks', '{:+.3f}')]:
        try:
            _l, _mb = per_seed_layer('hybrid_csa_dyn', _key)
            _l2, _ma = per_seed_layer('hybrid_csa_dyn_fuse', _key)
        except (KeyError, ValueError, TypeError):
            continue
        if _l != _l2:
            continue
        if len(_pair_ok_seeds) < len(SEEDS):
            _keep = np.zeros(len(SEEDS), dtype=bool)
            for _i, _s in enumerate(SEEDS):
                _keep[_i] = _s in _pair_ok_seeds
            _mb = np.where(_keep[:, None], _mb, np.nan)
            _ma = np.where(_keep[:, None], _ma, np.nan)
        _d = _ma - _mb
        if _d.size == 0 or not np.any(np.isfinite(_d)):
            continue
        _blk[_name] = float(np.nanmax(np.abs(_d)))
        _blk[_name + '_fmt'] = _fmt
    if _pair_missing:
        lines.append('> 注：分块偏移量未计算——至少一臂的记录缺失或不可计量，配对门禁无从谈起；本节的块长分布表只列各臂自身的描述值，不给出两臂偏移量。\n')
    elif not _pair_ok_seeds:
        lines.append(f'> 注：分块偏移量未计算——{_pgate_drop} 全部因两臂配置/预算不一致被 `pair_reason` 排除，没有可配对的 seed；本节的块长分布表只列各臂自身的描述值。\n')
    elif _pgate_drop:
        lines.append(f'> 注：分块偏移量的配对已按 `pair_reason` 过滤，{_pgate_drop} 因两臂配置/预算不一致被排除；下方偏移量是在 {_pair_ok_seeds} 上计算的。\n')
    for v, (panel_name, label, fused) in VARIANTS.items():
        try:
            _, Mm = per_seed_layer(v, 'len_mean')
            _, Ms = per_seed_layer(v, 'len_std')
            _, Mx = per_seed_layer(v, 'len_max')
            _, Mf = per_seed_layer(v, 'frac_at_min')
            _, Mb = per_seed_layer(v, 'blocks')
            _, Md = per_seed_layer(v, 'delta')
        except (KeyError, ValueError, TypeError) as e:
            lines.append(f'| `{v}` | — | — | — | — | — | （该变体数据缺失：{e}）|')
            continue
        def _nm(M):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                v = np.nanmean(M)
            return f'{v:.1f}' if np.isfinite(v) else '—'
        def _nm3(M):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                v = np.nanmean(M)
            return f'{v:.3f}' if np.isfinite(v) else '—'
        def _nmp(M):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                v = np.nanmean(M)
            return f'{v:+.3f}' if np.isfinite(v) else '—'
        lines.append(f'| `{v}` | {_nm3(Mm)} | {_nm3(Ms)} | {_nm(Mx)} | {_nm3(Mf)} | {_nm(Mb)} | {_nmp(Md)} |')
    lines.append('')
    _hyb_keep = None
    try:
        _hyb_keep = np.array([L.pair_reason(rec('hybrid_csa_dyn', s), rec('hybrid_csa_dyn_fuse', s)) is None for s in SEEDS], dtype=bool)
    except (KeyError, ValueError, TypeError):
        _hyb_keep = np.zeros(len(SEEDS), dtype=bool)
    _hyb_seeds = [s for s, k in zip(SEEDS, _hyb_keep) if k]
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (a, b, ttl) in zip(axes, [('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn', 'hybrid stack (in-panel paired)'), ('csa_dyn_fuse', 'csa_dynamic', 'pure CSA stack (cross-panel)')]):
        _in_panel = (a, b) == ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn')
        _seeds_ax = _hyb_seeds if _in_panel else SEEDS
        if _in_panel and not _seeds_ax:
            print('[fuse_analysis] NOTE: skipping the hybrid trajectory panel — no seed passes the run_cfg/budget pairing gate, so pooling the arms would mix configurations')
            ax.set_visible(False)
            continue
        for v, color, lab in [(b, 'steelblue', f'{b} (no-fuse)'), (a, 'darkorange', f'{a} (fuse)')]:
            try:
                hs = []
                for s in _seeds_ax:
                    _h = rec(v, s).get('ppl_history')
                    if not isinstance(_h, list):
                        raise ValueError(f'`ppl_history` of {v}::seed{s} is {type(_h).__name__}, not a list of [step, ppl] pairs')
                    hs.append(dict(_h))
                steps = _axis_steps(hs)
                Y = np.array([[h[st] for st in steps] for h in hs], dtype=float) if steps else None
            except (KeyError, ValueError, TypeError) as e:
                print(f'[fuse_analysis] NOTE: skipping trajectory of `{v}` — {e}')
                continue
            if not steps:
                continue
            Y[~np.isfinite(Y) | (Y <= 0)] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ax.plot(steps, np.nanmean(Y, axis=0), color=color, lw=2, label=lab)
                ax.fill_between(steps, np.nanmin(Y, axis=0), np.nanmax(Y, axis=0), color=color, alpha=0.18)
        ax.set_title(ttl)
        ax.set_xlabel('step')
        ax.set_ylabel('val PPL (log)')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
    _traj_n = f'{len(SEEDS)} seeds' if len(_hyb_seeds) == len(SEEDS) else f'{len(SEEDS)} seeds (hybrid panel gated to {len(_hyb_seeds)})'
    fig.suptitle(f'fuse vs no-fuse: validation PPL trajectories ({_traj_n}, min–max band)', y=1.0)
    fig.tight_layout()
    _save_fig(fig, 'fuse_ppl_traj.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    if not _hyb_seeds:
        print('[fuse_analysis] NOTE: skipping the boundary bar plot — no seed passes the run_cfg/budget pairing gate for the hybrid pair')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, key, ttl in zip(axes, ['bnd_f1', 'bnd_prec', 'bnd_rec'], ['boundary F1 (tol=1)', 'precision', 'recall']):
            try:
                layers, Mb = per_seed_layer('hybrid_csa_dyn', key)
                _lb, Ma = per_seed_layer('hybrid_csa_dyn_fuse', key)
                if _lb != layers:
                    raise ValueError(f'layer lists differ between the two arms ({layers} vs {_lb}) — not plotting side by side')
            except (KeyError, ValueError, TypeError) as e:
                print(f'[fuse_analysis] NOTE: skipping `{ttl}` boundary plot — {e}')
                continue
            Mb = np.where(_hyb_keep[:, None], Mb, np.nan)
            Ma = np.where(_hyb_keep[:, None], Ma, np.nan)
            x = np.arange(len(layers))
            w = 0.38
            ax.bar(x - w / 2, _lay_nanmean(Mb), w, yerr=sample_std(Mb, axis=0), capsize=3, color='steelblue', label='no-fuse')
            ax.bar(x + w / 2, _lay_nanmean(Ma), w, yerr=sample_std(Ma, axis=0), capsize=3, color='darkorange', label='fuse')
            ax.set_xticks(x)
            ax.set_xticklabels([lk.split('_')[0] for lk in layers])
            ax.set_title(f'{ttl} — hybrid panel')
            ax.grid(alpha=0.3, axis='y')
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, 'fuse_boundary.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    PLOTTED = [('hybrid_csa_dyn', 'steelblue'), ('hybrid_csa_dyn_fuse', 'darkorange'), ('csa_dynamic', 'seagreen'), ('csa_dyn_fuse', 'firebrick')]
    _plotted_layers = {}
    for v, _c in PLOTTED:
        try:
            _plotted_layers[v] = layers_of(v)
        except (KeyError, ValueError, TypeError) as e:
            print(f'[fuse_analysis] NOTE: skipping layer index contribution of `{v}` — {e}')
    all_idx = sorted({i for v, _ in PLOTTED for i in layer_idx(_plotted_layers.get(v, []))})
    ax = axes[0]
    for v, color in PLOTTED:
        try:
            layers, Mm = per_seed_layer(v, 'len_mean')
            _, Ms = per_seed_layer(v, 'len_std')
        except (KeyError, ValueError, TypeError) as e:
            print(f'[fuse_analysis] NOTE: skipping len_mean curve of `{v}` — {e}')
            continue
        x = np.array(layer_idx(layers))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ax.errorbar(x, np.nanmean(Mm, axis=0), yerr=np.nanmean(Ms, axis=0), marker='o', ms=4, lw=1.6, color=color, label=v, alpha=0.9)
    ax.set_xticks(all_idx)
    ax.set_xticklabels([f'L{i}' for i in all_idx])
    ax.set_title('block length: mean ± std (dynamic layers only)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    for v, color in PLOTTED:
        try:
            layers, Mf = per_seed_layer(v, 'frac_at_min')
        except (KeyError, ValueError, TypeError) as e:
            print(f'[fuse_analysis] NOTE: skipping frac_at_min curve of `{v}` — {e}')
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ax.plot(np.array(layer_idx(layers)), np.nanmean(Mf, axis=0), marker='s', ms=4, lw=1.6, color=color, label=v)
    ax.set_xticks(all_idx)
    ax.set_xticklabels([f'L{i}' for i in all_idx])
    ax.set_title('frac_at_min (degenerate min-length blocks)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, 'fuse_blocklen.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    def _pv(k, fmt='{:+.4f}'):
        v = probe.get(k)
        return '未测量' if v is None or not np.isfinite(v) else fmt.format(v)

    def _psd(k):
        v = probe.get(k)
        if v is None or not np.isfinite(v):
            return '未测量'
        return f'{('-' if v < 0 else '')}{abs(float(v)):.4f}'
    if _blk:
        _blk_txt = '、'.join((f'{_n} {_blk[_n + '_fmt'].format(_blk[_n])}' for _n in ('len_mean', 'len_std', 'frac_at_min', 'blocks') if _n in _blk))
        _blk_max = max((_blk[_n] for _n in ('len_mean', 'len_std', 'frac_at_min', 'blocks') if _n in _blk))
        if _blk_max <= 0.01:
            _blk_line = f'2. **分块行为不变（已由产物核验）**：同面板两臂的最大单层单种子偏移为 **{_blk_txt}**（各统计量取逐层逐 seed 差的绝对值上界，越界阈值 0.01），全部落在小数点后 1~2 位内——恒等初始化的融合卷积只经 comp_reg 的弱梯度训练，1500 步内基本没有离开恒等映射，既没有改变「在哪里切」，也没有改变「切多长」。\n'
        else:
            _blk_line = f'2. **分块行为在产物上并未重合（与「惰性旋钮」读法冲突，需复核）**：同面板两臂的最大单层单种子偏移为 **{_blk_txt}**（各统计量取逐层逐 seed 差的绝对值上界，越界阈值 0.01），即在该面板上 fuse 确实改变了分块行为；下文第 4 条的「没有改变块长分布」在本轮产物上**不成立**，应先查清是哪一层/哪个 seed 发生偏移再作结论。\n'
    else:
        _blk_line = '2. **分块行为的偏移量未测量**：本节无法从产物算出两臂的逐层块长差（层表不一致或统计量缺失），因此**不对「分块行为是否不变」作断言**。\n'
    lines.append('\n## 机制结论（与上方计算结果一致）\n')
    _n_p = probe.get('n')
    _n_pp = probe.get('n_ppl')
    _f1_d, _f1_p = (probe.get('d_f1'), probe.get('p_f1'))
    _ppl_d, _ppl_p = (probe.get('d_ppl'), probe.get('p_ppl'))
    _f1_sig = _f1_p is not None and _f1_p <= 0.05
    _ppl_sig = _ppl_p is not None and _ppl_p <= 0.05
    _blk_stable = bool(_blk) and _blk_max <= 0.01
    if _f1_d is None:
        _c1 = f'1. **边界 F1 的同面板配对差未测量**（n={_n_p or 0} 的可用配对不足），本节不对 F1 方向作断言。\n'
    elif _f1_sig:
        _c1 = f'1. **同面板配对下 fuse 显著改变了边界 F1**：hybrid 栈按 seed 配对（n={_n_p}），ΔF1 = **{_pv('d_f1')}**（±{_psd('sd_f1')}），p(exact) = **{_pv('p_f1', '{:.3f}')}** < 0.05——「F1 差不稳固」的读法在本轮产物上不成立。\n'
    else:
        _f1_floor = 2.0 / 2 ** _n_p if isinstance(_n_p, int) and _n_p > 0 else float('nan')
        _f1_floor_txt = f'p 值打在 n={_n_p} 的分辨率下限上，不能排除是噪声' if _f1_p is not None and math.isfinite(_f1_floor) and (abs(_f1_p - _f1_floor) < 1e-12) else f'p={_pv('p_f1', '{:.3f}')}，高于 n={_n_p or '?'} 的分辨率下限 {_f1_floor:.3f}，不能排除是噪声'
        _x_f1 = probe.get('x_d_f1')
        if _x_f1 is None or not np.isfinite(_x_f1):
            _x_f1_txt = '跨面板的纯 CSA 对比未测量'
        else:
            _x_f1_txt = f'只有在跨面板的纯 CSA 对比里才看到 {_pv('x_d_f1')} 的 F1 差' + (f'（p={_pv('x_p_f1', '{:.3f}')}）' if probe.get('x_d_f1_guarded') else '（该对比未通过本仓库的可配对门禁，故未执行检验、不给 p 值）') + '，而该对比非严格配对（面板/实现差异未受控）'
        _c1 = f'1. **连「F1 提升」本身都不稳固**：在最干净的同面板配对（hybrid 栈，仅 csa_dyn 层，按 seed 配对，n={_n_p or '?'}）中，fuse 与 no-fuse 的边界 F1 差为 **{_pv('d_f1')}**（±{_psd('sd_f1')}），p(exact) = **{_pv('p_f1', '{:.3f}')}**，即 {_f1_floor_txt}；{_x_f1_txt}，不足以支撑机制性主张。\n'
    if _ppl_d is None:
        _c3 = '3. **验证 PPL 的配对差未测量**，本节不对 PPL 方向作断言。\n'
    elif _ppl_sig:
        _c3 = f'3. **PPL 存在可分辨的差异**：同面板配对的终点差为 **{_pv('d_ppl', '{:+.2f}')} PPL**（±{_psd('sd_ppl')}，p={_pv('p_ppl', '{:.3f}')} < 0.05），「PPL 不变」的读法在本轮产物上不成立。\n'
    else:
        _ppl_floor = 2.0 / 2 ** _n_pp if isinstance(_n_pp, int) and _n_pp > 0 else float('nan')
        _ppl_floor_txt = f'配对符号翻转 p 值打在 n={_n_pp} 的分辨率下限上，任何差异不能排除是种子噪声' if _ppl_p is not None and math.isfinite(_ppl_floor) and (abs(_ppl_p - _ppl_floor) < 1e-12) else f'配对符号翻转 p={_pv('p_ppl', '{:.3f}')}，高于 n={_n_pp or '?'} 的分辨率下限 {_ppl_floor:.3f}，任何差异不能排除是种子噪声'
        _x_ppl = probe.get('x_d_ppl_csa')
        _x_ppl_txt = f'，跨面板纯 CSA 对比为 {_pv('x_d_ppl_csa', '{:+.2f}')} PPL' if _x_ppl is not None and np.isfinite(_x_ppl) else '，跨面板纯 CSA 对比未测量'
        _c3 = f'3. **PPL 同样不变**：两条验证 PPL 轨迹全程重叠（见 `fuse_ppl_traj.png`），同面板配对的终点差为 **{_pv('d_ppl', '{:+.2f}')} PPL**（±{_psd('sd_ppl')}，p={_pv('p_ppl', '{:.3f}')}）{_x_ppl_txt}——{_ppl_floor_txt}。\n'
    if _blk_stable and not _f1_sig and not _ppl_sig and (_f1_d is not None) and (_ppl_d is not None):
        _c4 = '4. **结论写法（可直接引用）**：在 1500 步 budget 区间，用于边界检测的邻域表示融合是一个**惰性旋钮（inert knob）**——它既没有稳定改善动态分块的边界对齐质量，在上文第 2 条核验通过的范围内也没有改变块长分布，更没有转化为语言建模收益。这与全套实验的主结论一致：该 budget 下 LM 损失对分块质量不敏感（v4 全部消融臂停在同一 PPL 水平），且动态分块母轴本身在 4 个面板（核心表/20k 长跑/RoPE/scale）都与固定分块贴平。\n'
    else:
        _c4 = '4. **结论写法**：本轮产物**不满足**「惰性旋钮」的全部前置条件（第 1–3 条中至少一条报错、显著或越界）——不作「fuse 是惰性旋钮」的断言，以第 1–3 条各自的实测结果为准。\n'
    lines.append(_c1 + _blk_line + _c3 + _c4 + '5. **遗留（如需更强断言）**：本分析限于 tol=1（落盘唯一容差）与 1500 步 budget 区间；其它容差或收敛区间（20k 步）的断言需要新 run。\n')
    md = '\n'.join(lines)
    L.atomic_write_text(os.path.join(OUT, 'fuse_report.md'), md)
    L.atomic_write_json(os.path.join(OUT, 'stats.json'), stats_out, indent=1)
    print(md[:1200])
    print('\n[done] ->', OUT)
    return 0
if __name__ == '__main__':
    try:
        sys.exit(main())
    except FileNotFoundError as _e:
        print(f'\n[fuse_analysis] {_e}', file=sys.stderr)
        sys.exit(2)
