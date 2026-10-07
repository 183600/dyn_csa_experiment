#!/usr/bin/env python3
import collections
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
REPO = os.path.dirname(os.path.abspath(__file__))
PAIR_STEP_KEYS = ('steps', 'tokens_seen')
_LOADED = {}

def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as _e:
        print(f'[build_report] WARNING: {path} exists but cannot be read ({type(_e).__name__}: {_e}) — treating it as absent rather than silently using partial data')
        return default

def agg(outdir):
    key = ('agg', outdir)
    if key not in _LOADED:
        _LOADED[key] = load(os.path.join(REPO, outdir, 'aggregate.json'), {}) or {}
    return _LOADED[key]

def summary(outdir):
    key = ('summary', outdir)
    if key not in _LOADED:
        _LOADED[key] = load(os.path.join(REPO, outdir, 'summary.json'), {}) or {}
    return _LOADED[key]

def fm(x, nd=2):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) else 'n/a'

def sp(x, nd=2):
    return f'±{x:.{nd}f}' if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) else ''

def _mins(v):
    return v / 60.0 if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None

def _hist_points(r):
    for _pt in (r.get('ppl_history') or []) if isinstance(r, dict) else []:
        try:
            _st, _pv = _pt
            _st = int(_st)
        except (TypeError, ValueError):
            continue
        yield (_st, _pv)

def sample_std(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    return (sum(((v - mu) ** 2 for v in vals)) / (n - 1)) ** 0.5

def exact_signflip(deltas):
    deltas = [float(d) for d in deltas if isinstance(d, (int, float)) and (not isinstance(d, bool)) and math.isfinite(d)]
    n = len(deltas)
    if n == 0:
        return None
    obs = abs(sum(deltas))
    mu = sum(deltas) / n
    std = (sum(((d - mu) ** 2 for d in deltas)) / (n - 1)) ** 0.5 if n > 1 else 0.0
    if n > 20:
        B = 200000
        try:
            import numpy as _np
        except ImportError:
            _np = None
        if _np is not None:
            _rng = _np.random.default_rng(0)
            cnt = 1
            _step = max(1, min(B, (1 << 24) // max(n, 1)))
            for _off in range(0, B, _step):
                _bs = min(_step, B - _off)
                _signs = _np.where(_rng.random((_bs, n)) < 0.5, 1.0, -1.0)
                _sums = _np.abs((_signs * _np.asarray(deltas)[None, :]).sum(1))
                cnt += int((_sums >= obs - 1e-12).sum())
        else:
            rng = random.Random(0)
            cnt = 1
            for _ in range(B):
                s = 0.0
                for d in deltas:
                    s += d if rng.random() < 0.5 else -d
                if abs(s) >= obs - 1e-12:
                    cnt += 1
        return {'n': n, 'mean': mu, 'std': std, 'p_exact_signflip': cnt / (B + 1)}
    cnt = 0
    for mask in range(1 << n):
        s = sum((d if mask >> i & 1 else -d for i, d in enumerate(deltas)))
        if abs(s) >= obs - 1e-12:
            cnt += 1
    mu = sum(deltas) / n
    return {'n': n, 'mean': mu, 'std': (sum(((d - mu) ** 2 for d in deltas)) / (n - 1)) ** 0.5 if n > 1 else 0.0, 'p_exact_signflip': cnt / (1 << n)}

def _measurable(r):
    v = r.get('ppl')
    return not r.get('synthesized') and _ppl_ok(v)

def _ppl_ok(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) and (v > 0.0)

def _finite_or_none(v):
    return float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None

def _int_or(v, default):
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and float(v).is_integer() else default
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return default

def _code_semantics():
    try:
        with open(os.path.join(REPO, 'exp_lib.py'), encoding='utf-8') as f:
            m = re.search("^CODE_SEMANTICS = '([^']+)'", f.read(), re.M)
            return m.group(1) if m else None
    except Exception:
        return None
_CUR_CS = _code_semantics()

def _finite_ppls(r):
    ppls = r.get('ppls') or r.get('ppl_list') or []
    return [x for x in ppls if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) and (x > 0)]

_WARM_KEY_RE = re.compile('.*::w(\\d+)::seed\\d+$')

def _legacy_warm_tags(s):
    _legacy = {}
    for key, rec in s.items():
        if not isinstance(rec, dict) or 'seed' not in rec or 'ppl' not in rec:
            continue
        if not _ppl_ok(rec.get('ppl')) or rec.get('synthesized'):
            continue
        if ('protocol' in rec and rec['protocol'] is not None) or rec.get('warm_steps') is not None:
            continue
        m = _WARM_KEY_RE.match(str(key))
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        _legacy.setdefault(str(_var), set()).add(m.group(1) if m else None)
    return {v: frozenset(tags) for v, tags in _legacy.items() if None not in tags}

def _tag_of(r, key=None, legacy_ok=None):
    if r.get('protocol') is not None:
        return str(r['protocol'])
    if r.get('warm_steps') is not None:
        return f"w{r['warm_steps']}"
    if key is not None:
        m = _WARM_KEY_RE.match(str(key))
        if m:
            ok = legacy_ok
            if isinstance(ok, dict):
                _var = r.get('variant')
                if _var is None:
                    _var = str(key).split('::', 1)[0]
                ok = ok.get(str(_var))
                if ok is None:
                    return ''
            if ok is not None and m.group(1) in ok:
                return f'w{m.group(1)}'
    return ''

def _prefer_one_cfg(items, cfg_of, who=''):
    by = collections.defaultdict(list)
    for it in items:
        by[cfg_of(it)].append(it)
    per = collections.defaultdict(list)
    for (label, cfg), lst in by.items():
        per[label].append((cfg, lst))
    kept, dropped = ([], [])
    for label, gl in per.items():
        if len(gl) == 1:
            kept.extend(gl[0][1])
            continue
        gl.sort(key=lambda kv: (not (isinstance(kv[0], str) and _CUR_CS and kv[0].endswith(f'_cs{_CUR_CS}')), -len(kv[1])))
        kept.extend(gl[0][1])
        for _cfg, lst in gl[1:]:
            dropped.extend(lst)
    if dropped:
        print(f'[report] {who}: {len(dropped)} record(s) sit in a minority run_cfg group of their variant — pooling across configurations is not allowed, so one group per variant is kept (the current code stamp preferred) and the rest are dropped from this table')
    return kept
PARAM_MATCH_TOL = 0.015
PARAM_MATCH_SMALL = 0.3
PARAM_MATCH_ARCH = 0.06

def param_gap(a, b, tol=PARAM_MATCH_TOL):
    pa = (a or {}).get('params')
    pb = (b or {}).get('params')
    if not (isinstance(pa, (int, float)) and isinstance(pb, (int, float))):
        return None
    if pa <= 0 or pb <= 0:
        return None
    return abs(pa - pb) / max(pa, pb)

def is_param_matched(a, b, tol=PARAM_MATCH_TOL):
    g = param_gap(a, b, tol)
    return None if g is None else g <= tol

def gap_is_architecture(a, b, tol=PARAM_MATCH_TOL, arch=PARAM_MATCH_ARCH):
    g = param_gap(a, b, tol)
    if g is None:
        return None
    return tol < g <= arch

def unmatched_tag(a, b, tol=PARAM_MATCH_TOL, small=PARAM_MATCH_SMALL):
    g = param_gap(a, b, tol)
    if g is None:
        return ' (⚠ params 未知)'
    if g <= tol:
        return ''
    if g > small:
        return f' (⚠ params 差异过大 {100 * g:.1f}%)'
    if gap_is_architecture(a, b, tol):
        return f' (⚠ 参数差 {100 * g:.1f}%，属架构差异)'
    return f' (⚠ 参数不匹配 {100 * g:.1f}%)'

def per_seed_records(outdir):
    _ck = ('psr', outdir)
    if _ck in _LOADED:
        return _LOADED[_ck]
    s = summary(outdir)
    _legacy_ok = _legacy_warm_tags(s)
    out = collections.defaultdict(dict)
    for _k, r in s.items():
        if isinstance(r, dict) and 'ppl' in r and ('variant' in r) and ('seed' in r) and _measurable(r):
            _sv = r['seed']
            if isinstance(_sv, bool):
                print(f"[report] record {_k!r} in {outdir} carries a boolean seed — skipped for pairing")
                continue
            if isinstance(_sv, float):
                _sd = int(_sv) if math.isfinite(_sv) and _sv.is_integer() else None
            else:
                try:
                    _sd = int(_sv)
                except (TypeError, ValueError, OverflowError):
                    _sd = None
            if _sd is None:
                print(f"[report] record {_k!r} in {outdir} carries a non-integer seed ({r.get('seed')!r}) — skipped for pairing")
                continue
            _slot = out[r['variant']]
            _tk = (_tag_of(r, _k, _legacy_ok), _sd)
            if _tk in _slot:
                print(f'[report] ambiguous (variant, protocol, seed) = ({r['variant']!r}, {_tk[0]!r}, {_sd}) in {outdir}: two measurable records found; keeping the FIRST (key {_k!r} ignored for pairing)')
                continue
            _slot[_tk] = r
    out = dict(out)
    _LOADED[_ck] = out
    return out

def _pair_reason(ra, rb):
    fa, fb = (ra.get('run_cfg'), rb.get('run_cfg'))
    if fa is None or fb is None:
        return 'unverifiable run_cfg'
    if fa != fb:
        return 'different run_cfg'
    bad = [k for k in PAIR_STEP_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
    if not bad:
        return None
    absent = [k for k in bad if ra.get(k) is None and rb.get(k) is None]
    unknown = [k for k in bad if k not in absent and (ra.get(k) is None) != (rb.get(k) is None)]
    if unknown:
        return f'unverifiable {sorted(unknown)}'
    if absent:
        return f'unverifiable {sorted(absent)}'
    return f'disagreeing {sorted(bad)}'

def paired_row(a_records, b_records):
    common = sorted(set(a_records) & set(b_records))
    if not common:
        return None
    _by_tag = {}
    for _tk in common:
        _by_tag.setdefault(_tk[0] if isinstance(_tk, tuple) else '', []).append(_tk)
    if len(_by_tag) > 1:
        _ranked = sorted(_by_tag.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
        _keep_tag, _kept_keys = _ranked[0]
        print(f"[report] paired_row: the two arms share {len(_by_tag)} protocol conditions ({sorted(_by_tag)}) — one sign-flip test cannot pool deltas across protocols (that would fake extra seeds), so only the largest shared condition `{_keep_tag or '(untagged)'}` ({len(_kept_keys)} seed(s)) is tested; the rest ({sorted((t for t in _by_tag if t != _keep_tag))}) is left to its own per-condition section")
        common = _kept_keys
    dl, kept, skipped = ([], [], [])
    for s in common:
        reason = _pair_reason(a_records[s], b_records[s])
        if reason:
            skipped.append(s)
            continue
        dl.append(a_records[s]['ppl'] - b_records[s]['ppl'])
        kept.append(s)
    if skipped:
        print(f'[report] {len(skipped)}/{len(common)} seed(s) NOT paired ({skipped}); excluded from the sign-flip test')
    if not dl:
        print('[report] no usable pair left after the run_cfg/steps checks — comparison omitted rather than quoting a cross-config delta')
        return None
    st = exact_signflip(dl)
    return (kept, dl, st)

def sec_header():
    return f'# CSA / HCA 受控机制研究 — 补充实验报告 (v7)\n\n> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n> 参考论文：arXiv:2606.19348（DeepSeek-V4 稀疏注意力的受控复现与机制剖析）\n> 定位：**受控机制研究**（controlled mechanism study），非论文全量复现。\n> 说明：本报告全部数字由 `build_report.py` 从 `results_*/`、`analysis_v7/`\n> 的落盘产物计算得到，无手填数值。\n\n本报告针对外部评审提出的 3 项 P0 阻断项与 P1/P2 缺口逐一补做实验。所有新变体\n（RoPE+QK-norm、dense-warmup、topk 扫描、m=1、长度外推）都经由 **完全相同的\n训练/优化器/LR/评测/TimeGuard 代码路径**（对 `exp_lib` 的 `make_layer_cfgs`/\n`SmallGPT` 做 monkey-patch 注入），以保证可比性。长上下文证据采用\n**wikitext 长度外推探针**（train@512 → eval 512..4096）：合成 NIAH 探针在\n本模型规模（d=256/6 层）下两臂均停留在随机水平（无区分度），故以自然语料的\n长度外推替代，同样回应「无长上下文评测」的质疑。\n\n'
_MODULE_NOTE = '本模块只读：从 `results_*/` 与 `analysis_v7/` 的落盘产物计算\n报告全文，不 import torch（也不 import exp_lib）。上面那段\n模块说明文字以 `_MODULE_NOTE` 命名保留，不参与报告正文的渲染。'

def sec_p0r():
    rope = agg('results_lm_v7_rope')
    ab = agg('results_lm_v3_1500')
    if not rope or not ab:
        return '## P0-3 位置编码对照 — （无结果）\n\n'
    pairs = [('full', 'full_rope'), ('csa_fixed', 'csa_fixed_rope'), ('csa_dynamic', 'csa_dynamic_rope'), ('hybrid_fixed', 'hybrid_fixed_rope')]
    lines = ['## P0-3 位置编码错配是否为混淆项？（RoPE + QK-RMSNorm 对照）', '', '**评审论断**：论文用 partial RoPE（末 64 维）+ 输出侧反向 RoPE(pos=−i) + core attention 前 Q/KV RMSNorm；仓库用 `nn.Embedding(max_seq,d)` 学习式绝对位置编码、无 QK-norm。怀疑 PE 错配造成了 sparse 落后。', '', '**做法**：新增 `*_rope` 变体族，seed 0/1/2 × 1500 步 × seq 512，其余训练配置与 absPE 面板完全一致。', '', '| 变体 | absPE (v3_1500) | RoPE+QKnorm (v7) | Δ(RoPE−abs) | params(abs/rope) |', '|---|---|---|---|---|']
    for a, b in pairs:
        ea, eb = (_agg_entry(ab, a), _agg_entry(rope, b))
        if ea is not None and eb is not None:
            pa, pb = (_finite_or_none(ea.get('ppl_mean')), _finite_or_none(eb.get('ppl_mean')))
            if pa is None or pb is None:
                print(f'[report] sec_p0r: `{a}`/`{b}` lacks a finite ppl_mean — row omitted rather than printed as n/a arithmetic')
                continue
            lines.append(f'| `{a}` | {fm(pa)} {sp(ea.get('ppl_std'))} | {fm(pb)} {sp(eb.get('ppl_std'))} | **{pb - pa:+.2f}** | {_params_m(ea)} / {_params_m(eb)} |')
        else:
            print(f'[report] sec_p0r: `{a}`/`{b}` not both present (config-split or absent) — row omitted')
    lines += ['', '**组间差距（决定评审论点是否成立）**：', '']
    _cf, _fu = (_agg_entry(ab, 'csa_fixed'), _agg_entry(ab, 'full'))
    _cf_r, _fu_r = (_agg_entry(rope, 'csa_fixed_rope'), _agg_entry(rope, 'full_rope'))
    _mt = _agg_entry(ab, 'full_matched')
    if _cf is not None and _cf_r is not None and (_fu_r is not None) and (_fu is not None):
        _means = [_finite_or_none(_e.get('ppl_mean')) for _e in (_cf, _fu, _cf_r, _fu_r)]
        if any((m is None for m in _means)):
            print('[report] sec_p0r: one of the four arm entries lacks a finite ppl_mean — group-gap block omitted')
            return '\n'.join(lines) + '\n'
        g_abs = _means[0] - _means[1]
        _mtm = _finite_or_none((_mt or {}).get('ppl_mean'))
        g_abs_m = _cf['ppl_mean'] - _mtm if _mtm is not None else float('nan')
        g_rope = _cf_r['ppl_mean'] - _fu_r['ppl_mean']
        d_abs = _fu['ppl_mean'] - _fu_r['ppl_mean']
        d_sp = _cf['ppl_mean'] - _cf_r['ppl_mean']
        _has_mt = _mtm is not None
        rope_unmatched = unmatched_tag(_cf_r, _fu_r)
        _pgap_r = param_gap(_cf_r, _fu_r)
        rope_is_defect = gap_is_architecture(_cf_r, _fu_r) is False or _pgap_r is None
        _w_abs = '领先' if g_abs < 0 else '落后'
        _w_absm = '领先' if g_abs_m < 0 else '落后'
        _w_rope = '领先' if g_rope < 0 else '落后'
        _d_abs_t = ('改善' if d_abs > 0 else '变差') + f' **{abs(d_abs):.2f}**'
        _d_sp_t = ('改善' if d_sp > 0 else '变差') + f' **{abs(d_sp):.2f}**'
        _gain_conf = [f'dense 臂{unmatched_tag(_fu, _fu_r)}' for _ in [1] if unmatched_tag(_fu, _fu_r)] + [f'sparse 臂{unmatched_tag(_cf, _cf_r)}' for _ in [1] if unmatched_tag(_cf, _cf_r)]
        _gain_note = f'（注：{'；'.join(_gain_conf)}，对应对比同时含容量效应）' if _gain_conf else ''
        lines += [f'- absPE 面板：`csa_fixed` {_w_abs} dense **{abs(g_abs):.2f}** PPL（参数对齐 `full_matched` 时{_w_absm} **{abs(g_abs_m):.2f}**；两者相差 **{g_abs - g_abs_m:+.2f} PPL**，即容量差异贡献的部分）{unmatched_tag(_cf, _mt)}。' if _has_mt else f'- absPE 面板：`csa_fixed` {_w_abs} dense **{abs(g_abs):.2f}** PPL。⚠ **本面板没有参数对齐的 dense 臂**（`full_matched` 缺失），该差距因此**含未剥离的容量效应**，不能作为机制性结论的依据。', f'- RoPE 面板：`csa_fixed_rope` {_w_rope} `full_rope` **{abs(g_rope):.2f}** PPL{rope_unmatched}。', f'- RoPE 使 dense {_d_abs_t}、sparse {_d_sp_t}{_gain_note}。', '']
        if rope_unmatched and rope_is_defect and _has_mt:
            lines += [f'> **本表的限制（务必先读）**：RoPE 面板里**没有**参数对齐的 dense 基线——落盘记录显示 `full_rope` 的 MLP 停在标准宽度（{_params_m(_fu_r)}），而 `csa_fixed_rope` 保持全宽（{_params_m(_cf_r)}），{f'相差 {100 * _pgap_r:.1f}%' if _pgap_r is not None else '参数差未知'}。对照 absPE 面板，同样的容量差异会贡献约 {g_abs - g_abs_m:.1f} PPL。因此**上面 RoPE 的组间差距不能与 absPE 的组间差距直接相减**，「差距几乎不变」的读法在当前产物上不成立。需**重跑 P0R 面板**（得到参数对齐的 dense 臂）才能给出该结论；在那之前，**P0-3 的证伪只由 absPE 面板的参数对齐数字支持**。', '']
        _arms_gain = d_abs > 0 and d_sp > 0
        _gain_txt = 'RoPE+QK-norm 对两臂都有真实增益，值得保留为新默认' if _arms_gain else f'RoPE+QK-norm 并未对两臂都带来增益（dense {_d_abs_t}、sparse {_d_sp_t}）'
        _pair_mt = None
        if _has_mt:
            _ps_ab = per_seed_records('results_lm_v3_1500')
            _pair_mt = paired_row(_ps_ab.get('csa_fixed', {}), _ps_ab.get('full_matched', {}))
        if _has_mt:
            if _pair_mt is not None:
                _st_mt = _pair_mt[2]
                _mt_stat = f'配对符号翻转 n={_st_mt['n']}、p={_st_mt['p_exact_signflip']:.3f}（Δ mean={_st_mt['mean']:+.2f}）'
                _mt_sig = _st_mt['p_exact_signflip'] <= 0.05
                _mt_gap = _st_mt['mean']
            else:
                _mt_stat = '本面板没有可通过配对门禁的同配置种子对，显著性无法检验'
                _mt_sig = False
                _mt_gap = g_abs_m
            if _mt_gap > 0:
                if _mt_sig:
                    _p0r_concl = f'**结论**：PE 错配 **不是** sparse 劣势的来源。（1）{_gain_txt}；（2）参数对齐的 absPE 数字（`csa_fixed` 对 `full_matched`）仍显著落后（{_mt_stat}），**P0-3 的混淆假设被证伪**，反而**强化**了论文的负结果——sparse 在该训练量下的落后是机制性的。'
                else:
                    _p0r_concl = f'**结论**：PE 错配 **不是** sparse 劣势的来源。（1）{_gain_txt}；（2）参数对齐的 absPE 数字（`csa_fixed` 对 `full_matched`）方向仍为落后（{_mt_stat}），**P0-3 的混淆假设被证伪**——sparse 在该训练量下的落后是机制性的；其显著性按配对检验的实际分辨率表述，不作超出分辨率的显著性主张。'
            else:
                _p0r_concl = f'**结论**：{_gain_txt}；但参数对齐的 absPE 数字中 `csa_fixed` 不再落后 `full_matched`（Δ={_mt_gap:+.2f} PPL，{_mt_stat}）——与原论断方向相反，**P0-3 的混淆假设在本轮未被证伪**，PE 错配不能排除在 sparse 的差距之外。'
        else:
            _p0r_concl = f'**结论（受限）**：`csa_fixed` 在 absPE 面板{_w_abs}未匹配的 `full`，{_gain_txt}。但**参数对齐的 absPE 臂（`full_matched`）在本面板缺失**，所以「容量差贡献了多少」无法剥离，**P0-3 的证伪在本轮没有证据支持**——本条只作方向性表述，须待 `full_matched` 产出后方可作结论。'
        lines += [_p0r_concl, '']
    return '\n'.join(lines) + '\n'

def sec_p0w():
    s = summary('results_lm_v7_warmup')
    if not s:
        return '## P0-1 dense→sparse warmup — （无结果）\n\n'
    _legacy_ok = _legacy_warm_tags(s)
    rows = []
    for k, r in s.items():
        if not isinstance(r, dict):
            continue
        _w = _int_or(r.get('warm_steps'), None)
        if _w is None:
            _t = _tag_of(r, k, _legacy_ok)
            _w = int(_t[1:]) if _t.startswith('w') and _t[1:].isdigit() else -1
        _sd = _int_or(r.get('seed'), -1)
        if _sd < 0 and _measurable({'ppl': r.get('ppl'), 'synthesized': r.get('synthesized', False)}):
            print(f"[report] sec_p0w: record {k!r} carries a non-integer seed — it is skipped, not counted")
            continue
        if _w < 0 and _measurable({'ppl': r.get('ppl'), 'synthesized': r.get('synthesized', False)}):
            print(f"[report] sec_p0w: record {k!r} carries no parseable warmup tag — it is skipped, not counted")
            continue
        rows.append((r.get('variant', '?'), _w, _sd, r.get('ppl'), r.get('ppl_at_switch'), _mins(r.get('train_time_s')), r.get('synthesized', False), k))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    rows = [row for row in rows if _measurable({'ppl': row[3], 'synthesized': row[6]})]
    rows = _prefer_one_cfg(rows, lambda row: ((row[0], row[1]), (s.get(row[-1]) or {}).get('run_cfg')), who='sec_p0w')
    _kept_keys = {row[-1] for row in rows}
    rec_by_s = collections.defaultdict(dict)
    for _k, _r in s.items():
        if _k not in _kept_keys:
            continue
        if not (isinstance(_r, dict) and _measurable(_r)):
            continue
        _sd = _int_or(_r.get('seed'), -1)
        if _sd < 0:
            continue
        _w = _int_or(_r.get('warm_steps'), None)
        if _w is None:
            _t = _tag_of(_r, _k, _legacy_ok)
            _w = int(_t[1:]) if _t.startswith('w') and _t[1:].isdigit() else -1
        _slot = rec_by_s[_r.get('variant', '?'), _w]
        if _sd in _slot:
            print(f"[report] sec_p0w: ambiguous (variant, warm_steps, seed) = ({_r.get('variant', '?')!r}, {_w}, {_sd}): two measurable records found; keeping the FIRST (key {_k!r} ignored)")
            continue
        _slot[_sd] = _r
    grp = collections.defaultdict(list)
    for (v, w), _m in rec_by_s.items():
        for _sd, _r in sorted(_m.items()):
            grp[v, w].append((_r.get('ppl'), _r.get('ppl_at_switch'), _mins(_r.get('train_time_s')), _sd))
    lines = ['## P0-1 dense→sparse warmup 是否能让 sparse 追平 dense？（评审「必做」项）', '', '**评审论断**：论文 §4.2.2 先训 1T token 的 dense 再在 seq 64K 切 sparse；仓库所有 sparse run 都是 from-scratch，故「渐近线更差」可能是没 warmup 的产物。不给这个实验，「渐近线反转」立不住。', '', '**做法**：`train_warmup` 用**相同全长度 cosine LR**，前 `warm_steps` 步走 dense 前向（复用各 csa/hca 层自己的 `W_kvhead` per-token KV 路径 + 同一 sink，不引入新参数），到点切换为 sparse 并记录切换瞬间 PPL。网格 `warm ∈ {0,5000,10000}`（每格 seed 数以下表 n 列为准），20k 步、seq 512。**两臂参数/步数/token 数/初始 PPL/LR 曲线完全相同**，唯一差异是切点。', '', '| variant | warm_steps | PPL (mean±std) | n | 切换时 PPL |', '|---|---|---|---|---|']
    for (v, w), vals in sorted(grp.items()):
        ppls = [p for p, _s, _m, _sd in vals]
        mean = sum(ppls) / len(ppls)
        std = sample_std(ppls)
        sws = [s for _p, s, _m, _sd in vals if _ppl_ok(s)]
        sw_s = f'{sum(sws) / len(sws):.2f}' if sws else '—'
        lines.append(f'| `{v}` | {w} | {mean:.2f} ± {std:.2f} | {len(ppls)} | {sw_s} |')
    base = grp.get(('csa_fixed', 0))
    if base and len(base) >= 1:
        b = sum((p for p, _s, _m, _sd in base)) / len(base)
        lines += ['', f'**from-scratch 基线**（warm=0）：{b:.2f} PPL。相对基线的变化：', '']
        for (v, w), vals in sorted(grp.items()):
            if v != 'csa_fixed' or w == 0:
                continue
            m = sum((p for p, _s, _m, _sd in vals)) / len(vals)
            lines.append(f'- `{v}` warm={w}：{m:.2f} PPL（{m - b:+.2f} vs 基线，n={len(vals)}）')
        curves = collections.defaultdict(lambda: collections.defaultdict(list))
        _dropped_pts = 0
        for (v, w), _m in rec_by_s.items():
            for _sd, r in sorted(_m.items()):
                for step, pv in _hist_points(r):
                    if not _ppl_ok(pv):
                        _dropped_pts += 1
                        continue
                    curves[v, w][int(step)].append(float(pv))
        if _dropped_pts:
            lines += ['', f'> **⚠ 轨迹表剔除了 {_dropped_pts} 个非有限/非正的 PPL 曲线点**（单次失败的 eval），它们不再参与 seed 平均。', '']
        if curves and any((len(c) > 1 for c in curves.values())) and any((_v == 'csa_fixed' for _v, _w in curves)):
            steps_all = sorted({st for c in curves.values() for st in c})
            lines += ['', '**验证 PPL 轨迹（seed 平均）**：', '', '| step | ' + ' | '.join((f'w={w}' for _v, w in sorted(curves) if _v == 'csa_fixed')) + ' |', '|' + '---|' * (1 + len([1 for _v, w in curves if _v == 'csa_fixed']))]
            for st in steps_all:
                cells = []
                for _v, w in sorted(curves):
                    if _v != 'csa_fixed':
                        continue
                    vals = curves[_v, w].get(st)
                    cells.append(f'{sum(vals) / len(vals):.1f}' if vals else '—')
                lines.append(f'| {st} | ' + ' | '.join(cells) + ' |')
        _conc = ['', '**结论**：']
        _parts = []
        _ds = []
        _ds_sig = []
        for v, w in sorted(grp):
            if v != 'csa_fixed' or w == 0:
                continue
            _wm = rec_by_s.get((v, w), {})
            _bm = rec_by_s.get(('csa_fixed', 0), {})
            _shared = sorted(set(_wm) & set(_bm))
            _dd = []
            _gated = 0
            for sd in _shared:
                reason = _pair_reason(_wm[sd], _bm[sd])
                if reason:
                    _gated += 1
                    continue
                _dd.append(_wm[sd]['ppl'] - _bm[sd]['ppl'])
            if _gated:
                print(f'[report] sec_p0w `{v}` warm={w}: {_gated}/{len(_shared)} seed(s) NOT paired (run_cfg/steps gate); excluded from the sign-flip test')
            _st = exact_signflip(_dd) if _dd else None
            if _st is None:
                continue
            _d = _st['mean']
            _floor = 2.0 / (1 << _st['n']) if _st['n'] and _st['n'] <= 20 else 1.0 / 200001.0
            _same_dir = all((x != 0 and (x > 0) == (_d > 0) for x in _dd))
            if _st['p_exact_signflip'] > 0.05:
                _verdict = f'n={_st['n']} 时 p={_st['p_exact_signflip']:.4f}（检验下限 {_floor:.3f}），**不构成显著**，' + ('各 seed 方向一致' if _same_dir else '方向不一致')
            else:
                _verdict = f'n={_st['n']} 时 p={_st['p_exact_signflip']:.4f}，**显著**'
            _parts.append(f'`{v}` warm={w}：{_d:+.2f} PPL vs from-scratch 基线（{_verdict}）')
            _ds.append(_d)
            _ds_sig.append(bool(_st['p_exact_signflip'] <= 0.05))
        if _parts:
            _conc.append('；'.join(_parts) + '。')
        else:
            _conc.append('（该面板无可配对的可测量记录，无法给出结论。）')
        if _ds and all((x > 0 for x in _ds)):
            if any(_ds_sig):
                _conc.append('即 dense 预热**不能**拯救 sparse 臂，过度预热反而有害——论文的 from-scratch 对比是保守的，**P0-1 的质疑被证伪**，负结果更稳固。')
            else:
                _conc.append('各 warm 档方向一致（预热更差），dense 预热在方向上**未能**拯救 sparse 臂，但没有一档达到配对检验的显著性（分辨率见上）——**P0-1 的质疑按方向不成立，按显著性则既不能证伪也不能确认**，须补更多 seeds 后再判。')
        elif _ds and all((x < 0 for x in _ds)):
            if any(_ds_sig):
                _conc.append('即 dense 预热对 sparse 臂**有**真实帮助——**P0-1 的质疑成立**，from-scratch 对比不利于 sparse，相关负结果须按带 warmup 的设定重审。')
            else:
                _conc.append('各 warm 档方向一致（预热更好），dense 预热在方向上对 sparse 臂有帮助，但没有一档达到配对检验的显著性（分辨率见上）——**P0-1 质疑既不能证伪也不能确认**，须补更多 seeds 后再判。')
        elif _ds:
            _conc.append('各 warm 档的方向不一致，P0-1 质疑既不能证伪也不能确认，须补更多 seeds/档位后再判。')
        _conc.append('')
        lines += _conc
    return '\n'.join(lines) + '\n'

def sec_p0e():
    d = agg('results_lm_v7_long40')
    if not d:
        return '## P0-2 渐近线是否反演（40k 长跑）— （无结果）\n\n'
    s = summary('results_lm_v7_long40')
    lines = ['## P0-2 「渐近线更差」是否成立？（延长到 40k 步 / ~246M tokens）', '', '**评审论断**：20k 步时两臂都未平台化、差距在收窄。要么延长，要么改述为「等训练量下收敛更慢」。', '', '**做法**：`csa_fixed` + `full`（seed 数见下表），同 LONG 配置延到 40k 步（seq 512、bs 12、同一 110M token 池、同一 cosine LR 全长度；eval_every=2000 记录全程轨迹）。', '', '| variant | PPL@40k (mean±std) | n | params |', '|---|---|---|---|']
    for v in sorted(d):
        r = d[v]
        lines.append(f'| `{v}` | {fm(r.get('ppl_mean'))} {sp(r.get('ppl_std'))} | {r.get('n_seeds', r.get('n', '?'))} | {_params_m(r)} |')
    _s_items = _prefer_one_cfg([_kv for _kv in s.items() if isinstance(_kv[1], dict) and 'variant' in _kv[1] and _measurable(_kv[1])], lambda kv: (kv[1].get('variant'), kv[1].get('run_cfg')), who='sec_p0e')
    trajs = collections.defaultdict(lambda: collections.defaultdict(dict))
    _dropped_pts = 0
    for _k, r in _s_items:
        if not isinstance(r, dict) or 'variant' not in r:
            continue
        if not _measurable(r):
            continue
        _sdv = str(r.get('seed'))
        for step, pv in _hist_points(r):
            if not _ppl_ok(pv):
                _dropped_pts += 1
                continue
            _cell = trajs[r['variant']][int(step)]
            if _sdv in _cell:
                print(f"[report] sec_p0e: ambiguous (variant, step, seed) = ({r['variant']!r}, {int(step)}, {_sdv}): two measurable points found; keeping the FIRST")
                continue
            _cell[_sdv] = float(pv)
    if _dropped_pts:
        lines += ['', f'> **⚠ 轨迹表剔除了 {_dropped_pts} 个非有限/非正的 PPL 曲线点**（单次失败的 eval），它们不再参与 seed 平均、尾段斜率与下方结论的判定。', '']
    if 'csa_fixed' in trajs and 'full' in trajs:
        steps = sorted(set(trajs['csa_fixed']) & set(trajs['full']))
        if steps:
            lines += ['', '**验证 PPL 轨迹（seed 平均，每 2000 步）**：', '', '| step | csa_fixed | full | gap (csa−full) |', '|---|---|---|---|']
            gaps = {}
            for st in steps:
                c = sum(trajs['csa_fixed'][st].values()) / len(trajs['csa_fixed'][st])
                f_ = sum(trajs['full'][st].values()) / len(trajs['full'][st])
                gaps[st] = c - f_
                lines.append(f'| {st} | {c:.1f} | {f_:.1f} | **{gaps[st]:+.1f}** |')
            tail = steps[-5:] if len(steps) >= 5 else steps
            _tail_span_k = (steps[-1] - tail[0]) / 1000.0
            _paired_raw = sorted(set.intersection(*(set(trajs[v][st]) for v in ('csa_fixed', 'full') for st in tail))) if tail else []
            _all_seeds = sorted(set.union(*(set(trajs[v][st]) for v in ('csa_fixed', 'full') for st in tail))) if tail else []
            _incomplete = [sd for sd in _all_seeds if sd not in set(_paired_raw)]
            if _incomplete:
                print(f'[report] sec_p0e: seed(s) {_incomplete} miss at least one arm x tail-step cell and are excluded from the paired tail gap — pairing requires a measurement on EVERY tail step for BOTH arms')
            _rec_of = {}
            for _k, r in _s_items:
                if isinstance(r, dict):
                    _rec_of.setdefault((r.get('variant'), str(r.get('seed'))), r)
            _paired = []
            _pair_gated = []
            for _sd in _paired_raw:
                _ra = _rec_of.get(('csa_fixed', _sd))
                _rb = _rec_of.get(('full', _sd))
                _why = _pair_reason(_ra, _rb) if _ra is not None and _rb is not None else 'record missing'
                if _why is None:
                    _paired.append(_sd)
                else:
                    _pair_gated.append((_sd, _why))
            if _pair_gated:
                print(f'[report] sec_p0e: {len(_pair_gated)} shared seed(s) FAILED the run_cfg/steps gate ({sorted({w for _s, w in _pair_gated})}) and are excluded from the paired tail-gap — the pairing rules are the same as every other section')
            _unpaired_gap = not _paired
            if _unpaired_gap:
                print('[report] sec_p0e: the two arms share NO seed on the tail evaluation steps — the tail-gap slope and extrapolation below use the UNPAIRED per-step mean gap (each step averaged over whichever seeds measured it), not a paired per-seed difference; read it as indicative only')
            xs = [st / 1000.0 for st in tail]
            if _paired:
                ys = [sum((trajs['csa_fixed'][st][sd] - trajs['full'][st][sd] for sd in _paired)) / len(_paired) for st in tail]
            else:
                ys = [gaps[st] for st in tail]
            g_last = ys[-1]
            n_ = len(xs)
            mx, my = (sum(xs) / n_, sum(ys) / n_)
            slope = sum(((x - mx) * (y - my) for x, y in zip(xs, ys))) / sum(((x - mx) ** 2 for x in xs)) if n_ > 1 else 0.0

            def rate(var):
                if _paired:
                    yv = [sum((trajs[var][st][sd] for sd in _paired)) / len(_paired) for st in tail]
                else:
                    yv = [sum(trajs[var][st].values()) / len(trajs[var][st]) for st in tail]
                my2 = sum(yv) / n_
                return sum(((x - mx) * (y - my2) for x, y in zip(xs, yv))) / sum(((x - mx) ** 2 for x in xs)) if n_ > 1 else 0.0
            r_csa, r_full = (rate('csa_fixed'), rate('full'))

            def _gap_at(step):
                v = gaps.get(step)
                return f'**{v:+.1f}**' if isinstance(v, (int, float)) and math.isfinite(v) else '**未测量**（该面板无此步）'
            _g20 = _gap_at(20000)
            def _dir_word(r):
                return '下降' if r < 0 else ('上升' if r > 0 else '持平')
            _lastk = steps[-1] / 1000.0
            _tps = next((r['tokens_seen'] / r['steps'] for _k, r in _s_items if isinstance(r, dict) and isinstance(r.get('tokens_seen'), (int, float)) and isinstance(r.get('steps'), (int, float)) and r['steps']), None)
            _tok_m = f'{steps[-1] * _tps / 1e6:.0f}M' if _tps else 'token 数未知'
            if _unpaired_gap:
                _gap_calib = '非配对逐步均值差（两臂尾段无公共 seed，仅作方向性参考）'
            elif _pair_gated:
                _gap_calib = f'尾段斜率与终点差距为过门禁配对子集（{len(_paired)}/{len(_paired) + len(_pair_gated)} 个公共 seed；上表逐行仍是逐步全 seed 均值）'
            else:
                _gap_calib = 'seed 平均'
            lines += ['', f'**差距轨迹分析（尾段 {tail[0]:g}–{steps[-1]:g} 步，跨度 {_tail_span_k:g}k，{_gap_calib}）**：', '', f'- 20k 步差距：{_g20} PPL；{_lastk:g}k 步差距：**{g_last:+.1f}** PPL。', f'- 尾段差距斜率：**{slope:+.2f} PPL / 1k steps**（csa {_dir_word(r_csa)} {abs(r_csa):.2f}、dense {_dir_word(r_full)} {abs(r_full):.2f} PPL/1k）。']
            if _unpaired_gap:
                verdict = f'**结论（受限）：本轮不对「渐近线是否反演」作判定。** 两臂在尾段评测步上没有任何通过配置/训练量配对门禁的公共 seed，上方尾段斜率（{slope:+.2f} PPL/1k）与终点差距（{g_last:+.1f} PPL）是**非配对逐步均值差**——每个评测步由各自测得的 seed 平均，seed 组成随步变化，组成变化本身即可伪造或掩盖交叉。需补齐两臂共享 seed 的长跑记录后再按同一判据重算。'
            elif n_ < 2:
                verdict = f'**结论：尾段差距统计量不可用**（两臂只在 {n_} 个公共评测步上有测量，尾段回归需要至少 2 个点），**本轮不给出渐近线读法**——这不是「差距不再收窄」，是**无法判定**。需补跑或加密尾段评测网格。'
            elif not (math.isfinite(g_last) and math.isfinite(slope)):
                verdict = f'**结论：尾段差距统计量不可用**（`g_last` 或 `slope` 非有限值：g_last={g_last!r}、slope={slope!r}），**本轮不给出渐近线读法**——这不是「差距不再收窄」，是**无法判定**。上表已剔除非有限的曲线点；若此处仍出现，说明该面板的尾段整体缺失，需补跑。'
            elif g_last <= 0:
                verdict = f'**结论：渐近线在 {_lastk:g}k 内反演**——`csa_fixed` 追平并超过 dense，评审的 P0-2 质疑成立，此前「worse asymptote」的表述需撤回。'
            elif slope < -1e-06:
                x0 = steps[-1] - g_last * 1000.0 / slope
                if steps[-1] < x0 <= steps[-1] * 10:
                    _g20v = gaps.get(20000)
                    _g20c = f'由 20k 的 **{_g20v:+.1f}** ' if isinstance(_g20v, (int, float)) and math.isfinite(_g20v) else ''
                    verdict = f'- 按尾段斜率线性外推，差距将在 ~{x0 / 1000:.0f}k 步附近归零（外推仅供参考：学习率已 cosine 衰减到底，后期斜率通常进一步放缓）。\n\n**结论**：{_lastk:g}k 步内未发生反演，但差距仍在缓慢收窄。**保守表述**：「等 token 量下 CSA 收敛更慢，终点差距 {_g20c}收窄到 {_lastk:g}k 的 {g_last:+.1f}，未见交叉」。是否最终追平属外推，不属证据。'
                else:
                    verdict = f'- 线性外推的交叉点在 ~{x0 / 1000:.0f}k 步，超出可信外推范围。\n\n**结论**：{_lastk:g}k 步（~{_tok_m} tokens）仍未追平——终点差距 **{g_last:+.1f} PPL**。差距收窄速度在尾段为 {abs(slope):.2f} PPL/1k，即每多训 10k 步约收窄 {abs(slope) * 10:.0f} PPL。**「渐近线更差」在实验可达范围内成立**，更精确的措辞是「等训练量下收敛更慢且差距长期存在」。'
            else:
                verdict = f'**结论**：尾段差距已不再收窄（斜率 {slope:+.2f} PPL/1k），**「渐近线更差」在 {_lastk:g}k 步坐实**。'
            lines += ['', verdict, '']
    return '\n'.join(lines) + '\n'

def sec_p1t():
    d = agg('results_lm_v7_seq2k')
    if not d:
        return '## P1 选择率错配 / m=1 对照（seq 2048 topk 扫描）— （无结果）\n\n'
    order = ['csa_fix_m1', 'csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fixed']
    lines = ['## P1 选择率错配 与 m=1(纯 DSA) 对照（seq 2048, topk 扫描）', '', '**评审论断**：仓库 seq 512 时选择率 25%，论文在长序列约 0.2%。`topk` 是主控旋钮，需要扫描；且缺 m=1（纯 DSA，不压缩）对照。', '', '**做法**：seq 2048、1500 步（seed 数见下表），扫描 `topk ∈ {8,32,128,512}` 并加入 `csa_fix_m1`（block_size=1, overlap=0 → 逐 token 选择 = 纯 DSA）。', '', '| variant | PPL (mean±std) | n | 等效选择率@2048 (topk/512 blocks) |', '|---|---|---|---|']
    seen = set()
    for v in order + sorted(d):
        if v in seen or v not in d:
            continue
        seen.add(v)
        r = d[v]
        ppls = _finite_ppls(r)
        if ppls:
            mean = sum(ppls) / len(ppls)
            std = sample_std(ppls)
            n = len(ppls)
        elif r.get('ppls') or r.get('ppl_list'):
            mean, std, n = (None, None, 0)
        else:
            mean, std, n = (r.get('ppl_mean'), r.get('ppl_std'), r.get('n_seeds') if r.get('n_seeds') is not None else r.get('n', 0))
        _bv = v.split('@')[0].split('#')[0]
        if _bv == 'csa_fix_m1':
            sr = '1.5625% (m=1, topk=32)'
        elif _bv.startswith('csa_fixed_topk'):
            _kv = _bv.replace('csa_fixed_topk', '')
            try:
                k = int(_kv)
            except ValueError:
                print(f'[report] sec_p1t: cannot parse topk from aggregate key {v!r} — row omitted')
                continue
            sr = f'{min(k, 512) / 512 * 100:.5g}%'
        elif _bv == 'csa_fixed':
            sr = '6.25% (default topk=32)'
        else:
            sr = '—'
            print(f'[report] sec_p1t: aggregate key {v!r} is not one of the known sweep variants — its selection-rate cell is left blank rather than guessed')
        lines.append(f'| `{v}` | {fm(mean)} ± {fm(std)} | {n} | {sr} |')
    s = summary('results_lm_v7_seq2k')
    failed = {}
    _measured = set()
    for rid, r in s.items():
        if not isinstance(r, dict):
            continue
        if r.get('error'):
            v = r.get('variant', rid.split('::')[0])
            failed.setdefault(v, []).append(r.get('seed'))
        elif _measurable(r):
            _measured.add(r.get('variant', rid.split('::')[0]))
    _grid = ['csa_fix_m1', 'csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512']
    missing = [v for v in _grid if v not in _measured and v not in failed]
    for v in missing:
        print(f'[report] sec_p1t: sweep point `{v}` has NO record in results_lm_v7_seq2k (never ran, cap-skipped, or truncated) — it is absent from the table and is listed below')
    if failed:
        lines += ['', '**未完成的扫描点（如实记录）**：', '']
        for v in sorted(failed):
            seeds = ','.join((str(x) for x in sorted(failed[v], key=lambda x: (x is None, 0 if isinstance(x, (int, float)) and (not isinstance(x, bool)) else 1, x if isinstance(x, (int, float)) and (not isinstance(x, bool)) else str(x)))))
            lines.append(f'- `{v}`（seed {seeds}）：该扫描点在本配置与运行时长约束下未完成（记录为 error），扫描被截断。')
        lines.append('')
    if missing:
        lines += ['', '**无记录的扫描点（如实记录）**：', '']
        for v in missing:
            lines.append(f'- `{v}`：面板上没有任何记录（未运行、被时长门禁跳过或截断后未留下测量），扫描不完整。')
        lines.append('')
    ks = [8, 32, 128, 512]
    rows_k = []
    for k in ks:
        r = _agg_entry(d, f'csa_fixed_topk{k}')
        if r is not None:
            ppls = _finite_ppls(r)
            m = sum(ppls) / len(ppls) if ppls else _finite_or_none(r.get('ppl_mean'))
            if isinstance(m, (int, float)):
                rows_k.append((k, m))
    if len(rows_k) >= 3:
        lines += ['', '**剂量-反应（topk → PPL）**：', '']
        mono = all((rows_k[i][1] >= rows_k[i + 1][1] - 1e-09 for i in range(len(rows_k) - 1)))
        for k, m in rows_k:
            lines.append(f'- topk={k}（选择率 {k / 512 * 100:.5g}%）：{m:.2f} PPL')
        lines.append('')
        if mono:
            lines.append('PPL 随选择率单调下降（选择越多越好），说明在 seq 2048 / 1500 步的受控训练量下未出现「少选反而优」的甜点——检索质量不足时，选择率是硬上限。')
        else:
            lines.append('存在非单调点：存在一个「少选反而更好/更差」的转折（见上表），提示 topk 存在训练量相关的最优值。')
        lines.append('')
    m1 = _agg_entry(d, 'csa_fix_m1')
    t8 = _agg_entry(d, 'csa_fixed_topk8')
    t32 = _agg_entry(d, 'csa_fixed_topk32')

    def _mean(r):
        if not r:
            return None
        ppls = _finite_ppls(r)
        return sum(ppls) / len(ppls) if ppls else _finite_or_none(r.get('ppl_mean'))
    m1m, t8m, t32m = (_mean(m1), _mean(t8), _mean(t32))

    lines += ['', '**结论（基于已完成的扫描点）**：', '']
    _m1_ok = m1m is not None
    _t_ok = t8m is not None and t32m is not None
    if _m1_ok and _t_ok:
        _recs3 = per_seed_records('results_lm_v7_seq2k')
        _m1_r, _t8_r, _t32_r = (_recs3.get('csa_fix_m1', {}), _recs3.get('csa_fixed_topk8', {}), _recs3.get('csa_fixed_topk32', {}))
        _c3 = []
        _rej3 = 0
        for sk in sorted(set(_m1_r) & set(_t8_r) & set(_t32_r)):
            if _pair_reason(_m1_r[sk], _t8_r[sk]) is not None or _pair_reason(_m1_r[sk], _t32_r[sk]) is not None:
                _rej3 += 1
                continue
            _c3.append(sk)
        if _rej3:
            print(f'[report] sec_p1t: {_rej3} three-way shared seed(s) rejected by the run_cfg/steps gate — excluded from the per-seed direction count')
        _m1_s = {sk: float(_m1_r[sk]['ppl']) for sk in _c3}
        _t8_s = {sk: float(_t8_r[sk]['ppl']) for sk in _c3}
        _t32_s = {sk: float(_t32_r[sk]['ppl']) for sk in _c3}
        if _c3:
            _nw = sum((1 for sk in _c3 if _m1_s[sk] > _t8_s[sk] and (_m1_s[sk] > _t32_s[sk])))
            _nb = sum((1 for sk in _c3 if _m1_s[sk] < _t8_s[sk] and (_m1_s[sk] < _t32_s[sk])))
            _dir3 = f'（逐 seed：{len(_c3)} 个通过门禁的三方公共 seed 中，{_nw} 个 m1 最差、{_nb} 个 m1 最好）'
        else:
            _dir3 = '（无通过门禁的三方公共 seed，逐 seed 方向不可核验）'
        if m1m > t8m and m1m > t32m:
            lines.append(f'1. **m=1（纯 DSA、不压缩）是三点中最差的**（{fm(m1m)} vs topk8 {fm(t8m)} / topk32 {fm(t32m)}{_dir3}）：去掉压缩并没有拯救 sparse 臂——在 seq 2048 / 1500 步的受控训练量下，**压缩不是瓶颈**，评审「缺 m=1 对照」的质疑得到直接回答（方向与整体负结果一致）。')
        elif m1m < t8m and m1m < t32m:
            lines.append(f'1. **m=1（纯 DSA、不压缩）是三点中最好的**（{fm(m1m)} vs topk8 {fm(t8m)} / topk32 {fm(t32m)}{_dir3}）：去掉压缩反而占优——在 seq 2048 / 1500 步的受控训练量下，**压缩是当前的瓶颈之一**，评审「缺 m=1 对照」的质疑得到直接回答。')
        else:
            lines.append(f'1. m=1（{fm(m1m)}）介于 topk8（{fm(t8m)}）与 topk32（{fm(t32m)}）之间，三点排序非单调{_dir3}；评审「缺 m=1 对照」的质疑得到直接回答，但压缩是否瓶颈需结合显著性判断，此处只作方向性表述。')
    elif _m1_ok:
        lines.append(f'1. m=1 测得 {fm(m1m)}，但 topk8/topk32 至少其一缺失，三点排序不可比——本条只陈述事实，不作「压缩是否瓶颈」的读法。')
    else:
        lines.append('1. m=1 未测量（该扫描点缺失或失败），三点排序不可比——本条只陈述事实，不作「压缩是否瓶颈」的读法。')
    if _t_ok:
        if t8m <= t32m:
            _dir = f'topk8 {fm(t8m)} ≤ topk32 {fm(t32m)}：选得更少反而略好，与论文「长序列下低选择率足够」的设计方向一致'
        else:
            _dir = f'topk8 {fm(t8m)} > topk32 {fm(t32m)}：选得更少反而更差，整体为单调上升（选择越多越好）'
        _gap = abs(t32m - t8m)
        _recs = per_seed_records('results_lm_v7_seq2k')
        _pair = paired_row(_recs.get('csa_fixed_topk8', {}), _recs.get('csa_fixed_topk32', {}))
        if _pair is not None:
            _kept, _dl, _st = _pair
            _sig_txt = '显示该差距**达到 p≤0.05**，按显著性读法处理。' if _st['p_exact_signflip'] <= 0.05 else '显示该差距仍在噪声量级内，**不构成显著性主张**，只作方向性参考。'
            lines.append(f'2. 在已完成的两个选择率点上：{_dir}；配对的精确符号翻转检验（n={_st['n']}，p={_st['p_exact_signflip']:.3f}，Δ(topk8−topk32) mean={_st['mean']:+.2f} PPL）{_sig_txt}')
        else:
            lines.append(f'2. 在已完成的两个选择率点上：{_dir}；面板上没有可通过 pairing 门禁的同配置种子对，topk8 与 topk32 之差（{_gap:.2f} PPL）无法配对检验，只作方向性参考。')
    else:
        lines.append('2. topk8/topk32 至少其一缺失，两个选择率点之间的对比不可算。')
    if failed or missing:
        _miss = '、'.join((f'`{v}`' for v in sorted(set(failed) | set(missing))))
        lines.append(f'3. 未完成的点（{_miss}，见上）需后续在完整扫描面板补齐，「甜点位置」的完整刻画以补齐后的面板为准。')
    else:
        lines.append('3. 全部扫描点均已完成，「甜点位置」以本面板数据为准。')
    return '\n'.join(lines) + '\n'

def sec_p1l():
    s = summary('results_len')
    if not s:
        return '## P1 长上下文长度外推（train@512 → eval 512..4096）— （无结果）\n\n> 注：合成 NIAH 检索探针已实现并调试（碰撞无关布局、可达标签），但在本受控规模（d=256 / 6 层 / seq 512）下 absPE 与 RoPE 两臂均无法学到高于随机的检索率，探针面板无信息量，故以 wikitext长度外推替代（见报告头部说明）。\n\n'
    recs = [r for r in s.values() if isinstance(r, dict) and 'by_len' in r and (not r.get('synthesized')) and any((isinstance(c, dict) and _ppl_ok(c.get('ppl')) for c in r['by_len'].values()))]
    _by_var = {}
    for r in recs:
        _by_var.setdefault(r.get('variant'), []).append(r)
    recs = []
    _stale_vars = set()
    for _v, _rv in _by_var.items():
        _cur = [r for r in _rv if r.get('_code') == _CUR_CS]
        if _cur:
            if len(_cur) < len(_rv):
                print(f'[report] sec_p1l: {_v}: {len(_rv) - len(_cur)} by-length record(s) predate the current code semantics — the table uses only the {len(_cur)} current one(s)')
            recs.extend(_cur)
        else:
            print(f'[report] sec_p1l: {_v}: NONE of the {len(_rv)} by-length record(s) carries the current code-semantics stamp {_CUR_CS!r} — quoting STALE values for lack of a current one; re-run the length panel to refresh')
            _stale_vars.add(_v)
            recs.extend(_rv)
    if not recs:
        return '## P1 长上下文长度外推 — （无结果）\n\n'
    def _len_key(Ln):
        try:
            return int(Ln)
        except (TypeError, ValueError):
            return None
    lens = sorted({_k for _k in (_len_key(Ln) for r in recs for Ln in r['by_len']) if _k is not None})
    variants = sorted({r.get('variant') for r in recs if r.get('variant') is not None})
    _seed_key = lambda x: (0, float(x), '') if isinstance(x, (int, float)) and (not isinstance(x, bool)) else (1, 0.0, str(x))
    _seeds_by_v = {v: sorted({r.get('seed') for r in recs if r.get('variant') == v and r.get('seed') is not None}, key=_seed_key) for v in variants}
    lines = ['## P1 长上下文长度外推（train@512 → 同一权重 eval 512/1024/2048/4096）', '', '**评审论断**：仓库所有评测都在训练长度（512）上，没有任何长上下文证据。', '', '**做法**：4 变体（`full` / `csa_fixed` / `full_rope` / `csa_fixed_rope`）在 seq 512 训 3000 步（同一代码路径，各臂 seed 数见表），保存权重后用**同一份权重**在 512/1024/2048/4096 上评 wikitext PPL。absPE 变体 `max_seq=512`、超出部分只取末尾 `max_pos` 个 token 截断评测——**这正是 P0-3 的对照点**；RoPE 变体可原生外推。', '', '> **`~` = 位置受限（abs-PE）：该格只评了最后 `max_pos` 个 token，不是该长度的真实长上下文分数**；未标注的格是完整 `Ln` 长度评测。', '', '| variant | ' + ' | '.join((f'PPL@{Ln}' for Ln in lens)) + ' | 相对退化 ratio@{0}/@{1} |'.format(lens[-1], lens[0]), '|' + '---|' * (len(lens) + 2)]
    per_v = {}
    trunc_v = {}
    for v in variants:
        cells = []
        base_map, last_map = ({}, {})
        trunc_flags = []
        for Ln in lens:
            recs_l = []
            _seen_sd = set()
            for _r in recs:
                if _r.get('variant') != v:
                    continue
                _bl = _r.get('by_len') or {}
                _c = _bl.get(Ln, _bl.get(str(Ln)))
                if isinstance(_c, dict) and _ppl_ok(_c.get('ppl')):
                    _sd = _r.get('seed')
                    if _sd in _seen_sd:
                        print(f'[report] sec_p1l: ambiguous (variant, seed) = ({v!r}, {_sd}) at L{Ln}: two measurable records found; keeping the FIRST')
                        continue
                    _seen_sd.add(_sd)
                    recs_l.append((_sd, _c))
            ok = [x for _sd, x in recs_l if not x.get('truncated')]
            is_tr = bool(recs_l) and len(ok) < len(recs_l)
            vals = [float(x['ppl']) for x in (ok if ok else [x for _sd, x in recs_l])]
            trunc_flags.append(is_tr)
            mean = sum(vals) / len(vals) if vals else None
            if mean is None:
                cells.append('—')
            else:
                cells.append(f'~{mean:.1f}' if is_tr else f'{mean:.1f}')
            if recs_l and not is_tr:
                _tgt = base_map if Ln == lens[0] else last_map if Ln == lens[-1] else None
                if _tgt is not None:
                    for _sd, x in recs_l:
                        _tgt[str(_sd)] = float(x['ppl'])
        _matched = sorted(set(base_map) & set(last_map))
        ratio = sum((last_map[sd] / base_map[sd] for sd in _matched)) / len(_matched) if _matched and len(lens) >= 2 else None
        per_v[v] = ratio
        trunc_v[v] = trunc_flags
        rt_txt = f'×{ratio:.2f}' if _finite_or_none(ratio) is not None else '—'
        _vmark = '，⚠ 陈旧' if v in _stale_vars else ''
        lines.append(f'| `{v}`（n={len(_seeds_by_v.get(v, []))}{_vmark}） | ' + ' | '.join(cells) + f' | {rt_txt} |')
    if _stale_vars:
        lines += ['', '> **⚠ 陈旧数据**：' + '、'.join((f'`{v}`' for v in sorted(_stale_vars))) + f' 的按长度记录全部早于当前代码语义戳 `{_CUR_CS}`——这些数值由旧版代码测得，仅因缺少当前测量而列出，重跑长度面板后才会刷新；涉及它们的结论须按陈旧数据处理。', '']
    _any_trunc = any((any(f) for f in trunc_v.values()))
    if _any_trunc:
        lines += ['', '**位置受限（截断）臂**：' + '、'.join((f'`{v}`' for v, f in sorted(trunc_v.items()) if any(f))) + ' —— 其 span > max_pos 的格全部只覆盖最后 `max_pos` 个 token，与真正的长上下文分数不可比；比较外推能力时须以 RoPE 臂（无位置上限）为准。', '']
    if per_v:
        lines += ['', '**读法（相对退化，越低越好）**：', '']
        for v, rt in sorted(per_v.items(), key=lambda kv: kv[1] if _finite_or_none(kv[1]) is not None else 1000000000.0):
            if _finite_or_none(rt) is None:
                continue
            lines.append(f'- `{v}`：{lens[-1]}/{lens[0]} = ×{rt:.2f}')
        _rope_ok = 'csa_fixed_rope' in per_v and 'full_rope' in per_v and _finite_or_none(per_v['csa_fixed_rope']) is not None and _finite_or_none(per_v['full_rope']) is not None and (not (trunc_v.get('csa_fixed_rope', [0])[-1] or trunc_v.get('full_rope', [0])[-1]))
        if _rope_ok:
            _ra = per_v['csa_fixed_rope']
            _rb = per_v['full_rope']
            dv = _ra - _rb
            _ptag = ''
            _rl = {}
            for _r in recs:
                if _r.get('variant') in ('csa_fixed_rope', 'full_rope') and isinstance(_r.get('params'), (int, float)):
                    _rl.setdefault(_r.get('variant'), set()).add(_r['params'])
            _ok = set(_rl) == {'csa_fixed_rope', 'full_rope'} and all((len(v) == 1 for v in _rl.values()))
            if _ok:
                _ptag = unmatched_tag({'params': next(iter(_rl['csa_fixed_rope']))}, {'params': next(iter(_rl['full_rope']))})
            elif _rl:
                _ptag = ' (⚠ params 各 seed 不一致，无法判定是否对齐)'
            _tail = f'（`csa_fixed_rope` ×{_ra:.2f} vs `full_rope` ×{_rb:.2f}，差值 {abs(dv):.2f} 个比值单位{_ptag}）'
            if abs(dv) < 0.05:
                note = 'sparse 与 dense 的长度外推退化**基本相同**——在该受控规模，块级稀疏未额外损害长度外推。' + _tail
            elif dv > 0:
                note = f'`csa_fixed_rope` 的长度外推退化比 `full_rope` 多 {abs(dv):.2f} 个比值单位——压缩 KV / 块级检索在超出训练长度后损失放大，这与「长距依赖检索是 CSA 软肋」的机制一致。' + _tail
            else:
                note = f'`csa_fixed_rope` 的长度外推退化低于 `full_rope` {abs(dv):.2f} 个比值单位——稀疏掩码滤掉了远距噪声，外推更稳。' + _tail
            lines += ['', f'**结论**：{note}', '']
        elif 'csa_fixed_rope' in per_v and 'full_rope' in per_v:
            _ra_t = f"×{per_v['csa_fixed_rope']:.2f}" if _finite_or_none(per_v.get('csa_fixed_rope')) is not None else '—'
            _rb_t = f"×{per_v['full_rope']:.2f}" if _finite_or_none(per_v.get('full_rope')) is not None else '—'
            lines += ['', f'**结论**：外推对比只看两个 RoPE 臂（`csa_fixed_rope` {_ra_t} vs `full_rope` {_rb_t}）——absPE 臂在 `max_pos` 之外没有可比的读数，**不参与**该对比。', '']
        lines.append('')
    lines.append('图：`results_len/length_gen.png`（PPL 与相对比值 vs 评测长度）。')
    lines.append('')
    return '\n'.join(lines) + '\n'

def _agg_entry(d, v):
    if v in d:
        return d[v]
    cands = [(k, e) for k, e in d.items() if isinstance(e, dict) and (k.startswith(f'{v}@cfg') or k.startswith(f'{v}#'))]
    if not cands:
        return None

    def _is_cur(e):
        rc = e.get('run_cfg')
        return isinstance(rc, str) and bool(_CUR_CS) and rc.endswith(f'_cs{_CUR_CS}')
    if any((_is_cur(e) for _k, e in cands)):
        cands.sort(key=lambda kv: (not _is_cur(kv[1]), -(_int_or(kv[1].get('n_seeds'), 0))))
        _why = 'the one stamped with the current code semantics'
    else:
        cands.sort(key=lambda kv: -(_int_or(kv[1].get('n_seeds'), 0)))
        _why = 'the largest-n one (no group carries a current-code run_cfg stamp — re-run `aggregate` to stamp them)'
    if len(cands) > 1:
        print(f'[report] _agg_entry: `{v}` spans {len(cands)} cfg/protocol groups ({sorted((k for k, _e in cands))}) — quoting {_why} ({cands[0][0]}, n_seeds={cands[0][1].get('n_seeds')})')
    return cands[0][1]

def _params_m(r):
    p = r.get('params') if isinstance(r, dict) else None
    if not isinstance(p, (int, float)) or isinstance(p, bool) or (not p):
        return '—'
    return f'{p / 1000000.0:.3f}M'

def sec_p2s():
    d = agg('results_lm_v5_scale')
    if not d:
        return ''
    lines = ['## P2 v5 scale 面板补种子（至 4 seeds）+ 配对显著性', '', '**评审论断**：v5 scale 只有 2 seeds 且不显著，需 4 seeds + `full_matched`。', '', '| variant | PPL (mean±std) | n | params |', '|---|---|---|---|']
    for v in sorted(d):
        r = d[v]
        lines.append(f'| `{v}` | {fm(r.get('ppl_mean'))} {sp(r.get('ppl_std'))} | {r.get('n_seeds', r.get('n', '?'))} | {_params_m(r)} |')
    ps = per_seed_records('results_lm_v5_scale')
    pairs = [('csa_fixed', 'full'), ('csa_fixed', 'full_sw128_matched'), ('csa_fixed', 'full_matched'), ('csa_dynamic', 'full'), ('hybrid_dynamic', 'full')]
    _agg = d
    rows = []
    for a, b in pairs:
        if a in ps and b in ps:
            try:
                pr = paired_row(ps[a], ps[b])
            except Exception as e:
                print(f'[report] paired_row({a}, {b}) skipped: {type(e).__name__}: {e}')
                continue
            if pr:
                common, dl, st = pr
                rows.append((a, b, common, st))
    if rows:
        lines += ['', '**配对检验（同 seed 配对，精确符号翻转）**：', '', '| 比较 | n (配对 seeds) | Δ (a−b) mean±std | p (exact) | params |', '|---|---|---|---|---|']
        for a, b, common, st in rows:
            lines.append(f'| `{a}` − `{b}` | {len(common)} | {st['mean']:+.2f} ± {st['std']:.2f} | {st['p_exact_signflip']:.3f} | {unmatched_tag(_agg_entry(_agg, a), _agg_entry(_agg, b)) or '✓'} |')
        main = next((r for r in rows if r[0] == 'csa_fixed' and r[1] == 'full'), None)
        _mt = next((r for r in rows if r[0] == 'csa_fixed' and r[1] == 'full_matched'), None) if _agg_entry(_agg, 'full_matched') is not None else None
        _base = _mt if _mt else main
        if _base:
            _ba, _bb, common, st = _base
            _which = '`csa_fixed − full_matched`（参数对齐）' if _mt else '`csa_fixed − full`（未做容量匹配）'
            _tag = unmatched_tag(_agg_entry(_agg, _ba), _agg_entry(_agg, _bb)) if _mt else unmatched_tag(_agg_entry(_agg, 'csa_fixed'), _agg_entry(_agg, 'full'))
            if st['mean'] > 0:
                concl = f'scale 面板上 {_which} 的配对差为 **{st['mean']:+.2f} PPL**（n={len(common)}，p={st['p_exact_signflip']:.3f}）：短训练下的 sparse 早期优势在 d=384 / 8 层 / seq 1024 的规模上**已经消失并反转为劣势**——规模越大，dense 的容量红利显现越早，与 20k/40k 长跑的「渐近线反转」同一方向。{_tag or '✓'}'
            else:
                concl = f'scale 面板上 {_which} 的配对差为 **{st['mean']:+.2f} PPL**（n={len(common)}，p={st['p_exact_signflip']:.3f}）：该规模下 sparse 仍领先，早期优势未随规模消失。{_tag or '✓'}'
            lines += ['', f'**结论**：{concl}', '']
    return '\n'.join(lines) + '\n'

def sec_flops():
    d = load(os.path.join(REPO, 'analysis_v7', 'flops_analytic.json'))
    if not d:
        return '## 解析 FLOPs / KV-cache — （无结果）\n\n'
    cfg = d.get('config', {})
    rows = d.get('rows', [])
    pick = {512, 2048, 8192, 65536, 1048576}
    _xo = d.get('crossover_seq_len_csa_beats_dense')
    _xo_txt = f'**seq = {_xo}**' if _xo is not None else '**不存在**（所测长度内 CSA 均不低于 dense）'
    lines = ['## 解析 FLOPs / KV-cache 与交叉点', '', f'**配置**：{cfg}。**方法**：论文只给相对百分比、无闭式，故采用与硬件无关的解析 FLOPs / KV-cache 进行对比。', '', f'**交叉点**：CSA 的每 token 注意力 FLOPs 在 {_xo_txt} 处开始低于 dense。', '', '| seq | 选择率 | CSA/dense FLOPs | hybrid/dense | CSA KV/dense |', '|---|---|---|---|---|']
    for r in rows:
        if not isinstance(r, dict) or r.get('seq_len') not in pick:
            continue
        try:
            lines.append(f'| {r['seq_len']} | {r['sel_ratio']:.2%} | {r['csa_over_dense']:.3f} | {r['hybrid_over_dense']:.3f} | {r['csa_kv_over_dense']:.3f} |')
        except (KeyError, TypeError, ValueError) as _e:
            print(f'[report] sec_flops: a malformed row was skipped ({type(_e).__name__}: {_e})')
    _clean = sorted((r for r in rows if isinstance(r, dict) and isinstance(r.get('seq_len'), (int, float)) and (not isinstance(r.get('seq_len'), bool)) and math.isfinite(r['seq_len'])), key=lambda r: r['seq_len'])
    _rows = {}
    for r in _clean:
        _rows.setdefault(r['seq_len'], r)
    _x = d.get('crossover_seq_len_csa_beats_dense')
    _r512 = _rows.get(512)
    if _r512 is not None and _x is not None:
        _ratio = _r512['csa_over_dense']
        _sel = _r512['sel_ratio']
        _verdict = '省' if _ratio < 1.0 else '不省'
        _cross = f'交叉点本身就在 seq={_x}' if _x <= 512 else f'交叉点在 seq={_x}，训练序列（512）尚未到达'
        _ratios = [r['csa_over_dense'] for r in _clean]
        _mono = all((_ratios[i] >= _ratios[i + 1] - 1e-12 for i in range(len(_ratios) - 1)))
        _trend = f'比值单调降到 {min(_ratios):.3f}' if _mono else f'比值总体下行到 {min(_ratios):.3f}（非严格单调，逐点见 flops_analytic.csv）'
        lines += ['', f'**要点**：在训练用的短序列（512）下 CSA 相对 dense **{_verdict}**（csa/dense FLOPs 比 = {_ratio:.3f}，选择率 {_sel:.1%}）；{_cross}。优势随序列变长继续放大（{_trend}）。稀疏的收益是随长度增长的，并非在任意长度上都成立。', '']
    return '\n'.join(lines) + '\n'

def sec_stats():
    d = load(os.path.join(REPO, 'analysis_v7', 'stats.json'))
    if not d:
        return ''
    lines = ['## 配对符号翻转检验（exact sign-flip permutation）', '', 'n≤4 时 bootstrap 无意义，改用精确符号翻转：枚举全部 2^n 种符号组合，双侧 p 值（n 很小时 p 的分辨率有限是统计事实，报告 Δ 与方向同向性）。', '', '| 比较 | Δ(mean) | p (exact) |', '|---|---|---|']
    items = d.get('comparisons', {}) if isinstance(d, dict) else {}
    if not items and isinstance(d, dict):
        items = {k: v for k, v in d.items() if isinstance(v, dict) and (v.get('mean') is not None or v.get('delta') is not None)}
    for k, v in items.items():
        if isinstance(v, dict):
            if 'omitted' in v:
                lines.append(f'| {k} | —（{v['omitted']}） | — |')
                continue
            _dv = v.get('mean') if v.get('mean') is not None else v.get('delta')
            _pv = v.get('p_exact_signflip') if v.get('p_exact_signflip') is not None else v.get('p')
            lines.append(f'| {k} | {fm(_dv)} | {fm(_pv, 4)} |')
    lines.append('')
    return '\n'.join(lines) + '\n'

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    for a in argv:
        if a in ('-h', '--help'):
            print(__doc__ or '')
            print('Usage: python build_report.py\n  Zero GPU.  Reads results_*/*.json + analysis_v7/*.json and rewrites\n  REPORT_v7.md next to the repo root.  Takes no arguments.')
            return 0
        raise SystemExit(f'build_report: unknown argument {a!r} (this builder takes none; try -h)')
    parts = [sec_header(), '---\n']
    failed = []
    for sec in (sec_p0r, sec_p0w, sec_p0e, sec_p1t, sec_p1l, sec_p2s, sec_flops, sec_stats):
        try:
            parts.append(sec())
        except Exception as e:
            print(f'[report] section {sec.__name__} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            failed.append(sec.__name__)
            parts.append(f'## （{sec.__name__} 生成失败）\n\n> `{type(e).__name__}: {e}` —— 本节未能从产物计算，其余各节不受影响。\n\n')
    if failed:
        print(f'[report] refusing to publish a partial report — failed section(s): {', '.join(failed)}')
        return None
    txt = '\n'.join((p for p in parts if p))
    out = os.path.join(REPO, 'REPORT_v7.md')
    _d = os.path.dirname(os.path.abspath(out))
    _fd, _tmp = tempfile.mkstemp(dir=_d, prefix=os.path.basename(out) + '.', suffix='.tmp')
    try:
        with os.fdopen(_fd, 'w', encoding='utf-8') as f:
            f.write(txt)
            f.flush()
            os.fsync(f.fileno())
        os.replace(_tmp, out)
    except BaseException:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise
    print(f'[report] wrote {out} ({len(txt)} bytes)')
    return out
if __name__ == '__main__':
    _rc = main()
    sys.exit(0 if _rc is not None else 1)
