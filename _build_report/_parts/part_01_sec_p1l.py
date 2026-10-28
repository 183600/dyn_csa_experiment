
def sec_p1l():
    s = summary('results_len')
    if not s:
        return '## P1 长上下文长度外推（train@512 → eval 512..4096）— （无结果）\n\n> 注：合成 NIAH 检索探针已实现并调试（碰撞无关布局、可达标签）；本报告以 wikitext 长度外推覆盖长上下文评测（见报告头部说明）。\n\n'
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
            # Fail closed: a panel whose every record predates
            # the current code/recipe stamp is excluded from the table, the
            # ratio and every conclusion.  It is listed as stale below instead
            # of being quoted with a warning marker.
            print(f'[report] sec_p1l: {_v}: NONE of the {len(_rv)} by-length record(s) carries the current code-semantics stamp {_CUR_CS!r} — EXCLUDED from the table and conclusions; re-run the length panel to refresh')
            _stale_vars.add(_v)
    if not recs:
        _msg = '## P1 长上下文长度外推 — （无可验证结果）\n\n'
        if _stale_vars:
            _msg += ('> **陈旧记录已排除**：' + '、'.join(f'`{v}`' for v in sorted(_stale_vars)) +
                     f' 的全部按长度记录都早于当前代码语义戳 `{_CUR_CS}`。按「旧代码/旧配方不得进入主要结论」的规则，'
                     '本节不引用这些数值、不计算 ratio，也不给出外推结论。重跑长度面板后再纳入。\n\n')
        return _msg
    def _len_key(Ln):
        try:
            return int(Ln)
        except (TypeError, ValueError, OverflowError):
            return None
    lens = sorted({_k for _k in (_len_key(Ln) for r in recs for Ln in r['by_len']) if _k is not None})
    variants = sorted({r.get('variant') for r in recs if r.get('variant') is not None})
    _seed_key = lambda x: (0, float(x), '') if isinstance(x, (int, float)) and (not isinstance(x, bool)) else (1, 0.0, str(x))
    _seeds_by_v = {v: sorted({r.get('seed') for r in recs if r.get('variant') == v and r.get('seed') is not None}, key=_seed_key) for v in variants}
    lines = ['## P1 长上下文长度外推（train@512 → 同一权重 eval 512/1024/2048/4096）', '', '**待检验论断**：仓库所有评测都在训练长度（512）上，没有任何长上下文证据。', '', '**做法**：4 变体（`full` / `csa_fixed` / `full_rope` / `csa_fixed_rope`）在 seq 512 训 3000 步（同一代码路径，各臂 seed 数见表），保存权重后用**同一份权重**在 512/1024/2048/4096 上评 wikitext PPL。absPE 变体 `max_seq=512`、超出部分只取末尾 `max_pos` 个 token 截断评测——**这正是 P0-3 的对照点**；RoPE 变体可原生外推。', '', '> **`~` = 位置受限（abs-PE）：该格只评了最后 `max_pos` 个 token，不是该长度的真实长上下文分数**；`?` = 该格缺少 `truncated` 标记，按不可验证处理（不参与 ratio）；未标注的格是完整 `Ln` 长度评测。', '', '| variant | ' + ' | '.join((f'PPL@{Ln}' for Ln in lens)) + ' | 相对退化 ratio@{0}/@{1} |'.format(lens[-1], lens[0]), '|' + '---|' * (len(lens) + 2)]
    per_v = {}
    trunc_v = {}
    # Group by variant (preserving order).
    _recs_by_v = {}
    for _r in recs:
        _recs_by_v.setdefault(_r.get('variant'), []).append(_r)
    for v in variants:
        cells = []
        base_map, last_map = ({}, {})
        trunc_flags = []
        for Ln in lens:
            recs_l = []
            _seen_sd = set()
            for _r in _recs_by_v.get(v, []):
                _bl = _r.get('by_len') or {}
                _c = _bl.get(Ln, _bl.get(str(Ln)))
                if isinstance(_c, dict) and _ppl_ok(_c.get('ppl')):
                    _sd = _r.get('seed')
                    if _sd in _seen_sd:
                        print(f'[report] sec_p1l: ambiguous (variant, seed) = ({v!r}, {_sd}) at L{Ln}: two measurable records found; keeping the FIRST')
                        continue
                    _seen_sd.add(_sd)
                    recs_l.append((_sd, _c))
            # Fail closed on a missing truncation flag: a cell that does not
            # say whether it was truncated is unverifiable, not full-length.
            _unknown = [_sd for _sd, x in recs_l
                        if 'truncated' not in x
                        or not isinstance(x.get('truncated'), (bool, np.bool_))]
            if _unknown:
                print(f'[report] sec_p1l: {v} L{Ln}: {len(_unknown)} cell(s) lack a boolean `truncated` flag {sorted(_unknown)} — the column is marked unverifiable and never enters a ratio')
            ok = [x for _sd, x in recs_l
                  if isinstance(x.get('truncated'), (bool, np.bool_))
                  and not bool(x.get('truncated'))]
            is_tr = bool(recs_l) and (len(ok) < len(recs_l))
            vals = [float(x['ppl']) for x in ok]
            trunc_flags.append(is_tr)
            mean = sum(vals) / len(vals) if vals else None
            if mean is None:
                cells.append('—' if not recs_l else '?')
            else:
                cells.append(f'~{mean:.1f}' if is_tr else f'{mean:.1f}')
            if ok and not is_tr:
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
        lines += ['', '> **⚠ 陈旧记录已排除**：' + '、'.join((f'`{v}`' for v in sorted(_stale_vars))) + f' 没有任何按长度记录携带当前代码语义戳 `{_CUR_CS}`，因此**未进入上表、未参与 ratio 或外推结论**（旧代码/旧配方不得进入主要结论）。重跑长度面板后再纳入。', '']
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

_AGG_ENTRY_CACHE = {}

def _agg_entry(d, v):
    # The resolution below scans the whole dict; agg dicts are already
    # memoized in _LOADED, so cache the per-variant resolution on the dict's
    # identity (keeping a strong ref to `d` so the id cannot be recycled
    # while the entry is alive).  Results are identical to the uncached scan.
    _ck = id(d)
    _ent = _AGG_ENTRY_CACHE.get(_ck)
    if _ent is None or _ent[0] is not d:
        _ent = (d, {})
        _AGG_ENTRY_CACHE[_ck] = _ent
    if v not in _ent[1]:
        _ent[1][v] = _agg_entry_scan(d, v)
    return _ent[1][v]

def _agg_entry_scan(d, v):
    if v in d:
        e = d[v]
        # A single entry that *does* carry a run_cfg must be
        # stamped with the current code semantics; quoting it otherwise would
        # put a stale number in a main table without any visible marker.
        rc = e.get('run_cfg') if isinstance(e, dict) else None
        if isinstance(rc, str) and bool(_CUR_CS) and not rc.endswith(f'_cs{_CUR_CS}'):
            print(f'[report] _agg_entry: `{v}` carries run_cfg `{rc[-24:]}` which does not end in the current code stamp `_cs{_CUR_CS}` — REFUSED (stale aggregate; re-run the panel)')
            return None
        return e
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
        # Fail closed: with several incompatible config groups and no current
        # stamp, picking "the largest n" is exactly the stale-admission path
        # Report the refusal instead.
        print(f'[report] _agg_entry: `{v}` spans {len(cands)} cfg/protocol groups ({sorted((k for k, _e in cands))}) and NONE carries the current code stamp `_cs{_CUR_CS}` — REFUSED (unverifiable/stale aggregate; re-run the panel)')
        return None
    if len(cands) > 1:
        print(f'[report] _agg_entry: `{v}` spans {len(cands)} cfg/protocol groups ({sorted((k for k, _e in cands))}) — quoting {_why} ({cands[0][0]}, n_seeds={cands[0][1].get('n_seeds')})')
    return cands[0][1]

def _params_m(r):
    p = _as_finite(r.get('params')) if isinstance(r, dict) else None
    if p is None or not p:
        return '—'
    return f'{p / 1000000.0:.3f}M'

def sec_p2s():
    d = agg('results_lm_v5_scale')
    if not d:
        return ''
    lines = ['## P2 v5 scale 面板补种子（至 4 seeds）+ 配对显著性', '', '**待检验论断**：v5 scale 的种子数不足，需 4 seeds + `full_matched`。', '', '| variant | PPL (pooled/mean; ±seed sd) | 文档 bootstrap 95% CI | n | params |', '|---|---|---|---|---|']
    _variants = sorted({r.get('variant') for r in d.values() if isinstance(r, dict) and r.get('variant') is not None})
    for v in _variants:
        r = _agg_entry(d, v)
        if r is None:
            continue
        _dci = r.get('ppl_ci95_doc_bootstrap')
        _dcitxt = f'[{_dci[0]:.2f}, {_dci[1]:.2f}]' if isinstance(_dci, (list, tuple)) and len(_dci) == 2 else '—（缺逐文档 NLL）'
        lines.append(f'| `{v}` | {fm(_ppl_value(r))} {sp(r.get('ppl_std'))} | {_dcitxt} | {r.get('n_seeds', r.get('n', '?'))} | {_params_m(r)} |')
    lines += ['', '> 注：`±seed sd` 是**种子层**训练随机性；`文档 bootstrap 95% CI` 重采样固定的评测文档，只量化**文档抽样**不确定性，二者不是同一个估计量。评测文档少于 64 条时该 CI 不覆盖语言数据总体（design_notes §20）。', '']
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
        lines += ['', '**配对检验（同 seed 配对，符号翻转）**：', '', '| 比较 | n (配对 seeds) | Δ (a−b) mean±std | p | params |', '|---|---|---|---|---|']
        for a, b, common, st in rows:
            _pl = f'{st["p_exact_signflip"]:.3f}' + (' (saddlepoint)' if str(st.get('method', '')).startswith('saddlepoint') else ' (exact)')
            lines.append(f'| `{a}` − `{b}` | {len(common)} | {st['mean']:+.2f} ± {st['std']:.2f} | {_pl} | {unmatched_tag(_agg_entry(_agg, a), _agg_entry(_agg, b)) or '✓'} |')
        main = next((r for r in rows if r[0] == 'csa_fixed' and r[1] == 'full'), None)
        _mt = next((r for r in rows if r[0] == 'csa_fixed' and r[1] == 'full_matched'), None) if _agg_entry(_agg, 'full_matched') is not None else None
        _base = _mt if _mt else main
        if _base:
            _ba, _bb, common, st = _base
            _which = '`csa_fixed − full_matched`（参数对齐）' if _mt else '`csa_fixed − full`（未做容量匹配）'
            _tag = unmatched_tag(_agg_entry(_agg, _ba), _agg_entry(_agg, _bb)) if _mt else unmatched_tag(_agg_entry(_agg, 'csa_fixed'), _agg_entry(_agg, 'full'))
            if st['mean'] > 0:
                concl = f'scale 面板上 {_which} 的配对差为 **{st['mean']:+.2f} PPL**（n={len(common)}，p={st['p_exact_signflip']:.3f}）：' + ('该差距通过 n≥6 + Holm + 效应>噪声门槛，方向为 sparse 更差。' if _sig_gate(st, label=f'P2S scale {_which}', family='P2S') else '该差距**未通过** n≥6 + Holm + 效应>噪声门槛，只能作方向性读数，不能写成「已经反转」。') + '该面板同时改变宽度、层数、序列长度与训练量，**不能把差异单独归因于「dense 的容量红利」**；需要参数/计算/训练量匹配的规模臂才能做机制归因。' + (f'{_tag}' if _tag else '✓')
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
    lines = ['## 解析 FLOPs / KV-cache 与交叉点', '', f'**配置**：{cfg}。**方法**：论文只给相对百分比、无闭式，故采用与硬件无关的解析 FLOPs / KV-cache 进行对比。**这些是结构性估算**；任何「效率优势」主张必须用 `audit_gates.measure_prefill_decode` 在真实 checkpoint 上测量后另行报告。', '', f'**交叉点**：CSA 的每 token 注意力 FLOPs 在 {_xo_txt} 处开始低于 dense。', '', '| seq | 选择率 | CSA/dense FLOPs | hybrid/dense | CSA KV/dense |', '|---|---|---|---|---|']
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
        lines += ['', f'**要点**：在训练用的短序列（512）下 CSA 相对 dense **{_verdict}**（csa/dense FLOPs 比 = {_ratio:.3f}，选择率 {_sel:.1%}）；{_cross}。**按该解析口径**（仅注意力项，不是实测延迟/吞吐/显存），CSA/dense 比随长度下降（{_trend}）；这不等于真实 GPU 加速，实际收益须由 checkpoint 上的测量确认。', '']
    return '\n'.join(lines) + '\n'

def _sig_gate(st, alpha=0.05, min_n=6, label=None, family=None):
    """Conservative significance gate for report prose.

    Requires n>=6 (the exact sign-flip floor at n<6 is >alpha, so no n<6 panel
    can support a significance claim), raw p<alpha, Holm p<alpha when present,
    and |mean|>std (effect bigger than seed noise).  A missing Holm value is
    treated as NOT corrected — the report may say "nominal" but never
    "significant" until a family correction has actually been attached.

    `label` (with optional `family`) registers the decision in the report's
    cross-panel multiplicity audit (`sec_global_multiplicity`): every
    significance decision the report makes must be counted there, because a
    claim that survives its own family can still be one of many contrasts the
    report displays.  An unlabeled call is counted and flagged, never lost."""
    if not isinstance(st, dict):
        return False
    # A threshold outside (0,1) would make every p "significant" (alpha>1) or
    # none reachable (alpha<=0).  Fail closed instead of reporting under it.
    _alpha = _as_finite(alpha)
    if _alpha is None or not (0.0 < _alpha < 1.0):
        return False
    alpha = _alpha
    # A minimum-n floor is a gate parameter, not a measurement: a boolean or
    # fractional value must fail closed instead of silently becoming 6.
    _min_n = _int_or(min_n, None)
    if _min_n is None or _min_n < 1:
        return False
    global _GATE_CALLS, _UNREGISTERED_GATES
    _GATE_CALLS += 1
    if label is None:
        _UNREGISTERED_GATES += 1
    else:
        _note_comparison(label, st.get('p_exact_signflip'), family,
                         st.get('p_holm'), st.get('n'), st.get('mean'))
    n = _int_or(st.get('n'), 0)
    p = _as_finite(st.get('p_exact_signflip'))
    ph = _as_finite(st.get('p_holm'))
    mean, std = _as_finite(st.get('mean')), _as_finite(st.get('std'))
    try:
        if n < _min_n:
            return False
        # Booleans/non-finite values are not p-values, and a negative p must
        # not pass every "< alpha" comparison.
        if p is None or not (0.0 <= p < alpha):
            return False
        if ph is None or not (0.0 <= ph < alpha):
            return False
        # Holm is a step-down correction: it can never make a raw p more
        # significant.  Mirror exp_lib.reportable_claim's hard refusal.
        if ph < p - 1e-12:
            return False
        if mean is None or std is None or abs(mean) <= std:
            return False
        # A complete-case subset is not a complete paired panel: if any
        # planned/paired seed was dropped by the run_cfg/estimand gate, the
        # surviving seeds were selected by that failure and cannot support a
        # significance claim (mirror exp_lib.reportable_claim).  Missing
        # fields keep the legacy permissive path.
        _raw_n_input = st.get('n_input')
        _raw_n_dropped = st.get('n_dropped')
        _n_input = _int_or(_raw_n_input, None)
        _n_dropped = _int_or(_raw_n_dropped, None)
        # A present-but-malformed accounting field is corrupt, not missing;
        # silently falling back would let e.g. n_dropped=0.5 or True pass as
        # a complete panel.
        if _raw_n_input is not None and _n_input is None:
            return False
        if _raw_n_dropped is not None and _n_dropped is None:
            return False
        if _n_input is not None and (_n_input < 0 or n > _n_input):
            return False
        if _n_dropped is not None:
            # A negative drop count is corrupt accounting, not an alternative
            # spelling of "no attrition" (mirror exp_lib.reportable_claim).
            if _n_dropped < 0:
                return False
            if _n_input is not None and _n_dropped > _n_input:
                return False
            if _n_dropped > 0:
                return False
        # n survivors + dropped inputs must reconstruct n_input; a smaller n
        # with n_dropped=0 means observations vanished from the accounting.
        if _n_input is not None and _n_dropped is not None:
            if n != _n_input - _n_dropped:
                return False
        elif _n_input is not None and _n_dropped is None and n != _n_input:
            return False
        # `complete_pairs=False` says the paired panel is incomplete even when
        # the caller did not also carry n_dropped.
        if st.get('complete_pairs') is False:
            return False
        # Exact ties lower the sign-flip resolution.  A fabricated n_nonzero
        # (or one bigger than the panel) must not make n seeds look finer than
        # the test can actually resolve.
        _raw_n_nonzero = st.get('n_nonzero')
        _n_nonzero = _int_or(_raw_n_nonzero, None)
        if _raw_n_nonzero is not None and _n_nonzero is None:
            return False
        if _n_nonzero is not None:
            if _n_nonzero < 0 or _n_nonzero > n:
                return False
            _floor = 1.0 if _n_nonzero == 0 else _signflip_floor(_n_nonzero)
        else:
            _floor = _signflip_floor(n)
        if _floor is None or not math.isfinite(_floor) or _floor > alpha:
            return False
        # Seeds rejected by the same-recipe controlled-comparison gate are
        # also an incomplete paired panel (mirror exp_lib.reportable_claim).
        # Both field spellings are used by the v9/v10/v11 contrast builders.
        _n_skipped = st.get('n_skipped_config_mismatch')
        if _n_skipped is None:
            _n_skipped = st.get('n_skipped')
        if _n_skipped is not None:
            _sk = _int_or(_n_skipped, None)
            if _sk is None or _sk < 0 or _sk > 0:
                return False
        # Leave-one-seed-out instability: dropping any one seed flips the
        # mean sign, so the direction is not stable across seeds even when the
        # aggregate p is small.
        if st.get('seed_driven') is True:
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True

# Every comparison whose p-value this report DISPLAYS, or whose significance
# the prose asserts via `_sig_gate`, is registered here so the cross-panel
# multiplicity audit (sec_global_multiplicity) counts it.  A report that shows
# twenty contrasts and corrects inside each panel still selected on twenty —
# the global view is what a cross-panel quote has to survive.
_REPORTED_COMPARISONS = []
_GATE_CALLS = 0
_UNREGISTERED_GATES = 0


def _reset_claim_registry():
    global _REPORTED_COMPARISONS, _GATE_CALLS, _UNREGISTERED_GATES
    _REPORTED_COMPARISONS = []
    _GATE_CALLS = 0
    _UNREGISTERED_GATES = 0


def _note_comparison(label, p, family=None, p_holm=None, n=None, mean=None):
    _REPORTED_COMPARISONS.append(
        {'label': str(label), 'p': p, 'family': (str(family) if family else None),
         'p_holm': p_holm, 'n': n, 'mean': mean})


def global_multiplicity_adjust(named_pvals, alpha=0.05):
    """Local twin of exp_lib.global_multiplicity_audit (this builder
    intentionally does not import torch/exp_lib; test_review_fixes cross-checks
    the two implementations against each other so they cannot drift).

    Benjamini-Hochberg step-up q-values + Bonferroni over ALL displayed
    comparisons.  Invalid/missing p keeps a p=1 slot (fail-closed family size,
    same convention as holm_adjust).  Duplicate names are a hard error."""
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    try:
        a = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    if not (0.0 < a < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    seen = set()
    rows = []
    for i, item in enumerate(named_pvals if named_pvals is not None else []):
        if not isinstance(item, (tuple, list)) or not 2 <= len(item) <= 3:
            raise ValueError(f'global_multiplicity_adjust: entry #{i} must be (name, p) or (name, p, family), got {item!r}')
        name = str(item[0])
        if name in seen:
            raise ValueError(f'global_multiplicity_adjust: duplicate comparison name {name!r}')
        seen.add(name)
        p = item[1]
        valid = False
        if p is not None and not isinstance(p, (bool, np.bool_)):
            try:
                pf = float(p)
                valid = math.isfinite(pf) and 0.0 <= pf <= 1.0
            except (TypeError, ValueError, OverflowError):
                valid = False
        fam = item[2] if len(item) == 3 else None
        rows.append({'name': name, 'p_raw': (pf if valid else None),
                     'p_slot': (pf if valid else 1.0),
                     'family': (str(fam) if fam is not None else None)})
    m = len(rows)
    if m == 0:
        return {'m': 0, 'alpha': a, 'rows': [],
                'n_raw_significant': 0, 'n_bh_rejected': 0,
                'n_bonferroni_significant': 0, 'n_untestable': 0}
    ps = sorted(range(m), key=lambda i: rows[i]['p_slot'])
    q_by_rank = []
    prev = float('inf')
    for rank in range(m, 0, -1):
        qi = rows[ps[rank - 1]]['p_slot'] * m / rank
        prev = min(prev, qi, 1.0)
        q_by_rank.append(prev)
    q_by_rank.reverse()
    for idx_in_sorted, row_i in enumerate(ps):
        rows[row_i]['q_bh'] = q_by_rank[idx_in_sorted]
        rows[row_i]['p_bonferroni'] = min(1.0, rows[row_i]['p_slot'] * m)
        p_slot = rows[row_i]['p_slot']
        rows[row_i]['bh_rejected'] = bool(rows[row_i]['q_bh'] <= a + 1e-15 and rows[row_i]['p_raw'] is not None)
    n_raw = n_bh = n_bonf = n_untest = 0
    for r in rows:
        if r['p_raw'] is None:
            n_untest += 1
            r['global_status'] = 'untestable'
        elif r['p_raw'] >= a:
            r['global_status'] = 'not_significant'
        elif r['bh_rejected'] and r['p_bonferroni'] < a:
            r['global_status'] = 'globally_controlled'
        elif r['bh_rejected']:
            r['global_status'] = 'fdr_only'
        else:
            r['global_status'] = 'family_only'
        if r['p_raw'] is not None and r['p_raw'] < a:
            n_raw += 1
        if r['bh_rejected']:
            n_bh += 1
        if r['p_raw'] is not None and r['p_bonferroni'] < a:
            n_bonf += 1
    return {'m': m, 'alpha': a, 'rows': rows, 'n_raw_significant': n_raw,
            'n_bh_rejected': n_bh, 'n_bonferroni_significant': n_bonf,
            'n_untestable': n_untest}


def sec_global_multiplicity():
    """Cross-panel multiplicity section: what the report's displayed contrasts
    look like AFTER all of them are counted (BH-FDR + Bonferroni).

    Within-family Holm (sec_stats) corrects inside one panel.  A report that
    displays contrasts from several panels and then quotes the significant ones
    has selected over the union, so the union is what a cross-panel quote must
    survive.  This section does not delete any number; it fixes the allowed
    wording for each one."""
    if not _REPORTED_COMPARISONS and _GATE_CALLS == 0:
        return '## 跨面板多重性（全局 FDR / Bonferroni）\n\n> 本报告没有显示任何带 p 值的比较或做任何显著性判定，本节为空。\n\n'
    lines = ['## 跨面板多重性（全局 FDR / Bonferroni）', '',
             '族内 Holm 只控制**单个面板内**的错误率；本报告把多个面板的比较并排放示，跨面板引用（“在 v7 与 v10 都显著”）实际是对**展示并集**的选择。下表把本报告实际显示/判定的每一个对比放进全局视图：Benjamini-Hochberg FDR q 值与 Bonferroni p×m。它不删任何数字，只约束措辞：', '',
             '| 比较 | 面板 | p | 族内 p_Holm | q_BH | p_Bonferroni | 全局判定 |', '|---|---|---|---|---|---|---|']
    named = [(r['label'], r['p'], r['family']) for r in _REPORTED_COMPARISONS]
    holm_by_label = {}
    for r in _REPORTED_COMPARISONS:
        if r.get('p_holm') is not None:
            holm_by_label[r['label']] = r['p_holm']
    aud = global_multiplicity_adjust(named, alpha=0.05)
    by_name = {r['name']: r for r in aud['rows']}
    for r in _REPORTED_COMPARISONS:
        a = by_name.get(r['label']) or {}
        _status_txt = {'globally_controlled': '全局可控（可跨面板引用）',
                       'fdr_only': '仅 FDR 可控（不得写族-wise 确定）',
                       'family_only': '仅族内（不得写“显著”而不加“未全局校正”）',
                       'not_significant': '不显著',
                       'untestable': '不可检验（无有效 p）'}.get(a.get('global_status'), a.get('global_status'))
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            r['label'], (r['family'] or '—'), fm(r['p'], 4),
            fm(holm_by_label.get(r['label']), 4), fm(a.get('q_bh'), 4),
            fm(a.get('p_bonferroni'), 4), _status_txt))
    lines += ['', f'共 {aud["m"]} 个展示比较：原始 p<0.05 的 {aud["n_raw_significant"]} 个；BH-FDR q<0.05 的 {aud["n_bh_rejected"]} 个；Bonferroni p<0.05 的 {aud["n_bonferroni_significant"]} 个；无有效 p 的 {aud["n_untestable"]} 个（计入族大小，p 槽位=1）。', '']
    if aud['n_bh_rejected'] < aud['n_raw_significant']:
        lines.append('> **读法限制**：存在原始 p<0.05 但全局 BH 不拒绝的比较——跨面板引用它们时必须写成「族内显著、未通过全局校正」，不得写成多个独立发现。')
    if _UNREGISTERED_GATES:
        lines.append(f'> ⚠ 有 {_UNREGISTERED_GATES} 个 `_sig_gate` 判定未带 label 注册（本节族大小可能偏小）——这是报告构建的缺陷，应修复，而不是忽略。')
    lines.append('> 本节是**探索性展示的纪律层**，不是确认性证据：全局校正不把任何探索面板变成预注册结果（见 docs/design_notes.md §17/§20）。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def holm_adjust(named_pvals, alpha=0.05):
    """Holm step-down over the report's displayed family (local copy: this
    builder intentionally does not import torch/exp_lib).  Missing/invalid
    p-values keep a dummy p=1 slot so the family size is not silently shrunk.

    The scalar contract mirrors ``exp_lib.holm_adjust``: booleans are flags,
    not probabilities (``False`` must never enter the family as p=0), and a
    threshold outside (0,1) is rejected instead of silently changing which
    comparisons count as significant."""
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    try:
        _alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}') from None
    if not (0.0 < _alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    items = []
    _seen = set()
    for k, v in (named_pvals if named_pvals is not None else []):
        name = str(k)
        if name in _seen:
            raise ValueError(f'holm_adjust: duplicate comparison name {name!r}')
        _seen.add(name)
        if isinstance(v, (bool, np.bool_)):
            pv = float('nan')
        else:
            try:
                pv = float(v)
            except (TypeError, ValueError, OverflowError):
                pv = float('nan')
        if (not math.isfinite(pv)) or pv < 0.0 or pv > 1.0:
            pv = 1.0
        items.append((name, pv))
    m = len(items)
    order = sorted(range(m), key=lambda i: items[i][1])
    out = {}
    prev = 0.0
    for rank, i in enumerate(order, start=1):
        name, p = items[i]
        adj = min(1.0, max(prev, (m - rank + 1) * p))
        prev = adj
        out[name] = adj
    return out

def _result_summaries():
    out = {}
    try:
        names = sorted(os.listdir(REPO))
    except OSError:
        return out
    for name in names:
        if not name.startswith('results_') or not os.path.isdir(os.path.join(REPO, name)):
            continue
        s = summary(name)
        if isinstance(s, dict) and s:
            out[name] = s
    return out


def sec_training_health():
    rows = []
    for name, s in _result_summaries().items():
        for key, rec in s.items():
            if not isinstance(rec, dict):
                continue
            h = rec.get('curve_health')
            if not isinstance(h, dict):
                continue
            rows.append((name, key, rec, h))
    if not rows:
        return ''
    lines = ['## Training-curve health (train/validation gap, over/under-fitting)', '',
             '> Diagnostic only. `best_val_ppl` / `best_step` are reported alongside the final number; they do not replace the pre-registered final-step endpoint.', '',
             '| panel | cell | status | best PPL @ step | final PPL | final−best | steps since best | tail train slope | tail val slope | gap slope | gap | eval points | eval docs | trend reliable | last curve 95% CI | action |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, key, rec, h in rows[:200]:
        _hist_raw = rec.get('health_history')
        _hist = _hist_raw if isinstance(_hist_raw, list) else []
        _last = (_hist[-1] if _hist and isinstance(_hist[-1], dict) else {})
        _ci = ('[%s, %s]' % (fm(_last.get('val_ci95_low')), fm(_last.get('val_ci95_high')))) if _last else 'n/a'
        _acts_raw = h.get('actions')
        if isinstance(_acts_raw, str):
            _acts = [_acts_raw]
        elif isinstance(_acts_raw, (list, tuple)):
            _acts = list(_acts_raw)
        else:
            _acts = []
        _act = '; '.join(str(a) for a in _acts) or '—'
        lines.append('| %s | `%s` | %s | %s @ %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, key, h.get('status'), fm(h.get('best_val_ppl')), h.get('best_step'),
            fm(rec.get('ppl')), fm(h.get('final_minus_best')), fm(h.get('steps_since_best'), 0),
            fm(h.get('train_ce_slope_per_step'), 6), fm(h.get('val_ce_slope_per_step'), 6),
            fm(h.get('gap_slope_per_step'), 6),
            fm(h.get('gap_last'), 4), fm(h.get('n_eval_points'), 0),
            fm(rec.get('eval_subset_rows'), 0), h.get('trend_reliable'), _ci,
            _act.replace('|', '\\|')))
    lines += ['', '> `overfitting` = train CE falling while val CE rises; `underfitting_or_stalled` = neither falls; `nonfinite` = a NaN/Inf point was recorded (see the NaN diagnostics in the same record). A substantially positive `gap slope` is direct overfitting evidence even when the validation slope is still small. `steps since best` > 0 means the final step is not the best validation point; report both, and do not silently select the best step after looking at the validation split. Curve points use `eval_subset_rows` fixed documents while the final PPL uses the full fixed validation batch. `best PPL @ step` and `final−best` are subset-vs-subset (like-for-like); the full-batch final PPL is shown separately and must not be subtracted from the subset best. The 95% CI is a document-level interval for the curve estimate, so small differences inside the interval are evaluation noise, not a mechanism effect.', '']
    return '\n'.join(lines) + '\n'


def sec_train_curve_health():
    rows = []
    for name, s in _result_summaries().items():
        for key, rec in s.items():
            if not isinstance(rec, dict):
                continue
            th = rec.get('train_curve_health')
            if isinstance(th, dict):
                rows.append((name, key, th))
    if not rows:
        return ''
    lines = ['## Per-step training-loss curve (not only the final value)', '',
             '> This is the training-side fallback for panels without a validation grid. It detects non-finite steps, a rising tail, finite spikes and a plateau; it cannot diagnose overfitting, which needs held-out points.', '',
             '| panel | cell | status | points | last | min | first spike | tail slope | relative change | action |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, key, th in rows[:200]:
        _acts_raw = th.get('actions')
        if isinstance(_acts_raw, str):
            _acts = [_acts_raw]
        elif isinstance(_acts_raw, (list, tuple)):
            _acts = list(_acts_raw)
        else:
            _acts = []
        _act = '; '.join(str(a) for a in _acts) or '—'
        lines.append('| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            name, key, th.get('status'), fm(th.get('n_points'), 0),
            fm(th.get('last'), 4), fm(th.get('min'), 4),
            fm(th.get('first_spike_index'), 0),
            fm(th.get('tail_slope_per_step'), 6),
            fm(th.get('tail_relative_change'), 4),
            _act.replace('|', '\\|')))
    lines += ['', '> `diverging` = the tail is rising; `unstable_spikes` = isolated finite jumps; `stalled` = no measurable decline in the tail; `nonfinite` = a NaN/Inf step was recorded. A spike or divergence that never reaches NaN is still an optimization failure and must not be hidden by quoting the final loss.', '']
    return '\n'.join(lines) + '\n'


def efficiency_group_key(eff):
    """Comparability key for a measured efficiency record.

    A latency/PPL frontier is only meaningful when token count, device, torch
    build, KV semantics and (when recorded) batch size are identical.  Missing
    provenance yields None (unverifiable), never a shared default bucket.
    """
    if not isinstance(eff, dict):
        return None
    dev = eff.get('device')
    torch_v = eff.get('torch')
    n_tok = eff.get('n_tokens')
    kv = eff.get('decode_has_kv_cache')
    if not isinstance(dev, str) or not dev:
        return None
    if not isinstance(torch_v, str) or not torch_v:
        return None
    if isinstance(n_tok, (bool, np.bool_)) or not isinstance(n_tok, (int, np.integer)) or int(n_tok) < 1:
        return None
    if not isinstance(kv, (bool, np.bool_)):
        return None
    # The same token count at batch 1 vs batch 8 is a different execution
    # shape.  A recorded batch must agree; an invalid recorded batch is
    # unverifiable, never a shared default.  Legacy records without the field
    # keep the historical 4-tuple (they remain comparable to each other, but
    # never to a record that does record its batch).
    batch = eff.get('batch')
    if batch is None:
        return (dev, torch_v, int(n_tok), bool(kv))
    if isinstance(batch, (bool, np.bool_)):
        return None
    if isinstance(batch, (int, np.integer)):
        batch = int(batch)
    elif isinstance(batch, (float, np.floating)):
        f = float(batch)
        if not (math.isfinite(f) and f == math.floor(f)):
            return None
        batch = int(f)
    else:
        return None
    if batch < 1:
        return None
    return (dev, torch_v, int(n_tok), bool(kv), batch)


def efficiency_pareto(points):
    """Non-dominated (latency, PPL) points within ONE comparable group."""
    pts = []
    for p in (points or []):
        if not isinstance(p, (tuple, list)) or len(p) < 3:
            continue
        lat = _as_finite(p[0])
        ppl = _as_finite(p[1])
        # A zero/negative latency is not a measured time (it makes every
        # ratio meaningless and would occupy the frontier), and PPL must be
        # strictly positive to be a valid model score.
        if lat is None or ppl is None or lat <= 0.0 or ppl <= 0.0:
            continue
        pts.append((lat, ppl, p[2]))
    pts.sort(key=lambda t: (t[0], t[1], str(t[2])))
    frontier = []
    best_ppl = float('inf')
    for lat, ppl, label in pts:
        if ppl < best_ppl:
            best_ppl = ppl
            frontier.append((lat, ppl, label))
    return frontier


def efficiency_accuracy_summary(items):
    """Per-arm PPL uncertainty and measured cost within one comparable group.

    ``items`` are dicts with ``variant``, ``ppl``, ``prefill_ms_p50`` and
    optionally ``params`` / ``peak_mem_allocated_mb``.  This is descriptive:
    seeds are treated as independent draws, so ``distinguishable`` is not a
    paired test.  It exists to stop a sub-noise PPL change from being presented
    as a win and to make the latency/memory multiplier visible next to it.
    """
    by_arm = collections.OrderedDict()
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        v = it.get('variant')
        if not isinstance(v, str) or not v:
            continue
        ppl = _as_finite(it.get('ppl'))
        lat = _as_finite(it.get('prefill_ms_p50'))
        if ppl is None or lat is None or ppl <= 0.0 or lat <= 0.0:
            continue
        _params = _as_finite(it.get('params'))
        _mem = _as_finite(it.get('peak_mem_allocated_mb'))
        _tok = _as_finite(it.get('tokens_seen'))
        by_arm.setdefault(v, []).append({
            'ppl': ppl, 'lat': lat,
            'params': (_params if (_params is not None and _params > 0.0) else None),
            'mem': (_mem if (_mem is not None and _mem >= 0.0) else None),
            # A zero/negative token budget is not a trained arm; treat it as
            # missing so it cannot "match" another invalid record.
            'tok': (_tok if (_tok is not None and _tok > 0.0) else None)})
    rows = []
    for v, vals in by_arm.items():
        pp = [x['ppl'] for x in vals]
        lats = sorted(x['lat'] for x in vals)
        n = len(pp)
        mean = sum(pp) / n
        std = sample_std(pp) if n > 1 else None
        sem = (std / math.sqrt(n)) if (std is not None and n > 1) else None
        params = [x['params'] for x in vals if x['params'] is not None]
        mems = [x['mem'] for x in vals if x['mem'] is not None]
        toks = sorted({x['tok'] for x in vals if x['tok'] is not None})
        rows.append({'variant': v, 'n': n, 'ppl_mean': mean, 'ppl_std': std,
                     'ppl_sem': sem, 'ppl_min': min(pp),
                     # np.median averages the two middle values for an even
                     # number of seeds; indexing the sorted list's middle
                     # element silently reported the upper median instead.
                     'latency_median': float(np.median(lats)),
                     'params_median': (float(np.median(params)) if params else None),
                     'peak_mem_max': (max(mems) if mems else None),
                     'tokens_seen_values': toks,
                     'tokens_seen_missing': sum(1 for x in vals if x['tok'] is None),
                     'tokens_seen_median': (float(np.median(toks)) if toks else None)})
    if not rows:
        return []
    rows.sort(key=lambda r: (r['ppl_mean'], r['variant']))
    # A cost/accuracy claim needs the SAME training budget.  Missing budget
    # fields are unverifiable, and one arm mixing two budgets is inconsistent;
    # neither may be silently treated as matched (the old median-of-unions
    # also hid a mixed arm whose median happened to equal the other arm).
    if any(r['tokens_seen_missing'] for r in rows):
        _budget_status = 'unverifiable'
    elif any(len(r['tokens_seen_values']) != 1 for r in rows):
        _budget_status = 'inconsistent'
    elif len({r['tokens_seen_median'] for r in rows}) > 1:
        _budget_status = 'differs'
    else:
        _budget_status = 'matched'
    _budget_matched = _budget_status == 'matched'
    ref = rows[0]
    for r in rows:
        r['ppl_delta_pct_vs_best'] = ((r['ppl_mean'] - ref['ppl_mean']) / ref['ppl_mean'] * 100.0)
        r['latency_ratio_vs_best'] = (r['latency_median'] / ref['latency_median']
                                      if ref['latency_median'] > 0 else None)
        r['params_ratio_vs_best'] = ((r['params_median'] / ref['params_median'])
                                     if (r['params_median'] is not None and ref['params_median']) else None)
        r['memory_ratio_vs_best'] = ((r['peak_mem_max'] / ref['peak_mem_max'])
                                     if (r['peak_mem_max'] is not None and ref['peak_mem_max']) else None)
        if r is ref:
            r['distinguishable'] = None
            r['flags'] = []
        else:
            if r['ppl_sem'] is not None and ref['ppl_sem'] is not None:
                r['noise_2se'] = 2.0 * math.sqrt(r['ppl_sem'] ** 2 + ref['ppl_sem'] ** 2)
                # Zero sample variance from a tiny seed set is not evidence of
                # zero noise; it is an unverifiable uncertainty estimate.
                r['distinguishable'] = (None if r['noise_2se'] == 0.0
                                        else abs(r['ppl_mean'] - ref['ppl_mean']) > r['noise_2se'])
            else:
                r['noise_2se'] = None
                r['distinguishable'] = None
            flags = []
            if abs(r['ppl_delta_pct_vs_best']) < 0.5:
                flags.append('ppl_gain_below_0.5pct')
            if r['distinguishable'] is False:
                flags.append('ppl_gain_within_seed_noise')
            if r['distinguishable'] is None:
                flags.append('ppl_uncertainty_unverifiable')
            if r['noise_2se'] == 0.0:
                flags.append('zero_within_seed_variance_unverifiable')
            if r['latency_ratio_vs_best'] is not None and r['latency_ratio_vs_best'] >= 2.0:
                flags.append('latency_2x_or_more')
            if r['params_ratio_vs_best'] is not None and r['params_ratio_vs_best'] >= 2.0:
                flags.append('params_2x_or_more')
            if r['memory_ratio_vs_best'] is not None and r['memory_ratio_vs_best'] >= 2.0:
                flags.append('memory_2x_or_more')
            _cost_2x = any(x is not None and x >= 2.0
                           for x in (r['latency_ratio_vs_best'],
                                     r['params_ratio_vs_best'],
                                     r['memory_ratio_vs_best']))
            if _cost_2x and (abs(r['ppl_delta_pct_vs_best']) < 0.5
                             or r['distinguishable'] is False):
                flags.append('cost_increase_without_clear_gain')
            r['flags'] = flags
        if _budget_status == 'differs':
            r['flags'] = list(r.get('flags') or []) + ['train_budget_differs']
        elif _budget_status == 'inconsistent':
            r['flags'] = list(r.get('flags') or []) + ['train_budget_inconsistent_within_arm']
        elif _budget_status == 'unverifiable':
            r['flags'] = list(r.get('flags') or []) + ['train_budget_unverifiable']
        r['budget_matched'] = _budget_matched
        r['train_budget_status'] = _budget_status
    return rows


def sec_early_stop():
    rows = []
    for name, summ in _result_summaries().items():
        for key, rec in summ.items():
            if not isinstance(rec, dict):
                continue
            es = rec.get('early_stop')
            planned = rec.get('steps')
            done = rec.get('steps_completed')
            _stopped = isinstance(es, dict) and bool(es.get('stopped'))
            # Only show an actual early stop (or a completed/planned mismatch).
            # A normal new record carries steps_completed == steps and must not
            # flood this section.
            if not _stopped and (done is None or done == planned):
                continue
            rows.append((name, key, rec, es, planned, done))
    if not rows:
        return ''
    lines = ['## Early stop / completed training budget', '',
             '> Early stopping is a diagnostic policy, not a free accuracy gain: records stopped before the planned step saw fewer tokens. Pairing already refuses equal-budget comparisons whose `tokens_seen` differ; this table makes the unequal budget visible instead of hiding it behind the planned step count.', '',
             '| panel | cell | stopped | completed / planned steps | completion | best PPL @ step | reason |', '|---|---|---|---|---|---|---|']
    for name, key, rec, es, planned, done in rows[:200]:
        es = es if isinstance(es, dict) else {}
        p = _as_finite(planned)
        d = _as_finite(done)
        frac = ((d / p) if (p and d is not None) else None)
        lines.append('| %s | `%s` | %s | %s / %s | %s | %s @ %s | %s |' % (
            name, key, bool(es.get('stopped')), fm(d, 0), fm(p, 0),
            (f'{frac * 100:.1f}%' if frac is not None else 'n/a'),
            fm(es.get('best_val_ppl')), fm(es.get('best_step'), 0),
            str(es.get('reason') or 'completed the planned steps').replace('|', '\\|')))
    lines += ['', '> `steps_completed` is what `tokens_seen` and the time-guard use. A curve that is still improving at the stop point means the patience/min-delta policy was too aggressive for this run.', '']
    return '\n'.join(lines) + '\n'


def sec_nan_diagnostics():
    rows = []
    for name, s in _result_summaries().items():
        for key, rec in s.items():
            if not isinstance(rec, dict):
                continue
            d = rec.get('instability_diagnosis')
            if not isinstance(d, dict):
                o = rec.get('opt_diag_summary')
                h = rec.get('curve_health') if isinstance(rec.get('curve_health'), dict) else {}
                _hbad = str(h.get('status') or '') == 'nonfinite'
                if not isinstance(o, dict) and not _hbad:
                    continue
                n_loss = (o or {}).get('nonfinite_loss_steps') or 0
                n_grad = (o or {}).get('nonfinite_grad_steps') or 0
                guard = (o or {}).get('nan_guard') if isinstance((o or {}).get('nan_guard'), dict) else {}
                if not (n_loss or n_grad or guard.get('elements_cleaned') or _hbad):
                    continue
                d = {'status': 'undiagnosed',
                     'primary_suspect': ('validation_curve_numerics' if _hbad else 'unknown'),
                     'summary': ('validation curve is non-finite but this record predates the triage path'
                                 if _hbad else 'non-finite events are present but this record predates the triage path')}
            elif d.get('status') == 'clean':
                continue
            o = rec.get('opt_diag_summary') if isinstance(rec.get('opt_diag_summary'), dict) else {}
            rows.append((name, key, rec, d, o))
    if not rows:
        return ''
    lines = ['## Non-finite (NaN/Inf) diagnostics', '',
             '> Triage order: data NaN/Inf -> forward/loss and log(0)/div0/softmax -> fp16 gradient-scaler overflow -> gradient explosion or LR. A check that this record cannot answer is reported as unverifiable, not as clean.', '',
             '| panel | cell | status | first suspect | first bad step | loss / grad steps | fp16 overflows | guarded attention values | data audit | action |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, key, rec, d, o in rows[:200]:
        guard = o.get('nan_guard') if isinstance(o.get('nan_guard'), dict) else {}
        _da = rec.get('data_audit') if isinstance(rec.get('data_audit'), dict) else {}
        _ds = _da.get('sections') if isinstance(_da.get('sections'), dict) else {}
        _dt = _ds.get('train') if isinstance(_ds.get('train'), dict) else {}
        _dv = _ds.get('validation') if isinstance(_ds.get('validation'), dict) else {}
        _data_txt = 'n/a'
        if _da:
            _data_txt = ('%s; train %s tok (%s), val %s tok; label-shift %s'
                         % (_da.get('status'),
                            fm(_dt.get('n_checked'), 0),
                            ('sampled' if _dt.get('sampled') else 'full'),
                            fm(_dv.get('n_checked'), 0),
                            (str((_da.get('label_shift') or {}).get('ok'))
                             if isinstance(_da.get('label_shift'), dict) else 'unknown')))
        action = ''
        for c in (d.get('checks') or []):
            if isinstance(c, dict) and c.get('check') == d.get('primary_suspect'):
                action = c.get('action') or ''
                break
        lines.append('| %s | `%s` | %s | %s | %s | %s / %s | %s | %s | %s | %s |' % (
            name, key, d.get('status'), d.get('primary_suspect'),
            fm(o.get('first_bad_step'), 0), fm(o.get('nonfinite_loss_steps'), 0),
            fm(o.get('nonfinite_grad_steps'), 0), fm(o.get('amp_scaler_overflows'), 0),
            fm(guard.get('elements_cleaned'), 0), _data_txt.replace('|', '\\|'),
            str(action).replace('|', '\\|')))
    lines += ['', '> This section is diagnostic only. A skipped optimizer update is not a fix: inspect the first bad step in `opt_diag.events`, reproduce with `CSA_STRICT_NAN=1` when the attention/pooling guard fired, and record the action in the decision log before comparing the run with others.', '']
    return '\n'.join(lines) + '\n'


def _error_analysis_paths():
    base = os.path.join(REPO, 'analysis_errors')
    paths = [os.path.join(base, 'summary.json')]
    try:
        for name in sorted(os.listdir(base)):
            sub = os.path.join(base, name)
            if os.path.isdir(sub):
                paths.append(os.path.join(sub, 'summary.json'))
    except OSError:
        pass
    return paths


def _render_error_analysis(d):
    lines = []
    ckpt = d.get('checkpoint')
    if ckpt:
        lines += [f'- checkpoint: `{os.path.basename(str(ckpt))}`', f'- split: **{d.get("split") or "unknown"}**', f'- pooled CE/PPL: {fm(d.get("mean_ce"), 4)} / {fm(d.get("pooled_ppl"), 2)}', '']
    if d.get('rare_target_basis'):
        _rb = str(d.get('rare_target_basis'))
        _rp = d.get('rare_target_problem')
        lines += [f'- rare-target basis: `{_rb}`' + (f' — {_rp}' if _rp else ''), '']
    buckets = d.get('buckets') or {}
    if isinstance(buckets, dict) and buckets:
        lines += ['| bucket | n | mean CE | PPL |', '|---|---|---|---|']
        for k, v in sorted(buckets.items()):
            if not isinstance(v, dict):
                continue
            lines.append('| %s | %s | %s | %s |' % (k, fm(v.get('n'), 0), fm(v.get('mean_ce'), 3), fm(v.get('perplexity'), 2)))
        lines.append('')
    _ca = d.get('context_ambiguity') if isinstance(d.get('context_ambiguity'), dict) else {}
    if _ca:
        lines += [f'- prefix ambiguity: {_ca.get("n_ambiguous_worst")} / {_ca.get("n_worst_reviewed")} '
                  f'reviewed worst tokens share their prefix with a different target '
                  f'({_ca.get("n_contexts_with_multiple_targets")} ambiguous contexts in the evaluated rows)', '']
    counts = d.get('cause_counts') if isinstance(d.get('cause_counts'), dict) else {}
    _guide_raw = d.get('cause_guide')
    guide = _guide_raw if isinstance(_guide_raw, dict) else {}
    if isinstance(counts, dict) and counts:
        lines += ['**Heuristic cause inventory (worst tokens)**', '', '| cause | n | review action |', '|---|---|---|']
        _counts = []
        for c, n in counts.items():
            try:
                n = int(n)
            except (TypeError, ValueError, OverflowError):
                continue
            if n > 0:
                _counts.append((str(c), n))
        for c, n in sorted(_counts, key=lambda kv: (-kv[1], kv[0])):
            lines.append('| %s | %d | %s |' % (c, n, str((guide or {}).get(c) or '').replace('|', '\\|')))
        lines.append('')
    allc = d.get('cause_counts_all') or {}
    if isinstance(allc, dict) and allc:
        lines += ['**Whole-split cause tally (finite targets, heuristic):**', '',
                  '| cause | n | share | mean CE |', '|---|---|---|---|']
        _all = []
        for c, v in allc.items():
            if not isinstance(v, dict):
                continue
            try:
                n = int(v.get('n') or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if n > 0:
                _all.append((str(c), n, _as_finite(v.get('share')), _as_finite(v.get('mean_ce'))))
        for c, n, share, mce in sorted(_all, key=lambda t: (-t[1], t[0])):
            lines.append('| %s | %d | %s | %s |' % (
                c, n, (f'{share * 100:.1f}%' if share is not None else 'n/a'), fm(mce, 3)))
        lines.append('')
    _tokens_raw = d.get('worst_tokens')
    tokens = _tokens_raw if isinstance(_tokens_raw, list) else []
    if tokens:
        lines += ['| # | row | pos | CE | context (truncated) | target | predicted | cause |', '|---|---|---|---|---|---|---|---|']
        for i, t in enumerate(tokens[:12]):
            if not isinstance(t, dict):
                continue
            tgt = str(t.get('target_text') or t.get('target_id') or '').replace('|', '\\|').replace('\n', '\\n')[:60]
            prd = str(t.get('pred_text') or t.get('pred_id') or '').replace('|', '\\|').replace('\n', '\\n')[:60]
            ctx = str(t.get('context_text') or '').replace('|', '\\|').replace('\n', '\\n')[:120]
            lines.append('| %d | %s | %s | %s | %s | %s | %s | %s |' % (
                i + 1, fm(t.get('row'), 0), fm(t.get('position'), 0), fm(t.get('ce'), 3),
                ctx, tgt, prd, t.get('cause_primary') or 'n/a'))
        lines.append('')
    if d.get('warning'):
        lines += [f'> {d.get("warning")}', '']
    return lines


def _render_paired_error_inference(paired):
    """Compact split-half bucket inference for the main report."""
    fams = paired.get('bucket_inference') if isinstance(paired, dict) else None
    if not isinstance(fams, list) or not fams:
        return []
    lines = ['']
    reasons = paired.get('bucket_family_reasons')
    if isinstance(reasons, dict) and reasons:
        lines += ['- omitted bucket families: ' + '; '.join(
            f'`{k}` ({str(v)[:160]})' for k, v in sorted(reasons.items())), '']
    for fam in fams:
        if not isinstance(fam, dict):
            continue
        name = fam.get('family') or 'unnamed'
        status = fam.get('status') or 'unknown'
        split = fam.get('split') if isinstance(fam.get('split'), dict) else {}
        lines += [f'**Paired bucket inference — `{name}`**: `{status}`', '']
        if fam.get('reason'):
            lines += [f'- {fam.get("reason")}', '']
        if split:
            lines += [f'- split: {split.get("n_discovery")} discovery / {split.get("n_replication")} '
                      f'replication validation rows (disjoint, hash-seeded)', '']
        buckets = fam.get('buckets') if isinstance(fam.get('buckets'), dict) else {}
        if not buckets:
            continue
        sel = fam.get('selected') if isinstance(fam.get('selected'), dict) else {}
        sel_name = sel.get('bucket')
        lines += ['| bucket | tokens | clusters | full ΔCE | full 95% CI | Holm p | discovery Δ | replication Δ | replicated |',
                  '|---|---|---|---|---|---|---|---|---|']
        for b, rec in sorted(buckets.items()):
            if not isinstance(rec, dict):
                continue
            dd = rec.get('discovery') if isinstance(rec.get('discovery'), dict) else {}
            rr = rec.get('replication') if isinstance(rec.get('replication'), dict) else {}
            _lo, _hi = _as_finite(rec.get('ci95_low')), _as_finite(rec.get('ci95_high'))
            _ci = (f'[{_lo:+.3f}, {_hi:+.3f}]' if _lo is not None and _hi is not None else 'n/a')
            lines.append('| %s%s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                '**' if b == sel_name else '', str(b) + ('**' if b == sel_name else ''),
                fm(rec.get('n_tokens'), 0), fm(rec.get('n_clusters'), 0), fm(rec.get('mean')),
                _ci, fm(rec.get('p_holm'), 4), fm(dd.get('mean')), fm(rr.get('mean')),
                ('yes' if sel.get('replicated') else 'no') if b == sel_name else ''))
        if sel:
            _sel_ci = sel.get('replication_ci95')
            if not (isinstance(_sel_ci, (list, tuple)) and len(_sel_ci) == 2):
                _sel_ci = [None, None]
            _rlo, _rhi = _as_finite(_sel_ci[0]), _as_finite(_sel_ci[1])
            _rci = (f'[{_rlo:+.3f}, {_rhi:+.3f}]' if _rlo is not None and _rhi is not None else 'n/a')
            lines += ['', f'- selected on discovery half: `{sel_name}`; replication Δ={fm(sel.get("replication_mean"))}, '
                      f'95% CI {_rci}, **replicated={sel.get("replicated")}** '
                      f'({sel.get("replication_reason")}); shrinkage='
                      f'{fm(sel.get("shrinkage_replication_over_discovery"), 3)}; '
                      f'rank in replication={sel.get("rank_in_replication")}/'
                      f'{sel.get("n_ranked_replication_buckets")}', '']
    lines += ['> Bucket p-values on the full sample are Holm-adjusted within the declared family but remain post-hoc descriptions. '
              'The replication half is an internal split-half stability check on the same fixed validation text, not a population test.', '']
    return lines


def sec_error_analysis():
    docs = []
    seen = set()
    for path in _error_analysis_paths():
        d = load(path)
        if not isinstance(d, dict) or not d:
            continue
        ck = str(d.get('checkpoint') or path)
        if ck in seen:
            continue
        seen.add(ck)
        docs.append((path, d))
    if not docs:
        return ''
    lines = ['## Bad-case analysis (human review required)', '',
             '> This section is an error-analysis aid, not a result table. The cause labels are heuristics: only reading the context and the target can distinguish an annotation error from a genuinely ambiguous example from a model blind spot.', '']
    for path, d in docs:
        if len(docs) > 1:
            lines += [f'### `{os.path.basename(os.path.dirname(path)) or "analysis_errors"}`', '']
        lines += _render_error_analysis(d)
        _paired_path = os.path.join(os.path.dirname(path), 'paired_errors.json')
        _paired = load(_paired_path, default=None)
        if isinstance(_paired, dict):
            lines += _render_paired_error_inference(_paired)
    return '\n'.join(lines) + '\n'


def sec_efficiency():
    rows = []
    for name, s in _result_summaries().items():
        for key, rec in s.items():
            if not isinstance(rec, dict):
                continue
            eff = rec.get('efficiency')
            if not isinstance(eff, dict) or eff.get('status') != 'ok':
                continue
            rows.append((name, key, rec, eff))
    if not rows:
        return ''
    lines = ['## Measured efficiency (not analytic FLOPs)', '',
             '> Measured on the actual checkpoint/architecture with warmup + repeated runs. `prefix_prefill` is **not** incremental decode: the model has no KV cache, so each step re-runs the whole prefix. Training throughput is wall-clock tokens/s for the whole cell (it includes evaluation/checkpoint overhead, so it is not a pure kernel measurement).', '',
             '| panel | cell | params (M) | prefill p50/p95 ms | tokens/s | prefix-prefill p50 ms | peak alloc MB | peak reserved MB | train tokens/s | tokens seen |', '|---|---|---|---|---|---|---|---|---|---|']
    for name, key, rec, eff in rows[:200]:
        p = rec.get('params')
        pm = (p / 1e6) if isinstance(p, (int, float)) else None
        _ttps = rec.get('train_tokens_per_s')
        if not isinstance(_ttps, (int, float)) or isinstance(_ttps, bool) or not math.isfinite(float(_ttps)):
            _ttps = None
        lines.append('| %s | `%s` | %s | %s / %s | %s | %s | %s | %s | %s | %s |' % (
            name, key, fm(pm), fm(eff.get('prefill_ms_p50'), 2), fm(eff.get('prefill_ms_p95'), 2),
            fm(eff.get('tokens_per_s_prefill'), 1), fm(eff.get('prefix_prefill_ms_p50'), 2),
            fm(eff.get('peak_mem_allocated_mb'), 1), fm(eff.get('peak_mem_reserved_mb'), 1),
            fm(_ttps, 1), fm(rec.get('tokens_seen'), 0)))
    groups = collections.OrderedDict()
    ungrouped = []
    for name, key, rec, eff in rows:
        gk = efficiency_group_key(eff)
        if gk is None:
            ungrouped.append((name, key))
            continue
        groups.setdefault(gk, []).append((name, key, rec, eff))
    if groups:
        lines += ['', '**Pareto frontiers within comparable measurement groups**', '',
                  '> A frontier is drawn only inside one (device, torch, token-count, KV-semantics, recorded batch) group. A PPL difference between groups is confounded by input shape or hardware and is not an architecture comparison.', '']
        for gk, grp in groups.items():
            pts = [(eff.get('prefill_ms_p50'), rec.get('ppl'), f'{name} / `{key}`')
                   for name, key, rec, eff in grp]
            frontier = efficiency_pareto(pts)
            if not frontier:
                continue
            _batch_txt = (f', batch {gk[4]}' if len(gk) > 4 else ', batch unrecorded')
            lines += [f'- device `{gk[0]}`, torch `{gk[1]}`, {gk[2]} tokens, KV={gk[3]}{_batch_txt}:', '']
            for lat, ppl, label in frontier:
                lines.append(f'  - {label}: PPL {fm(ppl)}, prefill p50 {fm(lat, 2)} ms')
            summary = efficiency_accuracy_summary([
                {'variant': rec.get('variant') or str(key).split('::')[0],
                 'ppl': rec.get('ppl'), 'prefill_ms_p50': eff.get('prefill_ms_p50'),
                 'params': rec.get('params'),
                 'peak_mem_allocated_mb': (eff.get('peak_mem_allocated_mb')
                                           if eff.get('peak_mem_allocated_mb') is not None
                                           else ((rec.get('peak_memory') or {}).get('max_memory_allocated_mb')
                                                 if isinstance(rec.get('peak_memory'), dict) else None)),
                 'tokens_seen': rec.get('tokens_seen')}
                for name, key, rec, eff in grp])
            if len(summary) >= 2 and summary[0].get('latency_ratio_vs_best') is not None:
                lines += ['', '  Cost of the best-PPL arm in this group (seed spread, not a paired test):', '',
                          '  | arm | n | PPL mean ± SEM | Δ vs best | latency × best | flags |', '  |---|---|---|---|---|---|']
                for r in summary:
                    flags = ', '.join(r.get('flags') or []) or '—'
                    lines.append('  | %s | %d | %s ± %s | %s%% | %s | %s |' % (
                        r['variant'], r['n'], fm(r['ppl_mean'], 3), fm(r['ppl_sem'], 3),
                        fm(r.get('ppl_delta_pct_vs_best'), 2),
                        fm(r.get('latency_ratio_vs_best'), 2), flags))
                ref = summary[0]
                for r in summary[1:]:
                    gain = r.get('ppl_delta_pct_vs_best', 0.0)
                    _lr = r.get('latency_ratio_vs_best')
                    lat_ratio = (1.0 / _lr) if (_lr and _lr > 0) else 0.0
                    if (r.get('budget_matched') is not False
                            and r.get('distinguishable') is True and gain >= 0.5 and lat_ratio >= 1.5):
                        # `gain` is delta-vs-best (positive = r is worse), so
                        # the best arm IMPROVES PPL by that magnitude.  The
                        # old sentence printed the negative sign as a "gain".
                        lines += ['', f'  > `{ref["variant"]}` improves PPL by {abs(float(gain)):.2f}% over `{r["variant"]}` but is {lat_ratio:.1f}× slower; the accuracy/cost tradeoff must be justified explicitly.']
                lines.append('')
        if ungrouped:
            lines += ['', f'> {len(ungrouped)} measured record(s) were excluded from every frontier because device/torch/token-count/KV provenance is missing (unverifiable, not assumed equal): ' + ', '.join(f'`{n}`/`{k}`' for n, k in ungrouped[:20]) + ('.' if len(ungrouped) <= 20 else ', ...'), '']
    lines += ['', '> A PPL improvement should be read together with this table: a small gain accompanied by much higher latency or memory is not a deployable improvement. Compare only within a group; a latency measured at 512 tokens cannot be subtracted from one measured at 4096.', '']
    return '\n'.join(lines) + '\n'


def sec_stats():
    d = load(os.path.join(REPO, 'analysis_v7', 'stats.json'))
    if not d:
        return ''
    lines = ['## 配对符号翻转检验（sign-flip permutation）', '', '符号翻转检验：n≤20 枚举全部 2^n 种符号组合得精确双侧 p 值；20<n≤40 用 meet-in-the-middle 精确计数；n>40 用确定性的 Lugannani-Rice 鞍点尾近似（不是随机 Monte-Carlo）。n 很小时 p 的分辨率下限为 2/2^n，报告 Δ、方向同向性与该下限。', '', '> **证据等级：探索性（v7 面板）**。Holm 列在本报告展示的对比族内做 step-down 校正，防止把多格 p<0.05 读成多个发现；它不是预注册确认，也不能把 v7–v11 已看过的数据变成确认性证据（见 docs/design_notes.md §17）。', '', '| 比较 | Δ(mean) | p | p_Holm |', '|---|---|---|---|']
    items = d.get('comparisons', {}) if isinstance(d, dict) else {}
    if not items and isinstance(d, dict):
        items = {k: v for k, v in d.items() if isinstance(v, dict) and (v.get('mean') is not None or v.get('delta') is not None)}
    _pairs = []
    for k, v in items.items():
        _pv = None
        if isinstance(v, dict) and 'omitted' not in v:
            _pv = v.get('p_exact_signflip') if v.get('p_exact_signflip') is not None else v.get('p')
            try:
                _pv = float(_pv)
            except (TypeError, ValueError, OverflowError):
                _pv = None
        # Omitted/invalid comparisons keep a dummy p=1 slot so a report cannot
        # shrink its own correction family by dropping a failed comparison.
        _pairs.append((k, _pv if (_pv is not None and math.isfinite(_pv) and 0.0 <= _pv <= 1.0) else 1.0))
    _holm = holm_adjust(_pairs, alpha=0.05) if _pairs else {}
    # The displayed family is itself part of the report's global selection:
    # register every displayed comparison (invalid/omitted ones included) so
    # the cross-panel audit counts them with a p=1 slot, never drops them.
    for _k, _pv in _pairs:
        _note_comparison(f'STATS {_k}', _pv, family='v7 displayed comparisons',
                         p_holm=_holm.get(_k))
    for k, v in items.items():
        if not isinstance(v, dict):
            continue
        if 'omitted' in v:
            lines.append('| %s | —（%s） | — | — |' % (k, v.get('omitted')))
            continue
        _dv = v.get('mean') if v.get('mean') is not None else v.get('delta')
        _pv = v.get('p_exact_signflip') if v.get('p_exact_signflip') is not None else v.get('p')
        _ph = _holm.get(k)
        _method = str(v.get('method') or '')
        _pt = fm(_pv, 4) + (' (saddlepoint)' if _method.startswith('saddlepoint') else (' (MITM)' if _method == 'exact-mitm' else ''))
        lines.append('| %s | %s | %s | %s |' % (k, fm(_dv), _pt, fm(_ph, 4)))
    lines += ['', '> 若未校正 p<0.05 而 p_Holm≥0.05，只能写「名义上显著、族内未校正通过」，不能写成显著效应。', '']
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
    _reset_claim_registry()
    for sec in (sec_p0r, sec_p0w, sec_p0e, sec_p1t, sec_p1l, sec_p2s, sec_flops, sec_training_health, sec_train_curve_health, sec_early_stop, sec_nan_diagnostics, sec_error_analysis, sec_efficiency, sec_stats, sec_global_multiplicity):
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
def _run_cli():
    _rc = main()
    sys.exit(0 if _rc is not None else 1)
