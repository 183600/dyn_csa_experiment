#!/usr/bin/env python3
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
import v9_supp as V9
V = V9.V
L = V9.L
V8 = V9.V8
REPO = V9.REPO
DEVICE = L.DEVICE
BUDGET_V10 = dict(L.BUDGET)
BUDGET_V10.update(total_yuan=float(os.environ.get('V10_BUDGET_YUAN', 60.0)), price_per_hour=float(os.environ.get('V10_PRICE_PER_HOUR', 2.4)), state_path='autodl_budget_state_v10.json', already_spent_yuan=0.0)

def make_guard():
    g = L.CostGuard(BUDGET_V10)
    if not g.state.get('sps_by_class'):
        for src in ('autodl_budget_state_v9.json', 'autodl_budget_state_v8.json', 'autodl_budget_state_v7.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v10] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v10] CostGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v10] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g
P3MT_PAYLOAD = dict(outdir='results_v10_mech', variants=['csa_fixed_rope', 'full_rope'], eval_lens=[512, 1024, 2048, 4096], steps=3000, train_len=512, vocab=8192)
PROBE = dict(outdir='results_v10_mech', ckpt_dir='results_v10_mech/ckpt', summary='results_v10_mech/distractor.json', variants=['csa_fixed_rope', 'full_rope'], seeds=[0, 1, 2], eval_lens=[2048, 4096], rhos=[0.0, 0.125, 0.25, 0.5], target=512, n_seq=16, chunk=4, arms=['dense', 'learned', 'randidx', 'allblocks'])

def _arm_of(variant, arm):
    if variant == 'full_rope':
        return arm == 'dense'
    return arm in ('learned', 'randidx', 'allblocks')

@torch.no_grad()
def _distractor_ppls(model, val_ids, eval_len, rho, n_seq, seed, cfg):
    target = cfg['target']
    vocab = int(cfg.get('vocab') or P3MT_PAYLOAD['vocab'])
    far = eval_len - target
    n_far = far
    k = int(round(rho * n_far))
    nll_sum = []
    n_tok = []
    n_ch = max(1, int(cfg['chunk']))
    for i in range(0, n_seq, n_ch):
        rows = []
        for j in range(i, min(i + n_ch, n_seq)):
            ids = np.asarray(val_ids[j, :eval_len + 1], dtype=np.int64).copy()
            if k > 0:
                rng = np.random.default_rng((eval_len * 1000003 + j * 10007) * 1048576 + int(round(rho * 1048576)))
                pos = rng.choice(n_far, size=k, replace=False)
                ids[pos] = (ids[pos] + rng.integers(1, vocab, size=k)) % vocab
            rows.append(ids)
        ids = torch.from_numpy(np.stack(rows)).to(DEVICE)
        logits = model(ids[:, :-1])
        ce = F.cross_entropy(logits[:, -target:].reshape(-1, vocab), ids[:, 1:][:, -target:].reshape(-1), reduction='none')
        ce = ce.double().view(len(rows), -1)
        nll_sum += ce.sum(1).tolist()
        n_tok += [int(ce.shape[1])] * len(rows)
        del ids, logits, ce
    return (nll_sum, n_tok)

def _cell_ppl(nll_sum, n_tok):
    if not nll_sum:
        return float('nan')
    total = math.fsum((float(x) for x in nll_sum))
    nt = int(sum(n_tok))
    if nt <= 0:
        return float('nan')
    return math.exp(total / nt)

def _probe_fingerprint(cfg):
    return {'n_seq': int(cfg['n_seq']), 'chunk': int(cfg['chunk']), 'target': int(cfg['target']), 'vocab': int(cfg.get('vocab') or P3MT_PAYLOAD['vocab']), 'stat': 'ppl_pooled_nll_v3'}

def _probe_params_current(rec, fp):
    return isinstance(rec, dict) and rec.get('probe_params') == fp

def run_probe(cfg=PROBE, guard=None, label='v10 P3MP'):
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['summary']
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[resume] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    need = max(cfg['eval_lens']) + 1
    train_len = int(cfg.get('train_len') or P3MT_PAYLOAD['train_len'])
    _, val_ids, _, _, _ = L.load_wikitext(max(train_len, need), 4000000)
    mech_summ = {}
    _trdir = cfg.get('outdir') or P3MT_PAYLOAD['outdir']
    _msp = os.path.join(_trdir, 'summary.json')
    if os.path.exists(_msp):
        try:
            mech_summ = json.load(open(_msp, encoding='utf-8'))
        except Exception as _e:
            print(f'[p3mp] FATAL: {_msp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — without it no checkpoint can be attributed to a training recipe, so the probe would score unknown weights.  Move it aside to re-train.')
            raise
    print(f'[p3mp] val slice {val_ids.shape}, arms={cfg['arms']}, rhos={cfg['rhos']}')
    for v in cfg['variants']:
        for seed in cfg['seeds']:
            ck = os.path.join(cfg['ckpt_dir'], f'{v}_seed{seed}.pt')
            if not os.path.exists(ck):
                print(f'[p3mp] SKIP {v} s{seed}: no checkpoint {ck} (run phase P3MT first)')
                continue
            d = torch.load(ck, map_location='cpu', weights_only=False)
            if d.get('code') != V.CKPT_CODE:
                print(f'[p3mp] REFUSE {v} s{seed}: checkpoint predates the current code stamp (code={d.get('code')!r}) — re-run the training phase (P3MT/P4MT) first')
                del d
                continue
            _tr = (mech_summ or {}).get(f'{v}::seed{seed}') or {}
            if not _tr:
                print(f"[p3mp] REFUSE {v} s{seed}: {ck} exists but {_msp} has no '{v}::seed{seed}' training record — the weights have no traceable provenance, so they cannot be attributed to the current recipe")
                del d
                continue
            _vocab_ck = int(d.get('vocab', 8192))
            if cfg.get('vocab') is not None and int(cfg['vocab']) != _vocab_ck:
                print(f"[p3mp] {v} s{seed}: probe cfg vocab={int(cfg['vocab'])} differs from the checkpoint's vocab={_vocab_ck} — using the checkpoint's (the model is what is being scored)")
            _pcfg = dict(cfg, vocab=_vocab_ck)
            _fp = _probe_fingerprint(_pcfg)
            _defs = [(Ln, rho) for Ln in cfg['eval_lens'] for rho in cfg['rhos']]
            for arm in cfg['arms']:
                if not _arm_of(v, arm):
                    continue

                def _cell_key(_Ln, _rho, _arm=arm, _v=v, _seed=seed):
                    return f'{_v}::s{_seed}::{_arm}::L{_Ln}::r{_rho}'
                _stale_param = [f'L{Ln}::r{rho}' for Ln, rho in _defs if (summary.get(_cell_key(Ln, rho)) or {}).get('probe_params') is not None and (not _probe_params_current(summary.get(_cell_key(Ln, rho)), _fp))]
                if _stale_param:
                    print(f'[p3mp] {v} s{seed} {arm}: re-probing {len(_stale_param)} cell(s) whose stored probe parameters differ from the current config ({', '.join(_stale_param[:3])}{('…' if len(_stale_param) > 3 else '')})')
                cells = [(Ln, rho) for Ln, rho in _defs if not (L.result_is_current(summary.get(_cell_key(Ln, rho)), V.CKPT_CODE, 'ppl_mean') and L.ppl_is_usable((summary.get(_cell_key(Ln, rho)) or {}).get('ppl_mean')) and _probe_params_current(summary.get(_cell_key(Ln, rho)), _fp))]
                if not cells:
                    continue
                cfgs = []
                for c in d['cfg']:
                    c2 = copy.copy(c)
                    if arm == 'randidx':
                        c2.indexer_mode = 'random'
                    elif arm == 'allblocks':
                        c2.index_topk = 10 ** 6
                    cfgs.append(c2)
                model = L.SmallGPT(_vocab_ck, 256, 6, 8, 32, d.get('train_len', 512), cfgs, mlp_ratio=d['mlp_ratio']).to(DEVICE)
                model.load_state_dict(d['sd'])
                model.eval()
                t0 = time.time()
                for Ln, rho in cells:
                    key = _cell_key(Ln, rho)
                    nll_sum, n_tok = _distractor_ppls(model, val_ids, Ln, rho, _pcfg['n_seq'], seed, _pcfg)
                    cell_ppl = _cell_ppl(nll_sum, n_tok)
                    per_seq = [math.exp(s / t) for s, t in zip(nll_sum, n_tok)]
                    summary[key] = {'variant': v, 'seed': seed, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'ppl_mean': float(cell_ppl), 'ppl_std': float(np.std(per_seq, ddof=1)) if len(per_seq) > 1 else 0.0, 'ppls': [float(p) for p in per_seq], 'nll_sum': [float(x) for x in nll_sum], 'n_tok': [int(x) for x in n_tok], 'n_seq': len(per_seq), 'target': cfg['target'], '_code': V.CKPT_CODE, 'probe_params': _fp}
                    print(f'  [p3mp] {key:44s} PPL={cell_ppl:8.2f}', flush=True)
                    L.atomic_write_json(spath, summary, indent=1)
                if guard is not None:
                    guard.record_run(time.time() - t0, 0, 0, 0, 0, 0)
                del model
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            del d
    print(f'[p3mp] wrote {spath} ({len(summary)} cells)')
    return summary
P3SS_CFG = dict(L.RUN, outdir='results_lm_v10_scale_s', d=128, n_layers=4, n_heads=4, d_head=32, seq_len=512, batch_size=12, n_train_tokens=40000000, steps=8000, warmup=100, eval_every=250, eval_subset=64, variants=['csa_fixed', 'full'])
P3SL_CFG = dict(L.RUN, outdir='results_lm_v10_scale_l', d=512, n_layers=10, n_heads=16, d_head=32, seq_len=512, batch_size=12, n_train_tokens=40000000, steps=4000, warmup=100, eval_every=250, eval_subset=64, variants=['csa_fixed', 'full'])
PHASES = [('P3MT', 'lenphase', P3MT_PAYLOAD, [0, 1, 2], 1.6), ('P3MP', 'probe', PROBE, None, 0.3), ('P3T', 'run', V9.P2T_CFG, [2, 3], 1.8), ('P3SS', 'run', P3SS_CFG, [0, 1], 0.6), ('P3SL', 'run', P3SL_CFG, [0, 1], 9.0)]

def run_phase(name, guard):
    for pname, kind, payload, seeds, _h in PHASES:
        if pname != name:
            continue
        if kind == 'lenphase':
            return V.run_lenphase(payload, guard=guard, label=f'v10 {pname}')
        if kind == 'probe':
            return run_probe(payload, guard=guard, label=f'v10 {pname}')
        ckpt_on = pname == 'P3T'
        try:
            V9._V9_CKPT['on'] = ckpt_on
            s, _a = L.run(payload, seeds=seeds, guard=guard, label=f'v10 {pname}')
        finally:
            V9._V9_CKPT['on'] = False
        return s
    raise SystemExit(f'unknown phase {name}')

def _ppl_by_seed(outdir, full=False):
    return V9._ppl_by_seed(outdir, full=full)

def _hist_by_seed(outdir, variant):
    sp = os.path.join(outdir, 'summary.json')

    class _Curves(dict):

        def __init__(self):
            super().__init__()
            self.per_seed_synth = {}
    out = _Curves()
    synth_of = {}
    if not os.path.exists(sp):
        return out
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no curve for {variant}')
        return out
    seen = {}
    for _k, r in raw.items():
        if not (isinstance(r, dict) and r.get('variant') == variant and r.get('ppl_history')):
            continue
        s = int(r['seed'])
        try:
            fp = tuple(((int(t), type(v).__name__, round(float(v), 9)) for t, v in r['ppl_history'] if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v == v)))
        except (TypeError, ValueError):
            fp = None
        if fp is None:
            print(f'[v10 stats] {_k} has an unreadable ppl_history entry; its curve fingerprint cannot be de-duplicated reliably')
            fp = ('__unreadable__', id(r))
        if fp in seen:
            if not r.get('synthesized'):
                print(f'[stats] {sp}: seed {s} of `{variant}` is a REAL record but its curve is IDENTICAL to the one already admitted for seed {seen[fp]}; keeping seed {seen[fp]} (the first admission) and dropping seed {s} so the curve is not counted twice.')
            continue
        seen[fp] = s
        synth_of[s] = bool(r.get('synthesized'))
        out[s] = r['ppl_history']
    out.per_seed_synth = synth_of
    return out

def _crossover_step(gap_steps, gap_vals, smooth=1):
    g = np.asarray(gap_vals, dtype=float)
    s = np.asarray(gap_steps, dtype=float)
    if len(g) != len(s):
        raise ValueError(f'_crossover_step: gap_steps and gap_vals must be the same length ({len(s)} vs {len(g)}) — the returned step indexes the two together, so a silent misalignment would report the crossover at the wrong STEP while looking perfectly well-formed.')
    if smooth > 0 and len(g) > 2 * smooth:
        g = np.array([g[max(0, i - smooth):i + smooth + 1].mean() for i in range(len(g))])
    for i in range(len(g)):
        if g[i] > 0 and np.all(g[i:] > 0):
            return (float(s[i]), True)
    return (float(s[len(g) - 1]) if len(g) else float('nan'), False)

def _tokens_per_step(outdir, fallback=None):
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return None
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — no tokens/step for this panel')
        return None
    rates = []
    n_bad = 0
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('synthesized'):
            continue
        _ts, _st = (r.get('tokens_seen'), r.get('steps'))
        if isinstance(_ts, bool) or isinstance(_st, bool):
            n_bad += 1
            continue
        if not (isinstance(_ts, (int, float)) and isinstance(_st, (int, float))):
            if _ts is not None or _st is not None:
                n_bad += 1
            continue
        if not (math.isfinite(_ts) and math.isfinite(_st) and (_st > 0)):
            n_bad += 1
            continue
        rates.append(_ts / _st)
    if n_bad:
        print(f'[stats] WARNING: {outdir}: {n_bad} record(s) state a non-numeric or non-positive tokens_seen/steps and are ignored when establishing the token axis')
    if not rates:
        return None
    if len(set(rates)) > 1:
        print(f'[stats] WARNING: {outdir} reports {len(set(rates))} distinct tokens/step rates {sorted(set(rates))} — the token axis is ambiguous, so no crossover_tokens will be published')
        return None
    return float(rates[0])

def _scale_crossovers():
    panels = {'d128_L4': dict(outdir='results_lm_v10_scale_s', d=128, n_layers=4, batch=12, seq=512), 'd256_L6': dict(outdir='results_lm_v3_long', d=256, n_layers=6, batch=12, seq=512), 'd384_L8': dict(outdir='results_lm_v5_scale', d=384, n_layers=8, batch=8, seq=1024), 'd512_L10': dict(outdir='results_lm_v10_scale_l', d=512, n_layers=10, batch=12, seq=512)}
    out = {}
    for label, p in panels.items():
        ha = _hist_by_seed(p['outdir'], 'csa_fixed')
        hb = _hist_by_seed(p['outdir'], 'full')
        common = sorted(set(ha) & set(hb))
        if not common:
            out[label] = {'status': 'missing', **p}
            continue
        _paired = []
        _no_overlap = []
        for s in common:
            if set(dict(ha[s])) & set(dict(hb[s])):
                _paired.append(s)
            else:
                _no_overlap.append(s)
        if _no_overlap:
            print(f'[stats] {label}: seed(s) {_no_overlap} have NO eval step in common between `csa_fixed` and `full` — they enter no crossover statistic and are excluded from the reported n')
        if not _paired:
            print(f'[stats] {label}: no seed pairs `csa_fixed` against `full` on any shared eval step — the panel carries no crossover trajectory')
            out[label] = {'status': 'missing', **p}
            continue
        common = _paired
        tps = _tokens_per_step(p['outdir'])
        if tps is not None and tps != p['batch'] * p['seq']:
            print(f'[stats] {label}: panel tokens/step is {tps:g} but its declared batch*seq is {p['batch'] * p['seq']} — using the calibrated rate')
        per_seed = {}
        for s in common:
            da, db = (dict(ha[s]), dict(hb[s]))
            steps = sorted(set(da) & set(db))
            gap = [da[t] - db[t] for t in steps]
            xo, crossed = _crossover_step(steps, gap)
            rec = {'crossover_step': xo, 'crossed': crossed, 'n_evals': len(steps), 'last_step': steps[-1], 'final_gap': gap[-1]}
            if tps is not None and crossed and math.isfinite(xo):
                rec['crossover_tokens'] = xo * tps
            per_seed[s] = rec
        steps = sorted(set.intersection(*[set(dict(ha[s]).keys()) for s in common], *[set(dict(hb[s]).keys()) for s in common]))
        gap_mean = [float(np.mean([dict(ha[s])[t] for s in common]) - np.mean([dict(hb[s])[t] for s in common])) for t in steps]
        xo_m, crossed_m = _crossover_step(steps, gap_mean)
        entry = {'status': 'ok', **p, 'seeds': common, 'tokens_per_step': tps, 'per_seed': per_seed, 'per_seed_synth': {str(s): bool(getattr(ha, 'per_seed_synth', {}).get(s) or getattr(hb, 'per_seed_synth', {}).get(s)) for s in common}, 'mean_crossover_step': xo_m, 'mean_crossed': crossed_m, 'mean_traj': {'steps': steps, 'gap': gap_mean}}
        if tps is not None and crossed_m and math.isfinite(xo_m):
            entry['mean_crossover_tokens'] = xo_m * tps
        elif tps is None:
            print(f'[stats] {label}: no tokens/step on this panel — `crossover_tokens` omitted (step positions still published)')
        else:
            print(f'[stats] {label}: the mean trajectory is censored — `mean_crossover_tokens` omitted (a censored step is a lower bound, not a crossover position)')
        out[label] = entry
    return out

def v10_analysis(out='analysis_v10/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    probe = {}
    sp = PROBE['summary']
    if os.path.exists(sp):
        raw = json.load(open(sp, encoding='utf-8'))
        cells = {}
        _param_conflict = {}
        _usable, _bad_cells = ({}, [])
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
            _usable[_k] = r
        if _bad_cells:
            print(f'[v10 stats] {len(_bad_cells)} record(s) in {sp} carry no variant/seed/ppl_mean and are skipped (they are error or truncation stubs, not measurements): ' + ', '.join((f'{k}({why})' for k, why in _bad_cells[:4])))
        _n_bad_pre = len(_bad_cells)
        for k, r in _usable.items():
            p = r.get('probe_params')
            if p is None:
                continue
            key = (r['variant'], r['arm'], r['eval_len'], r['rho'])
            seen = _param_conflict.setdefault(key, set())
            seen.add(tuple(sorted(p.items())))
        _bad = {k: sorted(v) for k, v in _param_conflict.items() if len(v) > 1}
        if _bad:
            raise ValueError('v10_analysis: the P3MP probe panel mixes probe parameters under one cell key, so its mean and its paired contrasts would difference two different measurements. Re-probe the cell(s): ' + '; '.join((f'{k}' for k in sorted(_bad)[:3])))
        cells, _dup_cells = ({}, [])
        for k, r in _usable.items():
            if r.get('probe_params') is None:
                _bad_cells.append((k, 'no probe_params'))
                continue
            key = (r['variant'], r['arm'], r['eval_len'], r['rho'], str(r.get('_code')))
            grp = cells.setdefault(key, {})
            if r['seed'] in grp:
                _dup_cells.append((k, key, r['seed']))
                continue
            grp[r['seed']] = r
        if _dup_cells:
            print(f'[v10 stats] {len(_dup_cells)} probe record(s) share a (cell, seed) identity — their PPL is ambiguous and they are dropped from the pooling: ' + ', '.join((f'{k}' for k, _c, _s in _dup_cells[:4])))
        _new_bad = _bad_cells[_n_bad_pre:]
        if _new_bad:
            print(f'[v10 stats] {len(_new_bad)} record(s) in {sp} lack `probe_params` and are dropped from every mean/paired statistic: ' + ', '.join((f'{k}({why})' for k, why in _new_bad[:4])))
        probe_cells = []
        for (v, arm, Ln, rho, _cd), by_seed in sorted(cells.items(), key=lambda kv: kv[0]):
            means = {s: r['ppl_mean'] for s, r in by_seed.items()}
            assert means, (v, arm, Ln, rho)
            probe_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'seeds': sorted(means), 'ppl_by_seed': means, 'mean': float(np.mean(list(means.values()))), 'n': len(means)})
        contrasts = {}

        def cell_mean(v, arm, Ln, rho):
            hits = [c for c in probe_cells if (c['variant'], c['arm'], c['eval_len'], c['rho']) == (v, arm, Ln, rho)]
            if len(hits) > 1:
                print(f'[v10 stats] ({v}/{arm}/L{Ln}/r{rho}) holds {len(hits)} probe parameterisations — no unambiguous cell, so the contrast is omitted rather than pooled across them')
                return {}
            return hits[0]['ppl_by_seed'] if hits else {}
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho)
                for arm, tag in (('dense', 'full_rope(dense)'), ('randidx', 'csa+randidx'), ('allblocks', 'csa+allblocks')):
                    other = cell_mean('full_rope', 'dense', Ln, rho) if arm == 'dense' else cell_mean('csa_fixed_rope', arm, Ln, rho)
                    common = sorted(set(learned) & set(other))
                    if len(common) >= 2:
                        dl = [other[s] - learned[s] for s in common]
                        contrasts[f'L{Ln} r{rho}: {tag} - csa+learned'] = V.exact_sign_permutation(dl)
        probe = {'cells': probe_cells, 'contrasts': contrasts}
    xo = _scale_crossovers()

    def _is_recon(v):
        flags = v.get('per_seed_synth', {})
        return any((flags.get(str(s)) for s in v.get('seeds', [])))
    pts = [(v['d'], v['mean_crossover_tokens']) for v in xo.values() if v.get('status') == 'ok' and v.get('mean_crossed') and ('mean_crossover_tokens' in v) and (not _is_recon(v))]
    dropped = [k for k, v in xo.items() if v.get('status') == 'ok' and v.get('mean_crossed') and ('mean_crossover_tokens' not in v)]
    if dropped:
        print(f'[stats] crossover fit: {sorted(dropped)} crossed but carry no token rate — excluded from the log-log fit')
    recon = sorted((k for k, v in xo.items() if v.get('status') == 'ok' and v.get('mean_crossed') and ('mean_crossover_tokens' in v) and _is_recon(v)))
    if recon:
        print(f'[stats] crossover fit: {recon} carry a token rate but their trajectory is a RECONSTRUCTION, not a measurement — excluded from the log-log fit (the points count independent runs)')
    fit = {}
    if len(pts) >= 2:
        xs = np.log10([p[0] for p in pts])
        ys = np.log10([max(p[1], 1.0) for p in pts])
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
        pred = A @ np.array([slope, intercept])
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        fit = {'n_points': len(pts), 'slope': float(slope), 'intercept': float(intercept), 'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'), 'note': 'log10(crossover_tokens) ~ log10(d_model), untruncated crossovers only', 'excluded_no_rate': sorted(dropped), 'excluded_reconstructed': recon}
    seq2k = _ppl_by_seed('results_lm_v9_seq2k', full=True)
    seq2k_f = V9._paired_records('results_lm_v9_seq2k')
    comparisons = {}

    def add(name, panel, a, b, outdir):
        V9.add_paired(comparisons, name, panel, a, b, who='v10 ', outdir=outdir)
    add('seq2k(bs1): topk8 - m1', seq2k_f, 'csa_fixed_topk8', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk512 - m1', seq2k_f, 'csa_fixed_topk512', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk8 - topk512', seq2k_f, 'csa_fixed_topk8', 'csa_fixed_topk512', 'results_lm_v9_seq2k')
    panel = V8._panel_block
    out_d = {'probe': probe, 'crossover_panels': xo, 'crossover_fit': fit, 'seq2k_bs1_panel': panel(seq2k), 'comparisons': comparisons}
    L.atomic_write_json(out, out_d, indent=2)
    print(f'[v10 stats] wrote {out}')
    for k, v in comparisons.items():
        if 'omitted' in v:
            print(f'  {k:38s} omitted — {v['omitted']}')
            continue
        print(f'  {k:38s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}')
    for label, v in xo.items():
        if v.get('status') == 'ok':
            if not v['mean_crossed']:
                _stps = v['mean_traj']['steps']
                if _stps:
                    print(f'  crossover {label:10s} mean-traj: CENSORED (> {_stps[-1]:g} steps)')
                else:
                    print(f'  crossover {label:10s} mean-traj: no eval step shared by every seed — no censored bound')
            elif 'mean_crossover_tokens' in v:
                print(f'  crossover {label:10s} mean-traj: step {v['mean_crossover_step']:g} ({v['mean_crossover_tokens']:g} tokens)')
            else:
                print(f'  crossover {label:10s} mean-traj: step {v['mean_crossover_step']:g} (tokens/step unknown)')
    return out_d

def _fmt_pm(cell, std=None):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        mean, std = (cell['mean'], cell.get('std', 0.0))
    else:
        mean = cell
    return f'{mean:.2f} ±{std or 0.0:.2f}'

def build_report(out='REPORT_v10.md'):
    stats_p = 'analysis_v10/stats.json'
    if not os.path.exists(stats_p):
        v10_analysis()
    st = json.load(open(stats_p, encoding='utf-8'))
    probe = st['probe']
    xo = st['crossover_panels']
    fit = st['crossover_fit']
    seq2k = st['seq2k_bs1_panel']
    comp = st['comparisons']
    if os.path.exists('autodl_budget_state_v10.json'):
        v10_state = json.load(open('autodl_budget_state_v10.json', encoding='utf-8'))
    else:
        v10_state = {'booked_seconds': 0.0, 'runs': 0}
    price = BUDGET_V10['price_per_hour']
    v10_h = v10_state.get('booked_seconds', 0.0) / 3600.0
    mech_sum = {}
    mp = os.path.join(P3MT_PAYLOAD['outdir'], 'summary.json')
    if os.path.exists(mp):
        mech_sum = json.load(open(mp, encoding='utf-8'))
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v10 补实验报告（外推机制 + 反转点标度 + topk 补种子）')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348（DeepSeek-V4 稀疏注意力的受控复现与机制剖析）')
    A('> 说明：本报告全部数字由 `v10_supp.py report` 从 `results_*/`、`analysis_v10/` 的落盘产物计算得到，无手填数值。')
    A('')
    A(f'**预算**：v10 记账 {v10_state.get('runs', 0)} runs，估算花费 ¥{v10_h * price:.2f} / ¥{BUDGET_V10['total_yuan']:.2f}（AutoDL RTX 4090，按 ¥{price:.2f}/h 记账；v7/v8/v9 台账各自独立冻结）。')
    A('')
    A('v10 落实 v9 收官后评审的三项升级：(1) 把唯一正面发现（CSA+RoPE 长度外推 ×1.07 vs dense-RoPE ×1.57）从附带观察升级为机制性主结果——远距干扰注入实验直接检验「稀疏掩码滤除远距噪声」假设；(2) 新增 d=128/4L 与 d=512/10L 两个规模，与既有 d=256/d=384 面板组成 4 点反转交叉点标度读数（预注册交叉判据）；(3) 把 v9 唯一仍处 p=0.500 下限的面板（seq-2048 topk 扫描）从 n=2 补到 n=4。')
    A('')
    A('---')
    A('')
    A('## P3M 长度外推优势的机制验证（远距干扰注入）')
    A('')
    A('**背景**：v7 P1L 的 ×1.07 vs ×1.57 是全仓库唯一的 sparse 正面发现，但只是附带观察（n=3 时精确符号翻转 p 值下限 0.250）。P1L 权重未持久化，P3MT 先按 v7 P1L 原配方（seq 512、3000 步、3 seeds、AdamW lr 3e-4）重训两臂并保存权重——同时作为 ×1.07/×1.57 对比的独立复现。')
    A('')
    if mech_sum:
        A('**P3MT 复现对照**（train@512 → 同权重 eval；ratio = PPL@L / PPL@512，seed 平均）：')
        A('')
        A('| variant | PPL@512 | PPL@2048 | PPL@4096 | ratio@4096 |')
        A('|---|---|---|---|---|')
        for v in P3MT_PAYLOAD['variants']:
            rows = [r for r in mech_sum.values() if r.get('variant') == v and 'by_len' in r]
            if not rows:
                continue

            def _at(Ln, rows=rows):
                vals = [c['ppl'] for c in L.by_len_cells(rows, Ln)]
                return float(np.mean(vals)) if vals else float('nan')
            p512, p2048, p4096 = (_at(512), _at(2048), _at(4096))
            _ratio = '—' if not (np.isfinite(p512) and np.isfinite(p4096) and (p512 > 0)) else f'×{p4096 / p512:.2f}'
            A(f'| `{v}` | {p512:.1f} | {p2048:.1f} | {p4096:.1f} | {_ratio} |')
        A('')
    A('**P3MP 干扰注入设计**：eval 长度 2048/4096；把**远距上下文**（除最后 512 个干净目标 token 外的全部位置）中比例为 ρ ∈ {0, 1/8, 1/4, 1/2} 的 token 替换为均匀随机 token——同一 (序列, ρ) 的腐坏对所有臂逐字节相同（按 (eval_len, 序列号, ρ) 播种，与臂/模型种子无关），臂间严格配对。只在干净目标区计 PPL。四个臂：')
    A('')
    A('- `full_rope`（dense，必须直面远距噪声）')
    A('- `csa_fixed_rope` 学习选择（原机制）')
    A('- `csa_fixed_rope` + 随机 indexer（选择被打乱，v4 消融开关，eval 时注入）')
    A('- `csa_fixed_rope` + topk=全块（看得见一切，≈ 带压缩的 dense）')
    A('')
    A('「稀疏掩码滤除远距噪声」假设预测：**学习选择臂的 ρ 曲线平坦，其余三臂随 ρ 陡峭退化**。')
    A('')
    cells = probe.get('cells', [])
    if cells:
        lens = sorted({c['eval_len'] for c in cells})
        rhos = PROBE['rhos']
        arm_order = [('full_rope', 'dense'), ('csa_fixed_rope', 'learned'), ('csa_fixed_rope', 'randidx'), ('csa_fixed_rope', 'allblocks')]
        for Ln in lens:
            _n_ln = max([c.get('n', 0) for c in cells if c['eval_len'] == Ln] or [0])
            A(f'**目标区 PPL（eval_len={Ln}，{_n_ln or "?"} seeds 平均）**：')
            A('')
            A('| 臂 | ' + ' | '.join((f'ρ={r:g}' for r in rhos)) + ' |')
            A('|---|' + '---|' * len(rhos))
            for v, arm in arm_order:
                row = []
                for rho in rhos:
                    m = [c['mean'] for c in cells if (c['variant'], c['arm'], c['eval_len'], c['rho']) == (v, arm, Ln, rho)]
                    row.append(f'{m[0]:.2f}' if m else '—')
                A(f'| `{v}`+{arm} | ' + ' | '.join(row) + ' |')
            A('')
    contr = probe.get('contrasts', {})
    if contr:
        A('**配对检验（exact sign-flip，Δ = 对照臂 − 学习选择臂，>0 表示学习选择更抗噪）：**')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) |')
        A('|---|---|---|---|')
        for k, v in contr.items():
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
        A('')

        def _rng(tag):
            _lo, _hi, _nonf = L.finite_range(((k, v.get('mean')) for k, v in contr.items() if tag in k))
            if _nonf:
                print(f'[v10 report] WARNING: {len(_nonf)} contrast(s) in {tag!r} carry a non-finite mean and are excluded from the reported range: {sorted(_nonf)[:3]}')
            return None if _lo is None else (_lo, _hi)

        def _fmt(r, nd=1):
            return 'n/a' if not r else f'{r[0]:+.{nd}f}~{r[1]:+.{nd}f}'

        def _all_sig(tag, alpha=0.05):
            vs = [(k, v) for k, v in contr.items() if tag in k]
            if not vs:
                return False
            return all((v['mean'] > 0 and v.get('p_exact_signflip', 1.0) < alpha for _k, v in vs))
        dn, ab, ri = (_rng('dense'), _rng('allblocks'), _rng('randidx'))
        ri_sig = _all_sig('randidx')
        _n_by_c = [(k, v.get('n', 0)) for k, v in contr.items() if 'omitted' not in v]
        n_min = min((n for _k, n in _n_by_c), default=0)
        _bind = [k for k, n in _n_by_c if n == n_min]
        n_min_ri = min((n for k, n in _n_by_c if 'randidx' in k), default=0)
        p_floor = 2.0 / 2 ** n_min if n_min else float('nan')
        learned_helps = ri_sig
        A(f'**结论（区间实时算自上方配对检验，Δ = 对照臂 − 学习选择臂）：**(1) dense−learned Δ 区间 {_fmt(dn)} PPL，allblocks−learned Δ 区间 {_fmt(ab)} PPL——正值表示「看得见远距噪声就会受害」，支持限制远距注意力暴露带来抗噪性；(2) randidx−learned Δ 区间 {_fmt(ri)} PPL——' + ('全为正值且全部对照达到 p<0.05：学习到的检索对抗噪性有**独立贡献**（强版本成立）。' if learned_helps else f'未能在 p<0.05 上成立（randidx 对照的最小 n={n_min_ri}，全表最小 n={n_min}[{', '.join(sorted(_bind)[:2])}{('…' if len(_bind) > 2 else '')}]，精确符号翻转的 p 下限为 {p_floor:.3f}，即该面板在设计上就无法达到 0.05 显著性），「抗噪性来自学习到的检索质量」的强版本**不成立**；现有数据只支持「抗噪性主要来自稀疏归纳偏置本身」这一较弱表述。') + '）')
        A('')
    A('---')
    A('')
    A('## P3S 反转交叉点的标度读数（4 个模型规模）')
    A('')
    A('**判据（预注册）**：对每个规模的 `csa_fixed − full` 验证 PPL 差距轨迹（eval_every=250 或面板既有网格），取 ±1 点平滑后**首次转正且此后不再回穿**的 eval 步为交叉步；轨迹内未交叉记为删失（> 末步）。token 数 = 步数 × 每步 token。')
    A('')
    A('| 规模 | 面板 | seeds | 交叉步（seed 均值轨迹） | 交叉 token | 状态 |')
    A('|---|---|---|---|---|---|')
    _recon = {k for k, v in xo.items() if v.get('status') == 'ok' and any((v.get('per_seed_synth', {}).get(str(s)) for s in v.get('seeds', [])))}
    for label, v in xo.items():
        if v.get('status') != 'ok':
            A(f'| {label} | `{v['outdir']}` | — | — | — | 数据缺失 |')
            continue
        n = len(v['seeds'])
        _ntag = f'{n}（重构）' if label in _recon else f'{n}'
        if v['mean_crossed']:
            _tok = f'{v['mean_crossover_tokens'] / 1000000.0:.1f}M' if 'mean_crossover_tokens' in v else 'n/a'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | {v['mean_crossover_step']:g} | {_tok} | 已交叉 |')
        else:
            _stps = v['mean_traj']['steps']
            if not _stps:
                A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | — | — | 无公共 eval 步 |')
                continue
            last = _stps[-1]
            _tok = f'{last * v['tokens_per_step'] / 1000000.0:.1f}M' if v.get('tokens_per_step') else 'n/a'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | > {last:g} | > {_tok} | 删失（窗内未交叉） |')
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
    _xo256 = xo.get('d256_L6') or next((v for v in xo.values() if v.get('d') == 256), {})
    _xo256_step = _xo256.get('mean_crossover_step')
    _xo256_txt = f'~{_xo256_step:g} 步' if isinstance(_xo256_step, (int, float)) and math.isfinite(_xo256_step) else '步数未知（该规模无有效交叉读数）'
    A(f'**注意**：d=256 的轨迹取自 `results_lm_v3_long/summary.json`（v6 重构件，见 README 记账说明 #1）——逐种子轨迹不可独立恢复，该规模的逐种子交叉步互为副本，只有 seed 均值轨迹的交叉步（{_xo256_txt}，eval 网格 1000 步）是有效读数。')
    A('')
    if fit:
        _excl = []
        if fit.get('excluded_reconstructed'):
            _excl.append(f'{len(fit['excluded_reconstructed'])} 个规模（{', '.join(fit['excluded_reconstructed'])}）的轨迹是重构件而非测量')
        if fit.get('excluded_no_rate'):
            _excl.append(f'{len(fit['excluded_no_rate'])} 个规模无 token 速率')
        A(f'**趋势拟合（仅供参考）**：在 {fit['n_points']} 个已观测交叉点上，log10(交叉token) ~ log10(d) 斜率 {fit['slope']:.2f}（R²={fit['r2']:.2f}）——负斜率 = 规模越大交叉越早。' + (f'点数已扣除{'、'.join(_excl)}；' if _excl else '') + '点数少且部分格子删失，只作方向性证据，不作精确幂律主张。')
        A('')
    A('---')
    A('')
    A('## P3T seq-2048 topk 扫描补种子（n=2 → n=4）')
    A('')
    A('**背景**：v9 P2T 的完整 5 点扫描只有 2 seeds，精确符号翻转 p 值下限 0.500——v9 收官后唯一仍处该下限的面板。v10 按同一配方（seq 2048、bs 1、1500 步）补 seeds 2/3（同一 outdir，resume 跳过 seed 0/1）。')
    A('')
    A('| variant | PPL (mean±std) | n |')
    A('|---|---|---|')
    _p3t_done = 0
    for v in ['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']:
        if v in seq2k:
            p = seq2k[v]
            A(f'| `{v}` | {_fmt_pm(p)} | {p['n']} |')
            _p3t_done += 1
        else:
            A(f'| `{v}` | —（未完成） | 0 |')
    A('')
    A(f'**完成 {_p3t_done}/5 点**。')
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
    A('**注**：n=4 时精确符号翻转的 p 值分辨率为 0.125——仍跨不过 0.05（需 n=6 才有 0.031）；此行补齐的是与 scale 面板同级的统计分辨率，显著性主张仍以效应量 + 方向同向性表述。')
    A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A('| `results_v10_mech/` | P3M：两臂重训权重（ckpt/，*.pt 不入库）+ by-length 复现表 + 干扰注入探针 `distractor.json` |')
    A('| `results_lm_v10_scale_s/` | P3S：d=128/4L 交叉定位面板（8000 步） |')
    A('| `results_lm_v10_scale_l/` | P3S：d=512/10L 交叉定位面板（4000 步） |')
    A('| `results_lm_v9_seq2k/` | P3T：topk 扫描追加 seeds 2/3 后的 4-seed 面板 |')
    A('| `analysis_v10/stats.json` | 探针对比、交叉定位与配对符号翻转检验 |')
    A('| `autodl_budget_state_v10.json` | v10 CostGuard 台账 |')
    A('| `v10_supp.py` | 本阶段驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v10 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V10_NO_PUSH'):
        print(f'[git] push skipped (V10_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V10_NO_SHUTDOWN'):
        print('[v10] shutdown suppressed (V10_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v10] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v10] remaining ¥{guard.remaining_yuan():.2f} (cap ¥{guard.cap_yuan():.2f} @ ¥{guard.price:.2f}/h)')
    git_push('v10: supplementary driver (P3M distractor-injection mechanism + P3S crossover scaling + P3T topk seeds 3/4)')
    all_ok = True
    for pname, _kind, _payload, _seeds, est_h in PHASES:
        rem = guard.remaining_yuan()
        if rem < 1.0:
            print(f'[v10] stopping before {pname}: ¥{rem:.2f} left')
            break
        if not V.cuda_healthy():
            print(f'[v10] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v10 phase {pname} (~{est_h} h est, ¥{rem:.2f} left) =====')
        try:
            run_phase(pname, guard)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v10: phase {pname} results')
    try:
        v10_analysis()
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v10: stats + REPORT_v10.md (analysis_v10)')
    guard.report()
    print('\n[v10] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) tiny lenphase ckpt + distractor probe roundtrip')
    L.set_seed(0)
    payload = dict(P3MT_PAYLOAD, outdir='results_smoke_v10', seeds=[0], steps=40, eval_lens=[512, 1024])
    import shutil
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope']), guard=None, label='v10 smoke')
    pcfg = dict(PROBE, outdir='results_smoke_v10', ckpt_dir='results_smoke_v10/ckpt', summary='results_smoke_v10/distractor.json', variants=['csa_fixed_rope'], seeds=[0], eval_lens=[1024], rhos=[0.0, 0.5], n_seq=4, chunk=2)
    s = run_probe(pcfg, guard=None, label='v10 smoke')
    got = {(r['arm'], r['rho']) for r in s.values()}
    want = {('learned', 0.0), ('learned', 0.5), ('randidx', 0.0), ('randidx', 0.5), ('allblocks', 0.0), ('allblocks', 0.5)}
    assert want <= got, f'probe cells missing: {want - got}'
    print(f'  probe roundtrip OK ({len(s)} cells)')
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    print('[smoke] 2) forward/backward at the two new scales')
    for d, nl, nh in ((128, 4, 4), (512, 10, 16)):
        for v in ('csa_fixed', 'full'):
            cfgs = L.make_layer_cfgs(nl, v)
            m = L.SmallGPT(8192, d, nl, nh, 32, 512, cfgs).to(DEVICE)
            x = torch.randint(0, 8192, (2, 512), device=DEVICE)
            out = m(x)
            loss = F.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
            loss.backward()
            print(f'  d={d} L={nl} {v:10s} out={tuple(out.shape)} loss={loss.item():.3f} params={L.count_params(m) / 1000000.0:.2f}M')
            del m, out, loss, x
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    print('[smoke] 3) zero-GPU analysis/report on current artifacts')
    v10_analysis()
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
        print(__doc__ or f"[v10] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v10] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v10] mode 'phase' needs a phase name, e.g. `python v10_supp.py phase P3MT`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v10] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v10_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v10: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v10: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
