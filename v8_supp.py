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
import v7_supp as V
L = V.L
REPO = V.REPO
DEVICE = L.DEVICE
BUDGET_V8 = dict(L.BUDGET)
BUDGET_V8.update(total_yuan=float(os.environ.get('V8_BUDGET_YUAN', 45.0)), price_per_hour=float(os.environ.get('V8_PRICE_PER_HOUR', 2.4)), state_path='autodl_budget_state_v8.json', already_spent_yuan=0.0)

def make_guard():
    g = L.CostGuard(BUDGET_V8)
    if not g.state.get('sps_by_class') and os.path.exists('autodl_budget_state_v7.json'):
        try:
            st = json.load(open('autodl_budget_state_v7.json', encoding='utf-8'))
            g.state['sps_by_class'] = st.get('sps_by_class', {})
            g.state['norm_sps'] = st.get('norm_sps')
            g._save()
            print('[v8] CostGuard calibrated from the v7 speed ledger')
        except Exception as _e:
            print(f'[v8] WARNING: could not calibrate CostGuard from the v7 speed ledger ({type(_e).__name__}: {_e}); falling back to the built-in per-class defaults — budget estimates are NOMINAL, not calibrated')
    return g
P1S_CFG = dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full', 'full_matched', 'hybrid_dynamic'])
P2R_CFG = dict(L.RUN_LONG, outdir='results_lm_v8_rope20k', variants=['csa_fixed_rope', 'full_rope'])
PHASES = [('P1S', P1S_CFG, [0, 1, 2, 3], 3.0), ('P2R', P2R_CFG, [0, 1], 11.0)]

def run_phase(name, guard):
    for pname, cfg, seeds, _h in PHASES:
        if pname != name:
            continue
        s, _a = L.run(cfg, seeds=seeds, guard=guard, label=f'v8 {pname}')
        return s
    raise SystemExit(f'unknown phase {name}')

def _ppl_by_seed(outdir, full=False):
    return L.ppl_by_seed(outdir, full=full)

def _paired_records(outdir):
    return L.ppl_by_seed(outdir, full=True)

def _add_paired(comparisons, name, panel, a, b, who='', outdir=None):

    def _omit(reason):
        print(f'[stats] {who}{name}: comparison OMITTED — {reason}')
        comparisons[name] = {'omitted': reason, 'n': 0}
        return None
    if a not in panel or b not in panel:
        missing = [v for v in (a, b) if v not in panel]
        if outdir is not None:
            pres = L.format_variant_presence(L.variant_presence(outdir), missing)
            if pres is not None:
                return _omit(pres)
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

def _panel_block(panel):
    out = {}
    for v, d in sorted(panel.items()):
        vals = list(d.values())
        if not vals:
            out[v] = {'ppls': d, 'n': 0}
            continue
        out[v] = {'ppls': d, 'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 'n': len(vals)}
    return out

def v8_analysis(out='analysis_v8/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    scale = _ppl_by_seed('results_lm_v5_scale')
    rope20k = _ppl_by_seed('results_lm_v8_rope20k')
    scale_f = _paired_records('results_lm_v5_scale')
    rope20k_f = _paired_records('results_lm_v8_rope20k')
    abs_long_f = _paired_records('results_lm_v3_long')
    comparisons = {}

    def add(name, panel, a, b, outdir):
        _add_paired(comparisons, name, panel, a, b, who='v8 ', outdir=outdir)
    for a, b in [('csa_fixed', 'full'), ('csa_fixed', 'full_matched'), ('csa_fixed', 'full_sw128_matched'), ('csa_dynamic', 'full'), ('hybrid_dynamic', 'full')]:
        add(f'scale: {a} - {b}', scale_f, a, b, 'results_lm_v5_scale')
    add('rope20k: csa_fixed_rope - full_rope', rope20k_f, 'csa_fixed_rope', 'full_rope', 'results_lm_v8_rope20k')
    add('long20k(abs): csa_fixed - full', abs_long_f, 'csa_fixed', 'full', 'results_lm_v3_long')
    out_d = {'scale_panel': _panel_block(scale), 'rope20k_panel': _panel_block(rope20k), 'comparisons': comparisons}
    L.atomic_write_json(out, out_d, indent=2)
    print(f'[v8 stats] wrote {out}')
    for k, v in comparisons.items():
        if 'omitted' in v:
            print(f'  {k:38s} omitted — {v['omitted']}')
            continue
        print(f'  {k:38s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}')
    return out_d

def git_push(msg):
    if os.environ.get('V8_NO_PUSH'):
        print(f'[git] push skipped (V8_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V8_NO_SHUTDOWN'):
        print('[v8] shutdown suppressed (V8_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v8] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v8] remaining ¥{guard.remaining_yuan():.2f} (cap ¥{guard.cap_yuan():.2f} @ ¥{guard.price:.2f}/h)')
    git_push('v8: supplementary driver (P1S scale seed completion + P2R RoPE long run)')
    all_ok = True
    for pname, _cfg, _seeds, est_h in PHASES:
        rem = guard.remaining_yuan()
        if rem < 1.0:
            print(f'[v8] stopping before {pname}: ¥{rem:.2f} left')
            break
        if not V.cuda_healthy():
            print(f'[v8] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v8 phase {pname} (~{est_h} h est, ¥{rem:.2f} left) =====')
        try:
            run_phase(pname, guard)
        except Exception:
            traceback.print_exc()
        all_ok &= git_push(f'v8: phase {pname} results')
    try:
        v8_analysis()
    except Exception:
        traceback.print_exc()
    all_ok &= git_push('v8: paired sign-flip stats (analysis_v8)')
    guard.report()
    print('\n[v8] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) RoPE variant forward/backward (2 layers, seq 128)')
    L.set_seed(0)
    for v in ['full_rope', 'csa_fixed_rope']:
        cfgs = L.make_layer_cfgs(2, v)
        m = L.SmallGPT(8192, 256, 2, 8, 32, 128, cfgs).to(DEVICE)
        x = torch.randint(0, 8192, (2, 128), device=DEVICE)
        import torch.nn.functional as Fn
        out = m(x)
        loss = Fn.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
        loss.backward()
        print(f'  {v:18s} out={tuple(out.shape)} loss={loss.item():.3f} use_abs_pe={getattr(m, 'use_abs_pe', True)}')
        del m
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print('[smoke] 2) 60-step csa_fixed_rope probe (LONG recipe), for CostGuard calibration')
    guard = make_guard()
    train_ids, val_batch, vocab, _, vb = L.load_wikitext(512, 1000000)
    t0 = time.time()
    rec = L.train_variant('csa_fixed_rope', train_ids, val_batch, vocab, seed=0, steps=60, eval_every=30, eval_subset=16, log_every=30, val_bnd=vb)
    dt = time.time() - t0
    guard.record_run(dt, 60, 256, 6, 512, 12)
    print(f'  60 steps in {dt:.0f}s -> {dt / 60:.3f} s/step (ppl {rec['ppl']:.1f}); booked to the v8 guard for calibration')
    est = guard.estimate_seconds(20000, d=256, n_layers=6, seq_len=512, batch_size=12)
    print(f'  -> 20k-step RoPE run estimate: {est / 3600:.2f} h (¥{est / 3600 * guard.price:.2f})')
    print('\n[smoke] PASSED')
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'analysis', 'phase')
    if mode in ('-h', '--help', 'help'):
        print(__doc__ or f"[v8] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v8] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v8] mode 'phase' needs a phase name, e.g. `python v8_supp.py phase P1S`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v8] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            v8_analysis()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v8: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v8: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
