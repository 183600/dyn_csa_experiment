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
RUN_CAP_V8 = dict(L.RUN_CAP)
RUN_CAP_V8.update(total_hours=float(os.environ.get('V8_CAP_HOURS', 45.0 / 2.4)), state_path='run_time_state_v8.json', already_hours=0.0)

def make_guard():
    g = L.TimeGuard(RUN_CAP_V8)
    if not g.state.get('sps_by_class') and os.path.exists('run_time_state_v7.json'):
        try:
            st = json.load(open('run_time_state_v7.json', encoding='utf-8'))
            g.state['sps_by_class'] = st.get('sps_by_class', {})
            g.state['norm_sps'] = st.get('norm_sps')
            g._save()
            print('[v8] TimeGuard calibrated from the v7 speed log')
        except Exception as _e:
            print(f'[v8] WARNING: could not calibrate TimeGuard from the v7 speed log ({type(_e).__name__}: {_e}); falling back to the built-in per-class defaults — time estimates are NOMINAL, not calibrated')
    return g
P1S_CFG = dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full', 'full_matched', 'hybrid_dynamic'])
P2R_CFG = dict(L.RUN_LONG, outdir='results_lm_v8_rope20k', variants=['csa_fixed_rope', 'full_rope'], matched=set(V.PARAM_MATCHED_V7), mlp_match_ref='csa_fixed_rope')
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
        unstamped += (ra.get('run_cfg') is None) + (rb.get('run_cfg') is None)
        reason = L.pair_reason(ra, rb)
        if reason:
            why.add(reason)
            skipped.append(s)
            continue
        dl.append(ra['ppl'] - rb['ppl'])
    if skipped:
        print(f'[stats] {who}{name}: {len(skipped)}/{len(common)} seed(s) NOT paired — {sorted(why)}; excluded from the test')
    if not dl:
        return _omit(f'no seed survived the run_cfg/steps checks ({len(common)} shared, {len(skipped)} rejected: {sorted(why)})')
    res = V.exact_sign_permutation(dl)
    res['seeds'] = [s for s in common if s not in set(skipped)]
    res['n_skipped_config_mismatch'] = len(skipped)
    res['n_unstamped'] = unstamped
    comparisons[name] = res
    return res

def _panel_block(panel):
    out = {}
    for v, d in sorted(panel.items()):
        recs = dict(d)
        _stale_cfg = False
        if recs and all((isinstance(r, dict) for r in recs.values())):
            by_cfg = {}
            raw_cfg = {}
            for s, r in recs.items():
                _gk = json.dumps(r.get('run_cfg'), sort_keys=True, default=str)
                by_cfg.setdefault(_gk, {})[s] = r
                raw_cfg[_gk] = r.get('run_cfg')
            _cur_sfx = f'_cs{L.CODE_SEMANTICS}'
            _ranked = sorted(by_cfg.items(), key=lambda kv: (not (isinstance(raw_cfg.get(kv[0]), str) and raw_cfg[kv[0]].endswith(_cur_sfx)), -len(kv[1])))
            _kept_cfg = raw_cfg.get(_ranked[0][0])
            if len(by_cfg) > 1:
                keep = _ranked[0][1]
                _ties = [g for _k, g in _ranked[1:] if len(g) == len(keep)]
                print(f'[stats] _panel_block: variant `{v}` spans {len(by_cfg)} distinct run_cfg groups — pooling across configurations is not allowed, so the table keeps only one group ({len(keep)}/{len(recs)} seeds, preferring the one stamped with the current code semantics) and drops the rest')
                if _ties:
                    print(f'[stats] _panel_block: variant `{v}` run_cfg groups TIE at {len(keep)} seed(s) each — the kept group is {sorted(keep)}; the tied alternatives are dropped, so this panel contributes fewer seeds than were measured')
                recs = keep
            _stale_cfg = not (isinstance(_kept_cfg, str) and _kept_cfg.endswith(_cur_sfx))
            if _stale_cfg:
                print(f'[stats] _panel_block: variant `{v}` has NO run_cfg group stamped with the current code semantics ({_cur_sfx}) — the table entry quotes numbers measured by older code and is marked `_stale_cfg` so downstream readers can tell')
        vals = [v0 for v0 in ((r.get('ppl') if isinstance(r, dict) else r) for r in recs.values()) if isinstance(v0, (int, float)) and (not isinstance(v0, bool)) and np.isfinite(v0)]
        ppl_map = {s: (r.get('ppl') if isinstance(r, dict) else r) for s, r in recs.items()}
        if not vals:
            out[v] = {'ppls': ppl_map, 'n': 0, **({'_stale_cfg': True} if _stale_cfg else {})}
            continue
        out[v] = {'ppls': ppl_map, 'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 'n': len(vals), **({'_stale_cfg': True} if _stale_cfg else {})}
    return out

def v8_analysis(out='analysis_v8/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    scale = _ppl_by_seed('results_lm_v5_scale', full=True)
    rope20k = _ppl_by_seed('results_lm_v8_rope20k', full=True)
    abs_long_f = _paired_records('results_lm_v3_long')
    comparisons = {}

    def add(name, panel, a, b, outdir):
        _add_paired(comparisons, name, panel, a, b, who='v8 ', outdir=outdir)
    for a, b in [('csa_fixed', 'full'), ('csa_fixed', 'full_matched'), ('csa_fixed', 'full_sw128_matched'), ('csa_dynamic', 'full'), ('hybrid_dynamic', 'full')]:
        add(f'scale: {a} - {b}', scale, a, b, 'results_lm_v5_scale')
    add('rope20k: csa_fixed_rope - full_rope', rope20k, 'csa_fixed_rope', 'full_rope', 'results_lm_v8_rope20k')
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
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v8] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v8] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state['booked_seconds'] / 3600:.2f} h)')
    git_push('v8: supplementary driver (P1S scale seed completion + P2R RoPE long run)')
    all_ok = True
    for pname, _cfg, _seeds, est_h in PHASES:
        rem = guard.remaining_hours()
        if rem < 1.0 / 2.4:
            print(f'[v8] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            break
        if not V.cuda_healthy():
            print(f'[v8] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v8 phase {pname} (~{est_h} h est) =====')
        try:
            run_phase(pname, guard)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v8: phase {pname} results')
    try:
        v8_analysis()
    except Exception:
        traceback.print_exc()
        all_ok = False
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
        del m, out, loss, x
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print('[smoke] 2) 60-step csa_fixed_rope probe (LONG recipe), for TimeGuard calibration')
    guard = make_guard()
    t_data = time.time()
    train_ids, val_batch, vocab, _, vb = L.load_wikitext(512, 1000000)
    guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    t0 = time.time()
    rec = L.train_variant('csa_fixed_rope', train_ids, val_batch, vocab, seed=0, steps=60, eval_every=30, eval_subset=16, log_every=30, val_bnd=vb)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    guard.record_run(dt, 60, 256, 6, 512, 12, calib_seconds=rec.get('train_time_s'))
    print(f'  60 steps in {dt:.0f}s -> {dt / 60:.3f} s/step (ppl {rec['ppl']:.1f}); booked to the v8 guard for calibration')
    est = guard.estimate_seconds(20000, d=256, n_layers=6, seq_len=512, batch_size=12)
    print(f'  -> 20k-step RoPE run estimate: {est / 3600:.2f} h')
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
