#!/usr/bin/env python3
import json, os, shutil, subprocess, sys, traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
import exp_lib as L
import torch
LONG_GAP_VARIANTS = ['hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn']

def _num_or_none(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None

def _vocab_or_none(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return int(v) if v.is_integer() else v
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None

def git_push(msg):
    if os.environ.get('V6_NO_PUSH'):
        print(f'[git] push skipped (V6_NO_PUSH): {msg}')
        return True
    run = lambda *a: subprocess.run(a, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    run('git', 'add', '-A')
    r = run('git', 'commit', '-m', msg)
    committed = r.returncode == 0
    if not committed and 'nothing to commit' not in r.stdout + r.stderr:
        print(f'[git] commit problem: {(r.stdout + r.stderr)[-300:]}')
    rb = run('git', 'pull', '--rebase', '--autostash', 'origin', run('git', 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip())
    if committed and rb.returncode != 0:
        print(f'[git] rebase onto origin failed: {(rb.stdout + rb.stderr)[-300:]}')
    r = run('git', 'push', 'origin', 'HEAD')
    ok = r.returncode == 0
    tail = (r.stdout + r.stderr).strip()[-300:]
    print(f'[git] push {('OK' if ok else 'FAILED')}: {msg}' + ('' if ok else f'  ({tail})'))
    if not ok and committed:
        print('[git] WARNING: the commit is only on this instance — the work is NOT on the remote.  Re-push before shutting down.')
    return ok

def synthesize_long_summary():
    adir = os.path.join(REPO, 'results_lm_v3_long')
    agg_path = os.path.join(adir, 'aggregate.json')
    if not os.path.exists(agg_path):
        print(f'[synth] SKIP: {agg_path} is absent — the panel is a LOCAL experiment artifact and is not tracked in the repository, so there is nothing to synthesize from.  Run the v3 panel (or restore the artifact) if you need this reconstruction.')
        return {'skipped_missing_aggregate': agg_path}
    agg = json.load(open(agg_path, encoding='utf-8'))
    spath = os.path.join(adir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[synth] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    n_synth = n_kept = n_replaced = n_backfill = 0
    _unaligned = []
    for v, e in agg.items():
        _ppls = e.get('ppls') or []
        _seeds = e.get('seeds')
        _real = sorted((r.get('seed') for r in summary.values() if isinstance(r, dict) and r.get('variant') == v and (r.get('seed') is not None)))
        if _ppls and _seeds is None:
            _unaligned.append((v, len(_ppls), _real))
            continue
        if not _real:
            continue
        if _ppls and (len(_seeds) != len(_ppls) or sorted((int(s) for s in _seeds)) != _real):
            _unaligned.append((v, len(_ppls), _real))
    if _unaligned:
        for _v, _n, _real in _unaligned:
            print(f"[synth] REFUSING to synthesize `{_v}`: the aggregate holds {_n} PPL value(s) but carries no verified seed->ppl mapping (real per-seed records at seeds {_real}).  `enumerate` would attach the wrong seed's number to each key, replacing a genuine measurement with a copy.  Restore the per-seed records (or re-run the panel) — only this variant is skipped; the aligned variants are still rebuilt.")
    _unaligned = {v for v, _n, _r in _unaligned}
    for v, e in agg.items():
        if v in _unaligned:
            continue
        _seeds = e.get('seeds')
        for i, p in enumerate(e.get('ppls', [])):
            _seed = int(_seeds[i]) if _seeds is not None else i
            key = f'{v}::seed{_seed}'
            old = summary.get(key)
            if isinstance(old, dict) and 'ppl' in old and (old.get('steps') == 20000) and (not old.get('synthesized')):
                n_kept += 1
                if old.get('tokens_seen') is None and e.get('tokens_seen') is not None:
                    old['tokens_seen'] = e['tokens_seen']
                    n_backfill += 1
                continue
            if isinstance(old, dict) and 'ppl' in old and (not old.get('synthesized')):
                n_kept += 1
                print(f"[synth] keeping real record `{key}` (steps={old.get('steps')}); the aggregate's reconstructed value is NOT written over it")
                continue
            if isinstance(old, dict) and 'ppl' in old:
                n_replaced += 1
            rec = {'variant': v, 'seed': _seed, 'steps': 20000, 'tokens_seen': e.get('tokens_seen'), 'ppl': float(p), 'params': _num_or_none(e.get('params')), 'synthesized': True, 'note': 'reconstructed from committed aggregate.json (v6); means exact, secondary-metric stds approximate'}
            if 'ppl_curve' in e:
                rec['ppl_history'] = e['ppl_curve']
            summary[key] = rec
            n_synth += 1
    L.atomic_write_json(spath, summary, indent=2)
    print(f'[v6-synth] results_lm_v3_long/summary.json rebuilt: {n_synth} synthesized, {n_kept} real kept ({n_backfill} of them had `tokens_seen` backfilled from the aggregate so they stay pairable), {n_replaced} stale replaced')
    return summary

def schedule_shutdown(delay_s=90):
    if os.environ.get('V6_NO_SHUTDOWN'):
        print('[v6] shutdown suppressed (V6_NO_SHUTDOWN)')
        return
    marker = 'v6autoshutdown'
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; echo {marker}; shutdown'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v6] AutoDL instance will SHUT DOWN in {delay_s}s — only the data disk keeps billing afterwards. (cancel: pkill -f {marker})')

def run_full():
    guard = L.CostGuard(L.BUDGET)
    guard.report()
    synthesize_long_summary()
    push_ok = git_push('v6: rebuild results_lm_v3_long/summary.json from committed aggregate')
    L.run({**L.RUN_LONG, 'variants': LONG_GAP_VARIANTS}, seeds=[2], guard=guard, label='P-LONG hybrid seed2 (complete 3-seed table)')
    push_ok &= git_push('v6: results_lm_v3_long hybrid variants at seed 2 — 3-seed long-run table complete')
    L.run(L.RUN, seeds=[0, 1, 2], guard=guard, label='P-CORE 1500-step core table (results_lm_v3_1500)')
    push_ok &= git_push('v6: results_lm_v3_1500 — 9-variant core table, 1500 steps x 3 seeds')
    L.run(L.RUN_SCALE, seeds=[0, 1], guard=guard, label='P-SCALE d=384/8L/seq1024 probe (results_lm_v5_scale)')
    push_ok &= git_push('v6: results_lm_v5_scale — model-scale probe (d=384, 8L, seq1024, 5 variants x 2 seeds)')
    guard.report()
    print('\n[v6] ALL PHASES DONE.')
    schedule_shutdown(90 if push_ok else 2400)
    if not push_ok:
        print('[v6] WARNING: final push failed — instance stays up 40 min for a retry, then shuts down. Results are safe on the data disk either way.')

def run_smoke():
    synthesize_long_summary()
    smoke_cfg = dict(L.RUN, outdir='results_smoke', variants=['full', 'csa_fixed', 'hybrid_dynamic'], steps=60, eval_every=30, n_train_tokens=1000000, seeds=[0])
    L.run(smoke_cfg, seeds=[0], guard=None, label='SMOKE (60 steps)')
    shutil.rmtree('results_smoke', ignore_errors=True)
    print('\n[smoke] PASSED — the full pipeline works end to end. The synthesized results_lm_v3_long/summary.json is kept (it is the desired repo state); smoke outputs were deleted.')
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    if mode in ('-h', '--help'):
        print(__doc__ or '')
        raise SystemExit(0)
    if mode not in ('smoke', 'full'):
        print(f"[v6] unknown mode {mode!r}; expected 'smoke' or 'full' (no argument means 'full').  Nothing was started.")
        raise SystemExit(2)
    try:
        _gpu = torch.cuda.get_device_name(0)
    except Exception:
        _gpu = 'cpu'
    print(f'[v6] mode = {mode}   repo = {REPO}   device = {L.DEVICE} ({_gpu})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'full':
            run_full()
        else:
            raise SystemExit(f'unknown mode: {mode}')
    except Exception:
        traceback.print_exc()
        if mode == 'full':
            print('\n[v6] CRASHED — attempting a partial-results push, then the instance stays up 40 min for remote debugging before auto-shutdown (cancel with: pkill -f v6autoshutdown).')
            try:
                git_push('v6: PARTIAL — run_gapfill crashed, see run_v6.log (resume by re-running: completed runs are skipped)')
            except Exception:
                traceback.print_exc()
            schedule_shutdown(2400)
        else:
            print(f'\n[v6] CRASHED in `{mode}` mode — the tree can hold half-written throwaway output, so NOTHING was committed or pushed and no shutdown was scheduled.  Fix the defect and re-run; `smoke` exists to catch exactly this.')
        sys.exit(1)
