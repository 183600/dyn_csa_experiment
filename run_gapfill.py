#!/usr/bin/env python3
import json, math, os, shutil, statistics, subprocess, sys, traceback
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
_LONG_STEPS = L.RUN_LONG['steps']

def _num_or_none(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None

def git_push(msg):
    if os.environ.get('V6_NO_PUSH'):
        print(f'[git] push skipped (V6_NO_PUSH): {msg}')
        return True
    run = lambda *a: subprocess.run(a, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    ra = run('git', 'add', '-A')
    if ra.returncode != 0:
        print(f'[git] add FAILED: {(ra.stdout + ra.stderr)[-300:]} — NOT pushing (a commit could miss unstaged work); leaving the tree for inspection')
        return False
    r = run('git', 'commit', '-m', msg)
    committed = r.returncode == 0
    if not committed and 'nothing to commit' not in r.stdout + r.stderr:
        print(f'[git] commit problem: {(r.stdout + r.stderr)[-300:]}')
        if run('git', 'status', '--porcelain').stdout.strip():
            print('[git] the commit failed while changes are still uncommitted — NOT pushing (a push would publish stale history and report success); leaving the tree for inspection')
            return False
    _branch = run('git', 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    if _branch != 'HEAD':
        rb = run('git', 'pull', '--rebase', '--autostash', 'origin', _branch)
        if rb.returncode != 0:
            print(f'[git] rebase onto origin failed: {(rb.stdout + rb.stderr)[-300:]}')
            run('git', 'rebase', '--abort')
        r = run('git', 'push', 'origin', 'HEAD')
    else:
        _up = run('git', 'symbolic-ref', '-q', 'refs/remotes/origin/HEAD').stdout.strip()
        _dst = _up[len('refs/remotes/origin/'):] if _up.startswith('refs/remotes/origin/') else ''
        if not _dst or run('git', 'rev-parse', '--verify', '-q', f'refs/remotes/origin/{_dst}').returncode != 0:
            _dst = 'main'
        print(f'[git] detached HEAD — pushing explicitly to origin/{_dst}')
        r = run('git', 'push', 'origin', f'HEAD:{_dst}')
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
        return ({'skipped_missing_aggregate': agg_path}, None)
    try:
        agg = json.load(open(agg_path, encoding='utf-8'))
    except Exception as _e:
        print(f'[synth] SKIP: {agg_path} exists but cannot be parsed ({type(_e).__name__}: {_e}) — refusing to rebuild summary.json from a corrupt aggregate; re-derive the aggregate first.')
        return ({'skipped_unreadable_aggregate': agg_path}, None)
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
    _split = []
    _cfg_refused = []

    def _canon(v, e):
        if '@' in str(v):
            return None
        real_v = e.get('variant') if isinstance(e, dict) else None
        if not isinstance(real_v, str) or not real_v:
            real_v = str(v).split('#')[0]
        _proto = e.get('protocol') if isinstance(e, dict) else None
        return (real_v, str(_proto) if _proto else None)
    for v, e in agg.items():
        _ppls = e.get('ppls') or []
        _seeds = e.get('seeds')
        if not _ppls:
            continue
        _c = _canon(v, e)
        if _c is None:
            _split.append(v)
            continue
        real_v, _proto = _c
        _real = sorted({_num_or_none(r.get('seed')) for r in summary.values() if isinstance(r, dict) and r.get('variant') == real_v and (r.get('seed') is not None)} - {None})
        _exp_tok = _LONG_STEPS * int(L.RUN_LONG['batch_size']) * int(L.RUN_LONG['seq_len'])
        _ts = _num_or_none(e.get('tokens_seen'))
        if _ts is not None and _ts != _exp_tok:
            print(f"[synth] REFUSING to synthesize `{v}`: the aggregate's tokens_seen={_ts} does not match the current long recipe ({_LONG_STEPS} steps x {int(L.RUN_LONG['batch_size'])} x {int(L.RUN_LONG['seq_len'])} = {_exp_tok} tokens) — the snapshot was produced under a different configuration and reconstructing under the current step count would mislabel it.  Re-run the panel; only this variant is skipped.")
            _cfg_refused.append(v)
            continue
        if _seeds is None or len(_seeds) != len(_ppls):
            _unaligned.append((v, len(_ppls), _real))
            continue
        _agg_map = {}
        _bad_map = False
        for _si, _pi in zip(_seeds, _ppls):
            _si = _num_or_none(_si)
            if _si is None or not L.ppl_is_usable(_pi):
                _bad_map = True
                break
            _pi = float(_pi)
            if _si in _agg_map and abs(_agg_map[_si] - _pi) > 1e-06 * max(1.0, abs(_pi)):
                _bad_map = True
                break
            _agg_map[_si] = _pi
        _conflict = [_rs for _r in summary.values() if isinstance(_r, dict) and _r.get('variant') == real_v and (_r.get('seed') is not None) and (not _r.get('synthesized')) and (_r.get('steps') == _LONG_STEPS) and L.ppl_is_usable(_r.get('ppl')) for _rs in [_num_or_none(_r.get('seed'))] if _rs is None or _rs not in _agg_map or abs(float(_r['ppl']) - _agg_map[_rs]) > 1e-06 * max(1.0, abs(_agg_map[_rs]))]
        if _bad_map or _conflict:
            _unaligned.append((v, len(_ppls), _real))
    if _unaligned:
        for _v, _n, _real in _unaligned:
            print(f"[synth] REFUSING to synthesize `{_v}`: the aggregate holds {_n} PPL value(s) but carries no verified seed->ppl mapping (real per-seed records at seeds {_real}).  `enumerate` would attach the wrong seed's number to each key, replacing a genuine measurement with a copy.  Restore the per-seed records (or re-run the panel) — only this variant is skipped; the aligned variants are still rebuilt.")
    for _v in _split:
        print(f'[synth] REFUSING to synthesize `{_v}`: this aggregate key is a run_cfg-split group, so it cannot be mapped back to one canonical per-seed record — re-run the panel to get per-seed records; only this key is skipped')
    _unaligned = {v for v, _n, _r in _unaligned} | set(_split) | set(_cfg_refused)
    for v, e in agg.items():
        if v in _unaligned:
            continue
        _c = _canon(v, e)
        if _c is None:
            continue
        real_v, _proto = _c
        _seeds = e.get('seeds')
        for i, p in enumerate(e.get('ppls', [])):
            _seed = _num_or_none(_seeds[i])
            if _seed is None:
                print(f"[synth] REFUSING one reconstructed value of `{v}`: its seed ({_seeds[i]!r}) is not an integer, so it cannot be attached to any per-seed key without aliasing — skipping this value only")
                continue
            key = f'{real_v}::{_proto}::seed{_seed}' if _proto else f'{real_v}::seed{_seed}'
            old = summary.get(key)
            if isinstance(old, dict) and L.ppl_is_usable(old.get('ppl')) and (old.get('steps') == _LONG_STEPS) and (not old.get('synthesized')):
                n_kept += 1
                if old.get('tokens_seen') is None and e.get('tokens_seen') is not None:
                    old['tokens_seen'] = e['tokens_seen']
                    n_backfill += 1
                continue
            if isinstance(old, dict) and L.ppl_is_usable(old.get('ppl')) and (not old.get('synthesized')):
                n_kept += 1
                print(f"[synth] keeping real record `{key}` (steps={old.get('steps')}); the aggregate's reconstructed value is NOT written over it")
                continue
            _twin = next((_k for _k, _r in summary.items() if _k != key and isinstance(_r, dict) and _r.get('variant') == real_v and str(_r.get('protocol') or '') == str(_proto or '') and (_num_or_none(_r.get('seed')) == _seed) and (not _r.get('synthesized')) and (_r.get('steps') == _LONG_STEPS) and L.ppl_is_usable(_r.get('ppl')) and (abs(float(_r['ppl']) - float(p)) <= 1e-06 * max(1.0, abs(float(p))))), None)
            if _twin is not None:
                n_kept += 1
                print(f'[synth] real record `{_twin}` already covers `({real_v}, seed {_seed})` under a different key — the reconstructed value is not duplicated next to it')
                continue
            if isinstance(old, dict) and 'ppl' in old:
                n_replaced += 1
            rec = {'variant': real_v, 'seed': _seed, 'steps': _LONG_STEPS, 'tokens_seen': e.get('tokens_seen'), 'ppl': float(p), 'params': _num_or_none(e.get('params')), 'synthesized': True, 'note': 'reconstructed from committed aggregate.json (v6); means exact, secondary-metric stds approximate'}
            if _proto:
                rec['protocol'] = _proto
            if 'ppl_curve' in e:
                rec['ppl_history'] = e['ppl_curve']
            summary[key] = rec
            n_synth += 1
    L.atomic_write_json(spath, summary, indent=2)
    print(f'[v6-synth] results_lm_v3_long/summary.json rebuilt: {n_synth} synthesized, {n_kept} real kept ({n_backfill} of them had `tokens_seen` backfilled from the aggregate so they stay pairable), {n_replaced} stale replaced')
    return (summary, agg)

def merge_long_aggregate(adir, snapshot_agg):
    if not snapshot_agg:
        return
    agg_path = os.path.join(adir, 'aggregate.json')
    try:
        fresh = json.load(open(agg_path, encoding='utf-8'))
    except Exception as _e:
        print(f'[v6-merge] WARNING: cannot read the freshly written {agg_path} ({type(_e).__name__}: {_e}) — the snapshot is left unmerged; re-run `_finish_panel` on the panel to rebuild it')
        return
    merged = json.loads(json.dumps(snapshot_agg))

    def _seed_map(entry):
        out = {}
        if not isinstance(entry, dict):
            return out
        for _s, _p in zip(entry.get('seeds') or [], entry.get('ppls') or []):
            _s = _num_or_none(_s)
            if _s is not None and L.ppl_is_usable(_p):
                out[_s] = float(_p)
        return out
    n_folded = 0
    _folded_keys = set()
    for key, fentry in fresh.items():
        if not isinstance(fentry, dict):
            continue
        sentry = merged.get(key)
        if not isinstance(sentry, dict):
            merged[key] = fentry
            continue
        sm = _seed_map(sentry)
        fm = _seed_map(fentry)
        _overlap = set(sm) & set(fm)
        n_old, n_new = (len(sm) - len(_overlap), len(fm))
        sm.update(fm)
        if not sm:
            continue
        seeds_sorted = sorted(sm)
        vals = [sm[_s] for _s in seeds_sorted]
        sentry['seeds'] = seeds_sorted
        sentry['ppls'] = vals
        sentry['n_seeds'] = len(vals)
        sentry['ppl_mean'] = sum(vals) / len(vals)
        sentry['ppl_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        n_folded += 1
        _folded_keys.add(key)
        if n_old and n_new:
            w_old = n_old / (n_old + n_new)
            for _f in ('avg_dyn_block_len', 'avg_block_len', 'boundary_f1_dyn', 'boundary_excess_dyn', 'delta_logit_mean', 'tokens_per_step'):
                _vo, _vn = (sentry.get(_f), fentry.get(_f))
                if isinstance(_vo, (int, float)) and isinstance(_vn, (int, float)) and (not isinstance(_vo, bool)) and (not isinstance(_vn, bool)) and math.isfinite(_vo) and math.isfinite(_vn):
                    sentry[_f] = _vo * w_old + _vn * (1.0 - w_old)
            _co, _cn = (sentry.get('ppl_curve'), fentry.get('ppl_curve'))
            if _co and _cn:
                _cm = {}
                for _pt in _co:
                    try:
                        _cm[int(_pt[0])] = [float(_pt[1]) * n_old, n_old]
                    except (TypeError, ValueError, IndexError):
                        continue
                for _pt in _cn:
                    try:
                        _s, _v = (int(_pt[0]), float(_pt[1]))
                    except (TypeError, ValueError, IndexError):
                        continue
                    _acc = _cm.setdefault(_s, [0.0, 0])
                    _acc[0] += _v * n_new
                    _acc[1] += n_new
                sentry['ppl_curve'] = [[_s, _a / _w] for _s, (_a, _w) in sorted(_cm.items()) if _w]
    if _folded_keys:
        _base_full = _seed_map(merged.get('full'))
        _base_sw = _seed_map(merged.get('full_sw128_matched'))

        def _repair_delta(entry, base_map, mean_f, std_f, n_f):
            if not base_map:
                return
            vm = _seed_map(entry)
            common = sorted(set(vm) & set(base_map))
            if not common:
                return
            ds = [vm[_s] - base_map[_s] for _s in common]
            entry[mean_f] = sum(ds) / len(ds)
            entry[std_f] = statistics.stdev(ds) if len(ds) > 1 else 0.0
            entry[n_f] = len(ds)
        for key in _folded_keys:
            entry = merged.get(key)
            if not isinstance(entry, dict):
                continue
            _v = entry.get('variant') or str(key).split('#')[0].split('@')[0]
            if _v != 'full':
                _repair_delta(entry, _base_full, 'dPPL_vs_full_mean', 'dPPL_vs_full_std', 'n_paired')
            if _v not in ('full', 'full_sw128_matched'):
                _repair_delta(entry, _base_sw, 'dPPL_vs_sw128m_mean', 'dPPL_vs_sw128m_std', 'n_paired_sw128m')
    L.atomic_write_json(agg_path, merged, indent=2)
    try:
        L._save_csv(merged, adir)
    except Exception as _e:
        print(f'[v6-merge] WARNING: aggregate.json was merged but the CSV could not be refreshed ({type(_e).__name__}: {_e}) — regenerate it from aggregate.json before quoting the table')
    print(f'[v6-merge] aggregate.json: {n_folded} measured entr(ies) folded into the snapshot — per-seed PPL means/stds, the paired deltas, the seed-averaged curves and the pooled secondary means now cover every measured seed (measured values win per seed; secondary-metric stds keep their snapshot values, see the `note` on the synthesized records in summary.json)')

def schedule_shutdown(delay_s=90):
    if os.environ.get('V6_NO_SHUTDOWN'):
        print('[v6] shutdown suppressed (V6_NO_SHUTDOWN)')
        return
    marker = 'v6autoshutdown'
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; echo {marker}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v6] AutoDL instance will SHUT DOWN in {delay_s}s — only the data disk keeps billing afterwards. (cancel: pkill -f {marker})')

def run_full():
    guard = L.CostGuard(L.BUDGET)
    guard.report()
    _synth_summary, _long_snapshot = synthesize_long_summary()
    push_ok = git_push('v6: rebuild results_lm_v3_long/summary.json from committed aggregate')
    L.run({**L.RUN_LONG, 'variants': LONG_GAP_VARIANTS}, seeds=[2], guard=guard, label='P-LONG hybrid seed2 (complete 3-seed table)')
    merge_long_aggregate(os.path.join(REPO, 'results_lm_v3_long'), _long_snapshot)
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
    smoke_cfg = dict(L.RUN, outdir='results_smoke', variants=['full', 'csa_fixed', 'hybrid_dynamic'], steps=60, eval_every=30, n_train_tokens=1000000, seeds=[0])
    L.run(smoke_cfg, seeds=[0], guard=None, label='SMOKE (60 steps)')
    _sp = os.path.join('results_smoke', 'summary.json')
    _got = {}
    if os.path.exists(_sp):
        try:
            _got = json.load(open(_sp, encoding='utf-8'))
        except Exception:
            _got = {}
    _bad = [f'{v}::seed0' for v in smoke_cfg['variants'] if not L.ppl_is_usable((_got.get(f'{v}::seed0') or {}).get('ppl'))]
    if _bad:
        print(f'\n[smoke] FAILED: no usable PPL was measured for {_bad} — the pipeline is degraded; results_smoke/ is KEPT for inspection instead of being deleted.')
        sys.exit(1)
    shutil.rmtree('results_smoke', ignore_errors=True)
    print('\n[smoke] PASSED — the full pipeline works end to end; smoke outputs were deleted and no experiment directory was touched.')
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
