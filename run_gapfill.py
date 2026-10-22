#!/usr/bin/env python3
import json, math, os, shutil, statistics, subprocess, sys, time, traceback
import copy
import numpy as np
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
    # Booleans are flags, not counts/seed identities: float(True) == 1 would
    # silently turn a flag into seed 1 (or a budget of one token).
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(f) if math.isfinite(f) and f.is_integer() else None

def _announce_seed_plan(label, cfg, seeds):
    """Print the prospective family-corrected seed requirement for a phase.

    A panel launched at a seed count below `methodology_seed_plan`'s floor is
    DESCRIPTIVE, however large the observed effect: the exact sign-flip test
    cannot reach alpha under the declared Holm family.  Announce that before
    spending GPU time (and name the top-up path) instead of discovering it when
    the table is interpreted after the run.
    """
    plan = L.methodology_seed_plan(len(list(seeds)), variants=cfg.get('variants'))
    _need = plan.get('min_seeds_for_family_significance')
    print(f'[seed-plan] {label}: n={plan["n_seeds"]}, declared contrasts={plan["declared_contrasts"]}, '
          f'family-corrected minimum n={_need}, sufficient={plan["sufficient_for_family_rejection"]}')
    if not plan.get('sufficient_for_family_rejection'):
        print(f'[seed-plan] WARNING: {label} at n={plan["n_seeds"]} is exploratory/descriptive only '
              f'(need n>={_need} for any Holm-corrected claim). Add seeds with the seed-topup path before '
              'reading a family-corrected p as significant.')
    return plan


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

def _agg_seed_map(entry, tag=''):
    out = {}
    if not isinstance(entry, dict):
        return out
    if len(entry.get('seeds') or []) != len(entry.get('ppls') or []):
        print(f'[synth] NOTE: seeds/ppls length mismatch in aggregate entry{(" for " + tag) if tag else ""} — skipping it instead of silently truncating')
        return out
    for _s, _p in zip(entry.get('seeds') or [], entry.get('ppls') or []):
        _si = _num_or_none(_s)
        if _si is not None and L.ppl_is_usable(_p):
            out[_si] = float(_p)
    return out

def _fold_backup_into_aggregate(adir):
    agg_path = os.path.join(adir, 'aggregate.json')
    bak_path = agg_path + '.v6bak'
    if not os.path.exists(bak_path):
        return
    try:
        bak = json.load(open(bak_path, encoding='utf-8'))
    except Exception as _e:
        print(f'[synth] NOTE: the pre-retrain snapshot {bak_path} cannot be parsed ({type(_e).__name__}: {_e}) — ignoring it')
        return
    if not isinstance(bak, dict):
        print(f'[synth] NOTE: the pre-retrain snapshot {bak_path} is valid JSON '
              f'but not an object ({type(bak).__name__}) — ignoring it')
        return
    live = {}
    if os.path.exists(agg_path):
        try:
            live = json.load(open(agg_path, encoding='utf-8'))
        except Exception:
            live = {}
    if not isinstance(live, dict):
        live = {}
    merged = dict(live)
    recovered = {}
    for key, bentry in bak.items():
        if not isinstance(bentry, dict):
            continue
        lentry = merged.get(key)
        sm_b, sm_l = (_agg_seed_map(bentry, tag=str(key)), _agg_seed_map(lentry, tag=str(key)))
        missing = set(sm_b) - set(sm_l)
        if not missing:
            continue
        if not isinstance(lentry, dict):
            merged[key] = bentry
            recovered[key] = sorted(missing)
            continue
        sm = dict(sm_b)
        sm.update(sm_l)
        seeds_sorted = sorted(sm)
        vals = [sm[_s] for _s in seeds_sorted]
        lentry['seeds'] = seeds_sorted
        lentry['ppls'] = vals
        lentry['n_seeds'] = len(vals)
        lentry['ppl_mean'] = sum(vals) / len(vals)
        lentry['ppl_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # Every other aggregate was computed over the pre-recovery seed subset.
        # It is not safe to keep those values next to a larger `seeds` list, so
        # invalidate them explicitly instead of mixing a 1-seed mean with a
        # 2-seed ppl_mean.
        _derived = ('ppl_primary', 'ppl_primary_estimand', 'ppl_primary_missing',
                    'ppl_aggregation', 'nll_sum', 'n_tokens', 'ppl_curve',
                    'ppl_ci95', 'ppl_ci95_low', 'ppl_ci95_high',
                    'ppl_ci95_seed_t', 'ppl_ci95_doc_bootstrap',
                    'ppl_ci_seed_doc_combined', 'ppl_median', 'ppl_iqr',
                    'ppl_mean_arithmetic', 'dPPL_vs_full', 'dPPL_vs_csa_fixed',
                    'dPPL_vs_full_mean', 'dPPL_vs_full_std', 'n_paired',
                    'dPPL_vs_sw128m_mean', 'dPPL_vs_sw128m_std',
                    'n_paired_sw128m', 'avg_dyn_block_len', 'avg_block_len',
                    'boundary_f1_dyn', 'boundary_excess_dyn',
                    'delta_logit_mean', 'tokens_per_step', 'nll_pooled',
                    'ppl_pooled', 'ppl_pooled_std')
        for _f in _derived:
            lentry[_f] = None
        lentry['seed_derived_fields_invalidated'] = list(_derived)
        recovered[key] = sorted(missing)
    if recovered:
        L.atomic_write_json(agg_path, merged, indent=2)
        print(f'[synth] {agg_path} had lost per-seed coverage (a previous pass was interrupted between re-aggregation and the snapshot merge) — recovered from the pre-retrain snapshot: ' + ', '.join((f'{k}: seeds {v}' for k, v in sorted(recovered.items()))))

def synthesize_long_summary():
    adir = os.path.join(REPO, 'results_lm_v3_long')
    agg_path = os.path.join(adir, 'aggregate.json')
    _fold_backup_into_aggregate(adir)
    if not os.path.exists(agg_path):
        print(f'[synth] SKIP: {agg_path} is absent — the panel is a LOCAL experiment artifact and is not tracked in the repository, so there is nothing to synthesize from.  Run the v3 panel (or restore the artifact) if you need this reconstruction.')
        return ({'skipped_missing_aggregate': agg_path}, None)
    try:
        agg = json.load(open(agg_path, encoding='utf-8'))
    except Exception as _e:
        print(f'[synth] SKIP: {agg_path} exists but cannot be parsed ({type(_e).__name__}: {_e}) — refusing to rebuild summary.json from a corrupt aggregate; re-derive the aggregate first.')
        return ({'skipped_unreadable_aggregate': agg_path}, None)
    if not isinstance(agg, dict):
        print(f'[synth] SKIP: {agg_path} is valid JSON but not an object '
              f'({type(agg).__name__}) — refusing to rebuild summary.json from it.')
        return ({'skipped_non_object_aggregate': agg_path}, None)
    spath = os.path.join(adir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[synth] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
        if not isinstance(summary, dict):
            print(f'[synth] SKIP: {spath} is valid JSON but not an object '
                  f'({type(summary).__name__}) — refusing to treat it as a '
                  'summary mapping.')
            return ({'skipped_non_object_summary': spath}, None)
    n_synth = n_kept = n_replaced = n_backfill = 0
    _unaligned = []
    _split = []
    _cfg_refused = []
    _malformed = []

    def _canon(v, e):
        if '@' in str(v):
            return None
        real_v = e.get('variant') if isinstance(e, dict) else None
        if not isinstance(real_v, str) or not real_v:
            real_v = str(v).split('#')[0]
        _proto = e.get('protocol') if isinstance(e, dict) else None
        return (real_v, str(_proto) if _proto else None)
    for v, e in agg.items():
        if not isinstance(e, dict):
            _malformed.append(v)
            print(f'[synth] REFUSING to synthesize `{v}`: the aggregate entry '
                  f'is a {type(e).__name__}, not an object — re-run the panel '
                  'to get a valid aggregate; only this key is skipped')
            continue
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
    _unaligned = ({v for v, _n, _r in _unaligned} | set(_split)
                  | set(_cfg_refused) | set(_malformed))
    # Group the real (non-synthesized, usable, full-steps) records by
    # (variant, protocol, seed) once, then each reconstructed value only walks
    # its own bucket with the identical match conditions.
    _twin_idx = {}
    for _k, _r in summary.items():
        if not isinstance(_r, dict) or _r.get('synthesized'):
            continue
        if _r.get('steps') != _LONG_STEPS or not L.ppl_is_usable(_r.get('ppl')):
            continue
        _tw_seed = _num_or_none(_r.get('seed'))
        if _tw_seed is None:
            continue
        _twkey = (_r.get('variant'), str(_r.get('protocol') or ''), _tw_seed)
        _twin_idx.setdefault(_twkey, []).append((_k, float(_r['ppl'])))
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
            _twin = None
            _pf = float(p)
            for _k, _tp in _twin_idx.get((real_v, str(_proto or ''), _seed), ()):
                if _k != key and abs(_tp - _pf) <= 1e-06 * max(1.0, abs(_pf)):
                    _twin = _k
                    break
            if _twin is not None:
                n_kept += 1
                print(f'[synth] real record `{_twin}` already covers `({real_v}, seed {_seed})` under a different key — the reconstructed value is not duplicated next to it')
                continue
            if isinstance(old, dict) and L.ppl_is_usable(old.get('ppl')):
                n_replaced += 1
            rec = {'variant': real_v, 'seed': _seed, 'steps': _LONG_STEPS, 'tokens_seen': e.get('tokens_seen'), 'ppl': float(p), 'params': _num_or_none(e.get('params')), 'synthesized': True, 'note': 'reconstructed from the v6 aggregate.json snapshot; means exact, secondary-metric stds approximate'}
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
    if not isinstance(fresh, dict):
        print(f'[v6-merge] WARNING: {agg_path} is valid JSON but not an object '
              f'({type(fresh).__name__}) — the snapshot is left unmerged; re-run '
              '`_finish_panel` on the panel to rebuild it')
        return
    if not isinstance(snapshot_agg, dict):
        print(f'[v6-merge] WARNING: the snapshot aggregate is not an object '
              f'({type(snapshot_agg).__name__}) — using the freshly written '
              'aggregate instead')
        merged = {}
    else:
        merged = copy.deepcopy(snapshot_agg)

    def _seed_map(entry, tag=''):
        out = {}
        if not isinstance(entry, dict):
            return out
        if len(entry.get('seeds') or []) != len(entry.get('ppls') or []):
            print(f'[v6-merge] NOTE: seeds/ppls length mismatch in aggregate entry{(" for " + tag) if tag else ""} — skipping it instead of silently truncating')
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
        sm_old = _seed_map(sentry, tag=str(key))
        fm = _seed_map(fentry, tag=str(key))
        _overlap = set(sm_old) & set(fm)
        # Fresh values replace the old value for an overlapping seed; old-only
        # seeds keep their snapshot value.  Secondary means must therefore be
        # weighted by old-ONLY vs fresh draws, never by all old draws (which
        # would count an overlap seed twice).
        n_old_only = len(set(sm_old) - set(fm))
        n_fresh = len(fm)
        sm = dict(sm_old)
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
        # Every field below was computed over the snapshot's seed subset.
        # Keeping it next to the enlarged seeds/ppls/ppl_mean would mix a
        # 1-seed primary/CI/pooled value with a 3-seed mean.  The curve,
        # secondary means and paired deltas are recomputed below; the rest
        # stay explicitly null until the panel is re-aggregated.
        for _f in ('ppl_primary', 'ppl_primary_estimand', 'ppl_primary_missing',
                   'ppl_aggregation', 'ppl_pooled', 'nll_sum', 'n_tokens',
                   'ppl_median', 'ppl_iqr', 'ppl_mean_arithmetic',
                   'ppl_ci95', 'ppl_ci95_low', 'ppl_ci95_high',
                   'ppl_ci95_seed_t', 'ppl_ci95_doc_bootstrap',
                   'ppl_ci_seed_doc_combined', 'dPPL_vs_full_mean',
                   'dPPL_vs_full_std', 'n_paired', 'dPPL_vs_sw128m_mean',
                   'dPPL_vs_sw128m_std', 'n_paired_sw128m'):
            if _f in sentry:
                sentry[_f] = None
        n_folded += 1
        _folded_keys.add(key)
        if n_fresh:
            _fresh_covers = n_old_only == 0
            _overlap = bool(set(sm_old) & set(fm))
            # Partial overlap is not algebraically mergeable from aggregate
            # means: the snapshot mean covers all old seeds (including the
            # overlapping ones), so it cannot be split into old-only and
            # overlap components without per-seed values.  Weighting it by
            # `n_old_only` silently assumes the overlap mean equals the old
            # mean, which is exactly what a re-measurement may have changed.
            _partial_overlap = bool(_overlap and not _fresh_covers)
            for _f in ('avg_dyn_block_len', 'avg_block_len', 'boundary_f1_dyn', 'boundary_excess_dyn', 'delta_logit_mean', 'tokens_per_step'):
                _vo, _vn = (sentry.get(_f), fentry.get(_f))
                _vn_ok = isinstance(_vn, (int, float)) and (not isinstance(_vn, bool)) and math.isfinite(_vn)
                _vo_ok = isinstance(_vo, (int, float)) and (not isinstance(_vo, bool)) and math.isfinite(_vo)
                if not _vn_ok:
                    continue
                if _partial_overlap:
                    # Do not present an unverifiable seed-average as current.
                    sentry[_f] = None
                elif _fresh_covers or not _vo_ok:
                    sentry[_f] = _vn
                else:
                    w_old = n_old_only / (n_old_only + n_fresh)
                    sentry[_f] = _vo * w_old + _vn * (1.0 - w_old)
            _co, _cn = (sentry.get('ppl_curve'), fentry.get('ppl_curve'))
            if _partial_overlap:
                sentry['ppl_curve'] = None
            elif _fresh_covers:
                if _cn:
                    sentry['ppl_curve'] = _cn
            elif _co and _cn:
                _cm = {}
                for _pt in _co:
                    try:
                        _cm[int(_pt[0])] = [float(_pt[1]) * n_old_only, n_old_only]
                    except (TypeError, ValueError, IndexError):
                        continue
                for _pt in _cn:
                    try:
                        _s, _v = (int(_pt[0]), float(_pt[1]))
                    except (TypeError, ValueError, IndexError):
                        continue
                    _acc = _cm.setdefault(_s, [0.0, 0])
                    _acc[0] += _v * n_fresh
                    _acc[1] += n_fresh
                sentry['ppl_curve'] = [[_s, _a / _w] for _s, (_a, _w) in sorted(_cm.items()) if _w]
            if _partial_overlap:
                sentry['merge_stale_fields'] = 'partial_seed_overlap'
                sentry['merge_note'] = ('snapshot and fresh measurements overlap on only some '
                                        'seeds; per-seed PPLs are merged exactly, but the '
                                        'seed-averaged secondary fields and ppl_curve cannot '
                                        'be re-aggregated from aggregate means alone and were '
                                        'set to null. Re-run the panel to rebuild them.')
    if _folded_keys:
        _base_full = _seed_map(merged.get('full'), tag='full')
        _base_sw = _seed_map(merged.get('full_sw128_matched'), tag='full_sw128_matched')

        def _repair_delta(entry, base_map, mean_f, std_f, n_f):
            if not base_map:
                return
            vm = _seed_map(entry, tag=str(entry.get('variant') if isinstance(entry, dict) else ''))
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
    print(f'[v6] instance will SHUT DOWN in {delay_s}s. (cancel: pkill -f {marker})')

def run_overfit(cfg=None, variants=None, steps=None):
    """Pre-flight small-batch overfit sanity check.

    Runs BEFORE any full panel: one fixed batch, zero weight decay, zero
    compression penalty, no augmentation.  A failure is a hard stop (the full
    run must not start).  Passing is necessary, not sufficient — data
    augmentation / feature-alignment / label-mapping problems can still survive.
    The probe uses the panel's real architecture/MLP and a block-length
    sequence, and failed evidence is persisted before the hard stop."""
    if cfg is None:
        # L.RUN already carries the main-panel outdir, so setdefault never
        # fired and the sanity artifacts polluted the core panel directory.
        cfg = dict(L.RUN)
        cfg['outdir'] = 'results_overfit_sanity'
    else:
        cfg = dict(cfg)
        cfg.setdefault('outdir', 'results_overfit_sanity')
    variants = list(variants or ['full', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic', 'hybrid_hca_dyn', 'csa_dyn_nocomp'])
    # Seed BEFORE the corpus is served — this entry point can be the first
    # thing a process runs (`run_gapfill.py overfit` / the first phase of
    # `full`), and L.run()'s own pre-load set_seed is not reached yet.  The
    # loaders are deterministic today, but the seeding contract (every
    # corpus-construction path runs after a seed) must hold for this entry
    # point too, exactly like it does inside L.run().
    L.set_seed(int((cfg.get('seeds') or [0])[0]))
    train_ids, val_batch, vocab, _, _ = L.load_run_corpus(cfg)
    d = int(cfg.get('d', 256))
    n_layers = int(cfg.get('n_layers', 6))
    n_heads = int(cfg.get('n_heads', 8))
    d_head = int(cfg.get('d_head', 32))
    ratios = {}
    for v in variants:
        try:
            ratios[v] = L.variant_mlp_ratio(v, int(vocab), d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=int(cfg.get('seq_len', 512)), matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
        except Exception:
            ratios[v] = None
    probe_seq = L.probe_seq_len(cfg.get('seq_len', 512))
    arch = {'d': d, 'n_layers': n_layers, 'n_heads': n_heads, 'd_head': d_head}
    print(f'\n[overfit] small-batch sanity check: {len(variants)} variant(s), one fixed batch, wd=0, comp_lambda=0, no augmentation, arch={arch}, seq_len={probe_seq}')
    _prov = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    _pfp = L.preflight_fingerprint(cfg, stage='gapfill_overfit', vocab=int(vocab), data_fp=L._ids_fp(train_ids), probe_cfg={'arch': arch, 'seq_len': probe_seq, 'mlp_ratio': ratios})
    os.makedirs(cfg['outdir'], exist_ok=True)
    try:
        reports = L.overfit_guard(variants, steps=int(steps or 300), train_ids=train_ids,
                                  vocab=vocab, seq_len=probe_seq, arch=arch,
                                  mlp_ratio=ratios, force=True)
    except RuntimeError as _fe:
        _reports = [dict(r) for k, r in L._OVERFIT_REPORTS.items() if k[0] in set(variants)]
        L.atomic_write_json(os.path.join(cfg['outdir'], 'overfit_sanity.json'), {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'gapfill_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
            'passed': False, 'error': f'{_fe}', 'provenance': _prov,
            'reports': _reports,
        })
        L.log_experiment_event(cfg['outdir'], {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'gapfill_overfit', 'run_cfg': _pfp, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'error': f'{_fe}', 'sanity': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if not reports:
        return []
    ok = all(r['passed'] for r in reports)
    L.atomic_write_json(os.path.join(cfg['outdir'], 'overfit_sanity.json'), {
        'kind': 'small_batch_overfit_sanity_panel',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_cfg': _pfp,
        'passed': ok,
        'provenance': _prov,
        'reports': reports,
    })
    _seeds = [int(x) for x in (cfg.get('seeds') or [0])]
    _repro_seeds = list(dict.fromkeys([_seeds[0], _seeds[-1]]))
    try:
        _repro = L.reproducibility_guard(variants, arch=arch, train_ids=train_ids,
                                         vocab=vocab, mlp_ratio=ratios,
                                         seq_len=probe_seq, batch_size=2,
                                         seeds=_repro_seeds,
                                         strict=L._is_confirmatory_cfg(cfg))
    except RuntimeError as _fe:
        _reports = [dict(r) for k, r in L._REPRO_REPORTS.items() if k[0] in set(variants)]
        L.atomic_write_json(os.path.join(cfg['outdir'], 'reproducibility.json'), {
            'kind': 'duplicate_run_reproducibility_panel', 'stage': 'gapfill_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
            'passed': False, 'error': f'{_fe}', 'provenance': _prov,
            'reports': _reports,
        })
        L.log_experiment_event(cfg['outdir'], {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'gapfill_repro', 'run_cfg': _pfp, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'error': f'{_fe}', 'reproducibility': {'passed': False, 'n_reports': len(_reports)}})
        raise
    if _repro:
        L.atomic_write_json(os.path.join(cfg['outdir'], 'reproducibility.json'), {
            'kind': 'duplicate_run_reproducibility_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_cfg': _pfp,
            'passed': all(bool(r.get('passed')) for r in _repro),
            'provenance': _prov,
            'reports': _repro,
        })
    L.log_experiment_event(cfg['outdir'], {'kind': 'preflight', 'tracking_schema': 2, 'status': ('ok' if ok and all(bool(r.get('passed')) for r in _repro) else 'failed'), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code': L.code_stamp(), 'provenance': _prov, 'variants': variants, 'run_cfg': _pfp, 'sanity': {'passed': ok, 'n_reports': len(reports)}, 'reproducibility': {'passed': all(bool(r.get('passed')) for r in _repro), 'n_reports': len(_repro)}})
    if not ok:
        _bad = [r['variant'] for r in reports if not r['passed']]
        raise RuntimeError(f'small-batch overfit sanity check FAILED for {_bad} — do NOT start the full run; this points at a gradient/loss/label plumbing fault (see {cfg["outdir"]}/overfit_sanity.json)')
    print('[overfit] PASSED (necessary, not sufficient — keep the intermediate-result checks)')
    return reports

def run_full():
    run_overfit(cfg=dict(L.RUN, outdir='results_overfit_sanity'))
    guard = L.TimeGuard(L.RUN_CAP)
    guard.report()
    if os.environ.get('V6_SYNTH', '').strip().lower() in ('1', 'true', 'yes'):
        _synth_summary, _long_snapshot = synthesize_long_summary()
        if _long_snapshot:
            L.atomic_write_json(os.path.join(REPO, 'results_lm_v3_long', 'aggregate.json.v6bak'), _long_snapshot, indent=2)
        push_ok = git_push('v6: rebuild results_lm_v3_long/summary.json from committed aggregate')
    else:
        # Reconstruction is an explicit, opt-in provenance operation: a
        # synthesized record is a copy, not an independent run.
        print('[synth] skipped (set V6_SYNTH=1 to rebuild results_lm_v3_long/summary.json from an old aggregate)')
        _synth_summary, _long_snapshot = {}, None
        push_ok = True
    # Phase order: the statistically foundational panels first (core table ->
    # its leave-one-out ablations -> scale probe), the 20k-step long runs
    # LAST.  The n>=3 core/ablation/scale tables are secured before the long
    # runs; on a box that already holds the panels every phase just resumes
    # and the order adds nothing.
    _announce_seed_plan('P-CORE', L.RUN, [0, 1, 2])
    L.run(L.RUN, seeds=[0, 1, 2], guard=guard, label='P-CORE 1500-step core table (results_lm_v3_1500)')
    push_ok &= git_push('v6: results_lm_v3_1500 — 9-variant core table, 1500 steps x 3 seeds')
    _announce_seed_plan('P-ABL', L.RUN_ABL, [0, 1, 2])
    L.run(L.RUN_ABL, seeds=[0, 1, 2], guard=guard, label='P-ABL leave-one-out ablation table (results_lm_v4_abl)')
    push_ok &= git_push('v6: results_lm_v4_abl — leave-one-out ablation table (in-panel references + abl arms), 1500 steps x 3 seeds')
    _announce_seed_plan('P-SCALE', L.RUN_SCALE, [0, 1, 2])
    L.run(L.RUN_SCALE, seeds=[0, 1, 2], guard=guard, label='P-SCALE d=384/8L/seq1024 probe (results_lm_v5_scale)')
    push_ok &= git_push('v6: results_lm_v5_scale — model-scale probe (d=384, 8L, seq1024, 5 variants x 3 seeds)')
    _announce_seed_plan('P-LONG', {**L.RUN_LONG, 'variants': LONG_GAP_VARIANTS}, [0, 1, 2])
    L.run({**L.RUN_LONG, 'variants': LONG_GAP_VARIANTS}, seeds=[0, 1, 2], guard=guard, label='P-LONG hybrid 3-seed long-run table (results_lm_v3_long)')
    merge_long_aggregate(os.path.join(REPO, 'results_lm_v3_long'), _long_snapshot)
    push_ok &= git_push('v6: results_lm_v3_long hybrid variants at seeds 0-2 — 3-seed long-run table complete')
    guard.report()
    print('\n[v6] ALL PHASES DONE.')
    schedule_shutdown(90 if push_ok else 2400)
    if not push_ok:
        print('[v6] WARNING: final push failed — instance stays up 40 min for a retry, then shuts down. Results are safe on the data disk either way.')

def run_topup():
    """Seed top-up for the leave-one-out ablation panel (P-ABL).

    Re-runs the SAME `RUN_ABL` config with a longer seed list; completed
    (variant, seed) cells are skipped by the normal resume logic, so this only
    pays for the missing seeds.  The default target comes from
    `exp_lib.methodology_seed_plan` (the family-corrected exact-test floor),
    NOT from a hard-coded 3 — a 3-seed panel cannot reach significance at
    alpha=0.05 no matter how large the effect is.  Override with
    `ABL_SEED_TARGET=<n>`.
    """
    plan = L.methodology_seed_plan(0, baselines={}, variants=L.RUN_ABL['variants'])
    _default = plan.get('min_seeds_for_family_significance') or 6
    try:
        target = int(os.environ.get('ABL_SEED_TARGET') or _default)
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(f"ABL_SEED_TARGET must be an integer, got {os.environ.get('ABL_SEED_TARGET')!r}")
    if target < 1:
        raise SystemExit('ABL_SEED_TARGET must be >= 1')
    print(f'[topup] P-ABL seed target n={target} (declared contrasts={plan["declared_contrasts"]}, '
          f'family-corrected minimum={plan["min_seeds_for_family_significance"]}); existing seeds are resumed, only missing cells run.')
    guard = L.TimeGuard(L.RUN_CAP)
    guard.report()
    L.run(L.RUN_ABL, seeds=list(range(target)), guard=guard,
          label=f'P-ABL seed top-up to n={target} (results_lm_v4_abl)')
    _core_raw = os.environ.get('CORE_SEED_TARGET')
    if _core_raw:
        try:
            _core_target = int(_core_raw)
        except (TypeError, ValueError, OverflowError):
            raise SystemExit(f'CORE_SEED_TARGET must be an integer, got {_core_raw!r}')
        if _core_target < 1:
            raise SystemExit('CORE_SEED_TARGET must be >= 1')
        print(f'[topup] P-CORE seed target n={_core_target} (baseline claims live in this panel; resume-safe)')
        L.run(L.RUN, seeds=list(range(_core_target)), guard=guard,
              label=f'P-CORE seed top-up to n={_core_target} (results_lm_v3_1500)')
    ok = git_push(f'v6: P-ABL/P-CORE seed top-up (P-ABL n={target}' + (f", P-CORE n={_core_target}" if _core_raw else '') + '; resume-safe)')
    guard.report()
    print(f'\n[v6] TOP-UP DONE (target n={target}).')
    schedule_shutdown(90 if ok else 2400)

def run_smoke():
    # The smoke must exercise every forward path the `full` phases schedule:
    # dense, fixed-block CSA and the dynamic hybrid, PLUS the leave-one-out
    # ablation arms of P-ABL (random indexer selection, zero pooled content,
    # sink off, learned local fusion, the HCA layer kind, and the dense
    # sink control) — otherwise a broken ablation arm would only surface
    # after the full panels.
    smoke_cfg = dict(L.RUN, outdir='results_smoke', variants=['full', 'csa_fixed', 'hybrid_dynamic', 'csa_dyn_randidx', 'csa_dyn_zerocont', 'csa_dyn_nosink', 'csa_dyn_nocomp', 'csa_dyn_fuse', 'hybrid_hca_dyn', 'full_sink'], steps=60, eval_every=30, n_train_tokens=1000000, seeds=[0])
    run_overfit(cfg=smoke_cfg, variants=['full', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic', 'hybrid_hca_dyn', 'csa_dyn_nocomp'], steps=250)
    L.run(smoke_cfg, seeds=[0], guard=None, label='SMOKE (60 steps)', par=False)
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
    if mode not in ('smoke', 'overfit', 'topup', 'full'):
        print(f"[v6] unknown mode {mode!r}; expected 'smoke', 'overfit', 'topup' or 'full' (no argument means 'full').  Nothing was started.")
        raise SystemExit(2)
    try:
        _gpu = torch.cuda.get_device_name(0)
    except Exception:
        _gpu = 'cpu'
    print(f'[v6] mode = {mode}   repo = {REPO}   device = {L.DEVICE} ({_gpu})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'overfit':
            run_overfit()
        elif mode == 'topup':
            run_topup()
        elif mode == 'full':
            run_full()
        else:
            raise SystemExit(f'unknown mode: {mode}')
    except Exception:
        traceback.print_exc()
        if mode in ('full', 'topup'):
            print('\n[v6] CRASHED — attempting a partial-results push, then the instance stays up 40 min for remote diagnosis before auto-shutdown (cancel with: pkill -f v6autoshutdown).')
            try:
                git_push('v6: PARTIAL — run_gapfill crashed, see run_v6.log (resume by re-running: completed runs are skipped)')
            except Exception:
                traceback.print_exc()
            schedule_shutdown(2400)
        else:
            print(f'\n[v6] CRASHED in `{mode}` mode — the tree can hold half-written throwaway output, so NOTHING was committed or pushed and no shutdown was scheduled.  Repair the defect and re-run; `smoke` exists to catch exactly this.')
        sys.exit(1)
