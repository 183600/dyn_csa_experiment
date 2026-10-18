#!/usr/bin/env python3
"""baseline_tune.py — equal-search tuning for the simple baselines and the
proposed attention arm.

Why this exists: a sparse/dynamic arm that only beats a dense baseline at the
dense baseline's default hyperparameters has NOT shown that the mechanism is
better; it may simply have been tuned while the baseline was not.  This driver
runs the SAME declared grid (``exp_lib.BASELINE_TUNING_PROTOCOL``) for every
arm, selects on validation PPL, and writes the trial/selection record that
``exp_lib.baseline_tuning_audit`` verifies.

Subcommands:
  simple  fit and tune the trivial count-model floor (unigram / bigram) on the
          training split, select on validation, write simple_baseline.json
  smoke   tiny end-to-end run of the tuning + selection + audit round-trip
  full    the real tuning panel (defaults to exp_lib.RUN; BT_SEARCH_STEPS /
          BT_TRIAL_LIMIT / BT_SKIP_FINAL / BT_SKIP_SIMPLE can bound it); runs
          the count-model floor first unless BT_SKIP_SIMPLE=1
  report  zero-GPU: audit results_baseline_tuning/baseline_tuning.json and
          simple_baseline.json

The held-out test split is NEVER read here.  Hyperparameters are selected on
the validation split; the final comparison re-uses exactly those selections.
"""
import json
import os
import sys
import time

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


def _trial_cfg(base, outdir, arms, seeds, steps, knobs):
    cfg = dict(base)
    cfg.update(outdir=outdir, variants=list(arms), seeds=[int(s) for s in seeds],
               steps=int(steps), lr=float(knobs['lr']),
               weight_decay=float(knobs['weight_decay']),
               warmup=int(knobs['warmup']))
    return cfg


def _one_seed_record(summary, arm, seed):
    rec = summary.get(f'{arm}::seed{int(seed)}')
    if isinstance(rec, dict) and L.ppl_is_usable(rec.get('ppl')):
        return rec
    return None


def _mean_std(values):
    vals = [float(v) for v in values if L.ppl_is_usable(v)]
    if not vals:
        return float('nan'), float('nan')
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def run_tuning(base_cfg=None, outdir='results_baseline_tuning', protocol=None,
               trial_limit=None, search_steps=None, final=True):
    """Run the declared grid, select per arm on validation PPL, then optionally
    retrain each arm at its selected config across the final seed list."""
    protocol = dict(L.BASELINE_TUNING_PROTOCOL if protocol is None else protocol)
    declared_trials = list(protocol['trials'])
    base = dict(L.RUN if base_cfg is None else base_cfg)
    arms = list(protocol['arms'])
    trials = list(protocol['trials'])
    if trial_limit is not None:
        trials = trials[:max(1, int(trial_limit))]
    if search_steps is not None:
        search_steps = int(search_steps)
    else:
        search_steps = int(base['steps'])
    search_seed = int(protocol.get('search_seed', 0))
    # Tuning selection must not be decided by a single seed's validation
    # noise.  The same declared search-seed set is used for every arm and
    # every trial, and selection uses the mean validation PPL (with the
    # across-seed std recorded).  Legacy protocols that only declare
    # `search_seed` keep the single-seed path.
    _raw_search_seeds = protocol.get('search_seeds')
    if _raw_search_seeds is None:
        _raw_search_seeds = [search_seed]
    try:
        search_seeds = sorted({int(s) for s in _raw_search_seeds})
    except (TypeError, ValueError):
        raise ValueError(f'search_seeds must be integers, got {_raw_search_seeds!r}')
    if not search_seeds:
        raise ValueError('search_seeds must contain at least one seed')
    os.makedirs(outdir, exist_ok=True)
    print(f'[tune] equal-search grid: {len(arms)} arms x {len(trials)} trials x {search_steps} steps x {len(search_seeds)} seed(s), selection={protocol["selection_metric"]} on {protocol["selection_split"]}')
    L.overfit_guard(arms, steps=200, vocab=256)
    per_arm = {a: [] for a in arms}
    for ti, knobs in enumerate(trials):
        tdir = os.path.join(outdir, f'trial_{ti:02d}')
        cfg = _trial_cfg(base, tdir, arms, search_seeds, search_steps, knobs)
        L.run(cfg, seeds=search_seeds, guard=None, label=f'baseline tuning trial {ti + 1}/{len(trials)}')
        summary = json.load(open(os.path.join(tdir, 'summary.json'), encoding='utf-8'))
        for arm in arms:
            records = {}
            for seed in search_seeds:
                rec = _one_seed_record(summary, arm, seed)
                if rec is not None:
                    records[seed] = rec
            if not records:
                print(f'[tune] WARNING: {arm} has no measurable PPL in trial {ti} ({tdir})')
                continue
            if len(records) < len(search_seeds):
                print(f'[tune] WARNING: {arm} trial {ti} has only seeds {sorted(records)} of {search_seeds}; '
                      'selection uses the observed seeds and the artifact records the gap')
            mean, std = _mean_std([records[s]['ppl'] for s in sorted(records)])
            _steps = {int(records[s]['steps']) for s in records if records[s].get('steps') is not None}
            _tokens = {int(records[s]['tokens_seen']) for s in records if records[s].get('tokens_seen') is not None}
            per_arm[arm].append({
                'config': dict(knobs), 'ppl': mean, 'ppl_std': std,
                'n_seeds': len(records), 'seeds': sorted(records),
                'seed_ppls': {str(s): float(records[s]['ppl']) for s in sorted(records)},
                'params': records[sorted(records)[0]].get('params'),
                'steps': (next(iter(_steps)) if len(_steps) == 1 else None),
                'tokens_seen': (next(iter(_tokens)) if len(_tokens) == 1 else None),
                'budget_by_seed': {str(s): [records[s].get('steps'), records[s].get('tokens_seen')]
                                   for s in sorted(records)},
                'run_cfg': records[sorted(records)[0]].get('run_cfg'), 'outdir': tdir,
            })
    selected = {}
    for arm in arms:
        hits = [t for t in per_arm[arm] if L.ppl_is_usable(t.get('ppl'))]
        if not hits:
            raise RuntimeError(f'[tune] {arm}: no usable trial PPL — cannot select a config')
        best = min(hits, key=lambda t: float(t['ppl']))
        selected[arm] = {'selected': dict(best['config']),
                         'selected_metric_value': float(best['ppl']),
                         'params': best.get('params'), 'run_cfg': best.get('run_cfg'),
                         'trials': per_arm[arm]}
        print(f'[tune] {arm:22s} best mean PPL {best["ppl"]:.4f} +/- {best["ppl_std"]:.4f} '
              f'(n={best.get("n_seeds")}) at {best["config"]}')
    artifact_protocol = dict(protocol)
    artifact_protocol['trials'] = [dict(t) for t in trials]
    artifact_protocol['search_seeds'] = list(search_seeds)
    artifact = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': artifact_protocol,
        'selection_metric': protocol['selection_metric'],
        'selection_split': protocol['selection_split'],
        'direction': protocol['direction'],
        'search_seed': search_seed,
        'search_seeds': list(search_seeds),
        'search_steps': search_steps,
        'declared_trials': len(declared_trials),
        'trials_run': len(trials),
        'search_complete': bool(len(trials) == len(declared_trials)),
        'final_seeds': list(protocol.get('final_seeds') or []),
        'arms': selected,
        'code': L.code_stamp(),
    }
    path = os.path.join(outdir, 'baseline_tuning.json')
    L.atomic_write_json(path, artifact)
    audit = L.baseline_tuning_audit(path=path)
    print(f'[tune] artifact: {path}  audit={audit.get("status")}')
    if not audit.get('tuning_verified'):
        raise RuntimeError(f'[tune] tuning artifact failed its own audit: {audit.get("problems")}')
    if final:
        _run_final(base, outdir, protocol, selected)
        L.methodology_audit(json.load(open(os.path.join(outdir, 'final', 'summary.json'), encoding='utf-8')),
                            outdir=os.path.join(outdir, 'final'), verbose=True)
    return artifact


def _relaxed_pair_reason(a, b, tuned_arms):
    """Pairing rule for per-arm tuned configurations.

    The strict `pair_reason` gate requires identical run_cfg so that ONLY the
    architecture changes.  After independent tuning, the selected lr/wd/warmup
    may legitimately differ per arm.  This rule therefore requires every
    non-tuned field to match (data/val fingerprints, steps, tokens, params) and
    records the fact that both arms received the same declared search range.
    It is deliberately a separate rule: strict architectural contrasts and
    tuned-vs-tuned contrasts must not be silently pooled.
    """
    for k in ('steps', 'tokens_seen', 'data_fp', 'val_fp'):
        _va, _vb = a.get(k), b.get(k)
        if _va is None or _vb is None:
            return f'unverifiable {k} (at least one record omits it)'
        if _va != _vb:
            return f'{k} differs ({_va!r} vs {_vb!r})'
    # Tuning may change lr/wd/warmup, but it must never silently change the
    # optimizer, betas/eps, LR-schedule shape, grad-clip or augmentation.  The
    # strict `pair_reason` already enforces this for architecture contrasts;
    # the relaxed tuned rule must not become a back door around it.
    ka, kb = a.get('train_knobs'), b.get('train_knobs')
    if (ka is None) != (kb is None):
        return 'unverifiable train_knobs (only one record carries the optimizer/schedule/augmentation descriptor)'
    if ka is not None and ka != kb:
        return 'different train_knobs (optimizer / LR-schedule / grad-clip / augmentation)'
    try:
        pa, pb = int(a.get('params')), int(b.get('params'))
        if abs(pa - pb) / float(max(pa, pb, 1)) > 0.01:
            return f'parameter mismatch ({pa} vs {pb})'
    except (TypeError, ValueError):
        pass
    if tuned_arms.get(a.get('variant')) != tuned_arms.get(b.get('variant')):
        return 'arms did not receive the same declared tuning range'
    return None


def _run_final(base, outdir, protocol, selected):
    final_seeds = [int(s) for s in protocol.get('final_seeds', [])]
    final_dir = os.path.join(outdir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    merged = {}
    for arm, entry in selected.items():
        adir = os.path.join(final_dir, arm)
        cfg = dict(base, outdir=adir, variants=[arm], seeds=final_seeds)
        cfg.update(lr=float(entry['selected']['lr']),
                   weight_decay=float(entry['selected']['weight_decay']),
                   warmup=int(entry['selected']['warmup']))
        L.run(cfg, seeds=final_seeds, guard=None, label=f'final tuned {arm}')
        s = json.load(open(os.path.join(adir, 'summary.json'), encoding='utf-8'))
        for rec in s.values():
            if isinstance(rec, dict) and L.ppl_is_usable(rec.get('ppl')) and 'seed' in rec:
                merged[f'{arm}::seed{int(rec["seed"])}'] = rec
    L.atomic_write_json(os.path.join(final_dir, 'summary.json'), merged)
    tuning = json.load(open(os.path.join(outdir, 'baseline_tuning.json'), encoding='utf-8'))
    tuning['final_outdir'] = final_dir
    tuning['final_records'] = {k: {'ppl': v.get('ppl'), 'seed': v.get('seed'),
                                   'params': v.get('params'), 'run_cfg': v.get('run_cfg'),
                                   'hyperparams': v.get('hyperparams')}
                               for k, v in merged.items()}
    L.atomic_write_json(os.path.join(final_dir, 'baseline_tuning.json'), tuning)
    # Arm-level mean±std over the same final seed set, so the tuned table is
    # not a single aggregate number per arm.  The paired contrast below is
    # still computed per seed (paired by construction), not by subtracting
    # these two means.
    arm_summary = {}
    for arm, entry in selected.items():
        _recs = {int(r['seed']): float(r['ppl']) for r in merged.values()
                 if isinstance(r, dict) and r.get('variant') == arm
                 and r.get('seed') is not None and L.ppl_is_usable(r.get('ppl'))}
        _vals = [_recs[s] for s in sorted(_recs)]
        arm_summary[arm] = {
            'n_seeds': len(_vals), 'seeds': sorted(_recs),
            'ppl_mean': float(np.mean(_vals)) if _vals else float('nan'),
            'ppl_std': float(np.std(_vals, ddof=1)) if len(_vals) > 1 else 0.0,
            'selected': dict(entry.get('selected') or {}),
        }
    tuned_arms = {a: len(entry['trials']) for a, entry in selected.items()}
    comparison = {'pairing_rule': 'tuned_equal_budget',
                  'note': 'Each arm used its own validation-selected hyperparameters; data/val fingerprints, steps, tokens and parameters are matched. This is NOT the strict same-run_cfg architectural contrast and must be reported separately.',
                  'final_seeds': list(final_seeds),
                  'arm_summary': arm_summary,
                  'contrasts': {}}
    for method in ('csa_dynamic',):
        if method not in selected:
            continue
        for baseline in selected:
            if baseline == method:
                continue
            deltas, seeds, skipped = [], [], []
            _common = sorted({int(r['seed']) for r in merged.values() if r.get('variant') == method} &
                             {int(r['seed']) for r in merged.values() if r.get('variant') == baseline})
            for seed in _common:
                ra = merged.get(f'{method}::seed{seed}')
                rb = merged.get(f'{baseline}::seed{seed}')
                if not isinstance(ra, dict) or not isinstance(rb, dict):
                    continue
                why = _relaxed_pair_reason(ra, rb, tuned_arms)
                if why:
                    skipped.append({'seed': seed, 'reason': why})
                    continue
                deltas.append(float(ra['ppl']) - float(rb['ppl']))
                seeds.append(seed)
            st = L.paired_effect_stats(deltas)
            _base_mean = (arm_summary.get(baseline) or {}).get('ppl_mean')
            _rel = (100.0 * float(st['mean']) / _base_mean
                    if isinstance(_base_mean, (int, float)) and _base_mean > 0 else float('nan'))
            st.update({'with': method, 'without': baseline, 'seeds': seeds,
                       'n_shared': len(_common), 'n_skipped': len(skipped),
                       'skipped': skipped, 'pairing_rule': 'tuned_equal_budget',
                       'with_mean': (arm_summary.get(method) or {}).get('ppl_mean'),
                       'without_mean': _base_mean,
                       'relative_effect_pct': _rel,
                       'mean_pm_std': (f'{st["mean"]:+.4f} +/- {st["std"]:.4f}' if st.get('n') else 'n/a')})
            comparison['contrasts'][f'{method} - {baseline}'] = st
    # The tuned contrasts form their own family: each baseline is a separate
    # claim, so raw p values must not be read as if only one comparison was
    # run.  Apply the same Holm step-down used everywhere else and record the
    # exact-test resolution floor instead of leaving "not significant" to be
    # mistaken for a null when n is too small.
    _named = [(name, st.get('p_exact_signflip')) for name, st in comparison['contrasts'].items()]
    _holm = L.holm_adjust(_named, alpha=0.05) if _named else {}
    _m = len(_named)
    for name, st in comparison['contrasts'].items():
        _h = _holm.get(name) or {}
        st['p_holm'] = _h.get('p_holm')
        st['significant_holm'] = bool(_h.get('reject') and (not st.get('within_seed_noise'))
                                      and st.get('controlled', True))
    _final_counts = {a: int(v.get('n_seeds') or 0) for a, v in arm_summary.items()}
    comparison['family'] = {
        'name': 'tuned_equal_budget_baselines',
        'm': _m, 'alpha': 0.05, 'correction': 'holm step-down',
        'final_seed_count_by_arm': _final_counts,
        'final_seed_coverage_equal': (len(set(_final_counts.values())) == 1) if _final_counts else None,
        'min_seeds_for_family_significance': L.signflip_min_n_for_holm(_m, 0.05) if _m else None,
        'note': 'per-arm validation-selected hyperparameters; a tuned win is NOT a strict same-run_cfg architectural contrast',
    }
    if _final_counts:
        _min_final = min(_final_counts.values())
        _need = comparison['family']['min_seeds_for_family_significance']
        if _min_final < 3:
            comparison['family']['warning'] = 'fewer than 3 final seeds per arm — descriptive only'
        elif _need is not None and _min_final < _need:
            comparison['family']['warning'] = (
                f'{_min_final} final seeds cannot reach Holm-corrected significance for m={_m} '
                f'(needs n>={_need}); the contrast is resolution-limited, not a null')
    L.atomic_write_json(os.path.join(final_dir, 'tuned_comparison.json'), comparison)
    print('\n[tune] tuned-vs-tuned paired contrasts (Δ = method − baseline; negative PPL is better):')
    for name, st in comparison['contrasts'].items():
        print(f'  {name:48s} n={st["n"]}  Δ={st["mean"]:+.4f} ± {st["std"]:.4f}  p={st["p_exact_signflip"]:.4f}  p_Holm={st.get("p_holm")}  {st["effect_vs_noise"]}')


def run_simple(base_cfg=None, outdir='results_baseline_tuning', protocol=None):
    """Fit/tune the trivial count-model floor on the real corpus.

    This is deliberately the FIRST baseline: a unigram/bigram count model whose
    smoothing (and bigram interpolation) is selected on the validation split.
    If a neural mechanism cannot beat this floor, there is no point in asking
    whether it beats a tuned dense Transformer.
    """
    base = dict(L.RUN if base_cfg is None else base_cfg)
    # Serve the corpus selected by the panel config (WikiText-103, or the
    # B-tier FineWeb path), so the count-model floor and the neural panel use
    # the same corpus and vocabulary.
    train_ids, val_batch, vocab, _tok, _val_bnd = L.load_run_corpus(base)
    # Match the neural panel's FINAL evaluation slice exactly: train_variant
    # scores the full fixed val_batch (`eval_subset` only controls the
    # per-eval-step history).
    rows = max(1, int(val_batch.shape[0]))
    eval_slice = val_batch[:rows]
    artifact = L.simple_baseline_fit(train_ids, eval_slice, vocab, protocol=protocol)
    artifact.setdefault('data', {})['cache_val_fp'] = L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1))
    artifact['data']['eval_rows'] = int(rows)
    artifact['data']['eval_fp'] = L._ids_fp(np.ascontiguousarray(eval_slice).reshape(-1))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'simple_baseline.json')
    L.atomic_write_json(path, artifact)
    audit = L.simple_baseline_audit(path=path)
    print('[simple] best count-model floor: %s PPL=%.4f  audit=%s' % (
        artifact.get('best_baseline'), float(artifact.get('best_ppl') or float('nan')), audit.get('status')))
    for arm, entry in sorted((artifact.get('baselines') or {}).items()):
        print('  %-8s selected %s  validation PPL=%.4f' % (
            arm, entry.get('selected'), float(entry.get('selected_metric_value') or float('nan'))))
    if not audit.get('baseline_verified'):
        raise RuntimeError(f'[simple] count-model artifact failed its audit: {audit.get("problems")}')
    return artifact


def run_smoke():
    base = dict(L.RUN, d=32, n_layers=1, n_heads=2, d_head=16, seq_len=32, batch_size=2,
                n_train_tokens=8000, steps=2, warmup=1, eval_every=1, eval_subset=2,
                val_seqs=2, seeds=[0])
    protocol = dict(L.BASELINE_TUNING_PROTOCOL,
                    arms=['full', 'csa_fixed', 'csa_dynamic'],
                    trials=L.BASELINE_TUNING_PROTOCOL['trials'][:2],
                    final_seeds=[0])
    os.environ.setdefault('CSA_SKIP_OVERFIT', '1')
    run_tuning(base, outdir='results_smoke_baseline_tune', protocol=protocol,
               trial_limit=2, search_steps=2, final=True)


def run_report():
    path = os.path.join('results_baseline_tuning', 'baseline_tuning.json')
    report = L.baseline_tuning_audit(path=path)
    simple = L.simple_baseline_audit(path=os.path.join('results_baseline_tuning', 'simple_baseline.json'))
    print(json.dumps({'baseline_tuning': report, 'simple_baseline': simple}, ensure_ascii=False, indent=2))
    return {'baseline_tuning': report, 'simple_baseline': simple}


if __name__ == '__main__':
    _cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if _cmd == 'smoke':
        run_smoke()
    elif _cmd == 'simple':
        base = dict(L.RUN)
        if os.environ.get('BT_BASE_OUTDIR'):
            base['outdir'] = os.environ['BT_BASE_OUTDIR']
        run_simple(base, outdir=os.environ.get('BT_OUTDIR', 'results_baseline_tuning'))
    elif _cmd == 'full':
        base = dict(L.RUN)
        if os.environ.get('BT_BASE_OUTDIR'):
            base['outdir'] = os.environ['BT_BASE_OUTDIR']
        _od = os.environ.get('BT_OUTDIR', 'results_baseline_tuning')
        if os.environ.get('BT_SKIP_SIMPLE', '').strip().lower() not in ('1', 'true', 'yes'):
            run_simple(base, outdir=_od)
        run_tuning(base, outdir=_od,
                   trial_limit=(int(os.environ['BT_TRIAL_LIMIT']) if os.environ.get('BT_TRIAL_LIMIT') else None),
                   search_steps=(int(os.environ['BT_SEARCH_STEPS']) if os.environ.get('BT_SEARCH_STEPS') else None),
                   final=(os.environ.get('BT_SKIP_FINAL', '').strip().lower() not in ('1', 'true', 'yes')))
    elif _cmd == 'report':
        run_report()
    else:
        print(__doc__)
        raise SystemExit(2)
