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
  smoke   tiny end-to-end run of the tuning + selection + audit round-trip
  full    the real tuning panel (defaults to exp_lib.RUN; BT_SEARCH_STEPS /
          BT_TRIAL_LIMIT / BT_SKIP_FINAL can bound it)
  report  zero-GPU: audit results_baseline_tuning/baseline_tuning.json

The held-out test split is NEVER read here.  Hyperparameters are selected on
the validation split; the final comparison re-uses exactly those selections.
"""
import json
import os
import sys
import time

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


def _trial_cfg(base, outdir, arms, seed, steps, knobs):
    cfg = dict(base)
    cfg.update(outdir=outdir, variants=list(arms), seeds=[int(seed)], steps=int(steps),
               lr=float(knobs['lr']), weight_decay=float(knobs['weight_decay']),
               warmup=int(knobs['warmup']))
    return cfg


def _one_seed_record(summary, arm, seed):
    rec = summary.get(f'{arm}::seed{int(seed)}')
    if isinstance(rec, dict) and L.ppl_is_usable(rec.get('ppl')):
        return rec
    return None


def run_tuning(base_cfg=None, outdir='results_baseline_tuning', protocol=None,
               trial_limit=None, search_steps=None, final=True):
    """Run the declared grid, select per arm on validation PPL, then optionally
    retrain each arm at its selected config across the final seed list."""
    protocol = dict(L.BASELINE_TUNING_PROTOCOL if protocol is None else protocol)
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
    os.makedirs(outdir, exist_ok=True)
    print(f'[tune] equal-search grid: {len(arms)} arms x {len(trials)} trials x {search_steps} steps, selection={protocol["selection_metric"]} on {protocol["selection_split"]}')
    L.overfit_guard(arms, steps=200, vocab=256)
    per_arm = {a: [] for a in arms}
    for ti, knobs in enumerate(trials):
        tdir = os.path.join(outdir, f'trial_{ti:02d}')
        cfg = _trial_cfg(base, tdir, arms, search_seed, search_steps, knobs)
        L.run(cfg, seeds=[search_seed], guard=None, label=f'baseline tuning trial {ti + 1}/{len(trials)}')
        summary = json.load(open(os.path.join(tdir, 'summary.json'), encoding='utf-8'))
        for arm in arms:
            rec = _one_seed_record(summary, arm, search_seed)
            if rec is None:
                print(f'[tune] WARNING: {arm} has no measurable PPL in trial {ti} ({tdir})')
                continue
            per_arm[arm].append({'config': dict(knobs), 'ppl': float(rec['ppl']),
                                 'params': rec.get('params'), 'steps': rec.get('steps'),
                                 'tokens_seen': rec.get('tokens_seen'),
                                 'run_cfg': rec.get('run_cfg'), 'outdir': tdir})
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
        print(f'[tune] {arm:22s} best PPL {best["ppl"]:.4f} at {best["config"]}')
    artifact = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': protocol,
        'selection_metric': protocol['selection_metric'],
        'selection_split': protocol['selection_split'],
        'direction': protocol['direction'],
        'search_seed': search_seed,
        'search_steps': search_steps,
        'arms': selected,
        'code': L.code_stamp(),
    }
    path = os.path.join(outdir, 'baseline_tuning.json')
    L.atomic_write_json(path, artifact)
    audit = L.baseline_tuning_audit(path=path, protocol=protocol)
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
        if a.get(k) is not None and b.get(k) is not None and a.get(k) != b.get(k):
            return f'{k} differs ({a.get(k)!r} vs {b.get(k)!r})'
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
    tuned_arms = {a: len(entry['trials']) for a, entry in selected.items()}
    comparison = {'pairing_rule': 'tuned_equal_budget',
                  'note': 'Each arm used its own validation-selected hyperparameters; data/val fingerprints, steps, tokens and parameters are matched. This is NOT the strict same-run_cfg architectural contrast and must be reported separately.',
                  'contrasts': {}}
    for method in ('csa_dynamic',):
        if method not in selected:
            continue
        for baseline in selected:
            if baseline == method:
                continue
            deltas, seeds, skipped = [], [], []
            for seed in sorted({int(r['seed']) for r in merged.values() if r.get('variant') == method} &
                               {int(r['seed']) for r in merged.values() if r.get('variant') == baseline}):
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
            st.update({'with': method, 'without': baseline, 'seeds': seeds,
                       'skipped': skipped, 'pairing_rule': 'tuned_equal_budget',
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
    comparison['family'] = {
        'name': 'tuned_equal_budget_baselines',
        'm': _m, 'alpha': 0.05, 'correction': 'holm step-down',
        'min_seeds_for_family_significance': L.signflip_min_n_for_holm(_m, 0.05) if _m else None,
        'note': 'per-arm validation-selected hyperparameters; a tuned win is NOT a strict same-run_cfg architectural contrast',
    }
    L.atomic_write_json(os.path.join(final_dir, 'tuned_comparison.json'), comparison)
    print('\n[tune] tuned-vs-tuned paired contrasts (Δ = method − baseline; negative PPL is better):')
    for name, st in comparison['contrasts'].items():
        print(f'  {name:48s} n={st["n"]}  Δ={st["mean"]:+.4f} ± {st["std"]:.4f}  p={st["p_exact_signflip"]:.4f}  p_Holm={st.get("p_holm")}  {st["effect_vs_noise"]}')


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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    _cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if _cmd == 'smoke':
        run_smoke()
    elif _cmd == 'full':
        base = dict(L.RUN)
        if os.environ.get('BT_BASE_OUTDIR'):
            base['outdir'] = os.environ['BT_BASE_OUTDIR']
        run_tuning(base, outdir=os.environ.get('BT_OUTDIR', 'results_baseline_tuning'),
                   trial_limit=(int(os.environ['BT_TRIAL_LIMIT']) if os.environ.get('BT_TRIAL_LIMIT') else None),
                   search_steps=(int(os.environ['BT_SEARCH_STEPS']) if os.environ.get('BT_SEARCH_STEPS') else None),
                   final=(os.environ.get('BT_SKIP_FINAL', '').strip().lower() not in ('1', 'true', 'yes')))
    elif _cmd == 'report':
        run_report()
    else:
        print(__doc__)
        raise SystemExit(2)
