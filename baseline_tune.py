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


def _protocol_arm_groups(protocol, base, arms):
    """Resolve the protocol's arm groups, falling back to one legacy group.

    Groups exist because the v6 arms and the frozen v12 rope arms parameter-
    match against DIFFERENT reference arms (csa_dynamic vs csa_fixed_rope),
    so they cannot share a single panel config.  Within a group the matched
    set / mlp_match_ref override the base config; across groups the declared
    trials, search seeds and selection split stay identical, so the
    equal-search contract covers the union of arms.
    """
    declared = list(arms)
    declared_set = set(declared)
    raw = protocol.get('arm_groups')
    groups = []
    if raw:
        for g in raw:
            if not isinstance(g, dict) or not isinstance(g.get('arms'), (list, tuple)):
                raise ValueError(f'arm group must be a dict with an arms list, got {g!r}')
            ga = [a for a in g['arms'] if a in declared_set]
            if not ga:
                continue
            ng = dict(g)
            ng['arms'] = ga
            if isinstance(g.get('matched'), (list, tuple, set)):
                ng['matched'] = [a for a in g['matched'] if a in declared_set]
            groups.append(ng)
    else:
        groups = [{'arms': declared, 'matched': base.get('matched'),
                   'mlp_match_ref': base.get('mlp_match_ref', 'csa_dynamic')}]
    seen = []
    for g in groups:
        for a in g['arms']:
            if a in seen:
                raise ValueError(f'arm {a!r} appears in more than one tuning group')
            seen.append(a)
    if not raw:
        groups[0]['arms'] = declared
    else:
        _missing = [a for a in declared if a not in seen]
        if _missing:
            groups.append({'arms': _missing, 'matched': base.get('matched'),
                           'mlp_match_ref': base.get('mlp_match_ref', 'csa_dynamic')})
            seen.extend(_missing)
    return groups


def _group_overrides(base, group):
    cfg_over = {}
    matched = group.get('matched')
    if matched is not None:
        cfg_over['matched'] = {str(x) for x in matched}
    ref = group.get('mlp_match_ref')
    if ref:
        cfg_over['mlp_match_ref'] = str(ref)
    return cfg_over


def _search_final_seeds(protocol):
    """Return (search_seeds, final_seeds, overlap).

    The final records are the evidence for a tuned comparison.  If the same
    seeds were also used to select the hyper-parameters, the selected config
    has already seen those exact draws and the "final" mean is optimistically
    selected, not an independent estimate.  Keep the two partitions explicit
    and refuse the run when they overlap.
    """
    def _seeds(raw, label):
        if raw is None:
            return []
        if isinstance(raw, (str, bytes)):
            raw = L.normalize_seed_list(raw, default=())
        elif isinstance(raw, (int, np.integer)) and not isinstance(raw, (bool, np.bool_)):
            raw = [raw]
        try:
            vals = list(raw)
        except TypeError:
            raise ValueError(f'seed list must be iterable, got {raw!r}') from None
        out = []
        seen = set()
        for raw_seed in vals:
            seed = L._as_seed_int(raw_seed)
            if seed is None:
                raise ValueError(f'seeds must be integers, got {raw_seed!r}')
            if seed in seen:
                # Silently de-duplicating here made the declared seed count
                # (used by the power floor) disagree with the records that
                # actually ran.  A duplicated draw identity is a planning
                # error, not an extra replicate.
                raise ValueError(
                    f'{label} contains duplicate seed {seed}; a seed identity '
                    'contributes one paired draw and must be unique')
            seen.add(seed)
            out.append(seed)
        return sorted(out)

    raw_search = protocol.get('search_seeds')
    if raw_search is None:
        raw_search = [protocol.get('search_seed', 0)]
    search = _seeds(raw_search, 'search_seeds')
    # `or []` would turn the valid scalar seed 0 into an empty final plan.
    _raw_final = protocol.get('final_seeds')
    final = _seeds([] if _raw_final is None else _raw_final, 'final_seeds')
    if not search:
        raise ValueError('search_seeds must contain at least one seed')
    return search, final, sorted(set(search) & set(final))


def _contrast_plan(protocol, groups, selected):
    """Declared tuned comparisons: each group's method arm vs the other arms
    in the SAME parameter-matching group.

    The v6 and frozen-v12 arms are matched against different references, so a
    cross-group contrast would change the parameter-matching reference as well
    as the mechanism.  Legacy protocols without `arm_groups` keep the old
    single-method rule.
    """
    selected = set(selected)
    plan = []
    seen = set()

    def _add(method, baseline):
        if method == baseline or method not in selected or baseline not in selected:
            return
        key = (method, baseline)
        if key not in seen:
            seen.add(key)
            plan.append(key)

    raw_groups = protocol.get('arm_groups')
    if raw_groups:
        for group in groups:
            method = group.get('method_arm')
            if method not in selected:
                continue
            for baseline in group.get('arms') or []:
                _add(method, baseline)
    else:
        method = protocol.get('method_arm') or 'csa_dynamic'
        for baseline in selected:
            _add(method, baseline)
    return plan


def run_tuning(base_cfg=None, outdir='results_baseline_tuning', protocol=None,
               trial_limit=None, search_steps=None, final=True):
    """Run the declared grid, select per arm on validation PPL, then optionally
    retrain each arm at its selected config across the final seed list."""
    protocol = dict(L.BASELINE_TUNING_PROTOCOL if protocol is None else protocol)
    declared_trials = list(protocol['trials'])
    base = dict(L.RUN if base_cfg is None else base_cfg)
    arms = list(protocol['arms'])
    groups = _protocol_arm_groups(protocol, base, arms)
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
    search_seeds, final_seeds_declared, overlap = _search_final_seeds(protocol)
    if final and not final_seeds_declared:
        # An empty final partition used to fall through to L.run(..., seeds=[])
        # whose default is seed 0.  That silently retrained the "final" panel
        # on a seed that may also have selected the hyper-parameters, with no
        # artifact field recording the substitution.  Refuse instead.
        raise ValueError(
            'final=True requires a non-empty declared final_seeds partition; '
            'an empty list would silently fall back to the default seed 0 and '
            'could re-use a search draw')
    if final and final_seeds_declared and overlap:
        raise ValueError(
            'search_seeds and final_seeds overlap on %s — the final evaluation '
            'would re-use the draws that selected the hyper-parameters; declare '
            'a disjoint final seed partition' % overlap)
    if final and final_seeds_declared:
        _planned = _contrast_plan(protocol, groups, arms)
        _need = L.signflip_min_n_for_holm(len(_planned), 0.05)
        if _need is not None and len(final_seeds_declared) < _need:
            print(f'[tune] WARNING: {len(final_seeds_declared)} final seeds cannot reach '
                  f'Holm significance for the {len(_planned)}-contrast tuned family '
                  f'(needs n>={_need}); the final panel will be descriptive only')
    os.makedirs(outdir, exist_ok=True)
    print(f'[tune] equal-search grid: {len(arms)} arms ({len(groups)} group(s)) x {len(trials)} trials x {search_steps} steps x {len(search_seeds)} seed(s), selection={protocol["selection_metric"]} on {protocol["selection_split"]}')
    _arch = {'d': int(base.get('d', 256)), 'n_layers': int(base.get('n_layers', 6)),
             'n_heads': int(base.get('n_heads', 8)), 'd_head': int(base.get('d_head', 32))}
    _pre_vocab = int(base.get('vocab_size') or 1024)
    _pre_ratios = {}
    for _grp in groups:
        _ov = _group_overrides(base, _grp)
        for _a in _grp['arms']:
            try:
                _pre_ratios[_a] = L.variant_mlp_ratio(_a, _pre_vocab, d=_arch['d'], n_layers=_arch['n_layers'], n_heads=_arch['n_heads'], d_head=_arch['d_head'], seq_len=int(base.get('seq_len', 512)), matched=_ov.get('matched'), ref_variant=_ov.get('mlp_match_ref', 'csa_dynamic'))
            except Exception:
                _pre_ratios[_a] = None
    L.overfit_guard(arms, steps=200, vocab=_pre_vocab,
                    seq_len=L.probe_seq_len(base.get('seq_len', 512)),
                    arch=_arch, mlp_ratio=_pre_ratios)
    per_arm = {a: [] for a in arms}
    _prov_any = None
    _data_version_any = None
    # Nested validation: tune on the selection rows, estimate the final
    # contrast on the disjoint evaluation rows.  The fold indices depend only
    # on (n_rows, seed, n_folds), so every arm/trial uses the same documents.
    _row_spec = L.tuning_row_split_spec(protocol)
    _row_state = {'n_rows': None, 'selection': None, 'evaluation': None,
                  'meta': None}
    for ti, knobs in enumerate(trials):
        for _gi, _grp in enumerate(groups):
            _ov = _group_overrides(base, _grp)
            _suffix = '' if len(groups) == 1 else f'_g{_gi}'
            tdir = os.path.join(outdir, f'trial_{ti:02d}{_suffix}')
            cfg = _trial_cfg(base, tdir, _grp['arms'], search_seeds, search_steps, knobs)
            cfg.update(_ov)
            L.run(cfg, seeds=search_seeds, guard=None, label=f'baseline tuning trial {ti + 1}/{len(trials)} ({"/".join(_grp["arms"])})')
            summary = json.load(open(os.path.join(tdir, 'summary.json'), encoding='utf-8'))
            for arm in _grp['arms']:
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
                if _prov_any is None:
                    _prov_any = records[sorted(records)[0]].get('provenance')
                    _data_version_any = records[sorted(records)[0]].get('data_version')
                full_seed_ppls = {s: float(records[s]['ppl']) for s in sorted(records)}
                selection_ppls = None
                selection_stats = {}
                _split_reason = None
                if _row_spec is not None:
                    _lens = {len(records[s].get('nll_per_row') or []) for s in records}
                    if len(_lens) != 1:
                        _split_reason = 'search records expose different per-row NLL lengths'
                    else:
                        _n_rows = next(iter(_lens))
                        if _n_rows < 2:
                            _split_reason = f'validation slice has only {_n_rows} row(s)'
                        else:
                            if _row_state['meta'] is None:
                                _sel, _ev, _meta = L.validation_row_split(
                                    _n_rows, seed=_row_spec['seed'],
                                    n_folds=_row_spec['n_folds'],
                                    selection_fold=_row_spec['selection_fold'],
                                    evaluation_fold=_row_spec['evaluation_fold'])
                                _row_state.update(n_rows=_n_rows, selection=_sel,
                                                  evaluation=_ev, meta=_meta)
                            if _n_rows != _row_state['n_rows']:
                                _split_reason = ('validation row count changed between trials '
                                                 f'({_n_rows} vs {_row_state["n_rows"]})')
                            else:
                                _tmp = {}
                                for s in sorted(records):
                                    _st = L.record_subset_ppl(records[s], _row_state['selection'])
                                    if _st is None:
                                        _split_reason = 'a search record lacks usable per-row NLL accounting'
                                        _tmp = {}
                                        break
                                    _tmp[s] = _st
                                if _tmp and len(_tmp) == len(records):
                                    selection_stats = _tmp
                                    selection_ppls = {s: float(_tmp[s]['ppl']) for s in sorted(_tmp)}
                if selection_ppls is not None:
                    # Token-pooled within each seed and across search seeds:
                    # the same estimator as the panel's primary PPL.
                    _nll_tot = float(sum(selection_stats[s]['nll_sum'] for s in selection_stats))
                    _tok_tot = int(sum(selection_stats[s]['n_tokens'] for s in selection_stats))
                    mean = float(L._safe_exp_ppl(_nll_tot / _tok_tot))
                    std = _mean_std(list(selection_ppls.values()))[1]
                    _metric_ppls = selection_ppls
                    _seed_nll = {str(s): float(selection_stats[s]['nll_sum']) for s in sorted(selection_stats)}
                    _seed_ntok = {str(s): int(selection_stats[s]['n_tokens']) for s in sorted(selection_stats)}
                    _agg = 'pooled_token_nll'
                else:
                    mean, std = _mean_std(list(full_seed_ppls.values()))
                    _metric_ppls = full_seed_ppls
                    _seed_nll = None
                    _seed_ntok = None
                    _agg = 'arithmetic_seed_mean_fallback'
                    if _row_spec is not None:
                        print(f'[tune] WARNING: {arm} trial {ti}: nested selection rows unavailable ({_split_reason or "per-row NLL missing"}); falling back to full-validation means')
                _steps = {int(records[s]['steps']) for s in records if records[s].get('steps') is not None}
                _tokens = {int(records[s]['tokens_seen']) for s in records if records[s].get('tokens_seen') is not None}
                per_arm[arm].append({
                    'config': dict(knobs), 'ppl': mean, 'ppl_std': std,
                    'ppl_aggregation': _agg,
                    'nll_sum': (float(sum(selection_stats[s]['nll_sum'] for s in selection_stats))
                                if selection_ppls is not None else None),
                    'n_tokens': (int(sum(selection_stats[s]['n_tokens'] for s in selection_stats))
                                 if selection_ppls is not None else None),
                    'seed_nll_sum': _seed_nll, 'seed_n_tokens': _seed_ntok,
                    'n_seeds': len(records), 'seeds': sorted(records),
                    'seed_ppls': {str(s): float(_metric_ppls[s]) for s in sorted(records)},
                    'ppl_full_validation': _mean_std(list(full_seed_ppls.values()))[0],
                    'seed_ppls_full_validation': {str(s): float(full_seed_ppls[s]) for s in sorted(records)},
                    'selection_row_split_applied': bool(selection_ppls is not None),
                    'selection_row_fp': ((_row_state['meta'] or {}).get('selection_row_fp')
                                         if selection_ppls is not None else None),
                    'evaluation_row_fp': ((_row_state['meta'] or {}).get('evaluation_row_fp')
                                          if selection_ppls is not None else None),
                    'params': records[sorted(records)[0]].get('params'),
                    'steps': (next(iter(_steps)) if len(_steps) == 1 else None),
                    'tokens_seen': (next(iter(_tokens)) if len(_tokens) == 1 else None),
                    'budget_by_seed': {str(s): [records[s].get('steps'), records[s].get('tokens_seen')]
                                       for s in sorted(records)},
                    'run_cfg': records[sorted(records)[0]].get('run_cfg'), 'outdir': tdir,
                })
    # Selection must compare every trial on the SAME rows.  If one trial in
    # an arm could not use the declared nested selection fold while another
    # could, comparing the surviving mixture (nested for some configs, full
    # validation for others) is an invalid sample change.  Fall back to the
    # full-validation means for the WHOLE arm so all trials share one sample,
    # and leave the split flags false so downstream audits see that the nested
    # contract was not met.
    if _row_spec is not None:
        for arm in arms:
            rows = per_arm.get(arm) or []
            flags = [bool(t.get('selection_row_split_applied')) for t in rows]
            if any(flags) and not all(flags):
                print(f'[tune] WARNING: {arm}: trials mix nested and full-validation '
                      'selection samples; falling back to full validation for every '
                      'trial in this arm so the selected config is not chosen across '
                      'different row sets')
                for t in rows:
                    full = t.get('seed_ppls_full_validation') or {}
                    _full_vals = [v for v in full.values() if L.ppl_is_usable(v)]
                    _mu, _sd = _mean_std(_full_vals)
                    t['ppl'] = _mu
                    t['ppl_std'] = _sd
                    if full:
                        t['seed_ppls'] = {str(k): float(v) for k, v in full.items()}
                    t['nll_sum'] = None
                    t['n_tokens'] = None
                    t['seed_nll_sum'] = None
                    t['seed_n_tokens'] = None
                    t['ppl_aggregation'] = 'arithmetic_seed_mean_fallback'
                    t['selection_row_split_applied'] = False
                    t['selection_row_fp'] = None
                    t['evaluation_row_fp'] = None
    selected = {}
    for arm in arms:
        hits = [t for t in per_arm[arm] if L.ppl_is_usable(t.get('ppl'))]
        if not hits:
            raise RuntimeError(f'[tune] {arm}: no usable trial PPL — cannot select a config')
        best = min(hits, key=lambda t: float(t['ppl']))
        # A raw argmin over noisy trial means can crown a config that is
        # statistically indistinguishable from the runner-up.  Record the
        # margin in units of its standard error so the audit can warn that
        # "selected" is not "uniquely best".  This does NOT change selection;
        # it makes the uncertainty of the selection itself visible.
        _ranked = sorted(hits, key=lambda t: float(t['ppl']))
        _runner = _ranked[1] if len(_ranked) > 1 else None
        _sel_margin = _sel_margin_se = _sel_margin_z = None
        _sel_decisive = None
        _sel_margin_method = 'unavailable'
        if _runner is not None:
            _mu = L.tuning_selection_margin(best, _runner, metric='ppl',
                                            direction='lower_is_better')
            _sel_margin = _mu.get('margin')
            _sel_margin_se = _mu.get('se')
            _sel_margin_z = _mu.get('z')
            _sel_decisive = _mu.get('decisive')
            _sel_margin_method = _mu.get('method')
        _stability = L.tuning_selection_stability(
            per_arm[arm], metric=protocol['selection_metric'],
            direction=protocol['direction'])
        _best_sig = json.dumps(best['config'], sort_keys=True, default=str)
        _chosen_sig = json.dumps(_stability.get('choices', [{}])[0].get('config') or {}, sort_keys=True,
                                 default=str) if _stability.get('choices') else None
        selected[arm] = {'selected': dict(best['config']),
                         'selected_metric_value': float(best['ppl']),
                         'selection_margin': _sel_margin,
                         'selection_margin_se': _sel_margin_se,
                         'selection_margin_in_se': _sel_margin_z,
                         'selection_margin_method': _sel_margin_method,
                         'selection_decisive': _sel_decisive,
                         'runner_up': ({'config': dict(_runner['config']),
                                        'ppl': float(_runner['ppl'])}
                                       if _runner is not None else None),
                         'params': best.get('params'), 'run_cfg': best.get('run_cfg'),
                         'selection_stability': _stability,
                         'selected_is_loo_stable': bool(
                             _stability.get('stable') and _chosen_sig == _best_sig),
                         'trials': per_arm[arm]}
        print(f'[tune] {arm:22s} best mean PPL {best["ppl"]:.4f} +/- {best["ppl_std"]:.4f} '
              f'(n={best.get("n_seeds")}) at {best["config"]}  '
              f'LOO-stable={selected[arm]["selected_is_loo_stable"]}')
        if _stability.get('status') == 'ok' and not _stability.get('stable'):
            print(f'[tune] WARNING: {arm}: the selected config changes when one search seed is dropped '
                  f'({_stability.get("n_distinct_configs")} distinct winners) — the "tuned" pick is seed-driven')
        if _sel_decisive is False:
            print(f'[tune] WARNING: {arm}: the selected config is only {_sel_margin:.4g} PPL better '
                  f'than the runner-up (SE {_sel_margin_se:.4g}) — the winner is inside one standard '
                  'error; do not describe it as the uniquely best setting')
    artifact_protocol = dict(protocol)
    artifact_protocol['arms'] = list(arms)
    artifact_protocol['trials'] = [dict(t) for t in trials]
    artifact_protocol['search_seeds'] = list(search_seeds)
    if 'arm_groups' in protocol:
        artifact_protocol['arm_groups'] = [dict(g) for g in groups]
    _row_applied = {a: bool(per_arm[a]) and all(bool(t.get('selection_row_split_applied'))
                                                for t in per_arm[a]) for a in arms}
    _row_meta = dict(_row_state['meta'] or {})
    _row_meta.update({'applied': bool(_row_spec is not None and all(_row_applied.values())),
                      'applied_by_arm': _row_applied,
                      'declared': (_row_spec is not None)})
    if _row_spec is not None and not _row_meta['applied']:
        _row_meta['unavailable_arms'] = sorted(a for a, ok in _row_applied.items() if not ok)
        _row_meta['reason'] = ('at least one arm/trial lacks usable per-row NLL accounting; '
                               'its selection fell back to full-validation means')
    if _row_meta.get('applied'):
        print(f"[tune] nested row split applied: selection rows={_row_meta.get('n_selection_rows')} "
              f"(fp {str(_row_meta.get('selection_row_fp'))[:12]}…), evaluation rows="
              f"{_row_meta.get('n_evaluation_rows')} (fp {str(_row_meta.get('evaluation_row_fp'))[:12]}…)")
    artifact = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': artifact_protocol,
        'selection_metric': protocol['selection_metric'],
        'selection_split': protocol['selection_split'],
        'selection_row_split': _row_meta,
        'direction': protocol['direction'],
        'search_seed': search_seed,
        'search_seeds': list(search_seeds),
        'search_final_seed_overlap': list(overlap),
        'search_final_seed_disjoint': (not overlap),
        'search_steps': search_steps,
        'declared_trials': len(declared_trials),
        'trials_run': len(trials),
        'search_complete': bool(len(trials) == len(declared_trials)),
        'final_seeds': list(final_seeds_declared),
        'arms': selected,
        'code': L.code_stamp(),
        'provenance': _prov_any or L.code_provenance(),
        'data_version': _data_version_any,
    }
    path = os.path.join(outdir, 'baseline_tuning.json')
    L.atomic_write_json(path, artifact)
    audit = L.baseline_tuning_audit(path=path)
    print(f'[tune] artifact: {path}  audit={audit.get("status")}')
    if not audit.get('tuning_verified'):
        raise RuntimeError(f'[tune] tuning artifact failed its own audit: {audit.get("problems")}')
    if final:
        _run_final(base, outdir, protocol, selected)
        # The final summary keeps full-validation PPL for context, but the
        # audit must not silently re-introduce the selection documents.  Feed
        # it the held-out-row estimate whenever the nested split was applied.
        _final_summary = json.load(open(os.path.join(outdir, 'final', 'summary.json'), encoding='utf-8'))
        _audit_summary = {}
        for _k, _r in _final_summary.items():
            if not isinstance(_r, dict):
                continue
            _rr = dict(_r)
            if L.ppl_is_usable(_rr.get('ppl_holdout')):
                _rr['ppl'] = float(_rr['ppl_holdout'])
                # Keep the NLL accounting on the SAME sample as the reported
                # PPL.  Leaving full-validation nll_sum/n_tokens in place made
                # the token-weighted aggregate and the metric-aggregation
                # check describe a different sample than `ppl`.
                if _rr.get('nll_sum_holdout') is not None:
                    _rr['nll_sum'] = _rr['nll_sum_holdout']
                if _rr.get('n_tokens_holdout') is not None:
                    _rr['n_tokens'] = _rr['n_tokens_holdout']
                _rr['ppl_estimation_rows'] = 'nested_validation_holdout'
            else:
                _rr['ppl_estimation_rows'] = 'full_validation'
            _audit_summary[_k] = _rr
        L.methodology_audit(_audit_summary, outdir=os.path.join(outdir, 'final'), verbose=True)
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
    # The final tuned metric may be a held-out row fold; the strict
    # `pair_reason` already refuses a mixed evaluation slice, and the relaxed
    # tuned rule must not become a back door around it.  Legacy records with
    # no eval fingerprint on either side keep the permissive path.
    for _key in ('eval_fp', 'eval_subset_fp'):
        _ea, _eb = a.get(_key), b.get(_key)
        _ha = _ea is not None and str(_ea) != ''
        _hb = _eb is not None and str(_eb) != ''
        if _ha != _hb:
            return f'unverifiable {_key} (only one record carries the evaluation-slice fingerprint)'
        if _ha and str(_ea) != str(_eb):
            return f'{_key} differs ({str(_ea)[:12]}… vs {str(_eb)[:12]}…)'
    # The tuned estimate may be computed on a deterministic held-out row fold
    # (`ppl_holdout`).  Different holdout fingerprints mean the two arms are
    # not scored on the same rows even when the evaluation slice matches.
    _ra, _rb = a.get('holdout_row_fp'), b.get('holdout_row_fp')
    _hha = _ra is not None and str(_ra) != ''
    _hhb = _rb is not None and str(_rb) != ''
    if _hha != _hhb:
        return 'unverifiable holdout_row_fp (only one record carries the held-out row fingerprint)'
    if _hha and str(_ra) != str(_rb):
        return f'holdout_row_fp differs ({str(_ra)[:12]}… vs {str(_rb)[:12]}…)'
    # Tuning may change lr/wd/warmup, but it must never silently change the
    # optimizer, betas/eps, LR-schedule shape, grad-clip or augmentation.  The
    # strict `pair_reason` already enforces this for architecture contrasts;
    # the relaxed tuned rule must not become a back door around it.
    ka, kb = a.get('train_knobs'), b.get('train_knobs')
    if (ka is None) != (kb is None):
        return 'unverifiable train_knobs (only one record carries the optimizer/schedule/augmentation descriptor)'
    if ka is not None and ka != kb:
        return 'different train_knobs (optimizer / LR-schedule / grad-clip / augmentation)'
    # Seed equality does not imply initialization equality: different
    # architectures consume different amounts of RNG.  The shared-trunk digest
    # is the machine check that the paired seeded arms actually started equal.
    ia, ib = a.get('init_core_fp'), b.get('init_core_fp')
    if (ia is None) != (ib is None):
        return 'unverifiable init_core_fp (only one record carries the matched-initialization digest)'
    if ia is not None and ib is not None and ia != ib:
        return f'different init_core_fp ({str(ia)[:12]}… vs {str(ib)[:12]}…)'
    # Capacity control must fail CLOSED.  The old try/except turned a missing
    # `params` field (int(None) -> TypeError) into a silent pass, so a tuned
    # contrast could be reported with no capacity check at all.  The shared
    # helper also compares the recorded non-embedding/active breakdown, since
    # equal totals can hide a reallocated MLP or dead indexer projections.
    _param_why = L.param_pair_reason(a, b)
    if _param_why:
        return _param_why
    if tuned_arms.get(a.get('variant')) != tuned_arms.get(b.get('variant')):
        return 'arms did not receive the same declared tuning range'
    return None


def _run_final(base, outdir, protocol, selected):
    # Use the same normalised plan the tuning driver validated (integers,
    # unique).  Reading the raw protocol here let a duplicate seed reach the
    # runner, which would deduplicate it under the cover of the run key.
    search_seeds, final_seeds, overlap = _search_final_seeds(protocol)
    if not final_seeds:
        raise ValueError('_run_final: final_seeds must be a non-empty declared partition')
    if final_seeds and overlap:
        raise ValueError(
            'search_seeds and final_seeds overlap on %s — final tuned records '
            'must not be measured on the seeds that selected the config; '
            'declare a disjoint final partition' % overlap)
    _groups = _protocol_arm_groups(protocol, base, list(selected))
    _arm_group = {}
    for _grp in _groups:
        for _a in _grp['arms']:
            _arm_group[_a] = _grp
    final_dir = os.path.join(outdir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    merged = {}
    for arm, entry in selected.items():
        adir = os.path.join(final_dir, arm)
        cfg = dict(base, outdir=adir, variants=[arm], seeds=final_seeds)
        cfg.update(_group_overrides(base, _arm_group.get(arm) or {}))
        cfg.update(lr=float(entry['selected']['lr']),
                   weight_decay=float(entry['selected']['weight_decay']),
                   warmup=int(entry['selected']['warmup']))
        L.run(cfg, seeds=final_seeds, guard=None, label=f'final tuned {arm}')
        s = json.load(open(os.path.join(adir, 'summary.json'), encoding='utf-8'))
        for rec in s.values():
            if isinstance(rec, dict) and L.ppl_is_usable(rec.get('ppl')) and 'seed' in rec:
                merged[f'{arm}::seed{int(rec["seed"])}'] = dict(rec)
    # Nested validation, evaluation half: recompute each final record's
    # token-pooled PPL on the row fold that was NEVER used for selection.
    # Full-validation PPL stays in the record for context; the tuned contrast
    # below uses the held-out fold.
    _row_spec = L.tuning_row_split_spec(protocol)
    _final_rows_meta = {'declared': _row_spec is not None, 'applied': False}
    _eval_idx = None
    if _row_spec is not None:
        _lens = {len(r.get('nll_per_row') or []) for r in merged.values()} if merged else set()
        if len(_lens) == 1 and min(_lens) >= 2:
            _n_rows = next(iter(_lens))
            _sel, _eval_idx, _meta = L.validation_row_split(
                _n_rows, seed=_row_spec['seed'], n_folds=_row_spec['n_folds'],
                selection_fold=_row_spec['selection_fold'],
                evaluation_fold=_row_spec['evaluation_fold'])
            _final_rows_meta.update(_meta)
            # Compute every holdout estimate BEFORE mutating any record.  A
            # partial assignment would leave some records on the later row
            # fold and others on full validation, so the paired contrast would
            # silently mix two samples even though the metadata says the
            # nested split was not applied.  Either all records move to the
            # held-out fold or none do.
            _pending, _bad = [], []
            for k, r in merged.items():
                st = L.record_subset_ppl(r, _eval_idx)
                # A finite row index is not enough: an overflowed/underflowed
                # holdout PPL is unmeasured.  Treating it as applied would let
                # `_eval_metric` silently fall back to full validation for
                # THAT record while the artifact still claims every seed used
                # the held-out fold.
                if st is None or not L.ppl_is_usable(st.get('ppl')):
                    _bad.append(k)
                    continue
                _pending.append((r, st))
            _final_rows_meta['applied'] = not _bad
            if _bad:
                _final_rows_meta['reason'] = ('final records without usable per-row NLL '
                                              f'accounting: {sorted(_bad)[:5]}')
            else:
                for r, st in _pending:
                    r['ppl_full_validation'] = r.get('ppl')
                    r['ppl_holdout'] = st['ppl']
                    r['nll_sum_holdout'] = st['nll_sum']
                    r['n_tokens_holdout'] = st['n_tokens']
                    r['holdout_row_fp'] = _meta.get('evaluation_row_fp')
                    r['ppl_estimation_rows'] = 'nested_validation_holdout'
                    # Keep the per-row accounting on the SAME held-out rows as
                    # the reported PPL.  The full-validation arrays stay for
                    # context, but a paired row bootstrap must not resample
                    # rows that selected the hyperparameters.
                    r['nll_per_row_holdout'] = [float(r['nll_per_row'][int(i)])
                                                for i in _eval_idx]
                    r['n_tokens_per_row_holdout'] = [int(r['n_tokens_per_row'][int(i)])
                                                     for i in _eval_idx]
        else:
            _final_rows_meta['reason'] = ('final records expose different per-row NLL lengths '
                                          'or fewer than 2 rows')
        if not _final_rows_meta['applied']:
            print(f"[tune] WARNING: nested final holdout unavailable ({_final_rows_meta.get('reason')}); "
                  'the tuned contrast falls back to full-validation PPL')
        else:
            print(f"[tune] final tuned contrast uses the held-out row fold: "
                  f"{_final_rows_meta.get('n_evaluation_rows')} rows "
                  f"(fp {str(_final_rows_meta.get('evaluation_row_fp'))[:12]}…)")
    L.atomic_write_json(os.path.join(final_dir, 'summary.json'), merged)
    def _eval_metric(rec):
        v = rec.get('ppl_holdout') if isinstance(rec, dict) else None
        if L.ppl_is_usable(v):
            return float(v)
        return float(rec['ppl'])
    _live = [r for r in merged.values() if isinstance(r, dict)]
    _prov = next((r.get('provenance') for r in _live if r.get('provenance')), None) or L.code_provenance()
    _dv = next((r.get('data_version') for r in _live if r.get('data_version')), None)
    _final_cfg = {'seeds': final_seeds, 'variants': list(selected), 'outdir': final_dir, 'n_train_tokens': base.get('n_train_tokens'), 'seq_len': base.get('seq_len')}
    _final_fp = L.preflight_fingerprint({'selected': selected, 'seeds': final_seeds, 'base': base}, stage='baseline_final')
    _planned = [f'{a}::seed{s}' for a in selected for s in final_seeds]
    try:
        L.write_run_manifest(final_dir, _final_cfg, _final_fp, _prov, merged,
                             status='complete', planned_cells=_planned)
        L.log_experiment_event(final_dir, {'kind': 'panel_start', 'tracking_schema': 2, 'label': 'tuned final baselines', 'run_cfg': _final_fp, 'code': L.code_stamp(), 'provenance': _prov, 'data_version': _dv, 'seeds': final_seeds, 'variants': list(selected), 'n_cells_planned': len(_planned)})
        L.log_experiment_event(final_dir, {'kind': 'panel_finish', 'tracking_schema': 2, 'label': 'tuned final baselines', 'run_cfg': _final_fp, 'code': L.code_stamp(), 'provenance': _prov, 'data_version': _dv, 'n_cells_planned': len(_planned), 'n_cells_measured': len(merged), 'n_cells_missing': max(0, len(_planned) - len(merged))})
    except Exception as _te:
        print(f'[tune] WARNING: could not write the final-panel manifest/events ({type(_te).__name__}: {_te})')
    tuning = json.load(open(os.path.join(outdir, 'baseline_tuning.json'), encoding='utf-8'))
    tuning['final_outdir'] = final_dir
    tuning['final_records'] = {k: {'ppl': v.get('ppl'), 'seed': v.get('seed'),
                                   'ppl_holdout': v.get('ppl_holdout'),
                                   'nll_sum_holdout': v.get('nll_sum_holdout'),
                                   'n_tokens_holdout': v.get('n_tokens_holdout'),
                                   'holdout_row_fp': v.get('holdout_row_fp'),
                                   'params': v.get('params'), 'run_cfg': v.get('run_cfg'),
                                   'hyperparams': v.get('hyperparams')}
                               for k, v in merged.items()}
    tuning['final_evaluation_rows'] = _final_rows_meta
    L.atomic_write_json(os.path.join(final_dir, 'baseline_tuning.json'), tuning)
    # Arm-level mean±std over the same final seed set, so the tuned table is
    # not a single aggregate number per arm.  The paired contrast below is
    # still computed per seed (paired by construction), not by subtracting
    # these two means.
    arm_summary = {}
    for arm, entry in selected.items():
        _recs = {int(r['seed']): _eval_metric(r) for r in merged.values()
                 if isinstance(r, dict) and r.get('variant') == arm
                 and r.get('seed') is not None and L.ppl_is_usable(r.get('ppl'))}
        _vals = [_recs[s] for s in sorted(_recs)]
        _full = [float(r['ppl']) for r in merged.values()
                 if isinstance(r, dict) and r.get('variant') == arm
                 and r.get('seed') is not None and L.ppl_is_usable(r.get('ppl'))]
        arm_summary[arm] = {
            'n_seeds': len(_vals), 'seeds': sorted(_recs),
            'ppl_mean': float(np.mean(_vals)) if _vals else float('nan'),
            'ppl_std': float(np.std(_vals, ddof=1)) if len(_vals) > 1 else 0.0,
            'ppl_full_validation_mean': float(np.mean(_full)) if _full else float('nan'),
            'estimation_rows': ('nested_validation_holdout' if _final_rows_meta.get('applied')
                                else 'full_validation'),
            'selected': dict(entry.get('selected') or {}),
        }
    tuned_arms = {a: len(entry['trials']) for a, entry in selected.items()}
    plan = _contrast_plan(protocol, _groups, list(selected))
    comparison = {'pairing_rule': 'tuned_equal_budget',
                  'note': 'Each arm used its own validation-selected hyperparameters; data/val fingerprints, steps, tokens and parameters are matched. This is NOT the strict same-run_cfg architectural contrast and must be reported separately.',
                  'final_seeds': list(final_seeds),
                  'search_seeds': list(search_seeds),
                  'search_final_seed_disjoint': (not overlap),
                  'estimation_rows': ('nested_validation_holdout' if _final_rows_meta.get('applied')
                                      else 'full_validation'),
                  'selection_row_split': (tuning.get('selection_row_split') or {}),
                  'final_evaluation_rows': _final_rows_meta,
                  'declared_contrasts': [f'{a} - {b}' for a, b in plan],
                  'arm_summary': arm_summary,
                  'contrasts': {}}
    for method, baseline in plan:
        deltas, seeds, skipped, pairs = [], [], [], []
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
            deltas.append(_eval_metric(ra) - _eval_metric(rb))
            seeds.append(seed)
            pairs.append((ra, rb))
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
                   'mean_pm_std': (f'{st["mean"]:+.4f} +/- {st["std"]:.4f}' if st.get('n') else 'n/a'),
                   'paired_row_bootstrap_delta': L.paired_row_bootstrap_delta(pairs)})
        comparison['contrasts'][f'{method} - {baseline}'] = st
    # The tuned contrasts form their own DECLARED family.  A contrast that is
    # declared but unmeasurable keeps a p=1 slot; dropping it from the family
    # would make the remaining tests look better than the study design allows.
    _named = []
    for name, st in comparison['contrasts'].items():
        _p = st.get('p_exact_signflip')
        try:
            _p = float(_p)
        except (TypeError, ValueError, OverflowError):
            _p = float('nan')
        _named.append((name, _p if np.isfinite(_p) else 1.0))
    _holm = L.holm_adjust(_named, alpha=0.05) if _named else {}
    _m = len(plan)
    for name, st in comparison['contrasts'].items():
        _h = _holm.get(name) or {}
        st['p_holm'] = _h.get('p_holm')
        st['declared_family_m'] = _m
        st['significant_holm'] = bool(_h.get('reject') and (not st.get('within_seed_noise'))
                                      and st.get('controlled', True))
    _final_counts = {a: int(v.get('n_seeds') or 0) for a, v in arm_summary.items()}
    comparison['family'] = {
        'name': 'tuned_equal_budget_baselines',
        'm': _m, 'declared_m': _m, 'measured_contrasts': len(_named),
        'alpha': 0.05, 'correction': 'holm step-down',
        'method_arms': sorted({a for a, _b in plan}),
        'final_seed_count_by_arm': _final_counts,
        'final_seed_coverage_equal': (len(set(_final_counts.values())) == 1) if _final_counts else None,
        'min_seeds_for_family_significance': L.signflip_min_n_for_holm(_m, 0.05) if _m else None,
        'note': 'per-arm validation-selected hyperparameters; a tuned win is NOT a strict same-run_cfg architectural contrast. Contrasts are declared per parameter-matching group; unmeasurable declared pairs stay in the Holm family with p=1.',
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
    # Use the same nested chronological row split as the neural panel so the
    # floor's smoothing is selected on earlier rows and reported on disjoint
    # later rows.  With fewer than two rows per fold the split is not
    # meaningful, so fall back to the legacy full-validation path and say so.
    _proto = protocol if protocol is not None else L.SIMPLE_BASELINE_POLICY
    _row_spec = L.tuning_row_split_spec(_proto)
    _sel_rows = _eval_rows = _row_meta = None
    if _row_spec is not None and rows >= 4:
        _sel_rows, _eval_rows, _row_meta = L.validation_row_split(
            rows, seed=_row_spec['seed'], n_folds=_row_spec['n_folds'],
            selection_fold=_row_spec['selection_fold'],
            evaluation_fold=_row_spec['evaluation_fold'])
    artifact = L.simple_baseline_fit(
        train_ids, eval_slice, vocab, protocol=protocol,
        selection_rows=_sel_rows, evaluation_rows=_eval_rows, row_split=_row_meta)
    artifact.setdefault('data', {})['cache_val_fp'] = L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1))
    artifact['data']['eval_rows'] = int(len(_eval_rows) if _eval_rows is not None else rows)
    if _sel_rows is None:
        artifact['data']['nested_selection_unavailable_reason'] = (
            'validation has fewer than 4 rows (two per chronological fold)' if rows < 4
            else 'the simple-baseline protocol declares no nested row split')
        artifact['data']['eval_fp'] = L._ids_fp(np.ascontiguousarray(eval_slice).reshape(-1))
    artifact['data_version'] = L._data_version(artifact['data'])
    if isinstance(artifact.get('provenance'), dict):
        artifact['provenance']['data'] = dict(artifact['data'])
        artifact['provenance']['data']['data_version'] = artifact['data_version']
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'simple_baseline.json')
    L.atomic_write_json(path, artifact)
    audit = L.simple_baseline_audit(path=path)
    print('[simple] best count-model floor: %s PPL=%.4f (%s)  audit=%s' % (
        artifact.get('best_baseline'), float(artifact.get('best_ppl') or float('nan')),
        artifact.get('best_ppl_estimation_rows') or 'full_validation', audit.get('status')))
    for arm, entry in sorted((artifact.get('baselines') or {}).items()):
        _hold = entry.get('holdout_metric_value')
        print('  %-8s selected %s  selection PPL=%.4f%s' % (
            arm, entry.get('selected'), float(entry.get('selected_metric_value') or float('nan')),
            (f'  holdout PPL={float(_hold):.4f}' if _hold is not None else '')))
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
                    search_seeds=[7], final_seeds=[0])
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
