
def v10_analysis(out='analysis_v10/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    try:
        import explib.rope as _V7
        _niah_factor_rep = L.niah_contrast_factor_audit(
            factory=_V7.make_layer_cfgs_v7, verbose=False)
    except Exception as _e:
        _niah_factor_rep = {'status': 'not_verifiable', 'pairs': {},
                            'note': f'{type(_e).__name__}: {_e}'}
    probe = {}
    sp = PROBE['summary']
    if os.path.exists(sp):
        try:
            raw = json.load(open(sp, encoding='utf-8'))
        except Exception as _e:
            print(f'[stats] FATAL: {sp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — refusing to compute probe statistics from a truncated file; move it aside to re-probe.')
            raise
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
        cells, _dup_cells = ({}, [])
        for k, r in _usable.items():
            if r.get('probe_params') is None:
                _bad_cells.append((k, 'no probe_params'))
                continue
            if r.get('synthesized'):
                _bad_cells.append((k, 'synthesized reconstruction is not an independent measurement'))
                continue
            if r.get('_code') != L.code_stamp():
                _bad_cells.append((k, 'stale code stamp'))
                continue
            if r.get('val_fp') in (None, ''):
                _bad_cells.append((k, 'no validation fingerprint'))
                continue
            _mode = str(r.get('mode', 'uniform'))
            # the grouping key carries the mode EXPLICITLY (probe_params' mode
            # field is normalised out by _fp_norm): uniform and natural cells
            # at the same rho must never merge into one pooled cell.  val_fp is
            # part of the identity: different evaluation corpora are not one
            # cell.
            key = (r['variant'], r['arm'], r['eval_len'], r['rho'], _mode, json.dumps(_fp_norm(r.get('probe_params')), sort_keys=True), str(r.get('_code')), str(r.get('recipe')), str(r.get('val_fp')))
            grp = cells.setdefault(key, {})
            if r['seed'] in grp:
                _dup_cells.append((k, key[:5], r['seed']))
                continue
            grp[r['seed']] = r
        if _dup_cells:
            print(f'[v10 stats] {len(_dup_cells)} probe record(s) share a (cell, seed) identity — their PPL is ambiguous and they are dropped from the pooling: ' + ', '.join((f'{k}' for k, _c, _s in _dup_cells[:4])))
        _new_bad = _bad_cells[_n_bad_pre:]
        if _new_bad:
            print(f'[v10 stats] {len(_new_bad)} record(s) in {sp} lack `probe_params` and are dropped from every mean/paired statistic: ' + ', '.join((f'{k}({why})' for k, why in _new_bad[:4])))
        probe_cells = []
        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items(), key=lambda kv: kv[0]):
            means = {s: r['ppl_mean'] for s, r in by_seed.items()}
            assert means, (v, arm, Ln, rho, mode)
            probe_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'seeds': sorted(means), 'ppl_by_seed': means, 'probe_params': next(iter(by_seed.values())).get('probe_params'), '_code': _cd, 'recipe': _recipe, 'val_fp': _vfp, 'mean': float(np.mean(list(means.values()))), 'n': len(means)})
        contrasts = {}

        # The grouped lists are exactly what a scan of probe_cells would have
        # collected, so the ambiguity/unambiguous outcomes (and the picked
        # cell) are unchanged.
        _cell_idx = {}
        for c in probe_cells:
            _cell_idx.setdefault((c['variant'], c['arm'], c['eval_len'], c['rho'], c['mode']), []).append(c)

        def cell_mean(v, arm, Ln, rho, mode='uniform'):
            hits = _cell_idx.get((v, arm, Ln, rho, mode), [])
            if len(hits) > 1:
                print(f'[v10 stats] ({v}/{arm}/L{Ln}/r{rho}/{mode}) holds {len(hits)} distinct probe parameterisations/recipes/code stamps — no unambiguous cell, so the contrast is omitted rather than pooled across them')
                return {}
            return hits[0] if hits else {}

        def _pairable(a, b, Ln, rho, tag, mode='uniform'):
            if a.get('val_fp') != b.get('val_fp'):
                print(f"[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms were measured on DIFFERENT validation slices ({a.get('val_fp')} vs {b.get('val_fp')}) — pairing them would difference two different measurements, so the contrast is omitted")
                return False
            if _fp_norm(a.get('probe_params')) != _fp_norm(b.get('probe_params')):
                print(f'[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms were probed under DIFFERENT probe_params — pairing them would difference two different measurements, so the contrast is omitted')
                return False
            if a.get('_code') != b.get('_code') or _recipe_norm(a.get('recipe')) != _recipe_norm(b.get('recipe')):
                print(f"[v10 stats] L{Ln} r{rho} {mode} {tag}: the two arms carry DIFFERENT code/recipe stamps (learned {a.get('_code')}/{a.get('recipe')} vs {b.get('_code')}/{b.get('recipe')}) — pairing them would mix two measurement semantics, so the contrast is omitted; re-run the stale arm's phase to refresh it")
                return False
            return True

        def _add_contrast(name, a, b):
            common = sorted(set(a['ppl_by_seed']) & set(b['ppl_by_seed']))
            if len(common) >= 2:
                dl = [b['ppl_by_seed'][s] - a['ppl_by_seed'][s] for s in common]
                contrasts[name] = V.exact_sign_permutation(dl)
                contrasts[name]['seeds'] = common
        # ---- legacy uniform-mode contrasts (names unchanged for report compat) ----
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                for arm, tag in (('dense', 'full_rope(dense)'), ('randidx', 'csa+randidx'), ('allblocks', 'csa+allblocks')):
                    other = cell_mean('full_rope', 'dense', Ln, rho, 'uniform') if arm == 'dense' else cell_mean('csa_fixed_rope', arm, Ln, rho, 'uniform')
                    if not learned or not other:
                        continue
                    if not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag} - csa+learned', learned, other)
        # ---- from-scratch selector controls ----
        for Ln in PROBE['eval_lens']:
            for rho in PROBE['rhos']:
                learned = cell_mean('csa_fixed_rope', 'learned', Ln, rho, 'uniform')
                if not learned:
                    continue
                for tag, ov, oarm in (('csa_randtrain', 'csa_fixed_rope_randidx', 'learned'), ('csa_alltrain', 'csa_fixed_rope_allblocks', 'learned')):
                    other = cell_mean(ov, oarm, Ln, rho, 'uniform')
                    if not other or not _pairable(learned, other, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag} - csa+learned', learned, other)
                # intervention-vs-retrain gap (NOT a causal
                # mismatch decomposition): test-time intervention minus the
                # trained-with-that-selection control at the same (L, rho).
                # For losses the positive direction means the intervention
                # scores WORSE than retraining that selection — a gap that MAY
                # itself contain train/test-mismatch effects (plus optimisation
                # adaptation and parameter differences), so it must be read as
                # a raw difference, never as 'the part that cannot be
                # attributed to mismatch'.
                for tag, tt_arm, sv in (('retrain-gap(random): csa+randidx - csa_randtrain', 'randidx', 'csa_fixed_rope_randidx'), ('retrain-gap(allblocks): csa+allblocks - csa_alltrain', 'allblocks', 'csa_fixed_rope_allblocks')):
                    tt = cell_mean('csa_fixed_rope', tt_arm, Ln, rho, 'uniform')
                    sc = cell_mean(sv, 'learned', Ln, rho, 'uniform')
                    if not tt or not sc or not _pairable(tt, sc, Ln, rho, tag):
                        continue
                    _add_contrast(f'L{Ln} r{rho}: {tag}', sc, tt)
                # architecture-aligned dense: full_rope minus
                # the all-blocks CSA twin (same projections/norm/RoPE/sink)
                dense = cell_mean('full_rope', 'dense', Ln, rho, 'uniform')
                allt = cell_mean('csa_fixed_rope_allblocks', 'learned', Ln, rho, 'uniform')
                if dense and allt and _pairable(dense, allt, Ln, rho, 'arch-aligned'):
                    _add_contrast(f'L{Ln} r{rho}: arch-aligned: full_rope - csa_alltrain', allt, dense)
        # ---- delta-NLL on the corruption modes ----
        delta_nll = {}

        def _nll_tok(rec):
            try:
                ns = [float(x) for x in (rec.get('nll_sum') or [])]
                nt = [int(x) for x in (rec.get('n_tok') or [])]
            except (TypeError, ValueError, OverflowError):
                return float('nan')
            tot = sum(nt)
            if not nt or tot <= 0:
                return float('nan')
            v = math.fsum(ns) / tot
            return v if math.isfinite(v) else float('nan')

        for (v, arm, Ln, rho, mode, _pp, _cd, _recipe, _vfp), by_seed in sorted(cells.items()):
            if mode == 'uniform' and abs(float(rho)) < 1e-12:
                continue
            # the clean baseline is the SAME arm's rho=0 uniform cell under the
            # same parameterisation/code/recipe; deltas are per-seed paired.
            # `span` carries an extra span_len in probe_params (it changes the
            # corruption geometry), but the clean ρ=0 cell does not.  Strip only
            # the mode-specific key for the lookup; the measured cells keep it.
            # `_pp` is the canonical JSON string used in the grouping key, so
            # parse it before normalising (a string is not a dict).
            _pp_clean = _pp
            if isinstance(_pp, str):
                try:
                    _pp_obj = json.loads(_pp)
                except (TypeError, ValueError, OverflowError):
                    _pp_obj = None
                if isinstance(_pp_obj, dict):
                    _pp_clean = json.dumps(_clean_probe_params(_pp_obj), sort_keys=True)
            clean_key = (v, arm, Ln, 0.0, 'uniform', _pp_clean, _cd, _recipe, _vfp)
            clean_grp = cells.get(clean_key) or {}
            if not clean_grp:
                continue
            per_seed = {}
            for s, r in by_seed.items():
                cr = clean_grp.get(s)
                if cr is None:
                    continue
                a, b = (_nll_tok(r), _nll_tok(cr))
                if math.isfinite(a) and math.isfinite(b):
                    per_seed[s] = a - b
            if len(per_seed) >= 1:
                means = {s: per_seed[s] for s in sorted(per_seed)}
                _dl = list(means.values())
                entry = {'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho, 'mode': mode, 'val_fp': _vfp, 'probe_params': json.loads(_pp) if isinstance(_pp, str) else _pp, '_code': _cd, 'recipe': _recipe, 'n': len(_dl), 'mean': float(np.mean(_dl)), 'std': float(np.std(_dl, ddof=1)) if len(_dl) > 1 else 0.0, 'per_seed': means, 'clean_nll_by_seed': {s: _nll_tok(r) for s, r in clean_grp.items() if math.isfinite(_nll_tok(r))}, 'note': 'delta = NLL_corrupted - NLL_clean, nats per target token'}
                delta_nll[f'{v}::{arm}::L{Ln}::{mode}::r{rho}'] = entry
        # paired delta-NLL contrasts between arms at the same (L, mode, rho)
        for Ln in PROBE['eval_lens']:
            _modes = (PROBE.get('modes') or {}).keys() or ['uniform']
            for mode in [m for m in _modes if m != 'uniform'] + ['uniform']:
                _rhos = ((PROBE.get('modes') or {}).get(mode) or ([r for r in PROBE['rhos'] if r > 0] if mode == 'uniform' else []))
                for rho in [r for r in _rhos if r > 0]:
                    learned = delta_nll.get(f'csa_fixed_rope::learned::L{Ln}::{mode}::r{rho}')
                    for arm, tag in (('dense', 'full_rope::dense'), ('randidx', 'csa_fixed_rope::randidx'), ('allblocks', 'csa_fixed_rope::allblocks')):
                        other = delta_nll.get(f'{tag}::L{Ln}::{mode}::r{rho}')
                        if not learned or not other:
                            continue
                        # The delta entries also have to come from the same
                        # validation slice / probe / code, not merely share a
                        # (variant, arm, L, mode, rho) key.
                        if learned.get('val_fp') != other.get('val_fp'):
                            print(f"[v10 stats] ΔNLL L{Ln} {mode} r{rho} {arm}: the two arms were measured on DIFFERENT validation slices ({learned.get('val_fp')} vs {other.get('val_fp')}) — contrast omitted")
                            continue
                        if (_fp_norm(learned.get('probe_params')) != _fp_norm(other.get('probe_params'))
                                or learned.get('_code') != other.get('_code')
                                or _recipe_norm(learned.get('recipe')) != _recipe_norm(other.get('recipe'))):
                            print(f'[v10 stats] ΔNLL L{Ln} {mode} r{rho} {arm}: probe/code/recipe identity differs — contrast omitted')
                            continue
                        common = sorted(set(learned['per_seed']) & set(other['per_seed']))
                        if len(common) >= 2:
                            dl = [other['per_seed'][s] - learned['per_seed'][s] for s in common]
                            res = V.exact_sign_permutation(dl)
                            res['seeds'] = common
                            contrasts[f'ΔNLL L{Ln} {mode} r{rho}: {arm} - csa+learned'] = res
        # ---- Holm correction over the whole contrast family ----
        holm = {}
        _fam = _holm_family(contrasts)
        if _fam:
            holm = L.holm_adjust(_fam)
            for _k, _v in contrasts.items():
                if isinstance(_v, dict):
                    _v['p_holm'] = (holm.get(_k) or {}).get('p_holm') if _k in holm else None
            _rej = [k for k, v in holm.items() if v.get('reject')]
            print(f'[v10 stats] Holm step-down over the {len(_fam)}-contrast family (EXPLORATORY — not pre-registered): {len(_rej)} survive at alpha=0.05' + (f': {sorted(_rej)[:4]}' if _rej else ''))
        probe = {'cells': probe_cells, 'contrasts': contrasts, 'delta_nll': delta_nll, 'holm_exploratory': holm}
        # ---- NIAH-natural (far-necessity task) ----
        _nsp = (PROBE.get('niah') or {}).get('summary')
        if _nsp and os.path.exists(_nsp):
            try:
                nraw = json.load(open(_nsp, encoding='utf-8'))
            except Exception as _e:
                print(f'[stats] WARNING: {_nsp} exists but cannot be parsed ({type(_e).__name__}: {_e}) — the NIAH-natural panel is skipped')
                nraw = {}
            ncells = {}
            for _k, r in nraw.items():
                if not isinstance(r, dict) or r.get('task') != 'niah_natural':
                    continue
                if r.get('variant') is None or r.get('seed') is None or r.get('arm') is None or r.get('condition') is None:
                    continue
                if r.get('synthesized') or r.get('_code') != L.code_stamp() or r.get('val_fp') in (None, ''):
                    continue
                nkey = (r['variant'], r['arm'], r['eval_len'], r['condition'], json.dumps(_fp_norm(r.get('probe_params')), sort_keys=True), str(r.get('_code')), str(r.get('recipe')), str(r.get('val_fp')))
                grp = ncells.setdefault(nkey, {})
                if r['seed'] in grp:
                    continue
                grp[r['seed']] = r
            niah_cells = []
            for (v, arm, Ln, cond, _pp, _cd, _recipe, _vfp), by_seed in sorted(ncells.items(), key=lambda kv: kv[0]):
                accs = {s: r['acc'] for s, r in by_seed.items()}
                nlls = {s: r['nll'] for s, r in by_seed.items()}
                bals = {s: r.get('balanced_accuracy') for s, r in by_seed.items()}
                _f1m = {s: r.get('f1_macro') for s, r in by_seed.items()}
                _f1w = {s: r.get('f1_weighted') for s, r in by_seed.items()}
                _first = next(iter(by_seed.values()))
                _bal_vals = [float(x) for x in bals.values()
                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(float(x))]
                _f1m_vals = [float(x) for x in _f1m.values()
                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(float(x))]
                _f1w_vals = [float(x) for x in _f1w.values()
                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(float(x))]
                niah_cells.append({'variant': v, 'arm': arm, 'eval_len': Ln, 'condition': cond, 'seeds': sorted(accs), 'acc_by_seed': accs, 'nll_by_seed': nlls, 'balanced_by_seed': bals, 'f1_macro_by_seed': _f1m, 'f1_weighted_by_seed': _f1w, 'acc_mean': float(np.mean(list(accs.values()))), 'balanced_mean': (float(np.mean(_bal_vals)) if _bal_vals else None), 'f1_macro_mean': (float(np.mean(_f1m_vals)) if _f1m_vals else None), 'f1_weighted_mean': (float(np.mean(_f1w_vals)) if _f1w_vals else None), 'nll_mean': float(np.mean(list(nlls.values()))), 'recommended_primary_metric': _first.get('recommended_primary_metric'), 'copy_rule_acc': (_first.get('copy_rule_baseline') or {}).get('acc'), 'copy_rule_n_missing_context': (_first.get('copy_rule_baseline') or {}).get('n_missing_context'), 'minority_prevalence': _first.get('minority_prevalence'), 'chance_accuracy': _first.get('chance_accuracy'), 'majority_accuracy': _first.get('majority_accuracy'), 'n_classes': _first.get('n_classes'), 'n': len(accs), 'n_pairs': _first.get('n_pairs'), 'dist_mean': _first.get('dist_mean'), 'val_fp': _vfp, 'probe_params': _first.get('probe_params'), '_code': _cd, 'recipe': _recipe})
            niah_contrasts = {}
            niah_acc_contrasts = {}
            _nidx = {}
            for c in niah_cells:
                _nidx.setdefault((c['variant'], c['arm'], c['eval_len'], c['condition']), []).append(c)

            def _ncell(v, arm, Ln, cond):
                hits = _nidx.get((v, arm, Ln, cond), [])
                return hits[0] if len(hits) == 1 else {}

            def _n_pairable(a, b, Ln, cond, tag):
                if a.get('val_fp') != b.get('val_fp'):
                    print(f"[v10 stats] NIAH-nat L{Ln} {cond} {tag}: the two arms were measured on DIFFERENT validation slices ({a.get('val_fp')} vs {b.get('val_fp')}) — contrast omitted")
                    return False
                if (_fp_norm(a.get('probe_params')) != _fp_norm(b.get('probe_params'))
                        or a.get('_code') != b.get('_code')
                        or _recipe_norm(a.get('recipe')) != _recipe_norm(b.get('recipe'))):
                    print(f'[v10 stats] NIAH-nat L{Ln} {cond} {tag}: probe/code/recipe identity differs — contrast omitted')
                    return False
                return True
            for Ln in PROBE['eval_lens']:
                for cond in (PROBE.get('niah') or {}).get('conditions', ['clean', 'distract']):
                    learned = _ncell('csa_fixed_rope', 'learned', Ln, cond)
                    for arm, tag in (('dense', 'full_rope'), ('randidx', 'csa_fixed_rope'), ('allblocks', 'csa_fixed_rope')):
                        other = _ncell('full_rope', 'dense', Ln, cond) if arm == 'dense' else _ncell(tag, arm, Ln, cond)
                        if not learned or not other:
                            continue
                        if not _n_pairable(learned, other, Ln, cond, arm):
                            continue
                        common = sorted(set(learned['acc_by_seed']) & set(other['acc_by_seed']))
                        if len(common) >= 2:
                            res = V.exact_sign_permutation([other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common])
                            res['seeds'] = common
                            _attach_niah_factor_metadata(res, arm, _niah_factor_rep)
                            niah_contrasts[f'NIAH-nat L{Ln} {cond}: {arm} - learned (answer NLL)'] = res
                            # Accuracy is the declared primary endpoint for
                            # retrieval.  Testing only the secondary NLL would
                            # leave the stated goal untested and could report a
                            # significant NLL difference while the top-1
                            # accuracy is indistinguishable.
                            _ra = V.exact_sign_permutation([other['acc_by_seed'][s] - learned['acc_by_seed'][s] for s in common])
                            _ra['seeds'] = common
                            _attach_niah_factor_metadata(_ra, arm, _niah_factor_rep)
                            niah_acc_contrasts[f'NIAH-nat L{Ln} {cond}: {arm} - learned (answer accuracy)'] = _ra
                    for tag, ov in (('csa_randtrain', 'csa_fixed_rope_randidx'), ('csa_alltrain', 'csa_fixed_rope_allblocks')):
                        other = _ncell(ov, 'learned', Ln, cond)
                        if not learned or not other:
                            continue
                        if not _n_pairable(learned, other, Ln, cond, tag):
                            continue
                        common = sorted(set(learned['nll_by_seed']) & set(other['nll_by_seed']))
                        if len(common) >= 2:
                            res = V.exact_sign_permutation([other['nll_by_seed'][s] - learned['nll_by_seed'][s] for s in common])
                            res['seeds'] = common
                            _attach_niah_factor_metadata(res, tag, _niah_factor_rep)
                            niah_contrasts[f'NIAH-nat L{Ln} {cond}: {tag} - learned (answer NLL)'] = res
                            _ra = V.exact_sign_permutation([other['acc_by_seed'][s] - learned['acc_by_seed'][s] for s in common])
                            _ra['seeds'] = common
                            _attach_niah_factor_metadata(_ra, tag, _niah_factor_rep)
                            niah_acc_contrasts[f'NIAH-nat L{Ln} {cond}: {tag} - learned (answer accuracy)'] = _ra
            probe['niah_natural'] = {'cells': niah_cells, 'contrasts': niah_contrasts,
                                     'accuracy_contrasts': niah_acc_contrasts,
                                     'hypothesis_audit': L.hypothesis_audit(
                                         L.NIAH_HYPOTHESIS_DESIGN, 'niah')}
            if probe['niah_natural']['hypothesis_audit'].get('status') != 'ok':
                print('[stats] WARNING: NIAH hypotheses are incomplete or metric-misaligned: '
                      + '; '.join((probe['niah_natural']['hypothesis_audit'].get('warnings') or [])[:3]))
            # NIAH contrasts are built AFTER the corruption family was
            # corrected.  Leaving them out of the Holm family would report
            # uncorrected p-values under a "Holm" heading, so re-run the
            # correction over the union and overwrite both.
            _comb = dict(contrasts)
            _comb.update(niah_contrasts)
            _comb.update(niah_acc_contrasts)
            if _comb:
                _h2 = L.holm_adjust(_holm_family(_comb))
                for _k, _v in _comb.items():
                    if isinstance(_v, dict):
                        _e2 = _h2.get(_k)
                        _v['p_holm'] = (_e2.get('p_holm') if isinstance(_e2, dict) else None)
                holm = _h2
                probe['holm_exploratory'] = _h2
                probe['holm_family_note'] = ('Holm over the combined corruption + NIAH contrast families')
                print(f'[v10 stats] Holm re-attached over the combined {len(_comb)}-contrast family (corruption + NIAH)')
    xo = _scale_crossovers()

    def _is_recon(v):
        flags = v.get('per_seed_synth', {})
        return any((flags.get(str(s)) for s in v.get('seeds', [])))
    fit_entries = {}
    for _lab, _v in xo.items():
        if not isinstance(_v, dict):
            continue
        fit_entries[_lab] = {'label': _lab, 'd': _v.get('d'),
                             'crossover_tokens': _v.get('mean_crossover_tokens'),
                             'train_tokens': _v.get('train_tokens'),
                             'censored': bool(_v.get('censored') or _v.get('mean_crossed') is False),
                             'reconstructed': bool(_is_recon(_v))}
    # One point per width, no censored subset, comparable
    # training volume, at least 4 widths, and a slope WITH CI.  The helper
    # refuses instead of fitting whenever those conditions fail.
    fit = L.crossover_scaling_fit(fit_entries, min_points=4)
    for _why in (fit.get('reasons') or []):
        print(f'[stats] crossover fit refused: {_why}')
    if fit.get('deduped_by_d'):
        print(f'[stats] crossover fit: duplicate widths dropped (token-matched panel kept): {fit["deduped_by_d"]}')
    seq2k = _ppl_by_seed('results_lm_v9_seq2k', full=True)
    comparisons = {}

    def add(name, panel, a, b, outdir):
        V9.add_paired(comparisons, name, panel, a, b, who='v10 ', outdir=outdir,
                      factors=V9.SEQ2K_CONTRAST_FACTORS.get(name))
    add('seq2k(bs1): topk8 - m1', seq2k, 'csa_fixed_topk8', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk512 - m1', seq2k, 'csa_fixed_topk512', 'csa_fix_m1', 'results_lm_v9_seq2k')
    add('seq2k(bs1): topk8 - topk512', seq2k, 'csa_fixed_topk8', 'csa_fixed_topk512', 'results_lm_v9_seq2k')
    panel = V8._panel_block
    scale_conf = _equal_token_slices()
    out_d = {'probe': probe, 'crossover_panels': xo, 'crossover_fit': fit, 'scale_confounds': scale_conf, 'seq2k_bs1_panel': panel(seq2k), 'comparisons': comparisons, 'holm': L.attach_holm(comparisons, family='v10 seq2k comparisons')}
    out_d = L.stamped_analysis('v10_analysis', out_d, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
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

def build_report(out='REPORT_v10.md', stats_p='analysis_v10/stats.json'):
    v10_analysis(out=stats_p)
    st = json.load(open(stats_p, encoding='utf-8'))
    probe = st['probe']
    xo = st['crossover_panels']
    fit = st['crossover_fit']
    seq2k = st['seq2k_bs1_panel']
    comp = st['comparisons']
    mech_sum = {}
    mp = os.path.join(P3MT_PAYLOAD['outdir'], 'summary.json')
    if os.path.exists(mp):
        mech_sum = json.load(open(mp, encoding='utf-8'))
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v10 补实验报告（外推机制 + 反转点标度 + topk 补种子）')
    A('> **证据等级：探索性（exploratory）**。本面板的比较/种子是观察 validation 结果后选择的；Holm 校正只限制过度宣称，不等于预注册确认。')
    A('')
    A('')
    A(f'> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}')
    A('> 参考论文：arXiv:2606.19348。**范围声明**：本报告是小规模 WikiText-103 上的受控研究，不是 DeepSeek-V4 的全量复现，也不支持对其规模、语料或生产系统做外推。')
    A('> 说明：本报告全部数字由 `explib.corruption.py report` 从 `results_*/`、`analysis_v10/` 的落盘产物计算得到，无手填数值。')
    A('')

    A('---')
    A('')
    A('## P3M 长度外推优势的机制验证（远距干扰注入）')
    A('')
    A('**背景**：v7 P1L 的外推对比只是附带观察（n=3 时精确符号翻转 p 值下限 0.250）。P1L 权重未持久化，P3MT 先按 v7 P1L 原配方（seq 512、3000 步、3 seeds、AdamW lr 3e-4）重训两臂并保存权重——同时作为该外推对比的独立复现。')
    A('')
    if mech_sum:
        _rows_by_v = {}
        _stale_vars = set()
        for v in list(P3MT_PAYLOAD['variants']) + SCRATCH_CONTROLS:
            _rv = [r for r in mech_sum.values() if isinstance(r, dict) and r.get('variant') == v and (r.get('seed') is not None) and ('by_len' in r)]
            _cur = [r for r in _rv if r.get('_code') == L.code_stamp()]
            if _cur:
                if len(_cur) < len(_rv):
                    print(f'[v10 report] {v}: {len(_rv) - len(_cur)} by-length record(s) predate the current code semantics — the table uses only the {len(_cur)} current one(s)')
                _rv = _cur
            elif _rv:
                # Fail closed: a panel with no record under
                # the current code/recipe stamp is excluded from the table and
                # from every conclusion, not quoted with a warning marker.
                print(f'[v10 report] {v}: no by-length record carries the current code semantics ({L.code_stamp()}) — EXCLUDED from the table and conclusions ({len(_rv)} stale record(s) left on disk)')
                _stale_vars.add(v)
                _rv = []
            _rows_by_v[v] = _rv
        n_by_v = {v: {r['seed'] for r in _rv} for v, _rv in _rows_by_v.items() if _rv}
        if len(n_by_v) > 1:
            _common_seeds = set.intersection(*n_by_v.values())
            if len({frozenset(s) for s in n_by_v.values()}) > 1:
                if _common_seeds:
                    print(f'[v10 report] by-length panel is ragged ({ {v: sorted(s) for v, s in n_by_v.items()} }) — restricting the table to the {len(_common_seeds)}-seed intersection {sorted(_common_seeds)} so every column shares one seed set')
                    for _v in list(_rows_by_v):
                        _rows_by_v[_v] = [r for r in _rows_by_v[_v] if r['seed'] in _common_seeds]
                    n_by_v = {v: {r['seed'] for r in _rv} for v, _rv in _rows_by_v.items() if _rv}
                else:
                    print(f'[v10 report] WARNING: by-length arms share NO seed ({ {v: sorted(s) for v, s in n_by_v.items()} }) — the table below quotes per-arm means over DIFFERENT seed sets and is not a paired comparison')
        A(f'**P3MT 复现对照**（train@512 → 同权重 eval；ratio = PPL@L / PPL@512，seed 平均，n={(min((len(s) for s in n_by_v.values())) if n_by_v else 0)}）：')
        A('')
        A('| variant | PPL@512 | PPL@2048 | PPL@4096 | ratio@4096 |')
        A('|---|---|---|---|---|')
        for v in P3MT_PAYLOAD['variants']:
            rows = _rows_by_v.get(v, [])
            if not rows:
                continue

            def _at(Ln, rows=rows):
                cells = L.by_len_cells(rows, Ln)
                ok = [c for c in cells if isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated'))]
                if (ok if ok else cells) and len(ok if ok else cells) < len(rows):
                    print(f'[v10 report] {v} L{Ln}: the column mean covers {len(ok if ok else cells)}/{len(rows)} seeds — the rest hold no usable cell at this length (error/truncated cell), so the column is a sub-seed mean')
                if ok:
                    return (float(np.mean([c['ppl'] for c in ok])), False)
                return (float(np.mean([c['ppl'] for c in cells])), True) if cells else (float('nan'), False)

            def _paired_ratio(rows=rows):
                m512, m4096 = ({}, {})
                for r in rows:
                    bl = r.get('by_len')
                    if not isinstance(bl, dict):
                        continue
                    for Ln, mp in ((512, m512), (4096, m4096)):
                        c = bl.get(Ln, bl.get(str(Ln)))
                        if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')):
                            mp[r.get('seed')] = float(c['ppl'])
                both = sorted(set(m512) & set(m4096))
                if both and len(both) < len(rows):
                    print(f'[v10 report] {v} ratio@4096: computed over {len(both)}/{len(rows)} seeds — the rest lack a usable cell at 512 or 4096')
                if not both:
                    return None
                return float(np.mean([m4096[s] / m512[s] for s in both]))

            def _fmt_cell(Ln):
                val, tr = _at(Ln)
                return f'{val:.1f}~' if tr else f'{val:.1f}'
            p512, _512tr = _at(512)
            p4096, _4096tr = _at(4096)
            if _512tr or _4096tr:
                _ratio = '—（含截断 cell，不可比）'
            else:
                _pr = _paired_ratio()
                _ratio = f'×{_pr:.2f}' if _pr is not None and np.isfinite(_pr) else '—'
            A(f'| `{v}` | {_fmt_cell(512)} | {_fmt_cell(2048)} | {_fmt_cell(4096)} | {_ratio} |')
        A('')
        A('> 后缀 `~` 表示该列**只有位置受限（abs-PE）的截断 cell**：它是在比列名更短的 span 上评出来的分数，**不是**长上下文测量值。任何 `~` 行不可用于「外推是否稳健」的结论。')
        A('')
        if _stale_vars:
            A('> **⚠ 陈旧记录已排除**：' + '、'.join(f'`{x}`' for x in sorted(_stale_vars)) +
              f' 没有任何按长度记录携带当前代码语义戳 `{L.code_stamp()}`；按「旧代码/旧配方/旧数据指纹不得进入主要结论」的规则，'
              '本表不引用这些数值，也不据此计算 ratio 或外推结论。重跑对应训练阶段后再纳入。')
            A('')
    A(f'**P3MP 干扰注入设计**：eval 长度 2048/4096；把**远距上下文**（目标区之前的全部输入位置，即除最后 {PROBE['target']} 个干净目标 token 及其紧邻前置位置以外的位置）中比例为 ρ ∈ {{0, 1/8, 1/4, 1/2}} 的 token 替换为均匀随机 token——同一 (序列, ρ) 的腐坏对所有臂逐字节相同（按 (eval_len, 序列号, ρ) 播种，与臂/模型种子无关），臂间严格配对。只在干净目标区计 PPL。四个臂：')
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
        arm_order = [('full_rope', 'dense'), ('csa_fixed_rope', 'learned'), ('csa_fixed_rope', 'randidx'), ('csa_fixed_rope', 'allblocks'), ('csa_fixed_rope_randidx', 'learned'), ('csa_fixed_rope_allblocks', 'learned')]
        for Ln in lens:
            _n_ln = max([c.get('n', 0) for c in cells if c['eval_len'] == Ln] or [0])
            A(f'**目标区 PPL（eval_len={Ln}，至多 {_n_ln or "?"} seeds 平均；uniform 腐坏模式）**：')
            A('')
            A('| 臂 | ' + ' | '.join((f'ρ={r:g}' for r in rhos)) + ' |')
            A('|---|' + '---|' * len(rhos))
            for v, arm in arm_order:
                row = []
                for rho in rhos:
                    m = [c['mean'] for c in cells if (c['variant'], c['arm'], c['eval_len'], c['rho'], c.get('mode', 'uniform')) == (v, arm, Ln, rho, 'uniform')]
                    row.append(f'{m[0]:.2f}' if len(m) == 1 else '—')
                    if len(m) > 1:
                        print(f'[report] {v}+{arm} L={Ln} rho={rho}: {len(m)} parameterisations of the cell exist — the table omits the cell (the paired test refuses it too)')
                A(f'| `{v}`+{arm} | ' + ' | '.join(row) + ' |')
            A('')
    contr = probe.get('contrasts', {})
    _legacy_keys = [k for k in contr if ' - csa+learned' in k and any((t in k for t in ('full_rope(dense)', 'csa+randidx', 'csa+allblocks'))) and not k.startswith('ΔNLL')]
    _legacy = {k: contr[k] for k in _legacy_keys}
    if _legacy:
        A('**配对检验（exact sign-flip，Δ = 对照臂 − 学习选择臂，>0 表示学习选择更抗噪；uniform 模式）：**')
        A('')
        A('| 比较 | n | Δ mean±std | p (exact) | p_Holm |')
        A('|---|---|---|---|---|')
        for k, v in _legacy.items():
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} | {v.get('p_holm') if v.get('p_holm') is not None else '—'} |')
        A('')

        def _rng(tag):
            _lo, _hi, _nonf = L.finite_range(((k, v.get('mean')) for k, v in _legacy.items() if tag in k))
            if _nonf:
                print(f'[v10 report] WARNING: {len(_nonf)} contrast(s) in {tag!r} carry a non-finite mean and are excluded from the reported range: {sorted(_nonf)[:3]}')
            return None if _lo is None else (_lo, _hi)

        def _fmt(r, nd=1):
            return 'n/a' if not r else f'{r[0]:+.{nd}f}~{r[1]:+.{nd}f}'

        def _all_sig(tag, alpha=0.05):
            vs = [(k, v) for k, v in _legacy.items() if tag in k]
            if not vs:
                return False
            # Direction, raw p, Holm p, n floor and |mean|>std must ALL hold:
            # the raw-p version could call a resolution-limited or noise-level
            # contrast "strong evidence" while the table printed p_holm=1.0.
            return all((v.get('mean', 0) > 0 and
                        L.reportable_claim(v, family_size=len(_legacy), alpha=alpha,
                                           min_n=6, require_holm=True)['allowed'])
                       for _k, v in vs)
        dn, ab, ri = (_rng('dense'), _rng('allblocks'), _rng('randidx'))
        ri_sig = _all_sig('randidx')
        _n_by_c = [(k, v.get('n', 0)) for k, v in _legacy.items() if 'omitted' not in v]
        n_min = min((n for _k, n in _n_by_c), default=0)
        _bind = [k for k, n in _n_by_c if n == n_min]
        n_min_ri = min((n for k, n in _n_by_c if 'randidx' in k), default=0)
        p_floor = L.signflip_p_floor(n_min) if n_min else float('nan')
        learned_helps = ri_sig
        _dpos = bool(dn) and dn[0] > 0 and bool(ab) and ab[0] > 0
        _dtxt = '正值表示「看得见远距噪声就会受害」，支持限制远距注意力暴露带来抗噪性' if _dpos else '区间并非全正，「远距暴露致害」的方向在本轮产物上不成立，以配对数字为准'
        A(f'**结论（区间实时算自上方配对检验，Δ = 对照臂 − 学习选择臂）：**(1) dense−learned Δ 区间 {_fmt(dn)} PPL，allblocks−learned Δ 区间 {_fmt(ab)} PPL——{_dtxt}；(2) randidx−learned Δ 区间 {_fmt(ri)} PPL——' + ('全为正值且全部对照达到 p<0.05：学习到的检索对抗噪性有**独立贡献**（强版本成立）。' if learned_helps else f'未能在 p<0.05 上成立（randidx 对照的最小 n={n_min_ri}，全表最小 n={n_min}[{', '.join(sorted(_bind)[:2])}{('…' if len(_bind) > 2 else '')}]，精确符号翻转的 p 下限为 {p_floor:.3f}，' + ('该种子数下无法达到 0.05 显著性' if not (p_floor <= 0.05) else '分辨率已足以达到 0.05，未显著是数据结论而非分辨率上限') + '），「抗噪性来自学习到的检索质量」的强版本**不成立**；现有数据只支持「抗噪性主要来自稀疏归纳偏置本身」这一较弱表述。') + '）')
        A('')
    # ---- P3M-2：从头训练的选择器对照 ----
    _new_c = {k: v for k, v in contr.items() if k not in _legacy and not k.startswith('ΔNLL')}
    if _new_c:
        A('### P3M-2 从头训练的选择器对照（同规模）')
        A('')
        A('**背景**：测试时把已训练模型的选择器替换为随机/全块，只能回答「模型是否依赖原选择器」，不能回答「学习式选择是否优于训练充分的随机选择或全块读取」——后两者的测试时干预改变了模型训练时适应的输入分布，效果下降可能来自失配本身。P3MTC 按与 P3MT **完全相同的配方**（3000 步、同数据、同种子）从头训练两个对照：`csa_fixed_rope_randidx`（训练期选择即随机）与 `csa_fixed_rope_allblocks`（训练期就读全部块；与 `csa_fixed_rope` 逐项同构——同 c_kv 压缩、同 qk_norm、同 RoPE、同 sink/窗口，仅选择方式不同，因此它同时是**逐项对齐的 dense 对照**）。')
        A('')
        A('| 比较 | n | Δ mean±std | d_z | p (exact) | p 下限 |')
        A('|---|---|---|---|---|---|')
        for k, v in sorted(_new_c.items()):
            A(f'| {k} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v.get('cohens_dz', float('nan')):+.2f} | {v['p_exact_signflip']:.3f} | {v.get('p_floor', float('nan')):.3f} |')
        A('')
        A('> `retrain-gap(…)` 行 = 测试时干预 − 同规模从头训练对照（同 L、同 ρ）。对越低越好的损失，正值说明干预比重新训练**更差**，该差距本身**可能包含**训练—测试失配的代价，也混合了优化适应与参数差异——它是「干预与重新训练的效果差」，**不是**已完成的因果失配分解，不得把简单相减解释为「不能归因于失配的部分」。`arch-aligned` 行 = dense − 全块 CSA 孪生（两者仅剩 KV 压缩投影与分块读取方式之差）。')
        A('')
    # ---- ΔNLL：污染增量 ----
    dnll = probe.get('delta_nll') or {}
    if dnll:
        A('### P3M-3 污染增量 ΔNLL（按 NLL 口径，含干净条件基线）')
        A('')
        A('**口径**：ΔNLL = NLL_污染 − NLL_干净（nats/目标 token，逐种子配对）。腐坏模式：`uniform`（均匀随机 token，既有协议）、`natural`（其他真实序列同位置 token）、`mask`（远端整体屏蔽为 `<pad>`）、`delete`（远端整体删除，仅剩目标区）。`mask`/`delete` 回答「干净远端上下文到底带来多少收益」；`natural` 与 `uniform` 之差只回答「真实语料同位置 token 替换是否比均匀随机替换更难过滤」；它不是主题干扰、语义相近干扰、重复文本或对抗干扰，不能据此声称普遍抗噪性。')
        A('')
        A('| 臂 | L | 模式 | ρ | ΔNLL mean±std | n | 干净 NLL |')
        A('|---|---|---|---|---|---|---|')
        _clean_nll_cache = {}
        for k in sorted(dnll):
            e = dnll[k]
            _ck = f"{e['variant']}::{e['arm']}::L{e['eval_len']}"
            if _ck not in _clean_nll_cache:
                _cl = [x for x in dnll.values() if x['variant'] == e['variant'] and x['arm'] == e['arm'] and x['eval_len'] == e['eval_len'] and x['mode'] == 'uniform' and abs(float(x['rho'])) < 1e-12]
                _clean_nll_cache[_ck] = float(np.mean([np.mean(list(x.get('clean_nll_by_seed', {}).values())) for x in _cl])) if _cl else float('nan')
            A(f"| `{e['variant']}::{e['arm']}` | {e['eval_len']} | {e['mode']} | {e['rho']:g} | {e['mean']:+.4f} ± {e.get('std', 0):.4f} | {e['n']} | {_clean_nll_cache[_ck]:.4f} |")
        A('')
        _dnll_c = {k: v for k, v in contr.items() if k.startswith('ΔNLL')}
        if _dnll_c:
            A('| ΔNLL 配对比较（ΔΔNLL = 臂间污染增量之差） | n | Δ mean±std | p (exact) |')
            A('|---|---|---|---|')
            for k, v in sorted(_dnll_c.items()):
                A(f'| {k} | {v['n']} | {v['mean']:+.4f} ± {v.get('std', 0):.4f} | {v['p_exact_signflip']:.3f} |')
            A('')
        A('> **替代解释提示**：某臂 ΔNLL 小也可能因为它本来就很少利用远端上下文（干净 NLL 更高）——读表时必须把 ΔNLL 与干净 NLL 列一起看；`mask`/`delete` 行给出干净远端的收益幅度作为参照系。')
        A('')
    # ---- NIAH-natural：远端必要信息任务 ----
    niah = probe.get('niah_natural') or {}
    if niah.get('cells'):
        A('### P3M-4 远端必要信息任务（NIAH-natural，自然文本 + 自然干扰）')
        A('')
        A('**设计**：序列 = 远端自然文本（内嵌 8 个 key→value 耦合）+ 目标区（自然文本收尾 + 8 个查询 key）。查询 key 的下一个 token 的正确答案只在远端出现——正确预测**必须**利用远距信息；`distract` 条件把 ρ=0.5 的远端草料换成其他真实序列的 token（realistic-token 替换；不是主题/语义干扰），但**绝不腐坏耦合位置**。报告 answer top-1 准确率与 answer NLL（已按各臂 clean/distract 成对）。')
        A('')
        A('| 臂 | L | 条件 | acc (微) | macro-recall | F1 macro | F1 weighted | 均匀 chance | 多数类 floor | answer NLL | n |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for c in sorted(niah['cells'], key=lambda c: (c['eval_len'], c['condition'], c['variant'], c['arm'])):
            _bal = (f"{c['balanced_mean']:.3f}" if c.get('balanced_mean') is not None else 'n/a')
            _f1m = (f"{c['f1_macro_mean']:.3f}" if c.get('f1_macro_mean') is not None else 'n/a')
            _f1w = (f"{c['f1_weighted_mean']:.3f}" if c.get('f1_weighted_mean') is not None else 'n/a')
            _ch = (f"{c['chance_accuracy']:.3f}" if isinstance(c.get('chance_accuracy'), (int, float)) else 'n/a')
            _mj = (f"{c['majority_accuracy']:.3f}" if isinstance(c.get('majority_accuracy'), (int, float)) else 'n/a')
            A(f"| `{c['variant']}::{c['arm']}` | {c['eval_len']} | {c['condition']} | {c['acc_mean']:.3f} | {_bal} | {_f1m} | {_f1w} | {_ch} | {_mj} | {c['nll_mean']:.3f} | {c['n']} |")
        A('')
        if niah.get('contrasts'):
            A('| NIAH 配对比较（Δ = 对照臂 − 学习选择，answer NLL，>0 = 学习选择更好） | n | Δ mean±std | p (exact) | p (Holm) |')
            A('|---|---|---|---|---|')
            for k, v in sorted(niah['contrasts'].items()):
                _ph = v.get('p_holm')
                _phs = f'{_ph:.3f}' if isinstance(_ph, (int, float)) and math.isfinite(float(_ph)) else 'n/a'
                A(f"| {k} | {v['n']} | {v['mean']:+.3f} ± {v.get('std', 0):.3f} | {v['p_exact_signflip']:.3f} | {_phs} |")
            A('')
        if niah.get('accuracy_contrasts'):
            A('| NIAH 配对比较（Δ = 对照臂 − 学习选择，answer top-1 accuracy，>0 = 对照臂更好） | n | Δ mean±std | p (exact) | p (Holm) |')
            A('|---|---|---|---|---|')
            for k, v in sorted(niah['accuracy_contrasts'].items()):
                _ph = v.get('p_holm')
                _phs = f'{_ph:.3f}' if isinstance(_ph, (int, float)) and math.isfinite(float(_ph)) else 'n/a'
                A(f"| {k} | {v['n']} | {v['mean']:+.3f} ± {v.get('std', 0):.3f} | {v['p_exact_signflip']:.3f} | {_phs} |")
            A('')
        A('> 该任务直接检验「长距离信息**利用**能力」，与上面的干扰鲁棒性互补：鲁棒性高 + 检索利用不差，才支持「稀疏选择在滤噪的同时保留了有用远距信息」；若只有鲁棒性而检索差，则更符合「本来就少用远端」的替代解释。')
        A('')
        A('> **平凡基线**：`均匀 chance` = 1 / 实际出现的答案类别数；`多数类 floor` = 总是预测出现最多的答案 token。`acc (微)` 会被高频答案主导，`macro-recall` 对每个答案类别等权；`F1 macro` 对每个出现过的答案类别等权，`F1 weighted` 按答案支持度加权（单标签 top-1 任务的 `F1 micro` 恒等于 acc，已写在记录里）。读检索收益时必须把这些列一起看；PR-AUC/ROC-AUC 需要每类分数，top-1 记录不伪造。')
        A('')
    A('### 统计口径')
    A('')
    _n_all = sorted({v.get('n', 0) for v in contr.values() if isinstance(v, dict) and 'omitted' not in v})
    A(f'- 每个配对比较都给出逐种子 Δ、效应量 d_z 与该 n 下的双侧精确符号翻转 p 值下限 2/2^n（本表 n 集合 {_n_all}）。')
    A('- 上表全部 p 值按 **Holm step-down** 在本报告的对比族内做了多重校正（`analysis_v10/stats.json` 的 `probe.holm_exploratory`）；该校正族**未预先注册**，因此校正后的 p 只能防止过度宣称，不能当作确认性证据。')
    A('- v7→v11 的多轮追加实验与既有面板属于**探索性**证据；确认性检验见 `explib.confirmation.py`（冻结清单 `confirmation_frozen.json`：固定主比较、终点、种子数，用 wiki.test.raw 独立 test split 一次性评估）。')
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
            _tok = f'> {last * v['tokens_per_step'] / 1000000.0:.1f}M' if v.get('tokens_per_step') else '(tokens/step unknown)'
            A(f'| d={v['d']}/{v['n_layers']}L | `{v['outdir']}` | {_ntag} | > {last:g} | {_tok} | 删失（窗内未交叉） |')
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
    _xo256_recon = any(((_xo256.get('per_seed_synth') or {}).get(str(s)) for s in _xo256.get('seeds', [])))
    if _xo256_recon:
        A(f'**注意**：d=256 的轨迹取自 `results_lm_v3_long/summary.json`，其中部分记录是从旧聚合表重构的副本（`synthesized`），逐种子轨迹不可独立恢复——该规模的逐种子交叉步互为副本，只有 seed 均值轨迹的交叉步（{_xo256_txt}，eval 网格 1000 步）是有效读数。')
    elif _xo256.get('status') == 'ok':
        A(f'**注意**：d=256 面板（`results_lm_v3_long`）当前不含重构记录，逐种子交叉步（{_xo256_txt} 附近）与 seed 均值轨迹同为有效读数。')
    A('')
    if fit:
        if fit.get('status') != 'ok' or 'slope' not in fit:
            _why = '; '.join(fit.get('reasons') or [fit.get('note', 'no slope')])
            A(f'**趋势拟合：被拒绝**（status={fit.get("status")}）。{_why}。因此本版**不报告** log-log 斜率或幂律。')
            A('')
        else:
            _ci = fit.get('ci')
            _citxt = f'，95% CI [{_ci[0]:.2f}, {_ci[1]:.2f}]' if isinstance(_ci, (list, tuple)) and len(_ci) == 2 else ''
            _ptxt = f'，p={fit["p"]:.3f}' if isinstance(fit.get('p'), (int, float)) else ''
            _r2txt = f'，R²={fit["r2"]:.2f}' if isinstance(fit.get('r2'), (int, float)) else ''
            _lootxt = ''
            _loo = fit.get('leave_one_out_slope_range')
            if isinstance(_loo, (list, tuple)) and len(_loo) == 2:
                _lootxt = f'；留一斜率范围 [{_loo[0]:.2f}, {_loo[1]:.2f}]'
            _vol = fit.get('volume_ratio')
            A(f'**趋势拟合（描述性，非幂律主张）**：{fit.get("n_points")} 个去重后的规模点，'
              f'log10(交叉 token) ~ log10(d) 斜率 {fit["slope"]:.2f}{_citxt}{_ptxt}{_r2txt}{_lootxt}；'
              f'训练量比 {_vol:.2f}（要求 ≤1.25）。这是观测性回归，规模面板未随机化，不能写成 scaling law。')
            A('')
    sc = st.get('scale_confounds') or {}
    if sc:
        A('### P3S-2 规模混杂控制（等训练 token / 6ND 代理规模 / 规模内平台期）')
        A('')
        A('**背景**：P3SS（d=128/4L，8000 步 ≈ 49.15M token）与 P3SL（d=512/10L，4000 步 ≈ 24.58M token）的累计 token 相差一倍——跨规模终点差异不能单独归因于规模。本节把同一批曲线在**相同累计 token**处切片配对（gap_L − gap_S，逐种子），并给出等 token（`P3SL_TOK`，8000 步）与 6ND 参数量代理规模（`P3SL_FLOP`，按参数量比例缩步数——**代理规模，非实测等 FLOPs**）两个大模型臂，以及各面板的规模内平台期诊断（平台期 ≠ 收敛，只支持「规模内平台期后」读数）。')
        A('')
        _panels = sc.get('panels') or {}
        if _panels:
            A('| 面板 | 规模 | 末步 token | 末点 gap (csa−full) mean±std | n | 末点 p (exact) | 平台化种子 |')
            A('|---|---|---|---|---|---|---|')
            for name in ('P3SS', 'P3SL', 'P3SL_TOK', 'P3SL_FLOP'):
                e = _panels.get(name) or {}
                if not e or e.get('status') != 'ok':
                    A(f'| {name} | — | — | 数据缺失 | 0 | — | — |')
                    continue
                _ep = e.get('endpoint') or {}
                _tok = e.get('tokens_per_step')
                _ntok = (f"{(_ep.get('tokens') or 0) / 1000000.0:.2f}M" if _ep and _ep.get('tokens') else 'n/a')
                _conv = e.get('convergence') or {}
                _pl = [f"{_v.get('n_plateaued', 0)}/{_v.get('n', 0)}" for _v in _conv.values()]
                A(f"| {name} | {e.get('note', '')} | {_ntok} | {_ep.get('mean', float('nan')):+.2f} ± {_ep.get('std', 0):.2f} | {_ep.get('n', 0)} | {_ep.get('p_exact_signflip', float('nan')):.3f} | {'/'.join(_pl) or '—'} |")
            A('')
            _conv_warn = [name for name, e in _panels.items() if e.get('status') == 'ok' and any(((e.get('convergence') or {}).get(v) or {}).get('n_plateaued', 0) == 0 for v in ('csa_fixed', 'full'))]
            if _conv_warn:
                A(f'> **规模内平台期证据不足**：{_conv_warn} 面板在最后四分之一 eval 窗口的相对改进 ≥1%（未平台化）——「平台期后交叉」的主张在该面板上**不成立**，只有「窗口内交叉」的读数有效。')
                A('')
            A('> 即使全部种子平台化，也只允许「规模内平台期后交叉」的读数；「渐近优势反转」「收敛后更差」需要更长训练、学习率续训与重复验证，本面板不提供。')
            A('')
        _slices = sc.get('equal_token_slices') or {}
        if _slices:
            A('**等 token 切片（gap_L − gap_S，同累计 token 处逐种子配对；>0 = 大模型在同 token 处 CSA−dense 差距更大）：**')
            A('')
            A('| 切片 | n | Δ mean±std | d_z | p (exact) |')
            A('|---|---|---|---|---|')
            for k, v in sorted(_slices.items()):
                A(f'| {k} | {v.get('n', 0)} | {v.get('mean', float('nan')):+.2f} ± {v.get('std', 0):.2f} | {v.get('cohens_dz', float('nan')):+.2f} | {v.get('p_exact_signflip', float('nan')):.3f} |')
            A('')
            A('> `P3SL_FLOP` 的步数 = 8000 × N_small/N_large（参数量按面板两臂平均，CPU 上确定性计数；下限 250），即**参数量与步数的 6ND 代理对齐**——它不是实测 FLOPs、GPU 时间或能耗；该臂的 gap 只能读作「该代理近似相等时」的规模效应。')
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
    A('| 比较 | 因素 | n | Δ (a−b) mean±std | p (exact) |')
    A('|---|---|---|---|---|')
    for k, v in comp.items():
        if k.startswith('seq2k'):
            _fac = ('topk 剂量响应' if v.get('single_factor') else
                    '块几何束（多因素）' if 'single_factor' in v else '—')
            if 'omitted' in v:
                A(f'| {k} | {_fac} | 0 | —（{v['omitted']}） | — |')
                continue
            A(f'| {k} | {_fac} | {v['n']} | {v['mean']:+.2f} ± {v.get('std', 0):.2f} | {v['p_exact_signflip']:.3f} |')
    A('')
    A('**因素列（单因素契约）**：`topk 剂量响应` = 只差 `index_topk`；`块几何束（多因素）` = m1 对照同时改 block_size(4→1)/overlap(128→0) 与 topk 语义（逐 token 读取），等 token 预算但多因素——只能读作 sweep 点之间的系统级比较（v9 `SEQ2K_CONTRAST_FACTORS` 同一声明）。')
    A('')
    A('**注**：n=4 时精确符号翻转的 p 值分辨率为 0.125——仍跨不过 0.05（需 n=6 才有 0.031）；此行补齐的是与 scale 面板同级的统计分辨率，显著性主张仍以效应量 + 方向同向性表述。')
    A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A('| `results_v10_mech/` | P3M：两臂重训权重（ckpt/，*.pt 不入库）+ by-length 复现表 + 干扰注入探针 `distractor.json`（uniform/natural/mask/delete + span/doc/topical 模式；natural = realistic-token，§15 修订）+ NIAH-natural `niah_natural.json` |')
    A('| `results_lm_v10_scale_s/` | P3S：d=128/4L 交叉定位面板（8000 步） |')
    A('| `results_lm_v10_scale_l/` | P3S：d=512/10L 交叉定位面板（4000 步） |')
    A('| `results_lm_v10_scale_l8k/` | P3S-2：等 token 大模型臂（8000 步） |')
    A('| `results_lm_v10_scale_lflop/` | P3S-2：6ND 参数量代理规模大模型臂（非实测等 FLOPs） |')
    A('| `results_lm_v9_seq2k/` | P3T：topk 扫描追加 seeds 2/3 后的 4-seed 面板 |')
    A('| `analysis_v10/stats.json` | 探针对比（含 ΔNLL、Holm）、交叉定位、等 token 切片与配对符号翻转检验 |')
    A('| `explib.corruption.py` | 本阶段驱动（smoke/phase/analysis/report，可断点续跑） |')
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
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v10] instance shuts down in {delay_s}s.')

def run_full():
    guard = make_guard()
    guard.report()
    print(f'[v10] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state['booked_seconds'] / 3600:.2f} h)')
    git_push('v10: supplementary driver (P3M distractor-injection mechanism + P3S crossover scaling + P3T topk seeds 2/3 + scratch selector controls, corruption modes/NIAH, token/compute-matched scale arms)')
    import par
    _ctx = par.resolve_ctx(label='v10')
    all_ok = True
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v10] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            break
        if not V.cuda_healthy():
            print(f'[v10] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v10 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v10: phase {pname} results')
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v10: stats + REPORT_v10.md (analysis_v10)')
    guard.report()
    print('\n[v10] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) tiny lenphase ckpts + extended distractor probe roundtrip (all modes + NIAH + a scratch control)')
    L.set_seed(0)
    payload = dict(P3MT_PAYLOAD, outdir='results_smoke_v10', seeds=[0], steps=40, eval_lens=[512, 1024])
    import shutil
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope', 'csa_fixed_rope_randidx']), guard=None, label='v10 smoke', par=False)
    pcfg = dict(PROBE, outdir='results_smoke_v10', ckpt_dir='results_smoke_v10/ckpt', summary='results_smoke_v10/distractor.json', variants=['csa_fixed_rope', 'csa_fixed_rope_randidx'], seeds=[0], eval_lens=[1024], rhos=[0.0, 0.5], modes={'uniform': [0.0, 0.5], 'natural': [0.5], 'mask': [1.0], 'delete': [1.0]}, n_seq=4, chunk=2, niah=dict(NIAH_NATURAL, summary='results_smoke_v10/niah_natural.json', n_pairs=2, n_seq=4, conditions=['clean', 'distract']), arm_overrides={'csa_fixed_rope_randidx': ['learned']})
    s = run_probe(pcfg, guard=None, label='v10 smoke', par=False)
    got = {(r.get('mode', 'uniform'), r['arm'], r['rho']) for r in s.values()}
    _want = {('uniform', 'learned', 0.0), ('uniform', 'learned', 0.5), ('uniform', 'randidx', 0.0), ('uniform', 'randidx', 0.5), ('uniform', 'allblocks', 0.0), ('uniform', 'allblocks', 0.5), ('natural', 'learned', 0.5), ('natural', 'randidx', 0.5), ('natural', 'allblocks', 0.5), ('mask', 'learned', 1.0), ('mask', 'randidx', 1.0), ('mask', 'allblocks', 1.0), ('delete', 'learned', 1.0), ('delete', 'randidx', 1.0), ('delete', 'allblocks', 1.0)}
    assert _want <= got, f'probe cells missing: {_want - got} (got {sorted(got)})'
    niah_sum = {}
    _nsp = 'results_smoke_v10/niah_natural.json'
    if os.path.exists(_nsp):
        niah_sum = json.load(open(_nsp, encoding='utf-8'))
    _niah_got = {(r.get('arm'), r.get('condition')) for r in niah_sum.values() if isinstance(r, dict)}
    _niah_want = {('learned', 'clean'), ('learned', 'distract'), ('randidx', 'clean'), ('randidx', 'distract'), ('allblocks', 'clean'), ('allblocks', 'distract')}
    assert _niah_want <= _niah_got, f'NIAH cells missing: {_niah_want - _niah_got} (got {sorted(_niah_got)})'
    print(f'  probe roundtrip OK ({len(s)} corruption cells + {len(niah_sum)} NIAH cells)')
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    print('[smoke] 2) forward/backward at the two new scales + the scratch-control rope variants')
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
    for v in SCRATCH_CONTROLS:
        cfgs = L.make_layer_cfgs(6, v)
        m = L.SmallGPT(8192, 256, 6, 8, 32, 512, cfgs).to(DEVICE)
        x = torch.randint(0, 8192, (2, 512), device=DEVICE)
        out = m(x)
        loss = F.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
        loss.backward()
        print(f'  d=256 L=6 {v:32s} out={tuple(out.shape)} loss={loss.item():.3f} params={L.count_params(m) / 1000000.0:.2f}M')
        del m, out, loss, x
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    print('[smoke] 2b) flop-matched steps are deterministic')
    _fs1 = _flop_matched_steps(P3SS_CFG, P3SL_CFG)
    _fs2 = _flop_matched_steps(P3SS_CFG, P3SL_CFG)
    assert _fs1 == _fs2 and _fs1 >= 250, f'flop-matched steps not deterministic: {_fs1} vs {_fs2}'
    print(f'  flop-matched steps OK ({_fs1})')
    print('[smoke] 3) zero-GPU analysis/report on current artifacts (scratch outputs, real artifacts untouched)')
    build_report(out='results_smoke_v10/REPORT_v10.md', stats_p='results_smoke_v10/stats.json')
    shutil.rmtree('results_smoke_v10', ignore_errors=True)
    print('\n[smoke] PASSED')
def _run_cli():
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
        print(f"[v10] mode 'phase' needs a phase name, e.g. `python explib.corruption.py phase P3MT`.  Available: {[p[0] for p in PHASES]}")
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
