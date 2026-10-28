
def metric_policy_audit(summary=None, policy=None):
    """Check the declared metric contract and the records that claim to meet it.

    The audit never selects a metric for the experimenter.  It checks that each
    task's primary metric, direction, aggregation and imbalance policy are
    declared; that every F1 endpoint names its averaging rule (macro, micro,
    weighted or binary-positive); and that a panel which reports boundary,
    retrieval or classification results also reports the companion metrics
    needed to interpret them (precision/recall plus random baseline for
    boundaries; answer NLL for retrieval; PR-AUC alongside ROC-AUC for rare
    positives)."""
    policy = METRIC_POLICY if policy is None else policy
    out = {'status': 'ok', 'policies': {}, 'checks': [], 'warnings': []}
    if summary is not None and not isinstance(summary, dict):
        # A panel summary is a mapping of record ids; a scalar/list cannot be
        # audited record-by-record.  Keep the policy contract audit and warn
        # instead of crashing on `.values()`.
        out['warnings'].append(
            f'summary is a {type(summary).__name__}, not an object; '
            'the record-level metric contract cannot be audited')
        summary = None
    if not isinstance(policy, dict):
        # A policy is a declared mapping.  Iterating a scalar/list would crash
        # .items() instead of reporting that the contract is unusable.
        out['warnings'].append(
            f'metric policy is a {type(policy).__name__}, not an object; '
            'the metric contract cannot be audited')
        policy = {}
    for name, p in policy.items():
        if not isinstance(p, dict):
            # One malformed entry must not abort the audit of the rest.
            out['policies'][str(name)] = {
                'primary': None, 'primary_average': None, 'direction': None,
                'aggregation': None, 'imbalance': None, 'complete': False,
                'invalid': f'policy entry is {type(p).__name__}, not an object'}
            out['warnings'].append(
                f'metric policy `{name}` is a {type(p).__name__}, not an object')
            continue
        missing = [k for k in ('task', 'primary', 'direction', 'aggregation', 'imbalance') if not p.get(k)]
        primary = str(p.get('primary') or '')
        if primary.startswith('f1') and not p.get('primary_average'):
            missing.append('primary_average')
        out['policies'][name] = {'primary': p.get('primary'),
                                 'primary_average': p.get('primary_average'),
                                 'direction': p.get('direction'),
                                 'aggregation': p.get('aggregation'), 'imbalance': p.get('imbalance'),
                                 'complete': not missing}
        if missing:
            out['warnings'].append(f'metric policy `{name}` is missing {missing}')

    def _policy_entry(task):
        # Records carry the task name; a policy may be keyed by that name or
        # by a role key (e.g. ``binary_classification`` vs ``niah_retrieval``)
        # whose entry declares a different ``task``.  Use the CALLER's policy,
        # not the module default, so a supplied contract actually governs the
        # record checks.  Malformed entries are treated as absent (the top
        # loop already warned about them).
        _e = policy.get(task)
        if isinstance(_e, dict):
            return _e
        for _v in policy.values():
            if isinstance(_v, dict) and _v.get('task') == task:
                return _v
        return {}

    if summary:
        recs = [r for r in summary.values() if isinstance(r, dict)]
        lm = [r for r in recs if ppl_is_usable(r.get('ppl'))]
        # Token-weighted PPL is the declared estimand.  When a record exposes
        # the NLL accounting, verify that its PPL really is exp(sum NLL / sum
        # tokens) rather than an arithmetic mean of per-sequence PPLs.
        _lm_bad = []
        _lm_checked = 0
        for _r in lm:
            _nested = (_r.get('ppl_estimation_rows') == 'nested_validation_holdout')
            # Never mix the full-validation totals with a holdout primary
            # estimate.  If a nested record lacks either holdout total, use
            # its holdout per-row arrays; only a record with no holdout
            # accounting at all is treated as unverifiable here.
            if _nested:
                _nll = _finite_scalar(_r.get('nll_sum_holdout'))
                _ntok = _exact_count(_r.get('n_tokens_holdout'), minimum=1)
                if _nll is None or _ntok is None:
                    _nll = _ntok = None
            else:
                _nll = _finite_scalar(_r.get('nll_sum'))
                _ntok = _exact_count(_r.get('n_tokens'), minimum=1)
            if _nll is None or _ntok is None:
                _arrays = _primary_row_arrays(_r)
                _rn, _rt = (_arrays if isinstance(_arrays, tuple)
                            and len(_arrays) == 2 else (None, None))
                if isinstance(_rn, (list, tuple)) and isinstance(_rt, (list, tuple)) and _rn and len(_rn) == len(_rt):
                    _vals = [_finite_scalar(x) for x in _rn]
                    _toks = [_exact_count(x, minimum=0) for x in _rt]
                    if all(v is not None for v in _vals) and all(t is not None for t in _toks) and sum(_toks) > 0:
                        _nll, _ntok = float(sum(_vals)), int(sum(_toks))
            if _nll is None or _ntok is None:
                continue
            _lm_checked += 1
            _want = _safe_exp_ppl(_nll / _ntok)
            if not math.isfinite(float(_want)):
                _lm_bad.append(_r.get('variant') or _r.get('name') or '?')
                continue
            # Compare against the PPL on the SAME rows as the accounting
            # used above.  A nested record's `ppl` is full validation while
            # `nll_sum_holdout` describes the later row fold; validating one
            # against the other would flag a correct record (or, worse, let a
            # mismatched pair pass when the two errors happen to cancel).
            _got = _record_primary_ppl(_r)
            if _got is None or abs(_got - float(_want)) > 1e-6 * max(1.0, abs(float(_want))):
                _lm_bad.append(_r.get('variant') or _r.get('name') or '?')
        _lm_missing = max(0, len(lm) - _lm_checked)
        out['checks'].append({'metric': 'ppl', 'policy': 'lm_perplexity', 'records': len(lm),
                              'aggregation': 'token_weighted',
                              'records_with_nll_accounting': _lm_checked,
                              'records_without_nll_accounting': _lm_missing,
                              'aggregation_mismatch': sorted(set(map(str, _lm_bad)))[:8],
                              'ok': bool(lm) and not _lm_bad and _lm_missing == 0})
        if _lm_bad:
            out['warnings'].append(
                'LM record(s) ' + ', '.join(sorted(set(map(str, _lm_bad)))[:5])
                + ' report PPL that is not exp(nll_sum/n_tokens) — the declared token-weighted '
                'PPL estimand is not what the record measured')
        if _lm_missing:
            out['warnings'].append(
                f'{_lm_missing}/{len(lm)} LM record(s) expose no NLL/token accounting; '
                'their PPL cannot be verified as the declared token-weighted estimand '
                '(legacy records may be unverifiable rather than wrong)')
        bnd = []
        niah = []
        for r in recs:
            if r.get('task') == 'niah_natural':
                niah.append(r)
            for layer in (r.get('stats') or {}).values() if isinstance(r.get('stats'), dict) else []:
                if isinstance(layer, dict) and 'bnd_f1' in layer:
                    bnd.append(layer)
        if bnd:
            missing = [k for k in ('bnd_prec', 'bnd_rec', 'bnd_f1', 'bnd_rand')
                       if not all(k in b and b.get(k) is not None for b in bnd)]
            _bad_bnd = []
            _macro_unverifiable = []
            for _b in bnd:
                _p, _r, _f = (_finite_scalar(_b.get(k)) for k in ('bnd_prec', 'bnd_rec', 'bnd_f1'))
                _vals = [_p, _r, _f]
                # A present-but-None key is not a measurement.  The old
                # `if v is not None` filter removed exactly those values and
                # let an all-None boundary record pass as `ok`.
                if any(v is None or not (0.0 <= v <= 1.0) for v in _vals):
                    _bad_bnd.append('out-of-range')
                    continue
                _avg = _b.get('bnd_f1_average')
                if _avg is not None and _avg not in ('macro_over_sequences', 'macro', 'micro'):
                    _bad_bnd.append('unknown-f1-averaging')
                _have_micro = any(_b.get(k) is not None for k in
                                  ('bnd_f1_micro', 'bnd_tp', 'bnd_fp', 'bnd_fn'))
                if _have_micro:
                    _pm = _finite_scalar(_b.get('bnd_prec_micro'))
                    _rm = _finite_scalar(_b.get('bnd_rec_micro'))
                    _fm = _finite_scalar(_b.get('bnd_f1_micro'))
                    if _pm is None or _rm is None or _fm is None:
                        _bad_bnd.append('incomplete-micro-counts')
                    elif not (0.0 <= _pm <= 1.0 and 0.0 <= _rm <= 1.0 and 0.0 <= _fm <= 1.0):
                        _bad_bnd.append('micro-out-of-range')
                    elif abs(_fm - 2 * _pm * _rm / max(_pm + _rm, 1e-09)) > 1e-6:
                        _bad_bnd.append('micro-f1-not-harmonic')
                    _tp = _exact_count(_b.get('bnd_tp'), minimum=0)
                    _fp = _exact_count(_b.get('bnd_fp'), minimum=0)
                    _fn = _exact_count(_b.get('bnd_fn'), minimum=0)
                    if None not in (_tp, _fp, _fn):
                        _pm2 = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
                        _rm2 = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
                        if _pm is not None and abs(_pm - _pm2) > 1e-6:
                            _bad_bnd.append('micro-precision-inconsistent')
                        if _rm is not None and abs(_rm - _rm2) > 1e-6:
                            _bad_bnd.append('micro-recall-inconsistent')
                elif _avg in (None, 'macro_over_sequences', 'macro'):
                    _want = 2 * _p * _r / max(_p + _r, 1e-09)
                    if abs(_f - _want) > 1e-6 and _avg is None:
                        _bad_bnd.append('f1-not-harmonic')
                    if _avg is not None:
                        _macro_unverifiable.append(
                            'macro F1 is a mean of per-sequence F1s and cannot be reconstructed '
                            'from mean precision/recall; record bnd_f1_micro or per-sequence values')
                if _avg == 'micro' and abs(_f - 2 * _p * _r / max(_p + _r, 1e-09)) > 1e-6:
                    _bad_bnd.append('micro-declared-f1-not-harmonic')
                _rb = _finite_scalar(_b.get('bnd_rand'))
                if _rb is None or not (0.0 <= _rb <= 1.0):
                    _bad_bnd.append('random-out-of-range')
                _rk = _b.get('bnd_rand_kind')
                if _rk is not None:
                    try:
                        normalize_bnd_rand_kind(_rk)
                    except (TypeError, ValueError):
                        _bad_bnd.append('unknown-bnd-rand-kind')
            out['checks'].append({'metric': 'boundary_alignment', 'policy': 'boundary_alignment',
                                  'records': len(bnd), 'ok': not missing and not _bad_bnd,
                                  'missing': missing, 'invalid': sorted(set(_bad_bnd)),
                                  'f1_average': 'macro_over_sequences',
                                  'macro_f1_reconstructible': not _macro_unverifiable})
            if missing:
                out['warnings'].append(f'boundary metrics report F1 without {missing}; F1 alone cannot be interpreted against the random-cut baseline')
            if _bad_bnd:
                out['warnings'].append('boundary metrics contain invalid/out-of-range or non-harmonic '
                                       f'F1 values ({sorted(set(_bad_bnd))}); the reported F1 is not the '
                                       'declared micro precision/recall endpoint')
            if _macro_unverifiable:
                out['warnings'].append('boundary metrics declare macro-over-sequences F1 but expose no '
                                       'micro counts; the harmonic relation with mean precision/recall '
                                       'does not hold for macro averages, so F1 reproducibility is '
                                       'unverifiable from this record')
        if niah:
            missing = [k for k in ('acc', 'nll') if not all(k in b for b in niah)]
            _bad_niah = []
            _legacy_niah = []
            _legacy_f1 = []
            _f1_unverifiable = []
            for _b in niah:
                _a = _finite_scalar(_b.get('acc'))
                _nl = _finite_scalar(_b.get('nll'))
                _nq = _exact_count(_b.get('n_queries', _b.get('n')), minimum=1)
                if _a is None or not (0.0 <= _a <= 1.0):
                    _bad_niah.append('acc-out-of-range')
                if _nl is None:
                    _bad_niah.append('nll-not-finite')
                if _nq is None:
                    _bad_niah.append('query-count-missing')
                # A top-1 count makes the rate reproducible; a rate alone is
                # not evidence that the record scored n_correct queries.
                _nc = _exact_count(_b.get('n_correct'), minimum=0)
                _has_floor = any(k in _b for k in
                                 ('n_correct', 'chance_accuracy', 'majority_accuracy',
                                  'balanced_accuracy', 'answer_support'))
                # A missing/non-numeric accuracy is already reported above.
                # It must not be fed to float() here (which aborted the whole
                # audit instead of marking the single record unusable).
                if _a is not None and _nc is not None and _nq is not None:
                    if abs(float(_a) - (_nc / _nq)) > 1e-9:
                        _bad_niah.append('acc-count-inconsistent')
                elif _nc is None and _has_floor:
                    _bad_niah.append('correct-count-missing')
                # Recompute the trivial floors from the recorded answer support
                # whenever possible.  A raw accuracy without the uniform and
                # majority-answer floors cannot distinguish retrieval skill
                # from class-frequency luck.
                _supports = None
                _support_map = None
                if isinstance(_b.get('answer_support'), dict):
                    _raw_sup = list(_b['answer_support'].values())
                    _raw_sup_names = list(_b['answer_support'].keys())
                elif isinstance(_b.get('answer_support'), (list, tuple)):
                    _raw_sup = list(_b['answer_support'])
                    _raw_sup_names = None
                else:
                    _raw_sup = None
                    _raw_sup_names = None
                    if 'answer_support' in _b:
                        # A present-but-malformed support table is not the same
                        # as a legacy record that omits it: the record claims to
                        # expose the floor evidence, so it must be parseable.
                        _bad_niah.append('answer-support-invalid')
                if _raw_sup is not None:
                    _parsed_sup = [_exact_count(v, minimum=0) for v in _raw_sup]
                    if any(v is None for v in _parsed_sup):
                        _bad_niah.append('answer-support-invalid')
                    else:
                        _supports = _parsed_sup
                        if _raw_sup_names is not None:
                            _support_map = dict(zip(_raw_sup_names, _parsed_sup))
                if (_supports is not None and _nq is not None
                        and int(sum(_supports)) != int(_nq)):
                    # Per-class supports are the denominator of recall, the
                    # chance floor and the majority floor.  A support table
                    # that does not add up to the scored query count describes
                    # a different sample than `n_queries`.
                    _bad_niah.append('answer-support-sum-mismatch')
                _n_classes = _exact_count(_b.get('n_classes'), minimum=1)
                _observed_classes = (sum(1 for v in _supports if v > 0)
                                     if _supports is not None else None)
                if (_n_classes is not None and _observed_classes is not None
                        and _n_classes != _observed_classes):
                    _bad_niah.append('class-count-inconsistent')
                if _has_floor:
                    _ch = _finite_scalar(_b.get('chance_accuracy'))
                    _mj = _finite_scalar(_b.get('majority_accuracy'))
                    _ba = _finite_scalar(_b.get('balanced_accuracy'))
                    if _ch is None or not (0.0 < _ch <= 1.0):
                        _bad_niah.append('chance-floor-missing')
                    else:
                        _kc = _observed_classes if _observed_classes else _n_classes
                        if _kc:
                            if abs(_ch - 1.0 / _kc) > 1e-9:
                                _bad_niah.append('chance-floor-not-uniform')
                    if _mj is None or not (0.0 <= _mj <= 1.0):
                        _bad_niah.append('majority-floor-missing')
                    elif _supports and _nq is not None:
                        # An empty support table has no majority class; it is
                        # already flagged by the support-sum check and must not
                        # crash max() here.
                        if abs(_mj - max(_supports) / _nq) > 1e-9:
                            _bad_niah.append('majority-floor-inconsistent')
                    if _ba is None or not (0.0 <= _ba <= 1.0):
                        _bad_niah.append('macro-recall-missing')
                    _pcr = _b.get('per_class_recall')
                    if 'per_class_recall' in _b and not isinstance(_pcr, dict):
                        _bad_niah.append('per-class-recall-invalid')
                    if isinstance(_pcr, dict):
                        _rec_vals = [_finite_scalar(v) for v in _pcr.values()
                                     if v is not None]
                        if any(v is None or not (0.0 <= v <= 1.0) for v in _rec_vals):
                            _bad_niah.append('per-class-recall-invalid')
                        elif _rec_vals and _ba is not None:
                            if abs(_ba - float(np.mean(_rec_vals))) > 1e-9:
                                _bad_niah.append('macro-recall-inconsistent')
                        if _support_map is not None:
                            if set(_pcr) != set(_support_map):
                                _bad_niah.append('per-class-keys-disagree')
                            else:
                                _correct_sum = 0.0
                                _counts_ok = True
                                for _name, _rv in _pcr.items():
                                    _sp = _support_map.get(_name)
                                    if _rv is None:
                                        if _sp is not None and _sp > 0:
                                            # `retrieval_metrics` only leaves a
                                            # recall null for an ABSENT class.
                                            # Null for a class that was scored
                                            # drops it from macro recall and the
                                            # correct-count reconciliation.
                                            _bad_niah.append('per-class-recall-missing')
                                            _counts_ok = False
                                            break
                                        continue
                                    _rf = _finite_scalar(_rv)
                                    if _rf is None or _sp is None:
                                        _counts_ok = False
                                        break
                                    try:
                                        _cf = _rf * _sp
                                    except OverflowError:
                                        # A count above the float range is not
                                        # representable, but it must fail the
                                        # record closed instead of aborting
                                        # the audit.
                                        _counts_ok = False
                                        break
                                    _cr = int(round(_cf))
                                    if (abs(_cf - _cr) > 1e-7
                                            or _cr < 0 or _cr > _sp):
                                        _counts_ok = False
                                        break
                                    _correct_sum += _cr
                                if not _counts_ok:
                                    _bad_niah.append('per-class-recall-count-mismatch')
                                elif (_nc is not None
                                      and int(round(_correct_sum)) != int(_nc)):
                                    _bad_niah.append('per-class-correct-sum-mismatch')
                    # Multiclass F1 must name its averaging rule.  A top-1
                    # retrieval record is single-label, so micro F1 is exactly
                    # accuracy; macro and weighted F1 are recomputed from the
                    # per-class TP/FP/FN counts when those are present.
                    _f1_keys = ('f1_macro', 'f1_micro', 'f1_weighted')
                    _f1_present = [k for k in _f1_keys if k in _b]
                    if not _f1_present:
                        _legacy_f1.append(_b.get('variant') or _b.get('name') or '?')
                    elif len(_f1_present) != len(_f1_keys):
                        _bad_niah.append('f1-aggregate-missing')
                    else:
                        _f1_vals = [_finite_scalar(_b.get(k)) for k in _f1_keys]
                        if any(v is None or not (0.0 <= v <= 1.0) for v in _f1_vals):
                            _bad_niah.append('f1-aggregate-invalid')
                        elif _a is not None and abs(_f1_vals[1] - _a) > 1e-9:
                            _bad_niah.append('f1-micro-not-accuracy')
                        else:
                            _pc_f1 = _b.get('per_class')
                            _f1_rows = {}
                            _f1_recalc = isinstance(_pc_f1, dict) and bool(_pc_f1)
                            if _f1_recalc:
                                for _name, _row in _pc_f1.items():
                                    if not isinstance(_row, dict):
                                        _f1_recalc = False
                                        break
                                    _sp = _exact_count(_row.get('support'), minimum=0)
                                    _pr = _exact_count(_row.get('predicted'), minimum=0)
                                    _co = _exact_count(_row.get('correct'), minimum=0)
                                    if None in (_sp, _pr, _co):
                                        _f1_recalc = False
                                        break
                                    _f1_rows[str(_name)] = (_sp, _pr, _co)
                            if not _f1_recalc:
                                _f1_unverifiable.append(
                                    _b.get('variant') or _b.get('name') or '?')
                            else:
                                _sup_tot = int(sum(v[0] for v in _f1_rows.values()))
                                _pred_tot = int(sum(v[1] for v in _f1_rows.values()))
                                # Predictions of classes that never occur in the
                                # answer support are legitimate (a top-1 model
                                # may emit any token), so per-class predicted
                                # counts need not sum to n_queries.  When the
                                # record exposes that count, reconcile it;
                                # otherwise only require no over-count.
                                _n_und = _exact_count(
                                    _b.get('n_undeclared_predictions'), minimum=0)
                                _den_bad = False
                                if _nq is not None and _sup_tot != int(_nq):
                                    _den_bad = True
                                elif _pred_tot > (_sup_tot if _nq is None else int(_nq)):
                                    _den_bad = True
                                elif (_n_und is not None and _nq is not None
                                      and _pred_tot + _n_und != int(_nq)):
                                    _den_bad = True
                                if _den_bad:
                                    _bad_niah.append('f1-count-denominator-mismatch')
                                else:
                                    _f1_macro_pc = []
                                    _f1_sup_pc = []
                                    for _sp, _pr, _co in _f1_rows.values():
                                        if not _sp:
                                            continue
                                        _prec = (_co / _pr) if _pr else 0.0
                                        _rec = _co / _sp
                                        _fv = (2.0 * _prec * _rec / (_prec + _rec)
                                               if (_prec + _rec) > 0 else 0.0)
                                        _f1_macro_pc.append(_fv)
                                        _f1_sup_pc.append(_sp)
                                    _want_macro = (float(np.mean(_f1_macro_pc))
                                                   if _f1_macro_pc else 0.0)
                                    _want_weighted = (
                                        float(np.average(_f1_macro_pc,
                                                         weights=_f1_sup_pc))
                                        if _f1_macro_pc else 0.0)
                                    # Micro is over the full single-label
                                    # endpoint, including predictions of
                                    # unobserved classes, so it equals accuracy.
                                    _tp = int(sum(v[2] for v in _f1_rows.values()))
                                    _den = int(_sup_tot)
                                    _want_micro = 2.0 * _tp / max(
                                        2 * _tp + (_den - _tp) + (_den - _tp), 1)
                                    if (abs(_f1_vals[0] - _want_macro) > 1e-9
                                            or abs(_f1_vals[1] - _want_micro) > 1e-9
                                            or abs(_f1_vals[2] - _want_weighted) > 1e-9):
                                        _bad_niah.append('f1-aggregate-inconsistent')
                    _lch = _finite_scalar(_b.get('acc_lift_over_chance'))
                    if _lch is not None and _ch is not None and _a is not None:
                        if abs(_lch - (_a - _ch)) > 1e-9:
                            _bad_niah.append('chance-lift-inconsistent')
                    _lmj = _finite_scalar(_b.get('acc_lift_over_majority'))
                    if _lmj is not None and _mj is not None and _a is not None:
                        if abs(_lmj - (_a - _mj)) > 1e-9:
                            _bad_niah.append('majority-lift-inconsistent')
                    # Conditional endpoint: with an answer class rarer than
                    # 1:1000, top-1 accuracy is dominated by the frequent
                    # classes and macro recall becomes the policy primary.
                    # Require the record to declare that switch (and to keep
                    # top-1 when the imbalance is not extreme), instead of
                    # letting a bare top-1 headline stand in for retrieval
                    # skill on a rare answer class.
                    if _supports and _nq:
                        _positive = [v for v in _supports if v > 0]
                        try:
                            _minority = (min(_positive) / float(_nq)
                                         if _positive else None)
                        except (OverflowError, ValueError, TypeError):
                            # An astronomically large support count is valid
                            # exact evidence but not representable as a float
                            # prevalence; the conditional endpoint stays
                            # unverifiable instead of aborting the audit.
                            _minority = None
                        _extreme = bool(_extreme_imbalance(_minority))
                        _want_primary = ('macro_recall' if _extreme
                                         else 'top1_accuracy')
                        _declared_primary = _b.get('recommended_primary_metric')
                        if _declared_primary is None:
                            _bad_niah.append('conditional-primary-missing')
                        elif _declared_primary != _want_primary:
                            _bad_niah.append(
                                'conditional-primary-mismatch')
                        _rec_avail = _b.get('recommended_primary_available')
                        if _rec_avail is not None and not isinstance(
                                _rec_avail, (bool, np.bool_)):
                            _bad_niah.append('conditional-primary-availability-invalid')
                        elif _rec_avail is False:
                            _bad_niah.append('conditional-primary-unavailable')
                    # A deterministic lexical copy rule is the simplest
                    # baseline that can solve a key->value retrieval task.  If
                    # the model does not beat it, the learned indexer has not
                    # demonstrated even the copy operation; surface that next
                    # to the model score instead of letting the uniform floor
                    # stand in for a real baseline.
                    _cr = _b.get('copy_rule_baseline')
                    if _cr is not None:
                        if not isinstance(_cr, dict):
                            _bad_niah.append('copy-rule-invalid')
                        else:
                            _ca = _finite_scalar(_cr.get('acc'))
                            _cn = _exact_count(_cr.get('n'), minimum=1)
                            _cnc = _exact_count(_cr.get('n_correct'), minimum=0)
                            _cmc = _exact_count(_cr.get('n_missing_context'), minimum=0)
                            if _ca is None or not (0.0 <= _ca <= 1.0):
                                _bad_niah.append('copy-rule-acc-invalid')
                            if _cn is None or _cnc is None or _cnc > (_cn or 0):
                                _bad_niah.append('copy-rule-count-invalid')
                            elif (_ca is not None
                                  and abs(_ca - (_cnc / _cn)) > 1e-9):
                                _bad_niah.append('copy-rule-acc-count-inconsistent')
                            if _cmc is None or (_cn is not None and _cmc > _cn):
                                _bad_niah.append('copy-rule-missing-context-invalid')
                            if _cn is not None and _nq is not None and _cn != _nq:
                                _bad_niah.append('copy-rule-sample-mismatch')
                            if (_ca is not None and _a is not None and _a <= _ca
                                    and _cn is not None and _cn >= 2):
                                out['warnings'].append(
                                    f'{_b.get("variant") or _b.get("name") or "?"}: NIAH '
                                    f'top-1 accuracy {_a:.4f} does not beat the nearest-key '
                                    f'copy rule ({_ca:.4f}) — the learned mechanism has not '
                                    'shown a retrieval gain over an untrained lexical baseline')
                else:
                    _legacy_niah.append(_b.get('variant') or _b.get('name') or '?')
            out['checks'].append({'metric': 'niah_retrieval', 'policy': 'niah_retrieval',
                                  'records': len(niah), 'ok': not missing and not _bad_niah,
                                  'missing': missing, 'invalid': sorted(set(_bad_niah)),
                                  'legacy_without_trivial_floor': len(_legacy_niah),
                                  'legacy_without_f1': len(_legacy_f1),
                                  'f1_unverifiable': len(_f1_unverifiable)})
            if missing:
                out['warnings'].append(f'NIAH reports top-1 without {missing}; answer NLL is required alongside retrieval accuracy')
            if _bad_niah:
                out['warnings'].append('NIAH records contain invalid accuracy/NLL/query counts or '
                                       'unverifiable trivial floors '
                                       f'({sorted(set(_bad_niah))}); the retrieval endpoint is not usable')
            if _legacy_niah:
                out['warnings'].append(
                    f'{len(_legacy_niah)}/{len(niah)} NIAH record(s) carry no n_correct/trivial floor; '
                    'their top-1 accuracy cannot be distinguished from uniform/majority guessing '
                    '(legacy records are unverifiable, not a pass)')
            if _legacy_f1:
                out['warnings'].append(
                    f'{len(_legacy_f1)}/{len(niah)} NIAH record(s) carry no explicit '
                    'macro/micro/weighted F1; a multiclass F1 without a named averaging '
                    'rule is unverifiable (legacy records, not a pass)')
            if _f1_unverifiable:
                out['warnings'].append(
                    f'{len(_f1_unverifiable)} NIAH record(s) report F1 but expose no '
                    'per-class support/predicted/correct counts; the macro/micro/weighted '
                    'aggregates cannot be recomputed')
        classify_tasks = ('classification', 'binary_classification', 'multiclass_classification')
        _class_recs, _class_issues = [], []
        _allowed_f1_averages = ('macro', 'micro', 'weighted', 'binary_positive_class',
                                'binary_positive')
        for r in recs:
            task = r.get('task')
            if task not in classify_tasks:
                continue
            _class_recs.append(r)
            _label_raw = r.get('variant') or r.get('name') or '?'
            # A malformed record may carry a list/dict in the label slot.
            # Normalize it so set(...)/sorted(...) cannot raise on unhashable
            # metadata and abort the whole audit.
            _label = _label_raw if isinstance(_label_raw, str) else str(_label_raw)
            # A recorded trivial floor must be the actual most-frequent
            # predictor of this label distribution, and its aggregates must be
            # reproducible.  Without this, a record can claim a straw-man floor
            # that is easier to beat than the majority class.
            _tb = r.get('trivial_baseline')
            if isinstance(_tb, dict):
                _pc_tb = r.get('per_class')
                _labels_tb = r.get('labels')
                if isinstance(_pc_tb, dict) and isinstance(_labels_tb, (list, tuple)):
                    try:
                        _sup_tb = [_exact_count(_pc_tb[str(_l)]['support'], minimum=0)
                                   for _l in _labels_tb]
                    except (KeyError, TypeError, ValueError, OverflowError):
                        _sup_tb = None
                    if _sup_tb is not None and all(s is not None for s in _sup_tb):
                        _want_tb = _majority_class_floor(_labels_tb, _sup_tb)
                        for _k in ('label', 'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'):
                            _got_tb = _tb.get(_k)
                            _exp_tb = _want_tb.get(_k)
                            if _k == 'label':
                                if str(_got_tb) != str(_exp_tb):
                                    _class_issues.append(
                                        f'{_label}: trivial_baseline label {_got_tb!r} is not '
                                        f'the most frequent class {_exp_tb!r}')
                            else:
                                _g = _finite_scalar(_got_tb)
                                if _g is None or _exp_tb is None or abs(_g - float(_exp_tb)) > 1e-9:
                                    _class_issues.append(
                                        f'{_label}: trivial_baseline {_k}={_got_tb!r} is not '
                                        f'reproducible from per_class supports ({_exp_tb!r})')
                    elif _sup_tb is not None:
                        _class_issues.append(
                            f'{_label}: per_class support is not an exact non-negative '
                            'count; the trivial-baseline floor cannot be reproduced')
            # Generic guard: a record that exposes a bare `accuracy` for a
            # classification-style task but no F1 must be treated as incomplete.
            if 'accuracy' in r and 'f1_macro' not in r:
                out['warnings'].append(f"{_label} reports accuracy without F1/averaging for a classification task")
            # F1 without a stated averaging rule is not a reproducible metric.
            f1_keys = [k for k in ('f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'f1_positive') if k in r]
            _f1_avg = r.get('f1_primary_average') or r.get('f1_average')
            if f1_keys and not (r.get('f1_averaging') or _f1_avg):
                out['warnings'].append(f"{_label} reports {f1_keys} without naming macro/micro/weighted/binary averaging")
            if _f1_avg is not None and _f1_avg not in _allowed_f1_averages:
                _class_issues.append(f'{_label}: f1_primary_average={_f1_avg!r} is not one of {_allowed_f1_averages}')
            if task == 'multiclass_classification':
                _missing_f1 = [k for k in ('f1_macro', 'f1_micro', 'f1_weighted')
                               if _finite_scalar(r.get(k)) is None]
                if _missing_f1:
                    _class_issues.append(
                        f'{_label}: multiclass record is missing {_missing_f1}; '
                        'macro, micro and weighted F1 can disagree under class imbalance')
                _gap = _finite_scalar(r.get('macro_micro_f1_gap'))
                _fm = _finite_scalar(r.get('f1_macro'))
                _fi = _finite_scalar(r.get('f1_micro'))
                if (_gap is not None and _fm is not None and _fi is not None
                        and abs(_gap - (_fm - _fi)) > 1e-9):
                    _class_issues.append(
                        f'{_label}: macro_micro_f1_gap={_gap} is not '
                        f'f1_macro - f1_micro ({_fm - _fi})')
                _rank_keys = ('pr_auc_macro', 'pr_auc_micro', 'pr_auc_weighted',
                              'roc_auc_macro', 'roc_auc_micro', 'roc_auc_weighted')
                _have_rank = [k for k in _rank_keys if _finite_scalar(r.get(k)) is not None]
                if _have_rank and not (r.get('ranking_averaging')
                                       or r.get('ranking_average')):
                    out['warnings'].append(
                        f'{_label} reports {_have_rank} without naming the one-vs-rest '
                        'aggregation (macro/micro/weighted)')
                # When per-class ranking values and supports are present, the
                # reported macro/weighted aggregates must be reproducible.
                _pc_r = r.get('per_class')
                if isinstance(_pc_r, dict) and _pc_r:
                    _pa, _ra, _rw, _ww = [], [], [], []
                    for _k, _v in _pc_r.items():
                        if not isinstance(_v, dict):
                            continue
                        _ap = _finite_scalar(_v.get('pr_auc'))
                        _au = _finite_scalar(_v.get('roc_auc'))
                        _sup = _exact_count(_v.get('support'), minimum=0)
                        if _ap is None or _au is None or _sup is None:
                            continue
                        _pa.append(_ap)
                        _ra.append(_au)
                        _rw.append(_sup)
                        _ww.append(_sup)
                    _n_pc = _exact_count(r.get('n'), minimum=1)
                    if _n_pc is None:
                        _n_pc = sum(_rw) if _rw else None
                    if _n_pc:
                        for _k, _v in _pc_r.items():
                            if not isinstance(_v, dict):
                                continue
                            _sup = _exact_count(_v.get('support'), minimum=0)
                            _ap = _finite_scalar(_v.get('pr_auc'))
                            if _sup is None:
                                continue
                            _chance = float(_sup) / float(_n_pc)
                            _got_c = _finite_scalar(_v.get('pr_auc_chance'))
                            if _got_c is not None and abs(_got_c - _chance) > 1e-12:
                                _class_issues.append(
                                    f'{_label}: per-class PR-AUC chance for {_k}={_got_c} '
                                    f'is not its support prevalence ({_chance})')
                            _got_l = _finite_scalar(_v.get('pr_auc_lift_over_chance'))
                            if (_got_l is not None and _ap is not None
                                    and abs(_got_l - (_ap - _chance)) > 1e-12):
                                _class_issues.append(
                                    f'{_label}: per-class PR-AUC lift for {_k}={_got_l} '
                                    f'is not pr_auc - support prevalence ({_ap - _chance})')
                        # Micro PR-AUC pools K one-vs-rest decisions per sample,
                        # so its chance floor is 1/K rather than a class
                        # prevalence.  A missing/wrong floor makes a raw micro
                        # value look more skilful than it is.
                        _labels_r = r.get('labels')
                        _k_classes = (len(_labels_r) if isinstance(_labels_r, (list, tuple))
                                      and _labels_r else len(_pc_r))
                        if _k_classes > 0:
                            _mc = _finite_scalar(r.get('pr_auc_micro_chance'))
                            if (_mc is not None
                                    and abs(_mc - 1.0 / float(_k_classes)) > 1e-12):
                                _class_issues.append(
                                    f'{_label}: pr_auc_micro_chance={_mc} is not '
                                    f'1/{_k_classes} for the pooled one-vs-rest decisions')
                            _ml = _finite_scalar(r.get('pr_auc_micro_lift_over_chance'))
                            _pmic = _finite_scalar(r.get('pr_auc_micro'))
                            if (_ml is not None and _pmic is not None
                                    and abs(_ml - (_pmic - 1.0 / float(_k_classes))) > 1e-9):
                                _class_issues.append(
                                    f'{_label}: pr_auc_micro_lift_over_chance={_ml} is not '
                                    f'pr_auc_micro - 1/{_k_classes}')
                    _pm = _finite_scalar(r.get('pr_auc_macro'))
                    if _pa and _pm is not None and abs(_pm - float(np.mean(_pa))) > 1e-9:
                        _class_issues.append(
                            f'{_label}: pr_auc_macro={_pm} is not the mean of its '
                            f'per-class one-vs-rest PR-AUC values ({float(np.mean(_pa))})')
                    _rm = _finite_scalar(r.get('roc_auc_macro'))
                    if _ra and _rm is not None and abs(_rm - float(np.mean(_ra))) > 1e-9:
                        _class_issues.append(
                            f'{_label}: roc_auc_macro={_rm} is not the mean of its '
                            f'per-class one-vs-rest ROC-AUC values ({float(np.mean(_ra))})')
                    if _pa and sum(_rw) > 0:
                        _pw = float(np.average(_pa, weights=_ww))
                        _rpw = _finite_scalar(r.get('pr_auc_weighted'))
                        if _rpw is not None and abs(_rpw - _pw) > 1e-9:
                            _class_issues.append(
                                f'{_label}: pr_auc_weighted={_rpw} is not the support-weighted '
                                f'mean of its per-class PR-AUC values ({_pw})')
                        _n_tb = _exact_count(r.get('n'), minimum=1)
                        if _n_tb is None:
                            _n_tb = sum(_rw)
                        if _n_tb and _n_tb > 0:
                            _chance_macro = float(np.mean([s / _n_tb for s in _rw]))
                            _chance_weighted = float(
                                sum(w * (s / _n_tb) for w, s in zip(_ww, _rw)) / sum(_ww))
                            for _key, _want in (
                                    ('pr_auc_macro_chance', _chance_macro),
                                    ('pr_auc_weighted_chance', _chance_weighted)):
                                _got_c = _finite_scalar(r.get(_key))
                                if _got_c is not None and abs(_got_c - _want) > 1e-9:
                                    _class_issues.append(
                                        f'{_label}: {_key}={_got_c} is not the one-vs-rest '
                                        f'prevalence floor ({_want})')
                            for _key, _base_key, _chance in (
                                    ('pr_auc_macro_lift_over_chance', 'pr_auc_macro', _chance_macro),
                                    ('pr_auc_weighted_lift_over_chance', 'pr_auc_weighted', _chance_weighted)):
                                _got_l = _finite_scalar(r.get(_key))
                                _base = _finite_scalar(r.get(_base_key))
                                if (_got_l is not None and _base is not None
                                        and abs(_got_l - (_base - _chance)) > 1e-9):
                                    _class_issues.append(
                                        f'{_label}: {_key}={_got_l} is not '
                                        f'{_base_key} - chance ({_base - _chance})')
            # f1_primary must be the field its declared averaging names.
            if 'f1_primary' in r:
                _f1p = _finite_scalar(r.get('f1_primary'))
                _source = None
                if _f1_avg == 'binary_positive_class' or _f1_avg == 'binary_positive':
                    _source = 'f1_positive'
                elif _f1_avg in ('macro', 'micro', 'weighted'):
                    _source = f'f1_{_f1_avg}'
                elif _f1_avg is None:
                    _source = ('f1_positive' if task == 'binary_classification'
                               else ('f1_macro' if task == 'multiclass_classification' else None))
                if _f1p is None:
                    _class_issues.append(f'{_label}: f1_primary is not a finite scalar')
                elif _source is not None and _finite_scalar(r.get(_source)) is not None:
                    if abs(_f1p - float(_finite_scalar(r.get(_source)))) > 1e-9:
                        _class_issues.append(
                            f'{_label}: f1_primary={_f1p} does not equal {_source}='
                            f'{_finite_scalar(r.get(_source))} under averaging {_f1_avg!r}')
            # When per-class evidence is present, the aggregate F1s must be
            # reproducible from it rather than asserted.
            _pc = r.get('per_class')
            if isinstance(_pc, dict) and _pc:
                _f1s, _sup = [], []
                _counts = []
                _pc_bad = False
                for _k, _v in _pc.items():
                    if not isinstance(_v, dict):
                        _pc_bad = True
                        break
                    _f = _finite_scalar(_v.get('f1'))
                    _s = _exact_count(_v.get('support'), minimum=0)
                    if _f is None or not (0.0 <= _f <= 1.0) or _s is None:
                        _pc_bad = True
                        break
                    _f1s.append(float(_f))
                    _sup.append(int(_s))
                    _tp = _exact_count(_v.get('tp'), minimum=0)
                    _fp = _exact_count(_v.get('fp'), minimum=0)
                    _fn = _exact_count(_v.get('fn'), minimum=0)
                    _counts.append(None if None in (_tp, _fp, _fn) else (int(_tp), int(_fp), int(_fn)))
                if _pc_bad:
                    _class_issues.append(f'{_label}: per_class contains an invalid F1/support entry')
                elif _f1s:
                    _macro = float(np.mean(_f1s))
                    _rep_macro = _finite_scalar(r.get('f1_macro'))
                    if _rep_macro is not None and abs(_rep_macro - _macro) > 1e-9:
                        _class_issues.append(
                            f'{_label}: f1_macro={_rep_macro} is not the mean of its per-class F1 values ({_macro})')
                    if sum(_sup) > 0:
                        _weighted = float(np.average(_f1s, weights=_sup))
                        _rep_weighted = _finite_scalar(r.get('f1_weighted'))
                        if _rep_weighted is not None and abs(_rep_weighted - _weighted) > 1e-9:
                            _class_issues.append(
                                f'{_label}: f1_weighted={_rep_weighted} is not the support-weighted '
                                f'mean of its per-class F1 values ({_weighted})')
                    if all(c is not None for c in _counts):
                        _bad_cls = []
                        for (_k, _v), (tp, fp, fn) in zip(_pc.items(), _counts):
                            wp = tp / (tp + fp) if tp + fp else 0.0
                            wr = tp / (tp + fn) if tp + fn else 0.0
                            wf = 2 * wp * wr / (wp + wr) if wp + wr else 0.0
                            for _key, _want in (('precision', wp), ('recall', wr), ('f1', wf)):
                                _got = _finite_scalar(_v.get(_key))
                                if _got is None or abs(_got - _want) > 1e-9:
                                    _bad_cls.append(f'{_k}:{_key}')
                        if _bad_cls:
                            _class_issues.append(
                                f'{_label}: per_class counts do not reproduce '
                                f'{sorted(set(_bad_cls))[:8]}')
                        _stp = sum(c[0] for c in _counts)
                        _sfp = sum(c[1] for c in _counts)
                        _sfn = sum(c[2] for c in _counts)
                        _mp = _stp / (_stp + _sfp) if _stp + _sfp else 0.0
                        _mr = _stp / (_stp + _sfn) if _stp + _sfn else 0.0
                        _mf = 2 * _mp * _mr / (_mp + _mr) if _mp + _mr else 0.0
                        for _key, _want in (('f1_micro', _mf), ('precision_micro', _mp),
                                            ('recall_micro', _mr)):
                            _got = _finite_scalar(r.get(_key))
                            if _got is not None and abs(_got - _want) > 1e-9:
                                _class_issues.append(
                                    f'{_label}: {_key}={_got} is not the value reproduced '
                                    f'from per_class counts ({_want})')
                        _tot = sum(_sup)
                        _acc = _finite_scalar(r.get('accuracy'))
                        if _tot > 0 and _acc is not None and abs(_acc - _stp / _tot) > 1e-9:
                            _class_issues.append(
                                f'{_label}: accuracy={_acc} is not the correct-count fraction '
                                f'from per_class ({_stp / _tot})')
            # Threshold-dependent endpoints are only reproducible with the cut
            # that produced them.  An untuned 0.5 cut is legal, but it must be
            # named; a cut selected on the evaluation rows is not.
            _thr = r.get('decision_threshold')
            _thr_src = r.get('decision_threshold_source')
            _f1ish = any(k in r for k in ('f1', 'f1_positive', 'f1_primary',
                                          'precision_macro', 'recall_macro'))
            if (task == 'binary_classification'
                    and any(k in r for k in ('f1_positive', 'f1_primary'))
                    and r.get('positive_label') is None):
                out['warnings'].append(
                    f'{_label} reports a positive-class F1 without positive_label; '
                    'the reported class is not identifiable')
            if task == 'binary_classification' and _f1ish and _thr_src is None:
                out['warnings'].append(
                    f'{_label} reports threshold-dependent F1/precision/recall without '
                    'decision_threshold_source; report the cut and how it was chosen')
            if _thr is not None or _thr_src is not None:
                _t = _finite_scalar(_thr)
                if _thr is None and _thr_src not in ('caller_predictions', 'argmax'):
                    _class_issues.append(
                        f'{_label}: decision_threshold_source={_thr_src!r} is set but '
                        'decision_threshold is missing')
                elif _thr is not None and _t is None:
                    # The score is not necessarily a probability: logits and
                    # arbitrary ranking scores are supported, so only
                    # finiteness is a contract here.  A cut outside the
                    # observed [score_min, score_max] is caught by
                    # `threshold_within_score_range` below.
                    _class_issues.append(
                        f'{_label}: decision_threshold={_thr!r} is not a finite number')
                if _thr_src is None:
                    _class_issues.append(
                        f'{_label}: decision_threshold is reported without decision_threshold_source')
                elif _thr_src not in _BINARY_THRESHOLD_SOURCES:
                    _class_issues.append(
                        f'{_label}: decision_threshold_source={_thr_src!r} is not one of '
                        f'{list(_BINARY_THRESHOLD_SOURCES)}')
                if _thr_src == 'validation_selection':
                    if r.get('threshold_selection_split') != 'validation':
                        _class_issues.append(
                            f'{_label}: the tuned threshold does not declare '
                            'threshold_selection_split="validation"; selecting it on the '
                            'evaluation/test rows is leakage')
                    if not r.get('threshold_selection_metric'):
                        _class_issues.append(
                            f'{_label}: the tuned threshold does not name its '
                            'threshold_selection_metric')
                if _thr_src == 'default_0.5' and _t is not None and abs(_t - 0.5) > 1e-12:
                    _class_issues.append(
                        f'{_label}: source is default_0.5 but decision_threshold={_t}')
            for _key in ('roc_auc', 'pr_auc', 'brier_score', 'ece',
                         'roc_auc_macro', 'roc_auc_micro', 'roc_auc_weighted',
                         'pr_auc_macro', 'pr_auc_micro', 'pr_auc_weighted',
                         'pr_auc_macro_chance', 'pr_auc_micro_chance',
                         'pr_auc_weighted_chance'):
                if _key in r and r.get(_key) is not None:
                    _v = _finite_scalar(r.get(_key))
                    if _v is None or not (0.0 <= _v <= 1.0):
                        _class_issues.append(
                            f'{_label}: {_key}={r.get(_key)!r} is not a finite value in [0, 1]')
            # A recorded predicted-positive rate must be reproducible from
            # the per-class counts, and a degenerate cut must be visible.
            _ppr = _finite_scalar(r.get('predicted_positive_rate'))
            if _ppr is not None:
                if not (0.0 <= _ppr <= 1.0):
                    _class_issues.append(
                        f'{_label}: predicted_positive_rate={_ppr} is outside [0, 1]')
                _pc_pp = r.get('per_class')
                _pos_nm = r.get('positive_label')
                if (isinstance(_pc_pp, dict) and _pos_nm is not None
                        and isinstance(_pc_pp.get(str(_pos_nm)), dict)):
                    _pe = _pc_pp[str(_pos_nm)]
                    _tp = _exact_count(_pe.get('tp'), minimum=0)
                    _fp = _exact_count(_pe.get('fp'), minimum=0)
                    _n_pp = _exact_count(r.get('n'), minimum=1)
                    if None not in (_tp, _fp, _n_pp) and _n_pp:
                        _want_ppr = float(_tp + _fp) / float(_n_pp)
                        if abs(_ppr - _want_ppr) > 1e-12:
                            _class_issues.append(
                                f'{_label}: predicted_positive_rate={_ppr} is not '
                                f'(tp+fp)/n ({_want_ppr})')
            if r.get('degenerate_predictions') is True:
                # This is a property of the recorded operating point, not a
                # malformed artifact: a majority-class baseline is legitimately
                # degenerate.  Surface it without failing an otherwise
                # reproducible record.
                out.setdefault('degenerate_prediction_records', []).append(_label)
            if r.get('threshold_within_score_range') is False:
                _class_issues.append(
                    f'{_label}: decision_threshold is outside the observed score range '
                    '[score_min, score_max]; thresholded metrics describe a constant '
                    'predictor, not the model ranking')
            _roc = _finite_scalar(r.get('roc_auc'))
            if _roc is not None and _roc < 0.5:
                out['warnings'].append(
                    f'{_label}: ROC-AUC={_roc:.3f} is below chance; check the positive-class '
                    'score orientation and positive_label before interpreting the model as weak')
            # A rare-positive panel that reports only ROC-AUC is exactly the
            # optimistic-score failure mode this policy exists to prevent.
            # For multiclass records the minority class prevalence is the
            # minimum over the full class distribution, not a binary 2-entry
            # dict: a 1:1000 third class must trigger the same warning.
            prevalence = None
            for key in ('prevalence', 'positive_prevalence'):
                val = r.get(key)
                if isinstance(val, dict) and val and (task != 'binary_classification'
                                                      or len(val) == 2):
                    # A malformed prevalence entry is unverifiable, not a
                    # reason to abort the whole audit.  Only finite values in
                    # [0, 1] define a class prevalence.
                    _pv = [_finite_scalar(x) for x in val.values()]
                    if all(v is not None and 0.0 <= v <= 1.0 for v in _pv):
                        prevalence = min(_pv)
                        break
                _prev_f = _finite_scalar(val)
                if _prev_f is not None and 0.0 <= _prev_f <= 1.0:
                    prevalence = (min(_prev_f, 1.0 - _prev_f)
                                  if task == 'binary_classification' else _prev_f)
                    break
            if task == 'binary_classification' and prevalence is None:
                # A record can expose the PR-AUC chance floor without a
                # separate `prevalence` field.  Ignoring that floor would let
                # a rare-positive record evade the conditional PR-AUC check.
                _pr_floor = _finite_scalar(r.get('pr_auc_prevalence'))
                if _pr_floor is not None and 0.0 <= _pr_floor <= 1.0:
                    prevalence = min(_pr_floor, 1.0 - _pr_floor)
            # Conditional-endpoint checks for a binary task follow the POSITIVE
            # class prevalence (the policy's wording and PR-AUC's floor), not
            # the minority rate.  Legacy records without a positive label keep
            # the minority fallback.
            _cond_prev = prevalence
            if task == 'binary_classification' and prevalence is not None:
                # PR-AUC's chance floor is the POSITIVE-class prevalence, which
                # need not be the minority class.  When the positive endpoint
                # is the common label (a rare negative event), `prevalence`
                # above is the minority rate and using it here would flag a
                # correct record as a misstated floor.
                _pos_prev = None
                _pos_nm = r.get('positive_label')
                _pp_raw = r.get('positive_prevalence')
                _pp = _finite_scalar(_pp_raw)
                if _pp is not None and 0.0 <= _pp <= 1.0:
                    _pos_prev = _pp
                elif _pos_nm is not None:
                    _prev_map = r.get('prevalence')
                    if isinstance(_prev_map, dict):
                        _pp = _finite_scalar(_prev_map.get(str(_pos_nm)))
                        if _pp is not None and 0.0 <= _pp <= 1.0:
                            _pos_prev = _pp
                    if _pos_prev is None:
                        _pc_pos = r.get('per_class')
                        if isinstance(_pc_pos, dict):
                            _pe_pos = _pc_pos.get(str(_pos_nm))
                            if isinstance(_pe_pos, dict):
                                _sup_pos = _exact_count(_pe_pos.get('support'), minimum=0)
                                _n_pos_rec = _exact_count(r.get('n'), minimum=1)
                                if _sup_pos is not None and _n_pos_rec:
                                    _pos_prev = float(_sup_pos) / float(_n_pos_rec)
                _floor = _pos_prev if _pos_prev is not None else prevalence
                if _pos_prev is not None:
                    _cond_prev = _pos_prev
                _pr_p = _finite_scalar(r.get('pr_auc_prevalence'))
                if _pr_p is not None and abs(_pr_p - _floor) > 1e-6:
                    # Legacy records may omit positive_label; in that case the
                    # endpoint could be either class, so accept the mirror
                    # orientation rather than inventing an error.
                    _ok_mirror = (_pos_prev is None
                                  and abs(_pr_p - (1.0 - prevalence)) <= 1e-6)
                    if not _ok_mirror:
                        _class_issues.append(
                            f'{_label}: pr_auc_prevalence={_pr_p} differs from the recorded '
                            f'positive prevalence {_floor}; the PR-AUC floor is misstated')
                _pr = _finite_scalar(r.get('pr_auc'))
                _lift = _finite_scalar(r.get('pr_auc_lift_over_prevalence'))
                if _pr is not None and _lift is not None and abs(_lift - (_pr - _floor)) > 1e-6:
                    _mirror = (_pos_prev is None
                               and abs(_lift - (_pr - (1.0 - prevalence))) <= 1e-6)
                    if not _mirror:
                        _class_issues.append(
                            f'{_label}: pr_auc_lift_over_prevalence={_lift} is not '
                            f'pr_auc - positive prevalence ({_pr - _floor})')
                if (_thr_src == 'default_0.5' and prevalence < 1e-2
                        and (_finite_scalar(r.get('f1_primary')) is not None
                             or _finite_scalar(r.get('f1')) is not None)):
                    out['warnings'].append(
                        f'{_label}: F1 uses the default 0.5 cut at prevalence '
                        f'{prevalence:.2e}; report PR-AUC and, if a tuned cut is claimed, '
                        'select it on validation with select_binary_threshold and record the split')
            _declared_primary = r.get('primary_metric') or r.get('metric_primary')
            if _declared_primary is not None:
                _pol = _policy_entry(task) or None
                _allowed_primary = set()
                if _pol:
                    for _k in ('primary', 'primary_when_extreme_imbalance'):
                        _pv = _pol.get(_k)
                        if isinstance(_pv, str):
                            _allowed_primary.add(_pv)
                        elif isinstance(_pv, (list, tuple)):
                            _allowed_primary.update(str(x) for x in _pv)
                if _allowed_primary and (not isinstance(_declared_primary, str)
                                         or _declared_primary not in _allowed_primary):
                    _class_issues.append(
                        f'{_label}: declares primary_metric={_declared_primary!r} for task '
                        f'{task!r}; the policy primary is {sorted(_allowed_primary)}')
            # The prevalence-aware recommendation is itself a claim; verify it
            # against the declared policy instead of trusting the record.
            _rec_metric = r.get('recommended_primary_metric')
            if _rec_metric is not None:
                _pol_rec = _policy_entry(task)
                _rec_allowed = set()
                for _k in ('primary', 'primary_when_extreme_imbalance'):
                    _pv = _pol_rec.get(_k)
                    if isinstance(_pv, str):
                        _rec_allowed.add(_pv)
                    elif isinstance(_pv, (list, tuple)):
                        _rec_allowed.update(str(x) for x in _pv)
                if _rec_allowed and (not isinstance(_rec_metric, str)
                                     or _rec_metric not in _rec_allowed):
                    _class_issues.append(
                        f'{_label}: recommended_primary_metric={_rec_metric!r} is not '
                        f'defined for task {task!r} (allowed {sorted(_rec_allowed)})')
                if _cond_prev is not None:
                    # A malformed policy threshold is a contract fault, not a
                    # reason to abort the whole record audit; fall back to the
                    # documented default and surface the invalidity.
                    _thr_rec = _finite_scalar(
                        _pol_rec.get('extreme_imbalance_threshold', 1e-3))
                    if _thr_rec is None:
                        out['warnings'].append(
                            f'metric policy {task!r} has a non-finite '
                            'extreme_imbalance_threshold; using the documented '
                            '1e-3 default')
                        _thr_rec = 1e-3
                    _conditional = _pol_rec.get('primary_when_extreme_imbalance')
                    if _extreme_imbalance(_cond_prev, _thr_rec) and _conditional:
                        _want_rec = _conditional
                    else:
                        _want_rec = _pol_rec.get('primary')
                    if _rec_metric != _want_rec:
                        _class_issues.append(
                            f'{_label}: recommended_primary_metric={_rec_metric!r} does '
                            f'not match the policy endpoint {_want_rec!r} at prevalence '
                            f'{_cond_prev:.3e}')
            # `recommended_primary_available` is a claim that the recommended
            # endpoint was actually measured.  A record can otherwise name the
            # right metric and still omit its value (e.g. F1 without PR-AUC at
            # 1:1000), which would make the recommendation unusable.
            if 'recommended_primary_available' in r:
                _avail = r.get('recommended_primary_available')
                if not isinstance(_avail, (bool, np.bool_)):
                    _class_issues.append(
                        f'{_label}: recommended_primary_available={_avail!r} is not a boolean')
                elif isinstance(_rec_metric, str):
                    _fields = {
                        'f1': ('f1_positive', 'f1_primary', 'f1'),
                        'f1_macro': ('f1_macro',),
                        'f1_micro': ('f1_micro',),
                        'f1_weighted': ('f1_weighted',),
                        'pr_auc': ('pr_auc',),
                        'pr_auc_macro': ('pr_auc_macro',),
                        'pr_auc_micro': ('pr_auc_micro',),
                        'pr_auc_weighted': ('pr_auc_weighted',),
                    }.get(_rec_metric, (_rec_metric,))
                    _measured = any(_finite_scalar(r.get(k)) is not None
                                    for k in _fields)
                    if bool(_avail) and not _measured:
                        _class_issues.append(
                            f'{_label}: recommended_primary_available=True but the '
                            f'recommended endpoint {_rec_metric!r} has no finite value '
                            f'(looked for {list(_fields)})')
                    elif not bool(_avail) and _measured:
                        _class_issues.append(
                            f'{_label}: recommended_primary_available=False but the '
                            f'recommended endpoint {_rec_metric!r} is recorded')
            if (_extreme_imbalance(_cond_prev)):
                _cond = _policy_entry(task).get(
                    'primary_when_extreme_imbalance',
                    'pr_auc_macro' if task == 'multiclass_classification' else 'pr_auc')
                _is_mc = task == 'multiclass_classification'
                _roc_key = 'roc_auc_macro' if _is_mc else 'roc_auc'
                _pr_key = 'pr_auc_macro' if _is_mc else 'pr_auc'
                if r.get(_roc_key) is not None and r.get(_pr_key) is None:
                    out['warnings'].append(
                        f"{_label} reports {_roc_key} but no {_pr_key} at prevalence "
                        f"{prevalence:.2e} (at or beyond 1:1000); PR-AUC ({_pr_key}) is "
                        'the primary ranking metric here')
                if (not _is_mc and r.get('pr_auc') is not None
                        and r.get('pr_auc_prevalence') is None):
                    out['warnings'].append(f"{_label} reports a rare-positive PR-AUC without its prevalence floor; PR-AUC must be read against chance prevalence")
                if _declared_primary is not None and _declared_primary != _cond:
                    _class_issues.append(
                        f'{_label}: declares primary_metric={_declared_primary!r} at '
                        f'prevalence {prevalence:.2e} (at or beyond 1:1000); the '
                        f'conditional policy primary is {_cond!r}')
                if r.get(_pr_key) is None and (r.get('f1_primary') is not None
                                               or r.get('f1') is not None
                                               or r.get('f1_macro') is not None):
                    _class_issues.append(
                        f'{_label}: reports F1 without {_pr_key} at prevalence '
                        f'{prevalence:.2e} (at or beyond 1:1000); F1 is '
                        'threshold-dependent and cannot be the primary endpoint here')
        if _class_recs:
            out['checks'].append({'metric': 'classification', 'records': len(_class_recs),
                                  'ok': not _class_issues,
                                  'issues': sorted(set(_class_issues))[:12],
                                  'degenerate_prediction_records': sorted(set(
                                      out.get('degenerate_prediction_records') or []))[:12],
                                  'f1_averaging_required': list(_allowed_f1_averages),
                                  'threshold_sources': list(_BINARY_THRESHOLD_SOURCES),
                                  'threshold_default': 0.5,
                                  'extreme_imbalance_primary': 'pr_auc'})
            for _ci in sorted(set(_class_issues)):
                out['warnings'].append('classification metric contract: ' + _ci)
    out['status'] = 'warn' if out['warnings'] else 'ok'
    return out

def tuning_selection_margin(best, runner, *, metric='ppl', direction='lower_is_better'):
    """Margin between the selected config and the runner-up, in standard errors.

    The two trials were evaluated on the SAME search seeds, so their difference
    is paired.  Using the two marginal standard errors (``hypot(s1, s2)``)
    ignores that pairing: it can call a perfectly consistent per-seed gain a
    tie, or (with negatively correlated draws) call a noisy pick decisive.
    When both trial rows expose ``seed_ppls`` we therefore compute the
    per-seed difference, its sample SD and ``SE = SD/sqrt(n_paired)``.  Only
    legacy rows without per-seed values fall back to the independent-SE
    approximation, and the chosen method is returned explicitly.
    """
    out = {'margin': None, 'se': None, 'z': None, 'decisive': None,
           'method': 'unavailable', 'n_paired': 0, 'common_seeds': [],
           'se_independent': None, 'decisive_independent': None}
    if not isinstance(best, dict) or not isinstance(runner, dict):
        return out
    if direction not in ('lower_is_better', 'higher_is_better'):
        return out
    _b = _finite_scalar(best.get(metric))
    _r = _finite_scalar(runner.get(metric))
    if _b is None or _r is None:
        return out
    margin = (_r - _b) if direction == 'lower_is_better' else (_b - _r)
    out['margin'] = float(margin)

    def _seed_map(row):
        vals = row.get('seed_ppls')
        if not isinstance(vals, dict):
            return None, None
        out_map = {}
        for k, v in vals.items():
            seed = _as_seed_int(k)
            f = _finite_scalar(v)
            if seed is None or f is None:
                return None, 'seed_ppls contains non-numeric seeds/values'
            if seed in out_map:
                # Two JSON keys can denote one seed ("0" and 0).  Overwriting
                # would silently turn a repeated draw into one observation and
                # make the paired SE smaller than the evidence supports.
                return None, (f'seed_ppls repeats seed {seed} under two JSON '
                              'spellings; the repeated draw is not one observation')
            out_map[seed] = f
        return out_map, None

    bmap, bprob = _seed_map(best)
    rmap, rprob = _seed_map(runner)
    _paired_insufficient = False
    if bprob or rprob:
        # Malformed per-seed evidence is not the same as legacy evidence that
        # omits it: do NOT fall back to the independent-SE approximation.
        out.update({'method': 'invalid_seed_ppls',
                    'reason': bprob or rprob,
                    'se': None, 'z': None, 'decisive': None})
        return out
    if bmap is not None and rmap is not None:
        common = sorted(set(bmap) & set(rmap))
        if len(common) >= 2:
            diffs = [(rmap[s] - bmap[s]) if direction == 'lower_is_better'
                     else (bmap[s] - rmap[s]) for s in common]
            sd = float(np.std(np.asarray(diffs, dtype=float), ddof=1))
            se = sd / math.sqrt(len(diffs))
            # The paired SE is estimated on the COMMON seeds.  Comparing it
            # with a margin taken from the two rows' full (possibly
            # different-seed) means would put the numerator and denominator on
            # different samples.  Use the paired mean difference instead, and
            # keep the original row-level margin for transparency.
            paired_margin = float(np.mean(np.asarray(diffs, dtype=float)))
            out.update({'margin': paired_margin,
                        'margin_unpaired': float(margin),
                        'se': float(se),
                        'z': (float(paired_margin / se) if se > 0 else None),
                        'decisive': bool(paired_margin > se),
                        'method': 'paired_search_seeds',
                        'n_paired': len(common), 'common_seeds': common})
        else:
            # Both rows DO expose per-seed evidence, but they do not share
            # enough draws for a paired margin.  Falling back to the
            # independent marginal SE would silently change the sample (and
            # could call a decisive pick from the full row means), so leave
            # the margin unverifiable instead.
            out['method'] = 'insufficient_common_seeds'
            out['decisive'] = None
            _paired_insufficient = True
    else:
        out['method'] = 'seed_ppls_unavailable'
    # Independent marginal-SE fallback, also reported for legacy-artifact
    # validation.  It is NOT the preferred estimate when paired values exist.
    try:
        _std_b = _finite_scalar(best.get('ppl_std'))
        _std_r = _finite_scalar(runner.get('ppl_std'))
        if _std_b is not None and _std_r is not None:
            n1 = max(1, int(_exact_count(best.get('n_seeds'), minimum=1)
                            or (len(bmap) if bmap else 1)))
            n2 = max(1, int(_exact_count(runner.get('n_seeds'), minimum=1)
                            or (len(rmap) if rmap else 1)))
            s1 = abs(float(_std_b)) / math.sqrt(n1)
            s2 = abs(float(_std_r)) / math.sqrt(n2)
            se_i = math.hypot(s1, s2)
            out['se_independent'] = float(se_i)
            out['decisive_independent'] = bool(margin > se_i)
            if out['method'] != 'paired_search_seeds' and not _paired_insufficient:
                out.update({'se': float(se_i),
                            'z': (float(margin / se_i) if se_i > 0 else None),
                            'decisive': bool(margin > se_i),
                            'method': 'independent_marginal_se'})
        elif out['method'] != 'paired_search_seeds':
            # Missing/non-finite marginal SD is not zero uncertainty.  Without
            # per-seed evidence and without an SD the margin is unverifiable;
            # never report a decisive pick from absent variance.
            out['se'] = None
            out['z'] = None
            out['decisive'] = None
            if out['method'] == 'unavailable':
                out['method'] = 'seed_ppls_unavailable'
    except (TypeError, ValueError, OverflowError):
        pass
    return out


def tuning_selection_stability(trials, metric='ppl', direction='lower_is_better'):
    """Leave-one-search-seed-out stability of the selected tuning config.

    Selecting the minimum of a mean over three seeds can still be decided by
    one seed.  Re-running the selection with every seed dropped in turn makes
    that visible: if the winner changes, the "tuned" configuration is a
    noise draw, not a stable choice.  Requires trial rows that expose their
    per-seed values; returns status `unavailable` otherwise (never a pass).
    """
    rows = [t for t in (trials or []) if isinstance(t, dict)]
    if not rows:
        return {'status': 'unavailable', 'reason': 'no trial records'}
    seed_maps = []
    acc_maps = []
    for t in rows:
        ppls = t.get('seed_ppls')
        if not isinstance(ppls, dict):
            return {'status': 'unavailable', 'reason': 'trial records do not expose seed_ppls'}
        m = {}
        for k, v in ppls.items():
            seed = _as_seed_int(k)
            if seed is None:
                return {'status': 'unavailable', 'reason': 'seed_ppls contains non-numeric seeds/values'}
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
                return {'status': 'unavailable', 'reason': 'seed_ppls contains non-numeric seeds/values'}
            if seed in m:
                return {'status': 'unavailable',
                        'reason': f'seed_ppls repeats seed {seed} under two JSON spellings; '
                                  'the repeated draw would be silently collapsed'}
            f = _finite_scalar(v)
            # A non-finite draw is not a measurement.  Leaving that seed out
            # of the comparison set keeps every trial's mean over the SAME
            # draws instead of silently averaging different panels.
            if f is not None:
                m[seed] = f
        seed_maps.append(m)
        # Token accounting lets the stability scan re-select with the same
        # token-pooled estimand the driver uses.  It is optional so legacy
        # artifacts with only per-seed PPLs still work via the arithmetic mean.
        _nll = t.get('seed_nll_sum')
        _tok = t.get('seed_n_tokens')
        acc = None
        if isinstance(_nll, dict) and isinstance(_tok, dict):
            acc = {}
            for k, v in _nll.items():
                seed = _as_seed_int(k)
                if seed is None:
                    acc = None
                    break
                nv, tv = _finite_scalar(v), _finite_scalar(_tok.get(k))
                if nv is None or tv is None or tv <= 0 or seed in acc:
                    acc = None
                    break
                acc[seed] = (float(nv), float(tv))
        acc_maps.append(acc)
    common = set(seed_maps[0])
    for m in seed_maps[1:]:
        common &= set(m)
    if len(common) < 3:
        return {'status': 'insufficient_seeds',
                'reason': 'leave-one-out selection needs at least 3 common seeds',
                'common_seeds': sorted(common)}
    _use_acc = all(a is not None for a in acc_maps) and all(
        common <= set(a) for a in acc_maps if a is not None)
    sigs = [json.dumps(t.get('config') or {}, sort_keys=True, default=str) for t in rows]
    choices = []
    for drop in sorted(common):
        vals = []
        for i, (m, sig) in enumerate(zip(seed_maps, sigs)):
            # Use exactly the COMMON seeds for every trial.  A trial with extra
            # non-common draws must not change a configuration's mean (and
            # therefore the winner) when a common seed is dropped.
            keep = [s for s in common if s != drop]
            if not keep:
                return {'status': 'unavailable', 'reason': f'trial has no finite values after dropping seed {drop}'}
            if _use_acc:
                _nll = sum(acc_maps[i][s][0] for s in keep)
                _tok = sum(acc_maps[i][s][1] for s in keep)
                value = float(_safe_exp_ppl(_nll / _tok))
            else:
                value = float(np.mean([m[s] for s in keep]))
            vals.append((sig, value))
        best = (min(vals, key=lambda kv: kv[1]) if direction == 'lower_is_better'
                else max(vals, key=lambda kv: kv[1]))
        choices.append({'dropped_seed': drop, 'config': json.loads(best[0]),
                        'metric': best[1]})
    distinct = {json.dumps(c['config'], sort_keys=True, default=str) for c in choices}
    return {'status': 'ok', 'stable': (len(distinct) == 1),
            'n_distinct_configs': len(distinct), 'n_common_seeds': len(common),
            'common_seeds': sorted(common), 'choices': choices,
            'metric': metric, 'direction': direction,
            'metric_aggregation': ('pooled_token_nll' if _use_acc else 'arithmetic_seed_mean'),
            'note': 'drop one search seed -> re-select; a changed winner means the selected config is seed-driven'}


def _tuning_declared_contrasts(protocol, arms):
    """Declared tuned comparisons and the family they form.

    The power floor must be computed from the comparisons the protocol
    actually declares (each group's ``method_arm`` against the other arms in
    that group), not from ``len(arms) - 1``.  The two differ whenever arms are
    split into parameter-matching groups: the v6/v12 protocol declares four
    tuned contrasts over six arms.  Using the arm count inflated the
    multiplicity family and could make a panel look resolution-limited when
    its declared family was resolvable.

    Returns ``(plan, basis)`` where ``plan`` is a de-duplicated list of
    ``(method, baseline)`` pairs.  A protocol with no usable group declaration
    falls back to the historical all-arms rule so old artifacts without
    groups keep their original planning semantics.
    """
    present = set(arms)
    plan = []
    seen = set()

    def _add(method, baseline):
        if method == baseline:
            return
        key = (method, baseline)
        if key not in seen:
            seen.add(key)
            plan.append(key)

    raw_groups = protocol.get('arm_groups')
    if isinstance(raw_groups, (list, tuple)):
        for group in raw_groups:
            if not isinstance(group, dict):
                continue
            method = group.get('method_arm')
            if method not in present:
                continue
            group_arms = group.get('arms')
            if not isinstance(group_arms, (list, tuple, set)):
                group_arms = ()
            for baseline in group_arms:
                if baseline in present:
                    _add(method, baseline)
    if plan:
        return plan, 'declared_arm_groups'
    method = protocol.get('method_arm') or 'csa_dynamic'
    for baseline in present:
        _add(method, baseline)
    if plan:
        return plan, 'legacy_all_arms'
    return [], 'no_measurable_contrast'


def _plan_seed_list(raw, label, problems):
    """Strict seed partition for a plan (duplicates are a planning error)."""
    if raw is None:
        return []
    if isinstance(raw, (bool, np.bool_)):
        problems.append(f'{label} is a boolean, not a seed list')
        return []
    if isinstance(raw, numbers.Real):
        raw = [raw]
    elif isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = [raw.item()]
    elif isinstance(raw, (str, bytes)):
        raw = (raw.decode() if isinstance(raw, bytes) else raw).replace(',', ' ').split()
    try:
        vals = list(raw)
    except TypeError:
        problems.append(f'{label} is not iterable ({raw!r})')
        return []
    out, seen = [], set()
    for item in vals:
        seed = _as_seed_int(item)
        if seed is None:
            problems.append(f'{label} contains a non-integer seed {item!r}')
            continue
        if seed in seen:
            problems.append(f'{label} contains duplicate seed {seed}')
            continue
        seen.add(seed)
        out.append(seed)
    if not out:
        problems.append(f'{label} is empty; the panel would train an undeclared fallback seed')
    return out
