

def baseline_tuning_audit(selected=None, path=None, protocol=None):
    """Audit the equal-search baseline-tuning evidence.

    A win over an untuned dense baseline establishes only that the proposed
    model beats a straw man.  This audit requires, for every compared arm:
    the *same declared trial grid*, the *same number of trials*, the *same
    selection split* (validation, never test), the *same selection metric*, a
    selected config that really is the best recorded trial, and a
    `selected_metric_value` that is the value of **that** config (not merely
    the minimum of some other trial).  When trial records expose
    steps/tokens_seen, the training allocation must also agree across arms.  When
    they expose `seeds` / `seed_ppls`, the aggregate must be the mean of those
    per-seed values and every arm must select on the same search-seed set —
    otherwise one arm's winner can be a noise draw even though lr grids match.
    The final-seed coverage is checked for equality and reported with the
    family-resolution floor.
    Missing evidence is reported as `not_verifiable_locally`, never as a pass.
    """
    _explicit_protocol = protocol is not None
    if selected is None and path:
        if os.path.exists(path):
            try:
                selected = json.load(open(path, encoding='utf-8'))
            except Exception as e:
                return {'status': 'fail', 'tuning_verified': False, 'problems': [f'cannot parse {path}: {type(e).__name__}: {e}']}
        else:
            return {'status': 'not_verifiable_locally', 'tuning_verified': False,
                    'problems': [f'{path} absent — run baseline_tune.py before claiming a baseline win']}
    # An artifact records the protocol it was produced under.  Use it when the
    # caller did not pass one explicitly, so a smoke/subset run is audited
    # against its own declared grid.
    if not _explicit_protocol and isinstance(selected, dict) and isinstance(selected.get('protocol'), dict):
        protocol = selected['protocol']
    else:
        protocol = BASELINE_TUNING_PROTOCOL if protocol is None else protocol
    if not isinstance(protocol, dict):
        return {'status': 'fail', 'tuning_verified': False,
                'problems': [f'tuning protocol must be an object, got '
                             f'{type(protocol).__name__}']}
    if not isinstance(selected, dict) or not isinstance(selected.get('arms'), dict):
        return {'status': 'not_verifiable_locally', 'tuning_verified': False,
                'problems': ['no baseline_tuning artifact (expected {"arms": {...}})']}
    arms = selected['arms']
    # An explicit (even malformed) artifact field must not fall through to the
    # protocol default: `artifact.get(k) or default` silently rewrote an empty
    # metric/split/direction into the valid declared value, so the malformed
    # artifact passed instead of being reported.
    def _declared_field(key):
        return selected[key] if key in selected else protocol.get(key)
    metric = _declared_field('selection_metric')
    split = _declared_field('selection_split')
    direction = _declared_field('direction')
    problems = []
    warnings = []
    # An undeclared arm is not extra evidence: it enlarges the arm set that
    # the family-size arithmetic and coverage map see, and it can make a
    # truncated tuning artifact look broader than the protocol that was
    # pre-registered.  The artifact must describe exactly the declared arms.
    _declared_arm_list = protocol.get('arms')
    if isinstance(_declared_arm_list, (list, tuple, set)):
        _declared_names = {str(a) for a in _declared_arm_list}
        _extra_artifact_arms = sorted(set(arms) - _declared_names)
        if _extra_artifact_arms:
            problems.append(
                f'artifact contains undeclared arm(s) {_extra_artifact_arms}; '
                'the equal-search claim covers only the protocol\'s declared '
                'arms, so an added arm is a different tuning design')
    if not isinstance(metric, str) or not metric.strip():
        # Trial records are indexed by the metric name; a mapping/list would
        # raise TypeError instead of reporting the malformed artifact.
        problems.append(f'selection metric {metric!r} is not a metric name')
        metric = None
    if split != 'validation':
        problems.append(f'selection split is {split!r}; hyperparameters must be selected on validation, never test')
    _declared_split = protocol.get('selection_split')
    if (isinstance(_declared_split, str) and split != _declared_split):
        problems.append(
            f'selection split {split!r} differs from the declared '
            f'{_declared_split!r}; the audit is not checking the protocol that '
            'was actually pre-registered')
    if metric != protocol.get('selection_metric'):
        problems.append(f'selection metric {metric!r} differs from the declared {protocol.get("selection_metric")!r}')
    if direction not in ('lower_is_better', 'higher_is_better'):
        problems.append(f'unknown selection direction {direction!r}')
    else:
        # The artifact's direction is not free metadata: it decides whether
        # the selected trial must be the argmin or the argmax.  Trusting an
        # artifact that flips it makes a maximum-PPL trial look like the tuned
        # winner even though the protocol declares lower-is-better.
        _declared_direction = protocol.get('direction')
        if (isinstance(_declared_direction, str)
                and direction != _declared_direction):
            problems.append(
                f'selection direction {direction!r} differs from the declared '
                f'{_declared_direction!r} — the artifact selected on a different '
                'ordering than the pre-registered tuning protocol')
        if metric == 'ppl' and direction != 'lower_is_better':
            problems.append(
                f'selection direction {direction!r} is incompatible with PPL '
                '(lower_is_better); a maximum-PPL trial cannot be the tuned winner')
    # Nested-validation evidence: seed-disjoint tuning is not enough if the
    # final estimate is still measured on the documents that selected the
    # config.  Verify the row-fold contract when the artifact claims it.
    try:
        _declared_rows = tuning_row_split_spec(protocol)
    except ValueError as _rse:
        problems.append(f'malformed selection_row_split: {_rse}')
        _declared_rows = None
    _split_info = selected.get('selection_row_split')
    _row_split_applied = None
    _nested_final = None
    if _declared_rows is not None:
        if not isinstance(_split_info, dict):
            warnings.append('the tuning protocol declares a nested validation row split, but the '
                            'artifact records no selection_row_split; selection and final evaluation '
                            'may share validation documents')
        elif _split_info.get('applied') is True:
            _row_split_applied = True
            # With more than two folds only the two selected blocks are used,
            # so `covers_all_rows=False` is the correct metadata (the unused
            # middle/later folds are neither selection nor evaluation rows).
            # Requiring full coverage here would reject a valid n_folds>2
            # chronological split.  Disjointness is required for every layout.
            _meta_folds = _as_seed_int(_split_info.get('n_folds'))
            _needs_full_coverage = _meta_folds is None or _meta_folds <= 2
            if (_split_info.get('disjoint') is not True
                    or (_needs_full_coverage
                        and _split_info.get('covers_all_rows') is not True)):
                problems.append('selection_row_split claims applied=True but does not attest '
                                'disjoint selection/evaluation folds'
                                + (' covering every validation row'
                                   if _needs_full_coverage else ''))
            if not _split_info.get('selection_row_fp') or not _split_info.get('evaluation_row_fp'):
                problems.append('selection_row_split lacks the selection/evaluation row fingerprints')
            elif _split_info.get('selection_row_fp') == _split_info.get('evaluation_row_fp'):
                problems.append('selection and evaluation row fingerprints are identical — the folds are not disjoint')
            # A nested split only prevents selection bias if the final rows
            # come after the selection rows in stream order.  A seeded random
            # row permutation (the legacy policy) is disjoint but interleaves
            # future chunks into selection; an artifact that declares it must
            # not pass the nested-validation gate.
            _policy = _split_info.get('policy')
            if _policy is not None and _policy != 'chronological_contiguous_row_folds':
                problems.append(
                    f'selection_row_split policy is {_policy!r}, not '
                    "'chronological_contiguous_row_folds' — a shuffled row "
                    'split puts future validation chunks into selection')
            elif _policy is None:
                warnings.append('selection_row_split does not declare a fold policy; '
                                'chronological ordering cannot be independently verified')
            if _split_info.get('chronological') is False:
                problems.append('selection_row_split explicitly declares a non-chronological split')
            if _split_info.get('selection_precedes_evaluation') is False:
                problems.append('selection_row_split evaluates earlier rows than it selects on '
                                '(reverse-time nested validation)')
            _sr = _split_info.get('selection_row_range')
            _er = _split_info.get('evaluation_row_range')
            if isinstance(_sr, (list, tuple)) and isinstance(_er, (list, tuple)):
                try:
                    _ok_range = (len(_sr) == 2 and len(_er) == 2
                                 and int(_sr[0]) <= int(_sr[1])
                                 and int(_er[0]) <= int(_er[1])
                                 and int(_sr[1]) < int(_er[0]))
                except (TypeError, ValueError, OverflowError):
                    _ok_range = False
                if not _ok_range:
                    problems.append('selection_row_split row ranges are not ordered '
                                    'selection-before-evaluation')
            # Reproduce the declared folds from (n_rows, seed, n_folds, fold
            # indices).  The checks above only show that the metadata is
            # internally consistent; they cannot detect an edited range or a
            # fingerprint that describes a different split.  When the full
            # fold description is present, recompute it deterministically and
            # compare every recorded field.
            _fold_keys = ('n_rows', 'seed', 'n_folds', 'selection_fold',
                          'evaluation_fold')
            _meta_keys = ('n_selection_rows', 'n_evaluation_rows',
                          'selection_row_range', 'evaluation_row_range',
                          'selection_row_fp', 'evaluation_row_fp',
                          'covers_all_rows', 'disjoint', 'chronological',
                          'selection_precedes_evaluation')
            _fold_present = [k for k in _fold_keys if _split_info.get(k) is not None]
            if len(_fold_present) == len(_fold_keys):
                _re_meta = None
                _groups_rec = _split_info.get('groups')
                _nrows_i = _as_seed_int(_split_info.get('n_rows'))
                if _split_info.get('groups_aware') is True:
                    # Group-aware folds depend on the cached document ids, so
                    # a row-only replay is the WRONG expected split.  Recompute
                    # exactly when the group vector is persisted; otherwise
                    # audit the recorded layout for internal consistency.
                    if (isinstance(_groups_rec, (list, tuple))
                            and _nrows_i is not None
                            and len(_groups_rec) == _nrows_i):
                        try:
                            _re_sel, _re_ev, _re_meta = validation_row_split(
                                _nrows_i, seed=_split_info.get('seed'),
                                n_folds=_split_info.get('n_folds'),
                                selection_fold=_split_info.get('selection_fold'),
                                evaluation_fold=_split_info.get('evaluation_fold'),
                                groups=_groups_rec)
                        except (TypeError, ValueError, OverflowError) as _rfe:
                            problems.append('selection_row_split metadata cannot be '
                                            f'reproduced: {_rfe}')
                    else:
                        _n_folds_i = _as_seed_int(_split_info.get('n_folds'))
                        _sel_fold_i = _as_seed_int(_split_info.get('selection_fold'))
                        _ev_fold_i = _as_seed_int(_split_info.get('evaluation_fold'))
                        _fr = _split_info.get('fold_ranges')
                        _fr_ok = (isinstance(_fr, list) and _n_folds_i is not None
                                  and len(_fr) == _n_folds_i)
                        if _fr_ok:
                            for _r in _fr:
                                if (not isinstance(_r, (list, tuple)) or len(_r) != 2
                                        or _as_seed_int(_r[0]) is None
                                        or _as_seed_int(_r[1]) is None):
                                    _fr_ok = False
                                    break
                        if not _fr_ok:
                            problems.append('group-aware selection_row_split lacks usable '
                                            'fold_ranges; the declared folds cannot be verified')
                        elif _nrows_i is None or _nrows_i < 2:
                            problems.append('group-aware selection_row_split has an invalid n_rows')
                        elif (_sel_fold_i is None or _ev_fold_i is None
                              or not (0 <= _sel_fold_i < _n_folds_i)
                              or not (0 <= _ev_fold_i < _n_folds_i)
                              or _sel_fold_i >= _ev_fold_i):
                            problems.append('group-aware selection_row_split has invalid '
                                            'fold indices (selection must precede evaluation)')
                        else:
                            _ranges = [(int(a), int(b)) for a, b in _fr]
                            _ranges_ok = (_ranges[0][0] == 0
                                          and _ranges[-1][1] == _nrows_i - 1
                                          and all(_ranges[i][1] + 1 == _ranges[i + 1][0]
                                                  for i in range(len(_ranges) - 1))
                                          and all(a <= b for a, b in _ranges))
                            if not _ranges_ok:
                                problems.append('group-aware selection_row_split fold_ranges '
                                                'do not partition [0, n_rows) in order')
                            else:
                                _sel_r, _ev_r = _ranges[_sel_fold_i], _ranges[_ev_fold_i]
                                _sel_idx = np.arange(_sel_r[0], _sel_r[1] + 1)
                                _ev_idx = np.arange(_ev_r[0], _ev_r[1] + 1)
                                _want = {
                                    'n_selection_rows': int(_sel_idx.size),
                                    'n_evaluation_rows': int(_ev_idx.size),
                                    'selection_row_range': [int(_sel_r[0]), int(_sel_r[1])],
                                    'evaluation_row_range': [int(_ev_r[0]), int(_ev_r[1])],
                                    'selection_row_fp': _ids_fp(_sel_idx),
                                    'evaluation_row_fp': _ids_fp(_ev_idx),
                                    'covers_all_rows': bool(int(_sel_idx.size) + int(_ev_idx.size) == _nrows_i),
                                    'disjoint': True,
                                    'chronological': True,
                                    'selection_precedes_evaluation': True,
                                }
                                _re_mismatch = [k for k in _meta_keys
                                                if k in _split_info
                                                and _split_info.get(k) != _want[k]]
                                if _re_mismatch:
                                    problems.append(
                                        'group-aware selection_row_split metadata is '
                                        'internally inconsistent (mismatch: '
                                        f'{sorted(_re_mismatch)})')
                else:
                    try:
                        _re_sel, _re_ev, _re_meta = validation_row_split(
                            _split_info.get('n_rows'),
                            seed=_split_info.get('seed'),
                            n_folds=_split_info.get('n_folds'),
                            selection_fold=_split_info.get('selection_fold'),
                            evaluation_fold=_split_info.get('evaluation_fold'))
                    except (TypeError, ValueError, OverflowError) as _rfe:
                        problems.append('selection_row_split metadata cannot be '
                                        f'reproduced: {_rfe}')
                if _re_meta is not None:
                    _re_mismatch = []
                    for _k in _meta_keys:
                        if _k in _split_info and _split_info.get(_k) != _re_meta.get(_k):
                            _re_mismatch.append(_k)
                    if _re_mismatch:
                        problems.append(
                            'selection_row_split metadata does not match the '
                            'deterministic split implied by its declared '
                            f'n_rows/seed/folds (mismatch: {sorted(_re_mismatch)}) '
                            '— the recorded fold description is not reproducible')
            elif _fold_present:
                warnings.append('selection_row_split omits part of the fold '
                                'description (n_rows/seed/n_folds/selection_fold/'
                                'evaluation_fold); the recorded ranges cannot be '
                                'independently reproduced')
            _expected_sel_fp = _split_info.get('selection_row_fp')
            _expected_eval_fp = _split_info.get('evaluation_row_fp')
            for _arm, _entry in arms.items():
                if not isinstance(_entry, dict):
                    continue
                _entry_trials = _entry.get('trials')
                if not isinstance(_entry_trials, (list, tuple)):
                    _entry_trials = []
                for _t in _entry_trials:
                    # The flag must be the boolean True, not a truthy string or
                    # number: only an actual applied split attests the nested
                    # protocol.
                    if (not isinstance(_t, dict)
                            or _t.get('selection_row_split_applied') is not True):
                        problems.append(f'{_arm}: a tuning trial did not use the declared nested selection rows')
                        break
                    if _t.get('selection_row_fp') != _expected_sel_fp:
                        problems.append(f'{_arm}: a tuning trial selected on different rows than the artifact declares')
                        break
                    if _t.get('evaluation_row_fp') != _expected_eval_fp:
                        # The trial's held-out fold must be the SAME fold the
                        # final records are scored on; otherwise the artifact
                        # can pair a selection fold with a different final
                        # evaluation fold under one declared split.
                        problems.append(
                            f'{_arm}: a tuning trial declares a different evaluation '
                            'fold than the artifact selection_row_split')
                        break
        else:
            _row_split_applied = False
            _why = _split_info.get('reason') or 'the artifact did not apply the row split'
            warnings.append('the tuning protocol declares nested validation rows, but the artifact '
                            f'reports applied=False ({_why}); the final estimate may be selection-biased')
    _declared_trials = protocol.get('trials')
    if _declared_trials is not None and not isinstance(_declared_trials, (list, tuple)):
        problems.append(f'the tuning protocol trials field is not a list ({_declared_trials!r})')
        _declared_trials = []
    declared = {json.dumps(t, sort_keys=True, default=str)
                for t in (_declared_trials or [])}
    signatures = {}
    n_trials = {}
    budgets = {}
    search_seed_sets = {}
    selection_stabilities = {}
    eval_n_tokens = {}
    # Search-to-final data transfer: the selected hyperparameters were tuned
    # on a particular training corpus and validation slice.  A final estimate
    # on a different corpus/validation source is not the tuned configuration,
    # even if steps and token counts match.
    fingerprints_by_arm = {k: {} for k in
                           ('data_fp', 'val_fp', 'eval_fp', 'data_version')}
    for arm, entry in arms.items():
        trials = entry.get('trials') if isinstance(entry, dict) else None
        if not isinstance(trials, list) or not trials:
            problems.append(f'{arm}: no trial records')
            continue
        cfgs = [json.dumps(t.get('config') or {}, sort_keys=True, default=str) for t in trials if isinstance(t, dict)]
        signatures[arm] = tuple(sorted(cfgs))
        n_trials[arm] = len(cfgs)
        # Multi-seed selection: when trials expose their seed set, every trial
        # of every arm must average the same seeds.  A trial whose `ppl` is
        # not the mean of its own per-seed values is a fabricated aggregate.
        _set = set()
        for t in trials:
            if not isinstance(t, dict):
                continue
            _seeds = t.get('seeds')
            _ppls = t.get('seed_ppls')
            if _ppls is not None and not isinstance(_ppls, dict):
                problems.append(f'{arm}: seed_ppls is not a mapping')
                continue
            if _seeds is not None:
                try:
                    _raw_seeds = list(_seeds)
                except TypeError:
                    problems.append(f'{arm}: trial seeds are not an iterable ({_seeds!r})')
                    continue
                _norm_seeds = [_as_seed_int(s) for s in _raw_seeds]
                if any(s is None for s in _norm_seeds):
                    problems.append(f'{arm}: trial seeds are not integers ({_seeds!r})')
                    continue
                _ss = tuple(sorted(set(_norm_seeds)))
                if len(_ss) != len(_raw_seeds):
                    problems.append(f'{arm}: trial lists duplicate seeds')
                if _ppls is not None and set(str(s) for s in _ss) != set(_ppls):
                    problems.append(f'{arm}: trial seeds and seed_ppls keys disagree')
                _set.add(_ss)
            elif _ppls is not None:
                _norm_keys = [_as_seed_int(k) for k in _ppls]
                if any(k is None for k in _norm_keys):
                    problems.append(f'{arm}: seed_ppls keys are not integer seeds ({list(_ppls)[:3]!r})')
                    continue
                _set.add(tuple(sorted(set(_norm_keys))))
            _trial_ppl = _finite_scalar(t.get('ppl'))
            _trial_nll = _finite_scalar(t.get('nll_sum'))
            _trial_ntok = _exact_count(t.get('n_tokens'), minimum=1)
            _agg = t.get('ppl_aggregation')
            if _agg == 'pooled_token_nll' and (_trial_nll is None or _trial_ntok is None
                                               or _trial_ntok <= 0):
                problems.append(f'{arm}: trial claims pooled_token_nll but lacks a usable nll_sum/n_tokens')
            if isinstance(_ppls, dict) and _trial_ppl is not None:
                _vals = [f for f in (_finite_scalar(v) for v in _ppls.values())
                         if f is not None]
                if _vals:
                    if _trial_nll is not None and _trial_ntok is not None and _trial_ntok > 0:
                        _expected = float(_safe_exp_ppl(_trial_nll / _trial_ntok))
                        if not math.isfinite(_expected):
                            # The pooled value itself is beyond float64.  A
                            # finite reported PPL cannot equal it, and the old
                            # comparison `inf > inf` was False, so the artifact
                            # silently passed an unrepresentable aggregate.
                            problems.append(
                                f'{arm}: trial ppl cannot be verified: '
                                'exp(nll_sum/n_tokens) is not representable')
                        elif abs(_expected - _trial_ppl) > 1e-9 * max(1.0, abs(_expected)):
                            problems.append(f'{arm}: trial ppl is not exp(nll_sum/n_tokens)')
                    else:
                        # `np.mean` sums first: a finite per-seed panel near the
                        # float limit overflows to inf, and `inf > inf` is
                        # False, so a forged finite mean used to pass.  Use the
                        # same guarded mean as the writers.
                        _expected = float(_scaled_mean_or_raw(_vals))
                        if not math.isfinite(_expected):
                            problems.append(
                                f'{arm}: trial ppl cannot be verified: the '
                                'per-seed mean is not representable')
                        elif abs(_expected - _trial_ppl) > 1e-9 * max(1.0, abs(_expected)):
                            problems.append(f'{arm}: trial ppl is not the mean of its per-seed values')
                    if t.get('ppl_std') is not None and len(_vals) > 1:
                        # A non-finite/bool std is not a spread measurement;
                        # abs(NaN - s) > tol is False, so the old direct float()
                        # silently PASSED a corrupted ppl_std.
                        _reported_sd = _finite_scalar(t['ppl_std'])
                        if _reported_sd is None:
                            problems.append(f'{arm}: trial ppl_std is not numeric or finite ({t.get("ppl_std")!r})')
                        else:
                            # Recompute with the same guarded SD used by the
                            # writers: a finite PPL panel can still overflow the
                            # raw squared deviations at extreme magnitudes.
                            _ms = _scaled_mean_std_if_extreme(_vals, ddof=1)
                            _s = (float(_ms[1]) if _ms is not None
                                  else float(np.std(_vals, ddof=1)))
                            if not math.isfinite(_s):
                                # An unrepresentable sample SD cannot match a
                                # finite reported value; `inf > inf` would
                                # otherwise accept it.
                                problems.append(
                                    f'{arm}: trial ppl_std cannot be verified: '
                                    'the sample std is not representable')
                            elif abs(_s - _reported_sd) > 1e-9 * max(1.0, abs(_s)):
                                problems.append(f'{arm}: trial ppl_std is not the sample std of its per-seed values')
        if _set:
            search_seed_sets[arm] = _set
        if declared and any(c not in declared for c in cfgs):
            problems.append(f'{arm}: a trial is outside the declared grid')
        elif declared and set(cfgs) != declared:
            # Equal-but-partial search is allowed for smoke/bounded runs, but
            # it must be described as a subset, not as the full declared grid.
            _missing_cfg = sorted(declared - set(cfgs))
            warnings.append(f'{arm}: searched a strict subset of the declared grid (missing {_missing_cfg[:2]})')
        by_cfg = {}
        for t in trials:
            if not isinstance(t, dict):
                continue
            sig = json.dumps(t.get('config') or {}, sort_keys=True, default=str)
            by_cfg[sig] = (t.get('steps'), t.get('tokens_seen'))
        budgets[arm] = by_cfg
        # Trial PPLs are comparable only on the same evaluation sample.  A
        # trial evaluated on a different subset is a different measurement,
        # even if it used the same training budget and hyperparameter grid.
        _eval_tok = set()
        for t in trials:
            if not isinstance(t, dict) or t.get('n_tokens') is None:
                continue
            _nt = _exact_count(t.get('n_tokens'), minimum=1)
            if _nt is None:
                problems.append(f'{arm}: trial n_tokens is not an exact positive count ({t.get("n_tokens")!r})')
                continue
            _eval_tok.add(_nt)
        if len(_eval_tok) > 1:
            problems.append(f'{arm}: trials selected on different evaluation token counts {sorted(_eval_tok)} — their PPLs are not comparable')
        eval_n_tokens[arm] = sorted(_eval_tok)
        # Exact corpus/validation identities, when the artifact records them.
        # One trial with a different fingerprint means the grid was not even
        # searched on one sample; a partial descriptor is unverifiable.
        for _fk in fingerprints_by_arm:
            _vals_fp = set()
            _missing_fp = 0
            for t in trials:
                if not isinstance(t, dict):
                    continue
                _v_fp = t.get(_fk)
                if _v_fp is None or (isinstance(_v_fp, str) and not _v_fp.strip()):
                    _missing_fp += 1
                else:
                    _vals_fp.add(str(_v_fp))
            fingerprints_by_arm[_fk][arm] = {
                'values': sorted(_vals_fp), 'n_missing': _missing_fp,
                'n_trials': len([t for t in trials if isinstance(t, dict)])}
            if len(_vals_fp) > 1:
                problems.append(
                    f'{arm}: trials selected on different {_fk} values '
                    f'{sorted(_vals_fp)} — the tuning grid mixes corpora/'
                    'validation samples')
            elif _missing_fp:
                warnings.append(
                    f'{arm}: {_missing_fp} tuning trial(s) omit {_fk}; the '
                    'search-to-final data transfer is unverifiable')
        # Step/token budgets are exact counts.  Raw equality would treat
        # True == 1 and let a boolean/truncated budget pose as a matched one.
        _steps, _toks, _bad_budget = set(), set(), []
        for _b in by_cfg.values():
            for _idx, _label, _target in ((0, 'steps', _steps),
                                          (1, 'tokens_seen', _toks)):
                if _b[_idx] is None:
                    continue
                _cnt = _exact_count(_b[_idx], minimum=1)
                if _cnt is None:
                    _bad_budget.append((_label, _b[_idx]))
                else:
                    _target.add(_cnt)
        if _bad_budget:
            problems.append(f'{arm}: trial budget values are not exact positive counts ({_bad_budget})')
        if len(_steps) > 1:
            problems.append(f'{arm}: trials used different step budgets {sorted(_steps)}')
        if len(_toks) > 1:
            problems.append(f'{arm}: trials saw different token counts {sorted(_toks)}')
        chosen = entry.get('selected') or entry.get('selected_config')
        if not isinstance(chosen, dict):
            problems.append(f'{arm}: no selected config')
            continue
        chosen_sig = json.dumps(chosen, sort_keys=True, default=str)
        if chosen_sig not in cfgs:
            problems.append(f'{arm}: selected config is not one of its recorded trials')
            continue
        chosen_trial = next(t for t in trials if isinstance(t, dict)
                            and json.dumps(t.get('config') or {}, sort_keys=True, default=str) == chosen_sig)
        _stab = tuning_selection_stability(trials, metric=metric, direction=direction)
        selection_stabilities[arm] = _stab
        if _stab.get('status') == 'ok':
            _recorded = entry.get('selection_stability')
            if isinstance(_recorded, dict) and isinstance(_recorded.get('stable'), bool):
                if bool(_recorded['stable']) != bool(_stab.get('stable')):
                    problems.append(f'{arm}: recorded selection_stability={_recorded.get("stable")} '
                                    f'but leave-one-seed-out re-selection gives {_stab.get("stable")}')
            if not _stab.get('stable'):
                warnings.append(f'{arm}: the selected config is not leave-one-search-seed-out stable '
                                f'({_stab.get("n_distinct_configs")} distinct winners) — the tuned '
                                'pick may be a single-seed draw')
        elif _stab.get('status') == 'insufficient_seeds':
            warnings.append(f'{arm}: selections stability is unverifiable — {_stab.get("reason")}')
        # Selection uncertainty: the raw argmin may beat the runner-up by less
        # than one standard error, in which case the "winner" is not
        # distinguishable from the next config.  Recompute the margin from the
        # trial records (do not trust a self-reported value) and warn.
        _ranked = sorted([t for t in trials if isinstance(t, dict)
                          and _finite_scalar(t.get(metric)) is not None],
                         key=lambda t: _finite_scalar(t.get(metric)),
                         reverse=(direction == 'higher_is_better'))
        _distinct = []
        _seen_sig = set()
        for _t in _ranked:
            _sig = json.dumps(_t.get('config') or {}, sort_keys=True, default=str)
            if _sig in _seen_sig:
                continue
            _seen_sig.add(_sig)
            _distinct.append(_t)
        _margin = _margin_se = _margin_z = None
        _decisive = None
        _margin_method = 'unavailable'
        _margin_independent = _decisive_independent = None
        if len(_distinct) >= 2:
            _mu = tuning_selection_margin(_distinct[0], _distinct[1], metric=metric,
                                          direction=direction)
            _margin = _mu.get('margin')
            _margin_se = _mu.get('se')
            _margin_z = _mu.get('z')
            _decisive = _mu.get('decisive')
            _margin_method = _mu.get('method')
            _margin_independent = _mu.get('se_independent')
            _decisive_independent = _mu.get('decisive_independent')
        if _decisive is False:
            warnings.append(
                f'{arm}: the selected config beats the runner-up by only {_margin:.4g} '
                f'{metric} (SE {_margin_se:.4g}, {_margin_method}) — the pick is inside '
                'one standard error; report it as a tie/uncertain choice, not as the '
                'uniquely tuned optimum')
        _rec_margin = _finite_scalar(entry.get('selection_margin'))
        if _rec_margin is not None and _margin is not None and abs(_rec_margin - _margin) > 1e-9:
            problems.append(
                f'{arm}: recorded selection_margin={_rec_margin} does not match the '
                f'margin between the selected and runner-up trials ({_margin})')
        _rec_method = entry.get('selection_margin_method')
        _rec_se = _finite_scalar(entry.get('selection_margin_se'))
        if _rec_se is not None and _margin_se is not None:
            _se_candidates = [x for x in (_margin_se, _margin_independent) if x is not None]
            # A legacy artifact has no method tag and stored the independent
            # marginal-SE value; accept either estimate there, but a tagged
            # artifact must match its declared method.
            _se_ok = any(abs(_rec_se - x) <= 1e-9 * max(1.0, abs(x)) for x in _se_candidates)
            if isinstance(_rec_method, str) and _rec_method in ('paired_search_seeds', 'independent_marginal_se'):
                _want = _margin_se if _rec_method == _margin_method else None
                _se_ok = (_want is not None
                          and abs(_rec_se - _want) <= 1e-9 * max(1.0, abs(_want)))
            if not _se_ok:
                problems.append(
                    f'{arm}: recorded selection_margin_se={_rec_se} does not match the '
                    f'recomputed {_margin_method} value ({_margin_se})')
        _rec_decisive = entry.get('selection_decisive')
        if isinstance(_rec_decisive, (bool, np.bool_)) and _decisive is not None:
            _ok_dec = bool(_rec_decisive) == _decisive
            # Legacy artifacts have no method tag and recorded the independent
            # marginal-SE decision.  Accept that value only when they did not
            # declare a method; a tagged artifact must match its own method.
            if not _ok_dec and not isinstance(_rec_method, str):
                _ok_dec = (_decisive_independent is not None
                           and bool(_rec_decisive) == bool(_decisive_independent))
            if not _ok_dec:
                problems.append(
                    f'{arm}: recorded selection_decisive={bool(_rec_decisive)} but '
                    f'recomputed margin/SE gives {_decisive} ({_margin_method})')
        vals = [f for f in (_finite_scalar(t.get(metric)) for t in trials
                             if isinstance(t, dict)) if f is not None]
        chosen_value = entry.get('selected_metric_value')
        if chosen_value is None:
            chosen_value = chosen_trial.get(metric)
        chosen_f = _finite_scalar(chosen_value)
        if not vals:
            problems.append(f'{arm}: no finite {metric} in its trials')
        elif chosen_f is None:
            problems.append(f'{arm}: no finite selected {metric} value')
        else:
            _ctv = _finite_scalar(chosen_trial.get(metric))
            if _ctv is None:
                problems.append(f'{arm}: selected trial has no finite {metric}')
            elif abs(_ctv - chosen_f) > 1e-12:
                problems.append(f'{arm}: selected_metric_value={chosen_value} is not the value of the selected config ({chosen_trial.get(metric)})')
            if direction == 'lower_is_better':
                if chosen_f > min(vals) + 1e-12:
                    problems.append(f'{arm}: selected trial is not the minimum-{metric} trial')
            else:
                if chosen_f < max(vals) - 1e-12:
                    problems.append(f'{arm}: selected trial is not the maximum-{metric} trial')
    # Across arms, every COMPLETE per-arm fingerprint must agree.  Otherwise
    # the arms were tuned on different data and their selected configs are not
    # comparable.
    for _fk, _by_arm in fingerprints_by_arm.items():
        _complete = {a: row['values'][0] for a, row in _by_arm.items()
                     if len(row['values']) == 1 and row['n_missing'] == 0}
        if len(set(_complete.values())) > 1:
            problems.append(
                'arms were tuned on different %s values (%s) — equal search '
                'budgets do not make the selected configs comparable across '
                'different corpora/validation samples'
                % (_fk, ', '.join('%s=%s' % (a, v) for a, v in sorted(_complete.items()))))
    if signatures:
        if len(set(signatures.values())) != 1:
            problems.append('arms did not search the same trial grid — tuning budgets are unequal')
        if len(set(n_trials.values())) != 1:
            problems.append('arms did not receive the same number of tuning trials — tuning budgets are unequal')
    # Equal tuning includes equal SEARCH SEEDS.  If one arm selected on seeds
    # {0} and another on {0,1,2}, their selection noise differs even though the
    # lr grid is identical.  Only trials that expose their seed set are
    # checkable; legacy artifacts without it stay on the untested path.
    search_seed_balanced = None
    search_seeds_seen = None
    _declared_ss = protocol.get('search_seeds')
    if _declared_ss is not None:
        try:
            _raw_declared = list(_declared_ss)
            _norm_declared = [_as_seed_int(s) for s in _raw_declared]
            if any(s is None for s in _norm_declared):
                raise ValueError
            _declared_ss = tuple(sorted(set(_norm_declared)))
        except (TypeError, ValueError, OverflowError):
            problems.append(f'declared search_seeds are not integers ({_declared_ss!r})')
            _declared_ss = None
    if search_seed_sets:
        _no_seed_info = sorted(set(arms) - set(search_seed_sets))
        if _no_seed_info:
            warnings.append(f'arms {_no_seed_info} do not expose their search seeds — equal search seeds cannot be verified for them')
        # Each trial of an arm must use the SAME search-seed set.  Comparing
        # only the collection of sets across arms let an arm (or every arm)
        # search trial 0 on seeds {0,1} and trial 1 on seeds {2,3}; that is a
        # different selection draw per config, not an equal search.
        _multi_seed_arms = sorted(a for a, v in search_seed_sets.items() if len(v) > 1)
        if _multi_seed_arms:
            search_seed_balanced = False
            problems.append(
                'arm(s) %s used more than one search-seed set across trials — '
                'each config would be selected on different draws; equal-search '
                'tuning is not verified' % ', '.join(_multi_seed_arms))
        elif len(search_seed_sets) >= 2:
            search_seed_balanced = len({next(iter(v)) for v in search_seed_sets.values()}) == 1
            if not search_seed_balanced:
                problems.append('arms did not select on the same search-seed set — equal-search tuning is not verified')
        search_seeds_seen = sorted({s for v in search_seed_sets.values() for ss in v for s in ss})
        if _declared_ss is not None and set(search_seeds_seen) != set(_declared_ss):
            warnings.append(f'trials report search seeds {search_seeds_seen} but the declared set is {list(_declared_ss)}')
    elif _declared_ss is not None:
        warnings.append('trial records do not expose their search seeds — multi-seed selection cannot be verified from this artifact')
    _all_sigs = set().union(*(set(b) for b in budgets.values())) if budgets else set()
    for sig in sorted(_all_sigs):
        _vals = {arm: b[sig] for arm, b in budgets.items() if sig in b}
        _st = {_exact_count(v[0], minimum=1) for v in _vals.values()
               if v[0] is not None}
        _tk = {_exact_count(v[1], minimum=1) for v in _vals.values()
               if v[1] is not None}
        if None in _st or None in _tk or len(_st) > 1 or len(_tk) > 1:
            problems.append(f'trial {sig[:48]} uses different steps/tokens across arms (steps={sorted(_st, key=str)}, tokens={sorted(_tk, key=str)})')
            break
    if selected.get('search_complete') is False:
        warnings.append('the search grid was truncated (search_complete=False); it is equal across arms, but describe it as a bounded search, not the full declared grid')
    _final_records_raw = selected.get('final_records')
    if _final_records_raw is None:
        final_records = {}
    elif isinstance(_final_records_raw, dict):
        final_records = _final_records_raw
    else:
        # A scalar/list cannot describe record -> measurement rows; treating
        # it as an empty map would silently skip the final-seed coverage and
        # holdout checks instead of failing the artifact.
        problems.append(f'final_records must be a mapping, got {type(_final_records_raw).__name__}')
        final_records = {}
    final_seeds = selected.get('final_seeds') or protocol.get('final_seeds') or []
    # Selection/evaluation partition: the final tuned records must not be
    # measured on the same seed draws that chose the hyper-parameters.
    _declared_search = selected.get('search_seeds') or protocol.get('search_seeds')
    _search_set = set()
    if _declared_search is not None:
        try:
            _norm_search = [_as_seed_int(s) for s in _declared_search]
            if any(s is None for s in _norm_search):
                raise ValueError
            _search_set = set(_norm_search)
        except (TypeError, ValueError, OverflowError):
            problems.append(f'search_seeds are not integers ({_declared_search!r})')
    _final_set = set()
    try:
        _raw_final = list(final_seeds or [])
    except TypeError:
        problems.append(f'declared final_seeds are not an iterable ({final_seeds!r})')
        _raw_final = []
    for _s in _raw_final:
        _si = _as_seed_int(_s)
        if _si is None:
            # Dropping an unparseable seed would shrink the declared final
            # panel and could make the remaining coverage look complete.
            problems.append(f'declared final seed {_s!r} is not an integer')
            continue
        if _si in _final_set:
            problems.append(f'declared final seed {_si} is duplicated')
        _final_set.add(_si)
    _seed_overlap = sorted(_search_set & _final_set)
    if _seed_overlap:
        _msg = (f'search_seeds and final_seeds overlap on {_seed_overlap} — the '
                'final tuned records would be measured on the draws that selected '
                'the config (selection-biased); declare a disjoint final partition')
        if final_records:
            problems.append(_msg)
        else:
            warnings.append(_msg)
    # After the per-arm loop below, record final-record seeds that overlap the
    # declared search set too (a record can carry a different seed from the
    # declared list only when the artifact is inconsistent, which is already
    # checked; this makes the overlap visible either way).
    final_seed_coverage = {}
    if final_records:
        for arm, entry in arms.items():
            if not isinstance(entry, dict):
                # The per-arm loop above already recorded the malformed arm;
                # it has no selected config to line up with final records.
                continue
            chosen = entry.get('selected') or entry.get('selected_config')
            if not isinstance(chosen, dict):
                chosen = {}
            hits = [(k, r) for k, r in final_records.items()
                    if k.startswith(f'{arm}::') and isinstance(r, dict)]
            if not hits:
                problems.append(f'{arm}: tuning artifact claims final records but none are bound to this arm')
                continue
            _seeds = [s for _k, r in hits
                      if (s := _as_seed_int(r.get('seed'))) is not None]
            if len(_seeds) != len(hits):
                problems.append(f'{arm}: a final record has no usable integer seed')
            if len(_seeds) != len(set(_seeds)):
                problems.append(f'{arm}: duplicate final records for the same seed')
            final_seed_coverage[arm] = sorted(set(_seeds))
            if _final_set:
                _miss = sorted(_final_set - set(_seeds))
                if _miss:
                    problems.append(f'{arm}: final records miss declared seeds {_miss}')
                # The declared partition is the plan: a record for an
                # UNDECLARED seed silently enlarges n and can make an
                # underpowered contrast look resolvable.  Refuse it instead of
                # treating extra records as if the plan had included them.
                _extra = sorted(set(_seeds) - _final_set)
                if _extra:
                    problems.append(
                        f'{arm}: final records include undeclared seed(s) {_extra}; '
                        'the final partition is not the declared plan')
            for _k, r in hits:
                hp = r.get('hyperparams')
                # A final record that does not expose the knobs it ran cannot
                # attest that the selected configuration was actually
                # evaluated.  Treat a missing descriptor as a failed check,
                # not as "nothing to compare": otherwise an artifact can
                # claim a tuned final result while the record is silent.
                if not isinstance(hp, dict):
                    problems.append(
                        f'{arm}: final record {_k} has no structured hyperparams '
                        'descriptor — it cannot attest that the selected config was run')
                    hp = {}
                for _kn in ('lr', 'weight_decay', 'warmup'):
                    if _kn not in chosen:
                        continue
                    if hp.get(_kn) is None:
                        problems.append(
                            f'{arm}: final record {_k} does not record {_kn}; '
                            f'the selected {_kn}={chosen[_kn]!r} is unattested')
                        continue
                    try:
                        _hpv = float(hp[_kn])
                        _selv = float(chosen[_kn])
                    except (TypeError, ValueError, OverflowError):
                        problems.append(
                            f'{arm}: final record {_kn}={hp[_kn]!r} or the '
                            f'selected value {chosen[_kn]!r} is not numeric')
                        break
                    if _hpv != _selv:
                        problems.append(f'{arm}: final record {_kn}={hp[_kn]} does not match the selected {chosen[_kn]}')
                        break
                if r.get('variant') is not None and r.get('variant') != arm:
                    problems.append(f'{arm}: final record key {_k} carries variant={r.get("variant")!r}')
                if _row_split_applied:
                    _hold = _finite_scalar(r.get('ppl_holdout'))
                    if _hold is None:
                        problems.append(f'{arm}: nested selection was applied but final record {_k} '
                                        'has no held-out-row PPL (ppl_holdout) — the reported tuned '
                                        'contrast would fall back to the selection documents')
                    elif r.get('holdout_row_fp') != (_split_info or {}).get('evaluation_row_fp'):
                        problems.append(f'{arm}: final record {_k} was scored on different held-out '
                                        'rows than the declared evaluation fold')
                    else:
                        _nested_final = True
    if final_records and _row_split_applied and _nested_final is None:
        problems.append('nested selection was applied but no final record carries a held-out-row PPL')
    if final_records and _row_split_applied is False:
        warnings.append('the tuning artifact did not apply the nested row split; the final tuned '
                        'records may be measured on the rows that selected the config')
    # Budget transfer: a config selected on a SHORTER search horizon and then
    # evaluated at the final horizon is not the tuned optimum for the budget
    # that is actually reported.  Equal search across arms does not catch this,
    # because every arm could share the same too-short search.  The artifact
    # declares `search_steps`; the final records carry their completed/planned
    # steps.  A known mismatch is disclosed and gates the tuned claim (below),
    # while legacy artifacts that omit the descriptor stay unverifiable.
    _raw_search_steps = selected.get('search_steps')
    _search_steps = None
    _budget_transfer = {'search_steps': None, 'trial_steps_by_arm': {},
                        'final_steps_by_arm': {}, 'trial_tokens_by_arm': {},
                        'final_tokens_by_arm': {}, 'equal': None,
                        'tokens_equal': None, 'checked': False,
                        'mismatch_arms': [], 'token_mismatch_arms': []}
    if _raw_search_steps is not None:
        _search_steps = _exact_count(_raw_search_steps, minimum=1)
        if _search_steps is None:
            problems.append(f'search_steps={_raw_search_steps!r} is not an exact positive step count')
    _budget_transfer['search_steps'] = _search_steps
    _trial_steps_by_arm = {}
    for _arm, _by_cfg in budgets.items():
        _trial_steps = {_exact_count(v[0], minimum=1) for v in _by_cfg.values()}
        _trial_steps.discard(None)
        if _trial_steps:
            _trial_steps_by_arm[_arm] = sorted(_trial_steps)
    _trial_tokens_by_arm = {}
    for _arm, _by_cfg in budgets.items():
        _trial_tokens = {_exact_count(v[1], minimum=1) for v in _by_cfg.values()}
        _trial_tokens.discard(None)
        if _trial_tokens:
            _trial_tokens_by_arm[_arm] = sorted(_trial_tokens)
    _budget_transfer['trial_steps_by_arm'] = _trial_steps_by_arm
    _budget_transfer['trial_tokens_by_arm'] = _trial_tokens_by_arm
    if _search_steps is not None and _trial_steps_by_arm:
        for _arm, _ts in sorted(_trial_steps_by_arm.items()):
            if len(_ts) != 1 or _ts[0] != _search_steps:
                problems.append(
                    f'{_arm}: declared search_steps={_search_steps} but its tuning '
                    f'trials ran steps={_ts}; the artifact does not describe the '
                    'search budget that produced the selected config')
    if final_records:
        _final_steps_by_arm = {}
        _final_tokens_by_arm = {}
        _final_fps_by_arm = {k: {} for k in fingerprints_by_arm}
        for _arm in arms:
            _hits = [(k, r) for k, r in final_records.items()
                     if k.startswith(f'{_arm}::') and isinstance(r, dict)]
            _steps, _tokens = set(), set()
            _missing_step_records, _missing_token_records = [], []
            for _k, r in _hits:
                _st = _exact_count(r.get('steps'), minimum=1)
                _tk = _exact_count(r.get('tokens_seen'), minimum=1)
                _hp = r.get('hyperparams')
                if isinstance(_hp, dict):
                    if _st is None:
                        _st = _exact_count(_hp.get('steps'), minimum=1)
                    if _tk is None:
                        _tk = _exact_count(_hp.get('tokens_seen'), minimum=1)
                if _st is None:
                    _missing_step_records.append(_k)
                else:
                    _steps.add(_st)
                if _tk is None:
                    _missing_token_records.append(_k)
                else:
                    _tokens.add(_tk)
            if len(_steps) > 1:
                problems.append(f'{_arm}: final records ran different step budgets '
                                f'{sorted(_steps)} — the tuned contrast would mix horizons')
            elif len(_steps) == 1:
                _final_steps_by_arm[_arm] = next(iter(_steps))
            if len(_tokens) > 1:
                problems.append(f'{_arm}: final records saw different token budgets '
                                f'{sorted(_tokens)} — the tuned contrast would mix exposures')
            elif len(_tokens) == 1:
                _final_tokens_by_arm[_arm] = next(iter(_tokens))
            if _missing_step_records:
                warnings.append(f'{_arm}: {len(_missing_step_records)} final record(s) do not '
                                'record steps; the search-to-final horizon transfer is unverifiable')
            if _missing_token_records:
                warnings.append(f'{_arm}: {len(_missing_token_records)} final record(s) do not '
                                'record tokens_seen; the search-to-final exposure transfer is unverifiable')
            for _fk in fingerprints_by_arm:
                _final_fp_vals = {str(r.get(_fk)) for _k, r in _hits
                                  if r.get(_fk) is not None
                                  and not (isinstance(r.get(_fk), str)
                                           and not r.get(_fk).strip())}
                _final_fps_by_arm[_fk][_arm] = _final_fp_vals
                if len(_final_fp_vals) > 1:
                    problems.append(
                        f'{_arm}: final records carry different {_fk} values '
                        f'{sorted(_final_fp_vals)} — a tuned contrast cannot mix '
                        'corpora/validation samples')
        # The final estimate must run on the same data identity as the tuning
        # search.  Missing descriptors are unverifiable (legacy), not a pass.
        for _fk, _by_arm in _final_fps_by_arm.items():
            _complete_final = {a: next(iter(v)) for a, v in _by_arm.items()
                               if len(v) == 1}
            if len(set(_complete_final.values())) > 1:
                problems.append(
                    'final records for different arms were scored on different '
                    '%s values' % _fk)
            for _arm, _vals in sorted(_by_arm.items()):
                if not _vals:
                    warnings.append(
                        f'{_arm}: final record(s) omit {_fk}; the search-to-final '
                        'data transfer is unverifiable')
                    continue
                if len(_vals) != 1:
                    continue
                _trial_vals = (fingerprints_by_arm.get(_fk, {})
                               .get(_arm, {}).get('values') or [])
                if not _trial_vals:
                    warnings.append(
                        f'{_arm}: tuning trials omit {_fk}; the search-to-final '
                        'data transfer is unverifiable')
                elif _trial_vals != sorted(_vals):
                    problems.append(
                        f'{_arm}: final records use {_fk}={sorted(_vals)} but the '
                        f'tuning trials used {_trial_vals} — the selected config '
                        'was tuned on a different corpus/validation sample than the '
                        'reported final estimate')
        _budget_transfer['final_steps_by_arm'] = _final_steps_by_arm
        _budget_transfer['final_tokens_by_arm'] = _final_tokens_by_arm
        if _search_steps is not None and _final_steps_by_arm:
            _mismatch = sorted(a for a, s in _final_steps_by_arm.items()
                               if s != _search_steps)
            _budget_transfer['checked'] = True
            _budget_transfer['equal'] = not _mismatch
            _budget_transfer['mismatch_arms'] = _mismatch
            if _mismatch:
                warnings.append(
                    'the tuning search ran at %d step(s) but the final tuned records for '
                    '%s ran at %s — hyperparameters were selected on a different training '
                    'horizon than the reported evaluation, so this is NOT a fully tuned '
                    'baseline at the final budget'
                    % (_search_steps, ', '.join(_mismatch),
                       ', '.join('%s=%d' % (a, _final_steps_by_arm[a]) for a in _mismatch)))
        elif _search_steps is None:
            warnings.append('the artifact does not declare search_steps; the tuning search '
                            'horizon cannot be compared with the final evaluation horizon')
        if _trial_tokens_by_arm and _final_tokens_by_arm:
            _tok_mismatch = []
            for _arm, _final_tok in sorted(_final_tokens_by_arm.items()):
                _trial_tok = _trial_tokens_by_arm.get(_arm)
                if _trial_tok != [_final_tok]:
                    _tok_mismatch.append(_arm)
            _budget_transfer['tokens_equal'] = not _tok_mismatch
            _budget_transfer['token_mismatch_arms'] = _tok_mismatch
            if _tok_mismatch:
                warnings.append(
                    'the tuning trials and the final records for %s saw different '
                    'token budgets (%s vs %s) — equal steps do not imply equal '
                    'exposure, so the selected configuration was not tuned on the '
                    'final training budget'
                    % (', '.join(_tok_mismatch),
                       {a: _trial_tokens_by_arm.get(a) for a in _tok_mismatch},
                       {a: _final_tokens_by_arm.get(a) for a in _tok_mismatch}))
    _measured_final_seeds = sorted({s for _v in final_seed_coverage.values() for s in _v})
    _record_overlap = sorted(_search_set & set(_measured_final_seeds))
    if _record_overlap and final_records and not _seed_overlap:
        problems.append(f'final records use search seeds {_record_overlap} — the '
                        'final estimate is measured on the selection draws')
    final_seed_coverage_equal = None
    final_min_seed_count = None
    _final_family_floor = None
    # The tuned family is the set of comparisons the protocol DECLARES, not
    # one contrast per pair of arms.  A grouped protocol can have six arms but
    # only four method-vs-baseline contrasts; using `len(arms) - 1` inflates
    # the Holm multiplicity and misstates the planning floor.
    _final_plan, _final_family_basis = _tuning_declared_contrasts(
        protocol, list(arms))
    _final_family_m = (len(_final_plan) if _final_plan
                       else max(1, len(arms) - 1))
    if final_seed_coverage:
        final_seed_coverage_equal = len({tuple(v) for v in final_seed_coverage.values()}) == 1
        final_min_seed_count = min(len(v) for v in final_seed_coverage.values())
        if not final_seed_coverage_equal:
            problems.append('final records do not cover the same seeds in every arm — the paired comparison would silently use different n per arm')
        _final_family_floor = signflip_min_n_for_holm(_final_family_m, 0.05)
        if final_min_seed_count < 3:
            warnings.append(f'only {final_min_seed_count} final seed(s) per arm — the reporting floor for a key experiment is 3; treat the tuned contrast as descriptive')
        elif _final_family_floor is not None and final_min_seed_count < _final_family_floor:
            warnings.append(f'{final_min_seed_count} final seeds per arm cannot reach a Holm-corrected p<0.05 for {_final_family_m} declared tuned contrast(s) (needs n>={_final_family_floor}); the tuned contrast is resolution-limited, not a null')
    # Coverage: an arm is covered either by its own tuned record or by the arm
    # whose recipe it inherits (budget-matched intervention controls).
    _inherits = protocol.get('inherits_tuning')
    if _inherits is not None and not isinstance(_inherits, dict):
        problems.append(f'the tuning protocol inherits_tuning field is not an object ({_inherits!r})')
        _inherits = {}
    _inherits = _inherits or {}
    _declared_arms = protocol.get('arms')
    if _declared_arms is not None and not isinstance(_declared_arms, (list, tuple, set)):
        problems.append(f'the tuning protocol arms field is not a list ({_declared_arms!r})')
        _declared_arms = []
    _declared_arm_names = list(_declared_arms or [])
    coverage = {a: a for a in arms}
    for _arm, _ref in _inherits.items():
        if _ref in arms:
            coverage[str(_arm)] = str(_ref)
        elif _arm in _declared_arm_names or _arm in _inherits:
            warnings.append(f'{_arm}: inherits tuning from {_ref!r}, but no tuned record exists for {_ref!r}')
    required = set(_declared_arm_names)
    uncovered = sorted(a for a in required if a not in coverage)
    if uncovered:
        problems.append(f'missing tuning evidence for arms {uncovered}')
    _nonempty_eval = [tuple(v) for v in eval_n_tokens.values() if v]
    if len(set(_nonempty_eval)) > 1:
        problems.append('arms selected on different evaluation token counts '
                        f'{sorted(set(_nonempty_eval))} — the equal-search comparison is not on a common sample')
    _grid_complete = None
    if signatures and declared:
        _grid_complete = all(set(sig) == declared for sig in signatures.values())
    # `search_complete` is an attestation, not a free-form note.  A bounded
    # smoke/subset run may legitimately say False, but claiming True while the
    # recorded trials are a strict subset of the declared grid is a
    # contradiction: downstream gates would then promote a truncated search to
    # a "fully tuned" baseline.
    _artifact_search_complete = selected.get('search_complete')
    if (_artifact_search_complete is not None
            and not isinstance(_artifact_search_complete, (bool, np.bool_))):
        problems.append(
            f'search_complete must be a boolean, got '
            f'{_artifact_search_complete!r}')
        _artifact_search_complete = None
    _effective_search_complete = _grid_complete
    if _artifact_search_complete is not None:
        _effective_search_complete = bool(_artifact_search_complete)
        if _artifact_search_complete is True and _grid_complete is False:
            problems.append(
                'the artifact claims search_complete=True, but the recorded '
                'trials are a strict subset of the declared grid; the search '
                'was truncated and cannot attest a complete grid')
            _effective_search_complete = False
    # Grid-coverage adequacy: an equal search can still be an inadequate one.
    # If the selected learning rate / weight decay sits on the minimum or
    # maximum declared level, the optimum may lie outside the searched range;
    # a single level per knob cannot show that the optimum is bracketed.
    _grid_levels = {}
    for _kn in ('lr', 'weight_decay', 'warmup'):
        _vals = []
        for _t in (_declared_trials or []):
            if not isinstance(_t, dict):
                continue
            _f = _finite_scalar(_t.get(_kn))
            if _f is not None and _f not in _vals:
                _vals.append(_f)
        if _vals:
            _grid_levels[_kn] = sorted(_vals)
    grid_coverage = {}
    grid_factorial = {}
    _boundary_hits = []
    _coarse = []
    for arm, entry in arms.items():
        if not isinstance(entry, dict):
            continue
        chosen = entry.get('selected') or entry.get('selected_config')
        if not isinstance(chosen, dict):
            continue
        per_knob = {}
        _searched_by_knob = {}
        for _kn, _levels in _grid_levels.items():
            _sel = _finite_scalar(chosen.get(_kn))
            if _sel is None:
                continue
            # Boundary adequacy is about the levels this arm ACTUALLY searched,
            # not only the declared grid: a capped run that stopped at the
            # first two trials has a boundary even if more levels were declared.
            _searched = []
            _entry_trials = entry.get('trials')
            if not isinstance(_entry_trials, (list, tuple)):
                _entry_trials = []
            for _t in _entry_trials:
                if not isinstance(_t, dict):
                    continue
                _cfg_t = _t.get('config')
                if not isinstance(_cfg_t, dict):
                    _cfg_t = {}
                _f = _finite_scalar(_cfg_t.get(_kn))
                if _f is not None and _f not in _searched:
                    _searched.append(_f)
            _searched.sort()
            _searched_by_knob[_kn] = _searched
            _pos = None
            for _i, _v in enumerate(_levels):
                if abs(float(_v) - _sel) <= 1e-12 * max(1.0, abs(_sel)):
                    _pos = _i
                    break
            if _pos is None:
                problems.append(f'{arm}: selected {_kn}={_sel} is outside the declared trial levels {_levels}')
                per_knob[_kn] = {'selected': _sel, 'levels': _levels,
                                 'searched_levels': _searched,
                                 'n_levels': len(_levels), 'position': None,
                                 'at_boundary': None, 'outside_grid': True}
                continue
            _spos = None
            for _i, _v in enumerate(_searched):
                if abs(float(_v) - _sel) <= 1e-12 * max(1.0, abs(_sel)):
                    _spos = _i
                    break
            if _spos is None:
                problems.append(f'{arm}: selected {_kn}={_sel} is outside the searched levels {_searched}')
                per_knob[_kn] = {'selected': _sel, 'levels': _levels,
                                 'searched_levels': _searched,
                                 'n_levels': len(_levels), 'position': _pos,
                                 'at_boundary': None, 'outside_grid': True}
                continue
            _at_boundary = bool(len(_searched) >= 3 and _spos in (0, len(_searched) - 1))
            per_knob[_kn] = {'selected': _sel, 'levels': _levels,
                             'searched_levels': _searched,
                             'n_levels': len(_levels),
                             'n_searched_levels': len(_searched),
                             'position': _pos, 'searched_position': _spos,
                             'at_boundary': _at_boundary, 'outside_grid': False}
            if _at_boundary:
                _boundary_hits.append(f'{arm}:{_kn}')
            if _kn == 'lr' and 0 < len(_searched) < 3:
                _coarse.append(f'{arm}: lr has only {len(_searched)} searched level(s)')
        # Factorial/interaction coverage: the default design is
        # one-factor-at-a-time, so a marginal winner does not establish a
        # joint optimum.  Record the searched Cartesian coverage (overall and
        # per knob pair) instead of silently treating the OFAT winner as fully
        # tuned.
        _entry_trials = entry.get('trials')
        if not isinstance(_entry_trials, (list, tuple)):
            _entry_trials = []
        _arm_trials = [t for t in _entry_trials if isinstance(t, dict)]
        _knobs = sorted(_searched_by_knob)
        _combos = set()
        for _t in _arm_trials:
            _cfg = _t.get('config')
            if not isinstance(_cfg, dict):
                _cfg = {}
            _vals = tuple(_finite_scalar(_cfg.get(_kn)) for _kn in _knobs)
            if all(v is not None for v in _vals):
                _combos.add(_vals)
        _full_size = 1
        for _kn in _knobs:
            _full_size *= max(1, len(_searched_by_knob.get(_kn) or []))
        _pairwise = {}
        _untested_pairs = []
        for _i, _k1 in enumerate(_knobs):
            for _k2 in _knobs[_i + 1:]:
                _obs = {(c[_knobs.index(_k1)], c[_knobs.index(_k2)]) for c in _combos}
                _want = {(a, b) for a in (_searched_by_knob.get(_k1) or [])
                         for b in (_searched_by_knob.get(_k2) or [])}
                _missing = sorted(_want - _obs, key=str)
                _pairwise[f'{_k1}|{_k2}'] = {
                    'searched_combinations': len(_obs),
                    'full_factorial_size': len(_want),
                    'full_pairwise': bool(_want) and not _missing,
                    'missing_combinations': [list(x) for x in _missing[:8]],
                    'n_missing': len(_missing),
                }
                if _missing:
                    _untested_pairs.append(f'{arm}:{_k1}|{_k2}')
        grid_factorial[arm] = {
            'knobs': _knobs,
            'searched_combinations': len(_combos),
            'full_factorial_size': int(_full_size),
            'coverage_fraction': (float(len(_combos)) / float(_full_size)
                                  if _full_size else None),
            'full_factorial': bool(_combos and len(_combos) == _full_size),
            'pairwise': _pairwise,
        }
        grid_coverage[arm] = per_knob
    _factor_pairs = sorted({f'{_arm}:{_p}' for _arm, _cv in grid_factorial.items()
                            for _p, _pw in _cv.get('pairwise', {}).items()
                            if _pw.get('full_pairwise') is False})
    if _factor_pairs:
        _declared_design = protocol.get('search_design')
        if _declared_design in ('one_factor_at_a_time', 'marginal'):
            warnings.append(
                'the declared marginal/OFAT search does not cover every pairwise '
                'interaction (' + ', '.join(_factor_pairs) + ') — its winner is '
                'a marginal optimum, not evidence of a jointly tuned optimum; '
                'add the missing interaction cells to support the stronger claim')
        else:
            warnings.append(
                'the tuning grid does not cover every pairwise interaction (' +
                ', '.join(_factor_pairs) + ') — a one-factor-at-a-time winner is '
                'not evidence of a jointly tuned optimum; either declare the search '
                'as marginal/OFAT or add the missing interaction cells before '
                'calling the baseline fully tuned')
    if _boundary_hits:
        warnings.append('selected hyperparameter(s) sit on the boundary of the declared search grid ('
                        + ', '.join(_boundary_hits) + ') — the optimum may lie outside the searched range; '
                        'extend the grid and re-run the equal search before calling this baseline tuned')
    if _coarse:
        warnings.append('the declared search grid is coarse: ' + '; '.join(_coarse)
                        + ' — a single level cannot show that the optimum is bracketed')
    _grid_adequacy = 'ok'
    if _boundary_hits or _coarse:
        _grid_adequacy = 'warn'
    if not _grid_levels:
        _grid_adequacy = 'incomplete'
        warnings.append('the tuning protocol declares no numeric trial levels, so grid adequacy is unverifiable')
    # The search design must be declared, because "OFAT" and "incomplete
    # factorial" look identical in the trial list but support different
    # claims.  A declared full factorial must actually be complete.
    _valid_search_designs = {'one_factor_at_a_time', 'marginal', 'full_factorial',
                             'seeded_random', 'hand_designed_grid'}
    _search_design = protocol.get('search_design')
    _search_design_note = protocol.get('search_design_note')
    if _search_design is None:
        warnings.append('the tuning protocol does not declare a search_design '
                        '(OFAT/full_factorial/seeded_random/hand_designed_grid); '
                        'interaction coverage cannot be interpreted as intended')
    elif not isinstance(_search_design, str) or _search_design not in _valid_search_designs:
        problems.append(f'search_design={_search_design!r} is not one of '
                        f'{sorted(_valid_search_designs)}')
    elif _search_design == 'full_factorial':
        _incomplete_factorial = sorted(
            a for a, g in grid_factorial.items()
            if not g.get('full_factorial'))
        if _incomplete_factorial or _factor_pairs:
            problems.append(
                'the protocol declares a full_factorial search, but arm(s) '
                f'{_incomplete_factorial or "?"} do not cover the complete '
                'cartesian grid (missing pairwise cells: '
                f'{_factor_pairs[:6]})')
    if _search_design_note is not None and (not isinstance(_search_design_note, str)
                                            or not _search_design_note.strip()):
        problems.append('search_design_note must be a non-empty string when present')
    out = {'status': 'ok' if not problems else 'fail', 'tuning_verified': not problems,
           'search_design': _search_design,
           'search_design_declared': _search_design is not None,
           'search_design_note': _search_design_note,
           'grid_levels': _grid_levels, 'grid_coverage': grid_coverage,
           'grid_factorial': grid_factorial,
           'grid_boundary_hits': _boundary_hits, 'grid_adequacy': _grid_adequacy,
           'grid_interaction_untested': _factor_pairs,
           'eval_n_tokens': eval_n_tokens,
           'selection_metric': metric, 'selection_split': split,
           'n_trials': n_trials, 'same_search_space': (len(set(signatures.values())) <= 1) if signatures else None,
           'search_seeds': search_seeds_seen if search_seeds_seen is not None else (_declared_ss and list(_declared_ss)),
           'search_seed_balanced': search_seed_balanced,
           'search_final_seed_overlap': _seed_overlap,
           'search_final_seed_disjoint': (not _seed_overlap) if (_search_set or _final_set) else None,
           'final_seed_coverage': final_seed_coverage,
           'final_seed_coverage_equal': final_seed_coverage_equal,
           'final_min_seed_count': final_min_seed_count,
           'selection_row_split_applied': _row_split_applied,
           'selection_rows_disjoint_from_evaluation': _row_split_applied,
           'nested_final_evaluation': _nested_final,
           'selection_row_split': (_split_info if isinstance(_split_info, dict) else None),
           'coverage': coverage, 'uncovered_arms': uncovered,
           'inherits_tuning': {str(k): str(v) for k, v in _inherits.items()},
           'selection_stability': selection_stabilities,
           'selection_stable_all': (all(s.get('stable') for s in selection_stabilities.values())
                                    if selection_stabilities else None),
           'min_final_seeds_for_family_significance': _final_family_floor,
           'final_family_size': int(_final_family_m),
           'final_family_basis': _final_family_basis,
           'final_family_contrasts': [[a, b] for a, b in _final_plan],
           'search_complete': _effective_search_complete,
           'budget_transfer': _budget_transfer,
           'data_transfer': {
               'trial_fingerprints': fingerprints_by_arm,
               'final_fingerprints': ({k: {a: sorted(v) for a, v in by.items()}
                                       for k, by in _final_fps_by_arm.items()}
                                      if final_records else {}),
           },
           'grid_complete': _grid_complete,
           'arms': sorted(arms), 'problems': problems, 'warnings': warnings,
           'selected_configs': {a: (e.get('selected') or e.get('selected_config'))
                                for a, e in arms.items() if isinstance(e, dict)}}
    if warnings and not problems:
        out['status'] = 'warn'
    # A verified equal-search artifact is not yet a "well-tuned baseline"
    # claim: the grid may be too narrow, the search may be truncated, the
    # selected config may have been tuned at another horizon/exposure, or the
    # pick may be seed-driven.  Expose the gate here so every caller (methods,
    # gates, reports) gets the same verdict instead of recomputing it.
    _claim_reasons = []
    if not out.get('tuning_verified'):
        _claim_reasons.append('equal-search tuning artifact not verified')
    if out.get('grid_adequacy') not in (None, 'ok'):
        _claim_reasons.append('grid_adequacy=%r' % (out.get('grid_adequacy'),))
    if out.get('search_complete') is False:
        _claim_reasons.append('search grid was truncated')
    _transfer = out.get('budget_transfer') or {}
    if isinstance(_transfer, dict) and _transfer.get('equal') is False:
        _claim_reasons.append('search-to-final training horizon differs')
    if isinstance(_transfer, dict) and _transfer.get('tokens_equal') is False:
        _claim_reasons.append('search-to-final token exposure differs')
    _stab = out.get('selection_stability') or {}
    if any(isinstance(v, dict) and v.get('status') == 'ok'
           and v.get('stable') is False for v in _stab.values()):
        _claim_reasons.append('selected config is not leave-one-search-seed-out stable')
    out['tuned_claim_allowed'] = not _claim_reasons
    out['tuned_claim_reasons'] = _claim_reasons
    return out

def _variant_seed_map(summary):
    """{(variant, protocol, seed): rec} for measurable, non-synthesized records
    with an unambiguous seed, plus the set of (variant, protocol, seed) triples
    that hold more than one record (ambiguous → excluded, first key wins).
    This is the single reader `_print_pairs` and `methodology_audit` share."""
    if not isinstance(summary, dict):
        # Callers feed this reader directly from JSON artifacts.  A scalar or
        # list has no record mapping; treat it as empty evidence rather than
        # crashing on `.items()`.
        return {}, set()
    _legacy_ok = legacy_warm_tags(summary)
    recs = {}
    amb = set()
    for _k, r in summary.items():
        if not (isinstance(r, dict) and ppl_is_usable(r.get('ppl')) and ('seed' in r) and (not r.get('synthesized'))):
            continue
        rk = _record_identity(r, _k, _legacy_ok)
        if rk is None or rk in amb:
            continue
        if rk in recs:
            amb.add(rk)
            recs.pop(rk, None)
            continue
        recs[rk] = r
    return recs, amb

def _paired_seed_deltas(recs, a, b, *, strict_hyper=False):
    """Per-protocol-condition per-seed (a − b) deltas over seeds where BOTH arms
    have a measurable record AND `pair_reason` confirms identical run_cfg /
    training length.

    Protocol conditions are matched exactly as `_print_pairs` does: equal
    non-empty tags pair together, and a tagged arm may pair against a pure-base
    ('') arm.  Two DIFFERENT non-empty conditions are never pooled — collapsing
    them (keying only on the seed) would fabricate seed counts and compare
    different measurement protocols (design_notes §3.3).

    With `strict_hyper=True` (module ablations), the structured
    `hyperparams` copy must also agree on every `_CONTROL_HYPER_KEYS` field
    (notably `mlp_ratio`), so a capacity change cannot ride along with the
    module toggle.  Baseline contrasts leave it off because their MLP width is
    intentionally matched via a different parameter count.

    Returns {(tag_a, tag_b): (deltas, shared_seeds, skipped)} with
    skipped = [(seed, reason)]."""
    ra = {(t, s): r for (v, t, s), r in recs.items() if v == a}
    rb = {(t, s): r for (v, t, s), r in recs.items() if v == b}
    _a_t = {t for t, _s in ra}
    _b_t = {t for t, _s in rb}
    _a_pure = _a_t == {''}
    _b_pure = _b_t == {''}
    conds = []
    if '' in _a_t and '' in _b_t:
        conds.append(('', ''))
    for tag in sorted(_a_t | _b_t):
        if not tag:
            continue
        if tag in _a_t and tag in _b_t:
            conds.append((tag, tag))
        if tag in _a_t and _b_pure:
            conds.append((tag, ''))
        if tag in _b_t and _a_pure:
            conds.append(('', tag))
    out = {}
    for ts, bs in conds:
        seeds = sorted({s for t, s in ra if t == ts} & {s for t, s in rb if t == bs})
        deltas, skipped = [], []
        _estimands = set()
        for s in seeds:
            reason = pair_reason(ra[ts, s], rb[bs, s])
            if reason is None and strict_hyper:
                reason = _hyperparam_reason(ra[ts, s], rb[bs, s])
            if reason is not None:
                skipped.append((s, reason))
                continue
            _delta = _paired_primary_delta(ra[ts, s], rb[bs, s])
            if _delta is None:
                skipped.append((s, 'primary estimand missing/mismatched (nested vs '
                                      'full-validation rows, or no holdout accounting)'))
                continue
            _estimands.add(
                ra[ts, s].get('ppl_estimation_rows') == 'nested_validation_holdout')
            deltas.append(_delta)
        if len(_estimands) > 1:
            # Each seed is internally paired, but the deltas come from two
            # different samples (holdout vs full validation).  Their mean/p
            # would describe neither estimand.
            out[(ts, bs)] = ([], seeds,
                             [(s, 'mixed estimands across seeds (some pairs are '
                                  'nested-holdout, others full-validation)')
                              for s in seeds])
            continue
        out[(ts, bs)] = (deltas, seeds, skipped)
    return out

# ---- 面板 look history：把「第几次分析」变成可复核的落盘记录 --------------
# 补种子 pass 会对同一个 outdir 再次聚合；每次 seed 集合变化就是一次新的
# look。这里把**当时**参与检验的逐种子配对差与种子集合落盘，使
# `sequential_signflip_p` 可以在事后复核「最终显著是不是靠不断加种子得到的」。
# 该文件只记录分析事件，不进入 resume/pairing/冻结判定。
LOOK_HISTORY_SCHEMA = 1
LOOK_HISTORY_MAX_LOOKS = 500


def _contrast_look_key(kind, comp, with_arm, without_arm, tag_with, tag_without):
    return '%s::%s::%s|%s|%s|%s' % (kind, comp, with_arm, without_arm,
                                    tag_with or '', tag_without or '')


def _interaction_look_key(name):
    """Stable look-history key for one declared 2x2 factorial.

    The interaction is recomputed from the four cells at each aggregation, so
    the key is the factorial identity.  If the cell arms or their protocol
    tags change, the per-seed interaction values change too; the sequential
    reader then treats the earlier look as stale and excludes it by name
    rather than silently mixing two estimands.
    """
    return 'interaction::%s' % name
