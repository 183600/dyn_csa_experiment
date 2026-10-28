
def simple_baseline_audit(artifact=None, path=None, protocol=None):
    """Audit the tuned count-model floor artifact.

    The artifact must show a validation-only search, a selected config that is
    actually the best recorded trial, and consistent evaluation-token counts.
    A missing/invalid artifact is `not_verifiable_locally`, never a pass."""
    _explicit = protocol is not None
    if artifact is None and path:
        if os.path.exists(path):
            try:
                artifact = json.load(open(path, encoding='utf-8'))
            except Exception as e:
                return {'status': 'fail', 'baseline_verified': False,
                        'problems': [f'cannot parse {path}: {type(e).__name__}: {e}']}
        else:
            return {'status': 'not_verifiable_locally', 'baseline_verified': False,
                    'problems': [f'{path} absent — run baseline_tune.py simple before claiming the trivial floor was tuned']}
    if not _explicit and isinstance(artifact, dict) and isinstance(artifact.get('protocol'), dict):
        protocol = artifact['protocol']
    else:
        protocol = SIMPLE_BASELINE_POLICY if protocol is None else protocol
    if not isinstance(protocol, dict):
        return {'status': 'fail', 'baseline_verified': False,
                'problems': [f'simple-baseline protocol must be an object, got '
                             f'{type(protocol).__name__}']}
    if not isinstance(artifact, dict) or not isinstance(artifact.get('baselines'), dict):
        return {'status': 'not_verifiable_locally', 'baseline_verified': False,
                'problems': ['no simple_baseline artifact (expected {"baselines": {...}})']}
    problems = []
    warnings = []
    # Keep explicit values (including an empty string/false) as evidence; only
    # an ABSENT key may inherit the declared protocol default.
    def _declared_field(key):
        return artifact[key] if key in artifact else protocol.get(key)
    metric = _declared_field('selection_metric')
    split = _declared_field('selection_split')
    direction = _declared_field('direction')
    if not isinstance(metric, str) or not metric.strip():
        # A mapping/list is not a metric name; `trial.get(metric)` would raise
        # on the unhashable value instead of reporting the malformed artifact.
        problems.append(f'selection metric {metric!r} is not a metric name')
        metric = None
    if split != 'validation':
        problems.append(f'selection split is {split!r}; count-model smoothing must be selected on validation, never test')
    _declared_split = protocol.get('selection_split')
    if isinstance(_declared_split, str) and split != _declared_split:
        problems.append(
            f'selection split {split!r} differs from the declared '
            f'{_declared_split!r}; the count-model floor is not audited against '
            'the pre-registered split')
    if metric != protocol.get('selection_metric'):
        problems.append(f'selection metric {metric!r} differs from the declared {protocol.get("selection_metric")!r}')
    if direction != 'lower_is_better':
        problems.append(f'count-model selection direction is {direction!r}; PPL is lower-is-better')
    _declared_direction = protocol.get('direction')
    if isinstance(_declared_direction, str) and direction != _declared_direction:
        problems.append(
            f'count-model selection direction {direction!r} differs from the '
            f'declared {_declared_direction!r}')
    _declared = protocol.get('arms')
    if _declared is None:
        _declared = {}
    elif not isinstance(_declared, dict):
        # A list/scalar cannot declare per-arm orders/grids.  Treating it as
        # an empty declaration silently skipped the missing/extra-arm gate.
        problems.append(
            f'simple-baseline protocol arms field is not an object '
            f'({type(_declared).__name__}); the declared floor arms are unknown')
        _declared = {}
    declared_arms = set(_declared.keys())
    baselines = artifact['baselines']
    missing = sorted(declared_arms - set(baselines))
    if missing:
        problems.append(f'missing simple-baseline evidence for arms {missing}')
    # An undeclared arm is not neutral: it can become `best_baseline` and be
    # quoted as the tuned floor even though it was outside the declared
    # equal-search design.  The artifact must describe exactly the arms the
    # protocol tuned (a superset is a different search, not more evidence).
    _extra_arms = sorted(set(baselines) - declared_arms) if declared_arms else []
    if _extra_arms:
        problems.append(
            f'baselines contain undeclared arm(s) {_extra_arms}; the quoted '
            'floor must come from the protocol\'s declared arms only')
    n_tokens_by_arm = {}
    selected_values = {}
    _search_design = protocol.get('search_design')
    _valid_search_designs = {'one_factor_at_a_time', 'marginal', 'full_factorial',
                             'seeded_random', 'hand_designed_grid'}
    if _search_design is None:
        warnings.append('the count-model protocol does not declare search_design; '
                        'grid completeness cannot be interpreted')
    elif not isinstance(_search_design, str) or _search_design not in _valid_search_designs:
        problems.append(f'search_design={_search_design!r} is not one of '
                        f'{sorted(_valid_search_designs)}')
    _grid_coverage = {}
    _grid_adequacy_by_arm = {}
    _grid_boundary_hits = []
    _selection_margins = {}
    _selection_tie_arms = []
    for arm, entry in baselines.items():
        if not isinstance(entry, dict):
            problems.append(f'{arm}: malformed baseline entry')
            continue
        trials = entry.get('trials')
        if not isinstance(trials, list) or not trials:
            problems.append(f'{arm}: no trial records')
            continue
        cfgs = [json.dumps(t.get('config') or {}, sort_keys=True, default=str) for t in trials if isinstance(t, dict)]
        # Only an ABSENT `selected` may fall back to the legacy alias; an
        # explicit null/blank value is evidence that the selection was lost.
        chosen = (entry['selected'] if 'selected' in entry
                  else entry.get('selected_config'))
        if not isinstance(chosen, dict):
            problems.append(f'{arm}: no selected config')
            continue
        chosen_sig = json.dumps(chosen, sort_keys=True, default=str)
        if chosen_sig not in cfgs:
            problems.append(f'{arm}: selected config is not one of its recorded trials')
            continue
        chosen_trial = next(t for t in trials if isinstance(t, dict)
                            and json.dumps(t.get('config') or {}, sort_keys=True, default=str) == chosen_sig)
        vals = [f for f in (_finite_scalar(t.get(metric)) for t in trials
                             if isinstance(t, dict)) if f is not None]
        if not vals:
            problems.append(f'{arm}: no finite {metric} in its trials')
            continue
        chosen_value = (entry['selected_metric_value']
                        if 'selected_metric_value' in entry
                        else chosen_trial.get(metric))
        chosen_f = _finite_scalar(chosen_value)
        if chosen_f is None:
            problems.append(f'{arm}: no finite selected {metric} value')
            continue
        selected_values[arm] = chosen_f
        _chosen_f = _finite_scalar(chosen_trial.get(metric))
        if _chosen_f is None:
            problems.append(f'{arm}: selected trial has no finite {metric}')
        elif abs(chosen_f - _chosen_f) > 1e-12:
            problems.append(f'{arm}: selected_metric_value does not match the selected trial')
        if chosen_f > min(vals) + 1e-12:
            problems.append(f'{arm}: selected trial is not the minimum-{metric} trial')
        # Declared search design: full factorial must actually be complete.
        # A corpus whose protocol does not declare this arm's grid stays
        # unverifiable (legacy artifacts), not silently "complete".
        _declared_arm = isinstance(_declared, dict) and arm in _declared
        _meta = _declared.get(arm) if _declared_arm else {}
        if _meta is not None and not isinstance(_meta, dict):
            problems.append(f'{arm}: declared arm metadata is not an object')
            _meta = {}
        _order = (_as_seed_int((_meta or {}).get('order', 1))
                  if _declared_arm else None)
        if _declared_arm and _order not in (1, 2):
            problems.append(
                f'{arm}: declared model order {(_meta or {}).get("order")!r} is '
                'not an integer 1 or 2')
        # The arm identity is a DISCRETE choice, not only part of a config
        # string.  An artifact whose per-arm order (or selected config order)
        # disagrees with the declared order ran a different count model.
        if _declared_arm and _order in (1, 2):
            _entry_order = _as_seed_int(entry.get('order'))
            if entry.get('order') is not None and _entry_order != _order:
                problems.append(
                    f'{arm}: entry order {entry.get("order")!r} differs from the '
                    f'declared order {_order}')
            if 'order' in chosen:
                _chosen_order = _as_seed_int(chosen.get('order'))
                if _chosen_order != _order:
                    problems.append(
                        f'{arm}: selected config order {chosen.get("order")!r} '
                        f'differs from the declared order {_order}')
        _grids = protocol.get('grids')
        _grid = _grids.get(arm) if isinstance(_grids, dict) else None
        if _grid is not None and not isinstance(_grid, dict):
            problems.append(f'{arm}: declared grid is not an object')
            _grid = {}
        _grid = _grid or {}
        _grid_declared = (isinstance(_grid, dict) and 'alpha' in _grid
                          and _order in (1, 2))

        def _grid_levels(raw, default):
            # A single declared level is legal; list/tuple values are the
            # normal grid spelling.  Anything else scalar is one level, not an
            # iterable (list(1) used to raise TypeError).
            if raw is None:
                return [default]
            if isinstance(raw, (list, tuple)):
                return list(raw)
            return [raw]

        _alphas = _grid_levels(_grid.get('alpha'), 0.1)
        _interps = (_grid_levels(_grid.get('interpolation'), 1.0)
                    if _order == 2 else [0.0])
        if _grid_declared:
            _expected = set()
            _bad_levels = []
            for _a in _alphas:
                for _lam in _interps:
                    try:
                        _a_f = float(_a)
                        _lam_f = float(_lam)
                        _cfg = {'order': _order, 'alpha': _a_f}
                        if _order == 2:
                            _cfg['interpolation'] = _lam_f
                    except (TypeError, ValueError, OverflowError):
                        _bad_levels.append((_a, _lam))
                        continue
                    _expected.add(json.dumps(_cfg, sort_keys=True, default=str))
            if _bad_levels:
                problems.append(
                    f'{arm}: declared grid contains non-numeric level(s) '
                    f'{_bad_levels[:4]}')
            _actual = set(cfgs)
            _missing_cells = sorted(_expected - _actual)
            _extra_cells = sorted(_actual - _expected)
            _grid_coverage[arm] = {'expected_cells': len(_expected),
                                   'observed_cells': len(_actual),
                                   'missing_cells': _missing_cells,
                                   'extra_cells': _extra_cells,
                                   'complete': not _missing_cells and not _extra_cells}
            # Completeness is not adequacy: a full grid can still select the
            # smallest/largest smoothing level it tested, in which case the
            # optimum may lie outside the searched range.  Mirror the neural
            # grid-adequacy gate so a count-model floor selected on a boundary
            # is not described as tuned.
            _alpha_levels = sorted({float(f) for f in
                                    (_finite_scalar(x) for x in _alphas)
                                    if f is not None})
            _sel_alpha = _finite_scalar(chosen.get('alpha'))
            if _sel_alpha is not None and len(_alpha_levels) >= 3:
                _pos = None
                for _i, _v in enumerate(_alpha_levels):
                    if abs(_v - _sel_alpha) <= 1e-12 * max(1.0, abs(_sel_alpha)):
                        _pos = _i
                        break
                if _pos in (0, len(_alpha_levels) - 1):
                    _boundary = ('min' if _pos == 0 else 'max')
                    _grid_boundary_hits.append(f'{arm}:alpha={_sel_alpha:g}({_boundary})')
                    warnings.append(
                        f'{arm}: selected alpha={_sel_alpha:g} sits on the {_boundary} of '
                        f'the declared range {_alpha_levels} — the smoothing optimum may '
                        'lie outside the searched grid; extend the grid before calling '
                        'this count-model floor tuned')
                _grid_adequacy_by_arm[arm] = {'alpha_levels': _alpha_levels,
                                              'selected_alpha': _sel_alpha,
                                              'selected_position': _pos,
                                              'at_boundary': _pos in (0, len(_alpha_levels) - 1)}
            elif _sel_alpha is not None:
                _grid_adequacy_by_arm[arm] = {
                    'alpha_levels': _alpha_levels, 'selected_alpha': _sel_alpha,
                    'selected_position': None, 'at_boundary': None,
                    'reason': ('fewer than three alpha levels cannot show that the '
                               'optimum is bracketed')}
                warnings.append(f'{arm}: the alpha grid has only {len(_alpha_levels)} '
                                'level(s); a single/two-point grid cannot show that the '
                                'smoothing optimum is bracketed')
            else:
                _grid_adequacy_by_arm[arm] = {'alpha_levels': _alpha_levels,
                                              'selected_alpha': None,
                                              'selected_position': None,
                                              'at_boundary': None}
            # The bigram interpolation weight is a second continuous tuning
            # knob.  Selecting 0 or 1 means the optimum may lie beyond the
            # tested range (the pure-unigram/pure-bigram corners), so it must
            # be visible just like an alpha boundary.
            if _order == 2:
                _interp_levels = sorted({float(f) for f in
                                         (_finite_scalar(x) for x in _interps)
                                         if f is not None})
                _sel_interp = _finite_scalar(chosen.get('interpolation'))
                _ipos = None
                if _sel_interp is not None:
                    for _i, _v in enumerate(_interp_levels):
                        if abs(_v - _sel_interp) <= 1e-12 * max(1.0, abs(_sel_interp)):
                            _ipos = _i
                            break
                    if len(_interp_levels) >= 3 and _ipos in (0, len(_interp_levels) - 1):
                        _iboundary = ('min' if _ipos == 0 else 'max')
                        _grid_boundary_hits.append(
                            f'{arm}:interpolation={_sel_interp:g}({_iboundary})')
                        warnings.append(
                            f'{arm}: selected interpolation={_sel_interp:g} sits on the '
                            f'{_iboundary} of the declared range {_interp_levels} — the '
                            'bigram/unigram mix optimum may lie outside the searched grid; '
                            'extend the grid before calling this count-model floor tuned')
                _grid_adequacy_by_arm.setdefault(arm, {}).update({
                    'interpolation_levels': _interp_levels,
                    'selected_interpolation': _sel_interp,
                    'interpolation_position': _ipos,
                    'interpolation_at_boundary': (bool(_ipos in (0, len(_interp_levels) - 1))
                                                  if _ipos is not None and len(_interp_levels) >= 3
                                                  else None)})
                if _sel_interp is not None and len(_interp_levels) < 3:
                    warnings.append(
                        f'{arm}: the interpolation grid has only {len(_interp_levels)} '
                        'level(s); the bigram/unigram mix optimum is not bracketed')
            # A winner that is indistinguishable from the runner-up is not a
            # tuned optimum.  Record the relative gap so the artifact shows
            # whether the selection was decisive.
            # Keep duplicate values: two different configs with the same PPL
            # are a tie even though the set of *values* has one element.
            _ranked_vals = sorted(float(v) for v in vals)
            if len(_ranked_vals) >= 2 and math.isfinite(_ranked_vals[0]) and _ranked_vals[0] > 0:
                _gap = float(_ranked_vals[1] - _ranked_vals[0])
                _rel = _gap / float(_ranked_vals[0])
                _selection_margins[arm] = {
                    'best': _ranked_vals[0], 'runner_up': _ranked_vals[1],
                    'absolute': _gap, 'relative': _rel,
                    'decisive_at_0.1pct': bool(_rel > 1e-3)}
                if _rel <= 1e-3:
                    _selection_tie_arms.append(arm)
                    warnings.append(
                        f'{arm}: the selected config beats the runner-up by only '
                        f'{_gap:.3g} {metric} ({100.0 * _rel:.3g}%) — treat the pick '
                        'as a tie, not as a uniquely tuned optimum')
            else:
                _selection_margins[arm] = {'best': (_ranked_vals[0]
                                                    if _ranked_vals else None),
                                           'runner_up': None, 'absolute': None,
                                           'relative': None, 'decisive_at_0.1pct': None}
            # A trial outside the declared search space is not an equal-search
            # result under ANY design, not only a full-factorial claim.
            if _extra_cells:
                problems.append(
                    f'{arm}: trials include {len(_extra_cells)} cell(s) outside '
                    f'the declared search grid ({_extra_cells[:4]})')
            if _search_design == 'full_factorial' and _missing_cells:
                problems.append(
                    f'{arm}: declares a full_factorial count-model search but '
                    f'{len(_missing_cells)} declared cell(s) were not run '
                    f'({_missing_cells[:4]})')
        else:
            _grid_coverage[arm] = {'expected_cells': None,
                                   'observed_cells': len(set(cfgs)),
                                   'missing_cells': None,
                                   'extra_cells': None,
                                   'complete': None,
                                   'declared': False}
            if _declared_arm and _search_design == 'full_factorial':
                problems.append(
                    f'{arm}: declares a full_factorial count-model search but '
                    'the protocol declares no alpha grid for this arm')
        nt = set()
        for t in trials:
            if not isinstance(t, dict) or t.get('n_tokens') is None:
                continue
            _nt = _as_seed_int(t.get('n_tokens'))
            if _nt is None:
                problems.append(f'{arm}: trial n_tokens is not an integer ({t.get("n_tokens")!r})')
                break
            nt.add(_nt)
        if len(nt) > 1:
            problems.append(f'{arm}: trials evaluated different token counts {sorted(nt)} — PPLs are not comparable')
        if not nt:
            problems.append(f'{arm}: no trial records its evaluation token count — '
                            'the common denominator is unattested')
        n_tokens_by_arm[arm] = sorted(nt)
        # Optional per-row accounting: when present it must describe the same
        # sample as the reported floor.  A mismatch means the floor contrast
        # would resample rows that were not scored.
        _rr_nll = entry.get('reported_nll_per_row')
        _rr_tok = entry.get('reported_n_tokens_per_row')
        if (_rr_nll is None) != (_rr_tok is None):
            problems.append(f'{arm}: only one of reported_nll_per_row/reported_n_tokens_per_row is present')
        elif _rr_nll is not None:
            if (not isinstance(_rr_nll, list) or not isinstance(_rr_tok, list)
                    or not _rr_nll or len(_rr_nll) != len(_rr_tok)):
                problems.append(f'{arm}: reported per-row accounting is malformed')
            else:
                _vals_nll = [_finite_scalar(v) for v in _rr_nll]
                _vals_tok = [_exact_count(v, minimum=0) for v in _rr_tok]
                if any(v is None for v in _vals_nll) or any(v is None for v in _vals_tok):
                    problems.append(f'{arm}: reported per-row NLL/token values are not finite exact numbers')
                else:
                    _marker = entry.get('reported_estimation_rows')
                    if (_marker is not None
                            and _marker not in ('nested_validation_holdout',
                                                'full_validation')):
                        problems.append(
                            f'{arm}: reported_estimation_rows={_marker!r} is not '
                            'one of nested_validation_holdout/full_validation; '
                            'the reported rows cannot be identified')
                    if _marker == 'nested_validation_holdout':
                        _want_tok = _exact_count(entry.get('holdout_n_tokens'), minimum=1)
                        _want_nll = _finite_scalar(entry.get('holdout_nll'))
                        # holdout_nll is the mean NLL per token, not a sum.
                        if _want_nll is not None and _want_tok is not None:
                            _want_nll = _want_nll * _want_tok
                    else:
                        _want_tok = _exact_count(chosen_trial.get('n_tokens'), minimum=1)
                        _tnll = _finite_scalar(chosen_trial.get('nll'))
                        _want_nll = (_tnll * _want_tok
                                     if _tnll is not None and _want_tok is not None else None)
                    if _want_tok is not None and int(sum(_vals_tok)) != _want_tok:
                        problems.append(f'{arm}: reported per-row token counts sum to '
                                        f'{int(sum(_vals_tok))}, not the reported {_want_tok}')
                    if _want_nll is not None and abs(float(sum(_vals_nll)) - float(_want_nll)) > 1e-9 * max(1.0, abs(float(_want_nll))):
                        problems.append(f'{arm}: reported per-row NLL sum does not match the reported floor NLL')
    if len({tuple(v) for v in n_tokens_by_arm.values()}) > 1:
        problems.append('simple baselines evaluated different token counts; report a common token-weighted floor')
    data = artifact.get('data')
    if not isinstance(data, dict):
        data = {}
    if not data.get('val_fp'):
        warnings.append('artifact has no validation fingerprint — the panel comparison cannot verify it used the same validation slice')
    _fit_policy = data.get('fit_token_policy')
    if _fit_policy is not None and _fit_policy != 'deterministic_prefix':
        problems.append(
            f'count-model fit_token_policy={_fit_policy!r} is not the implemented '
            'deterministic_prefix policy; the recorded fit budget is not reproducible')
    # The fit-budget disclosure is only trustworthy if the flag and the token
    # counts agree.  A stale/forged `fit_truncated=False` would otherwise let a
    # prefix-trained floor pass the truncation gate below.  Missing legacy
    # counts stay unverifiable; present counts are checked exactly.
    _fit_flag = data.get('fit_truncated')
    _fit_truncated = _fit_flag is True
    if _fit_flag is not None and not isinstance(_fit_flag, (bool, np.bool_)):
        problems.append(f'fit_truncated={_fit_flag!r} must be a boolean')
        _fit_flag = None
    _nfit = _as_seed_int(data.get('n_fit_tokens'))
    _nsrc = _as_seed_int(data.get('n_source_tokens'))
    if _nfit is not None or _nsrc is not None:
        if _nfit is None or _nsrc is None:
            problems.append('count-model artifact records only one of n_fit_tokens/n_source_tokens; '
                            'the fit budget cannot be verified')
        elif _nfit <= 0 or _nsrc <= 0 or _nfit > _nsrc:
            problems.append(f'count-model fit/source token counts are not ordered positive '
                            f'counts (n_fit_tokens={_nfit!r}, n_source_tokens={_nsrc!r})')
        else:
            _counts_truncated = bool(_nfit < _nsrc)
            _cap_raw = data.get('max_fit_tokens')
            if _cap_raw is not None:
                _cap_i = _as_seed_int(_cap_raw)
                if _cap_i is None or _cap_i <= 0:
                    problems.append(f'max_fit_tokens={_cap_raw!r} is not a positive integer')
                else:
                    _expected_fit = min(_nsrc, _cap_i)
                    if _nfit != _expected_fit:
                        problems.append(
                            f'n_fit_tokens={_nfit} is not min(n_source_tokens={_nsrc}, '
                            f'max_fit_tokens={_cap_i})={_expected_fit}; the recorded '
                            'prefix does not describe a deterministic-prefix fit')
            if isinstance(_fit_flag, (bool, np.bool_)):
                if bool(_fit_flag) != _counts_truncated:
                    problems.append(
                        f'fit_truncated={bool(_fit_flag)} contradicts the token counts '
                        f'({_nfit}/{_nsrc}); the fit-budget disclosure is not reproducible')
            elif _counts_truncated:
                warnings.append('the token counts show a truncated fit but fit_truncated '
                                'is absent; treating the floor as prefix-trained')
            _fit_truncated = bool(_counts_truncated or _fit_truncated)
            if _fit_truncated:
                warnings.append(f'the count-model floor was fitted on a deterministic prefix of '
                                f'{_nfit:,}/{_nsrc:,} training tokens (max_fit_tokens={data.get("max_fit_tokens")}); '
                                'report the fit budget when quoting the floor')
    elif _fit_truncated:
        warnings.append('count-model artifact claims a truncated fit but lacks integer fit/source token counts')
    # Nested floor evidence: selecting the smoothing AND reporting the floor
    # on the same rows gives the count model a look at the neural panel's
    # held-out rows.  When the artifact declares a nested split, verify that
    # the fold description is chronological/disjoint and reproducible, that
    # every arm reports a held-out value, and that `best_ppl` is the held-out
    # (not selection-fold) value of the selected arm.
    row_split = data.get('row_split') if isinstance(data, dict) else None
    nested_claimed = bool(isinstance(data, dict) and data.get('nested_selection') is True)
    nested_applied = None
    if nested_claimed or isinstance(row_split, dict):
        nested_applied = True
        if not isinstance(row_split, dict):
            problems.append('nested simple-baseline artifact lacks row_split metadata')
            row_split = {}
        if row_split.get('policy') not in (None, 'chronological_contiguous_row_folds'):
            problems.append(f'nested simple-baseline row policy is {row_split.get("policy")!r}, '
                            'not chronological_contiguous_row_folds')
        if row_split.get('chronological') is False:
            problems.append('nested simple-baseline split explicitly declares a non-chronological layout')
        if row_split.get('selection_precedes_evaluation') is False:
            problems.append('nested simple-baseline split evaluates earlier rows than it selects on')
        if row_split.get('disjoint') is False:
            problems.append('nested simple-baseline selection and evaluation folds overlap')
        _fold_keys = ('n_rows', 'seed', 'n_folds', 'selection_fold', 'evaluation_fold')
        _meta_keys = ('n_selection_rows', 'n_evaluation_rows',
                      'selection_row_range', 'evaluation_row_range',
                      'selection_row_fp', 'evaluation_row_fp',
                      'chronological', 'selection_precedes_evaluation',
                      'disjoint')
        if all(row_split.get(k) is not None for k in _fold_keys):
            _re_meta = None
            _groups_rec = row_split.get('groups')
            if row_split.get('groups_aware') is True:
                # A document-group-aware split cannot be recomputed from
                # (n_rows, seed, folds) alone: the cached document ids pick
                # the legal boundaries.  Recompute exactly when the group
                # vector was persisted; otherwise verify the recorded layout
                # is internally consistent instead of pretending the row-only
                # split describes it (that made every group-aware artifact
                # fail its own audit).
                _nrows_i = _as_seed_int(row_split.get('n_rows'))
                if (isinstance(_groups_rec, (list, tuple))
                        and _nrows_i is not None
                        and len(_groups_rec) == _nrows_i):
                    try:
                        _re_sel, _re_eval, _re_meta = validation_row_split(
                            _nrows_i, seed=row_split.get('seed'),
                            n_folds=row_split.get('n_folds'),
                            selection_fold=row_split.get('selection_fold'),
                            evaluation_fold=row_split.get('evaluation_fold'),
                            groups=_groups_rec)
                    except (TypeError, ValueError, OverflowError) as _rfe:
                        problems.append(f'nested simple-baseline row split cannot be reproduced: {_rfe}')
                else:
                    _n_folds_i = _as_seed_int(row_split.get('n_folds'))
                    _sel_fold_i = _as_seed_int(row_split.get('selection_fold'))
                    _ev_fold_i = _as_seed_int(row_split.get('evaluation_fold'))
                    _fr = row_split.get('fold_ranges')
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
                        problems.append('group-aware simple-baseline row_split lacks usable '
                                        'fold_ranges; the declared folds cannot be verified')
                    elif _nrows_i is None or _nrows_i < 2:
                        problems.append('group-aware simple-baseline row_split has an invalid n_rows')
                    elif (_sel_fold_i is None or _ev_fold_i is None
                          or not (0 <= _sel_fold_i < _n_folds_i)
                          or not (0 <= _ev_fold_i < _n_folds_i)
                          or _sel_fold_i >= _ev_fold_i):
                        problems.append('group-aware simple-baseline row_split has invalid '
                                        'fold indices (selection must precede evaluation)')
                    else:
                        _ranges = [(int(a), int(b)) for a, b in _fr]
                        _ranges_ok = (_ranges[0][0] == 0
                                      and _ranges[-1][1] == _nrows_i - 1
                                      and all(_ranges[i][1] + 1 == _ranges[i + 1][0]
                                              for i in range(len(_ranges) - 1))
                                      and all(a <= b for a, b in _ranges))
                        if not _ranges_ok:
                            problems.append('group-aware simple-baseline fold_ranges do not '
                                            'partition [0, n_rows) in order')
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
                                'chronological': True,
                                'selection_precedes_evaluation': True,
                                'disjoint': True,
                            }
                            _mismatch = [k for k in _meta_keys
                                         if k in row_split and row_split.get(k) != _want[k]]
                            if _mismatch:
                                problems.append('group-aware simple-baseline row_split metadata is '
                                                'internally inconsistent (mismatch: '
                                                f'{sorted(_mismatch)})')
            else:
                try:
                    _re_sel, _re_eval, _re_meta = validation_row_split(
                        row_split.get('n_rows'), seed=row_split.get('seed'),
                        n_folds=row_split.get('n_folds'),
                        selection_fold=row_split.get('selection_fold'),
                        evaluation_fold=row_split.get('evaluation_fold'))
                except (TypeError, ValueError, OverflowError) as _rfe:
                    problems.append(f'nested simple-baseline row split cannot be reproduced: {_rfe}')
            if _re_meta is not None:
                _mismatch = [k for k in _meta_keys
                             if k in row_split and row_split.get(k) != _re_meta.get(k)]
                if _mismatch:
                    problems.append('nested simple-baseline row_split metadata does not match the '
                                    f'deterministic split implied by n_rows/seed/folds (mismatch: {sorted(_mismatch)})')
        else:
            warnings.append('nested simple-baseline row_split omits part of the fold description '
                            '(n_rows/seed/n_folds/selection_fold/evaluation_fold); the recorded '
                            'ranges cannot be independently reproduced')
        # The floor must use the SAME declared fold layout as the neural
        # panel.  A reproducible but different split would tune the count
        # model on different (and possibly more favourable) selection rows.
        try:
            _declared_split = tuning_row_split_spec(protocol)
        except ValueError as _dse:
            problems.append(f'malformed simple-baseline selection_row_split: {_dse}')
            _declared_split = None
        if _declared_split is not None:
            if nested_applied is not True:
                warnings.append('the simple-baseline protocol declares a nested row split, but the '
                                'artifact reports full-validation selection — the floor may have '
                                'selected on the same documents the panel reports on')
            else:
                _diff = [k for k in ('seed', 'n_folds', 'selection_fold', 'evaluation_fold')
                         if row_split.get(k) is not None
                         and row_split.get(k) != _declared_split.get(k)]
                if _diff:
                    problems.append('nested simple-baseline row split differs from the declared '
                                    f'selection_row_split on {_diff} — the floor and the panel were '
                                    'tuned on different validation folds')
        _sel_fp = data.get('selection_fp')
        _eval_fp = data.get('eval_fp')
        if not _sel_fp or not _eval_fp:
            problems.append('nested simple-baseline artifact lacks selection/evaluation token fingerprints')
        elif _sel_fp == _eval_fp:
            problems.append('nested simple-baseline selection and evaluation token fingerprints are identical')
        # Per-arm accounting must describe the SAME sample as the floor the
        # artifact quotes.  A nested artifact whose rows are labelled
        # full_validation (or vice versa) is comparing different samples.
        _want_marker = 'nested_validation_holdout'
        for _arm, _entry in baselines.items():
            if not isinstance(_entry, dict):
                continue
            _marker = _entry.get('reported_estimation_rows')
            if _marker is not None and _marker != _want_marker:
                problems.append(
                    f'{_arm}: nested baseline reports per-row accounting for '
                    f'{_marker!r}, not the held-out fold')
        _hold_tok = set()
        for arm, entry in baselines.items():
            if not isinstance(entry, dict):
                continue
            _hv = _finite_scalar(entry.get('holdout_metric_value'))
            _ht = _exact_count(entry.get('holdout_n_tokens'), minimum=1)
            if _hv is None:
                problems.append(f'{arm}: nested baseline has no finite held-out {metric}')
                continue
            if entry.get('selected_metric_value') is None:
                problems.append(f'{arm}: nested baseline lacks its selection-fold metric')
            _chosen = entry.get('selected') or {}
            _sig = json.dumps(_chosen, sort_keys=True, default=str)
            _ct = next((t for t in (entry.get('trials') or []) if isinstance(t, dict)
                        and json.dumps(t.get('config') or {}, sort_keys=True, default=str) == _sig), None)
            if _ct is None or _finite_scalar(_ct.get('holdout_ppl')) is None:
                problems.append(f'{arm}: selected config has no held-out trial record')
            elif abs(float(_ct['holdout_ppl']) - _hv) > 1e-12:
                problems.append(f'{arm}: holdout_metric_value does not match the selected trial\'s held-out PPL')
            if _ht is not None:
                _hold_tok.add(_ht)
        if len(_hold_tok) > 1:
            problems.append(f'nested simple baselines evaluated different held-out token counts {sorted(_hold_tok)}')
        _n_eval = _exact_count(data.get('n_eval_tokens'), minimum=1)
        if _n_eval is not None and _hold_tok and _n_eval not in _hold_tok:
            problems.append('nested simple-baseline n_eval_tokens does not match its held-out token count')
    elif data:
        nested_applied = False
    # `best_ppl_estimation_rows` is the reader-facing statement of which
    # sample the quoted floor describes.  It must agree with the nested
    # evidence the artifact actually carries, not merely be a free-form label.
    _reported_rows = artifact.get('best_ppl_estimation_rows')
    if _reported_rows is not None and nested_applied is not None:
        _want_rows = ('nested_validation_holdout' if nested_applied is True
                      else 'full_validation')
        if _reported_rows != _want_rows:
            problems.append(
                f'best_ppl_estimation_rows={_reported_rows!r} does not match the '
                f'artifact\'s nested evidence ({_want_rows!r}); the quoted floor '
                'describes a different sample than the one that was scored')
    elif _reported_rows is not None:
        problems.append(
            f'best_ppl_estimation_rows={_reported_rows!r} cannot be verified: '
            'the artifact carries no nested/full-validation row evidence')
    best_name = None
    best_ppl_value = None
    if selected_values:
        best_name = min(selected_values, key=selected_values.get)
        # An explicitly empty/None `best_baseline` must not silently fall back
        # to the legacy alias: the floor identity is part of the claim.
        reported = (artifact['best_baseline'] if 'best_baseline' in artifact
                    else artifact.get('best_arm'))
        reported_ppl = artifact.get('best_ppl')
        if 'best_baseline' in artifact and reported is None:
            problems.append('best_baseline is present but null; the quoted floor '
                            'has no arm identity')
        if (reported is not None and isinstance(reported, str)
                and not reported.strip()):
            problems.append('best_baseline is present but blank; the quoted floor '
                            'has no arm identity')
        if reported is not None and reported != best_name:
            problems.append(f'best_baseline={reported!r} is not the minimum selected arm ({best_name!r})')
        _reported_f = _finite_scalar(reported_ppl)
        if 'best_ppl' in artifact and _reported_f is None:
            problems.append(
                f'best_ppl={reported_ppl!r} is present but not a finite number; '
                'the quoted floor cannot be verified')
        if nested_applied is True:
            _expected = _finite_scalar((baselines.get(best_name) or {}).get('holdout_metric_value'))
            if _expected is None:
                problems.append('nested artifact cannot derive the selected arm\'s held-out floor')
            else:
                best_ppl_value = _expected
                if _reported_f is None or abs(_reported_f - _expected) > 1e-12:
                    problems.append('nested best_ppl is not the held-out value of the minimum selection-fold arm')
        else:
            best_ppl_value = selected_values.get(best_name)
            if _reported_f is not None and abs(_reported_f - selected_values[best_name]) > 1e-12:
                problems.append('best_ppl does not match the minimum selected arm')
    _reported_nll = _reported_tok = None
    _reported_rows = None
    if best_name and isinstance(baselines.get(best_name), dict):
        _best_entry = baselines[best_name]
        _rn = _best_entry.get('reported_nll_per_row')
        _rt = _best_entry.get('reported_n_tokens_per_row')
        if isinstance(_rn, list) and isinstance(_rt, list) and len(_rn) == len(_rt):
            _reported_nll, _reported_tok = _rn, _rt
            _reported_rows = _best_entry.get('reported_estimation_rows')
    # Zero-information sanity floor: a count model must be better than sampling
    # tokens uniformly from the vocabulary.  If it is not, the tokenizer /
    # smoothing path is broken (or the reported floor is not the count model),
    # and calling it a baseline would understate every neural arm.
    _uniform_raw = artifact.get('uniform_ppl')
    _uniform = _finite_scalar(_uniform_raw)
    _vocab_d = _exact_count(data.get('vocab_size'), minimum=1) if isinstance(data, dict) else None
    if 'uniform_ppl' in artifact and _uniform is None:
        problems.append(
            f'uniform_ppl={_uniform_raw!r} is present but not a finite number; '
            'the zero-information floor cannot be verified')
    _reported_ratio = None
    if 'best_ppl_vs_uniform' in artifact:
        _ratio_raw = artifact.get('best_ppl_vs_uniform')
        _reported_ratio = _finite_scalar(_ratio_raw)
        if _reported_ratio is None:
            problems.append(
                f'best_ppl_vs_uniform={_ratio_raw!r} is present but not a finite '
                'number; the reported floor ratio cannot be verified')
    if _uniform is not None:
        if _uniform <= 0:
            problems.append(f'uniform_ppl={_uniform!r} must be positive')
        if _vocab_d is not None and abs(_uniform - _vocab_d) > 1e-9:
            problems.append(
                f'uniform_ppl={_uniform} is not the vocabulary size {_vocab_d}')
        _best_f = _finite_scalar(best_ppl_value)
        if (_best_f is not None and _uniform > 0
                and _best_f > _uniform * (1.0 + 1e-9)):
            # A tiny excess on a small/random evaluation sample can be noise,
            # so this is a warning rather than a hard failure.  A count model
            # on real text should be far below uniform; if it is not, check
            # the tokenizer/smoothing/data path before using it as a floor.
            warnings.append(
                f'the count-model floor (PPL={_best_f:.4f}) is WORSE than the '
                f'uniform-token floor (PPL={_uniform:.4f}); the floor is no better '
                'than zero-information (check smoothing/tokenization, or it is a '
                'signal-free sample)')
        if (_reported_ratio is not None and _best_f is not None
                and _uniform > 0
                and abs(_reported_ratio - _best_f / _uniform) > 1e-9):
            problems.append(
                f'best_ppl_vs_uniform={_reported_ratio} is not best_ppl/uniform_ppl '
                f'({_best_f / _uniform})')
    out = {'status': 'ok' if not problems else 'fail',
           'baseline_verified': not problems,
           'uniform_ppl': _uniform,
           'best_ppl_vs_uniform': (float(best_ppl_value) / _uniform
                                   if _uniform and _finite_scalar(best_ppl_value) is not None
                                   else None),
           'search_design': _search_design,
           'search_design_declared': _search_design is not None,
           'search_design_note': protocol.get('search_design_note'),
           'grid_coverage': _grid_coverage,
           'grid_adequacy_by_arm': _grid_adequacy_by_arm,
           'grid_boundary_hits': sorted(_grid_boundary_hits),
           'grid_adequacy': ('incomplete' if any(
               not (v or {}).get('complete') for v in _grid_coverage.values())
               else ('warn' if _grid_boundary_hits or _selection_tie_arms else 'ok')),
           'selection_margins': _selection_margins,
           'selection_tie_arms': sorted(_selection_tie_arms),
           'selection_metric': metric, 'selection_split': split,
           'arms': sorted(baselines), 'selected_values': selected_values,
           'best_baseline': best_name,
           'best_ppl': best_ppl_value,
           'reported_nll_per_row': _reported_nll,
           'reported_n_tokens_per_row': _reported_tok,
           'reported_estimation_rows': _reported_rows,
           'best_ppl_estimation_rows': ('nested_validation_holdout' if nested_applied
                                        else ('full_validation' if nested_applied is False else None)),
           'nested_selection': nested_applied,
           'row_split': row_split,
           'n_eval_tokens': n_tokens_by_arm,
           'fit_truncated': _fit_truncated,
           'fit_budget': {
               'fit_token_policy': data.get('fit_token_policy'),
               'max_fit_tokens': data.get('max_fit_tokens'),
               'n_fit_tokens': data.get('n_fit_tokens'),
               'n_source_tokens': data.get('n_source_tokens'),
           },
           'data': data, 'problems': problems, 'warnings': warnings}
    if warnings and not problems:
        out['status'] = 'warn'
    # Structural verification is not adequacy: a boundary/tied smoothing pick
    # or a prefix-truncated fit must not be quoted as a well-tuned floor.
    _floor_reasons = []
    if not out.get('baseline_verified'):
        _floor_reasons.append('count-model artifact not verified')
    if out.get('grid_adequacy') not in (None, 'ok'):
        _floor_reasons.append('grid_adequacy=%r' % (out.get('grid_adequacy'),))
    if out.get('grid_boundary_hits'):
        _floor_reasons.append('selected smoothing is on a grid boundary')
    if out.get('selection_tie_arms'):
        _floor_reasons.append('selected smoothing ties the runner-up')
    if out.get('fit_truncated') is True:
        _floor_reasons.append('count model was fitted on a truncated prefix')
    out['tuned_floor_claim_allowed'] = not _floor_reasons
    out['tuned_floor_claim_reasons'] = _floor_reasons
    return out

# 指标口径表：每个任务的主指标、方向、不平衡处理与聚合方式都显式写出。
# `metric_policy_audit` 用它检查落盘记录，避免「只报 accuracy」「F1 没写
# macro/micro/weighted」「极度不平衡仍以 ROC-AUC 为主」这类不可复核的读数。
METRIC_POLICY = {
    'lm_perplexity': {
        'task': 'language_modeling',
        'primary': 'ppl',
        'direction': 'lower_is_better',
        'secondary': ['nll', 'next_token_accuracy'],
        'imbalance': 'not_applicable',
        'aggregation': 'token_weighted',
        'note': 'PPL is the primary endpoint; next-token accuracy is diagnostic only and must never replace it.',
    },
    'boundary_alignment': {
        'task': 'binary_boundary_detection',
        'primary': 'f1',
        'primary_average': 'macro_over_sequences',
        'direction': 'higher_is_better',
        'secondary': ['precision', 'recall', 'random_precision', 'precision_excess_over_random'],
        'imbalance': 'positives_are_a_small_fraction_of_cut_positions',
        'aggregation': 'macro_over_sequences',
        'note': 'Never report cut accuracy: the negative class dominates. Always report precision, recall, F1 and the matched-length random baseline together.',
    },
    'niah_retrieval': {
        'task': 'multi_class_retrieval',
        'primary': 'top1_accuracy',
        # When one answer class is at or beyond 1:1000 (prevalence <= 1e-3),
        # top-1 accuracy is dominated by the frequent answers.  Macro recall (balanced accuracy)
        # is the recall-aligned conditional endpoint; PR-AUC is deliberately
        # NOT used because a many-way retrieval record may carry only top-1
        # labels, and inventing one-vs-rest scores from them is not valid.
        'primary_when_extreme_imbalance': 'macro_recall',
        'extreme_imbalance_threshold': 1e-3,
        'direction': 'higher_is_better',
        'secondary': ['answer_nll', 'balanced_accuracy', 'macro_recall',
                      'f1_macro', 'f1_micro', 'f1_weighted',
                      'chance_accuracy', 'majority_accuracy',
                      'acc_lift_over_chance', 'acc_lift_over_majority',
                      'minority_prevalence', 'imbalance_ratio'],
        'chance': '1/n_observed_answer_classes (uniform); majority-answer floor also required',
        'imbalance': ('top-1 micro accuracy is dominated by frequent answer classes; '
                      'report macro recall and the uniform/majority floors alongside it. '
                      'When a class is at or beyond 1:1000 (prevalence <= 1e-3) the primary endpoint switches '
                      'to macro recall (balanced accuracy).'),
        'aggregation': 'query_unweighted',
        'note': ('Top-1 accuracy is reported with answer NLL, macro recall (balanced accuracy), '
                 'the trivial uniform/majority floors and explicitly named macro/micro/weighted '
                 'F1. Micro F1 equals top-1 accuracy for a single-label endpoint; macro treats '
                 'every observed answer class equally and weighted follows support, so the '
                 'averaging rule must accompany every F1 number. PR-AUC/ROC-AUC are not '
                 'substitutes for a many-way retrieval endpoint and must not be invented from '
                 'top-1 labels; a record without per-class scores leaves them unverifiable. '
                 'Under extreme answer-class imbalance, macro recall is the conditional primary.'),
    },
    'binary_classification': {
        'task': 'binary_classification',
        'primary': 'f1',
        'primary_average': 'binary_positive_class',
        # Under extreme imbalance the declared primary SWITCHES to PR-AUC:
        # ROC-AUC stays visually high because the negative base rate makes FPR
        # grow slowly, and F1 is threshold-dependent.  This is a conditional
        # primary, not merely a preference in prose.
        'primary_when_extreme_imbalance': 'pr_auc',
        'extreme_imbalance_threshold': 1e-3,
        'direction': 'higher_is_better',
        'secondary': ['precision', 'recall', 'pr_auc', 'roc_auc', 'balanced_accuracy',
                      'brier_score', 'ece'],
        'imbalance': 'report prevalence; when positives are at or beyond 1:1000 (prevalence <= 1e-3) the primary endpoint is PR-AUC, not F1 or ROC-AUC',
        'aggregation': 'binary',
        'decision_threshold': 0.5,
        'decision_threshold_policy': ('Report the cut with every F1/precision/recall/balanced '
                                      'accuracy. The default 0.5 is a reporting convention, not a '
                                      'tuned optimum; a tuned cut must be selected on validation '
                                      'rows disjoint from evaluation and recorded as '
                                      'decision_threshold_source="validation_selection" with its '
                                      'split and metric. PR-AUC/ROC-AUC/Brier/ECE are threshold-free.'),
        'note': 'F1 must state the averaging rule; for binary tasks it is the positive-class F1. For prevalence <= 1e-3 (1:1000 or more imbalanced), PR-AUC is the primary ranking endpoint. Calibration (Brier/ECE) is secondary: a good ranking must not be mistaken for usable probabilities.',
    },
    'multiclass_classification': {
        'task': 'multiclass_classification',
        'primary': 'f1_macro',
        'primary_average': 'macro',
        'primary_when_extreme_imbalance': 'pr_auc_macro',
        'extreme_imbalance_threshold': 1e-3,
        'direction': 'higher_is_better',
        'secondary': ['f1_micro', 'f1_weighted', 'balanced_accuracy', 'per_class_recall',
                      'macro_micro_f1_gap', 'pr_auc_macro', 'pr_auc_micro',
                      'pr_auc_weighted', 'pr_auc_macro_chance',
                      'pr_auc_macro_lift_over_chance', 'pr_auc_micro_chance',
                      'pr_auc_micro_lift_over_chance', 'roc_auc_macro',
                      'roc_auc_micro', 'roc_auc_weighted', 'brier_score', 'ece'],
        'imbalance': ('macro averaging is required when class frequencies differ; micro equals '
                      'accuracy for single-label tasks; when a class is at or beyond 1:1000 (prevalence <= 1e-3) the '
                      'primary ranking endpoint switches to one-vs-rest macro PR-AUC'),
        'aggregation': 'macro',
        'ranking': 'one-vs-rest macro/micro/weighted PR-AUC and ROC-AUC; the caller must name the averaging rule',
        'note': ('Every reported F1 must name macro, micro or weighted; report per-class recall '
                 'when macro and micro disagree. F1 is threshold-dependent, so a rare-class '
                 'panel must also report threshold-free one-vs-rest PR-AUC. Macro treats every '
                 'class equally, micro is dominated by frequent classes, and weighted follows '
                 'the support: they can rank models differently. A row-normalized '
                 'probability matrix is also scored for multiclass Brier and top-label ECE; '
                 'logits stay uncalibrated rather than being treated as probabilities.'),
    },
}

_HYPOTHESIS_REQUIRED = ('statement', 'mechanism', 'predicted_sign', 'primary_metric',
                        'requires_param_match', 'falsifier')


def _extreme_imbalance(prevalence, threshold=1e-3):
    """Is the minority-class prevalence AT OR BEYOND the declared boundary?

    The project policy says the conditional endpoint switches at "1:1000 or
    more imbalanced".  In prevalence terms a 1:1000 endpoint is exactly 1e-3,
    so the comparison must be inclusive: a strict ``<`` silently keeps the
    usual F1/accuracy headline for a panel that is already at the documented
    extreme.  Returns ``True``/``False``, or ``None`` when either argument is
    not a usable probability/threshold.
    """
    prev = _finite_scalar(prevalence)
    thr = _finite_scalar(threshold)
    if prev is None or thr is None:
        return None
    if not (0.0 <= prev <= 1.0) or not (0.0 < thr <= 1.0):
        return None
    return bool(prev <= thr)


def hypothesis_coverage_audit(design=None, baselines=None,
                              component_table=None, baseline_table=None):
    """Design-point coverage of the pre-registered hypothesis tables.

    `hypothesis_audit` checks that a hypothesis which EXISTS is complete and
    metric-aligned.  It cannot notice an experiment that was added without a
    hypothesis, because the table and the design are separate declarations.
    This audit closes that gap: every declared contrast family must have an
    entry, and every entry must correspond to a declared family (an orphan
    hypothesis describes an experiment the design no longer contains).

    Both directions are reported.  A missing hypothesis is a pre-registration
    defect; an orphan is a stale declaration.  Neither is silently defaulted.
    """
    design = ABLATION_DESIGN if design is None else design
    baselines = BASELINE_DESIGN if baselines is None else baselines
    component_table = (HYPOTHESIS_DESIGN if component_table is None
                       else component_table)
    baseline_table = (BASELINE_HYPOTHESIS_DESIGN if baseline_table is None
                      else baseline_table)
    if not isinstance(design, dict) or not isinstance(baselines, dict):
        return {'kind': 'hypothesis_coverage', 'status': 'fail',
                'problems': ['design and baselines must be mappings'],
                'warnings': []}
    if not isinstance(component_table, dict) or not isinstance(baseline_table, dict):
        return {'kind': 'hypothesis_coverage', 'status': 'fail',
                'problems': ['hypothesis tables must be mappings'],
                'warnings': []}
    out = {'kind': 'hypothesis_coverage', 'status': 'ok',
           'component': {}, 'baseline': {}, 'warnings': []}

    def _cover(label, declared, table):
        declared_keys = sorted(declared)
        table_keys = sorted(table)
        missing = [k for k in declared_keys if k not in table]
        orphan = [k for k in table_keys if k not in declared]
        empty = [k for k in declared_keys
                 if not isinstance(declared.get(k), (list, tuple))
                 or not declared.get(k)]
        out[label] = {'declared': declared_keys,
                      'with_hypothesis': table_keys,
                      'missing': missing, 'orphan': orphan,
                      'empty_contrast': empty,
                      'complete': not (missing or orphan or empty)}
        if missing:
            out['warnings'].append(
                f'{label}: declared contrast family/families {missing} have no '
                'pre-registered hypothesis — a result from them cannot be '
                'interpreted against an expected mechanism or direction')
        if orphan:
            out['warnings'].append(
                f'{label}: hypothesis entry/entries {orphan} do not correspond '
                'to a declared contrast family — the declaration has drifted '
                'from the experiment design')
        if empty:
            out['warnings'].append(
                f'{label}: declared family/families {empty} are empty — there '
                'is nothing for the hypothesis to be tested on')

    _cover('component', design, component_table)
    _cover('baseline', baselines, baseline_table)
    if out['warnings']:
        out['status'] = 'warn'
    return out


def _metric_policy_roles(metric, policy):
    """Return (policies where `metric` is primary, policies where it is secondary)."""
    primary, secondary = [], []
    if not isinstance(metric, str) or not isinstance(policy, dict):
        return primary, secondary
    for name, p in policy.items():
        if not isinstance(p, dict):
            continue
        if p.get('primary') == metric:
            primary.append(name)
        _alt = p.get('primary_when_extreme_imbalance')
        if isinstance(_alt, str) and _alt == metric:
            primary.append(name)
        elif isinstance(_alt, (list, tuple)) and metric in _alt:
            primary.append(name)
        sec = p.get('secondary')
        if isinstance(sec, str):
            sec = [sec]
        if isinstance(sec, (list, tuple)) and metric in sec:
            secondary.append(name)
    return primary, secondary


def hypothesis_audit(table=None, kind='component', names=None, policy=None):
    """Pre-registration completeness / metric-alignment audit.

    A hypothesis is only actionable before an experiment if it states the
    expected mechanism, the signed direction of the primary metric, and a
    falsifier.  This audit also checks that the declared primary metric is one
    the project's ``METRIC_POLICY`` actually defines, so a hypothesis cannot
    silently target an unmeasured quantity (e.g. accuracy for an imbalanced
    endpoint, or ROC-AUC for a 1:1000 task).  A metric that appears only in a
    policy's **secondary** list (for example ROC-AUC under
    ``binary_classification``) is rejected as the primary endpoint unless the
    hypothesis explicitly labels it ``metric_role='secondary'`` and gives a
    non-empty ``metric_justification``.  An optional ``metric_policy`` key
    binds the hypothesis to one named policy and requires the metric to belong
    to that policy.  `names` limits the audit to the declared comparisons in
    the current panel; missing table entries are reported by the caller as
    `hypotheses_missing`.

    Direction `predicted_sign=0` is legal for an exploratory "difference
    exists" hypothesis, but it can never support a directional superiority
    claim; such components are listed separately as direction_unspecified."""
    if table is None:
        table = HYPOTHESIS_DESIGN if kind == 'component' else BASELINE_HYPOTHESIS_DESIGN
    policy = METRIC_POLICY if policy is None else policy
    wanted = set(names) if names is not None else set(table)
    out = {'status': 'ok', 'kind': kind, 'hypotheses': {},
           'invalid': [], 'metric_unknown': [], 'metric_misaligned': [],
           'metric_secondary_justified': [], 'direction_unspecified': [],
           'warnings': []}
    if not isinstance(policy, dict):
        # A policy is a declared mapping; a list/scalar cannot be audited.
        out['warnings'].append(
            f'metric policy is a {type(policy).__name__}, not an object; '
            'the hypothesis metric contract cannot be checked')
        policy = {}
    allowed = set()
    for _pname, p in policy.items():
        if not isinstance(p, dict):
            # One malformed entry must not abort the rest of the audit.
            out['warnings'].append(
                f'metric policy `{_pname}` is a {type(p).__name__}, not an object')
            continue
        for key in ('primary', 'secondary', 'primary_when_extreme_imbalance'):
            val = p.get(key)
            if isinstance(val, str):
                allowed.add(val)
            elif isinstance(val, (list, tuple)):
                allowed.update(str(x) for x in val)
    for name in sorted(wanted):
        spec = table.get(name)
        if not isinstance(spec, dict):
            out['invalid'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` is missing from the pre-registered table')
            continue
        issues = []
        for key in _HYPOTHESIS_REQUIRED:
            val = spec.get(key)
            if key == 'predicted_sign':
                if isinstance(val, (bool, np.bool_)) or val not in (-1, 0, 1):
                    issues.append(f'predicted_sign={val!r} (must be -1, 0 or 1)')
            elif key == 'requires_param_match':
                # The claim must say up front whether equal capacity is part of
                # what it tests.  A Python int is not a declaration.
                if not isinstance(val, bool):
                    issues.append(f'requires_param_match={val!r} (must be True or False)')
            elif not isinstance(val, str) or not val.strip():
                issues.append(f'{key} is empty or not a string')
        pm = spec.get('primary_metric')
        metric_known = bool(isinstance(pm, str) and pm in allowed)
        primary_policies, secondary_policies = _metric_policy_roles(pm, policy)
        metric_role = spec.get('metric_role', 'primary')
        metric_policy = spec.get('metric_policy')
        metric_justification = spec.get('metric_justification')
        role_issue = None
        if not isinstance(metric_role, str) or metric_role not in ('primary', 'secondary'):
            role_issue = f'metric_role={metric_role!r} must be "primary" or "secondary"'
        if metric_policy is not None:
            if not isinstance(metric_policy, str) or metric_policy not in policy:
                role_issue = (f'metric_policy={metric_policy!r} is not a declared '
                              'METRIC_POLICY entry')
            elif not isinstance(policy.get(metric_policy), dict):
                role_issue = (f'metric_policy={metric_policy!r} is not a '
                              'metric-policy object')
            elif metric_known:
                _p = policy.get(metric_policy)
                _sec = _p.get('secondary')
                if isinstance(_sec, str):
                    _sec = [_sec]
                _alt = _p.get('primary_when_extreme_imbalance')
                if isinstance(_alt, str):
                    _alt = [_alt]
                _in_primary = ((_p.get('primary') == pm)
                               or (isinstance(_alt, (list, tuple)) and pm in _alt))
                _in_secondary = (isinstance(_sec, (list, tuple)) and pm in _sec)
                if not (_in_primary or _in_secondary):
                    role_issue = (f'primary_metric={pm!r} is not part of the '
                                  f'declared metric_policy {metric_policy!r}')
                elif metric_role == 'primary' and not _in_primary:
                    role_issue = (f'primary_metric={pm!r} is only a secondary '
                                  f'endpoint of {metric_policy!r}; a hypothesis '
                                  'must name the policy primary or declare '
                                  'metric_role="secondary" with a justification')
        if role_issue is None and metric_known:
            if metric_role == 'primary' and not primary_policies:
                role_issue = (f'primary_metric={pm!r} is only a SECONDARY endpoint '
                              f'in policy {sorted(secondary_policies)}; use the '
                              'declared primary metric, or declare '
                              'metric_role="secondary" with a justification')
            elif metric_role == 'secondary':
                if not secondary_policies:
                    role_issue = (f'metric_role="secondary" but {pm!r} is a primary '
                                  'endpoint (or not defined) in METRIC_POLICY')
                elif not isinstance(metric_justification, str) or not metric_justification.strip():
                    role_issue = ('metric_role="secondary" requires a non-empty '
                                  'metric_justification explaining why the policy '
                                  'primary is not the hypothesis endpoint')
        # Optional observed prevalence lets the hypothesis declare the
        # policy's CONDITIONAL endpoint up front.  A hypothesis written for
        # F1 on a 1:1000 endpoint is not aligned with the metric contract,
        # even though `f1` is the policy's normal primary.
        _prevalence = spec.get('prevalence')
        _prev_issue = None
        _prev_value = None
        if _prevalence is not None:
            _prev_value = _finite_scalar(_prevalence)
            if _prev_value is None or not (0.0 <= _prev_value <= 1.0):
                _prev_issue = (f'prevalence={_prevalence!r} must be a finite '
                               'number in [0, 1]')
            else:
                _bound = [metric_policy] if metric_policy else list(primary_policies)
                for _pn in _bound:
                    _pol = policy.get(_pn)
                    if not isinstance(_pol, dict):
                        continue
                    _cond = _pol.get('primary_when_extreme_imbalance')
                    _thr = _finite_scalar(_pol.get('extreme_imbalance_threshold'))
                    if (_cond and _thr is not None
                            and _extreme_imbalance(_prev_value, _thr)
                            and metric_role == 'primary' and pm != _cond):
                        _prev_issue = (
                            f'prevalence={_prev_value:.3e} is at or beyond the '
                            f'{_pn} extreme-imbalance threshold {_thr:g}; the '
                            f'policy primary switches to {_cond!r}, not {pm!r} '
                            '(or declare metric_role="secondary" with a justification)')
                        break
        if _prev_issue is not None and role_issue is None:
            role_issue = _prev_issue
        pred = spec.get('predicted_sign')
        row = {'statement': spec.get('statement'), 'mechanism': spec.get('mechanism'),
               'predicted_sign': pred, 'primary_metric': pm,
               'prevalence': _prev_value,
               'metric_role': metric_role, 'metric_policy': metric_policy,
               'metric_primary_policies': sorted(primary_policies),
               'metric_secondary_policies': sorted(secondary_policies),
               'metric_justification': metric_justification,
               'requires_param_match': spec.get('requires_param_match'),
               'falsifier': spec.get('falsifier'),
               'direction_specified': (pred in (-1, 1)),
               'metric_declared_in_policy': metric_known,
               'complete': not issues and role_issue is None}
        out['hypotheses'][name] = row
        if issues:
            out['invalid'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` is incomplete: ' + '; '.join(issues))
        if not metric_known:
            out['metric_unknown'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` declares primary_metric={pm!r}, which METRIC_POLICY does not define')
        if role_issue is not None:
            if name not in out['invalid']:
                out['invalid'].append(name)
            out['metric_misaligned'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` has a metric-role mismatch: {role_issue}')
        elif metric_known and metric_role == 'secondary':
            out['metric_secondary_justified'].append(name)
            _policy_primaries = sorted({policy[p].get('primary') for p in secondary_policies
                                        if isinstance(policy.get(p), dict)})
            if metric_policy:
                _bound = policy.get(metric_policy)
                if isinstance(_bound, dict) and _bound.get('primary'):
                    _policy_primaries = [_bound.get('primary')]
            out['warnings'].append(
                f'{kind} hypothesis `{name}` uses secondary endpoint {pm!r} as its '
                'test metric (justified); report it together with the policy primary '
                f'{_policy_primaries}')
        if pred == 0:
            out['direction_unspecified'].append(name)
            out['warnings'].append(f'{kind} hypothesis `{name}` has predicted_sign=0 — it can test for a difference but cannot support a directional advantage claim')
    out['status'] = 'warn' if out['warnings'] else 'ok'
    return out

def _binary_roc_auc(y, score):
    """Rank-based ROC-AUC for a binary endpoint (ties get average ranks)."""
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(score.size, dtype=float)
    sorted_score = score[order]
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def _binary_average_precision(y, score):
    """Average precision (the step-wise PR-AUC) for a binary endpoint."""
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score, kind='mergesort')
    ys, ss = y[order], score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.nonzero(np.diff(ss))[0], ss.size - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _preserve_label_array(values):
    """Materialise labels without coercing mixed Python types.

    ``np.asarray([1, '1'])`` silently turns the integer into the string '1',
    merging two distinct classes before any audit can see them.  ndarray inputs
    keep their declared dtype; plain sequences become object arrays.
    """
    if isinstance(values, np.ndarray):
        return values.ravel()
    try:
        return np.asarray(values, dtype=object).ravel()
    except (TypeError, ValueError, OverflowError):
        return np.asarray(values).ravel()


def _unique_label_values(values):
    """Unique label identities, falling back when NumPy cannot sort them.

    The common homogeneous path is unchanged (``np.unique``).  Mixed object
    labels (e.g. ``1`` and ``'1'``) have no total NumPy order, but they are
    still distinct classes; the fallback keeps each identity once in a stable
    type/repr order.  NumPy also compares some temporal pairs inconsistently
    (a Python datetime can equal an ``np.datetime64`` without NumPy's unique
    collapsing them), so the canonical label identity is applied after
    ``np.unique`` as well.
    """
    arr = values if isinstance(values, np.ndarray) else _preserve_label_array(values)
    try:
        uniq = np.unique(arr)
    except (TypeError, ValueError, OverflowError):
        uniq = None
    if uniq is not None:
        seen = {}
        for value in uniq:
            seen.setdefault(_hashable_label(value), value)
        if len(seen) == int(uniq.size):
            return uniq
        return np.asarray(list(seen.values()), dtype=object)
    seen = {}
    for value in arr.reshape(-1).tolist():
        seen.setdefault(_hashable_label(value), value)
    try:
        ordered = sorted(seen.values(), key=lambda v: (type(v).__name__, repr(v)))
    except (TypeError, ValueError, OverflowError):
        ordered = list(seen.values())
    return np.asarray(ordered, dtype=object)


def _display_label_names(labels):
    """Unique human-readable names for a declared label set.

    ``str()`` is the historical key, but distinct label identities can share a
    spelling (the integer ``1`` and the string ``'1'``).  A report keyed by
    ``str`` would then overwrite one class and double-count the other.  Keep
    the historical spelling when it is unambiguous; otherwise fall back to
    ``repr`` and, if even that collides, to positional suffixes so every
    declared class keeps exactly one row.
    """
    values = labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
    names = [str(v) for v in values]
    if len(set(names)) == len(names):
        return names
    names = [repr(v) for v in values]
    if len(set(names)) == len(names):
        return names
    return [f'{name}#{i}' for i, name in enumerate(names)]


def niah_copy_rule_baseline(ids, query_cols, answer_ids, *, target):
    """A deterministic copy-rule floor for the NIAH retrieval task.

    The NIAH-natural task plants a ``key -> value`` couple in the far context
    and repeats the key in the target zone; the correct next token is the
    value.  A rule system that needs no training can therefore look up the
    nearest PREVIOUS occurrence of the query key in the causal prefix and copy
    the token that follows it.  This is the task's simplest lexical baseline:
    if a learned indexer cannot match it, the mechanism is not even doing the
    copy operation the task is built from.  Conversely, a model that only
    barely beats this rule has not demonstrated a semantic retrieval gain.

    ``ids`` has shape ``[sequences, eval_len + 1]`` (the final column is the
    gold answer), ``query_cols`` gives the query-key column relative to the
    last ``target`` input positions, and ``answer_ids`` holds the gold answer
    tokens.  A query with no earlier occurrence of its key is counted wrong
    and reported as ``n_missing_context``.  The returned top-1 metrics include
    the same uniform/majority floors as ``retrieval_metrics``.
    """
    arr = np.asarray(ids)
    qc = np.asarray(query_cols)
    av = np.asarray(answer_ids)
    if arr.ndim != 2:
        raise ValueError('niah_copy_rule_baseline: ids must be 2-D (sequences x tokens)')
    if qc.shape != av.shape or qc.ndim != 2:
        raise ValueError('niah_copy_rule_baseline: query_cols and answer_ids must be '
                         'matching 2-D arrays')
    _target = _as_seed_int(target)
    if _target is None or _target < 1:
        raise ValueError('niah_copy_rule_baseline: target must be a positive integer')
    n_seq, n_tokens = arr.shape
    if qc.shape[0] != n_seq:
        raise ValueError('niah_copy_rule_baseline: ids and query_cols disagree on sequence count')
    eval_len = int(n_tokens) - 1
    if int(_target) > eval_len:
        raise ValueError(
            f'niah_copy_rule_baseline: target={int(_target)} exceeds the '
            f'{eval_len} scored input position(s); the query offset would be '
            'outside the evaluated sequence')
    tail_start = eval_len - int(_target)
    preds = []
    missing = 0
    for i in range(int(n_seq)):
        row = arr[i]
        for j in range(qc.shape[1]):
            _col = _as_seed_int(qc[i, j])
            if _col is None or _col < 0 or _col >= int(_target):
                preds.append(-1)
                missing += 1
                continue
            key_pos = tail_start + int(_col)
            if key_pos <= 0 or key_pos >= eval_len:
                preds.append(-1)
                missing += 1
                continue
            key = row[key_pos]
            prev = np.nonzero(row[:key_pos] == key)[0]
            if prev.size == 0:
                preds.append(-1)
                missing += 1
            else:
                preds.append(int(row[int(prev[-1]) + 1]))
    _n = int(av.size)
    preds_arr = np.asarray(preds, dtype=np.int64)
    _rm = retrieval_metrics(av.reshape(-1), preds_arr)
    out = dict(_rm)
    out.update({'rule': 'nearest_previous_key_copy',
                'n': _n, 'n_missing_context': int(missing),
                'target': int(_target),
                'note': ('Untuned lexical floor: copy the token after the nearest earlier '
                         'occurrence of the query key. Retrieval above this floor is the '
                         'minimum evidence that a learned selection mechanism adds value.')})
    return out
