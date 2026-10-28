from __future__ import annotations


def data_quality_report(X, y=None, *, train_idx=None, val_idx=None, test_idx=None,
                        groups=None, time=None, clip_z=3.0, shift_threshold=0.5,
                        known_labels=None,
                        fail_on_nan_inf=True, fail_on_label_conflict=True,
                        fail_on_nonfinite_labels=False,
                        fail_on_label_shift=False,
                        fail_on_unknown_labels=False,
                        fail_on_duplicate_rows=False, fail_on_outliers=False,
                        fail_on_shift=False, target_outlier_z=None,
                        fail_on_target_outliers=False):
    """Report the common data-cleaning failure modes before any fit.

    ``train_idx`` identifies the training rows.  All outlier and shift
    statistics are computed from those rows only.  If a holdout index is
    supplied, ``train_idx`` is REQUIRED: silently treating the whole table as
    the training reference would let validation/test rows define the
    statistics that are later applied back to them.

    Returns a JSON-friendly dict.  ``status`` is ``'fail'`` when exact duplicate
    rows cross split boundaries (row-level leakage), groups cross splits, time
    runs backwards at a split boundary, or a hard data-quality policy fires
    (NaN/Inf and duplicate-row label conflicts by default).  Outliers,
    distribution shift and within-split duplicates are warnings unless the
    corresponding ``fail_on_*`` flag is enabled; callers that consciously
    accept them must record the waiver in their run metadata.

    ``time`` is an optional per-row ordering key (timestamp, event id, ...).
    When supplied, the report verifies the same monotone partition as
    :func:`assert_no_split_leakage`: every earlier split must be strictly
    older than the next split, so a caller that built indices elsewhere cannot
    silently hand in a shuffled time series.  Equal timestamps may not straddle
    a boundary (callers should put the whole event block in one split or pass
    ``groups``).

    Label diagnostics include per-split missing/non-finite counts, exact-
    duplicate label conflicts, a per-split label histogram, and the
    train→validation/test categorical label shift (Jensen-Shannon).  Non-finite
    labels and label shift are warnings by default; ``fail_on_nonfinite_labels``
    and ``fail_on_label_shift`` turn them into hard failures.  The training
    labels are the only reference distribution.  ``target_outlier_z`` enables a
    regression-target outlier audit whose fence is also learned from the
    training rows only (``fail_on_target_outliers`` enables the audit with
    ``clip_z`` and makes any flagged target a hard failure).

    ``known_labels`` optionally declares the admissible label set (a typo or a
    label seen only in a holdout is then reported per split).
    ``fail_on_unknown_labels`` turns that into a hard failure.
    """
    # A NaN/negative threshold silently disables the corresponding gate
    # (NaN compares False against everything), which is worse than an error.
    if isinstance(clip_z, (bool, np.bool_)):
        raise ValueError(f'clip_z must be a positive finite number, got {clip_z!r}')
    try:
        clip_z = float(clip_z)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'clip_z must be a positive finite number, got {clip_z!r}') from None
    if not math.isfinite(clip_z) or clip_z <= 0.0:
        raise ValueError(f'clip_z must be a positive finite number, got {clip_z!r}')
    if isinstance(shift_threshold, (bool, np.bool_)):
        raise ValueError(
            f'shift_threshold must be a finite non-negative number, got '
            f'{shift_threshold!r}')
    try:
        shift_threshold = float(shift_threshold)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'shift_threshold must be a finite non-negative number, got '
            f'{shift_threshold!r}') from None
    if not math.isfinite(shift_threshold) or shift_threshold < 0.0:
        raise ValueError(
            f'shift_threshold must be a finite non-negative number, got '
            f'{shift_threshold!r}')
    if fail_on_target_outliers and target_outlier_z is None:
        target_outlier_z = clip_z
    if target_outlier_z is not None:
        if isinstance(target_outlier_z, (bool, np.bool_)):
            raise ValueError(
                f'target_outlier_z must be a positive finite number or None, '
                f'got {target_outlier_z!r}')
        try:
            target_outlier_z = float(target_outlier_z)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'target_outlier_z must be a positive finite number or None, '
                f'got {target_outlier_z!r}') from None
        if not (math.isfinite(target_outlier_z) and target_outlier_z > 0.0):
            raise ValueError(
                f'target_outlier_z must be a positive finite number or None, '
                f'got {target_outlier_z!r}')
    if target_outlier_z is not None and y is None:
        raise ValueError(
            'data_quality_report: target_outlier_z/fail_on_target_outliers '
            'requires y so the target fence can be learned and audited; '
            'refusing to silently skip the requested target-outlier gate')
    # These flags are only meaningful with labels.  Without y the old path
    # simply never entered the label block, so a caller that explicitly asked
    # for a hard label gate got a silently disabled check.
    _label_gates = {
        'fail_on_label_shift': fail_on_label_shift,
        'fail_on_nonfinite_labels': fail_on_nonfinite_labels,
        'fail_on_unknown_labels': fail_on_unknown_labels,
    }
    _requested_label_gates = [k for k, v in _label_gates.items() if v]
    if y is None and _requested_label_gates:
        raise ValueError(
            'data_quality_report: ' + '/'.join(_requested_label_gates)
            + ' requires y so the requested label gate can actually run')
    # Keep the caller's dtype for duplicate keys: the float64 view used by the
    # statistics below cannot distinguish int64 values above 2**53.
    _raw_X = np.asarray(X)
    X = _as_float_2d(X)
    n = X.shape[0]
    known = None
    if known_labels is not None:
        if isinstance(known_labels, (str, bytes)):
            raise ValueError(
                'known_labels must be an iterable of labels, not a single string')
        try:
            values = (known_labels.reshape(-1).tolist()
                      if isinstance(known_labels, np.ndarray)
                      else list(known_labels))
        except TypeError:
            raise ValueError('known_labels must be a flat iterable of labels') from None
        known = {_hashable_label(_label_key(v)) for v in values}
        known.discard(None)
        if y is None:
            raise ValueError('known_labels requires y so the labels can be audited')
    idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for name, v in idx.items():
        if v is not None:
            v = _validated_index(v, name=f'{name}_idx')
            if v.size and v.max() >= n:
                raise ValueError(f'{name}_idx is out of bounds for n={n}')
            idx[name] = v
    if all(v is None for v in idx.values()):
        # No split supplied: use the whole input as a single training
        # reference and leave the holdouts empty.  This is a pre-split smoke
        # check, not permission to fit preprocessing on all data.
        split_rows = {'train': np.arange(n, dtype=np.int64),
                      'val': _empty_index(), 'test': _empty_index()}
    else:
        if idx['train'] is None and any(idx[k] is not None for k in ('val', 'test')):
            raise ValueError(
                'data_quality_report: train_idx is required when val_idx or '
                'test_idx is supplied — the holdout rows must never be used as '
                'the training reference')
        # Missing holdout indices mean "no rows on that side", never "all
        # rows": the latter would manufacture cross-split duplicates.
        split_rows = {name: (_empty_index() if idx[name] is None else idx[name])
                      for name in SPLIT_NAMES}
    ref = split_rows['train']
    if ref.size == 0 and (split_rows['val'].size or split_rows['test'].size):
        raise ValueError(
            'data_quality_report: the training reference is empty while '
            'validation/test rows were supplied; holdout rows must never be '
            'used as the reference that defines the statistics')
    warnings = []
    hard_failures = []
    checks = {}

    # The split indices themselves must be disjoint.  Cross-split duplicate
    # keys below only catch equal feature rows; an index repeated in two
    # splits is leakage even if the feature table happened to be mutated
    # between indexing operations.
    index_overlap = {}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            shared = np.intersect1d(split_rows[a], split_rows[b])
            index_overlap[f'{a}__{b}'] = int(shared.size)
            if shared.size:
                hard_failures.append(
                    f'{shared.size} row indices are assigned to both {a} and {b}')
    checks['split_index_overlap'] = index_overlap
    checks['duplicate_split_indices'] = {
        name: int(rows.size - np.unique(rows).size)
        for name, rows in split_rows.items()
    }
    if any(checks['duplicate_split_indices'].values()):
        warnings.append(f'duplicate row indices within splits: '
                        f"{checks['duplicate_split_indices']}")

    finite = np.isfinite(X)
    checks['nan_count'] = int(np.isnan(X).sum())
    checks['inf_count'] = int(np.isinf(X).sum())
    checks['nonfinite_by_split'] = {
        name: int((~finite[rows]).sum()) for name, rows in split_rows.items()
    }
    if checks['nan_count'] or checks['inf_count']:
        _msg = (f"{checks['nan_count']} NaN and {checks['inf_count']} Inf values; "
                'impute with training statistics before fitting')
        if fail_on_nan_inf:
            hard_failures.append(_msg)
        else:
            warnings.append(_msg)

    # Exact duplicates within each split (keyed on the original dtype so
    # large integers are not rounded through float64).  Build each split's key
    # set once and reuse it for the cross-split check below instead of making
    # a second copy of every row key.
    dup = {}
    keys = _row_keys(_raw_X)
    split_key_sets = {}
    for name, rows in split_rows.items():
        _keys = {keys[int(i)] for i in rows}
        split_key_sets[name] = _keys
        dup[name] = int(np.asarray(rows).size - len(_keys))
    checks['duplicate_rows_within_split'] = dup
    if any(dup.values()):
        _msg = f'duplicate rows within splits: {dup}'
        if fail_on_duplicate_rows:
            hard_failures.append(_msg)
        else:
            warnings.append(_msg)

    # Exact duplicate rows across splits: a hard leakage signal.
    cross = {}
    hard_leak = False
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        sa = split_key_sets[a]
        sb = split_key_sets[b]
        shared = len(sa & sb)
        cross[f'{a}__{b}'] = shared
        hard_leak = hard_leak or shared > 0
    checks['duplicate_rows_across_splits'] = cross
    if hard_leak:
        hard_failures.append(f'exact duplicate rows cross split boundaries: {cross}')

    # Group integrity.
    if groups is not None:
        groups = _as_1d(groups, 'groups', n)
        gsets = {name: {_group_key(groups[i]) for i in split_rows[name]}
                 for name in SPLIT_NAMES}
        group_cross = {}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            shared = gsets[a] & gsets[b]
            group_cross[f'{a}__{b}'] = len(shared)
            if shared:
                hard_leak = True
                hard_failures.append(
                    f'{len(shared)} group ids cross {a}/{b}; whole groups must '
                    'stay in one split')
        checks['group_overlap'] = group_cross

    # Temporal ordering.  A caller may build its own split indices; checking
    # only row/group overlap would accept a randomly shuffled time series.  The
    # comparison is the same strict partition contract as
    # assert_no_split_leakage (earlier max < later min), including equal
    # timestamps straddling a boundary.
    if time is not None:
        time = _as_1d(time, 'time', n)
        _assert_order_finite(time, name='time')
        temporal = {}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            ia = np.asarray(split_rows[a], dtype=np.int64).reshape(-1)
            ib = np.asarray(split_rows[b], dtype=np.int64).reshape(-1)
            if ia.size == 0 or ib.size == 0:
                temporal[f'{a}__{b}'] = {'checked': False}
                continue
            try:
                earlier_max = _order_extreme(time[ia], maximum=True)
                later_min = _order_extreme(time[ib], maximum=False)
                leaked = bool(earlier_max >= later_min)
            except (TypeError, ValueError, OverflowError):
                hard_leak = True
                temporal[f'{a}__{b}'] = {'checked': False,
                                         'error': 'time values are not mutually orderable'}
                hard_failures.append(
                    f'{a}/{b}: time values are not mutually orderable; a '
                    'temporal split cannot be verified')
                continue
            temporal[f'{a}__{b}'] = {
                'checked': True,
                'leak': leaked,
                'earlier_max': repr(earlier_max),
                'later_min': repr(later_min),
            }
            if leaked:
                hard_leak = True
                hard_failures.append(
                    f'temporal leakage: max {a} time {earlier_max!r} >= min '
                    f'{b} time {later_min!r}; the same event block must not '
                    'span two splits')
        checks['temporal_order'] = temporal

    # Label noise / corruption.  A target is only valid if it is finite; a
    # missing (None/NaN) target is reported separately from an Inf target,
    # and the counts are broken down by split so a corrupted holdout cannot
    # hide behind a clean overall total.
    if y is not None:
        y = _label_array(y).reshape(-1)
        if y.shape[0] != n:
            raise ValueError(f'y has length {y.shape[0]}, expected {n}')
        raw_labels = y.tolist()
        labels = [_label_key(v) for v in raw_labels]
        missing_labels = int(sum(v is None for v in labels))
        absent_labels = int(sum(v is None for v in raw_labels))
        nonfinite_labels = missing_labels - absent_labels
        checks['missing_labels'] = absent_labels
        checks['nonfinite_labels'] = nonfinite_labels
        checks['missing_labels_by_split'] = {}
        checks['nonfinite_labels_by_split'] = {}
        for name, rows in split_rows.items():
            _rows = np.asarray(rows, dtype=np.int64).reshape(-1).tolist()
            checks['missing_labels_by_split'][name] = int(
                sum(labels[i] is None and raw_labels[i] is None for i in _rows))
            checks['nonfinite_labels_by_split'][name] = int(
                sum(labels[i] is None and raw_labels[i] is not None for i in _rows))
        if absent_labels:
            warnings.append(f'{absent_labels} missing labels')
        if nonfinite_labels:
            _msg = (f'{nonfinite_labels} non-finite labels (NaN/Inf); impute '
                    'or remove them before fitting')
            if fail_on_nonfinite_labels:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        # Only duplicate keys can carry conflicting labels.  Building one
        # set() per row (as the old code did) costs hundreds of bytes per
        # distinct row even when the table has no duplicates at all; find the
        # duplicated keys first and collect labels only for those.
        _seen_keys = set()
        _dup_keys = set()
        for k in keys:
            if k in _seen_keys:
                _dup_keys.add(k)
            else:
                _seen_keys.add(k)
        del _seen_keys
        by_row = {}
        if _dup_keys:
            for i, (k, lab) in enumerate(zip(keys, labels)):
                if lab is not None and k in _dup_keys:
                    by_row.setdefault(k, set()).add(_hashable_label(lab))
        conflicts = [k for k, labs in by_row.items() if len(labs) > 1]
        checks['label_conflicts_on_duplicate_rows'] = len(conflicts)
        if conflicts:
            _msg = (f'{len(conflicts)} exact-duplicate feature rows carry '
                    'different labels (label noise / ambiguous annotation)')
            if fail_on_label_conflict:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        if known is not None:
            unknown_by_split = {}
            for name, rows in split_rows.items():
                counts = {}
                for i in np.asarray(rows, dtype=np.int64).reshape(-1).tolist():
                    lab = labels[i]
                    if lab is None:
                        continue
                    key = _hashable_label(lab)
                    if key not in known:
                        text = str(lab)
                        counts[text] = counts.get(text, 0) + 1
                unknown_by_split[name] = counts
            checks['unknown_labels_by_split'] = unknown_by_split
            _n_unknown = sum(sum(v.values()) for v in unknown_by_split.values())
            if _n_unknown:
                _msg = (f'{_n_unknown} labels are outside the declared '
                        f'known_labels set: {unknown_by_split}')
                if fail_on_unknown_labels:
                    hard_failures.append(_msg)
                else:
                    warnings.append(_msg)
        # Stringify before counting so mixed/object label types cannot make
        # np.unique raise (and so a numpy scalar and its Python twin count as
        # one category).  This is an audit histogram, not a model input.
        def _label_hist(values):
            hist = {}
            for v in values:
                if v is not None:
                    hist[str(v)] = hist.get(str(v), 0) + 1
            return {k: hist[k] for k in sorted(hist)}
        checks['label_distribution'] = _label_hist(labels)
        checks['label_distribution_by_split'] = {
            name: _label_hist([labels[i] for i in
                               np.asarray(rows, dtype=np.int64).reshape(-1).tolist()])
            for name, rows in split_rows.items()
        }
        # Train -> holdout label shift.  The training labels define the
        # reference distribution; validation/test labels are never pooled
        # into it.  String identities keep the report well-defined for mixed
        # categorical labels.
        label_shift = {}
        # Use the type-qualified label identity, not str(): integer 1 and the
        # string '1' are different classes, and stringifying both would hide a
        # real train->holdout label shift behind a shared spelling.
        _train_lab = [_hashable_label(labels[i]) for i in
                      np.asarray(ref, dtype=np.int64).reshape(-1).tolist()
                      if labels[i] is not None]
        for name in ('val', 'test'):
            _rows = np.asarray(split_rows[name], dtype=np.int64).reshape(-1).tolist()
            if not _rows:
                continue
            _hold_lab = [_hashable_label(labels[i]) for i in _rows
                         if labels[i] is not None]
            if not _train_lab or not _hold_lab:
                continue
            _sh = categorical_shift_report(_train_lab, _hold_lab)
            label_shift[name] = _sh
            if _sh['js_divergence'] > shift_threshold:
                _msg = (f'{name} label shift: JS divergence '
                        f"{_sh['js_divergence']:.3f} > {shift_threshold}")
                if fail_on_label_shift:
                    hard_failures.append(_msg)
                else:
                    warnings.append(_msg)
        checks['label_distribution_shift_vs_train'] = label_shift

        if target_outlier_z is not None:
            # Regression-target outliers, with the fence learned from the
            # training labels only.  Non-numeric labels (class names) are not
            # treated as numbers, so a categorical target is skipped rather
            # than being coerced through float().
            def _numeric_target(value):
                if value is None or isinstance(value, (bool, np.bool_)):
                    return None
                if isinstance(value, (complex, np.complexfloating)):
                    return None
                if _numeric_is_nan(value):
                    return None
                if not isinstance(value, (numbers.Number, _decimal.Decimal)):
                    return None
                try:
                    f = float(value)
                except (TypeError, ValueError, OverflowError):
                    return None
                return f if math.isfinite(f) else None

            _train_t = [_numeric_target(labels[i]) for i in
                        np.asarray(ref, dtype=np.int64).reshape(-1).tolist()]
            _train_t = [v for v in _train_t if v is not None]
            target_outliers = {name: 0 for name in SPLIT_NAMES}
            target_fence = None
            if len(_train_t) >= 2:
                _train_arr = np.asarray(_train_t, dtype=np.float64)
                q1, q3 = _safe_quantiles(_train_arr, (0.25, 0.75))
                with np.errstate(over='ignore', invalid='ignore'):
                    iqr = float(q3 - q1)
                if not math.isfinite(iqr) or iqr <= _EPS:
                    lo, hi = float(np.min(_train_arr)), float(np.max(_train_arr))
                else:
                    with np.errstate(over='ignore', invalid='ignore'):
                        lo = float(q1 - target_outlier_z * iqr)
                        hi = float(q3 + target_outlier_z * iqr)
                    if not math.isfinite(lo):
                        lo = float(np.min(_train_arr))
                    if not math.isfinite(hi):
                        hi = float(np.max(_train_arr))
                target_fence = {'lo': lo, 'hi': hi,
                                'z': float(target_outlier_z),
                                'n_train_reference': int(len(_train_t))}
                for name, rows in split_rows.items():
                    for i in np.asarray(rows, dtype=np.int64).reshape(-1).tolist():
                        v = _numeric_target(labels[i])
                        if v is not None and (v < lo or v > hi):
                            target_outliers[name] += 1
            checks['target_outliers_by_split'] = target_outliers
            checks['target_outlier_fence'] = target_fence
            checks['target_outlier_fence_estimated'] = bool(target_fence is not None)
            if target_fence is None:
                # The caller explicitly requested the target-outlier audit,
                # but no numeric training fence can be formed (fewer than two
                # finite numeric labels, or a categorical target).  Silently
                # reporting zero outliers would certify a gate that never ran.
                _msg = (f'the requested target-outlier audit is not estimable: '
                        f'only {len(_train_t)} finite numeric training label(s); '
                        'a fence needs at least 2')
                if fail_on_target_outliers:
                    hard_failures.append(_msg)
                else:
                    warnings.append(_msg)
            _n_target_out = sum(target_outliers.values())
            if _n_target_out:
                _msg = (f'{_n_target_out} target value(s) fall outside the '
                        f'train-only target fence {target_fence}: '
                        f'{target_outliers}')
                if fail_on_target_outliers:
                    hard_failures.append(_msg)
                else:
                    warnings.append(_msg)

    # Outliers, using training quantiles/IQR only.
    Xt = X[ref]
    finite_t = np.isfinite(Xt)
    out_counts = {name: 0 for name in SPLIT_NAMES}
    q_lo = np.zeros(X.shape[1])
    q_hi = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        vals = Xt[:, j][finite_t[:, j]]
        if vals.size == 0:
            q_lo[j], q_hi[j] = -np.inf, np.inf
            continue
        q1, q3 = _safe_quantiles(vals, (0.25, 0.75))
        with np.errstate(over='ignore', invalid='ignore'):
            iqr = q3 - q1
        if not math.isfinite(iqr) or iqr <= _EPS:
            q_lo[j], q_hi[j] = np.min(vals), np.max(vals)
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                q_lo[j], q_hi[j] = q1 - clip_z * iqr, q3 + clip_z * iqr
            if not math.isfinite(q_lo[j]):
                q_lo[j] = np.min(vals)
            if not math.isfinite(q_hi[j]):
                q_hi[j] = np.max(vals)
        for name, rows in split_rows.items():
            col = X[rows, j]
            good = np.isfinite(col)
            out_counts[name] += int(np.sum(
                good & ((col < q_lo[j]) | (col > q_hi[j]))))
    checks['outliers_by_split'] = out_counts
    if out_counts['train'] or out_counts['val'] or out_counts['test']:
        _msg = f'outliers (training IQR rule): {out_counts}'
        if fail_on_outliers:
            hard_failures.append(_msg)
        else:
            warnings.append(_msg)

    # Distribution shift: standardized mean difference and missing-rate delta
    # relative to training.  This is a descriptive audit, not a test.
    shift = {}
    for name in ('val', 'test'):
        rows = split_rows[name]
        if rows.size == 0:
            continue
        Xh = X[rows]
        smd = []
        non_estimable = []
        missing_reference = []
        for j in range(X.shape[1]):
            tv = Xt[:, j]
            hv = Xh[:, j]
            tv = tv[np.isfinite(tv)]
            hv = hv[np.isfinite(hv)]
            if tv.size == 0 or hv.size == 0:
                smd.append(None)
                missing_reference.append(j)
                continue
            sd = float(np.std(tv, ddof=0))
            mean_t = float(np.mean(tv))
            mean_h = float(np.mean(hv))
            if not (math.isfinite(sd) and math.isfinite(mean_t)
                    and math.isfinite(mean_h)):
                # Finite inputs can still overflow the variance/mean (values
                # near the float64 limit).  An infinite/nan SMD is not a clean
                # zero-shift result.
                smd.append(None)
                non_estimable.append(j)
                continue
            if sd <= _EPS:
                delta = abs(mean_h - mean_t)
                if delta <= _EPS:
                    smd.append(0.0)
                else:
                    # A constant training feature has no finite SMD, but that
                    # is not evidence of "no shift": the holdout mean is
                    # different by construction.  Dropping it from the maximum
                    # would report a material shift as clean.
                    smd.append(None)
                    non_estimable.append(j)
            else:
                value = float((mean_h - mean_t) / sd)
                if not math.isfinite(value):
                    smd.append(None)
                    non_estimable.append(j)
                else:
                    smd.append(value)
        miss_train = 1.0 - np.mean(np.isfinite(Xt), axis=0)
        miss_hold = 1.0 - np.mean(np.isfinite(Xh), axis=0)
        finite_smd = [abs(v) for v in smd if v is not None]
        # A zero-column table is a valid (if uninformative) input.  np.max on
        # the resulting empty vector would raise and abort the whole report.
        _missing_delta = (float(np.max(np.abs(miss_hold - miss_train)))
                          if miss_train.size else 0.0)
        shift[name] = {
            'max_abs_standardized_mean_difference': (
                max(finite_smd) if finite_smd else 0.0),
            'per_feature_standardized_mean_difference': smd,
            'non_estimable_shift_features': non_estimable,
            'features_without_reference_or_holdout_values': missing_reference,
            'max_abs_missing_rate_delta': _missing_delta,
        }
        if non_estimable:
            _msg = (
                f'{name} shift not estimable for {len(non_estimable)} '
                'zero-training-variance feature(s) whose holdout mean differs; '
                'SMD is undefined, not zero')
            if fail_on_shift:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        if missing_reference:
            _msg = (
                f'{name} shift not estimable for {len(missing_reference)} '
                'feature(s) with no finite training or holdout values')
            if fail_on_shift:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        if shift[name]['max_abs_standardized_mean_difference'] > shift_threshold:
            _msg = (
                f'{name} covariate shift: max |SMD| = '
                f"{shift[name]['max_abs_standardized_mean_difference']:.3f} "
                f'(threshold {shift_threshold})')
            if fail_on_shift:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        if shift[name]['max_abs_missing_rate_delta'] > shift_threshold:
            _msg = (
                f'{name} missingness shift: max |delta| = '
                f"{shift[name]['max_abs_missing_rate_delta']:.3f}")
            if fail_on_shift:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
    checks['distribution_shift_vs_train'] = shift

    status = ('fail' if (hard_leak or hard_failures)
              else ('warnings' if warnings else 'ok'))
    return {
        'status': status,
        'n_samples': int(n),
        'n_features': int(X.shape[1]),
        'n_train_reference': int(ref.size),
        'checks': checks,
        'warnings': warnings,
        'hard_failures': hard_failures,
        'note': 'statistics are learned from train_idx only; pass the split '
                'indices produced before any preprocessing',
    }


def categorical_shift_report(reference, holdout, *, max_categories=None):
    """Jensen-Shannon divergence between two categorical/token distributions.

    Used for corpora whose "features" are token ids rather than numeric
    columns.  ``reference`` must be the training split; the holdout is never
    used to define the vocabulary or the reference frequencies.  Returns
    ``{'js_divergence', 'total_variation', 'n_reference', 'n_holdout',
    'unseen_rate'}``.

    Categories are compared by their hashable identity rather than by a NumPy
    sort.  A mixed object column (e.g. a short string code and an integer code)
    has no single natural ordering in NumPy 2, but it still has well-defined
    category frequencies; sorting the identity keys on ``(type, repr)`` keeps
    the report computable and deterministic without conflating distinct values.
    """
    def _category_vector(values):
        # Keep each supplied item as ONE category.  ``np.asarray(seq,
        # dtype=object)`` expands a sequence of equal-length tuples/lists into
        # a 2-D array, so reshape(-1) would split a tuple category into its
        # components and silently change the distribution.  ndarray inputs
        # keep their existing flattening contract (a 2-D token table is a
        # table of categories); a Python sequence is materialised element by
        # element.
        if isinstance(values, np.ndarray):
            return values.reshape(-1)
        try:
            seq = list(values)
        except TypeError:
            seq = [values]
        out = np.empty(len(seq), dtype=object)
        for i, value in enumerate(seq):
            out[i] = value
        return out

    ref = _category_vector(reference)
    hold = _category_vector(holdout)
    if max_categories is not None:
        max_categories = _check_count('max_categories', max_categories, positive=True)
        if max_categories < 2:
            raise ValueError(
                'max_categories must be at least 2: a single bin would collapse '
                'the reference distribution and report zero divergence while '
                'every holdout category is unseen')
    if hold.size == 0:
        return {'js_divergence': 0.0, 'total_variation': 0.0,
                'n_reference': int(ref.size), 'n_holdout': 0,
                'unseen_rate': 0.0}
    if ref.size == 0:
        # With no reference sample there is no reference distribution, so the
        # JS divergence is undefined (the old path silently scored it as if
        # the reference were a degenerate one-bin histogram, returning log(2)/2
        # regardless of the holdout).  Every holdout category is unseen, and
        # the total variation from an empty reference is the fixed maximum
        # half-mass; keep that separately for callers that only gate on
        # unseen_rate.
        return {'js_divergence': None, 'total_variation': 0.5,
                'n_reference': 0, 'n_holdout': int(hold.size),
                'unseen_rate': 1.0, 'js_defined': False,
                'note': ('no reference observations; JS divergence is '
                         'undefined and must not be read as a measured shift')}

    def _key(value):
        # ``_target_key`` distinguishes a genuinely missing value from a real
        # tuple/string category spelled like the old sentinel, and stays
        # deterministic across processes (no address-bearing repr).
        return _target_key(value)

    def _sort_key(value):
        return (type(value).__name__, repr(value))

    ref_keys = [_key(v) for v in ref.tolist()]
    hold_keys = [_key(v) for v in hold.tolist()]
    counts = {}
    for key in ref_keys:
        counts[key] = counts.get(key, 0) + 1
    if max_categories is not None:
        if len(counts) > max_categories:
            # Keep the most frequent reference categories plus an "other" bin.
            # Count ties are broken by the stable identity key, not by NumPy's
            # unstable argsort of object values.
            ranked = sorted(counts, key=lambda k: (-counts[k], _sort_key(k)))
            keep = sorted(ranked[:max_categories - 1], key=_sort_key)
        else:
            keep = sorted(counts, key=_sort_key)
    else:
        keep = sorted(counts, key=_sort_key)
    keep_index = {key: i for i, key in enumerate(keep)}

    def _hist(keys):
        h = np.zeros(len(keep) + 1, dtype=np.float64)
        if keep:
            for key in keys:
                pos = keep_index.get(key)
                if pos is None:
                    h[-1] += 1.0
                else:
                    h[pos] += 1.0
        else:
            h[-1] = len(keys)
        return h / max(len(keys), 1)

    p = _hist(ref_keys)
    q = _hist(hold_keys)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], _EPS))))

    tv = 0.5 * float(np.sum(np.abs(p - q)))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if counts:
        # Seen means present ANYWHERE in the reference, not merely retained in
        # the truncated top-`max_categories` histogram bins.  Comparing against
        # keep_index counted a genuine reference category as unseen whenever
        # max_categories folded it into the "other" bin.
        _unseen_n = sum(1 for key in hold_keys if key not in counts)
        unseen = float(_unseen_n / len(hold_keys)) if hold_keys else 0.0
    else:
        # No reference category exists (e.g. an empty training side): every
        # holdout token is unseen, not zero.
        unseen = 1.0 if hold_keys else 0.0
    return {
        'js_divergence': js,
        'total_variation': tv,
        'n_reference': int(ref.size),
        'n_holdout': int(hold.size),
        'unseen_rate': unseen,
    }
