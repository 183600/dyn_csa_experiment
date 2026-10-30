from __future__ import annotations


def clean_training_rows(X, y=None, *, train_idx, drop_duplicates=True,
                        label_conflicts='drop', drop_invalid_labels=False,
                        drop_nonfinite_features=False, outlier_iqr=None,
                        target_outlier_iqr=None):
    """Remove duplicate/ambiguous rows from the training split only.

    The holdout indices are not accepted by this function, so no validation or
    test row can be dropped, imputed or used to decide which duplicate to keep.
    Rows are grouped by their exact feature key; the first occurrence in the
    supplied training order is retained.  Categorical/object feature tables
    are supported for duplicate and label cleaning; the two numeric feature
    options below (``drop_nonfinite_features`` / ``outlier_iqr``) require a
    real-valued table.  When duplicate rows carry different
    non-missing labels, ``label_conflicts='drop'`` removes the whole ambiguous
    cluster (the safe choice before fitting), ``'keep'`` retains it and
    ``'raise'`` refuses to continue.  ``drop_invalid_labels=True`` also removes
    training rows whose target is missing or non-finite (a NaN/Inf target
    cannot supervise a fit); the decision uses training labels only.

    ``drop_nonfinite_features=True`` removes training rows with a NaN/Inf in
    any feature instead of relying on imputation, and ``outlier_iqr`` (a
    positive multiplier, e.g. ``3.0``) removes training rows whose finite
    values fall outside ``Q1 - k*IQR`` / ``Q3 + k*IQR`` for any feature.  Both
    bounds are computed from the training rows of this call only.  These are
    cleaning decisions, so they are applied before any preprocessor fit; a
    holdout row is never inspected and can never decide which training row to
    drop.

    ``target_outlier_iqr`` applies the same train-only IQR fence to numeric
    training labels (label-noise cleanup).  It requires ``y``; non-numeric
    (categorical) labels are left untouched, and the fence is learned only
    from the finite training labels that survived the earlier drop rules.
    Holdout labels are never inspected, so a corrupted validation/test target
    cannot widen or move the fence used for the training split.

    Returns ``train_idx`` (the cleaned index array), the dropped rows and
    counts suitable for a run manifest.
    """
    _raw_X = np.asarray(X)
    Xraw = _as_numeric_2d(X)
    n = int(Xraw.shape[0])
    # Duplicate/label cleaning needs only the exact row identity, which works
    # for categorical/object tables too.  Convert to float64 only when a
    # numeric feature audit was actually requested, so a categorical feature
    # table is not rejected just because its duplicate labels need cleaning.
    Xarr = None
    train_rows = _validated_index(train_idx, name='train_idx')
    if train_rows.size == 0:
        raise ValueError('clean_training_rows: the training split is empty')
    if int(train_rows.max()) >= n:
        raise ValueError(
            f'clean_training_rows: train_idx reaches {int(train_rows.max())} '
            f'but X has {n} rows')
    if np.unique(train_rows).size != train_rows.size:
        # A repeated training index would let one row enter the drop mask and
        # the duplicate counts more than once, so a manifest could not say how
        # many physical rows survived.  Split helpers never emit duplicates;
        # an external index that does is a caller bug, not a cleaning choice.
        raise ValueError(
            'clean_training_rows: train_idx contains duplicate row indices; '
            'every training row must appear at most once')
    if label_conflicts not in ('drop', 'keep', 'raise'):
        raise ValueError(
            f"label_conflicts must be 'drop', 'keep' or 'raise', "
            f"got {label_conflicts!r}")
    for _flag_name, _flag in (('drop_duplicates', drop_duplicates),
                              ('drop_invalid_labels', drop_invalid_labels),
                              ('drop_nonfinite_features', drop_nonfinite_features)):
        if not isinstance(_flag, (bool, np.bool_)):
            raise ValueError(
                f'{_flag_name} must be a boolean, got {_flag!r}')
    if y is None:
        # These options can only inspect/remove labels.  Without y the
        # requested cleaning step silently did nothing, so a caller could
        # believe ambiguous/missing training targets were removed when the
        # split was untouched.  Refuse the un-runnable request (the
        # target-outlier fence below already fails closed for the same
        # reason).
        if drop_invalid_labels:
            raise ValueError(
                'clean_training_rows: drop_invalid_labels=True requires y so '
                'missing/non-finite training labels can be identified')
        if label_conflicts != 'drop':
            raise ValueError(
                f'clean_training_rows: label_conflicts={label_conflicts!r} '
                'requires y so duplicate-row label conflicts can be detected')
    if outlier_iqr is not None:
        if isinstance(outlier_iqr, (bool, np.bool_)):
            raise ValueError(
                f'outlier_iqr must be a positive finite number or None, '
                f'got {outlier_iqr!r}')
        try:
            outlier_iqr = float(outlier_iqr)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'outlier_iqr must be a positive finite number or None, '
                f'got {outlier_iqr!r}') from None
        if not (math.isfinite(outlier_iqr) and outlier_iqr > 0.0):
            raise ValueError(
                f'outlier_iqr must be a positive finite number or None, '
                f'got {outlier_iqr!r}')
    if target_outlier_iqr is not None:
        if isinstance(target_outlier_iqr, (bool, np.bool_)):
            raise ValueError(
                f'target_outlier_iqr must be a positive finite number or None, '
                f'got {target_outlier_iqr!r}')
        try:
            target_outlier_iqr = float(target_outlier_iqr)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'target_outlier_iqr must be a positive finite number or None, '
                f'got {target_outlier_iqr!r}') from None
        if not (math.isfinite(target_outlier_iqr) and target_outlier_iqr > 0.0):
            raise ValueError(
                f'target_outlier_iqr must be a positive finite number or None, '
                f'got {target_outlier_iqr!r}')
    if drop_nonfinite_features or outlier_iqr is not None:
        Xarr = _as_float_2d(Xraw, name='X')
    labels = None
    if y is not None:
        yarr = _label_array(y)
        if yarr.ndim == 0 or yarr.shape[0] != n:
            raise ValueError(
                f'clean_training_rows: y has '
                f'{yarr.shape[0] if yarr.ndim else 1} rows, expected {n}')
        train_rows_needed = np.unique(train_rows).tolist()
        if yarr.ndim == 1:
            labels = {int(i): _label_key(yarr[i])
                      for i in train_rows_needed}
        else:
            rows = yarr.reshape(n, -1)
            labels = {int(i): tuple(_label_key(v) for v in rows[i].tolist())
                      for i in train_rows_needed}
    if target_outlier_iqr is not None and labels is None:
        raise ValueError(
            'clean_training_rows: target_outlier_iqr requires y so the '
            'target fence can be learned from training labels only')

    def _missing(lab):
        return _is_missing_scalar(lab) or (isinstance(lab, tuple)
                                           and all(_is_missing_scalar(v)
                                                   for v in lab))

    drop = np.zeros(train_rows.size, dtype=bool)
    n_invalid_label_rows = 0
    if drop_invalid_labels and labels is not None:
        for pos, row in enumerate(train_rows.tolist()):
            lab = labels[int(row)]
            if _is_missing_scalar(lab) or (isinstance(lab, tuple)
                                           and any(_is_missing_scalar(v)
                                                   for v in lab)):
                drop[pos] = True
                n_invalid_label_rows += 1
    n_nonfinite_feature_rows = 0
    if drop_nonfinite_features:
        _finite = np.isfinite(Xarr[train_rows]).all(axis=1)
        _bad = ~_finite
        n_nonfinite_feature_rows = int((_bad & ~drop).sum())
        drop |= _bad
    n_outlier_rows = 0
    if outlier_iqr is not None:
        Xt = Xarr[train_rows]
        _bad_cells = np.zeros(Xt.shape, dtype=bool)
        for j in range(Xt.shape[1]):
            col = Xt[:, j]
            finite = np.isfinite(col)
            vals = col[finite]
            if vals.size == 0:
                continue
            q1, q3 = _safe_quantiles(vals, (0.25, 0.75))
            with np.errstate(over='ignore', invalid='ignore'):
                iqr = float(q3 - q1)
            if not math.isfinite(iqr) or iqr <= _EPS:
                # A representable IQR is needed for a finite fence.  For an
                # unrepresentable range fall back to the observed min/max,
                # which cannot manufacture an outlier.
                lo, hi = float(np.min(vals)), float(np.max(vals))
            else:
                with np.errstate(over='ignore', invalid='ignore'):
                    lo = float(q1 - outlier_iqr * iqr)
                    hi = float(q3 + outlier_iqr * iqr)
                # A fence beyond the float64 range is represented by the
                # observed extreme: it drops nothing, rather than emitting a
                # non-finite fence into the run manifest.
                if not math.isfinite(lo):
                    lo = float(np.min(vals))
                if not math.isfinite(hi):
                    hi = float(np.max(vals))
            _bad_cells[:, j] = finite & ((col < lo) | (col > hi))
        _row_bad = _bad_cells.any(axis=1)
        n_outlier_rows = int((_row_bad & ~drop).sum())
        drop |= _row_bad
    n_target_outlier_rows = 0
    target_outlier_fence = None
    n_target_reference_labels = 0
    if target_outlier_iqr is not None:
        def _numeric_label(value):
            # Only scalar numeric labels define a regression target.  A
            # categorical string, a tuple (multi-output) or a missing/Inf
            # target has no place on a single target fence.  Decimal is an
            # exact real numeric target even though it is not a
            # numbers.Number subclass; its NaN spellings are missing.
            if (value is None
                    or isinstance(value, (bool, np.bool_))
                    or isinstance(value, (complex, np.complexfloating))
                    or _numeric_is_nan(value)
                    or not isinstance(value, (numbers.Number, _decimal.Decimal))):
                return None
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return f if math.isfinite(f) else None

        _row_target = [_numeric_label(labels[int(row)])
                       for row in train_rows.tolist()]
        _ref = [v for pos, v in enumerate(_row_target)
                if not drop[pos] and v is not None]
        n_target_reference_labels = int(len(_ref))
        if len(_ref) >= 2:
            _ref_arr = np.asarray(_ref, dtype=np.float64)
            q1, q3 = _safe_quantiles(_ref_arr, (0.25, 0.75))
            with np.errstate(over='ignore', invalid='ignore'):
                iqr = float(q3 - q1)
            if not math.isfinite(iqr) or iqr <= _EPS:
                lo, hi = float(np.min(_ref_arr)), float(np.max(_ref_arr))
            else:
                with np.errstate(over='ignore', invalid='ignore'):
                    lo = float(q1 - target_outlier_iqr * iqr)
                    hi = float(q3 + target_outlier_iqr * iqr)
                if not math.isfinite(lo):
                    lo = float(np.min(_ref_arr))
                if not math.isfinite(hi):
                    hi = float(np.max(_ref_arr))
            target_outlier_fence = {
                'lo': lo, 'hi': hi, 'k': float(target_outlier_iqr),
                'n_train_reference': int(len(_ref)),
            }
            for pos, value in enumerate(_row_target):
                if drop[pos] or value is None:
                    continue
                if value < lo or value > hi:
                    drop[pos] = True
                    n_target_outlier_rows += 1
    # Only the TRAIN rows need duplicate clusters here.  Hashing the whole
    # table materialised one bytes object per row (a multi-GB list on a
    # scale-up corpus) even when clean_train was applied to a small subset.
    keys = _row_keys(_take(_raw_X, train_rows))
    clusters = {}
    for pos, row in enumerate(train_rows.tolist()):
        if drop[pos]:
            continue
        clusters.setdefault(keys[pos], []).append(pos)
    n_duplicate_rows = 0
    n_conflict_rows = 0
    n_conflict_clusters = 0
    for positions in clusters.values():
        if len(positions) < 2:
            continue
        valid = set()
        if labels is not None:
            valid = {_hashable_label(labels[int(train_rows[p])])
                     for p in positions
                     if not _missing(labels[int(train_rows[p])])}
        if labels is not None and len(valid) > 1:
            n_conflict_clusters += 1
            if label_conflicts == 'raise':
                raise ValueError(
                    f'clean_training_rows: {len(positions)} duplicate training '
                    f'rows carry {len(valid)} different labels; resolve them '
                    'or use label_conflicts="drop"')
            if label_conflicts == 'drop':
                drop[positions] = True
                n_conflict_rows += len(positions)
                continue
            continue
        if drop_duplicates:
            keep = positions[0]
            if labels is not None:
                for p in positions:
                    if not _missing(labels[int(train_rows[p])]):
                        keep = p
                        break
            for p in positions:
                if p != keep:
                    drop[p] = True
                    n_duplicate_rows += 1
    out_idx = train_rows[~drop].copy()
    if out_idx.size == 0:
        raise ValueError(
            'clean_training_rows: every training row was dropped; refusing to '
            'fit on an empty split')
    return {
        'train_idx': out_idx,
        'dropped_idx': train_rows[drop].copy(),
        'n_train_rows_in': int(train_rows.size),
        'n_train_rows_out': int(out_idx.size),
        'n_duplicate_rows_dropped': int(n_duplicate_rows),
        'n_label_conflict_rows_dropped': int(n_conflict_rows),
        'n_label_conflict_clusters': int(n_conflict_clusters),
        'n_invalid_label_rows_dropped': int(n_invalid_label_rows),
        'n_nonfinite_feature_rows_dropped': int(n_nonfinite_feature_rows),
        'n_outlier_rows_dropped': int(n_outlier_rows),
        'n_target_outlier_rows_dropped': int(n_target_outlier_rows),
        'target_outlier_iqr': (None if target_outlier_iqr is None
                               else float(target_outlier_iqr)),
        'target_outlier_fence': target_outlier_fence,
        # Distinguish "no numeric labels were outside the fence" from "the
        # requested fence could not be estimated at all" (categorical or
        # multi-output target, or fewer than two finite scalar labels).  A
        # caller must not report a label-noise cleanup that never ran.
        'target_outlier_fence_estimated': (
            None if target_outlier_iqr is None
            else bool(target_outlier_fence is not None)),
        'n_target_reference_labels': int(n_target_reference_labels),
        'drop_duplicates': bool(drop_duplicates),
        'label_conflicts': label_conflicts,
        'drop_invalid_labels': bool(drop_invalid_labels),
        'drop_nonfinite_features': bool(drop_nonfinite_features),
        'outlier_iqr': (None if outlier_iqr is None else float(outlier_iqr)),
    }


def _split_index_fingerprint(idx):
    a = np.asarray(idx, dtype=np.int64).reshape(-1)
    h = hashlib.sha256()
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    if a.size:
        # Hash the contiguous buffer through a zero-copy view.  `.tobytes()`
        # materialised a second full copy of the index array, which is
        # needless host memory (and a peak allocation) when a large split is
        # fingerprinted.
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


def split_content_fingerprints(X, y, indices):
    """Per-split content fingerprints of exactly the rows that will be used.

    A single aggregate digest cannot prove that train, validation and test
    each kept the identity the analysis assumes.  These are order-sensitive
    content digests (row order is part of the served data), plus the index
    digest so a reordering/split-boundary change is visible even when the
    multiset of rows is unchanged.  Preprocessing must be fitted after this
    point; the fingerprints describe the raw split content.
    """
    X = np.asarray(X)
    yarr = None if y is None else np.asarray(y)
    out = {}
    for name in SPLIT_NAMES:
        idx = np.asarray(indices.get(name, []), dtype=np.int64).reshape(-1)
        Xs = _take(X, idx)
        rec = {'n_rows': int(idx.size),
               'index_fp': _split_index_fingerprint(idx),
               'content_fp': None}
        try:
            rec['content_fp'] = _matrix_fingerprint(Xs)
        except (TypeError, ValueError, OverflowError) as e:
            # A null fingerprint is not a neutral value: two unrelated
            # datasets both compare as None == None.  Keep an exact
            # dtype/shape/raw-byte digest instead, and mark it as a fallback
            # so the primary canonical fingerprint is never silently
            # replaced by "unknown".
            h = hashlib.sha256()
            _xs = np.asarray(Xs)
            h.update(np.dtype(_xs.dtype).str.encode('ascii'))
            h.update(str(tuple(_xs.shape)).encode('ascii'))
            if _xs.size:
                h.update(memoryview(np.ascontiguousarray(_xs)).cast('B'))
            rec['content_fp'] = h.hexdigest()
            rec['content_fp_fallback'] = 'raw_bytes'
            rec['content_fp_error'] = f'{type(e).__name__}: {e}'
        if yarr is not None:
            ys = _take(yarr, idx)
            try:
                rec['target_fp'] = _matrix_fingerprint(np.asarray(ys).reshape(-1, 1))
            except (TypeError, ValueError, OverflowError):
                h = hashlib.sha256()
                h.update(json.dumps([repr(v) for v in np.asarray(ys).reshape(-1).tolist()],
                                    ensure_ascii=True, sort_keys=True).encode('utf-8'))
                rec['target_fp'] = h.hexdigest()
        out[name] = rec
    return out


def split_and_prepare(X, y=None, *, time=None, groups=None,
                      train_frac=0.8, val_frac=0.1, test_frac=0.1,
                      train_size=None, val_size=None, test_size=None,
                      purge=0, preprocess=True, quality_report=False,
                      clean_train=False, drop_duplicates=True,
                      label_conflicts='drop',
                      drop_invalid_labels=False,
                      drop_nonfinite_features=False, outlier_iqr=None,
                      target_outlier_iqr=None,
                      check_cross_split_duplicates=True,
                      quality_kwargs=None, fail_on_quality=False,
                      **preprocessor_kwargs):
    """Split first, then fit every preprocessing statistic on train only.

    This is the fail-closed convenience entry point for the whole data path::

        out = split_and_prepare(X, y, time=t, groups=g, purge=2)
        Xtr, Xva, Xte = out['X']['train'], out['X']['val'], out['X']['test']

    The train/val/test indices are produced before any imputation, clipping or
    scaling.  ``LeakageSafePreprocessor`` is fitted on the training rows and
    only ``transform`` is applied to validation/test, so no holdout mean,
    variance, quantile or missing indicator can enter the fitted statistics.
    Passing ``quality_report=True`` additionally runs
    :func:`data_quality_report` on the raw table with the same split indices
    (still train-only statistics for the outlier/shift reference).
    ``fail_on_quality=True`` runs that report even when ``quality_report`` is
    false and refuses to return data when its status is ``'fail'`` — a hard
    data-quality failure (label noise, NaN/Inf, cross-split duplicates, ...)
    must not be silently carried into a model fit.  Warnings remain warnings.
    ``clean_train=True`` first removes exact duplicate training rows
    (``drop_duplicates=True``; set it to ``False`` only when duplicates are the
    intended observations) and, by default, ambiguous duplicate-label clusters
    via :func:`clean_training_rows`; ``drop_invalid_labels=True`` additionally
    removes rows with missing/non-finite training targets, and
    ``drop_nonfinite_features=True`` / ``outlier_iqr=k`` remove training rows
    with non-finite features or values outside the training IQR fence, and
    ``target_outlier_iqr=k`` removes numeric training labels outside a
    train-only target fence (label-noise cleanup).
    Validation/test rows are never dropped or inspected.  The returned
    ``indices['train']`` is the cleaned index, so the preprocessor and every
    downstream step see exactly the same rows.  Unless
    ``check_cross_split_duplicates=False``,
    exact duplicate feature rows spanning two splits are a hard error: a row
    that appears in both train and a holdout leaks the target/identity across
    the split and cannot be repaired by train-only scaling.

    Returns a dict with ``indices``, per-split ``X`` (and ``y``), the fitted
    ``preprocessor`` (or ``None`` when ``preprocess=False``), an optional
    ``cleaning`` record and optionally the ``quality`` report.

    Options that would otherwise be silently ignored are refused: any
    cleaning choice requires ``clean_train=True`` and any preprocessor option
    requires ``preprocess=True``.  A caller that asked for label-noise removal
    or train-only scaling must not receive an unchanged, unfitted pipeline.
    """
    if not clean_train:
        _ignored = []
        if label_conflicts != 'drop':
            _ignored.append(f'label_conflicts={label_conflicts!r}')
        if drop_invalid_labels:
            _ignored.append('drop_invalid_labels=True')
        if drop_nonfinite_features:
            _ignored.append('drop_nonfinite_features=True')
        if outlier_iqr is not None:
            _ignored.append('outlier_iqr')
        if target_outlier_iqr is not None:
            _ignored.append('target_outlier_iqr')
        if drop_duplicates is not True:
            _ignored.append(f'drop_duplicates={drop_duplicates!r}')
        if _ignored:
            raise ValueError(
                'split_and_prepare: cleaning option(s) '
                + ', '.join(_ignored)
                + ' require clean_train=True; refusing to ignore a requested '
                'training-set cleaning step')
    if not preprocess and preprocessor_kwargs:
        raise ValueError(
            'split_and_prepare: preprocessor options '
            + ', '.join(sorted(preprocessor_kwargs))
            + ' were supplied while preprocess=False; refusing to ignore a '
            'requested train-only preprocessing step')
    if quality_kwargs and not quality_report and not fail_on_quality:
        raise ValueError(
            'split_and_prepare: quality_kwargs were supplied while both '
            'quality_report and fail_on_quality are false; refusing to ignore '
            'the requested data-quality audit')
    if quality_kwargs is not None and not isinstance(quality_kwargs, dict):
        raise ValueError(
            'split_and_prepare: quality_kwargs must be a mapping, got '
            f'{type(quality_kwargs).__name__}')
    # The audit must describe the exact split that is returned.  Letting a
    # caller inject its own train/val/test indices (or labels) either raises a
    # confusing duplicate-keyword TypeError or, for a differently shaped path,
    # would audit statistics learned from a different reference than the model
    # fit sees.  Those fields are owned by this helper.
    _quality_reserved = {'train_idx', 'val_idx', 'test_idx', 'y'}
    _quality_bad = sorted(set(quality_kwargs or {}) & _quality_reserved)
    if _quality_bad:
        raise ValueError(
            'split_and_prepare: quality_kwargs must not override '
            + ', '.join(_quality_bad)
            + '; the quality audit is bound to the split returned by this call')
    Xarr = _as_numeric_2d(X)
    n = int(Xarr.shape[0])
    yarr = None
    if y is not None:
        yarr = _label_array(y)
        if yarr.ndim == 0 or yarr.shape[0] != n:
            raise ValueError(f'y has length {yarr.shape[0] if yarr.ndim else 1}, '
                             f'expected {n}')
    idx = split_indices(n, time=time, groups=groups,
                        train_frac=train_frac, val_frac=val_frac,
                        test_frac=test_frac, train_size=train_size,
                        val_size=val_size, test_size=test_size, purge=purge)
    cleaning = None
    if clean_train:
        cleaning = clean_training_rows(
            Xarr, y=yarr, train_idx=idx['train'],
            drop_duplicates=drop_duplicates,
            label_conflicts=label_conflicts,
            drop_invalid_labels=drop_invalid_labels,
            drop_nonfinite_features=drop_nonfinite_features,
            outlier_iqr=outlier_iqr,
            target_outlier_iqr=target_outlier_iqr)
        idx['train'] = cleaning['train_idx']
        assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                                groups=groups, time=time)
    _dup_check = None
    if check_cross_split_duplicates:
        # Exact-duplicate rows across splits are leakage even when the row
        # indices, groups and timestamps are disjoint.  This is a cheap
        # content check (no statistics are learned from holdout values).
        _keys = _row_keys(Xarr)
        _sets = {name: {_keys[i] for i in idx[name].tolist()}
                 for name in SPLIT_NAMES}
        _dup_check = {}
        for _i, _a in enumerate(SPLIT_NAMES):
            for _b in SPLIT_NAMES[_i + 1:]:
                _dup_check[f'{_a}__{_b}'] = len(_sets[_a] & _sets[_b])
        _leaked = {k: v for k, v in _dup_check.items() if v}
        if _leaked:
            raise ValueError(
                f'split_and_prepare: exact duplicate feature rows cross split '
                f'boundaries {_leaked}; clean the source rows before splitting '
                'or pass check_cross_split_duplicates=False only for an '
                'explicitly reported sensitivity run')
    raw_X = {name: _take(Xarr, idx[name]) for name in SPLIT_NAMES}
    out = {'indices': idx, 'X': dict(raw_X)}
    # Bind each returned split to its raw content before any preprocessing
    # statistic is fitted; a downstream pairing can then prove it received the
    # same train/validation/test bytes rather than just the same row count.
    out['split_fingerprints'] = split_content_fingerprints(Xarr, yarr, idx)
    out['fingerprint_policy'] = ('per-split order-sensitive content + index digests '
                                 'computed BEFORE preprocessing; test is never used '
                                 'to fit any statistic')
    if _dup_check is not None:
        out['cross_split_duplicates'] = _dup_check
    else:
        # An explicit waiver is part of the record, not an invisible default.
        out['cross_split_duplicates_waived'] = True
    if cleaning is not None:
        out['cleaning'] = cleaning
    if yarr is not None:
        out['y'] = {name: _take(yarr, idx[name]) for name in SPLIT_NAMES}
    prep = None
    if preprocess:
        # fit_transform is used for the TRAIN rows only; holdouts are never
        # passed to fit.
        prep = LeakageSafePreprocessor(**preprocessor_kwargs).fit(raw_X['train'])
        out['X'] = {name: prep.transform(raw_X[name]) for name in SPLIT_NAMES}
        out['preprocessor'] = prep
    else:
        out['preprocessor'] = None
    if fail_on_quality:
        # A strict caller asked for the gate, so the report must run even if
        # quality_report was left at its default.
        quality_report = True
    if quality_report:
        kw = dict(quality_kwargs or {})
        # Forward the same ordering/group keys that produced the split so the
        # report can independently verify temporal and whole-group isolation.
        # An explicit quality_kwargs override wins (callers may pass a longer
        # external ordering array), but it must not be duplicated.  An
        # explicit ``None`` must not silently disable the temporal/group audit
        # that the split helper already knows how to perform.
        kw.setdefault('time', time)
        kw.setdefault('groups', groups)
        if kw.get('time') is None and time is not None:
            kw['time'] = time
        if kw.get('groups') is None and groups is not None:
            kw['groups'] = groups
        out['quality'] = data_quality_report(
            Xarr, y=yarr, train_idx=idx['train'], val_idx=idx['val'],
            test_idx=idx['test'], **kw)
        if fail_on_quality and out['quality'].get('status') == 'fail':
            _hard = list(out['quality'].get('hard_failures') or [])
            raise RuntimeError(
                'split_and_prepare: the post-split data-quality gate failed; '
                'refusing to fit preprocessing/model on this split: '
                + ('; '.join(_hard) if _hard else 'see the quality report'))
    return out


# ---------------------------------------------------------------------------
# Data-quality diagnostics
# ---------------------------------------------------------------------------

_HASH_BLOCK_BYTES = 64 << 20


def _hash_row_block(width, cap):
    """Rows per hashing block under a byte ceiling (at least one row)."""
    return max(1, min(int(cap), _HASH_BLOCK_BYTES // max(1, 8 * int(width))))


def _matrix_fingerprint(X):
    """Content fingerprint of a 1-D/2-D table, preserving integer precision.

    Casting every dtype through float64 would make two distinct int64 rows
    above 2**53 hash identically (the same precision loss that duplicate
    detection must avoid).  Rows are first hashed through the historical
    float64 block form (so an int and a float with the same exactly
    representable value still share a fingerprint); rows whose integer /
    datetime value is NOT exactly representable as float64 additionally append
    an exact per-row key.  Object rows use the canonical keys from
    :func:`_row_keys`.  Signed zero is normalised in both paths.
    """
    arr = np.asarray(X)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # A 1-D table is one column.  Hashing the original (n,) shape while
        # the object path reshaped to (n,1) made the same single-column table
        # fingerprint differently depending on its storage rank.
        arr = arr.reshape(-1, 1)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('ascii'))
    if np.iscomplexobj(arr):
        raise TypeError(
            'matrix fingerprint needs real-valued rows; casting complex rows '
            'to float64 would silently discard the imaginary part')
    if arr.dtype.kind in 'SUV':
        # Native unicode/bytes/structured tables have no float cast.  Route
        # them through the same canonical object-row keys as object columns.
        arr = arr.astype(object)
    if arr.dtype == object:
        # Numeric object tables go through the same float64 block form as a
        # native table (so an int/float/Decimal with the same exactly
        # representable value keeps the same digest), and only rows whose
        # value is NOT exactly representable append the canonical exact key.
        _obj = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        _w = int(_obj.shape[1])
        _step = _hash_row_block(_w, 1 << 20)

        def _row_float_exact(row):
            vals, exact = [], True
            for v in row:
                if v is None or (_is_missing_scalar(v) and not isinstance(
                        v, (np.datetime64, np.timedelta64,
                            _datetime.date, _datetime.datetime,
                            _datetime.timedelta))):
                    # None and every non-temporal missing spelling (numeric
                    # NaN, Decimal NaN, pandas NA) are one missing-like value
                    # in _row_keys; the digest must not split a Decimal NaN
                    # from None merely because Fraction(Decimal('NaN')) cannot
                    # be formed.  Temporal NaT stays a distinct temporal
                    # identity (matching _row_keys).
                    vals.append(float('nan'))
                    continue
                if isinstance(v, (complex, np.complexfloating)):
                    raise TypeError(
                        'matrix fingerprint needs real-valued rows; casting '
                        'complex rows to float64 would silently discard the '
                        'imaginary part')
                try:
                    f = float(v)
                except (TypeError, ValueError, OverflowError):
                    # A huge exact rational (Fraction/Decimal) has no finite
                    # float64 image, and a categorical string has no numeric
                    # image at all.  Use a placeholder here and let the exact
                    # canonical tail below carry the value; rejecting the
                    # table would make categorical object/unicode columns
                    # un-fingerprintable even though their rows are well
                    # defined.
                    vals.append(float('nan'))
                    exact = False
                    continue
                vals.append(f)
                if isinstance(v, (bool, np.bool_)):
                    if not (math.isfinite(f) and f.is_integer()
                            and int(f) == int(v)):
                        exact = False
                elif isinstance(v, numbers.Integral):
                    if not (math.isfinite(f) and f.is_integer()
                            and int(f) == int(v)):
                        exact = False
                elif isinstance(v, float):
                    pass
                elif isinstance(v, np.floating):
                    if v.dtype.itemsize > 8:
                        # A wider float is exact in this block only if its
                        # float64 image round-trips to the same value.
                        try:
                            _same = (math.isfinite(f)
                                     and np.longdouble(f) == v)
                        except (TypeError, ValueError, OverflowError):
                            _same = False
                        if not _same:
                            exact = False
                else:
                    try:
                        n, d = v.as_integer_ratio()
                        nf, df = f.as_integer_ratio()
                    except (AttributeError, TypeError, ValueError,
                            OverflowError):
                        try:
                            exact = exact and bool(v == f)
                        except Exception:
                            exact = False
                    else:
                        if (not math.isfinite(f)
                                or (int(n), int(d)) != (nf, df)):
                            exact = False
            return vals, exact

        # Exact-row tails go into a separate stream that is appended after
        # every block's float image.  Appending a tail immediately after its
        # block made the digest depend on the arbitrary block size (the same
        # rows hashed in one block vs several interleaved the tails
        # differently), so a memory-cap change silently changed a fitted
        # artifact's fingerprint.  A running hash keeps this O(1) memory even
        # when every row needs a tail (e.g. a large categorical object table).
        _tail = hashlib.sha256()
        for start in range(0, _obj.shape[0], _step):
            block = _obj[start:start + _step]
            _rows = block.tolist()
            parsed = [_row_float_exact(row) for row in _rows]
            block_f = np.asarray([p[0] for p in parsed], dtype=np.float64)
            if np.any(block_f == 0.0):
                block_f = block_f.copy()
                block_f[block_f == 0.0] = 0.0
            h.update(np.ascontiguousarray(block_f).tobytes())
            bad = [row for row, (_vals, ok) in zip(_rows, parsed) if not ok]
            if bad:
                # Build the object table explicitly: np.asarray(bad) would
                # expand tuple-valued cells (structured/void rows) into an
                # extra dimension instead of keeping one cell per column.
                _bad_obj = np.empty((len(bad), int(_obj.shape[1])), dtype=object)
                for _bi, _brow in enumerate(bad):
                    _bad_obj[_bi, :] = _brow
                # The block float image cannot distinguish these rows from an
                # exact row with the same float64 image (e.g. a rounded int
                # next to the exactly representable neighbour, or an
                # uncoercible string next to None).  Attach the absolute row
                # position so a permutation of those rows changes the digest.
                _bad_pos = [start + i for i, (_vals, ok) in enumerate(parsed)
                            if not ok]
                for pos, key in zip(_bad_pos, _row_keys(_bad_obj)):
                    _tail.update(b'|R')
                    _tail.update(str(int(pos)).encode('ascii'))
                    _tail.update(b'|')
                    _tail.update(key)
        h.update(b'|T')
        h.update(_tail.digest())
        return h.hexdigest()
    _w = int(arr.shape[1]) if arr.ndim > 1 else 1
    _step = _hash_row_block(_w, 1 << 20)
    _is_int = (np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_
               or np.issubdtype(arr.dtype, np.datetime64)
               or np.issubdtype(arr.dtype, np.timedelta64))
    # As in the object path, exact-row tails go into a separate running hash
    # that is folded in after all block float images.  Otherwise the same
    # table fingerprints differently when the memory cap changes how rows are
    # blocked, and buffering every tail would cost O(rows) memory.
    _tail = hashlib.sha256()
    for start in range(0, arr.shape[0], _step):
        block = arr[start:start + _step]
        with np.errstate(invalid='ignore', over='ignore'):
            block_f = np.asarray(block, dtype=np.float64)
        if np.any(block_f == 0.0):
            block_f = block_f.copy()
            block_f[block_f == 0.0] = 0.0
        h.update(np.ascontiguousarray(block_f).tobytes())
        if arr.dtype.kind == 'f' and arr.dtype.itemsize > 8:
            # Distinct longdouble values can share one float64 image; only
            # the rows that survive the float64 round-trip exactly can omit
            # the exact tail.  Appending a key for every row made a native
            # longdouble table fingerprint differently from the same values
            # in an object column (which only tails inexact rows), and the
            # old tail also lacked the row position used by the other paths.
            _ld_exact = (np.isfinite(block_f)
                         & (block_f.astype(np.longdouble) == block))
            _bad_rows = (~_ld_exact.all(axis=-1) if _ld_exact.ndim > 1
                         else ~_ld_exact)
            if np.any(_bad_rows):
                _bad = block[_bad_rows]
                if _bad.ndim == 1:
                    _bad = _bad.reshape(-1, 1)
                _bad_pos = np.nonzero(np.reshape(_bad_rows, -1))[0] + start
                for pos, key in zip(
                        _bad_pos.tolist(),
                        _row_keys(np.asarray(_bad, dtype=object))):
                    _tail.update(b'|R')
                    _tail.update(str(int(pos)).encode('ascii'))
                    _tail.update(b'|')
                    _tail.update(key)
            continue
        if not _is_int:
            continue
        # Which cells survive the float64 round-trip exactly?  Only inexact
        # rows need an exact tail, so the common (small-integer) case keeps
        # the same digest as an equivalent float64 table.
        _raw = block if not (np.issubdtype(arr.dtype, np.datetime64)
                             or np.issubdtype(arr.dtype, np.timedelta64)) else block.view(np.int64)
        _raw = np.asarray(_raw)
        if np.issubdtype(_raw.dtype, np.unsignedinteger):
            # Do not use int64 as the round-trip type: uint64 values above
            # int64's maximum are still exactly representable in float64
            # (e.g. 2**63, or every multiple of 2048 near 2**64).  Treating
            # all of them as inexact appended a tail that the equivalent
            # object/float row does not have, so the same value fingerprinted
            # differently by dtype.  Round-trip through uint64 with an
            # explicit representable-range guard instead.
            _castable = (np.isfinite(block_f) & (block_f >= 0.0)
                         & (block_f < 2.0 ** 64))
            _back_u = np.zeros(_raw.shape, dtype=np.uint64)
            if _castable.any():
                _back_u[_castable] = block_f[_castable].astype(np.uint64)
            _exact = _castable & (_back_u == _raw)
        else:
            # Guard the float64 -> int64 cast.  The float64 image of int64_max
            # is 2**63, which is outside int64; casting it directly is an
            # implementation-defined overflow that emits a RuntimeWarning (and
            # raises if warnings are errors).  Such a value is inexact by
            # definition, so mark it non-castable and let the exact tail below
            # carry it.
            _castable = (np.isfinite(block_f)
                         & (block_f >= -(2.0 ** 63))
                         & (block_f < (2.0 ** 63)))
            _back = np.zeros(_raw.shape, dtype=np.int64)
            if _castable.any():
                _back[_castable] = block_f[_castable].astype(np.int64)
            _exact = _castable & (_back == _raw)
        _bad_rows = ~_exact.all(axis=-1) if _exact.ndim > 1 else ~_exact
        if np.any(_bad_rows):
            _bad = _raw[_bad_rows]
            if _bad.ndim == 1:
                _bad = _bad.reshape(-1, 1)
            # Canonical object keys (not raw dtype bytes) so a native int64
            # row and the same Python-int object row share a digest.  Include
            # the absolute row position for the same reason as the object
            # path: two rows can share the rounded float64 image (an inexact
            # large integer next to its exactly representable neighbour), so
            # without a position a permutation would fingerprint identically.
            _bad_pos = np.nonzero(np.reshape(_bad_rows, -1))[0] + start
            for pos, key in zip(_bad_pos.tolist(),
                                _row_keys(np.asarray(_bad, dtype=object))):
                _tail.update(b'|R')
                _tail.update(str(int(pos)).encode('ascii'))
                _tail.update(b'|')
                _tail.update(key)
    h.update(b'|T')
    h.update(_tail.digest())
    return h.hexdigest()

def _wide_float_key(v):
    """Exact canonical key for a float wider than float64 (longdouble)."""
    if np.isnan(v):
        return b'Fnan'
    if np.isposinf(v):
        return b'F+inf'
    if np.isneginf(v):
        return b'F-inf'
    if v == 0:
        return b'I0'
    n, d = v.as_integer_ratio()
    n, d = int(n), int(d)
    return (b'I' + str(n).encode('ascii') if d == 1
            else b'R' + str(n).encode('ascii') + b'/' + str(d).encode('ascii'))


def _temporal_canonical_key(v):
    """Exact canonical bytes for a temporal scalar of any Python/NumPy kind.

    ``datetime.date``/``datetime.datetime`` compare equal to matching
    ``np.datetime64`` values (and to each other at midnight), while
    ``datetime.timedelta`` compares equal to some ``np.timedelta64`` values.
    Hashing their raw objects would split one instant into several identities.
    Timezone-aware datetimes share an identity by their UTC instant and carry
    an ``A`` prefix so a naive datetime with the same wall-clock text stays
    distinct (the two are not equal in Python).
    """
    if _pandas_missing(v):
        # pd.NaT is a datetime subclass but its underlying value is not a
        # valid datetime (np.datetime64(pd.NaT) raises).  Keep the same
        # temporal-NaT identity as np.datetime64('NaT') so an object column
        # and a native datetime column agree.
        if isinstance(v, (_datetime.datetime, _datetime.date)):
            return b'DNaT'
        return b'TNaT'
    if isinstance(v, np.datetime64):
        return _temporal_key(v)
    if isinstance(v, np.timedelta64):
        return _temporal_key(v)
    if isinstance(v, _datetime.datetime):
        if v.tzinfo is not None:
            try:
                utc = v.astimezone(_datetime.timezone.utc)
            except (OverflowError, OSError, ValueError):
                return b'AV:' + repr(v).encode('utf-8', 'backslashreplace')
            # Drop the zone only after converting: the canonical key carries
            # the UTC instant plus the aware-only prefix.
            return b'A' + _temporal_key(np.datetime64(utc.replace(tzinfo=None)))
        return _temporal_key(np.datetime64(v))
    if isinstance(v, _datetime.date):
        return _temporal_key(np.datetime64(v))
    if isinstance(v, _datetime.timedelta):
        # np.timedelta64(timedelta) uses microseconds and can overflow for
        # very large spans.  Build the exact nanosecond count directly (Python
        # timedelta has microsecond resolution), matching _temporal_key's
        # nanosecond canonical form for in-range values.
        ns = ((int(v.days) * 86400 + int(v.seconds)) * 1000000
              + int(v.microseconds)) * 1000
        if -(1 << 63) <= ns < (1 << 63):
            return b'TI' + str(ns).encode('ascii')
        # Outside the int64-nanosecond range use the same attosecond form as
        # _temporal_key's fallback, so an equal np.timedelta64 keeps one key.
        return b'TA' + str(ns * (10 ** 9)).encode('ascii')
    raise TypeError(f'not a temporal value: {type(v).__name__}')


# Exact attoseconds per fixed-width datetime/timedelta unit.  Python ints are
# unbounded, so the fallback below stays exact even when the nanosecond form
# would overflow int64.  Calendar units (Y/M) are first converted to days.
_TEMPORAL_AS_PER_UNIT = {
    'W': 7 * 86400 * 10 ** 18,
    'D': 86400 * 10 ** 18,
    'h': 3600 * 10 ** 18,
    'm': 60 * 10 ** 18,
    's': 10 ** 18,
    'ms': 10 ** 15,
    'us': 10 ** 12,
    'ns': 10 ** 9,
    'ps': 10 ** 6,
    'fs': 10 ** 3,
    'as': 1,
}


def _temporal_attoseconds(v):
    """Exact attoseconds since the epoch for a NumPy temporal scalar.

    ``astype('...64[ns]')`` overflows outside ~1678-2262, so the historical
    fallback keyed the raw unit count and dtype.  The same instant written in
    two units then got two identities (e.g. year 2500 in days vs seconds).
    Scaling a Python integer is exact for every unit, including sub-nanosecond
    units, and preserves the instant across units.
    """
    unit, count = np.datetime_data(v.dtype)
    if unit in ('Y', 'M'):
        target = ('datetime64[D]' if isinstance(v, np.datetime64)
                  else 'timedelta64[D]')
        conv = v.astype(target)
        if np.isnat(conv):
            raise ValueError('temporal value is outside the day range')
        raw = int(conv.view('i8'))
        factor = _TEMPORAL_AS_PER_UNIT['D']
        # `conv` is already an absolute day count, so the original unit
        # multiple (e.g. datetime64[2Y]) must NOT be applied again.  The old
        # `raw * count` turned year 3000 in a 2Y dtype into year 6000 and made
        # the same instant disagree by a factor of two.
        count = 1
    else:
        raw = int(v.view('i8'))
        factor = _TEMPORAL_AS_PER_UNIT.get(unit)
        if factor is None:
            raise ValueError(f'unsupported temporal unit {unit!r}')
    return raw * int(count) * factor


def _temporal_key(v):
    """Exact canonical key for an object-cell ``datetime64``/``timedelta64``.

    The generic object path below used to call ``float(v)`` on these values,
    which rounds a timestamp to the nearest float64.  Adjacent nanosecond
    timestamps (or a timestamp and an integer with the same rounded value)
    then hashed identically and were reported as exact duplicates, so a clean
    split could be refused or distinct training rows dropped.  Native temporal
    arrays were already hashed in their own dtype; object columns need the
    same exactness.  Values are canonicalised to nanoseconds when that does
    not overflow, so the same instant expressed in different units keeps one
    identity, with a unit-qualified fallback outside the nanosecond range.
    """
    is_datetime = isinstance(v, np.datetime64)
    prefix = b'D' if is_datetime else b'T'
    if np.isnat(v):
        return prefix + b'NaT'
    target = 'datetime64[ns]' if is_datetime else 'timedelta64[ns]'
    try:
        canon = v.astype(target)
    except (TypeError, ValueError, OverflowError):
        canon = None
    if canon is not None and not np.isnat(canon):
        # NumPy can wrap (not NaT) when casting a far-future/far-past value to
        # nanoseconds.  Only accept the nanosecond form when casting back to
        # the original dtype reproduces the same instant; otherwise use the
        # exact attosecond fallback, so the same instant in D/Y/2Y spellings
        # keeps one identity instead of silently becoming 1830.
        try:
            _back = canon.astype(v.dtype)
            _roundtrips = (not np.isnat(_back)) and bool(_back == v)
        except (TypeError, ValueError, OverflowError):
            _roundtrips = False
        if _roundtrips:
            try:
                return prefix + b'I' + str(int(canon.view('i8'))).encode('ascii')
            except (TypeError, ValueError, OverflowError):
                pass
    try:
        # Outside the nanosecond range, scale the exact unit count to
        # attoseconds instead of keying the raw dtype/count: two units that
        # denote the same instant must share one identity.
        return (prefix + b'A'
                + str(_temporal_attoseconds(v)).encode('ascii'))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        raw = int(v.view('i8'))
    except (TypeError, ValueError, OverflowError):
        return prefix + b'V:' + repr(v).encode('utf-8', 'backslashreplace')
    return (prefix + str(v.dtype).encode('ascii') + b':'
            + str(raw).encode('ascii'))


def _row_keys(X):
    """Stable per-row keys for exact-duplicate detection (NaN-safe)."""
    src = np.asarray(X)
    if src.ndim == 0:
        raise ValueError('duplicate detection needs a 1-D or 2-D table, got a scalar')
    if src.ndim > 2:
        raise ValueError(
            f'duplicate detection needs a 1-D or 2-D table, got shape {src.shape}')
    if np.iscomplexobj(src):
        raise TypeError(
            'duplicate detection needs real-valued rows; casting complex rows '
            'to float64 would silently discard the imaginary part')
    if src.ndim == 1:
        src = src.reshape(-1, 1)
    if src.dtype.kind == 'f' and src.dtype.itemsize > 8:
        # float128/longdouble cannot round-trip through float64 without
        # collapsing distinct values.  Hash an exact rational per cell.
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        for start in range(0, src.shape[0], _step):
            for row in src[start:start + _step].tolist():
                h = hashlib.blake2b(digest_size=16)
                for v in row:
                    h.update(_wide_float_key(v))
                    h.update(b'|')
                keys.append(h.digest())
        return keys
    if src.dtype.kind in 'SUV':
        # Native unicode/bytes/structured columns are not float-castable.
        # Canonical object keys keep their type-qualified identity (so b'a',
        # 'a' and 1 remain distinct) instead of crashing duplicate detection.
        src = src.astype(object)
    _canonical_int = (np.issubdtype(src.dtype, np.integer)
                      or src.dtype == np.bool_)
    if _canonical_int:
        # Hashing the native bytes made the same integer row keyed differently
        # by width (int32 [1, 2] != int64 [1, 2]), so a train/test duplicate
        # stored at different dtypes was reported as zero overlap.  Canonical
        # int64 rows are hashed below (per block, so a large uint16 cache is
        # not duplicated in memory); uint64 values above int64's maximum fall
        # through to the exact object path.
        if (src.dtype.kind == 'u' and src.dtype.itemsize >= 8 and src.size
                and int(src.max()) > int(np.iinfo(np.int64).max)):
            src = src.astype(object)
            _canonical_int = False
    if src.dtype == object:
        # Object columns are common after pandas/None handling.  Casting a
        # whole object row through float64 (the generic fallback below) would
        # collapse distinct integers above 2**53 into false duplicates, which
        # can turn a clean split into a false leakage alarm (or, after
        # rounding, hide a genuine duplicate).  Build an exact canonical token
        # per value: integers keep all their bits, integral floats compare
        # equal to the same integer (matching Python/NumPy equality), -0.0 is
        # normalised, and NaN/Inf remain distinct except that all NaNs are one
        # missing-like key.  Non-numeric objects (strings/categoricals) get a
        # type-qualified representation key; they must not crash the generic
        # float fallback or compare equal to their numeric spelling.
        # Do NOT flatten the entire object table just to reject complex
        # values: one Python reference per cell is pure peak memory on a
        # large table, and the hash loop below already visits every row.
        # The complex check is therefore done per bounded hash block.

        def _real_key(v):
            if v is None:
                # Match the generic float path: None is a missing value and
                # compares equal to a NaN scalar in the same position.
                return b'Fnan'
            if isinstance(v, (np.datetime64, np.timedelta64,
                              _datetime.date, _datetime.datetime,
                              _datetime.timedelta)):
                # A timestamp is not a float or an integer: adjacent
                # nanosecond values share one float64 image, and
                # ``np.timedelta64`` even registers as numbers.Integral.
                # Keep the exact temporal identity, including Python's
                # date/datetime/timedelta values that compare equal to a NumPy
                # temporal scalar.
                return _temporal_canonical_key(v)
            if _pandas_missing(v):
                # pandas NA is the library's missing scalar; NaT is a temporal
                # missing value and keeps the same identity as NumPy NaT.
                if isinstance(v, (_datetime.datetime, _datetime.date)):
                    return b'DNaT'
                return b'Fnan'
            if _numeric_is_nan(v):
                # Decimal('sNaN') raises in float(), so the generic path below
                # would key it as an ordinary categorical string.  A numeric
                # NaN of any encoding is the same missing-like value here.
                return b'Fnan'
            if isinstance(v, (bool, np.bool_)):
                return b'I' + str(int(v)).encode('ascii')
            if isinstance(v, numbers.Integral):
                return b'I' + str(int(v)).encode('ascii')
            if isinstance(v, np.floating) and v.dtype.itemsize > 8:
                return _wide_float_key(v)
            if isinstance(v, (str, bytes)):
                # Numeric strings are categorical labels, not numbers: '1'
                # and 1 are different row values.
                return (b'V' + type(v).__name__.encode('ascii')
                        + b':' + repr(v).encode('utf-8', 'backslashreplace'))
            # Exact rationals (Fraction, Decimal, ...) keep numerator and
            # denominator instead of being rounded through float64.  Decimal
            # is not a numbers.Real subclass, so it would otherwise fall
            # through to the lossy generic path below.
            if not isinstance(v, (float, np.floating)):
                try:
                    n, d = v.as_integer_ratio()
                except (AttributeError, TypeError, ValueError, OverflowError):
                    pass
                else:
                    n, d = int(n), int(d)
                    return (b'I' + str(n).encode('ascii') if d == 1
                            else b'R' + str(n).encode('ascii') + b'/'
                            + str(d).encode('ascii'))
            try:
                f = float(v)
            except (TypeError, ValueError, OverflowError):
                # A categorical/string value has no numeric identity.  Keep
                # its type in the key so string '1' is not the integer 1.
                return (b'V' + type(v).__name__.encode('utf-8', 'backslashreplace')
                        + b':' + repr(v).encode('utf-8', 'backslashreplace'))
            if math.isnan(f):
                return b'Fnan'
            if math.isinf(f):
                return b'F+inf' if f > 0 else b'F-inf'
            if f == 0.0:
                return b'I0'
            if f.is_integer():
                return b'I' + str(int(f)).encode('ascii')
            # Use the exact binary ratio so a Decimal/Fraction that equals
            # this float exactly produces the same key (two different floats
            # can never share a ratio).
            n, d = f.as_integer_ratio()
            return b'R' + str(n).encode('ascii') + b'/' + str(d).encode('ascii')

        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        for start in range(0, src.shape[0], _step):
            _block = src[start:start + _step]
            if any(isinstance(v, (complex, np.complexfloating))
                   for v in _block.flat):
                raise TypeError(
                    'duplicate detection needs real-valued rows; casting complex '
                    'rows to float64 would silently discard the imaginary part')
            for row in _block.tolist():
                # An all-numeric row (None counts as the missing-like NaN)
                # uses the SAME row bytes as the native int/float branch, so
                # an object column and a numeric column holding equal values
                # share one duplicate identity instead of two.  This includes
                # the non-finite mask/code encoding, not only finite rows.
                _vals = []
                _numeric = True
                for v in row:
                    if v is None or (_is_missing_scalar(v) and not isinstance(
                            v, (np.datetime64, np.timedelta64,
                                _datetime.date, _datetime.datetime,
                                _datetime.timedelta))):
                        # Every non-temporal missing scalar (None, NaN,
                        # Decimal NaN, pandas NA) is one missing-like NaN in
                        # the numeric fast path; temporal NaT is routed below
                        # so it keeps its distinct temporal identity.
                        _vals.append(float('nan'))
                        continue
                    if isinstance(v, (np.datetime64, np.timedelta64,
                                      _datetime.date, _datetime.datetime,
                                      _datetime.timedelta)):
                        # np.timedelta64 is numbers.Integral and [ns] even
                        # converts to float successfully, so it used to be
                        # hashed as a raw nanosecond integer instead of its
                        # temporal identity (and could collide with a real
                        # integer row).  Route it to _real_key below.
                        _numeric = False
                        break
                    if (not isinstance(v, (numbers.Number, _decimal.Decimal))
                            or isinstance(v, (complex, np.complexfloating))):
                        _numeric = False
                        break
                    # A longdouble has more than float64's 53-bit mantissa;
                    # the native >8-byte branch hashes its exact wide key, so
                    # do not round it into the common float64 path here.
                    if (isinstance(v, np.floating)
                            and v.dtype.itemsize > 8):
                        _numeric = False
                        break
                    # Decimal('sNaN') raises in float(); it is still one
                    # missing-like NaN identity, not a categorical value.
                    if _numeric_is_nan(v):
                        _vals.append(float('nan'))
                        continue
                    try:
                        _vals.append(float(v))
                    except (TypeError, ValueError, OverflowError):
                        _numeric = False
                        break
                if _numeric:
                    _frow = np.asarray(_vals, dtype=np.float64).reshape(-1)
                    _exact = True
                    for v, fv in zip(row, _frow.tolist()):
                        if math.isfinite(fv):
                            # NumPy integer scalars compare equal to a rounded
                            # float (np.int64(2**53+1) == float(2**53+1) is
                            # True), so `v == fv` accepted values that the
                            # float64 image cannot represent and collapsed
                            # distinct rows.  Require an exact integer image.
                            if isinstance(v, numbers.Integral) and not isinstance(
                                    v, (bool, np.bool_)):
                                if not (float(fv).is_integer()
                                        and int(fv) == int(v)):
                                    _exact = False
                                    break
                            elif v is None or not (v == fv):
                                _exact = False
                                break
                        # Non-finite object values map to their float image;
                        # NaN/Inf identity is carried by the code table below.
                    if _exact:
                        if np.any(_frow == 0.0):
                            _frow = _frow.copy()
                            _frow[_frow == 0.0] = 0.0
                        _finite = np.isfinite(_frow)
                        h = hashlib.blake2b(digest_size=16)
                        h.update(np.ascontiguousarray(_frow[_finite]).tobytes())
                        h.update(b'|')
                        h.update(np.packbits(~_finite).tobytes())
                        if not _finite.all():
                            _code = np.zeros(_frow.shape, dtype=np.uint8)
                            _code[np.isnan(_frow)] = 1
                            _code[np.isposinf(_frow)] = 2
                            _code[np.isneginf(_frow)] = 3
                            h.update(_code.tobytes())
                        keys.append(h.digest())
                        continue
                    # All-int rows that float64 cannot represent use the same
                    # raw int64 bytes as the native integer fallback.
                    if all(isinstance(v, numbers.Integral) for v in row):
                        try:
                            _irow = np.asarray(row, dtype=np.int64)
                        except (TypeError, ValueError, OverflowError):
                            _irow = None
                        if (_irow is not None and _irow.size == len(row)
                                and all(int(v) == int(i)
                                        for v, i in zip(row, _irow.tolist()))):
                            h = hashlib.blake2b(digest_size=16)
                            h.update(memoryview(np.ascontiguousarray(_irow)).cast('B'))
                            keys.append(h.digest())
                            continue
                h = hashlib.blake2b(digest_size=16)
                for v in row:
                    h.update(_real_key(v))
                    h.update(b'|')
                keys.append(h.digest())
        return keys
    if (np.issubdtype(src.dtype, np.datetime64)
            or np.issubdtype(src.dtype, np.timedelta64)):
        # A native temporal row must follow its instant, not the storage unit.
        # Hashing the raw dtype bytes made the same instant keyed differently
        # by unit (``datetime64[D]`` 2020-01-01 != the same value in ``[s]`` or
        # ``[ns]``), so an exact duplicate stored at a different resolution was
        # reported as zero overlap.  Build the same canonical per-cell keys as
        # the object path (which also keeps native and object temporal rows
        # comparable) and include NaT as one missing-like identity.
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        for start in range(0, src.shape[0], _step):
            for row in src[start:start + _step]:
                h = hashlib.blake2b(digest_size=16)
                for v in row:
                    h.update(_temporal_key(v))
                    h.update(b'|')
                keys.append(h.digest())
        return keys
    # Integer/bool rows are hashed in a canonical int64 form.  Converting them
    # to float64 first collapses distinct int64 values above 2**53 into false
    # duplicates.
    if np.issubdtype(src.dtype, np.integer) or src.dtype == np.bool_:
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        _tail = np.packbits(np.zeros(int(src.shape[1]), dtype=bool)).tobytes()
        for start in range(0, src.shape[0], _step):
            block = src[start:start + _step]
            if _canonical_int:
                block = block.astype(np.int64, copy=False)
            # An integer has a float64 twin exactly when the round-trip is
            # lossless.  Hash those ROWS through the SAME float64 bytes the
            # float branch uses, so an int64 row and an equal float64 row get
            # one key (the old int-byte key made train/test duplicates stored
            # at different numeric dtypes invisible to the leakage gate).
            # Values that are not exactly representable stay on the integer
            # bytes; no equal float64 exists for them, so precision is kept.
            #
            # The decision MUST be per row.  A block-wide `all()` made the key
            # of every row in a hash block depend on whether some unrelated
            # row in the same block was float64-exact; once a table exceeded
            # one hash block, the same duplicate could hash differently on
            # either side of the boundary and the leakage/duplicate gates
            # silently missed it.
            _i = np.ascontiguousarray(block, dtype=np.int64)
            _f = np.ascontiguousarray(block, dtype=np.float64)
            _safe = np.isfinite(_f) & (_f >= -(2.0 ** 63)) & (_f < (2.0 ** 63))
            _exact = np.zeros(_f.shape, dtype=bool)
            if _safe.any():
                _exact[_safe] = (_f[_safe].astype(np.int64) == _i[_safe])
            _exact_row = _exact.all(axis=1)
            for _r in range(int(_i.shape[0])):
                h = hashlib.blake2b(digest_size=16)
                if bool(_exact_row[_r]):
                    h.update(memoryview(_f[_r]))
                    h.update(b'|')
                    h.update(_tail)
                else:
                    h.update(memoryview(_i[_r]))
                keys.append(h.digest())
        return keys
    keys = []
    # Hash row blocks; each block is converted to float64 before hashing.
    _step = _hash_row_block(int(src.shape[1]), 1 << 16)
    for start in range(0, src.shape[0], _step):
        arr = np.ascontiguousarray(np.asarray(src[start:start + _step], dtype=np.float64))
        # ``-0.0 == 0.0`` but their byte patterns differ, so raw hashing would
        # miss an exact duplicate split across train/val/test.  Normalize the
        # sign of zero before hashing (NaN values are handled separately).
        # The copy is only made when a zero is actually present, so the common
        # all-nonzero block is not duplicated.
        if np.any(arr == 0.0):
            arr = arr.copy()
            arr[arr == 0.0] = 0.0
        finite = np.isfinite(arr)
        if finite.all():
            # Fast path: every row is finite, so the old key is exactly
            # ``row_bytes + b'|' + packbits(all-False)``.  Hashing the
            # contiguous row view with the precomputed all-zero mask tail
            # reproduces the historical key byte-for-byte while avoiding the
            # per-row boolean-index copy and packbits call.
            _tail = np.packbits(np.zeros(int(src.shape[1]), dtype=bool)).tobytes()
            for row_view in arr.view(np.uint8):
                h = hashlib.blake2b(digest_size=16)
                h.update(memoryview(row_view))
                h.update(b'|')
                h.update(_tail)
                keys.append(h.digest())
            continue
        for row, ok in zip(arr, finite):
            # NaN/Inf positions are represented explicitly so two rows are equal
            # only when their finite values AND their non-finite values match.
            # A mask alone would collapse NaN, +Inf and -Inf in the same
            # positions into one key: those rows are not duplicates of each
            # other, and treating them as such manufactures cross-split
            # leakage (and false label conflicts) from corrupted values.  The
            # byte code keeps NaN/NaN equal while separating the infinities.
            h = hashlib.blake2b(digest_size=16)
            h.update(np.ascontiguousarray(row[ok]).tobytes())
            h.update(b'|')
            h.update(np.packbits(~ok).tobytes())
            # Only a row that actually contains a non-finite value gets the
            # extra value-code tail.  Appending an all-zero tail to finite rows
            # in a mixed block would make the same finite row hash differently
            # depending on which block it landed in (and therefore hide a
            # duplicate whenever the two copies straddle a block boundary).
            if not ok.all():
                _code = np.zeros(row.shape, dtype=np.uint8)
                _code[np.isnan(row)] = 1
                _code[np.isposinf(row)] = 2
                _code[np.isneginf(row)] = 3
                h.update(_code.tobytes())
            keys.append(h.digest())
    return keys


def duplicate_mask(X):
    """Boolean mask marking every row that exactly duplicates an earlier row."""
    keys = _row_keys(X)
    seen = set()
    out = np.zeros(len(keys), dtype=bool)
    for i, k in enumerate(keys):
        if k in seen:
            out[i] = True
        else:
            seen.add(k)
    return out


def _split_array(X, idx):
    if idx is None:
        return X
    return X[np.asarray(idx, dtype=np.int64).reshape(-1)]


def _hashable_label(value):
    if _is_missing_scalar(value):
        # Missing labels must share one identity.  pandas NA is otherwise a
        # perfectly hashable object and becomes a real level in conflict and
        # shift audits.
        return None
    if isinstance(value, (np.datetime64, np.timedelta64,
                          _datetime.date, _datetime.datetime,
                          _datetime.timedelta)):
        # Python dates and NumPy temporal scalars can compare equal while
        # hashing differently.  A label/level identity must follow equality,
        # otherwise the same label is counted as two classes in a conflict or
        # shift audit.
        return _temporal_group_key(value)
    try:
        hash(value)
    except TypeError:
        # Opaque identity: a real tuple category equal to the old tuple
        # sentinel must not be conflated with a genuinely unhashable value.
        return _UnhashableKey(type(value).__name__, repr(value))
    return value


def _label_array(y):
    """Keep heterogeneous labels heterogeneous.

    ``np.asarray([1, '1'])`` silently coerces the integer to the string '1',
    so the two classes become indistinguishable before the shift/conflict
    audit sees them.  ndarray inputs keep their declared dtype; a Python
    sequence is materialised as an object array so each element retains its
    own type.
    """
    if isinstance(y, np.ndarray):
        return y
    return np.asarray(y, dtype=object)


def _label_key(v):
    if _is_missing_scalar(v):
        # None, NaN, NaT and pandas NA are all missing labels.  Checking only
        # None/NumPy-NaT let pd.NA through as an ordinary class label.
        return None
    # Only actual numeric scalars are checked for NaN/Inf.  Coercing every
    # value through float() turned legitimate categorical labels spelled
    # like "nan"/"inf" into missing targets and also conflated numeric
    # strings with their numeric twins.  Numbers.Real includes NumPy scalar
    # types but excludes complex/bool.
    if (isinstance(v, (bool, np.bool_))
            or isinstance(v, (complex, np.complexfloating))
            or not isinstance(v, numbers.Number)):
        return v
    if _numeric_is_nan(v):
        return None
    try:
        f = float(v)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return v
    # NaN and infinities are not valid class/regression targets.  Treating
    # +inf as an ordinary label would hide a corrupted target from the
    # missing-label audit.  Decimal/Fraction are Number but not Real, so the
    # check uses the wider numeric base class.
    if not math.isfinite(f):
        return None
    return v
