from __future__ import annotations


class LeakageSafePreprocessor:
    """Impute, clip and scale features using training statistics only.

    Parameters
    ----------
    scale : {'standard', 'robust', 'minmax', 'none'}
        ``standard`` uses the training mean/std, ``robust`` the median/IQR,
        ``minmax`` the training min/range and ``none`` leaves values unscaled.
    missing : {'median', 'mean', 'zero'} or a finite float
        Imputation value learned from the training rows for each feature.
    clip_quantiles : (float, float) or None
        Lower/upper training quantiles used to clip raw values before scaling.
        ``None`` disables clipping.  Clipping is what keeps a holdout outlier
        from producing an unbounded transformed value without changing the
        learned scale.
    clip_z : float or None
        Optional additional clipping in scaled units *after* scaling, using
        only training statistics (e.g. 8.0).  It applies to the standard,
        robust and minmax scales; ``scale='none'`` has no defined scaled unit,
        so combining it with ``clip_z`` is refused instead of silently
        ignoring the requested clip.
    """

    def __init__(self, *, scale='standard', missing='median',
                 clip_quantiles=(0.001, 0.999), clip_z=None, eps=_EPS):
        if scale not in ('standard', 'robust', 'minmax', 'none'):
            raise ValueError(f'unknown scale={scale!r}')
        if isinstance(missing, (bool, np.bool_)):
            raise ValueError(
                f'missing must be a finite constant or one of '
                f"'median'/'mean'/'zero', got {missing!r}")
        if isinstance(missing, (numbers.Real, np.integer, np.floating)):
            try:
                missing = float(missing)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f'missing must be a finite constant or one of '
                    f"'median'/'mean'/'zero', got {missing!r}") from None
            if not math.isfinite(missing):
                raise ValueError(
                    f'missing must be a finite constant or one of '
                    f"'median'/'mean'/'zero', got {missing!r}")
        elif not (isinstance(missing, str) and
                  missing in ('median', 'mean', 'zero')):
            raise ValueError(f'unknown missing={missing!r}')
        if clip_quantiles is not None:
            lo, hi = clip_quantiles
            if any(isinstance(v, (bool, np.bool_)) for v in (lo, hi)):
                raise ValueError('clip_quantiles must be numeric, not booleans')
            try:
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    'clip_quantiles must be two finite numbers in [0, 1]') from None
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError('clip_quantiles must satisfy 0 <= lo < hi <= 1')
            clip_quantiles = (lo, hi)
        if isinstance(clip_z, (bool, np.bool_)):
            raise ValueError('clip_z must be positive or None, not a boolean')
        if clip_z is not None:
            try:
                clip_z = float(clip_z)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    'clip_z must be a finite positive number or None') from None
            if not (math.isfinite(clip_z) and clip_z > 0.0):
                raise ValueError('clip_z must be a finite positive number or None')
            if scale == 'none':
                raise ValueError(
                    "clip_z needs a scaled feature space; scale='none' "
                    'has no standardized/minmax units to clip in')
        if isinstance(eps, (bool, np.bool_)):
            raise ValueError('eps must be a finite non-negative number, not a boolean')
        try:
            eps = float(eps)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'eps must be a finite non-negative number, got {eps!r}') from None
        if not (math.isfinite(eps) and eps >= 0.0):
            raise ValueError(f'eps must be a finite non-negative number, got {eps!r}')
        self.scale = scale
        self.missing = missing
        self.clip_quantiles = clip_quantiles
        self.clip_z = None if clip_z is None else float(clip_z)
        self.eps = eps
        self.fitted_ = False

    # -- fitting ----------------------------------------------------------
    def _fill_value(self, values):
        if isinstance(self.missing, (int, float)):
            return float(self.missing)
        if values.size == 0:
            return 0.0
        if self.missing == 'mean':
            return _safe_mean(values)
        if self.missing == 'median':
            return _safe_median(values)
        return 0.0

    def fit(self, X, y=None):
        """Learn imputation/clipping/scaling statistics from ``X`` (train)."""
        X = _as_numeric_2d(X)
        n, p = X.shape
        if n == 0:
            raise ValueError('cannot fit the preprocessor on zero training rows')
        stats = []
        for j in range(p):
            # Convert one column at a time: the full matrix never needs a
            # second float64 copy (large inputs can be float32/uint8).
            base = _col_float64(X, j)
            col = np.where(np.isfinite(base), base, np.nan)
            finite = col[np.isfinite(col)]
            fill = self._fill_value(finite)
            if not np.isfinite(fill):
                fill = 0.0
            if self.clip_quantiles is not None:
                if finite.size:
                    q_lo, q_hi = _safe_quantiles(finite, self.clip_quantiles)
                else:
                    q_lo, q_hi = -np.inf, np.inf
                if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
                    q_lo, q_hi = -np.inf, np.inf
            else:
                q_lo, q_hi = -np.inf, np.inf
            clipped = np.clip(col, q_lo, q_hi) if np.isfinite(q_lo) else col
            # Clip observed values, then use the imputation constant verbatim.
            # transform() does exactly the same, so a callable fitted on the
            # training rows and then applied to them reproduces the fitted
            # statistics.  (Clipping the fill itself here but not in
            # transform() would make fit and transform disagree whenever the
            # requested constant lies outside the training quantiles.)
            imputed = np.where(np.isfinite(clipped), clipped, fill)
            # Scaling statistics are computed on the training values after the
            # same impute/clip operation that transform() will apply.  They are
            # never recomputed from validation/test data.
            center, factor = 0.0, 1.0
            unit_scale = None
            if self.scale == 'standard':
                center = _safe_mean(imputed)
                sd = _safe_std(imputed)
                factor = sd if sd > self.eps else 1.0
            elif self.scale == 'robust':
                center = _safe_median(imputed)
                q1, q3 = _safe_quantiles(imputed, (0.25, 0.75))
                with np.errstate(over='ignore', invalid='ignore'):
                    iqr = float(q3 - q1)
                if math.isfinite(iqr) and iqr > self.eps:
                    factor = iqr
                else:
                    sd = _safe_std(imputed)
                    factor = sd if sd > self.eps else 1.0
            elif self.scale == 'minmax':
                center = float(np.min(imputed))
                with np.errstate(over='ignore', invalid='ignore'):
                    span = float(np.max(imputed) - center)
                if not math.isfinite(span):
                    # The true training range exceeds float64 (e.g. -1e308
                    # to +1e308).  Keep the fit representable and finite by
                    # scaling with the largest magnitude; transform uses the
                    # divided form below, so no intermediate subtraction can
                    # overflow.
                    scaled, m = _scaled_maxabs(imputed)
                    if scaled is not None:
                        _smin, _smax = float(np.min(scaled)), float(np.max(scaled))
                        if _smax > _smin:
                            center = _smin * m
                            factor = m
                            # The original span is not representable; after
                            # x/m - min/m the range is (_smax-_smin), so a
                            # finite post-scale restores the documented [0,1].
                            unit_scale = 1.0 / (_smax - _smin)
                factor = span if math.isfinite(span) and span > self.eps else factor
            if not np.isfinite(center):
                center = 0.0
            if not np.isfinite(factor) or factor <= self.eps:
                factor = 1.0
            # Every persisted descriptive statistic must itself be finite:
            # from_stats() refuses a non-finite mean/std/median, so an
            # overflowed statistic used to make a successfully fitted
            # preprocessor impossible to reload.
            stats.append({
                'fill': float(fill),
                'clip_lo': float(q_lo) if np.isfinite(q_lo) else None,
                'clip_hi': float(q_hi) if np.isfinite(q_hi) else None,
                'center': center,
                'factor': factor,
                'unit_scale': unit_scale,
                'mean': _safe_mean(imputed),
                'std': _safe_std(imputed),
                'median': _safe_median(imputed),
                'min': float(np.min(imputed)),
                'max': float(np.max(imputed)),
            })
        self.feature_stats_ = stats
        self.n_features_in_ = int(p)
        self.n_train_ = int(n)
        self.train_fp_ = _matrix_fingerprint(X)
        self.fitted_ = True
        return self

    # -- applying ---------------------------------------------------------
    def transform(self, X):
        """Apply training statistics to ``X``; never refits anything."""
        if not self.fitted_:
            raise RuntimeError('LeakageSafePreprocessor.transform called before fit')
        X = _as_numeric_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {X.shape[1]} features, but the preprocessor was fitted '
                f'on {self.n_features_in_}')
        out = np.empty(X.shape, dtype=np.float64)
        for j, st in enumerate(self.feature_stats_):
            col = _col_float64(X, j)
            col = np.where(np.isfinite(col), col, np.nan)
            lo, hi = st['clip_lo'], st['clip_hi']
            if lo is not None and hi is not None:
                col = np.clip(col, lo, hi)
            col = np.where(np.isnan(col), st['fill'], col)
            if self.scale != 'none':
                # Divide before subtracting.  ``(x - center) / factor`` can
                # overflow for finite x/center of opposite sign near the
                # float64 limit even though the scaled result is small;
                # x/factor - center/factor is mathematically identical and
                # stays finite whenever the scaled value is representable.
                col = col / st['factor'] - st['center'] / st['factor']
                if st.get('unit_scale') is not None:
                    col = col * st['unit_scale']
            if self.clip_z is not None and self.scale != 'none':
                col = np.clip(col, -self.clip_z, self.clip_z)
            out[:, j] = col
        if not np.all(np.isfinite(out)):
            raise ValueError(
                'non-finite value produced by transform(); check the fitted '
                'statistics and the raw input for overflow')
        return out

    def fit_transform(self, X, y=None):
        """Fit on **training** data and transform it (never call on test)."""
        return self.fit(X, y).transform(X)

    # -- persistence ------------------------------------------------------
    def get_stats(self):
        if not self.fitted_:
            raise RuntimeError('preprocessor has not been fitted')
        return {
            'scale': self.scale,
            'missing': self.missing,
            'clip_quantiles': self.clip_quantiles,
            'clip_z': self.clip_z,
            'eps': self.eps,
            'n_features_in': self.n_features_in_,
            'n_train': self.n_train_,
            'train_fp': self.train_fp_,
            'features': [dict(s) for s in self.feature_stats_],
        }

    @classmethod
    def from_stats(cls, stats):
        """Rebuild a fitted preprocessor from :meth:`get_stats` output."""
        features, n_features = _persisted_features(
            stats, what='LeakageSafePreprocessor')
        what = 'LeakageSafePreprocessor'
        for j, st in enumerate(features):
            for key in ('fill', 'center', 'factor'):
                _stat_float(st, key, what=what, index=j)
            if _stat_float(st, 'factor', what=what, index=j) <= 0.0:
                raise ValueError(
                    f'{what}.from_stats: feature {j} has a non-positive '
                    'scale factor')
            if st.get('unit_scale') is not None:
                _us = _stat_float(st, 'unit_scale', what=what, index=j)
                if _us is None or _us <= 0.0:
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has an invalid '
                        'minmax unit_scale')
            lo = _stat_float(st, 'clip_lo', what=what, index=j,
                             allow_none=True)
            hi = _stat_float(st, 'clip_hi', what=what, index=j,
                             allow_none=True)
            if (lo is None) != (hi is None):
                raise ValueError(
                    f'{what}.from_stats: feature {j} has only one clip bound; '
                    'both bounds must be present or both absent')
            if lo is not None and lo > hi:
                raise ValueError(
                    f'{what}.from_stats: feature {j} has clip_lo={lo} > '
                    f'clip_hi={hi}')
            for key in ('mean', 'std', 'median', 'min', 'max'):
                if key in st:
                    _stat_float(st, key, what=what, index=j,
                                allow_none=True)
        if 'scale' not in stats or 'missing' not in stats:
            raise ValueError(
                f'{what}.from_stats: stats must declare scale and missing')
        try:
            clip_quantiles = (tuple(stats['clip_quantiles'])
                              if stats.get('clip_quantiles') else None)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'{what}.from_stats: malformed clip_quantiles') from None
        obj = cls(scale=stats['scale'], missing=stats['missing'],
                  clip_quantiles=clip_quantiles,
                  clip_z=stats.get('clip_z'), eps=stats.get('eps', _EPS))
        obj.feature_stats_ = [dict(s) for s in features]
        obj.n_features_in_ = int(n_features)
        obj.n_train_ = _check_count('n_train', stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


def _as_categorical_2d(X, *, name='X'):
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 1-D or 2-D, got shape {arr.shape}')
    return arr


def _numeric_is_nan(value):
    """True for a real numeric scalar that represents NaN in any encoding.

    ``float(Decimal('sNaN'))`` raises instead of returning NaN, so relying on
    ``math.isnan(float(value))`` silently turns a signaling NaN into an
    ordinary category/group/label.  Exact numeric objects expose ``is_nan``;
    fall back to the float image for types that do not.
    """
    if (isinstance(value, (bool, np.bool_))
            or isinstance(value, (complex, np.complexfloating))
            or not isinstance(value, numbers.Number)):
        return False
    is_nan = getattr(value, 'is_nan', None)
    if callable(is_nan):
        try:
            return bool(is_nan())
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            pass
    try:
        return bool(math.isnan(float(value)))
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return False


def _category_is_missing(value):
    if value is None:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    # Decimal (and other exact numeric objects) can represent NaN too;
    # treating it as an ordinary category would hide a missing value.
    return _numeric_is_nan(value)


def _sorted_categories(values):
    unique = {}
    for value in values:
        if _category_is_missing(value):
            continue
        # Key by the same canonical identity transform() uses.  Hashing the raw
        # objects would keep, say, datetime.date(2020, 1, 1) and
        # np.datetime64('2020-01-01') as two levels with one persisted key;
        # the transform lookup then silently drops one of them.
        unique.setdefault(_target_key(value), value)
    return sorted(unique.values(),
                  key=lambda v: (type(v).__name__, repr(v)))


def _categorical_fingerprint(X):
    """Content fingerprint of a categorical table under encoder identity.

    ``repr()`` split equal categories written as different temporal types
    (``datetime.date`` vs ``np.datetime64``) even though the encoder maps them
    to the same level.  The target encoder uses this fingerprint to refuse
    transforming its own training rows, so a repr difference would silently
    re-enable the full-table leak.  Hash the same canonical ``_target_key``
    identity that ``transform`` uses instead.
    """
    h = hashlib.sha256()
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    h.update(str(arr.shape).encode('ascii'))
    for row in arr.tolist():
        for value in row:
            key = _target_key(value)
            h.update(str(len(key)).encode('ascii'))
            h.update(b':')
            h.update(key.encode('utf-8', errors='replace'))
            h.update(b'|')
        h.update(b'\x1f')
    return h.hexdigest()


def _categorical_row_set_fingerprint(X):
    """Order-insensitive identity of the rows in a categorical table.

    ``_categorical_fingerprint`` is order-sensitive.  The target encoder uses
    this companion digest to refuse the training rows even when the caller
    presents the same rows in a different order: a reordered training matrix
    still contains each row's own label, so allowing it through ``transform``
    would restore the full-table leak.  Only a commutative 128-bit sum/xor of
    per-row digests is retained, so large tables are not kept in memory.
    """
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mask = (1 << 128) - 1
    total = 0
    xored = 0
    for row in arr.tolist():
        h = hashlib.blake2b(digest_size=16)
        for value in row:
            key = _target_key(value)
            h.update(str(len(key)).encode('ascii'))
            h.update(b':')
            h.update(key.encode('utf-8', errors='replace'))
            h.update(b'|')
        digest = int.from_bytes(h.digest(), 'little')
        total = (total + digest) & mask
        xored ^= digest
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode('ascii'))
    h.update(b'|')
    h.update(str(int(arr.shape[0])).encode('ascii'))
    h.update(b'|')
    h.update(str(total).encode('ascii'))
    h.update(b'|')
    h.update(str(xored).encode('ascii'))
    return h.hexdigest()


def _json_safe_category(value):
    """Represent a category value without changing its learned identity.

    The persisted ``keys`` field is what ``transform`` uses, so the readable
    ``value`` copy may be sanitized.  Without this, a bytes/datetime/custom
    level makes ``json.dumps(get_stats())`` fail even though the encoder is
    otherwise fully fitted.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.datetime64, np.timedelta64)):
        # ``np.timedelta64`` is an ``np.integer``; int() raises for day-or-
        # larger units and would silently shrink a nanosecond delta to its raw
        # unit count.  Keep the readable value and dtype instead.
        return {'__numpy_datetime__': str(value), '__dtype__': str(value.dtype)}
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bytes):
        return {'__bytes_hex__': value.hex()}
    if isinstance(value, tuple):
        return {'__tuple__': [_json_safe_category(v) for v in value]}
    try:
        json.dumps(value, allow_nan=True)
    except (TypeError, ValueError, OverflowError):
        return {'__repr__': repr(value),
                '__type__': f'{type(value).__module__}.{type(value).__qualname__}'}
    return value


def _category_lookup(st):
    """Per-level lookup plus the key function that matches its stored keys.

    New stats persist canonical string keys so a JSON round-trip cannot change
    a tuple category into a different (list) identity.  Legacy stats without
    keys keep the historical ``_hashable_label`` behaviour.
    """
    categories = list(st.get('categories') or [])
    keys = st.get('keys')
    if (isinstance(keys, list) and len(keys) == len(categories)
            and all(isinstance(k, str) for k in keys)):
        return ({k: i for i, k in enumerate(keys)}, _target_key)
    return ({_hashable_label(c): i for i, c in enumerate(categories)},
            _hashable_label)


class LeakageSafeCategoricalEncoder:
    """Ordinal/one-hot encode categories learned from training rows only.

    ``fit`` records the category set of each column from the training split.
    ``transform`` maps validation/test values into that fixed space and never
    adds a level, so an unseen holdout category cannot widen the feature matrix
    or shift the encoding of a training row.  Missing values and unknown
    categories always have their own reserved level.
    """

    def __init__(self, *, encoding='ordinal', handle_unknown='unknown'):
        if encoding not in ('ordinal', 'onehot'):
            raise ValueError(f"encoding must be 'ordinal' or 'onehot', got {encoding!r}")
        if handle_unknown not in ('unknown', 'error'):
            raise ValueError(
                f"handle_unknown must be 'unknown' or 'error', got {handle_unknown!r}")
        self.encoding = encoding
        self.handle_unknown = handle_unknown
        self.fitted_ = False

    def fit(self, X, y=None):
        Xarr = _as_categorical_2d(X)
        n, p = Xarr.shape
        if n == 0:
            raise ValueError('cannot fit a categorical encoder on zero training rows')
        stats = []
        for j in range(p):
            categories = _sorted_categories(Xarr[:, j].tolist())
            stats.append({
                'categories': categories,
                # Persist a JSON-safe identity per level.  A tuple category is
                # serialized as a list by JSON, so rebuilding the lookup from
                # the round-tripped value would silently turn a training level
                # into an unseen one.
                'keys': [_target_key(c) for c in categories],
                'n_categories': int(len(categories)),
                'unknown_index': int(len(categories)),
                'missing_index': int(len(categories) + 1),
            })
        self.feature_stats_ = stats
        self.n_features_in_ = int(p)
        self.n_train_ = int(n)
        self.train_fp_ = _categorical_fingerprint(Xarr)
        self.fitted_ = True
        return self

    def _check_transform_input(self, X):
        if not self.fitted_:
            raise RuntimeError(
                'LeakageSafeCategoricalEncoder.transform called before fit')
        Xarr = _as_categorical_2d(X)
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {Xarr.shape[1]} features, but the encoder was fitted '
                f'on {self.n_features_in_}')
        return Xarr

    def transform(self, X):
        Xarr = self._check_transform_input(X)
        n = int(Xarr.shape[0])
        if self.encoding == 'ordinal':
            out = np.empty((n, self.n_features_in_), dtype=np.float64)
            for j, st in enumerate(self.feature_stats_):
                lookup, keyfn = _category_lookup(st)
                for i, value in enumerate(Xarr[:, j].tolist()):
                    if _category_is_missing(value):
                        out[i, j] = st['missing_index']
                        continue
                    code = lookup.get(keyfn(value))
                    if code is None:
                        if self.handle_unknown == 'error':
                            raise ValueError(
                                f'unseen category {value!r} in column {j}; '
                                'the encoder was fitted on training rows only')
                        code = st['unknown_index']
                    out[i, j] = code
            return out
        width = sum(s['n_categories'] + 2 for s in self.feature_stats_)
        out = np.zeros((n, width), dtype=np.float64)
        offset = 0
        for j, st in enumerate(self.feature_stats_):
            lookup, keyfn = _category_lookup(st)
            span = st['n_categories'] + 2
            for i, value in enumerate(Xarr[:, j].tolist()):
                if _category_is_missing(value):
                    out[i, offset + st['missing_index']] = 1.0
                    continue
                code = lookup.get(keyfn(value))
                if code is None:
                    if self.handle_unknown == 'error':
                        raise ValueError(
                            f'unseen category {value!r} in column {j}; '
                            'the encoder was fitted on training rows only')
                    code = st['unknown_index']
                out[i, offset + code] = 1.0
            offset += span
        return out

    def fit_transform(self, X, y=None):
        """Fit on **training** categories and transform them (never test)."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        if not self.fitted_:
            raise RuntimeError('encoder has not been fitted')
        if self.encoding == 'ordinal':
            return [f'x{j}' for j in range(self.n_features_in_)]
        names = []
        for j, st in enumerate(self.feature_stats_):
            names.extend(f'x{j}={c!r}' for c in st['categories'])
            names.append(f'x{j}=__unknown__')
            names.append(f'x{j}=__missing__')
        return names

    def get_stats(self):
        if not self.fitted_:
            raise RuntimeError('encoder has not been fitted')
        return {
            'encoding': self.encoding,
            'handle_unknown': self.handle_unknown,
            'n_features_in': self.n_features_in_,
            'n_train': self.n_train_,
            'train_fp': self.train_fp_,
            # Copy the category lists too: a caller mutating the returned
            # snapshot must not change the fitted encoder (and vice versa).
            'features': [dict(s,
                              categories=[_json_safe_category(c)
                                          for c in (s.get('categories') or [])],
                              keys=list(s.get('keys') or []))
                         for s in self.feature_stats_],
        }

    @classmethod
    def from_stats(cls, stats):
        """Rebuild a fitted encoder from :meth:`get_stats` output."""
        features, n_features = _persisted_features(
            stats, what='LeakageSafeCategoricalEncoder')
        what = 'LeakageSafeCategoricalEncoder'
        if 'encoding' not in stats:
            raise ValueError(f'{what}.from_stats: stats must declare encoding')
        obj = cls(encoding=stats['encoding'],
                  handle_unknown=stats.get('handle_unknown', 'unknown'))
        rebuilt = []
        for j, st in enumerate(features):
            categories = st.get('categories')
            if categories is None:
                categories = []
            elif not isinstance(categories, (list, tuple)):
                raise ValueError(
                    f'{what}.from_stats: feature {j} categories are not a list')
            else:
                categories = list(categories)
            ncat = _check_count('n_categories', st.get('n_categories'))
            if ncat != len(categories):
                raise ValueError(
                    f'{what}.from_stats: feature {j} declares {ncat} '
                    f'categories but carries {len(categories)}')
            unknown = _check_count('unknown_index', st.get('unknown_index'))
            missing = _check_count('missing_index', st.get('missing_index'))
            if unknown != ncat or missing != ncat + 1:
                raise ValueError(
                    f'{what}.from_stats: feature {j} has inconsistent '
                    f'unknown/missing indices ({unknown}/{missing}) for '
                    f'{ncat} categories')
            keys = st.get('keys')
            _keys_declared = keys is not None
            if keys is None:
                # Legacy stats carried no key table; the fallback below uses
                # the readable categories, which is only safe for values that
                # survived JSON unchanged.  Non-native categories were already
                # sanitised into dicts (tuple/bytes/datetime), so their
                # original identity cannot be reconstructed: fail closed
                # instead of silently treating every holdout value as unseen.
                if any(not (c is None or isinstance(c, (bool, int, float, str)))
                       for c in categories):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has non-native '
                        'categories but no persisted key table; rebuild the '
                        'encoder instead of degrading its identity')
                keys = []
            else:
                if not isinstance(keys, (list, tuple)):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a non-list '
                        'category-key table')
                keys = list(keys)
                # An explicitly declared table must describe every category.
                # Treating an empty/truncated list as "legacy" silently fell
                # back to the JSON-sanitised readable values and changed the
                # identity of non-native categories (e.g. tuples).
                if (len(keys) != ncat
                        or not all(isinstance(k, str) for k in keys)
                        or len(set(keys)) != len(keys)):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a malformed or '
                        'duplicate category-key table')
            rebuilt.append(dict(st, categories=categories, keys=keys))
        obj.feature_stats_ = rebuilt
        obj.n_features_in_ = int(n_features)
        obj.n_train_ = _check_count('n_train', stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


_MIX64_MASK = (1 << 64) - 1


def _target_key(value):
    """JSON-safe canonical key for a target-encoding category.

    ``get_stats`` has to round-trip through JSON so a fitted encoder can be
    persisted with a run manifest.  The historical dict used the raw category
    object as a key and a tuple sentinel for missing values; a tuple key is not
    JSON-serialisable, and a user category equal to that sentinel collided with
    a genuinely missing value.  Encode both into strings instead.

    Numeric objects keep Python's equality grouping (``True == 1``,
    ``1 == 1.0 == Fraction(1,1)``) by canonicalising integral/reals to an
    integer or exact rational text form.
    """
    if _category_is_missing(value):
        return '__missing__'
    if isinstance(value, (np.datetime64, np.timedelta64,
                          _datetime.date, _datetime.datetime,
                          _datetime.timedelta)):
        # ``np.timedelta64`` is a numbers.Integral, but int() either returns a
        # unit-dependent raw count or raises for larger units.  That made
        # timedelta(1 ns) collide with integer 1 and crashed on timedelta(1 D).
        # Persist the exact temporal identity instead.  Python datetime/date/
        # timedelta values compare equal to their NumPy counterparts, so they
        # must share that identity too.
        return 'p:' + _temporal_canonical_key(value).decode(
            'ascii', 'backslashreplace')
    if isinstance(value, (bool, np.bool_)):
        return 'i:' + str(int(value))
    if isinstance(value, _decimal.Decimal):
        # Decimal is a numbers.Number but not a numbers.Real/float, and
        # Fraction(Decimal('Infinity')) raises, so infinities used to fall
        # through to a type-qualified repr.  Python considers
        # Decimal('Infinity') == float('inf'), so both spellings must share
        # the same level identity (same as the generic float path below).
        if value.is_infinite():
            return 'f:+inf' if value > 0 else 'f:-inf'
    if isinstance(value, numbers.Integral):
        return 'i:' + str(int(value))
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        if c.imag == 0.0:
            return _target_key(c.real)
        return 'c:' + repr(c)
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        # Python float() rounds longdouble to 53 bits.  Distinct extended
        # values must not become one category; keep the float path only when
        # the value is exactly representable (so 1L == 1 still shares 'i:1').
        try:
            _f = float(value)
            _vfinite = bool(np.isfinite(value))
            if not _vfinite:
                # True inf/NaN keeps the shared non-finite identity.
                f = _f
            elif math.isfinite(_f) and np.longdouble(_f) == value:
                f = _f
            else:
                # Finite but beyond float64 range (or with extra mantissa
                # bits): keep the exact extended value, never collapse it to
                # the infinity key.
                return 'w:' + _wide_float_key(value).hex()
        except (TypeError, ValueError, OverflowError):
            return 'w:' + _wide_float_key(value).hex()
    elif isinstance(value, (float, np.floating)):
        f = float(value)
    else:
        f = None
    if f is not None:
        if math.isnan(f):
            return '__missing__'
        if math.isinf(f):
            return 'f:+inf' if f > 0 else 'f:-inf'
        if f.is_integer():
            return 'i:' + str(int(f))
        fr = Fraction(f)
        return 'r:' + str(fr.numerator) + '/' + str(fr.denominator)
    if isinstance(value, str):
        return 's:' + value
    if isinstance(value, bytes):
        return 'y:' + value.hex()
    if isinstance(value, tuple):
        # Length-prefix every element.  Joining the element keys with a bare
        # comma is ambiguous: ('a', 'b') and ('a,s:b',) both render as
        # ``t:(s:a,s:b)`` because a string category may itself contain the
        # delimiter.  Appending the length of each element key makes the
        # encoding self-delimiting at every nesting level.
        parts = [_target_key(v) for v in value]
        return ('t:' + str(len(parts)) + ':'
                + ''.join(str(len(part)) + ':' + part for part in parts))
    if isinstance(value, numbers.Number):
        # Decimal/Fraction (not covered by float/Integral above) also compare
        # exactly against integers/rationals, so use the same canonical form.
        try:
            fr = Fraction(value)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if fr.denominator == 1:
                return 'i:' + str(fr.numerator)
            return 'r:' + str(fr.numerator) + '/' + str(fr.denominator)
    return 'v:' + type(value).__module__ + '.' + type(value).__qualname__ + ':' + repr(value)


def _mix64(value):
    """SplitMix64 integer mixing for deterministic, process-independent folds."""
    z = (int(value) + 0x9E3779B97F4A7C15) & _MIX64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MIX64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MIX64_MASK
    return (z ^ (z >> 31)) & _MIX64_MASK


def _stable_key_hash(key):
    """Deterministic 64-bit hash of a group/category identity."""
    return int.from_bytes(
        hashlib.blake2b(repr(key).encode('utf-8', 'backslashreplace'),
                        digest_size=8).digest(), 'little')
