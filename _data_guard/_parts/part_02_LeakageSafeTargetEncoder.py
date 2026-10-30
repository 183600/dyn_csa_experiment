from __future__ import annotations


class LeakageSafeTargetEncoder:
    """Smoothed target encoding learned from training labels only.

    Target encoding replaces a categorical level with the mean target observed
    for that level.  It is a powerful feature-engineering step for
    high-cardinality columns, but it is also a classic leakage source: encoding
    a training row with a mean that includes that row's own label (or, for
    grouped/temporal data, a related row's label) trains the model on the
    target.  This implementation therefore has two distinct paths:

    * :meth:`transform` uses the **full** training-level table and is intended
      for validation/test rows only.  Holdout labels are never read.
    * :meth:`fit_transform` returns **out-of-fold** encodings for the training
      rows.  A row is encoded from rows in other folds only.  Exact duplicate
      feature rows share one fold (they are one observation, or one ambiguous
      annotation), so a duplicate's label cannot leak into its twin's encoding.
      When ``groups`` is supplied, whole groups share a fold, so a
      document/user cannot leak its label into another row of the same group.
      When ``time`` is supplied, folds are contiguous chronological blocks and
      encoding is expanding: a row is encoded only from strictly earlier
      blocks (the first block uses an uninformative prior of 0 because no past
      label exists).

    Missing categories are their own training level; an unseen holdout level
    falls back to the training prior.  Non-finite/missing labels are ignored
    when estimating a level mean, and categorical (non-numeric) targets are
    refused because a mean target would be undefined.
    """

    def __init__(self, *, smoothing=10.0, n_folds=5, seed=0):
        if isinstance(smoothing, (bool, np.bool_)):
            raise ValueError(
                f'smoothing must be a finite non-negative number, got '
                f'{smoothing!r}')
        try:
            smoothing = float(smoothing)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'smoothing must be a finite non-negative number, got '
                f'{smoothing!r}') from None
        if not (math.isfinite(smoothing) and smoothing >= 0.0):
            raise ValueError(
                f'smoothing must be a finite non-negative number, got '
                f'{smoothing!r}')
        n_folds = _check_count('n_folds', n_folds, positive=True)
        if n_folds < 2:
            raise ValueError(f'n_folds must be at least 2, got {n_folds}')
        seed = _check_count('seed', seed)
        self.smoothing = smoothing
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.fitted_ = False

    def _target_array(self, y, n):
        if y is None:
            raise ValueError('LeakageSafeTargetEncoder requires y')
        arr = np.asarray(y)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f'y must be 1-D with {n} rows, got shape {arr.shape}')
        out = np.full(n, np.nan, dtype=np.float64)
        for i, value in enumerate(arr.tolist()):
            if _is_missing_scalar(value) or isinstance(value, (bool, np.bool_)):
                # None/NaN/NaT/pandas NA are all missing targets.  pd.NA is
                # not a numbers.Number, so without this it fell through to the
                # categorical-label rejection.
                continue
            if isinstance(value, (np.datetime64, np.timedelta64)):
                # NaT is a missing label, exactly like None/NaN.  Without this
                # branch datetime64 NaT fell through to the type check and
                # aborted the fit, while timedelta64 NaT happened to be
                # skipped as a non-finite float: two spellings of the same
                # missing target had different outcomes.  A real datetime is
                # still not a regression target; a timedelta is a numeric
                # duration and keeps the historical float conversion.
                if np.isnat(value):
                    continue
                if isinstance(value, np.datetime64):
                    raise TypeError(
                        'LeakageSafeTargetEncoder requires a numeric target; '
                        f'got a {type(value).__name__} label at row {i}')
                # A timedelta is a numeric duration.  float() on a NumPy
                # timedelta is unit-dependent (it works for ns but raises for
                # day/hour units), so the old fallback silently dropped most
                # durations while keeping one raw unit count.  Canonicalise to
                # exact nanoseconds (attoseconds / 1e9) instead.
                try:
                    f = _temporal_attoseconds(value) / 1e9
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(f):
                    out[i] = f
                continue
            if isinstance(value, _datetime.timedelta):
                # Same canonical nanosecond duration for the Python twin.
                ns = ((int(value.days) * 86400 + int(value.seconds))
                      * 1000000 + int(value.microseconds)) * 1000
                f = float(ns)
                if math.isfinite(f):
                    out[i] = f
                continue
            if isinstance(value, (complex, np.complexfloating)):
                continue
            # Decimal is not a numbers.Number subclass, but it is an exact
            # real numeric target (and its NaN/Inf spellings must be skipped,
            # not rejected as categorical).  Handle it explicitly rather than
            # silently refusing a numeric label type this module already
            # canonicalises for row/category identity.
            if _numeric_is_nan(value):
                continue
            if not isinstance(value, (numbers.Number, _decimal.Decimal)):
                raise TypeError(
                    'LeakageSafeTargetEncoder requires a numeric target; '
                    f'got a {type(value).__name__} label at row {i}')
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(f):
                out[i] = f
        if not np.isfinite(out).any():
            raise ValueError(
                'LeakageSafeTargetEncoder found no finite numeric training '
                'labels; a target mean cannot be estimated')
        return out

    def _group_folds(self, groups, n):
        g = _as_1d(groups, 'groups', n)
        keys = [_group_key(v) for v in g.tolist()]
        unique = {}
        for key in keys:
            unique.setdefault(key, None)
        if len(unique) < 2:
            raise ValueError(
                'LeakageSafeTargetEncoder needs at least 2 groups for '
                'out-of-fold encoding')
        items = sorted(unique, key=lambda key: (
            _stable_key_hash(key) ^ int(self.seed), repr(key)))
        folds = np.empty(n, dtype=np.int64)
        fold_of = {key: i % self.n_folds for i, key in enumerate(items)}
        for i, key in enumerate(keys):
            folds[i] = fold_of[key]
        return folds

    def _coalesce_folds(self, folds, group_keys, row_keys):
        """Force every group/duplicate-row component into one fold.

        Group folds alone cannot see that the same feature row appears in two
        different groups; the duplicate's label would then encode its twin.
        Union rows sharing a group *or* an exact feature key and move each
        whole component to the smallest fold it already occupies (a pure fold
        merge, so no held row starts using a later/time label).
        """
        n = int(len(folds))
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

        if group_keys is not None:
            first = {}
            for i, key in enumerate(group_keys):
                j = first.setdefault(key, i)
                if j != i:
                    union(j, i)
        if row_keys is not None:
            first = {}
            for i, key in enumerate(row_keys):
                j = first.setdefault(key, i)
                if j != i:
                    union(j, i)
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        members_list = list(comps.values())
        conflict = any(len({int(folds[i]) for i in members}) > 1
                       for members in members_list if len(members) > 1)
        if not conflict:
            # No group/duplicate merge spans folds: keep the historical
            # assignment untouched (only duplicates already in one fold).
            for members in members_list:
                target = min(int(folds[i]) for i in members)
                for i in members:
                    folds[i] = target
            return folds
        # Merging components can make every merged component carry the same
        # minimum fold (e.g. two duplicate pairs originally in folds 0/1).
        # Re-assign whole components in first-row/first-fold order so the
        # merged design still has distinct out-of-fold references.
        _original = folds.copy()
        members_list.sort(key=lambda m: (min(int(folds[i]) for i in m), m[0]))
        for rank, members in enumerate(members_list):
            target = rank % self.n_folds
            for i in members:
                folds[i] = target
        if np.unique(folds).size < 2:
            # Every group is connected through duplicate rows.  Collapsing to
            # one fold would remove all out-of-fold references; keep the
            # original group/time folds and let fit_transform exclude the
            # duplicate labels from every reference set instead.
            return _original
        return folds

    def _row_folds(self, n, keys=None):
        """Deterministic row folds; exact-duplicate rows stay together.

        Without ``keys`` this is the historical per-row assignment.  With
        duplicate-detection keys, all rows sharing a feature key are assigned
        the same fold: otherwise one duplicate's label would be used to encode
        its twin, which is exactly the self-target leak that out-of-fold
        encoding is meant to prevent.  The representative (first occurrence)
        keeps the historical order, so a table without duplicates is unchanged.
        """
        if keys is None:
            reps = list(range(n))
            rep_of = None
        else:
            if len(keys) != n:
                raise ValueError(
                    f'internal duplicate-key count {len(keys)} does not match '
                    f'n={n}')
            rep_of = {}
            reps = []
            for i, key in enumerate(keys):
                if key not in rep_of:
                    rep_of[key] = i
                    reps.append(i)
        order = sorted(reps, key=lambda i: (
            _mix64(int(i) ^ int(self.seed)), i))
        rep_folds = {}
        for rank, i in enumerate(order):
            rep_folds[i] = rank % self.n_folds
        folds = np.empty(n, dtype=np.int64)
        if rep_of is None:
            for i in reps:
                folds[i] = rep_folds[i]
        else:
            for i, key in enumerate(keys):
                folds[i] = rep_folds[rep_of[key]]
        return folds

    def _group_time_folds(self, groups, time, n):
        """Whole-group, expanding chronological folds.

        Groups are ordered by their earliest timestamp (first appearance breaks
        ties) and consumed as contiguous blocks, so no group is split and every
        later fold contains only groups that start after the earlier folds.
        """
        time = _as_1d(time, 'time', n)
        _assert_order_finite(time, name='time')
        g = _as_1d(groups, 'groups', n)
        keys = [_group_key(v) for v in g.tolist()]
        first = {}
        members = {}
        minkey = {}
        maxkey = {}
        try:
            for i, key in enumerate(keys):
                if key not in first:
                    first[key] = i
                    minkey[key] = time[i]
                    maxkey[key] = time[i]
                else:
                    if time[i] < minkey[key]:
                        minkey[key] = time[i]
                    if time[i] > maxkey[key]:
                        maxkey[key] = time[i]
                members.setdefault(key, []).append(i)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                'time values are not mutually orderable; cannot order groups '
                'chronologically') from None
        if len(first) < self.n_folds:
            raise ValueError(
                f'cannot form {self.n_folds} non-empty group folds from '
                f'{len(first)} groups')
        try:
            order = sorted(first, key=lambda key: (minkey[key], first[key]))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                'time values are not mutually orderable; cannot order groups '
                'chronologically') from None
        sizes = [len(members[key]) for key in order]
        prefix = [0]
        for size in sizes:
            prefix.append(prefix[-1] + size)
        # Pick group boundaries near the balanced row targets, but reserve at
        # least one group for every later fold so no fold can be empty.
        bounds = [0]
        prev = 0
        for k in range(1, self.n_folds):
            lo = prev + 1
            hi = len(order) - (self.n_folds - k)
            target = (int(k) * int(n)) // int(self.n_folds)
            j = min(range(lo, hi + 1),
                    key=lambda pos: (abs(prefix[pos] - target), pos))
            bounds.append(j)
            prev = int(j)
        bounds.append(len(order))
        folds = np.empty(n, dtype=np.int64)
        prefix_max = None
        for k in range(self.n_folds):
            group_slice = order[bounds[k]:bounds[k + 1]]
            try:
                fold_min = min(minkey[key] for key in group_slice)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    'time values are not mutually orderable; cannot order '
                    'groups chronologically') from None
            if prefix_max is not None:
                try:
                    leaked = bool(prefix_max >= fold_min)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        'time values are not mutually orderable; cannot '
                        'order groups chronologically') from None
                if leaked:
                    raise ValueError(
                        'grouped chronological target-encoding folds have '
                        'overlapping group time windows across a fold '
                        f'boundary (earlier max {prefix_max!r} >= later min '
                        f'{fold_min!r}); future/past labels would leak across '
                        'the fold — merge/remove the overlapping groups or '
                        'use fewer folds')
            for key in group_slice:
                if prefix_max is None:
                    prefix_max = maxkey[key]
                elif maxkey[key] > prefix_max:
                    prefix_max = maxkey[key]
                folds[members[key]] = k
        return folds

    def _time_folds(self, time, n):
        time = _as_1d(time, 'time', n)
        _assert_order_finite(time, name='time')
        try:
            order = np.argsort(time, kind='stable')
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                'time values are not mutually orderable; cannot build '
                'chronological target-encoding folds') from None
        sorted_time = np.asarray(time)[order]
        if n < self.n_folds:
            raise ValueError(
                f'cannot form {self.n_folds} non-empty chronological folds '
                f'from {n} rows')
        try:
            changed = np.asarray(sorted_time[1:] != sorted_time[:-1]).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            changed = np.asarray([True] * (n - 1))
        cuts = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.nonzero(changed)[0].astype(np.int64) + 1,
            np.full(1, n, dtype=np.int64),
        ))
        # Target boundaries at equal row counts, snapped forward to a legal
        # timestamp-change cut so one event block never straddles two folds.
        chosen = []
        prev_idx = 0
        for k in range(1, self.n_folds):
            target = (int(k) * int(n)) // int(self.n_folds)
            pos = int(np.searchsorted(cuts, target, side='left'))
            remaining = self.n_folds - k
            pos = max(pos, prev_idx + 1)
            pos = min(pos, len(cuts) - 1 - remaining)
            if pos <= prev_idx or pos >= len(cuts) - 1:
                raise ValueError(
                    'cannot form the requested chronological target-encoding '
                    'folds without splitting an equal-timestamp block; use '
                    'fewer folds or pass groups')
            chosen.append(pos)
            prev_idx = pos
        bounds = [0] + chosen + [len(cuts) - 1]
        folds = np.empty(n, dtype=np.int64)
        for k in range(self.n_folds):
            start = int(cuts[bounds[k]])
            end = int(cuts[bounds[k + 1]])
            if start >= end:
                raise ValueError(
                    'a chronological target-encoding fold is empty; use '
                    'fewer folds')
            folds[order[start:end]] = k
        return folds

    def _fit_fold(self, Xarr, y, rows, cols=None):
        """Build per-column level statistics from ``rows`` only."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        values = y[rows]
        finite = np.isfinite(values)
        if finite.any():
            # ``_safe_mean`` first tries the ordinary mean (so normal data is
            # bit-identical) and only falls back to a max-scaled sum when the
            # finite labels are near float64's limit.
            prior = _safe_mean(values[finite])
        else:
            prior = 0.0
        stats = []
        for j in range(Xarr.shape[1]):
            entries = {}
            col = Xarr[:, j]
            for i in rows.tolist():
                if not math.isfinite(float(y[i])):
                    continue
                value = col[i]
                key = _target_key(value)
                if key in entries:
                    entries[key][1] += 1
                    entries[key][2] += float(y[i])
                else:
                    entries[key] = [value, 1, float(y[i])]
            out = {}
            for key, (value, count, total) in entries.items():
                if self.smoothing > 0.0:
                    encoded = ((total + self.smoothing * prior)
                               / (count + self.smoothing))
                else:
                    encoded = total / count
                if not math.isfinite(encoded):
                    # The running total overflowed even though every label is
                    # finite.  Recompute this level's mean from its own values
                    # in max-scaled space so the emitted target feature stays
                    # finite and the persisted table can be reloaded.
                    _vals = [float(y[i]) for i in rows.tolist()
                             if math.isfinite(float(y[i]))
                             and _target_key(col[i]) == key]
                    _mean = _safe_mean(np.asarray(_vals, dtype=np.float64))
                    if self.smoothing > 0.0:
                        _scale = max(abs(_mean), abs(prior))
                        if _scale > 0.0 and math.isfinite(_scale):
                            encoded = (((count * (_mean / _scale))
                                        + self.smoothing * (prior / _scale))
                                       / (count + self.smoothing)) * _scale
                        else:
                            encoded = _mean
                    else:
                        encoded = _mean
                    if not math.isfinite(encoded):
                        encoded = _mean if math.isfinite(_mean) else 0.0
                out[key] = float(encoded)
            stats.append({'lookup': out, 'prior': float(prior),
                          'n_reference_rows': int(rows.size),
                          'n_reference_labels': int(finite.sum())})
        return stats

    def fit(self, X, y, *, groups=None, time=None):
        """Learn the full training-level target table (for holdout transform)."""
        Xarr = _as_categorical_2d(X)
        n, p = Xarr.shape
        if n == 0:
            raise ValueError(
                'cannot fit a target encoder on zero training rows')
        yarr = self._target_array(y, n)
        if groups is not None:
            _as_1d(groups, 'groups', n)
        if time is not None:
            _as_1d(time, 'time', n)
            _assert_order_finite(time, name='time')
        self.feature_stats_ = self._fit_fold(
            Xarr, yarr, np.arange(n, dtype=np.int64))
        self.n_features_in_ = int(p)
        self.n_train_ = int(n)
        self.prior_ = _safe_mean(yarr[np.isfinite(yarr)])
        self.train_fp_ = _categorical_fingerprint(Xarr)
        self.train_row_set_fp_ = _categorical_row_set_fingerprint(Xarr)
        self.fitted_ = True
        return self

    def _check_transform_input(self, X):
        if not self.fitted_:
            raise RuntimeError(
                'LeakageSafeTargetEncoder.transform called before fit')
        Xarr = _as_categorical_2d(X)
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {Xarr.shape[1]} features, but the encoder was fitted '
                f'on {self.n_features_in_}')
        return Xarr

    @staticmethod
    def _apply(stats, Xarr, rows=None):
        n = Xarr.shape[0]
        if rows is None:
            rows = np.arange(n, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.empty((rows.size, len(stats)), dtype=np.float64)
        for j, st in enumerate(stats):
            lookup = st['lookup']
            prior = float(st['prior'])
            col = Xarr[:, j]
            for pos, i in enumerate(rows.tolist()):
                value = col[i]
                key = _target_key(value)
                out[pos, j] = lookup.get(key, prior)
        return out

    def transform(self, X, *, allow_train=False):
        """Encode holdout rows with the full training-level table.

        The full-table lookup includes every training label, so applying it to
        the training rows themselves is the classic target-encoding leak.  The
        training matrix is recognised by its fingerprint and refused; use
        :meth:`fit_transform` for training rows (out-of-fold encodings).  A
        deliberate diagnostic may pass ``allow_train=True``.
        """
        Xarr = self._check_transform_input(X)
        if not allow_train and self.train_fp_ is not None:
            _same_rows = _categorical_fingerprint(Xarr) == self.train_fp_
            if (not _same_rows and self.train_row_set_fp_ is not None
                    and int(Xarr.shape[0]) == int(self.n_train_)
                    and _categorical_row_set_fingerprint(Xarr)
                    == self.train_row_set_fp_):
                _same_rows = True
            if _same_rows:
                raise ValueError(
                    'LeakageSafeTargetEncoder.transform was given the exact '
                    'training rows (possibly reordered), but its full-table '
                    'encoding contains their own labels; use fit_transform for '
                    'training rows (out-of-fold encodings) and reserve '
                    'transform for validation/test, or pass allow_train=True '
                    'for an explicit diagnostic')
        return self._apply(self.feature_stats_, Xarr)

    def fit_transform(self, X, y, *, groups=None, time=None):
        """Return out-of-fold training encodings (safe for model fitting)."""
        Xarr = _as_categorical_2d(X)
        n = Xarr.shape[0]
        if n < 2:
            # Out-of-fold encoding needs at least one row outside the held
            # fold.  With a single row the "encoding" would be an arbitrary
            # prior with no reference labels, silently pretending the target
            # feature was estimated.  Refuse instead of emitting that value.
            raise ValueError(
                'LeakageSafeTargetEncoder.fit_transform needs at least 2 '
                'training rows so a held row has out-of-fold reference '
                f'labels, got {n}')
        self.fit(X, y, groups=groups, time=time)
        yarr = self._target_array(y, n)
        # Exact duplicate feature rows are one observation.  This must hold on
        # every fold policy, including grouped/time folds: a duplicate in
        # another group would otherwise train its twin's encoding.
        _row_keys_n = _row_keys(Xarr)
        _group_keys_n = None
        if groups is not None:
            _g = _as_1d(groups, 'groups', n)
            _group_keys_n = [_group_key(v) for v in _g.tolist()]
        if time is not None and groups is not None:
            folds = self._group_time_folds(groups, time, n)
            policy = 'grouped_chronological_expanding_oof'
        elif time is not None:
            folds = self._time_folds(time, n)
            policy = 'chronological_expanding_oof'
        elif groups is not None:
            folds = self._group_folds(groups, n)
            policy = 'grouped_oof'
        else:
            folds = self._row_folds(n, keys=_row_keys_n)
            policy = 'row_oof'
        if policy == 'grouped_oof':
            # Group-only folds are free to merge connected groups.  Time folds
            # must stay chronological, so they keep their blocks and remove
            # duplicate labels from the reference set below instead.
            folds = self._coalesce_folds(folds, _group_keys_n, _row_keys_n)
        if np.unique(folds).size < 2:
            # Every row is one duplicate/group component: that fold has no
            # out-of-fold reference rows.  Silently emitting the prior would
            # fabricate an encoding, exactly like the single-row case.
            raise ValueError(
                'LeakageSafeTargetEncoder.fit_transform needs at least 2 '
                'distinct feature/group components so a held component has '
                'out-of-fold reference labels')
        out = np.zeros((n, Xarr.shape[1]), dtype=np.float64)
        # Duplicate feature rows are one observation even when the chosen
        # group/time folds could not be merged (e.g. every group is connected
        # by duplicates).  Never let such a twin contribute to a held row's
        # reference mean.
        _dup_key_counts = {}
        for _rk in _row_keys_n:
            _dup_key_counts[_rk] = _dup_key_counts.get(_rk, 0) + 1
        for k in range(int(folds.max()) + 1):
            held = np.nonzero(folds == k)[0]
            if held.size == 0:
                continue
            if policy in ('chronological_expanding_oof',
                          'grouped_chronological_expanding_oof'):
                # Only strictly earlier blocks may contribute a label.
                ref = np.nonzero(folds < k)[0]
            else:
                ref = np.nonzero(folds != k)[0]
            _held_keys = {_row_keys_n[i] for i in held}
            _exclude = np.fromiter(
                (_dup_key_counts[_row_keys_n[i]] > 1
                 and _row_keys_n[i] in _held_keys for i in ref),
                dtype=bool, count=int(ref.size))
            if _exclude.any():
                ref = ref[~_exclude]
            stats = self._fit_fold(Xarr, yarr, ref)
            out[held] = self._apply(stats, Xarr, held)
        self.oof_policy_ = policy
        self.oof_fold_fp_ = hashlib.sha256(
            np.ascontiguousarray(folds).tobytes()).hexdigest()[:16]
        return out

    def get_feature_names(self):
        if not self.fitted_:
            raise RuntimeError('target encoder has not been fitted')
        return [f'x{j}_target' for j in range(self.n_features_in_)]

    def get_stats(self):
        if not self.fitted_:
            raise RuntimeError('target encoder has not been fitted')
        return {
            'smoothing': float(self.smoothing),
            'n_folds': int(self.n_folds),
            'seed': int(self.seed),
            'n_features_in': self.n_features_in_,
            'n_train': self.n_train_,
            'prior': float(self.prior_),
            'train_fp': self.train_fp_,
            'train_row_set_fp': getattr(self, 'train_row_set_fp_', None),
            'oof_policy': getattr(self, 'oof_policy_', None),
            'oof_fold_fp': getattr(self, 'oof_fold_fp_', None),
            'features': [
                {
                    'lookup': dict(st['lookup']),
                    'prior': float(st['prior']),
                    'n_reference_rows': int(st['n_reference_rows']),
                    'n_reference_labels': int(st['n_reference_labels']),
                }
                for st in self.feature_stats_
            ],
        }

    @classmethod
    def from_stats(cls, stats):
        """Rebuild a fitted target encoder from :meth:`get_stats` output."""
        features, n_features = _persisted_features(
            stats, what='LeakageSafeTargetEncoder')
        what = 'LeakageSafeTargetEncoder'
        rebuilt = []
        for j, st in enumerate(features):
            lookup = st.get('lookup')
            if not isinstance(lookup, dict):
                raise ValueError(
                    f'{what}.from_stats: feature {j} has no lookup mapping')
            clean_lookup = {}
            for key, value in lookup.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a non-string '
                        'or empty lookup key')
                try:
                    value = float(value)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a non-numeric '
                        'lookup value') from None
                if not math.isfinite(value):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a non-finite '
                        'lookup value')
                clean_lookup[key] = value
            if 'prior' not in st:
                raise ValueError(
                    f'{what}.from_stats: feature {j} has no prior')
            prior = _stat_float(st, 'prior', what=what, index=j)
            n_rows = _check_count('n_reference_rows',
                                  st.get('n_reference_rows', 0))
            n_labels = _check_count('n_reference_labels',
                                    st.get('n_reference_labels', 0))
            if n_labels > n_rows:
                raise ValueError(
                    f'{what}.from_stats: feature {j} has more reference '
                    'labels than reference rows')
            rebuilt.append({'lookup': clean_lookup, 'prior': prior,
                            'n_reference_rows': n_rows,
                            'n_reference_labels': n_labels})
        obj = cls(smoothing=stats.get('smoothing', 10.0),
                  n_folds=stats.get('n_folds', 5),
                  seed=stats.get('seed', 0))
        obj.feature_stats_ = rebuilt
        obj.n_features_in_ = int(n_features)
        obj.n_train_ = _check_count('n_train', stats.get('n_train', 0))
        obj.prior_ = _stat_float(stats, 'prior',
                                 what=what, index='top-level')
        # The full-table lookup contains every training row's own label, so
        # ``transform`` refuses an exact training matrix using this
        # fingerprint.  A persisted artifact without it would silently lose
        # that guard and expose the classic in-sample target-encoding leak;
        # fail closed instead of accepting a degraded encoder.
        _train_fp = stats.get('train_fp')
        if obj.n_train_ > 0:
            if not isinstance(_train_fp, str) or not _train_fp:
                raise ValueError(
                    f'{what}.from_stats: n_train={obj.n_train_} requires a '
                    'non-empty train_fp so in-sample target encoding can be '
                    'refused')
        elif _train_fp is not None and not isinstance(_train_fp, str):
            raise ValueError(
                f'{what}.from_stats: train_fp must be a string or None')
        # A forged n_train smaller than the reference table also disables the
        # reordered-training-row check (it compares the input row count with
        # n_train).  The full-table fit builds every feature's reference table
        # from exactly the training rows, so an inconsistent count must fail
        # rather than weaken the leak guard.
        for j, st in enumerate(rebuilt):
            if int(st['n_reference_rows']) != obj.n_train_:
                raise ValueError(
                    f'{what}.from_stats: feature {j} references '
                    f"{st['n_reference_rows']} rows but n_train="
                    f'{obj.n_train_}; the persisted table was not fit on the '
                    'declared training rows')
        _row_set_fp = stats.get('train_row_set_fp')
        if _row_set_fp is not None and not isinstance(_row_set_fp, str):
            raise ValueError(
                f'{what}.from_stats: train_row_set_fp must be a string or None')
        if obj.n_train_ > 0 and (not isinstance(_row_set_fp, str)
                                 or not _row_set_fp.strip()):
            # Without the order-insensitive row-set digest, transform() can no
            # longer recognise the same training rows in a different order and
            # would serve each row's own label back as its feature.
            raise ValueError(
                f'{what}.from_stats: n_train={obj.n_train_} requires a '
                'non-empty train_row_set_fp so a reordered training matrix is '
                'still refused')
        obj.train_fp_ = _train_fp
        obj.train_row_set_fp_ = _row_set_fp
        obj.oof_policy_ = stats.get('oof_policy')
        obj.oof_fold_fp_ = stats.get('oof_fold_fp')
        obj.fitted_ = True
        return obj


def prepare_target_features(X_train, y_train, X_val=None, X_test=None,
                            *, groups=None, time=None, **kwargs):
    """Out-of-fold target encoding for train, full-table transform for holdouts."""
    encoder = LeakageSafeTargetEncoder(**kwargs)
    out = {'train': encoder.fit_transform(
               X_train, y_train, groups=groups, time=time),
           'encoder': encoder}
    if X_val is not None:
        out['val'] = encoder.transform(X_val)
    if X_test is not None:
        out['test'] = encoder.transform(X_test)
    return out


class LeakageSafeFrequencyEncoder:
    """Count/frequency encode categories using training rows only.

    High-cardinality categorical features cannot be one-hot encoded without
    exploding the feature width.  A frequency encoding is a compact
    alternative that is also safe for holdouts: the category counts are
    learned from ``X_train`` and ``transform`` only looks values up in that
    fixed table.  An unseen or missing holdout category encodes as ``0``
    (or raises under ``handle_unknown='error'``) and never widens or shifts
    the training feature matrix.  No label is used, so this transform cannot
    leak the target.
    """

    def __init__(self, *, normalize=True, handle_unknown='zero'):
        if not isinstance(normalize, (bool, np.bool_)):
            raise ValueError(f'normalize must be a boolean, got {normalize!r}')
        if handle_unknown not in ('zero', 'error'):
            raise ValueError(
                f"handle_unknown must be 'zero' or 'error', got "
                f'{handle_unknown!r}')
        self.normalize = bool(normalize)
        self.handle_unknown = handle_unknown
        self.fitted_ = False

    def fit(self, X, y=None):
        Xarr = _as_categorical_2d(X)
        n, p = Xarr.shape
        if n == 0:
            raise ValueError(
                'cannot fit a frequency encoder on zero training rows')
        stats = []
        for j in range(p):
            entries = {}
            n_missing = 0
            for value in Xarr[:, j].tolist():
                if _category_is_missing(value):
                    n_missing += 1
                    continue
                # Group by the persisted key, not the raw hash: equal
                # temporal spellings (date vs np.datetime64, ...) share one
                # level, and the transform lookup rebuilt from these keys
                # cannot collapse two counts into one.
                key = _target_key(value)
                if key in entries:
                    entries[key]['count'] += 1
                else:
                    entries[key] = {'value': value, 'count': 1,
                                    'key': key}
            ordered = sorted(entries.values(),
                             key=lambda e: (type(e['value']).__name__,
                                            repr(e['value'])))
            n_observed = int(sum(e['count'] for e in ordered))
            stats.append({
                'entries': [{'value': e['value'], 'count': int(e['count']),
                             'key': e['key']} for e in ordered],
                'n_observed': n_observed,
                'n_missing': int(n_missing),
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
                'LeakageSafeFrequencyEncoder.transform called before fit')
        Xarr = _as_categorical_2d(X)
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {Xarr.shape[1]} features, but the encoder was fitted '
                f'on {self.n_features_in_}')
        return Xarr

    def transform(self, X):
        Xarr = self._check_transform_input(X)
        n = int(Xarr.shape[0])
        out = np.zeros((n, self.n_features_in_), dtype=np.float64)
        for j, st in enumerate(self.feature_stats_):
            _entries = list(st.get('entries') or [])
            if _entries and all(isinstance(e.get('key'), str) for e in _entries):
                lookup = {e['key']: int(e['count']) for e in _entries}
                keyfn = _target_key
            else:
                lookup = {_hashable_label(e['value']): int(e['count'])
                          for e in _entries}
                keyfn = _hashable_label
            denom = int(st['n_observed'])
            for i, value in enumerate(Xarr[:, j].tolist()):
                if _category_is_missing(value):
                    # Missing is not a learned frequency level.  Under the
                    # strict contract it is rejected just like any other value
                    # that the training table cannot encode; the default
                    # contract maps it to the reserved zero level.
                    if self.handle_unknown == 'error':
                        raise ValueError(
                            f'missing value in column {j} under '
                            "handle_unknown='error'; the frequency table "
                            'was fitted on training rows only')
                    continue
                count = lookup.get(keyfn(value))
                if count is None:
                    if self.handle_unknown == 'error':
                        raise ValueError(
                            f'unseen category {value!r} in column {j}; the '
                            'frequency table was fitted on training rows only')
                    continue
                out[i, j] = (float(count) / float(denom)
                             if self.normalize and denom > 0
                             else float(count))
        return out

    def fit_transform(self, X, y=None):
        """Fit on **training** categories and transform them (never test)."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        if not self.fitted_:
            raise RuntimeError('frequency encoder has not been fitted')
        return [f'x{j}_freq' for j in range(self.n_features_in_)]

    def get_stats(self):
        if not self.fitted_:
            raise RuntimeError('frequency encoder has not been fitted')
        return {
            'normalize': bool(self.normalize),
            'handle_unknown': self.handle_unknown,
            'n_features_in': self.n_features_in_,
            'n_train': self.n_train_,
            'train_fp': self.train_fp_,
            'features': [
                {
                    'entries': [dict(e, value=_json_safe_category(e.get('value')))
                                for e in st.get('entries') or []],
                    'n_observed': int(st.get('n_observed', 0)),
                    'n_missing': int(st.get('n_missing', 0)),
                }
                for st in self.feature_stats_
            ],
        }

    @classmethod
    def from_stats(cls, stats):
        """Rebuild a fitted frequency encoder from :meth:`get_stats` output."""
        features, n_features = _persisted_features(
            stats, what='LeakageSafeFrequencyEncoder')
        what = 'LeakageSafeFrequencyEncoder'
        obj = cls(normalize=stats.get('normalize', True),
                  handle_unknown=stats.get('handle_unknown', 'zero'))
        rebuilt = []
        for j, st in enumerate(features):
            entries = st.get('entries')
            if not isinstance(entries, list):
                raise ValueError(
                    f'{what}.from_stats: feature {j} has no entry table')
            clean_entries = []
            n_observed = 0
            for pos, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} entry {pos} is not '
                        'a mapping')
                count = _check_count(f'entries[{pos}].count',
                                     entry.get('count'))
                if count == 0:
                    raise ValueError(
                        f'{what}.from_stats: feature {j} entry {pos} has a '
                        'zero count')
                key = entry.get('key')
                if key is not None and (not isinstance(key, str) or not key):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} entry {pos} has a '
                        'non-string or empty key')
                if 'value' not in entry and key is None:
                    raise ValueError(
                        f'{what}.from_stats: feature {j} entry {pos} has '
                        'neither a value nor a key')
                clean_entries.append(dict(entry, count=count))
                n_observed += count
            declared = _check_count('n_observed', st.get('n_observed'))
            if declared != n_observed:
                raise ValueError(
                    f'{what}.from_stats: feature {j} declares n_observed='
                    f'{declared} but its entries sum to {n_observed}')
            n_missing = _check_count('n_missing', st.get('n_missing', 0))
            keys = [e.get('key') for e in clean_entries]
            if any(k is not None for k in keys):
                if not all(isinstance(k, str) for k in keys):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} mixes keyed and '
                        'unkeyed entries')
                if len(set(keys)) != len(keys):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has duplicate '
                        'entry keys')
            elif any(not (e.get('value') is None
                          or isinstance(e.get('value'), (bool, int, float, str)))
                     for e in clean_entries):
                # Same legacy-identity rule as the categorical encoder: a
                # sanitised non-native value cannot be matched to the caller's
                # original tuple/bytes/datetime level.
                raise ValueError(
                    f'{what}.from_stats: feature {j} has non-native entries '
                    'but no persisted key table; rebuild the encoder instead '
                    'of degrading its identity')
            rebuilt.append({'entries': clean_entries,
                            'n_observed': n_observed,
                            'n_missing': n_missing})
        obj.feature_stats_ = rebuilt
        obj.n_features_in_ = int(n_features)
        obj.n_train_ = _check_count('n_train', stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


def _fit_and_transform(encoder, X_train, X_val, X_test, fitted_key):
    """Fit-on-train / transform-holdouts result bundle.

    ``fitted_key`` keeps the historical return key for the fitted object
    (``encoder`` for the feature encoders, ``preprocessor`` for the
    preprocessor) and its position in the dictionary.
    """
    out = {'train': encoder.transform(X_train), fitted_key: encoder}
    if X_val is not None:
        out['val'] = encoder.transform(X_val)
    if X_test is not None:
        out['test'] = encoder.transform(X_test)
    return out


def prepare_frequency_features(X_train, X_val=None, X_test=None, **kwargs):
    """Fit frequency tables on train, then transform the holdouts."""
    encoder = LeakageSafeFrequencyEncoder(**kwargs).fit(X_train)
    return _fit_and_transform(encoder, X_train, X_val, X_test, 'encoder')


def fit_categorical_encoder(X_train, **kwargs):
    """Fit a categorical encoder on training rows only."""
    return LeakageSafeCategoricalEncoder(**kwargs).fit(X_train)


def prepare_categorical_features(X_train, X_val=None, X_test=None, **kwargs):
    """Encode train categories first, then transform the holdouts."""
    encoder = LeakageSafeCategoricalEncoder(**kwargs).fit(X_train)
    return _fit_and_transform(encoder, X_train, X_val, X_test, 'encoder')


# Conventional short alias.
SafeScaler = LeakageSafePreprocessor


def fit_preprocessor(X_train, **kwargs):
    """Fit a preprocessor on training rows only."""
    return LeakageSafePreprocessor(**kwargs).fit(X_train)


def prepare_features(X_train, X_val=None, X_test=None, **kwargs):
    """Split-aware preprocessing helper.

    The preprocessor is fitted on ``X_train`` and then applied to the holdouts
    with ``transform``.  No holdout row contributes to any statistic.
    """
    prep = LeakageSafePreprocessor(**kwargs).fit(X_train)
    return _fit_and_transform(prep, X_train, X_val, X_test, 'preprocessor')


def _stats_match_float(a, b, *, atol, rtol):
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)


def _holdout_parts(values, *, name='X_holdout'):
    """Distinguish one list-of-rows table from several holdout tables.

    A Python list is ambiguous: ``[[1, 2], [3, 4]]`` is a 2-row table, while
    ``[val_matrix, test_matrix]`` is two tables.  Shape settles it: ndim<=2 is
    one table, ndim==3 (or a ragged list that NumPy cannot stack) is a list of
    parts.  Treating every list as parts silently transposed a list-of-rows
    holdout into a column vector.
    """
    if not isinstance(values, (list, tuple)):
        return [values]
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        return list(values)
    if arr.ndim <= 2:
        return [values]
    if arr.ndim == 3:
        return list(values)
    # Higher-rank input is not a valid table or a list of 2-D tables; let the
    # per-part ``_as_*_2d`` validator produce the precise error.
    return list(values)


def audit_preprocessor_fit(preprocessor, X_train, X_holdout=None, *,
                           atol=1e-10, rtol=1e-8):
    """Prove that a fitted preprocessor learned its statistics on TRAIN only.

    The classic preprocessing leak is fitting a scaler/imputer on the full
    table and then calling ``transform`` on each split.  Index-disjoint splits
    do not catch this: every column's mean/std/quantiles already contains
    holdout rows.  Given the training rows (and optionally the holdout rows),
    this audit recomputes the preprocessor's statistics from scratch:

    * ``train_only_verified`` — the persisted stats and ``train_fp`` equal a
      fresh fit on ``X_train``;
    * ``leak_detected`` — they do NOT match train, but DO match a fit on
      ``train + holdout`` (the holdout rows were used to fit it);
    * ``unverifiable`` — neither matches (the object was fitted on another
      table, was manually edited, or ``X_holdout`` was not supplied).

    Only a verified status is evidence that a reported preprocessing step was
    fit on the training split.  The audit never inspects holdout labels and
    never changes the fitted object.
    """
    if not getattr(preprocessor, 'fitted_', False):
        raise ValueError('audit_preprocessor_fit needs a fitted preprocessor')
    try:
        stats = preprocessor.get_stats()
    except Exception as e:
        raise ValueError(
            f'audit_preprocessor_fit: preprocessor.get_stats() failed '
            f'({type(e).__name__}: {e})') from None
    if not isinstance(stats, dict) or not isinstance(stats.get('features'), list):
        raise ValueError('audit_preprocessor_fit: malformed preprocessor stats')
    if isinstance(atol, (bool, np.bool_)) or isinstance(rtol, (bool, np.bool_)):
        raise ValueError('audit_preprocessor_fit: tolerances must be numbers')
    try:
        atol, rtol = float(atol), float(rtol)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('audit_preprocessor_fit: tolerances must be finite numbers') from None
    if not (math.isfinite(atol) and atol >= 0.0 and math.isfinite(rtol) and rtol >= 0.0):
        raise ValueError('audit_preprocessor_fit: tolerances must be finite and non-negative')
    Xtr = _as_float_2d(X_train, name='X_train')
    if Xtr.shape[0] == 0:
        raise ValueError('audit_preprocessor_fit: X_train is empty')
    # Refit on the ORIGINAL dtype, not the float64 view.  The fitted
    # ``train_fp`` hashes the table it actually saw; for int64/timestamp
    # values above 2**53 the native-int branch preserves bits that float64
    # rounds away.  Comparing against a float refit made a genuine train-only
    # fit permanently ``unverifiable`` (and weakened the leak check for the
    # same rows).
    Xtr_native = _as_numeric_2d(X_train, name='X_train')
    if Xtr_native.shape != Xtr.shape:
        raise ValueError('audit_preprocessor_fit: X_train changed shape during validation')

    def _ref_stats(rows):
        ref = LeakageSafePreprocessor(
            scale=stats.get('scale', 'standard'),
            missing=stats.get('missing', 'median'),
            clip_quantiles=(tuple(stats['clip_quantiles'])
                            if stats.get('clip_quantiles') else None),
            clip_z=stats.get('clip_z'),
            eps=stats.get('eps', _EPS))
        return ref.fit(rows).get_stats()

    def _compare(ref):
        if int(ref.get('n_features_in', -1)) != int(stats.get('n_features_in', -2)):
            return False, list(range(max(len(stats.get('features') or []),
                                         len(ref.get('features') or []))))
        if str(ref.get('train_fp')) != str(stats.get('train_fp')):
            return False, list(range(len(stats.get('features') or [])))
        got_features = stats.get('features') or []
        ref_features = ref.get('features') or []
        if len(got_features) != len(ref_features):
            # zip() would silently truncate to the shorter list, so a stats
            # artifact with a dropped/extra column could be certified as a
            # train-only fit without ever comparing that column.
            return False, list(range(max(len(got_features), len(ref_features))))
        bad = []
        for j, (got, want) in enumerate(zip(got_features, ref_features)):
            keys = ('fill', 'clip_lo', 'clip_hi', 'center', 'factor',
                    'mean', 'std', 'median', 'min', 'max')
            if not all(_stats_match_float(got.get(k), want.get(k),
                                          atol=atol, rtol=rtol) for k in keys):
                bad.append(j)
        return (not bad), bad

    train_ref = _ref_stats(Xtr_native)
    train_ok, train_bad = _compare(train_ref)
    out = {
        'status': 'unverifiable',
        'leak_detected': False,
        'stats_match_train': bool(train_ok),
        'stats_match_train_plus_holdout': None,
        'train_fp_match': (str(stats.get('train_fp')) == str(train_ref.get('train_fp'))),
        'n_train': int(Xtr.shape[0]),
        'n_features': int(Xtr.shape[1]),
        'n_holdout': None,
        'mismatched_features_on_train': train_bad,
        'note': ('stats and train_fp equal a fresh train-only fit; the '
                 'preprocessing step is verified to have used training rows only'
                 if train_ok else
                 'stats do not equal a fresh fit on this X_train; pass the '
                 'holdout rows to distinguish a different training table from '
                 'the classic train+holdout leak'),
    }
    if train_ok:
        out['status'] = 'train_only_verified'
        return out
    if X_holdout is None:
        return out
    parts = _holdout_parts(X_holdout)
    # Validate exactly as before (complex/ragged input is rejected), but keep
    # the native dtype for the refit so the fingerprint still sees exact
    # int64/timestamp bits above 2**53.
    _parts_native = []
    for part in parts:
        _as_float_2d(part, name='X_holdout')
        _parts_native.append(_as_numeric_2d(part, name='X_holdout'))
    parts = [p for p in _parts_native if p.shape[0] > 0]
    if not parts:
        return out
    if any(p.shape[1] != Xtr.shape[1] for p in parts):
        raise ValueError(
            'audit_preprocessor_fit: holdout features do not match X_train')
    Xh = np.vstack(parts)
    full_ref = _ref_stats(np.vstack([Xtr_native, Xh]))
    full_ok, full_bad = _compare(full_ref)
    out.update({'n_holdout': int(Xh.shape[0]),
                'stats_match_train_plus_holdout': bool(full_ok),
                'mismatched_features_on_train_plus_holdout': full_bad})
    if full_ok:
        out['status'] = 'leak_detected'
        out['leak_detected'] = True
        out['note'] = ('the fitted statistics match a fit on train + holdout, '
                       'not a train-only fit: holdout rows participated in '
                       'imputation/clipping/scaling statistics; refit on train '
                       'and transform the holdouts')
    else:
        out['note'] = ('the fitted statistics match neither a train-only nor a '
                       'train+holdout fit on the supplied rows; the fitting '
                       'table is not verifiable from these arrays')
    return out


def audit_encoder_fit(encoder, X_train, X_holdout=None):
    """Prove that a categorical/frequency encoder was fit on TRAIN only.

    The analogue of :func:`audit_preprocessor_fit` for encoders whose fitted
    state is a category set or a count table.  A holdout category silently
    added to the training vocabulary is leakage even when the encoded values
    happen to be in range: the feature space was defined with holdout data.

    Supports :class:`LeakageSafeCategoricalEncoder` and
    :class:`LeakageSafeFrequencyEncoder`.  Returns ``train_only_verified``,
    ``leak_detected`` (fitted state equals a train+holdout fit) or
    ``unverifiable``.  The target encoder is intentionally excluded: its
    training features are out-of-fold by construction, so comparing a single
    fitted table would not describe the encodings actually used.
    """
    if isinstance(encoder, LeakageSafeCategoricalEncoder):
        kind = 'categorical'
        def _make(stats):
            return LeakageSafeCategoricalEncoder(
                encoding=stats.get('encoding', 'ordinal'),
                handle_unknown=stats.get('handle_unknown', 'unknown'))
        def _fingerprint(stats):
            feats = []
            for st in stats.get('features') or []:
                categories = list(st.get('categories') or [])
                keys = st.get('keys')
                if (isinstance(keys, list) and len(keys) == len(categories)
                        and all(isinstance(k, str) for k in keys)):
                    # Persisted keys share the encoder's canonical identity,
                    # so equal temporal spellings do not make the audit
                    # "unverifiable" merely because their readable categories
                    # were sanitized differently.
                    cat_id = tuple(keys)
                else:
                    cat_id = tuple((type(v).__name__, repr(v))
                                   for v in categories)
                feats.append((
                    cat_id,
                    int(st.get('n_categories', 0)),
                    int(st.get('unknown_index', 0)),
                    int(st.get('missing_index', 0))))
            return tuple(feats)
    elif isinstance(encoder, LeakageSafeFrequencyEncoder):
        kind = 'frequency'
        def _make(stats):
            return LeakageSafeFrequencyEncoder(
                normalize=stats.get('normalize', True),
                handle_unknown=stats.get('handle_unknown', 'zero'))
        def _fingerprint(stats):
            feats = []
            for st in stats.get('features') or []:
                entries = []
                for e in (st.get('entries') or []):
                    key = e.get('key')
                    ident = (key if isinstance(key, str)
                             else (type(e.get('value')).__name__,
                                   repr(e.get('value'))))
                    entries.append((ident, int(e.get('count', 0))))
                feats.append((tuple(entries), int(st.get('n_observed', 0)),
                              int(st.get('n_missing', 0))))
            return tuple(feats)
    else:
        raise TypeError(
            'audit_encoder_fit supports LeakageSafeCategoricalEncoder and '
            f'LeakageSafeFrequencyEncoder, got {type(encoder).__name__}')
    if not getattr(encoder, 'fitted_', False):
        raise ValueError('audit_encoder_fit needs a fitted encoder')
    stats = encoder.get_stats()
    Xtr = _as_categorical_2d(X_train, name='X_train')
    if Xtr.shape[0] == 0:
        raise ValueError('audit_encoder_fit: X_train is empty')

    def _ref(rows):
        return _make(stats).fit(rows).get_stats()

    def _ok(ref):
        return (int(ref.get('n_features_in', -1)) == int(stats.get('n_features_in', -2))
                and str(ref.get('train_fp')) == str(stats.get('train_fp'))
                and _fingerprint(ref) == _fingerprint(stats))

    train_ref = _ref(Xtr)
    out = {'kind': kind, 'status': 'unverifiable', 'leak_detected': False,
           'stats_match_train': bool(_ok(train_ref)),
           'stats_match_train_plus_holdout': None,
           'n_train': int(Xtr.shape[0]), 'n_features': int(Xtr.shape[1]),
           'n_holdout': None,
           'note': ('the fitted categories/counts equal a fresh train-only fit'
                    if _ok(train_ref) else
                    'the fitted state does not equal a fresh fit on this '
                    'X_train; pass the holdout rows to distinguish a different '
                    'training table from a holdout-widened vocabulary')}
    if out['stats_match_train']:
        out['status'] = 'train_only_verified'
        return out
    if X_holdout is None:
        return out
    parts = _holdout_parts(X_holdout)
    parts = [_as_categorical_2d(part, name='X_holdout') for part in parts]
    parts = [p for p in parts if p.shape[0] > 0]
    if not parts:
        return out
    if any(p.shape[1] != Xtr.shape[1] for p in parts):
        raise ValueError('audit_encoder_fit: holdout features do not match X_train')
    Xh = np.vstack(parts)
    full_ok = _ok(_ref(np.vstack([Xtr, Xh])))
    out.update({'n_holdout': int(Xh.shape[0]),
                'stats_match_train_plus_holdout': bool(full_ok)})
    if full_ok:
        out['status'] = 'leak_detected'
        out['leak_detected'] = True
        out['note'] = ('the encoder state matches a fit on train + holdout, not '
                       'a train-only fit: holdout categories/counts widened '
                       'the feature space; refit on train and transform the '
                       'holdouts')
    else:
        out['note'] = ('the fitted state matches neither a train-only nor a '
                       'train+holdout fit on the supplied rows; the fitting '
                       'table is not verifiable from these arrays')
    return out
