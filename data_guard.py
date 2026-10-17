"""Leakage-safe data splitting and preprocessing.

This module is deliberately dependency-free (NumPy only) so the same contract
can be used by the language-model corpus pipeline and by any future tabular /
time-series / grouped-data experiments in this repository.

The rules encoded here are:

* **Split first, fit later.**  Every statistic used to impute, clip or scale a
  feature is learned from the training rows only.  ``transform`` never updates
  those statistics; ``fit_transform`` is a training-set convenience only.
* **Temporal data is split in time order.**  The split is contiguous and never
  shuffled.  Shuffling *within* the training split during SGD is a separate,
  allowed operation.
* **Groups stay together.**  When rows share a group id (a document, user,
  object, ...), the whole group is assigned to exactly one split.
* **Tests are sealed.**  Helpers here never inspect test labels to choose a
  split, imputation value, clip bound or scale factor.  Test data is only
  transformed with training statistics.

The module also provides data-quality diagnostics for the failure modes that
commonly invalidate a split: NaN/Inf, exact duplicates (within and across
splits), label noise on duplicate rows, outliers, and train/holdout
distribution shift.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

__all__ = [
    'SPLIT_NAMES',
    'chronological_split',
    'chronological_split_indices',
    'grouped_chronological_split',
    'grouped_chronological_split_indices',
    'split_indices',
    'train_val_test_split',
    'assert_no_split_leakage',
    'LeakageSafePreprocessor',
    'SafeScaler',
    'prepare_features',
    'fit_preprocessor',
    'data_quality_report',
    'categorical_shift_report',
    'duplicate_mask',
]

SPLIT_NAMES = ('train', 'val', 'test')
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Split sizing / validation
# ---------------------------------------------------------------------------

def _check_fraction(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number in [0, 1], got {value!r}') from None
    if not (0.0 <= value <= 1.0) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number in [0, 1], got {value!r}')
    return value


def _resolve_sizes(n, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                   train_size=None, val_size=None, test_size=None):
    """Return deterministic (n_train, n_val, n_test) sizes summing to ``n``.

    The default is chronological fractions.  When an explicit size is given it
    is honoured; unspecified split sizes are derived from the fractions and the
    last unspecified split absorbs the remainder.  All sizes are non-negative
    and must sum to ``n``.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f'n must be non-negative, got {n}')
    train_frac = _check_fraction('train_frac', train_frac)
    val_frac = _check_fraction('val_frac', val_frac)
    test_frac = _check_fraction('test_frac', test_frac)
    explicit = (train_size, val_size, test_size)
    if any(v is not None for v in explicit):
        sizes = []
        for name, v in zip(SPLIT_NAMES, explicit):
            if v is not None:
                v = int(v)
                if v < 0:
                    raise ValueError(f'{name}_size must be non-negative, got {v}')
                sizes.append(v)
            else:
                sizes.append(None)
        if sum(v for v in sizes if v is not None) > n:
            raise ValueError(f'explicit split sizes {sizes} exceed n={n}')
        # Fill unspecified entries, letting the last one absorb the remainder.
        fracs = (train_frac, val_frac, test_frac)
        n_unspecified = sum(v is None for v in sizes)
        for i in range(3):
            if sizes[i] is None:
                if n_unspecified == 1 or i == max(j for j in range(3) if sizes[j] is None):
                    sizes[i] = n - sum(v for v in sizes if v is not None)
                else:
                    sizes[i] = int(math.floor(n * fracs[i]))
                n_unspecified -= 1
        if sum(sizes) != n:
            raise ValueError(f'could not reconcile split sizes {sizes} with n={n}')
        return tuple(int(v) for v in sizes)

    frac_sum = train_frac + val_frac + test_frac
    if frac_sum <= 0:
        raise ValueError('at least one split fraction must be positive')
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(
            f'train_frac + val_frac + test_frac must sum to 1, got {frac_sum}')
    sizes = [int(math.floor(n * f)) for f in (train_frac, val_frac, test_frac)]
    # Small n: guarantee one row for every requested non-empty split.  Larger
    # splits are reduced first (from the end) so the chronological order is
    # never broken.
    requested = [f > 0 for f in (train_frac, val_frac, test_frac)]
    if n < sum(requested):
        raise ValueError(
            f'n={n} is too small for {sum(requested)} non-empty chronological splits')
    for i in range(3):
        if requested[i] and sizes[i] == 0:
            sizes[i] = 1
            for j in (2, 1, 0):
                if j != i and sizes[j] > 1:
                    sizes[j] -= 1
                    break
    sizes[2] = n - sizes[0] - sizes[1]
    if sizes[2] < 0 or sizes[0] < 0 or sizes[1] < 0:
        raise ValueError(f'inconsistent split sizes {sizes} for n={n}')
    return tuple(int(v) for v in sizes)


def _as_1d(values, name, n):
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f'{name} has length {arr.shape[0]}, expected {n}')
    return arr


def _empty_index():
    return np.empty(0, dtype=np.int64)


def _take(data, idx):
    """Index an array/sequence with an integer index array."""
    try:
        return data[idx]
    except TypeError:
        if isinstance(data, (list, tuple)):
            return type(data)(data[int(i)] for i in idx)
        # Fall back to NumPy for other array-likes (e.g. pandas objects are
        # better indexed explicitly by the caller).
        return np.asarray(data)[idx]


def chronological_split_indices(n, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                                *, train_size=None, val_size=None, test_size=None):
    """Contiguous, order-preserving train/val/test index arrays.

    Rows ``[0, n_train)`` train, the next block validates and the final block
    tests.  No shuffle and no resampling is performed, so the split is valid
    for time-series data (and is the safe default for generic data too).
    """
    n_train, n_val, n_test = _resolve_sizes(
        n, train_frac, val_frac, test_frac,
        train_size, val_size, test_size)
    train_end = n_train
    val_end = n_train + n_val
    return {
        'train': np.arange(0, train_end, dtype=np.int64),
        'val': np.arange(train_end, val_end, dtype=np.int64),
        'test': np.arange(val_end, val_end + n_test, dtype=np.int64),
    }


def chronological_split(data, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                        *, train_size=None, val_size=None, test_size=None):
    """Split an array/sequence chronologically and return the three parts.

    ``data`` may be any object that supports integer indexing (NumPy array,
    list, tuple, ...).  The exact same index blocks are returned for all three
    parts, so features and labels can be split consistently by calling
    :func:`chronological_split_indices` once.
    """
    idx = chronological_split_indices(
        len(data), train_frac, val_frac, test_frac,
        train_size=train_size, val_size=val_size, test_size=test_size)
    return {name: _take(data, idx[name]) for name in SPLIT_NAMES}


# Backwards/ergonomic alias: the safe default split is chronological.
train_val_test_split = chronological_split


def _group_order(groups, order=None):
    """Order group ids by first appearance or by their minimum timestamp."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        groups = groups.reshape(-1)
    first = {}
    for i, g in enumerate(groups.tolist()):
        if g not in first:
            first[g] = i
    if order is None:
        return sorted(first, key=lambda g: first[g]), first
    order = np.asarray(order)
    if order.ndim != 1 or order.shape[0] != groups.shape[0]:
        raise ValueError('order must be a 1-D array with one value per row')
    key = {}
    for i, g in enumerate(groups.tolist()):
        v = order[i]
        if g not in key or v < key[g]:
            key[g] = v
    return sorted(first, key=lambda g: (key[g], first[g])), first


def grouped_chronological_split_indices(n, groups, train_frac=0.8, val_frac=0.1,
                                        test_frac=0.1, *, order=None,
                                        train_size=None, val_size=None,
                                        test_size=None):
    """Chronological split that assigns every group to exactly one side.

    Group order is the order of first appearance (or the minimum ``order``
    value, e.g. a timestamp).  Whole groups are consumed until the requested
    train size is reached, then whole groups fill validation, and the remainder
    goes to test.  A group is therefore never split across train/val/test.
    """
    n = int(n)
    groups = _as_1d(groups, 'groups', n)
    if groups.shape[0] != n:
        raise ValueError('groups must have one entry per row')
    targets = _resolve_sizes(n, train_frac, val_frac, test_frac,
                             train_size, val_size, test_size)
    order_groups, first = _group_order(groups, order)
    members = {}
    for i, g in enumerate(groups.tolist()):
        members.setdefault(g, []).append(i)
    out = {name: [] for name in SPLIT_NAMES}
    current = 0
    for g in order_groups:
        while (current < 2 and
               len(out[SPLIT_NAMES[current]]) >= targets[current]):
            current += 1
        out[SPLIT_NAMES[current]].extend(members[g])
    res = {name: np.asarray(sorted(v), dtype=np.int64) for name, v in out.items()}
    # Groups must be pairwise disjoint even if the caller supplied a target
    # that forced a large group to one side.
    _assert_index_and_group_disjoint(res, groups)
    return res


def grouped_chronological_split(data, groups, train_frac=0.8, val_frac=0.1,
                                test_frac=0.1, *, order=None,
                                train_size=None, val_size=None, test_size=None):
    """Array-returning wrapper around :func:`grouped_chronological_split_indices`."""
    idx = grouped_chronological_split_indices(
        len(data), groups, train_frac, val_frac, test_frac, order=order,
        train_size=train_size, val_size=val_size, test_size=test_size)
    return {name: _take(data, idx[name]) for name in SPLIT_NAMES}


def split_indices(n, *, time=None, groups=None, train_frac=0.8, val_frac=0.1,
                  test_frac=0.1, train_size=None, val_size=None, test_size=None):
    """Unified leakage-safe split entry point.

    * ``groups`` given: whole-group chronological split (ordered by ``time``
      when supplied, else first appearance).
    * ``time`` given without groups: rows are ordered by time before the
      contiguous split, so the returned arrays are the original row indices in
      chronological order.
    * neither: contiguous split in the input order (no shuffle).
    """
    n = int(n)
    if groups is not None:
        return grouped_chronological_split_indices(
            n, groups, train_frac, val_frac, test_frac, order=time,
            train_size=train_size, val_size=val_size, test_size=test_size)
    if time is not None:
        time = _as_1d(time, 'time', n)
        order = np.argsort(time, kind='stable')
        sizes = _resolve_sizes(n, train_frac, val_frac, test_frac,
                               train_size, val_size, test_size)
        blocks = []
        start = 0
        for size in sizes:
            blocks.append(np.sort(order[start:start + size]))
            start += size
        return dict(zip(SPLIT_NAMES, blocks))
    return chronological_split_indices(
        n, train_frac, val_frac, test_frac,
        train_size=train_size, val_size=val_size, test_size=test_size)


def _assert_index_and_group_disjoint(split_idx, groups=None):
    sets = {name: set(np.asarray(split_idx[name]).reshape(-1).tolist())
            for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f'{a} and {b} indices overlap')
    if groups is not None:
        groups = np.asarray(groups)
        gsets = {name: {groups[i] for i in sets[name]} for name in SPLIT_NAMES}
        for i, a in enumerate(SPLIT_NAMES):
            for b in SPLIT_NAMES[i + 1:]:
                if gsets[a] & gsets[b]:
                    raise ValueError(
                        f'{a} and {b} share group ids: '
                        f'{sorted(map(repr, gsets[a] & gsets[b]))[:5]}')


def assert_no_split_leakage(train_idx, val_idx, test_idx, *, groups=None,
                            time=None):
    """Raise ``ValueError`` if splits overlap in rows, groups or time.

    Time is checked as a monotone partition: every training time must be no
    later than every validation time, and every validation time no later than
    every test time.  This catches random shuffling of time-series rows even
    when the row indices themselves are disjoint.
    """
    split_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    _assert_index_and_group_disjoint(split_idx, groups)
    if time is not None:
        time = np.asarray(time)
        if time.ndim != 1:
            time = time.reshape(-1)
        for a, b in (('train', 'val'), ('val', 'test'), ('train', 'test')):
            ia = np.asarray(split_idx[a], dtype=np.int64).reshape(-1)
            ib = np.asarray(split_idx[b], dtype=np.int64).reshape(-1)
            if ia.size == 0 or ib.size == 0:
                continue
            if np.max(time[ia]) > np.min(time[ib]):
                raise ValueError(
                    f'temporal leakage: max {a} time '
                    f'{np.max(time[ia])!r} > min {b} time {np.min(time[ib])!r}')
    return True


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing
# ---------------------------------------------------------------------------

def _as_float_2d(X, *, name='X'):
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 1-D or 2-D, got shape {arr.shape}')
    try:
        arr = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        raise TypeError(
            f'{name} must be numeric; encode categorical features before '
            'calling the leakage-safe preprocessor') from None
    return arr


def _finite_values(col):
    return col[np.isfinite(col)]


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
        Optional additional clipping in standardized units *after* scaling,
        using only training statistics (e.g. 8.0).
    """

    def __init__(self, *, scale='standard', missing='median',
                 clip_quantiles=(0.001, 0.999), clip_z=None, eps=_EPS):
        if scale not in ('standard', 'robust', 'minmax', 'none'):
            raise ValueError(f'unknown scale={scale!r}')
        if not (isinstance(missing, (int, float)) or
                missing in ('median', 'mean', 'zero')):
            raise ValueError(f'unknown missing={missing!r}')
        if clip_quantiles is not None:
            lo, hi = clip_quantiles
            lo, hi = float(lo), float(hi)
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError('clip_quantiles must satisfy 0 <= lo < hi <= 1')
            clip_quantiles = (lo, hi)
        if clip_z is not None and not (float(clip_z) > 0.0):
            raise ValueError('clip_z must be positive or None')
        self.scale = scale
        self.missing = missing
        self.clip_quantiles = clip_quantiles
        self.clip_z = None if clip_z is None else float(clip_z)
        self.eps = float(eps)
        self.fitted_ = False

    # -- fitting ----------------------------------------------------------
    def _fill_value(self, values):
        if isinstance(self.missing, (int, float)):
            return float(self.missing)
        if values.size == 0:
            return 0.0
        if self.missing == 'mean':
            return float(np.mean(values))
        if self.missing == 'median':
            return float(np.median(values))
        return 0.0

    def fit(self, X, y=None):
        """Learn imputation/clipping/scaling statistics from ``X`` (train)."""
        X = _as_float_2d(X)
        n, p = X.shape
        if n == 0:
            raise ValueError('cannot fit the preprocessor on zero training rows')
        Xn = np.where(np.isfinite(X), X, np.nan)
        stats = []
        for j in range(p):
            col = Xn[:, j]
            finite = col[np.isfinite(col)]
            fill = self._fill_value(finite)
            if not np.isfinite(fill):
                fill = 0.0
            if self.clip_quantiles is not None:
                if finite.size:
                    q_lo, q_hi = np.quantile(finite, self.clip_quantiles)
                else:
                    q_lo, q_hi = -np.inf, np.inf
                if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
                    q_lo, q_hi = -np.inf, np.inf
            else:
                q_lo, q_hi = -np.inf, np.inf
            clipped = np.clip(col, q_lo, q_hi) if np.isfinite(q_lo) else col
            imputed = np.where(np.isfinite(clipped), clipped, min(max(fill, q_lo), q_hi)
                               if np.isfinite(q_lo) else fill)
            # Scaling statistics are computed on the training values after the
            # same impute/clip operation that transform() will apply.  They are
            # never recomputed from validation/test data.
            center, factor = 0.0, 1.0
            if self.scale == 'standard':
                center = float(np.mean(imputed))
                sd = float(np.std(imputed, ddof=0))
                factor = sd if sd > self.eps else 1.0
            elif self.scale == 'robust':
                center = float(np.median(imputed))
                q1, q3 = np.quantile(imputed, (0.25, 0.75))
                iqr = float(q3 - q1)
                if iqr > self.eps:
                    factor = iqr
                else:
                    sd = float(np.std(imputed, ddof=0))
                    factor = sd if sd > self.eps else 1.0
            elif self.scale == 'minmax':
                center = float(np.min(imputed))
                span = float(np.max(imputed) - center)
                factor = span if span > self.eps else 1.0
            if not np.isfinite(center):
                center = 0.0
            if not np.isfinite(factor) or factor <= self.eps:
                factor = 1.0
            stats.append({
                'fill': float(fill),
                'clip_lo': float(q_lo) if np.isfinite(q_lo) else None,
                'clip_hi': float(q_hi) if np.isfinite(q_hi) else None,
                'center': center,
                'factor': factor,
                'mean': float(np.mean(imputed)),
                'std': float(np.std(imputed, ddof=0)),
                'median': float(np.median(imputed)),
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
        X = _as_float_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {X.shape[1]} features, but the preprocessor was fitted '
                f'on {self.n_features_in_}')
        out = np.empty_like(X, dtype=np.float64)
        for j, st in enumerate(self.feature_stats_):
            col = X[:, j]
            col = np.where(np.isfinite(col), col, np.nan)
            lo, hi = st['clip_lo'], st['clip_hi']
            if lo is not None and hi is not None:
                col = np.clip(col, lo, hi)
            col = np.where(np.isnan(col), st['fill'], col)
            if self.scale != 'none':
                col = (col - st['center']) / st['factor']
            if self.clip_z is not None and self.scale in ('standard', 'robust'):
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
        obj = cls(scale=stats['scale'], missing=stats['missing'],
                  clip_quantiles=(tuple(stats['clip_quantiles'])
                                  if stats.get('clip_quantiles') else None),
                  clip_z=stats.get('clip_z'), eps=stats.get('eps', _EPS))
        obj.feature_stats_ = [dict(s) for s in stats['features']]
        obj.n_features_in_ = int(stats['n_features_in'])
        obj.n_train_ = int(stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


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
    out = {'train': prep.transform(X_train), 'preprocessor': prep}
    if X_val is not None:
        out['val'] = prep.transform(X_val)
    if X_test is not None:
        out['test'] = prep.transform(X_test)
    return out


# ---------------------------------------------------------------------------
# Data-quality diagnostics
# ---------------------------------------------------------------------------

def _matrix_fingerprint(X):
    arr = np.asarray(X, dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('ascii'))
    # Hash in bounded row blocks so fingerprinting a large training matrix
    # does not allocate a second full-size copy.
    for start in range(0, arr.shape[0], 1 << 20):
        h.update(np.ascontiguousarray(arr[start:start + (1 << 20)]).tobytes())
    return h.hexdigest()[:16]


def _row_keys(X):
    """Stable per-row keys for exact-duplicate detection (NaN-safe)."""
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    finite = np.isfinite(arr)
    keys = []
    for row, ok in zip(arr, finite):
        # NaN/Inf positions are represented explicitly so two rows are equal
        # only when their finite values AND their non-finite patterns match.
        h = hashlib.blake2b(digest_size=8)
        h.update(np.ascontiguousarray(row[ok]).tobytes())
        h.update(b'|')
        h.update(np.packbits(~ok).tobytes())
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


def _label_key(v):
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
    except (TypeError, ValueError):
        pass
    return v


def data_quality_report(X, y=None, *, train_idx=None, val_idx=None, test_idx=None,
                        groups=None, clip_z=3.0, shift_threshold=0.5):
    """Report the common data-cleaning failure modes before any fit.

    ``train_idx`` identifies the training rows.  All outlier and shift
    statistics are computed from those rows only; when no index is supplied the
    whole input is treated as the training reference (useful as a pre-split
    smoke check, not as a license to fit preprocessing on all data).

    Returns a JSON-friendly dict.  ``status`` is ``'fail'`` when exact duplicate
    rows cross split boundaries (row-level leakage), ``'warnings'`` when
    NaN/Inf, within-split duplicates, label conflicts, outliers or large shift
    are present, and ``'ok'`` otherwise.
    """
    X = _as_float_2d(X)
    n = X.shape[0]
    idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for name, v in idx.items():
        if v is not None:
            v = np.asarray(v, dtype=np.int64).reshape(-1)
            if v.size and (v.min() < 0 or v.max() >= n):
                raise ValueError(f'{name}_idx is out of bounds for n={n}')
            idx[name] = v
    if all(v is None for v in idx.values()):
        # No split supplied: use the whole input as a single training
        # reference and leave the holdouts empty.  This is a pre-split smoke
        # check, not permission to fit preprocessing on all data.
        split_rows = {'train': np.arange(n, dtype=np.int64),
                      'val': _empty_index(), 'test': _empty_index()}
    else:
        # Missing holdout indices mean "no rows on that side", never "all
        # rows": the latter would manufacture cross-split duplicates.
        split_rows = {name: (_empty_index() if idx[name] is None else idx[name])
                      for name in SPLIT_NAMES}
        if idx['train'] is None:
            split_rows['train'] = np.arange(n, dtype=np.int64)
    ref = split_rows['train']
    warnings = []
    checks = {}

    finite = np.isfinite(X)
    checks['nan_count'] = int(np.isnan(X).sum())
    checks['inf_count'] = int(np.isinf(X).sum())
    checks['nonfinite_by_split'] = {
        name: int((~finite[rows]).sum()) for name, rows in split_rows.items()
    }
    if checks['nan_count'] or checks['inf_count']:
        warnings.append(
            f"{checks['nan_count']} NaN and {checks['inf_count']} Inf values; "
            'impute with training statistics before fitting')

    # Exact duplicates within each split.
    dup = {}
    keys = _row_keys(X)
    for name, rows in split_rows.items():
        sub = [keys[i] for i in rows]
        dup[name] = int(len(sub) - len(set(sub)))
    checks['duplicate_rows_within_split'] = dup
    if any(dup.values()):
        warnings.append(f'duplicate rows within splits: {dup}')

    # Exact duplicate rows across splits: a hard leakage signal.
    cross = {}
    hard_leak = False
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        sa = {keys[i] for i in split_rows[a]}
        sb = {keys[i] for i in split_rows[b]}
        shared = len(sa & sb)
        cross[f'{a}__{b}'] = shared
        hard_leak = hard_leak or shared > 0
    checks['duplicate_rows_across_splits'] = cross
    if hard_leak:
        warnings.append(f'exact duplicate rows cross split boundaries: {cross}')

    # Group integrity.
    if groups is not None:
        groups = _as_1d(groups, 'groups', n)
        gsets = {name: {groups[i] for i in split_rows[name]}
                 for name in SPLIT_NAMES}
        group_cross = {}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            shared = gsets[a] & gsets[b]
            group_cross[f'{a}__{b}'] = len(shared)
            if shared:
                hard_leak = True
                warnings.append(
                    f'{len(shared)} group ids cross {a}/{b}; whole groups must '
                    'stay in one split')
        checks['group_overlap'] = group_cross

    # Label noise on exact duplicate feature rows.
    if y is not None:
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n:
            raise ValueError(f'y has length {y.shape[0]}, expected {n}')
        labels = [_label_key(v) for v in y.tolist()]
        missing_labels = int(sum(v is None for v in labels))
        checks['missing_labels'] = missing_labels
        if missing_labels:
            warnings.append(f'{missing_labels} missing labels')
        by_row = {}
        for i, (k, lab) in enumerate(zip(keys, labels)):
            if lab is not None:
                by_row.setdefault(k, set()).add(lab)
        conflicts = [k for k, labs in by_row.items() if len(labs) > 1]
        checks['label_conflicts_on_duplicate_rows'] = len(conflicts)
        if conflicts:
            warnings.append(
                f'{len(conflicts)} exact-duplicate feature rows carry different '
                'labels (label noise / ambiguous annotation)')
        vals, counts = np.unique(
            [v for v in labels if v is not None], return_counts=True)
        checks['label_distribution'] = {
            str(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())
        }

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
        q1, q3 = np.quantile(vals, (0.25, 0.75))
        iqr = q3 - q1
        if iqr <= _EPS:
            q_lo[j], q_hi[j] = np.min(vals), np.max(vals)
        else:
            q_lo[j], q_hi[j] = q1 - clip_z * iqr, q3 + clip_z * iqr
        for name, rows in split_rows.items():
            col = X[rows, j]
            good = np.isfinite(col)
            out_counts[name] += int(np.sum(
                good & ((col < q_lo[j]) | (col > q_hi[j]))))
    checks['outliers_by_split'] = out_counts
    if out_counts['train'] or out_counts['val'] or out_counts['test']:
        warnings.append(f'outliers (training IQR rule): {out_counts}')

    # Distribution shift: standardized mean difference and missing-rate delta
    # relative to training.  This is a descriptive audit, not a test.
    shift = {}
    for name in ('val', 'test'):
        rows = split_rows[name]
        if rows.size == 0:
            continue
        Xh = X[rows]
        smd = []
        for j in range(X.shape[1]):
            tv = Xt[:, j]
            hv = Xh[:, j]
            tv = tv[np.isfinite(tv)]
            hv = hv[np.isfinite(hv)]
            if tv.size == 0 or hv.size == 0:
                smd.append(None)
                continue
            sd = float(np.std(tv, ddof=0))
            if sd <= _EPS:
                smd.append(0.0 if abs(float(np.mean(hv)) - float(np.mean(tv))) <= _EPS else None)
            else:
                smd.append(float((np.mean(hv) - np.mean(tv)) / sd))
        miss_train = 1.0 - np.mean(np.isfinite(Xt), axis=0)
        miss_hold = 1.0 - np.mean(np.isfinite(Xh), axis=0)
        finite_smd = [abs(v) for v in smd if v is not None]
        shift[name] = {
            'max_abs_standardized_mean_difference': (
                max(finite_smd) if finite_smd else 0.0),
            'per_feature_standardized_mean_difference': smd,
            'max_abs_missing_rate_delta': float(np.max(np.abs(miss_hold - miss_train))),
        }
        if shift[name]['max_abs_standardized_mean_difference'] > shift_threshold:
            warnings.append(
                f'{name} covariate shift: max |SMD| = '
                f"{shift[name]['max_abs_standardized_mean_difference']:.3f} "
                f'(threshold {shift_threshold})')
        if shift[name]['max_abs_missing_rate_delta'] > shift_threshold:
            warnings.append(
                f'{name} missingness shift: max |delta| = '
                f"{shift[name]['max_abs_missing_rate_delta']:.3f}")
    checks['distribution_shift_vs_train'] = shift

    status = 'fail' if hard_leak else ('warnings' if warnings else 'ok')
    return {
        'status': status,
        'n_samples': int(n),
        'n_features': int(X.shape[1]),
        'n_train_reference': int(ref.size),
        'checks': checks,
        'warnings': warnings,
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
    """
    ref = np.asarray(reference).reshape(-1)
    hold = np.asarray(holdout).reshape(-1)
    if ref.size == 0 or hold.size == 0:
        return {'js_divergence': 0.0, 'total_variation': 0.0,
                'n_reference': int(ref.size), 'n_holdout': int(hold.size),
                'unseen_rate': 0.0}
    if max_categories is not None:
        cat = np.unique(ref)
        if cat.size > max_categories:
            # Keep the most frequent reference categories plus an "other" bin.
            vals, counts = np.unique(ref, return_counts=True)
            keep = np.sort(vals[np.argsort(-counts)[:int(max_categories) - 1]])
        else:
            keep = cat
    else:
        keep = np.unique(ref)

    def _hist(a):
        h = np.zeros(keep.size + 1, dtype=np.float64)
        if keep.size:
            pos = np.searchsorted(keep, a)
            inb = (pos < keep.size)
            inb[inb] = keep[pos[inb]] == a[inb]
            h[:keep.size] = np.bincount(pos[inb], minlength=keep.size)
            h[-1] = np.sum(~inb)
        else:
            h[-1] = a.size
        return h / max(a.size, 1)

    p = _hist(ref)
    q = _hist(hold)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], _EPS))))

    tv = 0.5 * float(np.sum(np.abs(p - q)))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    unseen = 0.0
    if keep.size:
        unseen = float(np.mean(~np.isin(hold, keep)))
    return {
        'js_divergence': js,
        'total_variation': tv,
        'n_reference': int(ref.size),
        'n_holdout': int(hold.size),
        'unseen_rate': unseen,
    }
