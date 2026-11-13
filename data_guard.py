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
* **Time windows can be purged.**  An optional ``purge``/embargo gap keeps
  overlapping lookback windows from straddling a split boundary.
* **Split first, then prepare.**  ``split_and_prepare`` performs the split
  before fitting imputation/clipping/scaling, and the fitted preprocessor is
  applied to the holdouts with ``transform`` only.

The module also provides data-quality diagnostics for the failure modes that
commonly invalidate a split: NaN/Inf, exact duplicates (within and across
splits), label noise on duplicate rows, outliers, and train/holdout
distribution shift.
"""
from __future__ import annotations

import hashlib
import math
import numbers
from fractions import Fraction

import numpy as np

__all__ = [
    'SPLIT_NAMES',
    'LeakageSafeCategoricalEncoder',
    'LeakageSafePreprocessor',
    'SafeScaler',
    'assert_no_split_leakage',
    'categorical_shift_report',
    'chronological_split',
    'chronological_split_indices',
    'clean_training_rows',
    'data_quality_report',
    'duplicate_mask',
    'fit_categorical_encoder',
    'fit_preprocessor',
    'prepare_categorical_features',
    'grouped_chronological_split',
    'grouped_chronological_split_indices',
    'prepare_features',
    'split_and_prepare',
    'split_indices',
    'train_val_test_split',
]

SPLIT_NAMES = ('train', 'val', 'test')
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Split sizing / validation
# ---------------------------------------------------------------------------

def _check_fraction(name, value):
    # Booleans are not fractions: float(True) == 1.0 would silently turn a
    # mistaken flag into a 100% split.  Temporal values are not fractions
    # either.  Same contract as _check_count.
    if isinstance(value, (bool, np.bool_, np.datetime64, np.timedelta64)):
        raise ValueError(f'{name} must be a number in [0, 1], got {value!r}')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a number in [0, 1], got {value!r}') from None
    if not (0.0 <= value <= 1.0) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number in [0, 1], got {value!r}')
    return value


def _check_count(name, value, *, positive=False):
    """Validate an integer count without silently coercing/truncating it."""
    if isinstance(value, (bool, np.bool_, np.datetime64, np.timedelta64)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if isinstance(value, numbers.Integral):
        value = int(value)
    elif isinstance(value, numbers.Real):
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be an integer, got {value!r}') from None
        if (not math.isfinite(f) or f != math.floor(f)
                or abs(f) >= 2 ** 53):
            raise ValueError(f'{name} must be an integer, got {value!r}')
        value = int(f)
    else:
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if value < 0:
        raise ValueError(f'{name} must be non-negative, got {value}')
    if positive and value == 0:
        raise ValueError(f'{name} must be positive, got {value}')
    return value


def _explicit_count(name, value):
    """Validate an explicit split size without silent int() coercion."""
    return _check_count(f'{name}_size', value)


def _resolve_sizes(n, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                   train_size=None, val_size=None, test_size=None):
    """Return deterministic (n_train, n_val, n_test) sizes summing to ``n``.

    The default is chronological fractions.  When an explicit size is given it
    is honoured; unspecified split sizes are derived from the fractions and the
    last unspecified split absorbs the remainder.  All sizes are non-negative
    and must sum to ``n``.
    """
    n = _check_count('n', n)
    train_frac = _check_fraction('train_frac', train_frac)
    val_frac = _check_fraction('val_frac', val_frac)
    test_frac = _check_fraction('test_frac', test_frac)
    explicit = (train_size, val_size, test_size)
    if any(v is not None for v in explicit):
        sizes = []
        for name, v in zip(SPLIT_NAMES, explicit):
            if v is not None:
                sizes.append(_explicit_count(name, v))
            else:
                sizes.append(None)
        if sum(v for v in sizes if v is not None) > n:
            raise ValueError(f'explicit split sizes {sizes} exceed n={n}')
        # Fill unspecified entries, letting the last one absorb the remainder.
        # Earlier unspecified splits take their fractional target only while
        # rows remain; clamping to the remaining count keeps every size
        # non-negative even when an explicit size or a large fraction would
        # otherwise overdraw it (the last split then takes whatever is left).
        # Evaluate n*frac exactly: a float product can overflow or lose the
        # low bits of an exact integer count.
        fracs = (Fraction(train_frac), Fraction(val_frac), Fraction(test_frac))
        unspecified = [i for i in range(3) if sizes[i] is None]
        remaining = n - sum(v for v in sizes if v is not None)
        for i in unspecified[:-1]:
            take = min(int(Fraction(n) * fracs[i]), remaining)
            sizes[i] = take
            remaining -= take
        if unspecified:
            sizes[unspecified[-1]] = remaining
        if sum(sizes) != n or any(v < 0 for v in sizes):
            raise ValueError(f'could not reconcile split sizes {sizes} with n={n}')
        return tuple(int(v) for v in sizes)

    if n == 0:
        # An empty dataset splits into three empty (non-overlapping) parts;
        # requiring one row per non-empty split would be impossible.
        return (0, 0, 0)
    frac_sum = train_frac + val_frac + test_frac
    if frac_sum <= 0:
        raise ValueError('at least one split fraction must be positive')
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(
            f'train_frac + val_frac + test_frac must sum to 1, got {frac_sum}')
    # The tolerance accepts sums within 1e-6 of 1.  At scale, n*frac can then
    # overshoot n by thousands of rows and the floor sum could exceed n
    # (negative remainder).  Renormalising the accepted fractions keeps the
    # targets summing to n exactly for every n; the deviation is already
    # inside the declared tolerance.
    _frac_total = (Fraction(train_frac) + Fraction(val_frac)
                   + Fraction(test_frac))
    fracs = tuple(Fraction(f) / _frac_total
                  for f in (train_frac, val_frac, test_frac))
    requested = [f > 0 for f in fracs]
    if n < sum(requested):
        raise ValueError(
            f'n={n} is too small for {sum(requested)} non-empty chronological splits')
    # Largest-remainder apportionment: floor every target, then give the
    # leftover rows to the splits with the largest fractional parts.  This
    # keeps small-n splits close to their requested proportions and never
    # puts rows into a split whose requested fraction is zero.  Exact
    # rational arithmetic keeps the targets usable for any integer n.
    targets = [Fraction(n) * f for f in fracs]
    sizes = [int(t) for t in targets]
    remaining = n - sum(sizes)
    order = sorted(range(3), key=lambda i: (-(targets[i] - sizes[i]), -i))
    eligible = [i for i in order if requested[i]]
    if remaining > 0:
        # Flooring the rounded float targets can leave rows unassigned;
        # distribute them across the requested splits in one step.
        _base, _extra = divmod(remaining, len(eligible))
        for _pos, _i in enumerate(eligible):
            sizes[_i] += _base + (1 if _pos < _extra else 0)
    elif remaining < 0:
        # The float targets can also round UP enough that the floors sum to
        # more than n.  Remove the overshoot from the splits with the smallest
        # fractional parts first (largest split breaks ties) so every size
        # stays non-negative.
        _need = -remaining
        _drop_order = sorted(range(3), key=lambda i: (targets[i] - sizes[i],
                                                      -sizes[i], i))
        for _i in _drop_order:
            if _need <= 0:
                break
            _take = min(_need, sizes[_i])
            sizes[_i] -= _take
            _need -= _take
        if _need:
            raise ValueError(f'inconsistent split sizes {sizes} for n={n}')
    # A requested split that still rounds to zero (very small n) must borrow
    # one row from the largest split that can spare it.
    for i in range(3):
        if requested[i] and sizes[i] == 0:
            donors = [j for j in range(3) if j != i and sizes[j] > 1]
            if not donors:
                raise ValueError(f'inconsistent split sizes {sizes} for n={n}')
            j = max(donors, key=lambda k: (sizes[k], -k))
            sizes[j] -= 1
            sizes[i] = 1
    if sum(sizes) != n or any(v < 0 for v in sizes):
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


def _validated_index(values, *, name='index'):
    """Return a non-negative ``int64`` row-index array.

    Split arrays are compared by value, but NumPy also resolves negative
    indices from the end of an array.  ``-1`` and ``n - 1`` therefore address
    the same row while looking disjoint in a set comparison, which would let a
    holdout row appear in the training split.  Fractional values are rejected
    instead of being silently truncated (``0.5 -> 0``).  Bool arrays are
    rejected too: they are masks, not row indices.
    """
    arr = np.asarray(values)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return _empty_index()
    kind = flat.dtype.kind
    if kind in 'iu':
        out = flat.astype(np.int64, copy=False)
    elif kind == 'f':
        f = flat.astype(np.float64, copy=False)
        if not np.all(np.isfinite(f)):
            raise ValueError(f'{name} contains NaN/Inf; split indices must be finite')
        if np.any(f != np.floor(f)):
            raise ValueError(f'{name} contains non-integer row indices')
        # Same exactness boundary as the object branch: above 2**53 a float
        # cannot represent every integer, so it may silently alias another
        # row index.
        if np.any(np.abs(f) >= 2 ** 53):
            raise ValueError(
                f'{name} contains row indices outside the exact float-integer '
                'range (< 2**53)')
        out = f.astype(np.int64)
    elif kind == 'O':
        # Reject strings, datetimes and booleans instead of coercing them:
        # a timestamp column passed by mistake must not become row indices.
        vals = []
        for v in flat.tolist():
            if isinstance(v, (bool, np.bool_)):
                raise ValueError(
                    f'{name} must contain integer row indices, not booleans')
            if isinstance(v, (np.datetime64, np.timedelta64)):
                raise ValueError(
                    f'{name} must contain integer row indices, not temporal '
                    f'values ({type(v).__name__})')
            if isinstance(v, numbers.Integral):
                v = int(v)
            elif isinstance(v, numbers.Real):
                try:
                    f = float(v)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f'{name} must contain exact finite integer row indices, '
                        f'got {v!r}') from None
                if (not math.isfinite(f) or f != math.floor(f)
                        or abs(f) >= 2 ** 53):
                    raise ValueError(
                        f'{name} must contain exact finite integer row indices, '
                        f'got {v!r}')
                v = int(f)
            else:
                raise ValueError(
                    f'{name} must contain integer row indices, got '
                    f'{type(v).__name__}')
            if v < 0:
                raise ValueError(
                    f'{name} contains negative row indices; use non-negative '
                    'indices so a wraparound index cannot alias a row from '
                    'another split')
            if v > np.iinfo(np.int64).max:
                raise ValueError(f'{name} contains an out-of-range row index {v}')
            vals.append(v)
        out = np.asarray(vals, dtype=np.int64)
    else:
        raise ValueError(
            f'{name} must contain integer row indices, got dtype {flat.dtype}')
    if np.any(out < 0):
        raise ValueError(
            f'{name} contains negative row indices; use non-negative indices so '
            'a wraparound index cannot alias a row from another split')
    return out


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


def _check_purge(purge):
    """Validate a purge/embargo size (rows dropped before a holdout split).

    Only an actual integer (or an integral float such as ``2.0``) is accepted.
    Strings and non-integral numbers are refused instead of being truncated or
    coerced, which would silently change the embargo size while looking like a
    valid request.
    """
    if isinstance(purge, (bool, np.bool_, np.datetime64, np.timedelta64)):
        raise ValueError(f'purge must be a non-negative integer, got {purge!r}')
    if isinstance(purge, numbers.Integral):
        purge = int(purge)
    elif isinstance(purge, numbers.Real):
        try:
            f = float(purge)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'purge must be a non-negative integer, got {purge!r}') from None
        if (not math.isfinite(f) or f != math.floor(f)
                or abs(f) >= 2 ** 53):
            raise ValueError(f'purge must be a non-negative integer, got {purge!r}')
        purge = int(f)
    else:
        raise ValueError(f'purge must be a non-negative integer, got {purge!r}')
    if purge < 0:
        raise ValueError(f'purge must be a non-negative integer, got {purge}')
    return purge


def _purge_tail(idx, purge):
    """Drop ``purge`` trailing rows from train/val, leaving a real embargo gap.

    The dropped rows belong to neither side.  Removing them from the *end* of
    the earlier split keeps the held-out test slice intact, while the
    remaining timestamps are still strictly ordered (train < val < test), so a
    lookback window cannot overlap a row that is used for fitting.
    """
    if not purge:
        return idx
    out = {name: np.asarray(idx[name], dtype=np.int64) for name in SPLIT_NAMES}
    for name in ('train', 'val'):
        if out[name].size == 0:
            continue
        if out[name].size <= purge:
            raise ValueError(
                f'purge={purge} would leave the {name} split empty '
                f'(only {out[name].size} rows); reduce purge or the split size')
        out[name] = out[name][:-purge].copy()
    return out


def chronological_split_indices(n, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                                *, train_size=None, val_size=None, test_size=None,
                                purge=0):
    """Contiguous, order-preserving train/val/test index arrays.

    Rows ``[0, n_train)`` train, the next block validates and the final block
    tests.  No shuffle and no resampling is performed, so the split is valid
    for time-series data (and is the safe default for generic data too).

    ``purge`` drops that many trailing rows from train and validation, leaving
    an embargo gap between the splits.  The dropped rows are returned in no
    split; the test rows are untouched.  This prevents a feature window that
    reaches across a boundary from seeing a row from the other split.
    """
    purge = _check_purge(purge)
    n_train, n_val, n_test = _resolve_sizes(
        n, train_frac, val_frac, test_frac,
        train_size, val_size, test_size)
    train_end = n_train
    val_end = n_train + n_val
    idx = {
        'train': np.arange(0, train_end, dtype=np.int64),
        'val': np.arange(train_end, val_end, dtype=np.int64),
        'test': np.arange(val_end, val_end + n_test, dtype=np.int64),
    }
    return _purge_tail(idx, purge)


def chronological_split(data, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                        *, train_size=None, val_size=None, test_size=None,
                        purge=0):
    """Split an array/sequence chronologically and return the three parts.

    ``data`` may be any object that supports integer indexing (NumPy array,
    list, tuple, ...).  The exact same index blocks are returned for all three
    parts, so features and labels can be split consistently by calling
    :func:`chronological_split_indices` once.
    """
    idx = chronological_split_indices(
        len(data), train_frac, val_frac, test_frac,
        train_size=train_size, val_size=val_size, test_size=test_size,
        purge=purge)
    return {name: _take(data, idx[name]) for name in SPLIT_NAMES}


# Backwards/ergonomic alias: the safe default split is chronological.
train_val_test_split = chronological_split


def _assert_order_finite(order, *, name='order'):
    """Reject missing/non-finite ordering keys before they define a split.

    A NaN timestamp sorts to an arbitrary position and can silently place a
    group on the wrong side of a chronological boundary.  Object/string keys
    have no numeric finiteness contract and are left to the caller.
    """
    arr = np.asarray(order)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            if np.any(np.isnat(arr)):
                raise ValueError(f'{name} contains NaT; a temporal split needs '
                                 'a well-defined order')
        elif np.issubdtype(arr.dtype, np.number):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'{name} contains NaN/Inf; a temporal split '
                                 'needs a well-defined order')
        elif arr.dtype == object:
            # Object arrays have no vectorised isfinite/nat test.  Inspect the
            # scalar values so a Python float('nan') or np.datetime64('NaT')
            # cannot sort to an arbitrary position and silently define the
            # split.  Non-numeric/string keys are left to the caller's order.
            for value in arr.tolist():
                if isinstance(value, (np.datetime64, np.timedelta64)):
                    if np.isnat(value):
                        raise ValueError(
                            f'{name} contains NaT; a temporal split needs a '
                            'well-defined order')
                    continue
                # Only real numeric scalars can be missing.  Calling float()
                # on every object turned the legitimate string key "nan" (or
                # "inf") into a spurious NaN timestamp.
                if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
                    continue
                try:
                    f = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(f):
                    raise ValueError(
                        f'{name} contains NaN/Inf; a temporal split needs a '
                        'well-defined order')
    except TypeError:
        pass


def _order_extreme(values, *, maximum):
    """max/min of an ordering key across dtypes.

    NumPy has no ``maximum`` loop for some string dtypes (NumPy >= 2), even
    though Python can compare those scalars.  Falling back to the object
    comparison keeps string event ids / timestamps splittable instead of
    silently marking every such split "not orderable".
    """
    arr = np.asarray(values)
    try:
        return np.max(arr) if maximum else np.min(arr)
    except (TypeError, ValueError, OverflowError):
        flat = arr.reshape(-1).tolist()
        if not flat:
            raise ValueError('cannot take an extreme of an empty ordering key')
        return max(flat) if maximum else min(flat)


def _group_index_fast(groups, order=None):
    """Vectorised (order_groups, first, members) for non-object group arrays.

    Returns ``None`` for object dtypes, and for non-numeric/non-datetime
    ordering keys (e.g. string event ids), where the per-row dict form is the
    only safe implementation (values may be unorderable or their elementwise
    reductions may not have a NumPy loop).  Every returned value matches the
    per-row dict loops exactly:

    * ``first[g]`` is the row index of ``g``'s first appearance —
      ``np.unique(..., return_index=True)`` reports each unique value's
      first occurrence, and (numpy >= 2) collapses all NaNs into one value
      whose first occurrence is the earliest NaN row, exactly like
      ``_group_key`` collapsing NaN to the ``('__nan__',)`` singleton;
    * ``members[g]`` lists the group's row indices in increasing order
      (a stable argsort of the inverse ids keeps equal ids in ascending
      original positions);
    * ``order_groups`` is first-appearance order, or the
      ``(min order value, first appearance)`` order when ``order`` is given
      (``np.lexsort`` with the first indices as the secondary key).

    The dict keys are the same Python scalars ``groups.tolist()`` yields in
    the loop form (including the NaN sentinel tuple), so the two forms are
    interchangeable for any consumer.
    """
    if groups.dtype == object:
        return None
    if order is not None:
        order = np.asarray(order)
        if order.dtype == object:
            return None
        # String/bytes event ids are orderable in Python but have no
        # maximum/minimum ufunc loop in NumPy 2; use the per-row fallback.
        if order.dtype.kind not in 'iufmM':
            return None
        if order.ndim != 1 or order.shape[0] != groups.shape[0]:
            raise ValueError('order must be a 1-D array with one value per row')
        _assert_order_finite(order)
    uniq, first_idx, inv = np.unique(groups, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    first_idx = np.asarray(first_idx).reshape(-1)
    keys = [_group_key(v) for v in uniq.tolist()]
    n_uniq = uniq.shape[0]
    first = {keys[j]: int(first_idx[j]) for j in range(n_uniq)}
    perm = np.argsort(inv, kind='stable')
    inv_sorted = inv[perm]
    bounds = np.searchsorted(inv_sorted, np.arange(n_uniq, dtype=inv.dtype), side='left')
    ends = np.searchsorted(inv_sorted, np.arange(n_uniq, dtype=inv.dtype), side='right')
    members = {keys[j]: perm[bounds[j]:ends[j]] for j in range(n_uniq)}
    if order is None:
        ranked = np.argsort(first_idx, kind='stable')
    else:
        mins = np.minimum.reduceat(np.asarray(order)[perm], bounds.reshape(-1))
        ranked = np.lexsort((first_idx, mins))
    order_groups = [keys[j] for j in ranked.tolist()]
    return order_groups, first, members


def _group_order(groups, order=None):
    """Order group ids by first appearance or by their minimum timestamp."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        groups = groups.reshape(-1)
    fast = _group_index_fast(groups, order)
    if fast is not None:
        return fast[0], fast[1]
    first = {}
    for i, g in enumerate(groups.tolist()):
        g = _group_key(g)
        if g not in first:
            first[g] = i
    if order is None:
        return sorted(first, key=lambda g: first[g]), first
    order = np.asarray(order)
    if order.ndim != 1 or order.shape[0] != groups.shape[0]:
        raise ValueError('order must be a 1-D array with one value per row')
    _assert_order_finite(order)
    key = {}
    try:
        for i, g in enumerate(groups.tolist()):
            g = _group_key(g)
            v = order[i]
            if g not in key or v < key[g]:
                key[g] = v
        return sorted(first, key=lambda g: (key[g], first[g])), first
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            'order values are not mutually orderable; cannot order groups '
            'chronologically') from None


def grouped_chronological_split_indices(n, groups, train_frac=0.8, val_frac=0.1,
                                        test_frac=0.1, *, order=None,
                                        train_size=None, val_size=None,
                                        test_size=None, purge=0):
    """Chronological split that assigns every group to exactly one side.

    Group order is the order of first appearance (or the minimum ``order``
    value, e.g. a timestamp).  Whole groups are consumed until the requested
    train size is reached, then whole groups fill validation, and the remainder
    goes to test.  A group is therefore never split across train/val/test.
    ``purge`` is rejected here: dropping individual rows would break whole-
    group assignment.  Remove/merge whole groups explicitly when an embargo is
    required.
    """
    purge = _check_purge(purge)
    if purge:
        raise ValueError(
            'purge is not supported with grouped splits: it would drop rows '
            'from a group and break whole-group assignment; remove whole '
            'groups instead')
    n = _check_count('n', n)
    groups = _as_1d(groups, 'groups', n)
    if groups.shape[0] != n:
        raise ValueError('groups must have one entry per row')
    targets = _resolve_sizes(n, train_frac, val_frac, test_frac,
                             train_size, val_size, test_size)
    fast = _group_index_fast(groups, order)
    if fast is not None:
        order_groups, _first, members = fast
    else:
        order_groups, _first = _group_order(groups, order)
        members = {}
        for i, g in enumerate(groups.tolist()):
            members.setdefault(_group_key(g), []).append(i)
    out = {name: [] for name in SPLIT_NAMES}
    current = 0
    for g in order_groups:
        while (current < 2 and
               len(out[SPLIT_NAMES[current]]) >= targets[current]):
            current += 1
        out[SPLIT_NAMES[current]].extend(int(i) for i in members[g])
    # Preserve chronological order within each split when an ordering key is
    # supplied: sorting only by row index would hand back a shuffled
    # time-series split (the non-grouped path explicitly preserves time
    # order).  Without an order key, ascending row index is the deterministic
    # first-appearance order.
    res = {}
    for name, v in out.items():
        arr = np.asarray(v, dtype=np.int64)
        if arr.size and order is not None:
            _ord = np.asarray(order)
            try:
                arr = arr[np.argsort(_ord[arr], kind='stable')]
            except (TypeError, ValueError, OverflowError):
                arr = np.sort(arr)
        else:
            arr = np.sort(arr)
        res[name] = arr
    # A requested non-empty split must not silently be empty.  An empty
    # validation set is not a harmless corner case: selection/early stopping
    # then has no legal data and callers are tempted to fall back to test.
    for _name, _target in zip(SPLIT_NAMES, targets):
        if _target > 0 and res[_name].size == 0:
            raise ValueError(
                f'grouped split has no rows for the requested {_name} split '
                f'(target={_target}); the group sizes cannot satisfy the '
                'requested fractions — change the split targets or merge groups')
    # Groups must be pairwise disjoint even if the caller supplied a target
    # that forced a large group to one side.
    _assert_index_and_group_disjoint(res, groups)
    if order is not None:
        # Whole groups are on one side, but two groups whose time intervals
        # overlap must not land on opposite sides: that would still leak the
        # other side's future/past through the shared time window.  Refuse the
        # split instead of returning a silently leaking one.
        assert_no_split_leakage(res['train'], res['val'], res['test'],
                                groups=groups, time=order)
    return res


def grouped_chronological_split(data, groups, train_frac=0.8, val_frac=0.1,
                                test_frac=0.1, *, order=None,
                                train_size=None, val_size=None, test_size=None,
                                purge=0):
    """Array-returning wrapper around :func:`grouped_chronological_split_indices`."""
    idx = grouped_chronological_split_indices(
        len(data), groups, train_frac, val_frac, test_frac, order=order,
        train_size=train_size, val_size=val_size, test_size=test_size,
        purge=purge)
    return {name: _take(data, idx[name]) for name in SPLIT_NAMES}


def split_indices(n, *, time=None, groups=None, train_frac=0.8, val_frac=0.1,
                  test_frac=0.1, train_size=None, val_size=None, test_size=None,
                  purge=0):
    """Unified leakage-safe split entry point.

    * ``groups`` given: whole-group chronological split (ordered by ``time``
      when supplied, else first appearance).
    * ``time`` given without groups: rows are ordered by time before the
      contiguous split, so the returned arrays are the original row indices in
      chronological order.
    * neither: contiguous split in the input order (no shuffle).

    ``purge`` is an optional embargo (in rows) that removes the tail of train
    and validation from the returned indices, so overlapping windows cannot
    straddle a split boundary.  It is refused for grouped data because rows of
    a group must stay together.
    """
    n = _check_count('n', n)
    purge = _check_purge(purge)
    if groups is not None:
        return grouped_chronological_split_indices(
            n, groups, train_frac, val_frac, test_frac, order=time,
            train_size=train_size, val_size=val_size, test_size=test_size,
            purge=purge)
    if time is not None:
        time = _as_1d(time, 'time', n)
        _assert_order_finite(time, name='time')
        try:
            order = np.argsort(time, kind='stable')
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                'time values are not mutually orderable; cannot build a '
                'chronological split') from None
        sorted_time = time[order]
        sizes = _resolve_sizes(n, train_frac, val_frac, test_frac,
                               train_size, val_size, test_size)
        # Split boundaries must fall between timestamp groups: a repeated
        # timestamp is one event/batch and may not straddle train/val/test.
        # Build the list of legal cut positions (0, every timestamp change, n)
        # and snap each cumulative target to the first legal cut at or after
        # it.  This may move a few rows between adjacent splits, but never
        # splits an equal-timestamp block.
        if sorted_time.dtype == object:
            # Elementwise comparison on object arrays can raise per element;
            # the per-row form treats such a row as a cut.
            cuts = [0]
            for i in range(1, n):
                try:
                    changed = bool(sorted_time[i] != sorted_time[i - 1])
                except (TypeError, ValueError, OverflowError):
                    changed = True
                if changed:
                    cuts.append(i)
            cuts.append(n)
        else:
            # Same booleans as the per-row loop, vectorised: element i of
            # `changed` is exactly `sorted_time[i] != sorted_time[i-1]` for
            # i in 1..n-1, and every True element contributes the same cut
            # position i (stored at i-1, hence the +1 shift).
            changed = np.asarray(sorted_time[1:] != sorted_time[:-1]).reshape(-1)
            cuts = np.concatenate((
                np.zeros(1, dtype=np.int64),
                np.nonzero(changed)[0].astype(np.int64) + 1,
                np.full(1, n, dtype=np.int64),
            ))
        blocks = []
        start = 0
        start_pos = 0
        cumulative = 0
        _cuts_arr = np.asarray(cuts)
        for _si, size in enumerate(sizes):
            cumulative += int(size)
            # Reserve one legal cut for every later non-empty split, so an
            # earlier legal boundary can satisfy every split instead of
            # consuming the last timestamp block for the first one.
            _later = sum(1 for _s in sizes[_si + 1:] if int(_s) > 0)
            max_pos = len(cuts) - 1 - _later
            pos = int(np.searchsorted(_cuts_arr, cumulative, side='left'))
            # Never move backwards, and force progress for a non-empty split.
            pos = max(pos, start_pos + (1 if int(size) > 0 else 0))
            pos = min(max(pos, 0), max_pos)
            if pos <= start_pos and int(size) > 0:
                raise ValueError(
                    'cannot form the requested non-empty chronological split '
                    'without cutting an equal-timestamp block; supply groups '
                    '(grouped_chronological_split_indices) or fewer split rows')
            end = int(cuts[pos])
            end = max(end, start)
            if int(size) > 0 and end <= start:
                raise ValueError(
                    'cannot form the requested non-empty chronological split '
                    'without cutting an equal-timestamp block; supply groups '
                    '(grouped_chronological_split_indices) or fewer split rows')
            # Preserve the chronological order: the caller indexes its data
            # with these arrays, so sorting the row indices would silently
            # return each split in original row order (e.g. val times
            # [7, 6]) instead of time order.
            blocks.append(order[start:end].copy())
            start = end
            start_pos = pos
        return _purge_tail(dict(zip(SPLIT_NAMES, blocks)), purge)
    return chronological_split_indices(
        n, train_frac, val_frac, test_frac,
        train_size=train_size, val_size=val_size, test_size=test_size,
        purge=purge)


def _group_key(value):
    """Hashable group identity; NaN/NaT rows collapse into one missing group.

    A missing group id must not behave like "unique per row": otherwise rows
    of the same missing group can be split across train/val/test and no
    group-overlap check will ever fire.
    """
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return ('__nan__',)
    if isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value):
        return ('__nat__',)
    try:
        hash(value)
    except TypeError:
        # Group ids are normally hashable scalars, but an object column of
        # lists/arrays is grouped by a stable identity.
        return ('__unhashable__', type(value).__name__, repr(value))
    return value


def _assert_index_and_group_disjoint(split_idx, groups=None):
    # Row-index disjointness is a pure set operation on the int64 arrays.
    arrays = {name: np.asarray(split_idx[name]).reshape(-1)
              for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            if np.intersect1d(arrays[a], arrays[b], assume_unique=False).size:
                raise ValueError(f'{a} and {b} indices overlap')
    if groups is not None:
        sets = {name: set(arrays[name].tolist()) for name in SPLIT_NAMES}
        groups = np.asarray(groups)
        gsets = {name: {_group_key(groups[i]) for i in sets[name]}
                 for name in SPLIT_NAMES}
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
    split_idx = {
        'train': _validated_index(train_idx, name='train_idx'),
        'val': _validated_index(val_idx, name='val_idx'),
        'test': _validated_index(test_idx, name='test_idx'),
    }
    max_index = -1
    for name in SPLIT_NAMES:
        arr = split_idx[name]
        if arr.size:
            max_index = max(max_index, int(arr.max()))
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1:
            groups = groups.reshape(-1)
        if max_index >= 0 and groups.size <= max_index:
            raise ValueError(
                f'groups has length {groups.size}, but split indices reach '
                f'{max_index}; pass one group id per row')
    if time is not None:
        time = np.asarray(time)
        if time.ndim != 1:
            time = time.reshape(-1)
        if max_index >= 0 and time.size <= max_index:
            raise ValueError(
                f'time has length {time.size}, but split indices reach '
                f'{max_index}; pass one timestamp per row')
        # One validator for every dtype (numeric NaN/Inf, datetime64 NaT and
        # object scalars).  The ordering checks below assume a well-defined
        # order; skipping this for object arrays let a NaN timestamp pass.
        _assert_order_finite(time, name='time')
    _assert_index_and_group_disjoint(split_idx, groups)
    if time is not None:
        for a, b in (('train', 'val'), ('val', 'test'), ('train', 'test')):
            ia = np.asarray(split_idx[a], dtype=np.int64).reshape(-1)
            ib = np.asarray(split_idx[b], dtype=np.int64).reshape(-1)
            if ia.size == 0 or ib.size == 0:
                continue
            # ``>=`` (not ``>``): rows sharing a timestamp are one event/time
            # block.  Putting the same timestamp on two sides leaks the block's
            # future/past into the other split even though the row indices are
            # disjoint.  Callers must pass groups (or split at a timestamp
            # boundary) when timestamps repeat.
            try:
                earlier_max = _order_extreme(time[ia], maximum=True)
                later_min = _order_extreme(time[ib], maximum=False)
                leaked = bool(earlier_max >= later_min)
            except (TypeError, ValueError, OverflowError):
                # Mixed/unorderable object timestamps cannot prove a monotone
                # partition; fail closed rather than letting a TypeError
                # escape from what is supposed to be a verification helper.
                raise ValueError(
                    f'temporal leakage cannot be verified for {a}/{b}: '
                    'time values are not mutually orderable') from None
            if leaked:
                raise ValueError(
                    f'temporal leakage: max {a} time '
                    f'{earlier_max!r} >= min {b} time {later_min!r} '
                    '(the same timestamp/event block must not span two splits)')
    return True


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing
# ---------------------------------------------------------------------------

def _as_numeric_2d(X, *, name='X'):
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 1-D or 2-D, got shape {arr.shape}')
    return arr


def _as_float_2d(X, *, name='X'):
    arr = _as_numeric_2d(X, name=name)
    if np.iscomplexobj(arr):
        # NumPy would cast complex -> float64 with only a warning, silently
        # discarding the imaginary part.  Two different complex rows could
        # then collapse into the same "duplicate" and the fitted statistics
        # would describe data the caller never supplied.
        raise TypeError(
            f'{name} must be real-valued; complex features are not supported '
            'by the leakage-safe preprocessor')
    try:
        arr = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(
            f'{name} must be numeric; encode categorical features before '
            'calling the leakage-safe preprocessor') from None
    return arr


def _col_float64(arr, j, name='X'):
    col = arr[:, j]
    if np.iscomplexobj(col):
        raise TypeError(
            f'{name} must be real-valued; complex features are not supported '
            'by the leakage-safe preprocessor')
    try:
        return np.asarray(col, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(
            f'{name} must be numeric; encode categorical features before '
            'calling the leakage-safe preprocessor') from None


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
        if isinstance(missing, (bool, np.bool_)):
            raise ValueError(
                f'missing must be a finite constant or one of '
                f"'median'/'mean'/'zero', got {missing!r}")
        if isinstance(missing, (numbers.Real, np.integer, np.floating)):
            missing = float(missing)
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
            lo, hi = float(lo), float(hi)
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError('clip_quantiles must satisfy 0 <= lo < hi <= 1')
            clip_quantiles = (lo, hi)
        if isinstance(clip_z, (bool, np.bool_)):
            raise ValueError('clip_z must be positive or None, not a boolean')
        if clip_z is not None:
            clip_z = float(clip_z)
            if not (math.isfinite(clip_z) and clip_z > 0.0):
                raise ValueError('clip_z must be a finite positive number or None')
        if isinstance(eps, (bool, np.bool_)):
            raise ValueError('eps must be a finite non-negative number, not a boolean')
        eps = float(eps)
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
            return float(np.mean(values))
        if self.missing == 'median':
            return float(np.median(values))
        return 0.0

    def fit(self, X, y=None):
        """Learn imputation/clipping/scaling statistics from ``X`` (train)."""
        X = _as_numeric_2d(X)
        n, p = X.shape
        if n == 0:
            raise ValueError('cannot fit the preprocessor on zero training rows')
        stats = []
        for j in range(p):
            # Convert one column at a time to float64.
            base = _col_float64(X, j)
            col = np.where(np.isfinite(base), base, np.nan)
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


def _as_categorical_2d(X, *, name='X'):
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 1-D or 2-D, got shape {arr.shape}')
    return arr


def _category_is_missing(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _sorted_categories(values):
    unique = {}
    for value in values:
        if _category_is_missing(value):
            continue
        unique.setdefault(_hashable_label(value), value)
    return sorted(unique.values(),
                  key=lambda v: (type(v).__name__, repr(v)))


def _categorical_fingerprint(X):
    h = hashlib.sha256()
    arr = np.asarray(X, dtype=object)
    h.update(str(arr.shape).encode('ascii'))
    for row in arr.tolist():
        h.update(repr(row).encode('utf-8', errors='replace'))
        h.update(b'\x1f')
    return h.hexdigest()[:16]


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
                lookup = {_hashable_label(c): i
                          for i, c in enumerate(st['categories'])}
                for i, value in enumerate(Xarr[:, j].tolist()):
                    if _category_is_missing(value):
                        out[i, j] = st['missing_index']
                        continue
                    code = lookup.get(_hashable_label(value))
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
            lookup = {_hashable_label(c): i
                      for i, c in enumerate(st['categories'])}
            span = st['n_categories'] + 2
            for i, value in enumerate(Xarr[:, j].tolist()):
                if _category_is_missing(value):
                    out[i, offset + st['missing_index']] = 1.0
                    continue
                code = lookup.get(_hashable_label(value))
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
            'features': [dict(s, categories=list(s.get('categories') or []))
                         for s in self.feature_stats_],
        }

    @classmethod
    def from_stats(cls, stats):
        """Rebuild a fitted encoder from :meth:`get_stats` output."""
        obj = cls(encoding=stats['encoding'],
                  handle_unknown=stats.get('handle_unknown', 'unknown'))
        obj.feature_stats_ = [dict(s, categories=list(s.get('categories') or []))
                              for s in stats['features']]
        obj.n_features_in_ = int(stats['n_features_in'])
        obj.n_train_ = int(stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


def fit_categorical_encoder(X_train, **kwargs):
    """Fit a categorical encoder on training rows only."""
    return LeakageSafeCategoricalEncoder(**kwargs).fit(X_train)


def prepare_categorical_features(X_train, X_val=None, X_test=None, **kwargs):
    """Encode train categories first, then transform the holdouts."""
    encoder = LeakageSafeCategoricalEncoder(**kwargs).fit(X_train)
    out = {'train': encoder.transform(X_train), 'encoder': encoder}
    if X_val is not None:
        out['val'] = encoder.transform(X_val)
    if X_test is not None:
        out['test'] = encoder.transform(X_test)
    return out


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


def clean_training_rows(X, y=None, *, train_idx, drop_duplicates=True,
                        label_conflicts='drop'):
    """Remove duplicate/ambiguous rows from the training split only.

    The holdout indices are not accepted by this function, so no validation or
    test row can be dropped, imputed or used to decide which duplicate to keep.
    Rows are grouped by their exact feature key; the first occurrence in the
    supplied training order is retained.  When duplicate rows carry different
    non-missing labels, ``label_conflicts='drop'`` removes the whole ambiguous
    cluster (the safe choice before fitting), ``'keep'`` retains it and
    ``'raise'`` refuses to continue.

    Returns ``train_idx`` (the cleaned index array), the dropped rows and
    counts suitable for a run manifest.
    """
    Xraw = _as_numeric_2d(X)
    Xarr = _as_float_2d(Xraw)
    n = int(Xarr.shape[0])
    train_rows = _validated_index(train_idx, name='train_idx')
    if train_rows.size == 0:
        raise ValueError('clean_training_rows: the training split is empty')
    if int(train_rows.max()) >= n:
        raise ValueError(
            f'clean_training_rows: train_idx reaches {int(train_rows.max())} '
            f'but X has {n} rows')
    if label_conflicts not in ('drop', 'keep', 'raise'):
        raise ValueError(
            f"label_conflicts must be 'drop', 'keep' or 'raise', "
            f"got {label_conflicts!r}")
    labels = None
    if y is not None:
        yarr = np.asarray(y)
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

    def _missing(lab):
        # Any missing output makes the multi-output label unusable, matching
        # data_quality_report / _label_key.  Requiring every output to be
        # missing let a partially missing label masquerade as a real label and
        # trigger a false duplicate-label conflict.
        return lab is None or (isinstance(lab, tuple)
                               and any(v is None for v in lab))

    keys = _row_keys(Xraw)
    clusters = {}
    for pos, row in enumerate(train_rows.tolist()):
        clusters.setdefault(keys[row], []).append(pos)
    drop = np.zeros(train_rows.size, dtype=bool)
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
        'drop_duplicates': bool(drop_duplicates),
        'label_conflicts': label_conflicts,
    }


def split_and_prepare(X, y=None, *, time=None, groups=None,
                      train_frac=0.8, val_frac=0.1, test_frac=0.1,
                      train_size=None, val_size=None, test_size=None,
                      purge=0, preprocess=True, quality_report=False,
                      clean_train=False, label_conflicts='drop',
                      quality_kwargs=None, **preprocessor_kwargs):
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
    ``clean_train=True`` first removes exact duplicate training rows and, by
    default, ambiguous duplicate-label clusters via
    :func:`clean_training_rows`; validation/test rows are never dropped or
    inspected.  The returned ``indices['train']`` is the cleaned index, so the
    preprocessor and every downstream step see exactly the same rows.

    Returns a dict with ``indices``, per-split ``X`` (and ``y``), the fitted
    ``preprocessor`` (or ``None`` when ``preprocess=False``), an optional
    ``cleaning`` record and optionally the ``quality`` report.
    """
    Xarr = _as_numeric_2d(X)
    n = int(Xarr.shape[0])
    yarr = None
    if y is not None:
        yarr = np.asarray(y)
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
            label_conflicts=label_conflicts)
        idx['train'] = cleaning['train_idx']
        assert_no_split_leakage(idx['train'], idx['val'], idx['test'],
                                groups=groups, time=time)
    raw_X = {name: _take(Xarr, idx[name]) for name in SPLIT_NAMES}
    out = {'indices': idx, 'X': dict(raw_X)}
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
    if quality_report:
        kw = dict(quality_kwargs or {})
        # Forward the same ordering/group keys that produced the split so the
        # report can independently verify temporal and whole-group isolation.
        # An explicit quality_kwargs override wins (callers may pass a longer
        # external ordering array), but it must not be duplicated.
        kw.setdefault('time', time)
        kw.setdefault('groups', groups)
        out['quality'] = data_quality_report(
            Xarr, y=yarr, train_idx=idx['train'], val_idx=idx['val'],
            test_idx=idx['test'], **kw)
    return out


# ---------------------------------------------------------------------------
# Data-quality diagnostics
# ---------------------------------------------------------------------------

_HASH_BLOCK_BYTES = 64 << 20


def _hash_row_block(width, cap):
    """Rows per hashing block under a byte ceiling (at least one row)."""
    return max(1, min(int(cap), _HASH_BLOCK_BYTES // max(1, 8 * int(width))))


def _canonical_complex_cell(v):
    """Canonical identity of a complex scalar (object or NumPy spelling).

    The typed table's ``.tolist()`` yields a Python complex, while an object
    table may hold ``np.complex64``/``np.complex128`` scalars.  Both spellings
    are the same (real, imaginary) pair; the parts use the same NaN/Inf/zero
    canonicalisation as the real path so a missing/overflowed component is one
    identity and ``-0.0`` matches ``0.0``.
    """
    z = complex(v)
    def _part(x):
        if math.isnan(x):
            return ('nan',)
        if math.isinf(x):
            return ('inf', 1 if x > 0 else -1)
        return ('num', 0.0 if x == 0.0 else x)
    return ('cplx', _part(z.real), _part(z.imag))


def _temporal_identity(v):
    """Unit-independent identity for a datetime64/timedelta64 value.

    NumPy compares the same instant in different units as equal, so the
    duplicate/fingerprint contracts must do the same.  Normalise to
    microseconds (the finest unit that covers ordinary dates without
    overflowing); fall back to the textual spelling only when that conversion
    is out of range.
    """
    if np.isnat(v):
        return ('nat', type(v).__name__)
    try:
        u = (v.astype('datetime64[us]') if isinstance(v, np.datetime64)
             else v.astype('timedelta64[us]'))
        if not np.isnat(u):
            return ('dt', type(v).__name__, int(u.astype(np.int64)))
    except (OverflowError, ValueError, TypeError):
        pass
    return ('dt', type(v).__name__, str(v))


def _canonical_object_cell(v):
    if v is None:
        # None is the object spelling of a missing numeric cell, the same
        # value the numeric paths materialise as NaN.
        return ('nan',)
    if isinstance(v, (bytes, bytearray, np.bytes_)):
        # A bytearray is the mutable spelling of the same byte string, and an
        # object-serialised np.bytes_ must match its Python-bytes twin; use the
        # bytes content so an object table and a fixed-width bytes table share
        # one identity.  (np.asarray(list_of_bytearray, dtype=bytes) cannot
        # convert a MIXED bytes/bytearray column at all.)
        v = bytes(v)
    elif isinstance(v, str):
        # np.str_ is a str subclass whose type name differs; normalise to the
        # plain str spelling so an object table and its fixed-width Unicode
        # twin share one canonical identity (and fingerprint).
        v = str(v)
    if isinstance(v, (np.datetime64, np.timedelta64)):
        # Checked before the integer branches: np.timedelta64 is registered as
        # np.integer, but int() on a unit-bearing value is not an integer.
        return _temporal_identity(v)
    if isinstance(v, (bool, np.bool_)):
        return ('num', int(v))
    if isinstance(v, (int, np.integer)):
        return ('num', int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return ('nan',)
        if math.isinf(f):
            return ('inf', 1 if f > 0 else -1)
        return ('num', 0.0 if f == 0.0 else f)
    if isinstance(v, (complex, np.complexfloating)):
        return _canonical_complex_cell(v)
    try:
        hash(v)
    except TypeError:
        return ('unhashable', type(v).__name__, repr(v))
    return ('obj', type(v).__name__, v)


def _matrix_fingerprint(X):
    arr = np.asarray(X)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('ascii'))
    # Convert and hash one row block at a time; the block size follows the
    # actual width.
    _w = int(arr.shape[1]) if arr.ndim > 1 else 1
    _step = _hash_row_block(_w, 1 << 20)
    # float64 preserves integers up to 2**53 exactly.  Beyond that two
    # different token ids can round to the same float and produce a false
    # corpus fingerprint, so those integral tables are hashed as exact
    # uint64 instead.  In-range tables keep the same fingerprint.
    _integral = (arr.dtype != object
                 and (np.issubdtype(arr.dtype, np.integer)
                      or np.issubdtype(arr.dtype, np.bool_)))
    _obj_int = False
    _obj_fits = True
    _obj_big = False
    # Fixed-width Unicode/bytes and complex tables use the canonical per-cell
    # identity: float64 conversion would discard a complex imaginary part.
    _obj_canon = arr.dtype.kind in ('U', 'S', 'c')
    # Native datetime/timedelta tables share the canonical temporal identity
    # of an object column of np.datetime64/np.timedelta64 scalars.
    _temporal = arr.dtype.kind in ('M', 'm')
    if arr.dtype == object and arr.size:
        # Object-dtype integer tables must keep the same exactness contract:
        # the float64 fallback collapses 2**53 and 2**53+1.
        _vals = arr.reshape(-1).tolist()
        # np.timedelta64 is registered as numbers.Integral, but int() on a
        # unit-bearing value returns datetime.timedelta (or raises), so it is
        # a temporal value, not an integer cell.  Route every datetime-like
        # object cell through the canonical temporal identity.
        _obj_dt = any(isinstance(v, (np.datetime64, np.timedelta64))
                      for v in _vals)
        # Booleans are integral in the row-key contract (0/1), so include them
        # here too: an object table such as [[True, 2**53]] stays on the exact
        # integer path, where 2**53 and 2**53+1 remain distinct as the
        # duplicate detector requires.
        _obj_int = (not _obj_dt) and all(
            isinstance(v, numbers.Integral)
            or isinstance(v, (bool, np.bool_))
            for v in _vals)
        if _obj_int:
            _obj_fits = all(-(2 ** 63) <= int(v) <= 2 ** 64 - 1 for v in _vals)
            if not _obj_fits:
                # No fixed-width dtype can hold these values exactly, so no
                # cross-width twin exists.  Hash their canonical decimal
                # spelling (length-prefixed) to keep distinct integers
                # distinct.
                _obj_int = False
                _obj_big = True
    if arr.dtype == object and arr.size and not _obj_int and not _obj_big:
        _vals = arr.reshape(-1).tolist()
        # None is a missing numeric cell (the numeric paths turn it into NaN),
        # so it shares the float64 fingerprint path with real values.  A
        # datetime/timedelta object cell is temporal metadata, never a float.
        if (_obj_dt
                or not all(isinstance(v, numbers.Real) or v is None for v in _vals)):
            _obj_canon = True
        elif any(isinstance(v, numbers.Integral) and abs(int(v)) > 2 ** 53
                 for v in _vals):
            _obj_canon = True
    _integral = _integral or _obj_int
    _wide_int = False
    _wide_neg = False
    if _integral and arr.size:
        _lim = 2 ** 53
        if arr.dtype == object:
            _vals_wide = [int(v) for v in arr.reshape(-1).tolist()]
            # 2**53 is exactly representable as float64; use the same strict
            # comparison as the typed path below so object and typed tables
            # share one boundary convention.
            _wide_int = bool(any(abs(v) > _lim for v in _vals_wide))
            _wide_neg = bool(any(v < 0 for v in _vals_wide))
        elif np.issubdtype(arr.dtype, np.unsignedinteger):
            _wide_int = bool((arr > _lim).any())
        else:
            _wide_int = bool(((arr > _lim) | (arr < -_lim)).any())
            _wide_neg = bool((arr < 0).any())
    for start in range(0, arr.shape[0], _step):
        _blk = arr[start:start + _step]
        if _temporal:
            h.update(b'OBJ_CANON')
            # Iterate the native scalars (not .tolist(), which converts a
            # datetime64 to date/int and would lose the unit identity).
            for v in _blk.reshape(-1):
                _s = repr(_canonical_object_cell(v)).encode('utf-8', 'backslashreplace')
                h.update(len(_s).to_bytes(8, 'little'))
                h.update(_s)
            continue
        if _obj_big:
            h.update(b'OBJ_BIG')
            for v in _blk.reshape(-1).tolist():
                _s = str(int(v)).encode('ascii')
                h.update(len(_s).to_bytes(8, 'little'))
                h.update(_s)
            continue
        if _obj_canon:
            h.update(b'OBJ_CANON')
            for v in _blk.reshape(-1).tolist():
                _s = repr(_canonical_object_cell(v)).encode('utf-8', 'backslashreplace')
                h.update(len(_s).to_bytes(8, 'little'))
                h.update(_s)
            continue
        if _wide_int:
            if _blk.dtype == object:
                _flat_vals = [int(v) for v in _blk.reshape(-1).tolist()]
                _flat = [v & (2 ** 64 - 1) for v in _flat_vals]
                block = np.asarray(_flat, dtype=np.uint64).reshape(_blk.shape)
                if _wide_neg:
                    h.update(np.ascontiguousarray(
                        [v < 0 for v in _flat_vals], dtype=np.uint8).tobytes())
            else:
                block = np.ascontiguousarray(_blk, dtype=np.uint64)
                if _wide_neg:
                    # Encode the sign so -1 and 2**64-1 have distinct
                    # fingerprints.
                    h.update(np.ascontiguousarray(_blk < 0, dtype=np.uint8).tobytes())
        else:
            block = np.asarray(_blk, dtype=np.float64)
            # The content fingerprint must use the same logical-equality
            # rule as the duplicate detector.  `-0.0 == 0.0` but their byte
            # patterns differ, so canonicalise the sign of zero.  Every NaN
            # is likewise one missing value to `_row_keys` (it hashes the
            # non-finite POSITIONS and a NaN/Inf code, not the NaN payload),
            # so normalise NaN's sign/payload as well instead of letting a
            # cache round-trip change the corpus fingerprint.
            if np.any(block == 0.0) or np.any(np.isnan(block)):
                block = block.copy()
                block[block == 0.0] = 0.0
                block[np.isnan(block)] = np.nan
        h.update(np.ascontiguousarray(block).tobytes())
    return h.hexdigest()[:16]


def _row_keys(X):
    """Stable per-row keys for exact-duplicate detection (NaN-safe)."""
    src = np.asarray(X)
    if src.ndim == 0:
        raise ValueError('duplicate detection needs a 1-D or 2-D table, got a scalar')
    if src.ndim > 2:
        raise ValueError(
            f'duplicate detection needs a 1-D or 2-D table, got shape {src.shape}')
    if src.ndim == 1:
        src = src.reshape(-1, 1)
    if src.dtype == object:
        # An all-integer object table has an exact fixed-width twin.  Route it
        # through the same uint64/int64 canonicalisation as a typed table so
        # the same logical row has the same key whether a cache serialised it
        # as object, int64 or uint64, keeping cross-dtype duplicates visible
        # to the duplicate audit.
        _flat_obj = src.reshape(-1).tolist()
        # np.timedelta64 is numbers.Integral but int() on a unit-bearing value
        # is not an integer; keep every datetime-like cell on the temporal
        # canonical path below.
        _flat_dt = any(isinstance(v, (np.datetime64, np.timedelta64))
                       for v in _flat_obj)
        # Booleans are integral in Python/NumPy's canonical integer path, so
        # include them here as 0/1: an object-dtype bool table must produce the
        # same row keys as its typed bool/int twin.
        if _flat_obj and not _flat_dt and all(
                isinstance(v, numbers.Integral)
                or isinstance(v, (bool, np.bool_))
                for v in _flat_obj):
            _ivs = [int(v) for v in _flat_obj]
            if all(-(2 ** 63) <= v <= 2 ** 63 - 1 for v in _ivs):
                return _row_keys(np.asarray(_ivs, dtype=np.int64).reshape(src.shape))
            if all(0 <= v <= 2 ** 64 - 1 for v in _ivs):
                return _row_keys(np.asarray(_ivs, dtype=np.uint64).reshape(src.shape))
        # A mixed real-number object table is the same logical table as its
        # float64 twin.  Route it through the typed float path when every
        # integral value is exactly representable as float64; otherwise keep
        # the exact per-cell keys below (2**53 and 2**53+1 must stay
        # distinct).  Without this, an object-serialised cache produces
        # different duplicate keys from the typed cache with identical rows.
        # None is a missing numeric cell, the same value as NaN, so it shares
        # this path instead of becoming a distinct NoneType object key.
        if _flat_obj and not _flat_dt and all(
                isinstance(v, numbers.Real)
                or isinstance(v, (bool, np.bool_))
                or v is None
                for v in _flat_obj):
            _real_exact = all(v is None or not isinstance(v, numbers.Integral)
                              or abs(int(v)) <= 2 ** 53
                              for v in _flat_obj)
            if _real_exact:
                return _row_keys(np.asarray(_flat_obj, dtype=np.float64).reshape(src.shape))
        # A pure string object table is the logical twin of a fixed-width
        # Unicode/bytes table; route it through that path so a cache that
        # serialised strings as object dtype still matches its typed copy.
        if (_flat_obj and all(isinstance(v, (str, np.str_)) for v in _flat_obj)
                and all('\x00' not in str(v) for v in _flat_obj)):
            # Fixed-width NumPy strings cannot preserve embedded NULs (they
            # truncate at the first NUL).  Keep NUL-bearing object strings on
            # the exact per-cell path below so they cannot collide with a
            # shorter string.
            return _row_keys(np.asarray(_flat_obj, dtype=str).reshape(src.shape))
        if (_flat_obj and all(isinstance(v, (bytes, bytearray, np.bytes_))
                              for v in _flat_obj)
                and all(b'\x00' not in bytes(v) for v in _flat_obj)):
            # Convert each cell explicitly: np.asarray([b'a', bytearray(b'b')],
            # dtype=bytes) treats the bytearray as a sequence and refuses the
            # inhomogeneous table.  The byte content is the same identity.
            return _row_keys(np.asarray([bytes(v) for v in _flat_obj],
                                        dtype=bytes).reshape(src.shape))
        # Object tables are often mixed/non-contiguous; float64 does not apply
        # to string columns and loses integer precision above 2**53.  Build an
        # exact canonical key per cell instead.
        def _cell_key(v):
            if v is None:
                # Same missing-cell identity as the numeric float path.
                return ('nan',)
            if isinstance(v, (bytes, bytearray, np.bytes_)):
                # Match the fixed-width bytes path (and the canonical object
                # fingerprint): the byte content, not the mutable container.
                v = bytes(v)
            elif isinstance(v, str):
                # np.str_ is a str subclass with a different type name; use
                # the plain-str spelling for the same canonical identity.
                v = str(v)
            if isinstance(v, (np.datetime64, np.timedelta64)):
                # np.timedelta64 is np.integer, but int() on a unit-bearing
                # value is not an integer; handle temporal identity first.
                return _temporal_identity(v)
            if isinstance(v, (bool, np.bool_)):
                # Match the typed integral path, which hashes bool as 0/1.
                return ('num', int(v))
            if isinstance(v, (int, np.integer)):
                return ('num', int(v))
            if isinstance(v, (float, np.floating)):
                f = float(v)
                if math.isnan(f):
                    return ('nan',)
                if math.isinf(f):
                    return ('inf', 1 if f > 0 else -1)
                return ('num', 0.0 if f == 0.0 else f)
            if isinstance(v, (complex, np.complexfloating)):
                return _canonical_complex_cell(v)
            try:
                hash(v)
            except TypeError:
                return ('unhashable', type(v).__name__, repr(v))
            return ('obj', type(v).__name__, v)

        return [tuple(_cell_key(v) for v in row) for row in src]
    keys = []
    if src.dtype.kind == 'c':
        # A complex cell is a pair (real, imaginary); the raw-byte path below
        # is width-dependent and the float path discards the imaginary part.
        # Reuse the exact canonical per-cell identity of an object table.
        return _row_keys(src.astype(object))
    if src.dtype.kind in ('M', 'm'):
        # Native datetime/timedelta arrays must share the object-dtype twin's
        # temporal identity (including NaT).  Assign each scalar explicitly:
        # astype(object) would turn datetime64[D] into datetime.date and
        # datetime64[ns] into an int, losing both the type and the unit.
        obj = np.empty(src.shape, dtype=object)
        src_flat = src.reshape(-1)
        obj_flat = obj.reshape(-1)
        for i in range(int(src_flat.size)):
            obj_flat[i] = src_flat[i]
        return _row_keys(obj)
    if src.dtype.kind in ('U', 'S'):
        _tag = b'U' if src.dtype.kind == 'U' else b'S'
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        for start in range(0, src.shape[0], _step):
            for row in src[start:start + _step].tolist():
                h = hashlib.blake2b(digest_size=8)
                h.update(_tag)
                for cell in row:
                    _b = (cell.encode('utf-8', errors='backslashreplace')
                          if isinstance(cell, str) else bytes(cell))
                    h.update(len(_b).to_bytes(8, 'little'))
                    h.update(_b)
                keys.append(h.digest())
        return keys
    # Integer/bool/datetime/string columns have an exact fixed-width byte
    # representation.  Converting them through float64 loses integers above
    # 2**53 (2**53 and 2**53+1 would hash as the same row and one would be
    # dropped as a false duplicate), so hash the raw row bytes instead.
    # Integral tables are first canonicalised to uint64: the same logical
    # integer then hashes identically whether a cache stored it as int32 or
    # int64 (raw bytes are width-dependent and would miss cross-dtype
    # leakage).  uint64 keeps every integer value exact, including negatives
    # via two's complement.
    _integral = (src.dtype != object
                 and (np.issubdtype(src.dtype, np.integer)
                      or np.issubdtype(src.dtype, np.bool_)))
    if _integral or (src.dtype != object and not np.issubdtype(src.dtype, np.floating)):
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        for start in range(0, src.shape[0], _step):
            _blk = src[start:start + _step]
            if _integral:
                arr = np.ascontiguousarray(_blk, dtype=np.uint64)
            else:
                arr = np.ascontiguousarray(_blk)
            # A signed negative row must not collide with the raw unsigned
            # value (int64 -1 vs uint64 2**64-1).  Tag only rows that carry a
            # negative value, so the same nonnegative logical row still
            # matches across all integer widths.
            try:
                _neg_rows = np.asarray(_blk < 0).reshape(_blk.shape[0], -1).any(axis=1)
            except (TypeError, ValueError):
                _neg_rows = np.zeros(_blk.shape[0], dtype=bool)
            for _ri, row_view in enumerate(arr.view(np.uint8).reshape(arr.shape[0], -1)):
                h = hashlib.blake2b(digest_size=8)
                h.update(b'N' if bool(_neg_rows[_ri]) else b'P')
                h.update(memoryview(row_view))
                keys.append(h.digest())
        return keys
    # Hash row blocks; each floating block is converted to float64 before
    # hashing (values are preserved exactly for float16/32/64).
    _step = _hash_row_block(int(src.shape[1]), 1 << 16)
    for start in range(0, src.shape[0], _step):
        arr = np.ascontiguousarray(np.asarray(src[start:start + _step], dtype=np.float64))
        # ``-0.0 == 0.0`` but their byte patterns differ, so raw hashing would
        # miss an exact duplicate split across train/val/test.  Normalize the
        # sign of zero before hashing (NaN values are handled separately).
        if np.any(arr == 0.0):
            arr = arr.copy()
            arr[arr == 0.0] = 0.0
        finite = np.isfinite(arr)
        if finite.all():
            # Every row is finite, so the key is
            # ``row_bytes + b'|' + packbits(all-False)``.
            _tail = np.packbits(np.zeros(int(src.shape[1]), dtype=bool)).tobytes()
            for row_view in arr.view(np.uint8):
                h = hashlib.blake2b(digest_size=8)
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
            h = hashlib.blake2b(digest_size=8)
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
    try:
        hash(value)
    except TypeError:
        return ('__unhashable__', type(value).__name__, repr(value))
    return value


def _label_key(v):
    if v is None:
        return None
    if isinstance(v, (np.datetime64, np.timedelta64)) and np.isnat(v):
        # NaT is a missing target, just like NaN (object-dtype labels are not
        # normalised by ndarray.tolist() the way native datetime arrays are).
        return None
    # Only actual numeric scalars are checked for NaN/Inf.  Coercing every
    # value through float() turned legitimate categorical labels spelled
    # like "nan"/"inf" into missing targets and also conflated numeric
    # strings with their numeric twins.  Numbers.Real includes NumPy scalar
    # types but excludes complex/bool.
    if not isinstance(v, numbers.Real) or isinstance(v, (bool, np.bool_)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return v
    # NaN and infinities are not valid class/regression targets.  Treating
    # +inf as an ordinary label would hide a corrupted target from the
    # missing-label audit.
    if not math.isfinite(f):
        return None
    return v


def data_quality_report(X, y=None, *, train_idx=None, val_idx=None, test_idx=None,
                        groups=None, time=None, clip_z=3.0, shift_threshold=0.5,
                        known_labels=None,
                        fail_on_nan_inf=True, fail_on_label_conflict=True,
                        fail_on_nonfinite_labels=False,
                        fail_on_label_shift=False,
                        fail_on_unknown_labels=False,
                        fail_on_duplicate_rows=False, fail_on_outliers=False,
                        fail_on_shift=False):
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
    labels are the only reference distribution.

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
    Xraw = _as_numeric_2d(X)
    X = _as_float_2d(Xraw)
    n = X.shape[0]
    known = None
    if known_labels is not None:
        if isinstance(known_labels, (str, bytes)):
            raise ValueError(
                'known_labels must be an iterable of labels, not a single string')
        try:
            if isinstance(known_labels, np.ndarray) and known_labels.ndim > 1:
                # Multi-output y rows are tuples; flattening a 2-D known-label
                # table splits every tuple into scalar outputs and marks all
                # real labels unknown.
                values = [tuple(row) for row in
                          known_labels.reshape(known_labels.shape[0], -1).tolist()]
            else:
                values = (known_labels.reshape(-1).tolist()
                          if isinstance(known_labels, np.ndarray)
                          else list(known_labels))
        except TypeError:
            raise ValueError('known_labels must be a flat iterable of labels') from None
        _y_multi = (y is not None and np.asarray(y).ndim > 1)
        if _y_multi:
            values = [tuple(v) if isinstance(v, (list, tuple, np.ndarray)) else (v,)
                      for v in values]
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
    if ref.size == 0 and n > 0:
        # An explicit empty train_idx (with or without holdouts) audits
        # nothing; every vacuous check would otherwise report "ok".  An empty
        # input table remains a vacuous no-op rather than an error.
        raise ValueError(
            'data_quality_report: the training reference is empty; there are '
            'no training rows whose statistics can be audited')
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

    # Exact duplicates within each split.
    dup = {}
    keys = _row_keys(Xraw)
    for name, rows in split_rows.items():
        sub = [keys[i] for i in rows]
        dup[name] = int(len(sub) - len(set(sub)))
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
        sa = {keys[i] for i in split_rows[a]}
        sb = {keys[i] for i in split_rows[b]}
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
    # and the counts are reported per split so a corrupted holdout cannot
    # hide behind a clean overall total.
    if y is not None:
        yarr = np.asarray(y)
        if yarr.ndim == 0 or yarr.shape[0] != n:
            raise ValueError(
                f'y has {yarr.shape[0] if yarr.ndim else 1} rows, expected {n}')
        if yarr.ndim == 1:
            raw_labels = yarr.tolist()
            labels = [_label_key(v) for v in raw_labels]
        else:
            # Multi-output targets: flatten only along the output axis, not
            # across rows.  reshape(-1) would make the flattened length n*k
            # and reject every valid multi-output target with a misleading
            # length error (clean_training_rows already supports this shape).
            _rows_y = yarr.reshape(n, -1).tolist()
            raw_labels = [tuple(r) for r in _rows_y]
            labels = [tuple(_label_key(v) for v in r) for r in _rows_y]

        def _raw_missing_atom(v):
            if v is None:
                return True
            return (isinstance(v, (np.datetime64, np.timedelta64))
                    and np.isnat(v))

        def _raw_missing(v):
            # For a multi-output target, any missing output makes the row
            # unusable; requiring ALL outputs to be missing hid a partially
            # corrupted row from both the missing and non-finite counters.
            return _raw_missing_atom(v) or (isinstance(v, tuple)
                                            and any(_raw_missing_atom(x) for x in v))

        def _label_missing(v):
            return v is None or (isinstance(v, tuple)
                                 and any(x is None for x in v))

        def _row_nonfinite(label, raw):
            # A row can be BOTH partly missing and partly non-finite
            # (e.g. (None, inf)).  Subtracting the missing count from the
            # missing-label count hid the Inf entirely, so count the two
            # conditions independently at the output level.
            pairs = (zip(label, raw) if isinstance(label, tuple)
                     else ((label, raw),))
            return any(key is None and not _raw_missing_atom(atom)
                       for key, atom in pairs)

        missing_labels = int(sum(_raw_missing(v) for v in raw_labels))
        nonfinite_labels = int(sum(_row_nonfinite(lab, raw)
                                   for lab, raw in zip(labels, raw_labels)))
        checks['missing_labels'] = missing_labels
        checks['nonfinite_labels'] = nonfinite_labels
        checks['missing_labels_by_split'] = {}
        checks['nonfinite_labels_by_split'] = {}
        for name, rows in split_rows.items():
            _rows = np.asarray(rows, dtype=np.int64).reshape(-1).tolist()
            checks['missing_labels_by_split'][name] = int(
                sum(_raw_missing(raw_labels[i]) for i in _rows))
            checks['nonfinite_labels_by_split'][name] = int(
                sum(_row_nonfinite(labels[i], raw_labels[i]) for i in _rows))
        if missing_labels:
            warnings.append(f'{missing_labels} missing labels')
        if nonfinite_labels:
            _msg = (f'{nonfinite_labels} non-finite labels (NaN/Inf); impute '
                    'or remove them before fitting')
            if fail_on_nonfinite_labels:
                hard_failures.append(_msg)
            else:
                warnings.append(_msg)
        by_row = {}
        for i, (k, lab) in enumerate(zip(keys, labels)):
            if not _label_missing(lab):
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
                    if _label_missing(lab):
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
                if not _label_missing(v):
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
        _train_lab = [str(labels[i]) for i in
                      np.asarray(ref, dtype=np.int64).reshape(-1).tolist()
                      if not _label_missing(labels[i])]
        for name in ('val', 'test'):
            _rows = np.asarray(split_rows[name], dtype=np.int64).reshape(-1).tolist()
            if not _rows:
                continue
            _hold_lab = [str(labels[i]) for i in _rows
                         if not _label_missing(labels[i])]
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
            if sd <= _EPS:
                delta = abs(float(np.mean(hv)) - float(np.mean(tv)))
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
                smd.append(float((np.mean(hv) - np.mean(tv)) / sd))
        miss_train = 1.0 - np.mean(np.isfinite(Xt), axis=0)
        miss_hold = 1.0 - np.mean(np.isfinite(Xh), axis=0)
        finite_smd = [abs(v) for v in smd if v is not None]
        # A zero-column table is a valid (if uninformative) input, so the
        # empty-vector case is handled explicitly.
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
    def _categories(values):
        # Categories are hashable identities, not a flat scalar column: a
        # list of tuple labels must stay two categories, not be expanded into
        # four scalars by reshape(-1).  Rows of a genuine 2-D array are one
        # composite category each.
        def _item(v):
            # A list is the unhashable spelling of the same composite
            # category; normalise it so a row from a 2-D array and a tuple
            # supplied by the caller compare equal.
            return tuple(v) if isinstance(v, list) else v

        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                return [_item(values.item())]
            if values.ndim == 1:
                # A 1-D object array can still hold composite categories
                # (e.g. a list label); normalise them exactly like the 2-D
                # row case so a list and its tuple spelling compare equal.
                return [_item(v) for v in values.tolist()]
            return [_item(v) for v in values.tolist()]
        if isinstance(values, (str, bytes)):
            return [values]
        try:
            return [_item(v) for v in values]
        except TypeError:
            return [values]

    ref = _categories(reference)
    hold = _categories(holdout)
    if max_categories is not None:
        max_categories = _check_count('max_categories', max_categories, positive=True)
        if max_categories < 2:
            raise ValueError(
                'max_categories must be at least 2: a single bin would collapse '
                'the reference distribution and report zero divergence while '
                'every holdout category is unseen')
    if not hold:
        return {'js_divergence': 0.0, 'total_variation': 0.0,
                'n_reference': len(ref), 'n_holdout': 0,
                'unseen_rate': 0.0}

    def _key(value):
        return ('__missing__',) if _category_is_missing(value) else _hashable_label(value)

    def _sort_key(value):
        return (type(value).__name__, repr(value))

    ref_keys = [_key(v) for v in ref]
    hold_keys = [_key(v) for v in hold]
    counts = {}
    for key in ref_keys:
        counts[key] = counts.get(key, 0) + 1
    if max_categories is not None:
        if len(counts) > max_categories:
            # Keep the most frequent reference categories plus an "other" bin.
            # Count ties are resolved by the stable identity key, not by NumPy's
            # unstable argsort of object values.
            ranked = sorted(counts, key=lambda k: (-counts[k], _sort_key(k)))
            keep = sorted(ranked[:max_categories - 1], key=_sort_key)
        else:
            keep = sorted(counts, key=_sort_key)
    else:
        keep = sorted(counts, key=_sort_key)
    keep_index = {key: i for i, key in enumerate(keep)}
    # `keep` may be TRUNCATED by max_categories, but a holdout value that was a
    # genuine reference category (just not frequent enough to get its own bin)
    # is not "unseen" — it is represented by the reference's other-bin.  Track
    # the full reference vocabulary separately from the displayed bins.
    reference_keys = set(counts)

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
    if reference_keys:
        unseen = float(np.mean([key not in reference_keys for key in hold_keys]))
    else:
        # No reference category exists (e.g. an empty training side): every
        # holdout token is unseen, not zero.
        unseen = 1.0 if hold_keys else 0.0
    return {
        'js_divergence': js,
        'total_variation': tv,
        'n_reference': int(len(ref)),
        'n_holdout': int(len(hold)),
        'unseen_rate': unseen,
    }
