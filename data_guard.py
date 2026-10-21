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

import datetime as _datetime
import decimal as _decimal
import hashlib
import json
import math
import numbers
from fractions import Fraction

import numpy as np

__all__ = [
    'SPLIT_NAMES',
    'LeakageSafeCategoricalEncoder',
    'LeakageSafeFrequencyEncoder',
    'LeakageSafePreprocessor',
    'LeakageSafeTargetEncoder',
    'SafeScaler',
    'assert_no_split_leakage',
    'audit_encoder_fit',
    'audit_preprocessor_fit',
    'categorical_shift_report',
    'chronological_split',
    'chronological_split_indices',
    'clean_training_rows',
    'data_quality_report',
    'duplicate_mask',
    'fit_categorical_encoder',
    'fit_preprocessor',
    'prepare_categorical_features',
    'prepare_frequency_features',
    'prepare_target_features',
    'grouped_chronological_split',
    'grouped_chronological_split_indices',
    'prepare_features',
    'split_and_prepare',
    'split_indices',
    'train_val_test_split',
]

SPLIT_NAMES = ('train', 'val', 'test')
_EPS = 1e-12


class _UnhashableKey:
    """Opaque, deterministic identity for an unhashable category/group.

    A plain tuple sentinel such as ``('__unhashable__', type, repr)`` can
    collide with a real hashable tuple category using exactly those elements,
    silently merging two different labels/groups.  An instance of this class
    only compares equal to another instance with the same type name and repr,
    so user values cannot impersonate the sentinel.  The repr is deterministic
    (no object address), which keeps fold assignment and audit output stable
    across processes.
    """
    __slots__ = ('type_name', 'repr')

    def __init__(self, type_name, repr_text):
        self.type_name = str(type_name)
        self.repr = str(repr_text)

    def __hash__(self):
        return hash((type(self), self.type_name, self.repr))

    def __eq__(self, other):
        return (type(other) is type(self)
                and self.type_name == other.type_name
                and self.repr == other.repr)

    def __repr__(self):
        return f'<_UnhashableKey {self.type_name}:{self.repr}>'


# One identity for every representation of a missing group id (None, NaN,
# NaT).  ``('__nan__',)`` is the historical public sentinel (tests and older
# artifacts inspect it); what matters for leakage is that all missing ids share
# it, so they can never act as per-row unique groups.
_MISSING_GROUP_KEY = ('__nan__',)


# ---------------------------------------------------------------------------
# Split sizing / validation
# ---------------------------------------------------------------------------

def _check_fraction(name, value):
    # Booleans are not fractions: float(True) == 1.0 would silently turn a
    # mistaken flag into a 100% split.  Same contract as _check_count.
    if isinstance(value, (bool, np.bool_)):
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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if isinstance(value, numbers.Integral):
        value = int(value)
    elif isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f) or f != math.floor(f):
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


def _fraction_value(frac):
    """Recover the intended simple fraction behind a declared float.

    ``4/7`` is stored as a binary approximation whose exact product with 7 is
    slightly below 4, so using ``Fraction(float(frac))`` would turn an
    intended 4-row split into 3.  Recover the small rational the caller almost
    certainly meant; fall back to the exact binary value when no simple
    rational round-trips.
    """
    exact = Fraction(float(frac))
    simple = exact.limit_denominator(10 ** 9)
    return simple if abs(float(simple) - float(exact)) <= 1e-15 else exact


def _floor_fraction_product(n, frac):
    """Exact ``floor(n * frac)`` for a validated float fraction.

    ``n * frac`` is evaluated in binary floating point, so at large row counts
    the product can round across an integer boundary and move a row between
    splits.  The product with the integer ``n`` is computed exactly before
    flooring.
    """
    return int(math.floor(Fraction(int(n)) * _fraction_value(frac)))


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
        fracs = (train_frac, val_frac, test_frac)
        unspecified = [i for i in range(3) if sizes[i] is None]
        remaining = n - sum(v for v in sizes if v is not None)
        for i in unspecified[:-1]:
            take = min(_floor_fraction_product(n, fracs[i]), remaining)
            sizes[i] = take
            remaining -= take
        if unspecified:
            _last = unspecified[-1]
            # A split the caller declared empty (frac=0) must stay empty even
            # when it is the last unspecified slot.  Give the remainder to the
            # last unspecified split that actually has a positive fraction;
            # only fall back to the empty slot when no such target exists.
            if fracs[_last] > 0.0 or not any(fracs[i] > 0.0 for i in unspecified):
                sizes[_last] = remaining
            else:
                _targets = [i for i in unspecified if fracs[i] > 0.0]
                sizes[_last] = 0
                if _targets:
                    sizes[_targets[-1]] += remaining
                else:
                    sizes[_last] = remaining
        if sum(sizes) != n or any(v < 0 for v in sizes):
            raise ValueError(f'could not reconcile split sizes {sizes} with n={n}')
        return tuple(int(v) for v in sizes)

    # Validate the fraction contract even for an empty table.  Returning
    # (0, 0, 0) before this check made n=0 accept contradictory arguments
    # that the same call rejects for n>0; a caller that later appends rows
    # would then be surprised by a different split.
    frac_sum = train_frac + val_frac + test_frac
    if frac_sum <= 0:
        raise ValueError('at least one split fraction must be positive')
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(
            f'train_frac + val_frac + test_frac must sum to 1, got {frac_sum}')
    if n == 0:
        # An empty dataset splits into three empty (non-overlapping) parts;
        # requiring one row per non-empty split would be impossible.
        return (0, 0, 0)
    sizes = [_floor_fraction_product(n, f)
             for f in (train_frac, val_frac, test_frac)]
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
    # Put the remainder where the declared fractions are still most under-
    # filled.  Assigning it unconditionally to the last requested split made
    # an 80/10/10 split at n=8 into (4,1,3): guaranteeing one validation/test
    # row had already taken rows from train, and the last split then took the
    # remainder as well.  Ties go to the later split (preserving the historical
    # zero-fraction contract), while a genuinely under-filled earlier split
    # still wins.  Splits the caller declared empty stay empty.
    _remainder = n - sum(sizes)
    if _remainder < 0:
        raise ValueError(f'inconsistent split sizes {sizes} for n={n}')
    if _remainder:
        _targets = (train_frac, val_frac, test_frac)
        # Rank by the exact fractional shortfall, not a rounded float
        # product: at large n a 1-ulp difference can decide which split
        # receives the remainder row.
        _best = max((i for i, flag in enumerate(requested) if flag),
                    key=lambda i: (_fraction_value(_targets[i]) * int(n)
                                   - int(sizes[i]), i))
        sizes[_best] += _remainder
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
        # An unsigned index can hold values above int64's maximum.  Casting
        # those would wrap to a negative int64, so the overlap/negativity
        # checks below would reject a valid-looking split with a misleading
        # "negative index" message (or, for some widths, silently alias).  Fail
        # on the actual out-of-range value first.
        if kind == 'u' and flat.size:
            if int(flat.max()) > int(np.iinfo(np.int64).max):
                raise ValueError(
                    f'{name} contains an out-of-range row index '
                    f'{int(flat.max())}; indices must fit in int64')
        out = flat.astype(np.int64, copy=False)
    elif kind == 'f':
        f = flat.astype(np.float64, copy=False)
        if not np.all(np.isfinite(f)):
            raise ValueError(f'{name} contains NaN/Inf; split indices must be finite')
        if np.any(f != np.floor(f)):
            raise ValueError(f'{name} contains non-integer row indices')
        # float64 cannot represent int64_max exactly; its nearest float is
        # 2**63, which is already outside the int64 range.  Check the exact
        # boundary before the cast instead of relying on the wrapped result.
        if np.any(f >= 2.0 ** 63):
            raise ValueError(
                f'{name} contains an out-of-range row index '
                f'{float(f.max())!r}; indices must fit in int64')
        out = f.astype(np.int64)
    elif kind == 'O':
        # Reject strings, datetimes and booleans instead of coercing them:
        # a timestamp column passed by mistake must not become row indices.
        vals = []
        for v in flat.tolist():
            if isinstance(v, (bool, np.bool_)):
                raise ValueError(
                    f'{name} must contain integer row indices, not booleans')
            if isinstance(v, numbers.Integral):
                v = int(v)
            elif isinstance(v, numbers.Real):
                f = float(v)
                if not math.isfinite(f) or f != math.floor(f):
                    raise ValueError(
                        f'{name} must contain finite integer row indices, '
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
    if isinstance(purge, (bool, np.bool_)):
        raise ValueError(f'purge must be a non-negative integer, got {purge!r}')
    if isinstance(purge, numbers.Integral):
        purge = int(purge)
    elif isinstance(purge, numbers.Real):
        f = float(purge)
        if not math.isfinite(f) or f != math.floor(f):
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
                if value is None:
                    raise ValueError(
                        f'{name} contains a missing value; a temporal split '
                        'needs a well-defined order')
                if isinstance(value, (np.datetime64, np.timedelta64)):
                    if np.isnat(value):
                        raise ValueError(
                            f'{name} contains NaT; a temporal split needs a '
                            'well-defined order')
                    continue
                if isinstance(value, (str, bytes)):
                    # Strings are ordering keys, not numbers.  Casting the
                    # literal string 'nan'/'inf' through float() would reject
                    # a perfectly orderable event id; leave the caller's
                    # lexicographic order alone.
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
    # Iterate the native scalars for temporal dtypes: ``tolist()`` turns a
    # datetime64[ns] into a bare nanosecond integer and would bypass the
    # canonical temporal identity used by the per-row path.
    keys = ([_group_key(v) for v in uniq]
            if groups.dtype.kind in 'Mm'
            else [_group_key(v) for v in uniq.tolist()])
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
    for i, g in enumerate(groups):
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
        for i, g in enumerate(groups):
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
    value, e.g. a timestamp).  The two split boundaries are placed as close as
    possible to the requested row targets while keeping every whole group on
    exactly one side.  Only whole groups are moved across a boundary, so a
    large group may shift some rows between adjacent splits; it is never cut.
    A split whose target is exactly zero stays empty (the remainder is absorbed
    by the last requested non-empty split).

    This boundary placement is what makes the helper complete: the historical
    greedy "consume groups until this split is full, then move on" could put a
    large group in validation, leave test empty and raise even though a
    feasible whole-group partition existed (e.g. groups of 1,1,2 rows with
    targets 2/1/1).  Whole-group isolation is still verified after allocation.
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
        # Iterate the native scalars, not ``tolist()``: a native
        # datetime64[ns] group column becomes bare integers through
        # ``tolist()``, which would not match the canonical temporal keys
        # produced by ``_group_order`` (and would raise KeyError below).
        for i, g in enumerate(groups):
            members.setdefault(_group_key(g), []).append(i)
    n_groups = len(order_groups)
    t_train, t_val, t_test = (int(v) for v in targets)
    # Prefix row counts let us place a boundary at any group edge in O(log G)
    # while preserving the chronological group order.  Every group has at
    # least one row, so the prefix is strictly increasing.
    sizes = [len(members[g]) for g in order_groups]
    prefix = np.cumsum(np.asarray(sizes, dtype=np.int64))
    prefix = np.concatenate((np.zeros(1, dtype=np.int64), prefix))

    def _nearest(target, lo, hi):
        """Group-edge index in [lo, hi] whose row count is closest to target."""
        pos = int(np.searchsorted(prefix, int(target), side='left'))
        cands = [p for p in (pos - 1, pos) if lo <= p <= hi]
        if not cands:
            return lo if int(target) <= int(prefix[lo]) else hi
        # Ties go to the earlier (more conservative) boundary, matching the
        # historical first-crossing behaviour for exact row targets.
        return min(cands, key=lambda p: (abs(int(prefix[p]) - int(target)), p))

    def _no_rows_error():
        _positive = [name for name, target in zip(SPLIT_NAMES, targets)
                     if int(target) > 0]
        if n_groups < len(_positive):
            _missing = _positive[n_groups]
        else:
            # No count shortfall: the requested boundaries themselves are
            # impossible (e.g. a zero-sized side next to an oversized group).
            _missing = _positive[-1] if _positive else SPLIT_NAMES[0]
        return ValueError(
            f'grouped split has no rows for the requested {_missing} split '
            f'(targets={tuple(targets)}, groups={n_groups}, rows={n}); the '
            'group sizes cannot satisfy the requested fractions — change the '
            'split targets or merge groups')

    if n_groups == 0:
        if t_train or t_val or t_test:
            raise _no_rows_error()
        c1 = c2 = 0
    else:
        # First boundary.  A zero-target split must stay empty; positive splits
        # must each retain at least one whole group for the following sides.
        if t_train == 0:
            c1 = 0
        elif t_val == 0 and t_test == 0:
            c1 = n_groups
        else:
            lo = 1
            hi = n_groups - (1 if t_val > 0 else 0) - (1 if t_test > 0 else 0)
            if lo > hi:
                raise _no_rows_error()
            c1 = _nearest(t_train, lo, hi)
        # Second boundary.  c2 == c1 means an empty validation block; c2 ==
        # n_groups means an empty test block.
        if t_test == 0:
            c2 = n_groups
        elif t_val == 0:
            c2 = c1
        else:
            lo = c1 + 1
            hi = n_groups - 1
            if lo > hi:
                raise _no_rows_error()
            c2 = _nearest(int(prefix[c1]) + t_val, lo, hi)
        if not (0 <= c1 <= c2 <= n_groups):
            raise ValueError(
                f'grouped split had inconsistent boundaries {c1}/{c2} for '
                f'{n_groups} groups')
    out = {name: [] for name in SPLIT_NAMES}
    bounds = (0, c1, c2, n_groups)
    for k, name in enumerate(SPLIT_NAMES):
        for g in order_groups[bounds[k]:bounds[k + 1]]:
            out[name].extend(int(i) for i in members[g])
    res = {name: np.asarray(sorted(v), dtype=np.int64) for name, v in out.items()}
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
        cuts = np.asarray(cuts, dtype=np.int64)
        blocks = []
        start = 0
        start_idx = 0
        cumulative = 0
        for i, size in enumerate(sizes):
            size = int(size)
            cumulative += size
            if size <= 0:
                # A zero-sized split consumes no timestamp block.  Advancing
                # here would steal rows from a later requested split.
                blocks.append(order[start:start].copy())
                continue
            # Reserve at least one cut for every later positive split.  The
            # last positive split must still be able to end at n.
            rem_after = sum(1 for _s in sizes[i + 1:] if int(_s) > 0)
            max_idx = len(cuts) - 1 - rem_after
            pos = int(np.searchsorted(cuts, cumulative, side='left'))
            pos = max(pos, start_idx + 1)
            if pos > max_idx:
                # The target snaps past the last legal cut (typically to n).
                # Fall back to the latest cut that still leaves a block for
                # every remaining positive split instead of consuming all
                # rows and then failing the final split.
                pos = max_idx
            if pos <= start_idx or pos > max_idx or pos >= len(cuts):
                raise ValueError(
                    'cannot form the requested non-empty chronological split '
                    'without cutting an equal-timestamp block; supply groups '
                    '(grouped_chronological_split_indices) or fewer split rows')
            end = int(cuts[pos])
            # Preserve the chronological order: the caller indexes its data
            # with these arrays, so sorting the row indices would silently
            # return each split in original row order (e.g. val times
            # [7, 6]) instead of time order.
            blocks.append(order[start:end].copy())
            start = end
            start_idx = pos
        return _purge_tail(dict(zip(SPLIT_NAMES, blocks)), purge)
    return chronological_split_indices(
        n, train_frac, val_frac, test_frac,
        train_size=train_size, val_size=val_size, test_size=test_size,
        purge=purge)


def _temporal_group_key(value):
    """Canonical, equality-consistent identity for a temporal group id.

    NumPy and the stdlib do not agree on hashing: a ``datetime.date`` compares
    equal to some ``np.datetime64`` values but hashes differently.  Returning
    the raw objects as dict/set keys then let the same instant appear as two
    groups, so a document/user id could cross a split without the overlap
    check firing.  Canonicalising through the same nanosecond instant (falling
    back to a unit-qualified raw value when nanoseconds overflow) makes equal
    temporal ids hash equally.  Midnight values collapse with their date,
    which can only merge groups (a conservative, leak-avoiding direction).
    Timezone-aware datetimes are canonicalised to UTC, so the same instant
    written with different offsets is still one group; they stay distinct from
    a naive datetime with the same wall-clock text (which Python also does not
    compare equal to it).
    """
    if isinstance(value, (np.datetime64, np.timedelta64)):
        if np.isnat(value):
            return _MISSING_GROUP_KEY
    elif not isinstance(value, (_datetime.date, _datetime.datetime,
                                _datetime.timedelta)):
        # pragma: no cover - guarded by the caller
        return value
    try:
        # The opaque sentinel cannot be impersonated by a real tuple/list group
        # id whose elements happen to spell the same components.
        return _UnhashableKey(
            '__temporal__',
            _temporal_canonical_key(value).decode('ascii', 'backslashreplace'))
    except (TypeError, ValueError, OverflowError):
        return _UnhashableKey(
            '__temporal_repr__', f'{type(value).__name__}:{value!r}')


def _group_key(value):
    """Hashable group identity; all missing ids collapse into one group.

    A missing group id must not behave like "unique per row": otherwise rows
    of the same missing group can be split across train/val/test and no
    group-overlap check will ever fire.  ``None``, NaN and NaT share the
    historical ``('__nan__',)`` missing identity.
    """
    if value is None:
        return _MISSING_GROUP_KEY
    if isinstance(value, (np.datetime64, np.timedelta64,
                          _datetime.date, _datetime.datetime,
                          _datetime.timedelta)):
        return _temporal_group_key(value)
    if _numeric_is_nan(value):
        return _MISSING_GROUP_KEY
    try:
        hash(value)
    except TypeError:
        # Group ids are normally hashable scalars, but an object column of
        # lists/arrays should still be groupable by a stable identity rather
        # than crashing inside a set/dict downstream.  The opaque key cannot
        # equal a real user tuple.
        return _UnhashableKey(type(value).__name__, repr(value))
    return value


def _assert_index_and_group_disjoint(split_idx, groups=None):
    # Row-index disjointness is a pure set operation; intersecting the int64
    # arrays avoids materialising every index as a Python int (the set form
    # costs ~28 bytes/row and dominates memory on a scale-up split).
    arrays = {name: np.asarray(split_idx[name]).reshape(-1)
              for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            if np.intersect1d(arrays[a], arrays[b], assume_unique=False).size:
                raise ValueError(f'{a} and {b} indices overlap')
    if groups is not None:
        groups = np.asarray(groups)
        # Do not materialise a Python set of every row index just to map rows
        # to group ids: on a large corpus that is one pointer object per row
        # for each split.  Iterating the int64 index array directly keeps the
        # same group sets at a fraction of the memory.
        gsets = {
            name: {_group_key(groups[int(i)]) for i in arrays[name]}
            for name in SPLIT_NAMES
        }
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


def _scaled_maxabs(values):
    """Values divided by their largest magnitude, plus that magnitude.

    Scaling by the maximum absolute value keeps every intermediate <= 1, so a
    sum/quantile cannot overflow even when the original finite values are near
    float64's limit.  Returns ``(None, 0.0)`` for an empty/all-zero column.
    """
    if values.size == 0:
        return None, 0.0
    m = float(np.max(np.abs(values)))
    if not math.isfinite(m) or m == 0.0:
        return None, 0.0
    return values / m, m


def _safe_mean(values):
    """Float64 mean that falls back to a max-scaled sum on overflow."""
    if values.size == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.mean(values))
    if math.isfinite(value):
        return value
    scaled, m = _scaled_maxabs(values)
    if scaled is None:
        return 0.0
    value = float(np.mean(scaled)) * m
    return value if math.isfinite(value) else 0.0


def _safe_std(values):
    """Population std that falls back to a max-scaled computation."""
    if values.size == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.std(values, ddof=0))
    if math.isfinite(value):
        return value
    scaled, m = _scaled_maxabs(values)
    if scaled is None:
        return 0.0
    value = float(np.std(scaled, ddof=0)) * m
    return value if math.isfinite(value) else 0.0


def _safe_median(values):
    """Median without the ``(a + b) / 2`` overflow of NumPy's even case."""
    if values.size == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.median(values))
    if math.isfinite(value):
        return value
    scaled, m = _scaled_maxabs(values)
    if scaled is None:
        return 0.0
    value = float(np.median(scaled)) * m
    return value if math.isfinite(value) else 0.0


def _safe_quantiles(values, qs):
    """Quantiles that fall back to max-scaled values on overflow."""
    if values.size == 0:
        return (-math.inf, math.inf)
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.quantile(values, qs)
    out = tuple(float(v) for v in np.asarray(out).reshape(-1))
    if all(math.isfinite(v) for v in out):
        return out
    scaled, m = _scaled_maxabs(values)
    if scaled is None:
        return (-math.inf, math.inf)
    out = np.quantile(scaled, qs) * m
    out = tuple(float(v) for v in np.asarray(out).reshape(-1))
    if all(math.isfinite(v) for v in out):
        return out
    return (-math.inf, math.inf)


def _persisted_features(stats, *, what):
    """Validate a persisted ``features`` table against ``n_features_in``.

    ``transform`` loops over the fitted feature stats and writes one output
    column per entry.  A truncated/edited stats dict whose list is shorter
    than ``n_features_in`` would leave output columns uninitialised (leaking
    whatever memory NumPy returned), and a longer list would write past the
    declared width.  A persisted artifact must fail closed instead.
    """
    if not isinstance(stats, dict):
        raise ValueError(f'{what}.from_stats needs a mapping, got {type(stats).__name__}')
    features = stats.get('features')
    if not isinstance(features, list):
        raise ValueError(f'{what}.from_stats needs a features list')
    n_features = _check_count('n_features_in', stats.get('n_features_in'))
    if len(features) != n_features:
        raise ValueError(
            f'{what}.from_stats: {len(features)} feature statistics do not '
            f'match n_features_in={n_features}; the persisted stats are '
            'truncated or describe a different table')
    for j, st in enumerate(features):
        if not isinstance(st, dict):
            raise ValueError(
                f'{what}.from_stats: feature {j} statistics are not a mapping')
    return features, n_features


def _stat_float(st, key, *, what, index, allow_none=False):
    """Read one finite float statistic from a persisted feature entry.

    A missing/NaN/Inf bound would silently disable clipping, or poison every
    transformed value.  Persisted artifacts are untrusted input, so they fail
    closed instead of changing the fitted transform.
    """
    if key not in st:
        if allow_none:
            return None
        raise ValueError(
            f'{what}.from_stats: feature {index} has no {key!r} statistic')
    value = st[key]
    if value is None:
        if allow_none:
            return None
        raise ValueError(
            f'{what}.from_stats: feature {index} has a null {key!r} statistic')
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f'{what}.from_stats: feature {index} has a boolean {key!r} '
            'statistic')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{what}.from_stats: feature {index} has a non-numeric {key!r} '
            'statistic') from None
    if not math.isfinite(value):
        raise ValueError(
            f'{what}.from_stats: feature {index} has a non-finite {key!r} '
            'statistic')
    return value


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
            if scale == 'none':
                raise ValueError(
                    "clip_z needs a scaled feature space; scale='none' "
                    'has no standardized/minmax units to clip in')
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
                        center = float(np.min(scaled)) * m
                        factor = m
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
    return h.hexdigest()[:16]


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
    return h.hexdigest()[:16]


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
            if keys is None:
                keys = []
            else:
                keys = list(keys)
            if keys:
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
    if isinstance(value, numbers.Integral):
        return 'i:' + str(int(value))
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        if c.imag == 0.0:
            return _target_key(c.real)
        return 'c:' + repr(c)
    if isinstance(value, (float, np.floating)):
        f = float(value)
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
            if value is None or isinstance(value, (bool, np.bool_)):
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
            prior = float(np.mean(values[finite]))
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
        self.prior_ = float(np.mean(yarr[np.isfinite(yarr)]))
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
            folds = self._row_folds(n, keys=_row_keys(Xarr))
            policy = 'row_oof'
        out = np.zeros((n, Xarr.shape[1]), dtype=np.float64)
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
                if not isinstance(key, str):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} has a non-string '
                        'lookup key')
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
                if key is not None and not isinstance(key, str):
                    raise ValueError(
                        f'{what}.from_stats: feature {j} entry {pos} has a '
                        'non-string key')
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
            rebuilt.append({'entries': clean_entries,
                            'n_observed': n_observed,
                            'n_missing': n_missing})
        obj.feature_stats_ = rebuilt
        obj.n_features_in_ = int(n_features)
        obj.n_train_ = _check_count('n_train', stats.get('n_train', 0))
        obj.train_fp_ = stats.get('train_fp')
        obj.fitted_ = True
        return obj


def prepare_frequency_features(X_train, X_val=None, X_test=None, **kwargs):
    """Fit frequency tables on train, then transform the holdouts."""
    encoder = LeakageSafeFrequencyEncoder(**kwargs).fit(X_train)
    out = {'train': encoder.transform(X_train), 'encoder': encoder}
    if X_val is not None:
        out['val'] = encoder.transform(X_val)
    if X_test is not None:
        out['test'] = encoder.transform(X_test)
    return out


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
        bad = []
        for j, (got, want) in enumerate(zip(stats.get('features') or [],
                                            ref.get('features') or [])):
            keys = ('fill', 'clip_lo', 'clip_hi', 'center', 'factor',
                    'mean', 'std', 'median', 'min', 'max')
            if not all(_stats_match_float(got.get(k), want.get(k),
                                          atol=atol, rtol=rtol) for k in keys):
                bad.append(j)
        return (not bad), bad

    train_ref = _ref_stats(Xtr)
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
    parts = [_as_float_2d(part, name='X_holdout') for part in parts]
    parts = [p for p in parts if p.shape[0] > 0]
    if not parts:
        return out
    if any(p.shape[1] != Xtr.shape[1] for p in parts):
        raise ValueError(
            'audit_preprocessor_fit: holdout features do not match X_train')
    Xh = np.vstack(parts)
    full_ref = _ref_stats(np.vstack([Xtr, Xh]))
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
        return lab is None or (isinstance(lab, tuple)
                               and all(v is None for v in lab))

    drop = np.zeros(train_rows.size, dtype=bool)
    n_invalid_label_rows = 0
    if drop_invalid_labels and labels is not None:
        for pos, row in enumerate(train_rows.tolist()):
            lab = labels[int(row)]
            if lab is None or (isinstance(lab, tuple)
                               and any(v is None for v in lab)):
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
            q1, q3 = np.quantile(vals, (0.25, 0.75))
            iqr = float(q3 - q1)
            if iqr <= _EPS:
                lo, hi = float(np.min(vals)), float(np.max(vals))
            else:
                lo = float(q1 - outlier_iqr * iqr)
                hi = float(q3 + outlier_iqr * iqr)
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
            # target has no place on a single target fence.
            if (value is None
                    or isinstance(value, (bool, np.bool_))
                    or isinstance(value, (complex, np.complexfloating))
                    or not isinstance(value, numbers.Number)):
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
            q1, q3 = np.quantile(_ref, (0.25, 0.75))
            iqr = float(q3 - q1)
            if iqr <= _EPS:
                lo, hi = float(min(_ref)), float(max(_ref))
            else:
                lo = float(q1 - target_outlier_iqr * iqr)
                hi = float(q3 + target_outlier_iqr * iqr)
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
    keys = _row_keys(_raw_X)
    clusters = {}
    for pos, row in enumerate(train_rows.tolist()):
        if drop[pos]:
            continue
        clusters.setdefault(keys[row], []).append(pos)
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
                if v is None:
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
                    h.update(b'|R')
                    h.update(str(int(pos)).encode('ascii'))
                    h.update(b'|')
                    h.update(key)
        return h.hexdigest()[:16]
    _w = int(arr.shape[1]) if arr.ndim > 1 else 1
    _step = _hash_row_block(_w, 1 << 20)
    _is_int = (np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_
               or np.issubdtype(arr.dtype, np.datetime64)
               or np.issubdtype(arr.dtype, np.timedelta64))
    for start in range(0, arr.shape[0], _step):
        block = arr[start:start + _step]
        with np.errstate(invalid='ignore', over='ignore'):
            block_f = np.asarray(block, dtype=np.float64)
        if np.any(block_f == 0.0):
            block_f = block_f.copy()
            block_f[block_f == 0.0] = 0.0
        h.update(np.ascontiguousarray(block_f).tobytes())
        if arr.dtype.kind == 'f' and arr.dtype.itemsize > 8:
            # Distinct longdouble values can share one float64 image; append
            # the exact per-row keys so the fingerprint cannot collide.
            for key in _row_keys(block):
                h.update(b'|')
                h.update(key)
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
            _safe = _raw <= np.iinfo(np.int64).max
            _back = np.zeros(_raw.shape, dtype=np.int64)
            if _safe.any():
                _back[_safe] = block_f[_safe].astype(np.int64)
            _exact = _safe & (_back == _raw)
        else:
            _back = block_f.astype(np.int64)
            _exact = _back == _raw
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
                h.update(b'|R')
                h.update(str(int(pos)).encode('ascii'))
                h.update(b'|')
                h.update(key)
    return h.hexdigest()[:16]

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
                h = hashlib.blake2b(digest_size=8)
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
        _flat = src.reshape(-1).tolist()

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

        if any(isinstance(v, (complex, np.complexfloating)) for v in _flat):
            raise TypeError(
                'duplicate detection needs real-valued rows; casting complex '
                'rows to float64 would silently discard the imaginary part')

        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        for start in range(0, src.shape[0], _step):
            for row in src[start:start + _step].tolist():
                h = hashlib.blake2b(digest_size=8)
                for v in row:
                    h.update(_real_key(v))
                    h.update(b'|')
                keys.append(h.digest())
        return keys
    # Integer/bool/datetime/timedelta rows are hashed in their own dtype.
    # Converting them to float64 first collapses distinct int64 values above
    # 2**53 (and nanosecond timestamps) into false duplicates.
    if (np.issubdtype(src.dtype, np.integer) or src.dtype == np.bool_
            or np.issubdtype(src.dtype, np.datetime64)
            or np.issubdtype(src.dtype, np.timedelta64)):
        _step = _hash_row_block(int(src.shape[1]), 1 << 16)
        keys = []
        for start in range(0, src.shape[0], _step):
            arr = np.ascontiguousarray(src[start:start + _step])
            for row_view in arr.view(np.uint8).reshape(arr.shape[0], -1):
                h = hashlib.blake2b(digest_size=8)
                h.update(memoryview(row_view))
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
    if v is None:
        return None
    # NaT is a missing timestamp label, not a category: treating it as an
    # ordinary value would hide it from the missing/non-finite label audit.
    if isinstance(v, (np.datetime64, np.timedelta64)):
        return None if np.isnat(v) else v
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
        by_row = {}
        for i, (k, lab) in enumerate(zip(keys, labels)):
            if lab is not None:
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
                if not isinstance(value, numbers.Number):
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
                q1, q3 = np.quantile(_train_t, (0.25, 0.75))
                iqr = float(q3 - q1)
                if iqr <= _EPS:
                    lo, hi = float(min(_train_t)), float(max(_train_t))
                else:
                    lo = float(q1 - target_outlier_z * iqr)
                    hi = float(q3 + target_outlier_z * iqr)
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
    ref = np.asarray(reference, dtype=object).reshape(-1)
    hold = np.asarray(holdout, dtype=object).reshape(-1)
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
