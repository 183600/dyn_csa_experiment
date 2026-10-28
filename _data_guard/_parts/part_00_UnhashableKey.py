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
    'split_content_fingerprints',
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
    if simple == exact:
        return exact
    # Never round a positive fraction down to exactly zero: at a large row
    # count a 1e-16 split still contains rows (1e18 * 1e-16 = 100), and
    # ``limit_denominator`` cannot represent values below 1e-9 other than 0.
    # The tolerance is only meant to recover simple rationals such as 4/7,
    # not to erase a genuine positive target.
    if simple == 0:
        return exact
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
    _targets = (train_frac, val_frac, test_frac)
    _exact_targets = [_fraction_value(f) * int(n) for f in _targets]
    _remainder = n - sum(sizes)
    if _remainder < 0:
        # The fraction sum is allowed to be within 1e-6 of 1.  At large n
        # even a 1e-9 overshoot can make the three floored targets add up to
        # more than n (0.01 + 1e-9 + 0.99 does exactly that at n > 1e9).  A
        # valid split must still sum to n: return the excess while keeping
        # every requested split non-empty.  A zero-fraction split has zero
        # slack and is never touched.
        _excess = -_remainder
        # Remove the excess in BULK: the tolerated 1e-6 slack can be ~1e12
        # rows at a 1e18-row corpus, so a one-row-at-a-time loop would hang.
        # Every removed row adds exactly one to the L1 distance from the exact
        # target (the floors already sit below their targets), so the split of
        # the removal does not change that objective.  Distribute it in
        # proportion to each split's slack above its one-row floor, which
        # keeps every requested split non-empty and never touches an empty
        # split.  The leftover from integer division is at most 2 rows.
        _mins = [1 if requested[i] and n >= sum(requested) else 0
                 for i in range(3)]
        _slack = [max(0, int(sizes[i]) - _mins[i]) for i in range(3)]
        _total_slack = sum(_slack)
        if _excess > _total_slack or _total_slack <= 0:
            raise ValueError(
                f'could not reconcile split sizes {sizes} with n={n}')
        _remove = [(_excess * _slack[i]) // _total_slack
                   for i in range(3)]
        _left = _excess - sum(_remove)
        if _left:
            # Largest proportional remainder first; ties go to the later
            # split, matching the positive-remainder rule below.
            _order = sorted(range(3),
                            key=lambda i: ((_excess * _slack[i]) % _total_slack, i),
                            reverse=True)
            for i in _order:
                if not _left:
                    break
                _take = min(_slack[i] - _remove[i], _left)
                _remove[i] += _take
                _left -= _take
        sizes = [int(sizes[i]) - _remove[i] for i in range(3)]
        _remainder = 0
    if _remainder:
        # Rank by the exact fractional shortfall, not a rounded float
        # product: at large n a 1-ulp difference can decide which split
        # receives the remainder row.
        _best = max((i for i, flag in enumerate(requested) if flag),
                    key=lambda i: (_exact_targets[i] - int(sizes[i]), i))
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
        # 2**63, which is already outside the int64 range.  Check both exact
        # boundaries before the cast instead of relying on an
        # implementation-defined overflow (which also warns) for values far
        # outside the range.
        if np.any(f >= 2.0 ** 63) or np.any(f < -(2.0 ** 63)):
            raise ValueError(
                f'{name} contains an out-of-range row index '
                f'{float(f.min() if f.min() < -(2.0 ** 63) else f.max())!r}; '
                'indices must fit in int64')
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
                # Exact numeric NaNs (notably Decimal('sNaN')) raise inside
                # float() instead of returning NaN.  The old ``continue``
                # treated that as an ordinary orderable key, then the sort
                # raised decimal.InvalidOperation outside the intended
                # ValueError contract.  Detect the missing value first.
                if _numeric_is_nan(value):
                    raise ValueError(
                        f'{name} contains NaN/Inf; a temporal split needs a '
                        'well-defined order')
                try:
                    f = float(value)
                except (TypeError, ValueError, ArithmeticError, OverflowError):
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
    bounds = (0, c1, c2, n_groups)
    # Build int64 arrays directly instead of materialising N Python integers
    # and then sorting a second Python list.  `members[g]` is already in
    # ascending row order (stable argsort), so only the concatenation across
    # groups needs sorting; `np.sort` keeps the same final value order.
    res = {}
    for k, name in enumerate(SPLIT_NAMES):
        _parts = [np.asarray(members[g], dtype=np.int64)
                  for g in order_groups[bounds[k]:bounds[k + 1]]]
        res[name] = (np.sort(np.concatenate(_parts))
                     if _parts else np.empty(0, dtype=np.int64))
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
        if time is not None:
            # Accept a one-column time array exactly like the row-only path;
            # the grouped helper itself requires a 1-D ordering key.
            time = _as_1d(time, 'time', n)
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
    """Quantiles that avoid float64 overflow in interpolation.

    ``np.quantile`` computes interpolation as ``a + (b - a) * t``; for finite
    values near the float64 limit that subtraction can overflow and silently
    return a *wrong but finite* quantile.  When the magnitude is large enough
    for that to happen, compute on max-scaled values instead.  Ordinary data
    keeps the direct (bit-identical) path.
    """
    if values.size == 0:
        return (-math.inf, math.inf)
    _scaled, _m = _scaled_maxabs(values)
    if _scaled is not None and _m > 1e150:
        out = np.quantile(_scaled, qs) * _m
        out = tuple(float(v) for v in np.asarray(out).reshape(-1))
        if all(math.isfinite(v) for v in out):
            return out
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.quantile(values, qs)
    out = tuple(float(v) for v in np.asarray(out).reshape(-1))
    if all(math.isfinite(v) for v in out):
        return out
    if _scaled is None:
        return (-math.inf, math.inf)
    out = np.quantile(_scaled, qs) * _m
    out = tuple(float(v) for v in np.asarray(out).reshape(-1))
    if all(math.isfinite(v) for v in out):
        return out
    return (-math.inf, math.inf)


def _ks_statistic(reference, holdout):
    """Exact two-sample Kolmogorov-Smirnov distance between finite samples.

    The empirical-CDF difference can change value only at an observation of
    either sample, so evaluating it at both sorted samples is exact and avoids
    materialising the pooled union.  Returns ``None`` for an empty side (the
    statistic is undefined, not zero).
    """
    a = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(holdout, dtype=np.float64).reshape(-1))
    if a.size == 0 or b.size == 0:
        return None
    # At each reference value the reference CDF is its right rank; the holdout
    # CDF is the count <= that value.  The symmetric evaluation at holdout
    # values covers every possible extremum of the two step functions.
    fa_at_a = np.searchsorted(a, a, side='right') / float(a.size)
    fb_at_a = np.searchsorted(b, a, side='right') / float(b.size)
    fa_at_b = np.searchsorted(a, b, side='right') / float(a.size)
    fb_at_b = np.searchsorted(b, b, side='right') / float(b.size)
    return float(max(np.max(np.abs(fa_at_a - fb_at_a)),
                     np.max(np.abs(fa_at_b - fb_at_b))))


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
