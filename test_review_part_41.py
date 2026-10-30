"""Regression checks for JSON-safe extended-precision and temporal scalars.

NumPy's ``longdouble``/``clongdouble`` scalars return another NumPy scalar
from ``.item()``.  Recursing on that call therefore never reaches a Python
type, so a perfectly finite ``np.longdouble`` value used to raise
``RecursionError`` inside the artifact writer.  ``np.timedelta64`` is also an
``np.integer`` subclass, so a dtype-first conversion can call ``int()`` on a
``datetime.timedelta`` and crash.  These checks cover the shared converter and
the provenance converter that previously returned raw NumPy scalars.
"""
import json
import math
import os
import sys as _sys
from decimal import Decimal

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_json_clean_handles_extended_precision_scalars():
    cleaned = L._json_clean(np.longdouble('1.5'))
    assert cleaned == 1.5 and isinstance(cleaned, float), repr(cleaned)
    assert L._json_clean(np.longdouble('nan')) is None
    assert L._json_clean(np.longdouble('inf')) is None
    assert L._json_clean(np.longdouble('-inf')) is None
    assert L._json_clean(np.clongdouble(1.5 + 2.5j)) == {
        'real': 1.5, 'imag': 2.5}
    # A whole array must be writable under a strict JSON writer.
    payload = {'values': np.array([np.longdouble('1.5'),
                                   np.longdouble('nan')])}
    text = json.dumps(L._json_clean(payload), allow_nan=False)
    assert json.loads(text) == {'values': [1.5, None]}


def test_json_clean_keeps_temporal_scalar_identity():
    # np.timedelta64 is an np.integer subclass; it must still go through its
    # temporal identity, not int()/float(), and the result must be finite.
    assert L._json_clean(np.timedelta64(2, 's')) == 2.0
    assert L._json_clean(np.timedelta64(5, 'D')) == 432000.0
    assert L._json_clean(np.timedelta64(5, 'ns')) == 5e-9
    assert L._json_clean(np.datetime64('2020-01-01')) == '2020-01-01'
    assert (L._json_clean(np.datetime64('2020-01-01', 'ns'))
            == '2020-01-01T00:00:00.000000000')
    assert (L._json_clean(np.array(['2020-01-01'], dtype='datetime64[ns]'))
            == ['2020-01-01T00:00:00.000000000'])
    assert (L._json_clean(np.array([1, 2], dtype='timedelta64[ns]'))
            == [1e-9, 2e-9])
    assert L._json_clean(np.datetime64('NaT', 's')) is None
    assert L._json_clean(np.timedelta64('NaT', 's')) is None


def test_json_clean_handles_bytes_complex_and_exact_numerics():
    assert L._json_clean(b'abc') == 'abc'
    assert L._json_clean(1 + 2j) == {'real': 1.0, 'imag': 2.0}
    assert L._json_clean(Decimal('1.5')) == 1.5
    assert L._json_clean(Decimal('NaN')) is None
    assert L._json_clean(Decimal('Infinity')) is None
    json.dumps(L._json_clean([b'x', 1 + 2j, Decimal('1.5')]),
               allow_nan=False)


def test_jsonable_provenance_uses_the_same_conversion():
    assert L._jsonable(np.longdouble('1.5')) == 1.5
    assert L._jsonable(np.clongdouble(1 + 2j)) == {'real': 1.0, 'imag': 2.0}
    assert L._jsonable(np.datetime64('2020-01-01')) == '2020-01-01'
    assert L._jsonable(np.timedelta64(2, 's')) == 2.0
    json.dumps(L._jsonable({'x': [np.longdouble('nan')]}), allow_nan=False)


def test_atomic_write_json_serialises_extended_precision_values():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'value.json')
        L.atomic_write_json(path, {'v': np.longdouble('1.5'),
                                   'arr': np.array([np.longdouble('inf')])})
        with open(path, encoding='utf-8') as fh:
            loaded = json.load(fh)
        assert loaded['v'] == 1.5
        assert loaded['arr'] == [None]
        assert math.isfinite(loaded['v'])


def test_dedup_protocol_fingerprint_accepts_numpy_scalars():
    assert (L.dedup_protocol_fp({'x': np.bool_(True)})
            == L.dedup_protocol_fp({'x': True}))
    assert (L.dedup_protocol_fp({'x': np.int64(3)})
            == L.dedup_protocol_fp({'x': 3}))
    assert (L.dedup_protocol_fp({'x': np.timedelta64(2, 's')})
            == L.dedup_protocol_fp({'x': 2.0}))
    assert (L.dedup_protocol_fp({'x': np.datetime64('2020-01-01')})
            == L.dedup_protocol_fp({'x': '2020-01-01'}))
    assert (L.dedup_protocol_fp({'x': b'1'})
            == L.dedup_protocol_fp({'x': '1'}))
    assert (L.dedup_protocol_fp({'x': np.array([1, 2])})
            == L.dedup_protocol_fp({'x': [1, 2]}))


def test_longdouble_nan_is_one_missing_row_identity_across_dtypes():
    import data_guard as DG

    # A longdouble NaN is the same missing-like value as a float64 NaN and an
    # object-column NaN, including when it is the whole row.
    native = DG._row_keys(np.array([[np.longdouble('nan')]],
                                   dtype=np.longdouble))[0]
    object_row = DG._row_keys(np.array([[np.longdouble('nan')]],
                                       dtype=object))[0]
    float_row = DG._row_keys(np.array([[np.nan]], dtype=np.float64))[0]
    assert native == object_row == float_row
    assert (DG._matrix_fingerprint(np.array([[np.longdouble('nan')]],
                                            dtype=np.longdouble))
            == DG._matrix_fingerprint(np.array([[np.longdouble('nan')]],
                                               dtype=object)))
    # Mixed exact + NaN rows keep the same wide/float split in both storages.
    mixed_native = np.array([[np.longdouble('1.5'),
                              np.longdouble('nan')]], dtype=np.longdouble)
    mixed_object = np.array([[np.longdouble('1.5'),
                              np.longdouble('nan')]], dtype=object)
    assert DG._row_keys(mixed_native)[0] == DG._row_keys(mixed_object)[0]
    assert (DG._matrix_fingerprint(mixed_native)
            == DG._matrix_fingerprint(mixed_object))
    # The exactness fix must not collapse distinct wide values.
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        one = np.longdouble('1.0000000000000000001')
        two = np.longdouble('1.0000000000000000002')
        wide = np.array([[one], [two]], dtype=np.longdouble)
        assert DG.duplicate_mask(wide).tolist() == [False, False]
        assert (DG._matrix_fingerprint(wide[:1])
                != DG._matrix_fingerprint(wide[1:]))

