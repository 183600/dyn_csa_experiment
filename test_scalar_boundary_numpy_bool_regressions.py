"""NumPy boolean flags must not enter numeric measurement boundaries.

``np.bool_`` is not a subclass of Python ``bool``.  A reader that only checks
``isinstance(value, bool)`` therefore accepts ``float(np.bool_(True)) == 1.0``:
a registered target power becomes 100% and a replay tolerance becomes 1.0.
The same readers must also refuse an integer too large for CPython's
decimal-conversion limit instead of failing later while hashing/writing JSON.

These are CPU-only, zero-experiment checks.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_preparation_clearance_scalar_readers_reject_numpy_bools():
    for flag in (np.bool_(True), np.bool_(False)):
        assert L._prep_number(flag) is None
        assert L._prep_int(flag) is None
    assert L._prep_number(0.8) == 0.8
    assert L._prep_int(np.int64(7)) == 7
    assert L._prep_int(np.float64(7.0)) == 7


def test_preparation_clearance_int_reader_refuses_unserialisable_int():
    assert L._prep_int(10 ** 10000) is None
    assert L._prep_int(10 ** 4000) == 10 ** 4000


def test_power_gate_cannot_use_a_numpy_boolean_as_target_power():
    protocol = {
        'study': {'level': 'confirmatory', 'alpha': 0.05},
        'sample_size': {
            'target_power': np.bool_(True), 'power_q': 0.9,
            'family_size': 1, 'planned_seeds': 3, 'min_n': 2,
        },
        'comparisons': [{'name': 'a', 'predicted_sign': 1}],
    }
    power = {
        'status': 'pass', 'all_pass': True, 'alpha': 0.05, 'm_family': 1,
        'power_q_for_test': 0.9, 'min_power_for_test': 0.8,
        'comparisons': {
            'a': {'power_ok': True, 'n': 3,
                  'power': {'power': {'q=0.90': 0.99}}},
        },
    }
    out = L._prep_power_check(protocol, power)
    assert out['target_power'] is None
    assert out['status'] == 'fail'
    assert any('comparable target power' in problem
               for problem in out['problems']), out['problems']


def test_reproduction_tolerance_rejects_numpy_bool():
    for flag in (np.bool_(True), np.bool_(False)):
        with pytest.raises(ValueError, match='finite non-negative'):
            L._repro_validate_tolerance('rtol', flag)
    assert L._repro_validate_tolerance('rtol', 0.25) == 0.25


def test_validation_transfer_replay_reports_huge_int_without_raising():
    block = {
        'kind': L.VALIDATION_TRANSFER_KIND,
        'schema': L.VALIDATION_TRANSFER_SCHEMA,
        'family_size': 10 ** 10000,
    }
    out = L.replay_validation_transfer(block)
    assert out['status'] == 'fail'
    assert any('family_size' in problem for problem in out['problems']), out
    json.dumps(out, allow_nan=False)


def test_budget_truncation_identifier_reader_survives_huge_int():
    problems = []
    assert L._bt_declared_list([10 ** 10000, 'arm'], 'seeds', problems) == ['arm']
    assert any('too large' in problem for problem in problems), problems
