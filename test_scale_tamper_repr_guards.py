"""Scale/tamper guards for validation diagnostics.

A hand-edited artifact is untrusted JSON.  A finite Python integer can still
be far beyond CPython 3.12's integer-to-decimal digit limit, so formatting a
bad status/kind with ``repr`` raises ``ValueError`` while the validator is
trying to *report* the malformed field.  The launch/registration gates must
fail closed with a strict-JSON verdict instead of aborting.

These checks are CPU-only and read no run artifacts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


HUGE = 10 ** 10000


def _strict_json(value):
    json.dumps(value, allow_nan=False)
    return value


def test_analysis_rehearsal_tampered_scalars_fail_closed():
    out = _strict_json(L.validate_analysis_rehearsal({
        'kind': HUGE,
        'schema': HUGE,
        'status': HUGE,
        'n_failures': HUGE,
        'scenarios': {HUGE: {'ok': False}},
    }))
    assert out['status'] == 'fail', out
    assert any('rehearsal.kind' in p for p in out['problems']), out
    assert any('unrepresentable int' in p for p in out['problems']), out


def test_design_protocol_tampered_kind_fails_closed():
    out = _strict_json(L.validate_design_protocol({
        'kind': HUGE,
        'schema': HUGE,
        'study': {'level': 'draft', 'alpha': HUGE},
        'analysis': {'correction': HUGE},
        'data': {'test_access': {'method': HUGE}},
    }))
    assert out['status'] == 'fail', out
    assert any('protocol.kind' in p for p in out['problems']), out
    assert any('unrepresentable int' in p for p in out['problems']), out


def test_preparation_clearance_tampered_status_and_check_keys_fail_closed():
    out = _strict_json(L.validate_preparation_clearance({
        'kind': 'preparation_clearance',
        'schema': 1,
        'status': HUGE,
        'checks': {HUGE: {'status': HUGE, 'name': HUGE},
                   'protocol': {'status': HUGE, 'name': HUGE}},
        'required_checks': [],
    }))
    assert out['status'] == 'fail', out
    assert any('clearance.status' in p for p in out['problems']), out
    assert any('unrepresentable int' in p for p in out['problems']), out
    assert any('unknown checks' in p for p in out['problems']), out


def test_part_list_hash_is_explicitly_importable():
    # The hash function is a Makefile/API entry point; it must not depend on a
    # later fragment happening to import `json` into the module globals.
    value = L.part_list_hash()
    assert isinstance(value, str) and len(value) == 64, value
    int(value, 16)
