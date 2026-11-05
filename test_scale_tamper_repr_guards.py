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


def test_factor_audits_tampered_keys_fail_closed():
    # The sweep audit was hardened first; the NIAH/probe readers use the same
    # untrusted declaration shape and must not abort while naming a bad key.
    niah = _strict_json(L.niah_contrast_factor_audit(
        {HUGE: {'factors': ['kind'], 'single_factor': False}}, verbose=False))
    assert niah['status'] == 'fail', niah
    assert any('unrepresentable int' in v for v in niah['violations']), niah

    probe = _strict_json(L.probe_contrast_factor_audit(
        {'x': {'factors': ['kind'], 'single_factor': False,
               'override': HUGE}},
        observed_tags=[HUGE], verbose=False))
    assert probe['status'] == 'fail', probe
    assert any('unrepresentable int' in v for v in probe['violations']), probe

    rec = L.attach_probe_factor_metadata({}, HUGE, {'pairs': {}})
    assert rec['factor_declaration_verified'] is False, rec
    assert 'unrepresentable int' in rec['factors_note'], rec


def test_power_planners_oversized_counts_fail_closed():
    planned = _strict_json(L.planned_power_audit({
        'seeds': [],
        'statistics': {'n_seeds': HUGE},
        'primary_comparisons': [],
    }))
    assert planned['status'] == 'fail', planned
    assert planned['n_seeds'] is None, planned
    assert planned['n_seeds_bit_length'] == HUGE.bit_length(), planned

    paired = _strict_json(L.paired_design_sensitivity(HUGE, 5))
    assert paired['status'] == 'fail', paired
    assert paired['n_seeds'] is None, paired

    attrition = _strict_json(L.attrition_power_audit(6, HUGE, q=0.5))
    assert attrition['status'] == 'fail', attrition
    assert attrition['m_family'] is None, attrition

    # Below the scan cap, an oversized optional count must still be reported
    # rather than formatted into an exception.
    optional = _strict_json(L.attrition_power_audit(
        6, 5, q=0.5, min_complete_pairs=HUGE))
    assert optional['status'] in ('ok', 'not_declared', 'fail'), optional


def test_exact_floors_survive_extreme_seed_counts_and_tiny_alpha():
    assert L.signflip_p_floor(HUGE) == 0.0
    assert L.signflip_p_floor(6) == 0.03125

    for alpha in (1e-320, 5e-324):
        out = _strict_json(L.maxt_min_n_for_alpha(alpha))
        assert out['min_n'] >= 2, out
        assert 0.0 <= out['p_floor'] <= alpha, out


def test_design_audits_tampered_component_keys_fail_closed():
    # Older design audits also accept hand-written mappings.  A huge integer
    # component name must produce a normal verdict (or a normal failure), not
    # a stringification error while the audit names the bad component.
    audit_calls = (
        ('ablation_completeness',
         lambda: L.ablation_completeness_audit(design={HUGE: {HUGE: HUGE}})),
        ('lesion_design', lambda: L.lesion_design_audit({HUGE: {}})),
        ('factorial_design', lambda: L.factorial_design_audit(
            design={HUGE: {'factors': ['kind'], 'single_factor': False}})),
        ('hypothesis_coverage', lambda: L.hypothesis_coverage_audit(
            design={HUGE: {'factors': ['kind'], 'single_factor': False}})),
        ('replication_independence',
         lambda: L.replication_independence_audit({HUGE: {}})),
        ('factorial_power', lambda: L.factorial_power_audit({HUGE: {}})),
    )
    for label, call in audit_calls:
        out = _strict_json(call())
        assert isinstance(out, dict), (label, out)
