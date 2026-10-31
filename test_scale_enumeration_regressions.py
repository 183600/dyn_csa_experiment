"""Scale-safety checks for full-factorial completeness enumeration.

`ablation_completeness_audit(arms_by_combo=...)` used to materialise every
subset of the declared components (2**k tuples) even when the declaration was
missing most cells.  A large or malformed artifact could therefore exhaust
memory during a zero-GPU audit.  The exact small-family behaviour must stay
unchanged; above the bound the audit must still report an exact completeness
verdict and missing count without claiming the unenumerated remainder.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _components(n):
    return [f'C{i}' for i in range(n)]


def test_large_full_factorial_is_bounded_and_truncation_is_explicit():
    comps = _components(20)
    arms = {(): 'none'}
    arms.update({(c,): 'only-' + c for c in comps})
    rep = L.ablation_completeness_audit(arms_by_combo=arms)
    assert rep['complete'] is False, rep
    assert rep['n_required'] == 1 << 20, rep
    assert rep['n_mapped'] == 21, rep['n_mapped']
    assert rep['missing_combos'] == [], rep['missing_combos']
    assert rep['missing_combos_truncated'] is True, rep
    assert rep['missing_combos_count'] == (1 << 20) - 21, rep
    json.dumps(rep, allow_nan=False)


def test_complete_large_factorial_needs_no_enumeration_and_stays_complete():
    comps = _components(17)  # 131072 cells: above the exact-enumeration bound
    arms = {}
    for mask in range(1 << len(comps)):
        combo = tuple(comps[i] for i in range(len(comps)) if mask >> i & 1)
        arms[combo] = 'arm-' + (','.join(combo) or 'none')
    rep = L.ablation_completeness_audit(arms_by_combo=arms)
    assert rep['complete'] is True, (rep['n_required'], rep['n_mapped'],
                                     rep['missing_combos_count'],
                                     rep['structural_errors'])
    assert rep['missing_combos_count'] == 0, rep
    assert rep['n_required'] == 1 << 17, rep


def test_small_factorial_keeps_the_exact_missing_cell_list():
    arms = {(): 'full', ('A',): 'full-A', ('B',): 'full-B'}
    rep = L.ablation_completeness_audit(arms_by_combo=arms)
    assert rep['complete'] is False, rep
    assert rep['missing_combos'] == [['A', 'B']], rep['missing_combos']
    assert rep['missing_combos_truncated'] is False, rep
    assert rep['missing_combos_count'] == 1, rep


def test_non_mapping_factorial_declaration_fails_closed():
    for bad in ([], 'A+B', 3):
        rep = L.ablation_completeness_audit(arms_by_combo=bad)
        assert rep['complete'] is False, (bad, rep)
        assert rep['structural_errors'], (bad, rep)
        json.dumps(rep, allow_nan=False)


def test_explicit_split_registry_artifact_list_is_bounded():
    import audit_gates as A
    meta = {}
    paths = A._split_registry_artifact_paths(
        [f'/tmp/nonexistent-registry-{i}.json' for i in range(600)],
        meta=meta)
    assert len(paths) == A._SPLIT_REGISTRY_MAX_FILES, len(paths)
    assert meta.get('files_truncated') is True, meta


def test_split_registry_record_cap_does_not_mark_an_exact_bound_as_truncated():
    import audit_gates as A

    def records(n):
        return [{'source_tree': {'split_registry': {'sha256': 'a' * 64}}}
                for _ in range(n)]

    meta = {}
    found = A._split_registry_find(records(200), meta=meta)
    assert len(found) == 200, len(found)
    assert meta.get('records_truncated') is None, meta

    meta = {}
    found = A._split_registry_find(records(201), meta=meta)
    assert len(found) == 200, len(found)
    assert meta.get('records_truncated') is True, meta


def test_unhashable_present_arm_collection_fails_closed():
    rep = L.ablation_completeness_audit(arms_by_combo={(): 'none'},
                                        present=[[]])
    assert rep['complete'] is False, rep
    assert rep['structural_errors'], rep
    json.dumps(rep, allow_nan=False)
