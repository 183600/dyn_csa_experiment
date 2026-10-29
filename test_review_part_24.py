"""Reproduction identity edge cases that would otherwise fail open.

The comparator added in review 23 is a gate: a malformed count, a boolean
configuration value, a one-sided data fingerprint or a corrupt cell must not
be normalised into "absent" and silently produce a green reproduced verdict.
These checks only use synthetic summaries (no training, GPU or test split).
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403
from test_review_part_23 import _record, _summary

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_token_denominator_must_be_an_exact_count_on_both_sides():
    a = _record()
    for bad in (None, 0, 1000.9, True, -5, '1000.5', float('nan')):
        b = _record()
        b['n_tokens'] = bad
        d = L.reproduction_diagnosis(a, b)
        assert d['verdict'] == 'metric_mismatch' and d['reproduced'] is False, (bad, d)
        assert d['metric']['status'] == 'denominator_unverified', (bad, d['metric'])
        assert d['metric']['n_tokens']['rerun'] is None, (bad, d['metric'])
    # Both sides missing is also unverifiable, not equal by default.
    a2, b2 = _record(), _record()
    a2.pop('n_tokens')
    b2.pop('n_tokens')
    both = L.reproduction_diagnosis(a2, b2)
    assert both['verdict'] == 'metric_mismatch', both
    assert both['metric']['status'] == 'denominator_unverified', both['metric']
    # An integral-valued float is still an exact count and stays comparable.
    b3 = _record()
    b3['n_tokens'] = 1000.0
    assert L.reproduction_diagnosis(a, b3)['verdict'] == 'identical'


def test_boolean_config_values_are_not_treated_as_absent():
    a, b = _record(), _record()
    a['eager_batch'] = False
    b['eager_batch'] = True
    d = L.reproduction_diagnosis(a, b)
    assert d['verdict'] == 'config_changed' and d['reproduced'] is False, d
    assert d['dimensions']['config']['status'] == 'mismatch', d['dimensions']['config']
    field = [f for f in d['dimensions']['config']['fields']
             if f['field'] == 'eager_batch'][0]
    assert field['status'] == 'mismatch', field
    # Same boolean on both sides remains a match.
    b2 = _record()
    b2['eager_batch'] = False
    assert L.reproduction_diagnosis(a, b2)['verdict'] == 'identical'
    # A value present on one side only is unverified, never an absent match.
    b3 = _record()
    b3['eager_batch'] = None
    one = L.reproduction_diagnosis(a, b3)
    assert one['verdict'] == 'provenance_incomplete', one
    assert one['dimensions']['config']['status'] == 'unverified', one['dimensions']['config']


def test_one_sided_or_malformed_data_fingerprints_are_unverified():
    a, b = _record(), _record()
    b['data_fp'] = None
    b['provenance']['data']['data_fp'] = None
    d = L.reproduction_diagnosis(a, b)
    assert d['verdict'] == 'provenance_incomplete' and d['reproduced'] is False, d
    assert d['dimensions']['data']['status'] == 'unverified', d['dimensions']['data']
    field = [f for f in d['dimensions']['data']['fields']
             if f['field'] == 'data_fp'][0]
    assert field['status'] == 'unverified' and field['original'] == 'data', field
    malformed = _record()
    malformed['val_fp'] = ['not', 'a', 'fingerprint']
    bad = L.reproduction_diagnosis(a, malformed)
    assert bad['verdict'] == 'provenance_incomplete', bad
    assert bad['dimensions']['data']['status'] == 'unverified', bad['dimensions']['data']


def test_malformed_cells_cannot_disappear_from_the_comparison():
    a = _summary('csa_dynamic::seed0', 'csa_dynamic::seed1')
    b = copy.deepcopy(a)
    b['bad::seed0'] = 'not a record'
    rep = L.compare_reproductions(a, b)
    assert rep['verdict'] == 'incomplete' and rep['reproduced'] is False, rep
    assert rep['n_compared'] == 2 and rep['n_only_rerun'] == 0, rep
    assert rep['n_malformed_rerun'] == 1 and rep['malformed_rerun'] == ['bad::seed0'], rep
    assert any('not an object record' in p for p in rep['problems']), rep['problems']

    # The same corrupt key on both sides is still not a comparison of it.
    a2, b2 = copy.deepcopy(a), copy.deepcopy(a)
    a2['bad::seed0'] = None
    b2['bad::seed0'] = None
    rep2 = L.compare_reproductions(a2, b2)
    assert rep2['verdict'] == 'incomplete' and rep2['reproduced'] is False, rep2
    assert rep2['n_malformed_original'] == 1, rep2
    assert rep2['n_malformed_rerun'] == 1, rep2

    # Non-cell aggregate entries are still allowed to be present.
    b3 = copy.deepcopy(a)
    b3['aggregate'] = {'note': 'metadata, not a cell'}
    assert L.compare_reproductions(a, b3)['verdict'] == 'reproduced'


def test_byte_paths_point_at_the_same_summary_file():
    a = _summary('csa_dynamic::seed0')
    with tempfile.TemporaryDirectory() as td:
        for name in ('a', 'b'):
            os.makedirs(os.path.join(td, name))
            with open(os.path.join(td, name, 'summary.json'), 'w',
                      encoding='utf-8') as fh:
                json.dump(a, fh)
        rep = L.compare_reproductions(os.fsencode(os.path.join(td, 'a')),
                                      os.fsencode(os.path.join(td, 'b')))
        assert rep['verdict'] == 'reproduced', rep


def test_non_finite_or_malformed_descriptors_are_unverified_not_equal():
    a = _record()
    nan = _record()
    nan['exec_fp'] = float('nan')
    d = L.reproduction_diagnosis(a, nan)
    assert d['verdict'] == 'provenance_incomplete' and d['reproduced'] is False, d
    assert d['dimensions']['environment']['status'] == 'unverified', d['dimensions']['environment']

    bad_stamp = _record()
    bad_stamp['provenance']['code']['code_semantics'] = ['not', 'a', 'stamp']
    c = L.reproduction_diagnosis(a, bad_stamp)
    assert c['verdict'] == 'provenance_incomplete' and c['reproduced'] is False, c
    assert c['dimensions']['code']['status'] == 'unverified', c['dimensions']['code']

    bad_knobs = _record()
    bad_knobs['train_knobs'] = {'optimizer': 'adamw'}  # partial descriptor
    k = L.reproduction_diagnosis(a, bad_knobs)
    assert k['verdict'] == 'provenance_incomplete' and k['reproduced'] is False, k
    assert k['dimensions']['config']['status'] == 'unverified', k['dimensions']['config']
    field = [f for f in k['dimensions']['config']['fields']
             if f['field'] == 'train_knobs'][0]
    assert field['status'] == 'unverified', field
