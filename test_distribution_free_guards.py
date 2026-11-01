"""Replay guards for the distribution-free prediction artifact.

The interval itself is a small, local computation, but an audit gate must not
trust the artifact's wrapper: a wrong artifact kind/schema, a number outside
float64, or a boolean substituted for a measured endpoint has to fail closed
instead of being replayed (or crashing the audit).  These checks are CPU-only
and do not read a split.
"""
import copy
import json
import os
import tempfile

from test_support import *  # noqa: F401,F403


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh)


def test_huge_integer_delta_is_a_failed_replay_not_an_overflow():
    artifact = {
        'kind': L.DISTRIBUTION_FREE_PREDICTION_ARTIFACT_KIND,
        'schema': L.DISTRIBUTION_FREE_PREDICTION_ARTIFACT_SCHEMA,
        'inputs': {'deltas': [10 ** 400, 1.0], 'conf': 0.95,
                   'direction': None},
        'result': {},
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'huge.json')
        _write_json(path, artifact)
        audit = A.audit_distribution_free_prediction(path)
    assert audit['status'] == 'fail', audit
    assert any('float64' in p or 'finite' in p for p in audit['problems']), audit
    json.dumps(audit, allow_nan=False)


def test_artifact_kind_and_schema_are_part_of_the_replay_contract():
    good = L.build_distribution_free_prediction([-1.0, -0.8, -1.2, -0.9])
    with tempfile.TemporaryDirectory() as td:
        for field, bad_value in (('kind', 'some_other_artifact'),
                                 ('schema', 999),
                                 ('schema', True)):
            artifact = copy.deepcopy(good)
            artifact[field] = bad_value
            path = os.path.join(td, f'bad_{field}_{bad_value!r}.json')
            _write_json(path, artifact)
            audit = A.audit_distribution_free_prediction(path)
            assert audit['status'] == 'fail', (field, bad_value, audit)
            assert any(field in p for p in audit['problems']), audit


def test_extra_result_fields_cannot_smuggle_an_unchecked_claim():
    good = L.build_distribution_free_prediction([-1.0, -0.8, -1.2, -0.9])
    for key, value in (('significant', True), ('p', 0.01), ('claim', 'confirmed')):
        artifact = copy.deepcopy(good)
        artifact['result'][key] = value
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'extra.json')
            _write_json(path, artifact)
            audit = A.audit_distribution_free_prediction(path)
        assert audit['status'] == 'fail', (key, audit)
        assert any('frozen-decision' in p or 'outside the replay schema' in p
                   for p in audit['problems']), audit


def test_boolean_endpoint_is_not_a_numeric_measurement():
    good = L.build_distribution_free_prediction([1.0, 2.0, 3.0])
    # The first order statistic is exactly 1.0; True == 1 in Python but a
    # boolean is a flag, not a measured bound.
    artifact = copy.deepcopy(good)
    artifact['result']['interval'][0] = True
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'bool.json')
        _write_json(path, artifact)
        audit = A.audit_distribution_free_prediction(path)
    assert audit['status'] == 'fail', audit
    assert any('interval[0]' in p for p in audit['problems']), audit


def test_zero_width_reason_names_the_trimmed_endpoints_not_a_constant_panel():
    # n=4 at a loose 1% request trims to j=2, so [x_(2), x_(3)] is zero-width
    # here even though the sample is not constant (it contains a 1.0).
    out = L.distribution_free_prediction_interval(
        [1.0, 2.0, 2.0, 2.0], conf=0.01)
    assert out['available'] is False, out
    assert 'endpoints are equal' in out['reason'], out
    assert 'all calibration effects are equal' not in out['reason'], out


def test_reportable_claim_keeps_the_interval_attainability_fields():
    stats = L.paired_effect_stats(
        [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95,
         -1.05, -0.85, -1.15, -0.98, -1.02, -1.08])
    claim = L.reportable_claim(stats)
    assert claim['distribution_free_prediction_95'] is not None, claim
    assert claim['distribution_free_conf_achievable'] is False, claim
    assert claim['distribution_free_n_needed'] == 39, claim
    # A malformed planning field is dropped, not coerced.
    bad = dict(stats, distribution_free_n_needed=True,
               distribution_free_conf_achievable='yes')
    out = L.reportable_claim(bad)
    assert 'distribution_free_n_needed' not in out, out
    assert 'distribution_free_conf_achievable' not in out, out
