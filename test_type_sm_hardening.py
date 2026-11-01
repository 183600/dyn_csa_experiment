"""Hardening checks for the design-stage Type S / Type M readout.

These are the replay/scale guards around the newest design block: an artifact
must not be a container for an invalid design, a JSON boolean must not stand in
for an integer or a flag, extra fields must not travel beside valid numbers,
and the required-n search must stay one pass per candidate when ``n_max`` is
raised for a larger panel.  No split is read and no frozen decision changes.
"""
import copy
import json

from test_support import *  # noqa: F401,F403


def _small_pilot():
    base = [-1.10, -0.90, -1.05, -0.95, -1.15, -0.85,
            -1.02, -0.98, -1.08, -0.92, -1.12, -0.88]
    deltas = [((x + 1.0) * 10.0 - 0.1) for x in base]
    return {'alpha': 0.05, 'm_family': 2,
            'comparisons': {'c1': {'deltas': deltas},
                            'c2': {'deltas': [-x for x in deltas]}}}


def test_from_deltas_keeps_its_own_provenance_kind():
    deltas = [-1.0, -1.1, -0.9, -1.05, -0.95, -1.15,
              -0.85, -1.02, -0.98, -1.08, -0.92, -1.12]
    direct = L.type_sm_design(12, 1.0, 0.1)
    anchored = L.type_sm_design_from_deltas(12, deltas)
    assert direct['kind'] == 'type_sm_design', direct
    assert anchored['kind'] == 'type_sm_design_from_deltas', anchored
    assert anchored['anchor'] == 'pilot_mean', anchored
    # The numerical body is still the same readout, only labelled with the
    # anchor it came from.
    assert anchored['type_m'] == L.type_sm_design(
        12, anchored['pilot']['sd'], abs(anchored['pilot']['mean']))['type_m']


def test_builder_accepts_explicit_critical_value_without_alpha_or_family():
    artifact = L.build_type_sm_design(
        12, 1.0, 0.3, alpha=None, family_size=None, critical_value=2.0)
    assert artifact['inputs']['alpha'] is None, artifact
    assert artifact['inputs']['family_size'] is None, artifact
    assert artifact['result']['critical_rule'] == 'explicit', artifact
    valid = L.validate_type_sm_design(artifact)
    assert valid['status'] == 'ok', valid


def test_builder_refuses_to_publish_an_invalid_design():
    base = {'n': 12, 'sigma': 1.0, 'effect': 0.1}
    for extra in ({'sigma': 0.0}, {'effect': 0.0},
                  {'target_power': 1.0}, {'max_type_s': 0.6},
                  {'max_type_m': 0.5}, {'n': 10 ** 400},
                  {'family_size': None}):
        kwargs = dict(base)
        kwargs.update(extra)
        try:
            L.build_type_sm_design(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'builder published invalid design {extra!r}')


def _tampered(mutator):
    artifact = L.build_type_sm_design(12, 1.0, 0.1)
    bad = copy.deepcopy(artifact)
    mutator(bad)
    return L.validate_type_sm_design(bad)


def test_artifact_schema_boolean_is_not_schema_one():
    verdict = _tampered(lambda a: a.__setitem__('schema', True))
    assert verdict['status'] == 'fail', verdict
    assert any('schema' in p for p in verdict['problems']), verdict


def test_extra_fields_cannot_travel_beside_a_valid_numeric_body():
    for where in ('result', 'inputs'):
        def add(artifact, where=where):
            artifact[where]['claim'] = 'confirmed'
        verdict = _tampered(add)
        assert verdict['status'] == 'fail', (where, verdict)
        assert any('outside the replay schema' in p
                   for p in verdict['problems']), (where, verdict)


def test_exact_fields_reject_boolean_coercion_and_missing_sections():
    def boolean_count(artifact):
        artifact['result']['min_n_for_power'] = True
    verdict = _tampered(boolean_count)
    assert verdict['status'] == 'fail', verdict
    assert any('min_n_for_power' in p for p in verdict['problems']), verdict

    def nested_flag(artifact):
        artifact['result']['meets']['power'] = 1
    verdict = _tampered(nested_flag)
    assert verdict['status'] == 'fail', verdict
    assert any('meets' in p for p in verdict['problems']), verdict

    def drop_policy(artifact):
        del artifact['result']['policy']
    verdict = _tampered(drop_policy)
    assert verdict['status'] == 'fail', verdict
    assert any('policy' in p for p in verdict['problems']), verdict


def test_duplicate_primary_names_are_not_silently_overwritten():
    manifest = {
        'seeds': list(range(12)),
        'primary_comparisons': [
            {'name': 'c1', 'a': 'a', 'b': 'b'},
            {'name': 'c1', 'a': 'a', 'b': 'c'},
        ],
        'statistics': {'alpha': 0.05},
    }
    out = L.type_sm_from_manifest(manifest, _small_pilot())
    assert out['status'] == 'fail', out
    assert any('duplicate' in p for p in out['problems']), out
    assert list(out['comparisons']) == ['c1'], out


def test_required_n_search_evaluates_each_candidate_once(monkeypatch):
    # An impossible 99.9% power target forces the full grid.  The old
    # implementation walked it three times and recomputed each erf/exp; the
    # single-pass version must call the readout at most once per candidate
    # (plus the one point estimate for n itself).
    calls = []
    real = L._tsm_metrics

    def counting(k, lam, z):
        calls.append(int(k))
        return real(k, lam, z)

    monkeypatch.setattr(L, '_tsm_metrics', counting)
    out = L.type_sm_design(
        12, 1.0, 0.1, target_power=0.999999, n_max=300,
        max_type_s=0.05, max_type_m=2.0)
    assert out['status'] in ('ok', 'warnings'), out
    assert out['min_n_for_power'] is None, out
    assert len(calls) <= 301, (len(calls), calls[:5])
    # The candidate grid is still evaluated in order and without gaps.
    assert calls[1:] == list(range(2, 301)), calls[:5]


def test_validation_verdict_is_strict_json():
    artifact = L.build_type_sm_design(12, 1.0, 0.1)
    bad = copy.deepcopy(artifact)
    bad['result']['interpretation'] = float('nan')
    verdict = L.validate_type_sm_design(bad)
    assert verdict['status'] == 'fail', verdict
    json.dumps(verdict, allow_nan=False)
