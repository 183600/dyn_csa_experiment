"""Prospective group-sequential sign-test design: exact alpha spending.

A study that permits an interim seed look must pre-register the look schedule
and the alpha spent at each look; otherwise an "add seeds until significant"
panel is only repairable post hoc.  These checks exercise the exact integer
random-walk boundaries, the exact FWER/power, the artifact replay, the
timing/manifest/protocol audit and the CLI/Makefile wiring.

No training, no split read, and no frozen p/Holm/confirmed change.
"""
import itertools
import json
import math
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _plan(looks=(2, 4), **kwargs):
    return L.sequential_boundaries(list(looks), **kwargs)


def _brute(looks, rows, sides, q=0.5):
    """Enumerate every sign sequence through the final look."""
    total = 1 << int(looks[-1])
    stop = [0.0] * len(rows)
    for xs in itertools.product((1, -1), repeat=int(looks[-1])):
        weight = 1.0
        for x in xs:
            weight *= (q if x == 1 else (1.0 - q))
        w = 0
        prev = 0
        for index, row in enumerate(rows):
            w += sum(xs[prev:int(row['n'])])
            prev = int(row['n'])
            if L._seq_crosses(w, row['b'], sides):
                stop[index] += weight
                break
    return sum(stop), stop


def test_exact_fwer_matches_brute_force_for_all_sides():
    for sides in ('two', 'greater', 'less'):
        plan = _plan(looks=(2, 4), alpha=0.2, sides=sides,
                     spending='obrien_fleming')
        expected, _ = _brute((2, 4), plan['boundaries'], sides)
        assert math.isclose(plan['fwer'], expected, rel_tol=1e-12,
                            abs_tol=1e-15), (sides, plan['fwer'], expected)
        assert plan['fwer'] <= 0.2 + 1e-15


def test_exact_power_and_stopping_distribution_match_brute_force():
    q = 0.7
    plan = _plan(looks=(2, 4), alpha=0.2, sides='two',
                 spending='obrien_fleming')
    power = L._seq_power(plan['boundaries'], q, 'two')
    expected, stops = _brute((2, 4), plan['boundaries'], 'two', q=q)
    assert math.isclose(power['power'], expected, rel_tol=1e-12,
                        abs_tol=1e-15)
    for got, want in zip(power['prob_stop_by_look'], stops):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(
        power['expected_n'],
        sum(row['n'] * p for row, p
            in zip(plan['boundaries'], power['prob_stop_by_look']))
        + plan['boundaries'][-1]['n'] * power['prob_no_stop'],
        rel_tol=1e-12)
    # A panel that almost never stops early is observed to n_max, not to the
    # average of the stopping looks.
    assert power['expected_n'] > plan['boundaries'][-1]['n'] * 0.8


def test_conditional_power_matches_future_enumeration():
    plan = _plan(looks=(2, 4), alpha=0.2, sides='two',
                 spending='obrien_fleming')
    for observed in (-2, 0, 2):
        for q in (0.5, 0.7):
            got = L.sequential_conditional_power(
                plan, look=1, observed_w=observed, q=q)
            expected = 0.0
            for xs in itertools.product((1, -1), repeat=2):
                weight = 1.0
                for x in xs:
                    weight *= q if x == 1 else (1.0 - q)
                if L._seq_crosses(observed + sum(xs),
                                  plan['boundaries'][1]['b'], 'two'):
                    expected += weight
            assert math.isclose(got['conditional_power'], expected,
                                rel_tol=1e-12, abs_tol=1e-15), (observed, q)
    stopped = L.sequential_conditional_power(
        plan, look=2, observed_w=4, q=0.7)
    assert stopped['already_crossed'] is True
    assert stopped['conditional_power'] == 1.0
    with pytest.raises(ValueError):
        L.sequential_conditional_power(plan, look=1, observed_w=1, q=0.7)


def test_power_at_null_is_exactly_the_fwer():
    plan = _plan(looks=(3, 6), alpha=0.1, sides='two',
                 spending='pocock')
    power = L._seq_power(plan['boundaries'], 0.5, 'two')
    assert math.isclose(power['power'], plan['fwer'], rel_tol=1e-12,
                        abs_tol=1e-15)


def test_fwer_is_never_above_the_declared_alpha():
    for spending in ('pocock', 'obrien_fleming', 'linear', 'none'):
        for family_size in (1, 2, 4):
            plan = _plan(looks=(6, 9, 12), alpha=0.05,
                         spending=spending, family_size=family_size)
            assert plan['fwer_ok'] is True
            assert plan['fwer'] <= 0.05 / family_size + 1e-15
            assert plan['familywise_fwer_bound'] <= 0.05 + 1e-15


def test_boundaries_are_integer_and_attainable_or_named():
    plan = _plan(looks=(2, 4, 6), alpha=0.2, sides='two',
                 spending='linear')
    for row in plan['boundaries']:
        if row['b'] is None:
            assert row['attainable'] is False
            assert row['note']
        else:
            assert isinstance(row['b'], int) and 1 <= row['b'] <= row['n']
            assert row['attainable'] is True
            assert 0.0 < row['nominal_p_boundary'] <= 1.0


def test_monotone_z_option_can_make_a_front_loaded_plan_infeasible():
    free = _plan(looks=(6, 9, 12), alpha=0.05, spending='pocock',
                 monotone_z=False)
    assert free['status'] == 'ok'
    assert any('higher than the previous look' in w for w in free['warnings'])
    strict = _plan(looks=(6, 9, 12), alpha=0.05, spending='pocock',
                   monotone_z=True)
    assert strict['status'] in ('ok', 'not_estimable')
    if strict['status'] == 'not_estimable':
        assert strict['reason']


def test_custom_spending_is_validated_and_replayed():
    plan = _plan(looks=(2, 4, 6), alpha=0.1, spending='custom',
                 cumulative_spend=[0.0, 0.32, 1.0])
    assert plan['fwer'] <= 0.1 + 1e-15
    artifact = L.build_sequential_design(
        n_max=6, look_sizes=[2, 4, 6], alpha=0.1, spending='custom',
        cumulative_spend=[0.0, 0.32, 1.0], pilot_sign_rate=0.7)
    assert L.validate_sequential_design(artifact)['status'] == 'ok'
    with pytest.raises(ValueError):
        _plan(looks=(2, 4), alpha=0.1, spending='custom',
              cumulative_spend=[0.0, 0.9])
    with pytest.raises(ValueError):
        _plan(looks=(2, 4), alpha=0.1, spending='custom',
              cumulative_spend=[0.5, 0.4])


def test_family_size_uses_bonferroni_not_a_holm_claim():
    plan = _plan(looks=(12,), alpha=0.05, family_size=4,
                 spending='none')
    assert math.isclose(plan['comparison_alpha'], 0.05 / 4)
    assert plan['comparison_alpha'] == 0.05 / 4
    assert any('Bonferroni' in w for w in plan['warnings'])


def test_build_round_trip_and_pilot_summary():
    artifact = L.build_sequential_design(
        n_max=12, look_sizes=[6, 9, 12], alpha=0.05,
        spending='obrien_fleming', pilot_deltas=[0.2] * 8 + [-0.1] * 4,
        pilot_ppl=30.0, target_power=0.8)
    assert artifact['kind'] == 'sequential_design'
    assert artifact['inputs']['pilot_sign_rate'] == pytest.approx(2 / 3)
    assert artifact['inputs']['pilot_n_positive'] == 8
    assert artifact['inputs']['pilot_n_negative'] == 4
    assert artifact['result']['mde']['log_ppl'] is not None
    verdict = L.validate_sequential_design(artifact)
    assert verdict['status'] == 'ok', verdict
    assert verdict['result_status'] == 'ok'


def test_tampered_result_or_hash_is_rejected():
    artifact = L.build_sequential_design(
        n_max=6, look_sizes=[3, 6], alpha=0.2, pilot_sign_rate=0.7)
    bad = json.loads(json.dumps(artifact))
    bad['result']['power'] = 0.999
    assert L.validate_sequential_design(bad)['status'] == 'fail'
    bad = json.loads(json.dumps(artifact))
    bad['result']['power'] = 0.999
    bad['content_sha256'] = L._seq_content_sha(bad)
    verdict = L.validate_sequential_design(bad)
    assert verdict['status'] == 'fail'
    assert any('differs' in p for p in verdict['problems'])
    bad = json.loads(json.dumps(artifact))
    bad['result']['confirmed'] = True
    verdict = L.validate_sequential_design(bad)
    assert verdict['status'] == 'fail'
    assert any('frozen-decision' in p for p in verdict['problems'])


def test_missing_or_nonfinite_pilot_is_fail_closed():
    with pytest.raises(ValueError):
        L.build_sequential_design(n_max=12, look_sizes=[12],
                                  pilot_sign_rate=0.6,
                                  pilot_deltas=[0.1, -0.1])
    with pytest.raises(TypeError):
        L.build_sequential_design(n_max=12, look_sizes=[12],
                                  pilot_deltas=[0.1, True])
    with pytest.raises(ValueError):
        L.build_sequential_design(n_max=12, look_sizes=[12],
                                  pilot_deltas=[0.1, float('nan')])
    with pytest.raises(ValueError):
        L.build_sequential_design(n_max=12, look_sizes=[12],
                                  pilot_sd=float('inf'))
    with pytest.raises(ValueError):
        L.build_sequential_design(n_max=12, look_sizes=[12],
                                  pilot_ppl=0.0)


def test_invalid_design_parameters_are_rejected():
    for kwargs in (
            dict(n_max=12, look_sizes=[12, 9]),
            dict(n_max=12, look_sizes=[9]),
            dict(n_max=12, look_sizes=[12], alpha=0.0),
            dict(n_max=12, look_sizes=[12], alpha=1.0),
            dict(n_max=12, look_sizes=[12], sides='up'),
            dict(n_max=12, look_sizes=[12], spending='magic'),
            dict(n_max=12, look_sizes=[12], family_size=0),
            dict(n_max=12, look_sizes=[12], target_power=0.0),
            dict(n_max=12, look_sizes=[12],
                 power_curve_q=[0.7, 1.2]),
    ):
        with pytest.raises((TypeError, ValueError)):
            L.build_sequential_design(**kwargs)


def test_resolution_limited_design_is_not_estimable():
    artifact = L.build_sequential_design(
        n_max=3, look_sizes=[3], alpha=0.05, pilot_sign_rate=0.7)
    assert artifact['result']['status'] == 'not_estimable'
    assert artifact['result']['reason']
    verdict = L.validate_sequential_design(artifact)
    assert verdict['status'] == 'not_estimable'


def test_manifest_reader_builds_from_declared_block():
    manifest = {
        'seeds': list(range(12)),
        'primary_comparisons': [{'name': 'a'}, {'name': 'b'}],
        'statistics': {
            'alpha': 0.05,
            'min_power_for_test': 0.8,
            'sequential': {
                'look_sizes': [6, 9, 12],
                'spending': 'obrien_fleming',
                'pilot_sign_rate': 0.7,
                'target_power': 0.8,
            },
        },
    }
    artifact = L.sequential_design_from_manifest(manifest)
    assert artifact is not None
    assert artifact['inputs']['n_max'] == 12
    assert artifact['inputs']['look_sizes'] == [6, 9, 12]
    assert artifact['inputs']['family_size'] == 2
    assert L.sequential_design_from_manifest({'seeds': [0]}) is None


def test_protocol_reader_builds_from_the_stopping_block():
    protocol = {
        'analysis': {'alpha': 0.05},
        'sample_size': {'planned_seeds': 12, 'target_power': 0.8},
        'research_question': {'primary_family': ['a', 'b']},
        'stopping': {'sequential': {
            'look_sizes': [6, 9, 12],
            'spending': 'obrien_fleming',
            'pilot_sign_rate': 0.7,
            'target_power': 0.8,
        }},
    }
    artifact = L.sequential_design_from_protocol(protocol)
    assert artifact is not None
    assert artifact['inputs']['n_max'] == 12
    assert artifact['inputs']['family_size'] == 2
    assert artifact['inputs']['look_sizes'] == [6, 9, 12]
    assert L.sequential_design_from_protocol({}) is None
    assert L.sequential_design_from_protocol({'stopping': {}}) is None


def test_writer_is_conflict_safe():
    artifact = L.build_sequential_design(
        n_max=6, look_sizes=[6], alpha=0.2, pilot_sign_rate=0.7)
    with tempfile.TemporaryDirectory() as td:
        first = L.write_sequential_design(td, artifact)
        assert first['status'] == 'written'
        second = L.write_sequential_design(td, artifact)
        assert second['status'] == 'unchanged'
        other = L.build_sequential_design(
            n_max=6, look_sizes=[6], alpha=0.1, pilot_sign_rate=0.7)
        conflict = L.write_sequential_design(td, other)
        assert conflict['status'] == 'conflict'
        replaced = L.write_sequential_design(td, other, overwrite=True)
        assert replaced['status'] == 'written'
        stored = json.load(open(replaced['path'], encoding='utf-8'))
        assert stored['inputs']['alpha'] == 0.1


def _manifest(n=12, sequential=None):
    manifest = {
        'seeds': list(range(n)),
        'primary_comparisons': [{'name': 'a'}],
        'statistics': {'alpha': 0.05, 'min_power_for_test': 0.8},
    }
    if sequential is not None:
        manifest['statistics']['sequential'] = sequential
    return manifest


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(obj, handle)


def test_audit_reports_not_applicable_for_fixed_design():
    with tempfile.TemporaryDirectory() as td:
        manifest_path = os.path.join(td, 'manifest.json')
        _write_json(manifest_path, _manifest())
        report = L.sequential_design_audit(
            manifest_path=manifest_path,
            events_path=os.path.join(td, 'no_events.jsonl'),
            look_history_path=os.path.join(td, 'no_looks.json'))
        assert report['status'] == 'not_applicable'
        assert report['declared_by_study'] is False


def test_audit_reports_declared_but_missing_artifact():
    with tempfile.TemporaryDirectory() as td:
        manifest_path = os.path.join(td, 'manifest.json')
        _write_json(manifest_path, _manifest(
            sequential={'look_sizes': [6, 9, 12]}))
        report = L.sequential_design_audit(
            manifest_path=manifest_path,
            events_path=os.path.join(td, 'no_events.jsonl'),
            look_history_path=os.path.join(td, 'no_looks.json'))
        assert report['status'] == 'not_verifiable_locally'
        assert report['declared_by_study'] is True


def test_audit_timing_and_manifest_cross_checks():
    with tempfile.TemporaryDirectory() as td:
        manifest_path = os.path.join(td, 'manifest.json')
        _write_json(manifest_path, _manifest())
        artifact = L.build_sequential_design(
            n_max=12, look_sizes=[6, 9, 12], alpha=0.05,
            pilot_sign_rate=0.7, created_at='2026-01-02T00:00:00')
        artifact_path = os.path.join(td, 'sequential_design.json')
        _write_json(artifact_path, artifact)
        history_path = os.path.join(td, 'look_history.json')
        _write_json(history_path, {'looks': [
            {'n': 6, 'created_at': '2026-01-01T00:00:00'}]})
        events_path = os.path.join(td, 'events.jsonl')
        with open(events_path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({'kind': 'panel_look', 'created_at': '2026-01-01T00:00:00'}) + '\n')
        report = L.sequential_design_audit(
            artifact_path, manifest_path=manifest_path,
            look_history_path=history_path, events_path=events_path)
        assert report['status'] == 'fail'
        assert any('after the first recorded' in p or 'reconstruction' in p
                   for p in report['problems'])
        # A prospective artifact is acceptable (warnings come from the
        # discrete plan itself, not from provenance).
        artifact = L.build_sequential_design(
            n_max=12, look_sizes=[6, 9, 12], alpha=0.05,
            pilot_sign_rate=0.7, created_at='2025-12-31T00:00:00')
        _write_json(artifact_path, artifact)
        report = L.sequential_design_audit(
            artifact_path, manifest_path=manifest_path,
            look_history_path=history_path, events_path=events_path)
        assert report['status'] in ('ok', 'warnings')
        assert report['timing_ok'] is True
        # Manifest n mismatch is a hard failure.
        short_manifest = os.path.join(td, 'short.json')
        _write_json(short_manifest, _manifest(n=6))
        report = L.sequential_design_audit(
            artifact_path, manifest_path=short_manifest,
            look_history_path=history_path, events_path=events_path)
        assert report['status'] == 'fail'
        assert any('seed count' in p for p in report['problems'])


def test_audit_off_schedule_look_and_fixed_protocol_conflict():
    with tempfile.TemporaryDirectory() as td:
        manifest_path = os.path.join(td, 'manifest.json')
        _write_json(manifest_path, _manifest())
        artifact = L.build_sequential_design(
            n_max=12, look_sizes=[6, 9, 12], alpha=0.05,
            pilot_sign_rate=0.7, created_at='2025-12-31T00:00:00')
        artifact_path = os.path.join(td, 'sequential_design.json')
        _write_json(artifact_path, artifact)
        history_path = os.path.join(td, 'look_history.json')
        _write_json(history_path, {'looks': [
            {'n': 8, 'created_at': '2026-01-01T00:00:00'}]})
        report = L.sequential_design_audit(
            artifact_path, manifest_path=manifest_path,
            look_history_path=history_path,
            events_path=os.path.join(td, 'none.jsonl'))
        assert report['status'] == 'fail'
        assert any('not in the registered schedule' in p
                   for p in report['problems'])
        protocol_path = os.path.join(td, 'protocol.json')
        _write_json(protocol_path, {
            'stopping': {'rule': 'fixed_plan_no_optional_stopping',
                         'test_looks': 1}})
        report = L.sequential_design_audit(
            artifact_path, manifest_path=manifest_path,
            protocol_path=protocol_path,
            look_history_path=os.path.join(td, 'none.json'),
            events_path=os.path.join(td, 'none.jsonl'))
        assert report['status'] == 'fail'
        assert any('fixed_plan_no_optional_stopping' in p
                   for p in report['problems'])


def test_cli_and_makefile_wiring():
    import audit_gates as A
    assert callable(A.audit_sequential_design)
    assert callable(getattr(A, 'audit_sequential_design_artifact', None))
    source = open(os.path.join(REPO, 'Makefile'), encoding='utf-8').read()
    assert 'sequential-design:' in source
    assert 'sequential_design' in source
    assert 'part_84_sequential_design.py' in open(
        os.path.join(REPO, 'explib', '__init__.py'), encoding='utf-8').read()
    assert 'part_50_audit_sequential_design.py' in open(
        os.path.join(REPO, '_audit_gates', '__init__.py'),
        encoding='utf-8').read()
    # The protected documentation/CI files must still exist.
    assert os.path.exists(os.path.join(REPO, 'README.md'))
    assert os.path.exists(os.path.join(REPO, 'README_zh.md'))
    assert os.path.isdir(os.path.join(REPO, '.github'))
