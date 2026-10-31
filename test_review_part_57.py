"""CPU checks for the design-stage endpoint-sensitivity gate.

These checks pin the new pre-run contract: a bounded endpoint has a measurable
beneficial headroom, an endpoint already at the bound cannot support an
improvement claim, a declared minimum meaningful effect (MEE) must fit inside
that headroom, repeated identical evaluations estimate the measurement
resolution, and the plan's MDE must be no larger than the MEE.  The audit is
fail-closed only on declared requirements and never reads the test split.
"""
import contextlib
import inspect
import io
import json
import os
import tempfile

from test_support import *  # noqa: F401,F403


def _manifest(**overrides):
    m = {
        'primary_comparisons': [
            {'name': 'c_acc', 'endpoint': 'niah.top1_accuracy@L4096'},
            {'name': 'c_ppl', 'endpoint': 'by_len.ppl@4096'},
        ],
        'endpoint_sensitivity': {
            'require_headroom': True,
            'endpoints': {
                'niah.top1_accuracy@L4096': {
                    'minimum_meaningful_effect': 0.05},
                'by_len.ppl@4096': {'minimum_meaningful_effect': 0.5},
            },
        },
    }
    m.update(overrides)
    return m


def _pilot(**overrides):
    p = {
        'split': 'validation',
        'endpoints': {
            'niah.top1_accuracy@L4096': {
                'control_level': 0.90,
                'repeat_values': [0.900, 0.901, 0.899],
                'paired_sd': 0.01,
                'n_units': 48,
            },
            'by_len.ppl@4096': {
                'control_level': 30.0,
                'repeat_values': [30.00, 30.01, 29.99],
                'paired_sd': 0.20,
                'n_units': 48,
            },
        },
    }
    p.update(overrides)
    return p


def test_endpoint_metric_uses_longest_delimited_token():
    assert L.endpoint_metric('boundary_f1.tol1@4096')['metric'] == 'boundary_f1'
    assert L.endpoint_metric('niah.top1_accuracy@L4096')['metric'] == 'top1_accuracy'
    assert L.endpoint_metric('by_len.ppl@4096')['metric'] == 'ppl'
    assert L.endpoint_metric('by_len.ppl@4096')['direction'] == 'lower_is_better'
    unknown = L.endpoint_metric('mystery_score@8')
    assert unknown['status'] == 'not_estimable' and unknown['metric'] is None
    assert L.endpoint_metric(None)['status'] == 'not_estimable'


def test_headroom_bounded_unbounded_and_best_arm_reference():
    h = L.endpoint_headroom('accuracy', 0.90, minimum_meaningful_effect=0.05)
    assert h['status'] == 'ok' and h['structurally_possible'] is True
    assert abs(h['headroom'] - 0.10) < 1e-12
    assert abs(h['headroom_relative'] - 0.10) < 1e-12
    # The better arm is the reference, so two levels cannot fake headroom.
    both = L.endpoint_headroom('accuracy', arm_levels={'a': 0.90, 'b': 0.96},
                               minimum_meaningful_effect=0.05)
    assert both['reference_level'] == 0.96
    assert abs(both['headroom'] - 0.04) < 1e-12
    assert both['status'] == 'uninformative'
    # A positive, unbounded-above PPL endpoint is lower-is-better: 30 -> 0.
    p = L.endpoint_headroom('ppl', 30.0, minimum_meaningful_effect=0.5)
    assert p['status'] == 'ok' and p['headroom'] == 30.0
    assert p['headroom_relative'] is None
    # An explicitly unbounded beneficial direction is not bound-limited.
    u = L.endpoint_headroom('custom', 5.0, metric='custom',
                            bounds=[0.0, None],
                            direction='higher_is_better')
    assert u['status'] == 'ok' and u['headroom'] is None
    assert u['bounded_beneficial'] is False


def test_headroom_saturation_low_and_invalid():
    sat = L.endpoint_headroom('accuracy', 1.0, minimum_meaningful_effect=0.05)
    assert sat['status'] == 'saturated'
    assert sat['exhausted'] is True and sat['structurally_possible'] is False
    low = L.endpoint_headroom('accuracy', 0.97)
    assert low['status'] == 'low_headroom' and low['structurally_possible'] is True
    near = L.endpoint_headroom('accuracy', 0.99, minimum_meaningful_effect=0.05)
    assert near['status'] == 'uninformative'
    assert near['headroom_covers_mee'] is False
    assert L.endpoint_headroom('accuracy')['status'] == 'not_verifiable'
    assert L.endpoint_headroom('accuracy', 1.2)['status'] == 'invalid'
    assert L.endpoint_headroom('accuracy', 0.5,
                               minimum_meaningful_effect=-1.0)['status'] == 'invalid'
    assert L.endpoint_headroom('accuracy', 0.5,
                               bounds=[2.0, 1.0])['status'] == 'invalid'
    assert L.endpoint_headroom('mystery')['status'] == 'not_estimable'


def test_measurement_resolution_from_identical_repeats():
    exact = L.endpoint_measurement_resolution([0.90, 0.90, 0.90])
    assert exact['status'] == 'ok' and exact['resolution'] == 0.0
    fine = L.endpoint_measurement_resolution(
        [0.900, 0.905, 0.910], minimum_meaningful_effect=0.02)
    assert fine['resolves_mee'] is True and fine['status'] == 'ok'
    coarse = L.endpoint_measurement_resolution(
        [0.900, 0.905, 0.910], minimum_meaningful_effect=0.005)
    assert coarse['resolves_mee'] is False
    assert coarse['status'] == 'resolution_limited'
    one = L.endpoint_measurement_resolution([0.9], minimum_meaningful_effect=0.01)
    assert one['status'] == 'not_verifiable'
    dirty = L.endpoint_measurement_resolution(
        [0.9, 'x', None, float('nan'), 0.92], minimum_meaningful_effect=0.03)
    assert dirty['n_repeats'] == 2 and dirty['dropped_nonfinite'] == 3
    assert dirty['status'] == 'ok'


def test_combined_claim_gate_and_mde():
    good = L.endpoint_sensitivity(
        'accuracy', control_level=0.90, minimum_meaningful_effect=0.05,
        repeat_values=[0.900, 0.901, 0.899], paired_sd=0.01, n_units=48)
    assert good['status'] == 'ok' and good['claim_allowed'] is True
    assert good['mde']['status'] == 'ok' and good['mde']['mde'] > 0.0
    assert good['mde']['mde_covers_mee'] is True
    sat = L.endpoint_sensitivity(
        'accuracy', control_level=1.0, minimum_meaningful_effect=0.05,
        repeat_values=[1.0, 1.0], paired_sd=0.01, n_units=48)
    assert sat['status'] == 'saturated' and sat['claim_allowed'] is False
    coarse = L.endpoint_sensitivity(
        'accuracy', control_level=0.90, minimum_meaningful_effect=0.005,
        repeat_values=[0.900, 0.905, 0.910], paired_sd=0.01, n_units=48)
    assert coarse['status'] == 'uninformative'
    assert any('resolution' in b for b in coarse['blockers'])
    under = L.endpoint_sensitivity(
        'accuracy', control_level=0.90, minimum_meaningful_effect=0.001,
        repeat_values=[0.900, 0.901, 0.899], paired_sd=0.20, n_units=4)
    assert under['status'] == 'uninformative'
    assert any('MDE' in b for b in under['blockers'])
    no_mee = L.endpoint_sensitivity(
        'accuracy', control_level=0.90, repeat_values=[0.900, 0.901])
    assert no_mee['informative'] is None and no_mee['claim_allowed'] is None
    assert no_mee['headroom']['structurally_possible'] is True
    bad = L.endpoint_sensitivity(
        'accuracy', control_level=0.90, minimum_meaningful_effect=0.05,
        paired_sd=0.0, n_units=48)
    assert bad['mde']['status'] == 'invalid'


def test_audit_ok_and_comparison_claims():
    out = L.endpoint_sensitivity_audit(_manifest(), _pilot(), verbose=False)
    assert out['status'] == 'ok', (out['problems'], out['warnings'])
    assert out['n_endpoints'] == 2 and out['n_comparisons'] == 2
    assert set(out['endpoint_sensitivity']) == {
        'by_len.ppl@4096', 'niah.top1_accuracy@L4096'}
    acc = out['endpoint_sensitivity']['niah.top1_accuracy@L4096']
    assert acc['claim_allowed'] is True
    assert out['comparisons']['c_acc']['claim_allowed'] is True
    assert out['comparisons']['c_ppl']['claim_allowed'] is True
    # The result must be strict-JSON (no numpy scalars / NaN / Infinity).
    json.dumps(out, allow_nan=False)


def test_audit_fails_on_saturation_and_missing_pilot():
    pilot = _pilot()
    pilot['endpoints']['niah.top1_accuracy@L4096']['control_level'] = 1.0
    sat = L.endpoint_sensitivity_audit(_manifest(), pilot, verbose=False)
    assert sat['status'] == 'fail'
    assert any('beneficial bound' in p for p in sat['problems'])
    assert sat['comparisons']['c_acc']['claim_allowed'] is False
    missing = L.endpoint_sensitivity_audit(_manifest(), None, verbose=False)
    assert missing['status'] == 'fail'
    assert missing['pilot_status'] == 'not_available'
    assert any('headroom is not_verifiable' in p for p in missing['problems'])


def test_audit_conflicting_mee_and_bad_flag():
    m = _manifest()
    m['endpoint_sensitivity']['endpoints'][
        'niah.top1_accuracy@L4096']['minimum_meaningful_effect'] = 0.10
    pilot = _pilot()
    pilot['endpoints']['niah.top1_accuracy@L4096'][
        'minimum_meaningful_effect'] = 0.05
    out = L.endpoint_sensitivity_audit(m, pilot, verbose=False)
    assert out['status'] == 'fail'
    assert any('different minimum meaningful effects' in p
               for p in out['problems'])
    bad = _manifest()
    bad['endpoint_sensitivity']['require_headroom'] = 'yes'
    out2 = L.endpoint_sensitivity_audit(bad, _pilot(), verbose=False)
    assert out2['status'] == 'fail'
    assert any('must be a boolean' in p for p in out2['problems'])


def test_audit_resolution_and_mde_requirements():
    m = _manifest()
    m['endpoint_sensitivity']['require_measurement_resolution'] = True
    m['endpoint_sensitivity']['require_mde_within_headroom'] = True
    good = L.endpoint_sensitivity_audit(m, _pilot(), verbose=False)
    assert good['status'] == 'ok', good['problems']
    pilot = _pilot()
    pilot['endpoints']['niah.top1_accuracy@L4096']['repeat_values'] = [
        0.90, 0.95, 0.85]
    coarse = L.endpoint_sensitivity_audit(m, pilot, verbose=False)
    assert coarse['status'] == 'fail'
    assert any('measurement resolution is not smaller' in p
               for p in coarse['problems'])
    m2 = _manifest()
    m2['endpoint_sensitivity']['require_mde_within_headroom'] = True
    pilot2 = _pilot()
    pilot2['endpoints']['niah.top1_accuracy@L4096']['paired_sd'] = 0.5
    pilot2['endpoints']['niah.top1_accuracy@L4096']['n_units'] = 4
    under = L.endpoint_sensitivity_audit(m2, pilot2, verbose=False)
    assert under['status'] == 'fail'
    assert any('planned MDE exceeds' in p for p in under['problems'])
    no_sd = _pilot()
    del no_sd['endpoints']['niah.top1_accuracy@L4096']['paired_sd']
    no_mde = L.endpoint_sensitivity_audit(m2, no_sd, verbose=False)
    assert no_mde['status'] == 'fail'
    assert any('MDE is not verifiable' in p for p in no_mde['problems'])
    # A requirement without a target margin is not a pass either.
    m3 = _manifest()
    for _ep in m3['endpoint_sensitivity']['endpoints'].values():
        _ep.pop('minimum_meaningful_effect', None)
    m3['endpoint_sensitivity']['require_measurement_resolution'] = True
    m3['endpoint_sensitivity']['require_mde_within_headroom'] = True
    pilot3 = _pilot()
    for _ep in pilot3['endpoints'].values():
        _ep.pop('minimum_meaningful_effect', None)
    no_target = L.endpoint_sensitivity_audit(m3, pilot3, verbose=False)
    assert no_target['status'] == 'fail'
    assert any('no MEE or max_relative_resolution' in p
               for p in no_target['problems'])
    assert any('no minimum_meaningful_effect is declared' in p
               for p in no_target['problems'])
    # A declared relative resolution target can replace the MEE, but a coarse
    # measured resolution then has to fail the requirement.
    m4 = _manifest()
    for _ep in m4['endpoint_sensitivity']['endpoints'].values():
        _ep.pop('minimum_meaningful_effect', None)
    m4['endpoint_sensitivity']['require_measurement_resolution'] = True
    m4['endpoint_sensitivity']['max_relative_resolution'] = 1e-6
    coarse_rel = L.endpoint_sensitivity_audit(m4, _pilot(), verbose=False)
    assert coarse_rel['status'] == 'fail'
    assert any('coarser than the declared maximum' in p
               for p in coarse_rel['problems'])


def test_audit_missing_declaration_is_warning_not_pass():
    m = {'primary_comparisons': [
        {'name': 'c', 'endpoint': 'by_len.ppl@4096'}]}
    out = L.endpoint_sensitivity_audit(m, None, verbose=False)
    assert out['status'] == 'warnings'
    assert out['requirements']['require_headroom'] is False
    assert any('no endpoint_sensitivity block' in w for w in out['warnings'])


def test_audit_declared_but_empty_and_required_mee():
    empty = {'primary_comparisons': [],
             'endpoint_sensitivity': {'require_headroom': True,
                                      'endpoints': {}}}
    out = L.endpoint_sensitivity_audit(empty, None, verbose=False)
    assert out['status'] == 'fail'
    assert any('no endpoint is listed' in p for p in out['problems'])
    m = _manifest()
    m['endpoint_sensitivity']['require_declared_mee'] = True
    for _ep in m['endpoint_sensitivity']['endpoints'].values():
        _ep.pop('minimum_meaningful_effect', None)
    out2 = L.endpoint_sensitivity_audit(m, _pilot(), verbose=False)
    assert out2['status'] == 'fail'
    assert any('require_declared_mee is set but no' in p
               for p in out2['problems'])
    test_pilot = _pilot()
    test_pilot['split'] = 'test'
    out3 = L.endpoint_sensitivity_audit(_manifest(), test_pilot, verbose=False)
    assert out3['status'] == 'fail'
    assert any('held-out test split' in p for p in out3['problems'])


def test_audit_unreadable_manifest_and_pilot():
    out = L.endpoint_sensitivity_audit(
        os.path.join(REPO, 'does-not-exist.json'), None, verbose=False)
    assert out['status'] == 'not_verifiable_locally'
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, 'bad.json')
        with open(bad, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        broken = L.endpoint_sensitivity_audit(_manifest(), bad, verbose=False)
        assert broken['status'] == 'fail'
        assert any('not readable JSON' in p for p in broken['problems'])


def test_audit_never_reads_the_test_split(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError('endpoint sensitivity must not read the test split')
    monkeypatch.setattr(L, 'load_wikitest', _boom, raising=False)
    out = L.endpoint_sensitivity_audit(_manifest(), _pilot(), verbose=False)
    assert out['status'] == 'ok', out['problems']


def test_cli_and_design_preflight_wiring():
    cli = inspect.getsource(A._run_cli)
    for alias in ('endpoint_sensitivity', 'endpoint_headroom', 'headroom',
                  'dynamic_range', 'instrument_sensitivity', 'ceiling_floor'):
        assert alias in cli, alias
    assert 'endpoint_sensitivity' in inspect.getsource(A.run_report)
    with contextlib.redirect_stdout(io.StringIO()):
        dp = A.audit_design_preflight()
    assert dp['status'] == 'ok', dp['violations']
    block = dp.get('endpoint_sensitivity') or {}
    assert block.get('status') == 'warnings'
    assert 'by_len.ppl@4096' in (block.get('endpoints') or {})
    with contextlib.redirect_stdout(io.StringIO()):
        gate = A.audit_endpoint_sensitivity(verbose=False)
    assert gate['status'] == 'warnings'
    assert gate['n_endpoints'] == 1
