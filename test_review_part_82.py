"""CPU review checks (split group 82): design-stage Type S / Type M risk.

Power and the minimum detectable effect answer "can this panel reject?".
They do not answer what a rejection would mean: how often the sign is wrong
and how much the magnitude is exaggerated.  These checks pin the normal
location formulas against a Monte Carlo, the family-adjusted critical value,
scale invariance, required-n searches, the pilot-derived anchor, the replay
artifact, and the manifest/gate wiring.  This is a design readout: no test
split, no p-value, no Holm member and no frozen verdict is touched.
"""
import inspect
import json
import math
import os
import tempfile

import numpy as np

from test_support import *  # noqa: F401,F403


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _toy_manifest(*, require=False, alpha=0.05, thresholds=None):
    stat = {'alpha': alpha,
            'test': 'exact two-sided sign-flip over paired per-seed deltas'}
    if thresholds is not None:
        stat['type_sm'] = thresholds
    elif require:
        stat['type_sm'] = {'require': True, 'target_power': 0.80,
                           'max_type_s': 0.05, 'max_type_m': 2.0}
    return {
        'seeds': list(range(12)),
        'primary_comparisons': [
            {'name': 'c1', 'a': 'a', 'b': 'b',
             'endpoint': 'by_len.ppl@4096', 'expected_sign': 'negative'},
            {'name': 'c2', 'a': 'a', 'b': 'c',
             'endpoint': 'by_len.ppl@4096', 'expected_sign': 'negative'},
        ],
        'statistics': stat,
    }


def _pilot(deltas, *, alpha=0.05, m_family=2):
    return {
        'kind': 'power_validation', 'alpha': alpha, 'm_family': m_family,
        'comparisons': {
            'c1': {'deltas': list(deltas)},
            'c2': {'deltas': [-x for x in deltas]},
        },
    }


def _small_deltas(scale=1.0):
    # mean/sd is about 0.1: deliberately underpowered at n=12.  The base has
    # mean -1.0 and SD 0.1, so re-centre to -0.1 and widen to SD 1.0.
    base = [-1.10, -0.90, -1.05, -0.95, -1.15, -0.85,
            -1.02, -0.98, -1.08, -0.92, -1.12, -0.88]
    return [scale * ((x + 1.0) * 10.0 - 0.1) for x in base]


def _strong_deltas(scale=1.0):
    base = [-5.0, -5.1, -4.9, -5.05, -4.95, -5.02,
            -4.98, -5.08, -4.92, -5.01, -4.99, -5.06]
    return [scale * x for x in base]


def test_analytic_type_sm_matches_monte_carlo():
    n, sigma, effect = 12, 1.0, 0.1
    out = L.type_sm_design(n, sigma, effect)
    assert out['available'] is True and out['status'] == 'warnings', out
    z, lam = out['critical_value'], out['noncentrality']
    rng = np.random.default_rng(20261101)
    draws = rng.normal(lam, 1.0, 400000)
    mask = np.abs(draws) > z
    assert mask.sum() > 10000, mask.sum()
    mc_power = float(mask.mean())
    mc_type_s = float(np.mean(draws[mask] < 0.0))
    mc_type_m = float(np.mean(np.abs(draws[mask])) / lam)
    mc_over_2 = float(np.mean(np.abs(draws[mask]) > 2.0 * lam))
    mc_under_half = float(np.mean(np.abs(draws[mask]) < 0.5 * lam))
    assert _close(out['power'], mc_power, 0.01), (out['power'], mc_power)
    assert abs(out['type_s'] - mc_type_s) < 0.02, (out['type_s'], mc_type_s)
    assert abs(out['type_m'] - mc_type_m) < 0.10, (out['type_m'], mc_type_m)
    assert _close(out['p_overstate_2x'], mc_over_2, 0.02), (
        out['p_overstate_2x'], mc_over_2)
    assert abs(out['p_understate_half'] - mc_under_half) < 0.02, (
        out['p_understate_half'], mc_under_half)
    # A mid-power effect exercises the non-degenerate overstatement band
    # (2*lam > z) and the empty understatement band (lam/2 <= z).
    mid = L.type_sm_design(12, 1.0, 0.5)
    z2, lam2 = mid['critical_value'], mid['noncentrality']
    draws2 = rng.normal(lam2, 1.0, 400000)
    mask2 = np.abs(draws2) > z2
    mc_over2 = float(np.mean(np.abs(draws2[mask2]) > 2.0 * lam2))
    mc_under2 = float(np.mean(np.abs(draws2[mask2]) < 0.5 * lam2))
    assert abs(mid['p_overstate_2x'] - mc_over2) < 0.01, (
        mid['p_overstate_2x'], mc_over2)
    assert abs(mid['p_understate_half'] - mc_under2) < 0.01, (
        mid['p_understate_half'], mc_under2)
    # The formulas must be internally consistent: Type S is the lower-tail
    # share of the rejection probability, and Type M cannot be below 1.
    assert _close(out['type_s'],
                  out['power_wrong_direction'] / out['power'])
    assert out['type_m'] >= 1.0 - 1e-12


def test_scale_invariance_sign_symmetry_and_monotonicity():
    ref = L.type_sm_design(20, 2.0, 0.7)
    assert ref['status'] in ('ok', 'warnings'), ref
    for factor in (1e-9, 1e-3, 1.0, 1e6):
        got = L.type_sm_design(20, 2.0 * factor, 0.7 * factor)
        assert got['status'] == ref['status'], (factor, got)
        for key in ('power', 'type_s', 'type_m', 'p_overstate_2x',
                    'p_understate_half'):
            assert _close(got[key], ref[key], 1e-9), (factor, key, got, ref)
    flipped = L.type_sm_design(20, 2.0, -0.7)
    assert _close(flipped['power'], ref['power'])
    assert _close(flipped['type_s'], ref['type_s'])
    assert _close(flipped['type_m'], ref['type_m'])
    assert flipped['expected_signed_estimate'] < 0.0 < ref['expected_signed_estimate']
    # A larger true effect cannot lower power or raise either error metric.
    grid = [L.type_sm_design(12, 1.0, e) for e in (0.1, 0.2, 0.4, 0.8, 1.6)]
    for left, right in zip(grid, grid[1:]):
        assert right['power'] >= left['power'] - 1e-12, (left, right)
        assert right['type_s'] <= left['type_s'] + 1e-12, (left, right)
        assert right['type_m'] <= left['type_m'] + 1e-12, (left, right)
    # 12 seeds cannot reach 80% power for a 0.1-SD effect within the search.
    assert grid[0]['min_n_for_power'] is None, grid[0]
    assert grid[-1]['min_n_for_power'] is not None, grid[-1]


def test_family_adjustment_and_explicit_critical_value():
    one = L.type_sm_design(12, 1.0, 0.3, family_size=1)
    six = L.type_sm_design(12, 1.0, 0.3, family_size=6)
    assert _close(one['alpha_adjusted'], 0.05), one
    assert _close(six['alpha_adjusted'], 0.05 / 6.0), six
    assert six['critical_value'] > one['critical_value'], (one, six)
    assert six['power'] < one['power'], (one, six)
    assert six['type_m'] > one['type_m'], (one, six)
    # An explicit critical value bypasses the alpha arithmetic but is recorded
    # as such instead of being silently labelled Bonferroni.
    explicit = L.type_sm_design(
        12, 1.0, 0.3, alpha=None, family_size=None, critical_value=2.0)
    assert explicit['status'] in ('ok', 'warnings'), explicit
    assert explicit['critical_rule'] == 'explicit', explicit
    assert explicit['alpha_adjusted'] is None, explicit
    assert _close(explicit['critical_value'], 2.0), explicit


def test_invalid_inputs_fail_closed_without_raising():
    bad = [
        {'n': 1, 'sigma': 1.0, 'effect': 0.1},
        {'n': 12, 'sigma': 0.0, 'effect': 0.1},
        {'n': 12, 'sigma': 1.0, 'effect': 0.0},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'alpha': 0.0},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'alpha': 1.0},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'family_size': 0},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'target_power': 1.0},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'max_type_s': 0.6},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'max_type_m': 0.9},
        {'n': 12, 'sigma': float('nan'), 'effect': 0.1},
        {'n': 12, 'sigma': float('inf'), 'effect': 0.1},
        {'n': True, 'sigma': 1.0, 'effect': 0.1},
        {'n': 12, 'sigma': '1', 'effect': 0.1},
        {'n': 12, 'sigma': 1.0, 'effect': 0.1, 'n_max': 1},
    ]
    for kwargs in bad:
        out = L.type_sm_design(**kwargs)
        assert out['status'] == 'invalid' and out['available'] is False, (
            kwargs, out)
        assert out['problems'], (kwargs, out)
    # The artifact builder is stricter: non-finite values are rejected rather
    # than turned into an artifact whose inputs cannot be replayed.
    for kwargs in ({'sigma': float('nan')}, {'effect': float('inf')},
                   {'n': 1}, {'family_size': 0}):
        args = {'n': 12, 'sigma': 1.0, 'effect': 0.1}
        args.update(kwargs)
        try:
            L.build_type_sm_design(**args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'builder accepted {args!r}')


def test_required_n_searches_hit_each_threshold():
    # Type S is the binding constraint here: the target power is deliberately
    # low so the required-n for power is smaller than the required-n for the
    # sign-error threshold.
    sign = L.type_sm_design(
        12, 1.0, 0.1, target_power=0.10, max_type_s=0.02, max_type_m=10.0,
        n_max=2000)
    assert sign['min_n_for_power'] == 43, sign
    assert sign['min_n_for_type_s'] == 69, sign
    for n in (sign['min_n_for_type_s'],):
        at = L.type_sm_design(
            n, 1.0, 0.1, target_power=0.10, max_type_s=0.02,
            max_type_m=10.0, n_max=2000)
        assert at['type_s'] <= 0.02 and at['power'] >= 0.10, at
        below = L.type_sm_design(
            n - 1, 1.0, 0.1, target_power=0.10, max_type_s=0.02,
            max_type_m=10.0, n_max=2000)
        assert below['type_s'] > 0.02 or below['power'] < 0.10, below
    # Type M is the binding constraint here.
    mag = L.type_sm_design(
        12, 1.0, 0.4, target_power=0.20, max_type_s=0.05, max_type_m=1.5,
        n_max=500)
    assert mag['min_n_for_power'] == 8, mag
    assert mag['min_n_for_type_m'] == 21, mag
    at = L.type_sm_design(
        mag['min_n_for_type_m'], 1.0, 0.4, target_power=0.20,
        max_type_s=0.05, max_type_m=1.5, n_max=500)
    assert at['type_m'] <= 1.5 and at['power'] >= 0.20, at
    below = L.type_sm_design(
        mag['min_n_for_type_m'] - 1, 1.0, 0.4, target_power=0.20,
        max_type_s=0.05, max_type_m=1.5, n_max=500)
    assert below['type_m'] > 1.5 or below['power'] < 0.20, below
    # The shortfall fields are derived from the same searches and never
    # negative at the planned n.
    for key in ('n_shortfall_for_power', 'n_shortfall_for_type_s',
                'n_shortfall_for_type_m'):
        assert mag[key] is not None and mag[key] >= 0, (key, mag)


def test_curve_is_anchor_relative_and_monotone():
    curve = L.type_sm_design_curve(
        12, 1.0, 0.4, factors=(1.0, 0.75, 0.5, 0.25))
    assert curve['status'] == 'ok' and curve['available'] is True, curve
    assert [p['factor'] for p in curve['points']] == [1.0, 0.75, 0.5, 0.25]
    for left, right in zip(curve['points'], curve['points'][1:]):
        assert right['power'] <= left['power'] + 1e-12, (left, right)
        assert right['type_s'] >= left['type_s'] - 1e-12, (left, right)
        assert right['type_m'] >= left['type_m'] - 1e-12, (left, right)
    # A curve is a diagnostic, not a set of claims: no p/Holm field names leak
    # into the generated payload (policy prose may *mention* verdicts).
    def _keys(obj):
        out = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                out.add(key)
                out |= _keys(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                out |= _keys(value)
        return out
    names = _keys(curve)
    for forbidden in ('p_holm', 'confirmed', 'verified', 'verdict'):
        assert forbidden not in names, (forbidden, curve)
    bad = L.type_sm_design_curve(12, 1.0, 0.4, factors=(1.0, -0.5))
    assert bad['status'] == 'invalid' and bad['problems'], bad


def test_pilot_anchor_zero_effect_and_named_drops():
    deltas = _small_deltas(0.1)
    out = L.type_sm_design_from_deltas(12, deltas)
    assert out['status'] in ('ok', 'warnings'), out
    assert out['anchor'] == 'pilot_mean', out
    assert out['pilot']['n_finite'] == 12 and out['pilot']['n_dropped'] == 0
    assert out['pilot']['mean'] < 0.0 and out['pilot']['sd'] > 0.0, out
    # A declared SESOI overrides the pilot mean as the assumed truth.
    declared = L.type_sm_design_from_deltas(12, deltas, sesoi=0.5)
    assert declared['anchor'] == 'declared_sesoi', declared
    assert _close(declared['anchor_effect'], 0.5), declared
    # A zero-mean pilot is not a zero-risk pass; Type M has no finite value at
    # a true zero effect.
    zero = L.type_sm_design_from_deltas(12, [-1.0, 1.0] * 6)
    assert zero['status'] == 'not_applicable', zero
    assert zero['available'] is False and zero['not_applicable_reason'], zero
    # Non-finite values are dropped by position, never interpolated.
    messy = _small_deltas(0.1)
    messy[2] = float('nan')
    messy[7] = float('inf')
    got = L.type_sm_design_from_deltas(12, messy)
    assert got['pilot']['n_dropped'] == 2, got
    assert got['pilot']['dropped_positions'] == [2, 7], got
    # An explicit SD overrides the recomputed one but must be reported.
    override = L.type_sm_design_from_deltas(
        12, _small_deltas(0.1), pilot_sd=0.5)
    assert any('pilot_sd overrides' in w for w in override['warnings']), override
    # A single finite delta has no sample SD; an explicit SD must still state
    # the assumed effect so the anchor is not silently invented.
    one = L.type_sm_design_from_deltas(12, [-0.1])
    assert one['status'] == 'invalid' and one['problems'], one
    no_anchor = L.type_sm_design_from_deltas(12, [], pilot_sd=1.0)
    assert no_anchor['status'] == 'invalid' and no_anchor['problems'], no_anchor
    anchored = L.type_sm_design_from_deltas(
        12, [], pilot_sd=1.0, sesoi=0.5)
    assert anchored['anchor'] == 'declared_sesoi', anchored


def test_artifact_replay_tamper_and_frozen_field_rejection():
    artifact = L.build_type_sm_design(12, 1.0, 0.1)
    assert artifact['kind'] == 'type_sm_design_artifact', artifact
    assert artifact['inputs']['n'] == 12, artifact
    valid = L.validate_type_sm_design(artifact)
    assert valid['status'] == 'ok', valid
    assert valid['result']['type_s'] == artifact['result']['type_s'], valid

    tampered = json.loads(json.dumps(artifact))
    tampered['result']['type_m'] = 1.0
    bad = L.validate_type_sm_design(tampered)
    assert bad['status'] == 'fail', bad
    assert any('type_m' in p for p in bad['problems']), bad

    no_hash = json.loads(json.dumps(artifact))
    del no_hash['content_sha256']
    bad = L.validate_type_sm_design(no_hash)
    assert bad['status'] == 'fail' and bad['problems'], bad

    frozen = json.loads(json.dumps(artifact))
    frozen['result']['confirmed'] = True
    bad = L.validate_type_sm_design(frozen)
    assert bad['status'] == 'fail', bad
    assert any('frozen-decision' in p for p in bad['problems']), bad

    # Rebuilding the same inputs with the same timestamp is byte-identical;
    # the timestamp itself is provenance, not part of the content identity.
    a = L.build_type_sm_design(12, 1.0, 0.1, created_at='2026-11-01T00:00:00')
    b = L.build_type_sm_design(12, 1.0, 0.1, created_at='2026-11-02T00:00:00')
    assert a['content_sha256'] == b['content_sha256'], (a, b)
    assert a['result']['type_m'] == b['result']['type_m'], (a, b)


def test_manifest_audit_statuses_and_required_gate():
    m = _toy_manifest()
    no_pilot = L.type_sm_from_manifest(m)
    assert no_pilot['status'] == 'not_verifiable', no_pilot
    assert no_pilot['pilot_status'] == 'not_available', no_pilot
    assert all(c['status'] == 'not_verifiable'
               for c in no_pilot['comparisons'].values()), no_pilot

    strong = L.type_sm_from_manifest(m, _pilot(_strong_deltas()))
    assert strong['status'] == 'ok', strong
    assert all(c['status'] == 'ok'
               for c in strong['comparisons'].values()), strong

    failed_pilot = _pilot(_strong_deltas())
    failed_pilot['status'] = 'fail'
    failed = L.type_sm_from_manifest(m, failed_pilot)
    assert failed['status'] == 'warnings', failed
    assert any('pilot reports status' in w for w in failed['warnings']), failed

    weak = L.type_sm_from_manifest(m, _pilot(_small_deltas(0.1)))
    assert weak['status'] == 'warnings', weak
    assert all(c['status'] == 'warnings'
               for c in weak['comparisons'].values()), weak
    assert any('Type M' in w or 'planning thresholds' in w
               for w in weak['warnings']), weak

    required = L.type_sm_from_manifest(
        _toy_manifest(require=True), _pilot(_small_deltas(0.1)))
    assert required['status'] == 'fail', required
    assert required['violations'], required

    # A required threshold cannot be satisfied by a missing pilot.
    required_missing = L.type_sm_from_manifest(_toy_manifest(require=True))
    assert required_missing['status'] == 'fail', required_missing

    # Alpha/family mismatches are binding: a pilot from another family cannot
    # silently anchor this manifest.
    wrong_alpha = L.type_sm_from_manifest(
        m, _pilot(_strong_deltas(), alpha=0.01))
    assert wrong_alpha['status'] == 'fail', wrong_alpha
    wrong_m = L.type_sm_from_manifest(
        m, _pilot(_strong_deltas(), m_family=1))
    assert wrong_m['status'] == 'fail', wrong_m

    # A pilot bound to another manifest hash cannot anchor this freeze.
    m_hash = _toy_manifest()
    m_hash['manifest_sha256'] = 'a' * 64
    p_hash = _pilot(_strong_deltas())
    p_hash['manifest_sha256'] = 'b' * 64
    assert L.type_sm_from_manifest(m_hash, p_hash)['status'] == 'fail'
    p_hash['manifest_sha256'] = 'a' * 64
    assert L.type_sm_from_manifest(m_hash, p_hash)['status'] == 'ok'

    # A pilot that omits one comparison is not silently complete.
    partial = _pilot(_strong_deltas())
    del partial['comparisons']['c2']
    partial_out = L.type_sm_from_manifest(m, partial)
    assert partial_out['status'] == 'warnings', partial_out
    assert partial_out['comparisons']['c2']['status'] == 'not_verifiable', partial_out


def test_gate_cli_report_and_plan_wiring():
    m = _toy_manifest()
    pilot = _pilot(_strong_deltas())
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, 'manifest.json')
        pp = os.path.join(td, 'pilot.json')
        with open(mp, 'w', encoding='utf-8') as fh:
            json.dump(m, fh)
        with open(pp, 'w', encoding='utf-8') as fh:
            json.dump(pilot, fh)
        audit = A.audit_type_sm(mp, pp, verbose=False)
        assert audit['status'] == 'ok', audit
        assert audit['manifest'] == mp and audit['pilot'] == pp, audit
        missing = A.audit_type_sm(mp, os.path.join(td, 'none.json'),
                                 verbose=False)
        assert missing['status'] == 'not_verifiable', missing
    assert hasattr(A, 'audit_type_sm_artifact'), 'artifact replay gate is not exported'
    assert "'type_sm'" in inspect.getsource(A.run_report), (
        'run_report does not register the type_sm block')
    source = inspect.getsource(A._run_cli)
    assert 'type_sm' in source and 'sign_magnitude' in source, source
    makefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'Makefile')
    text = open(makefile, encoding='utf-8').read()
    assert 'type-sm:' in text and 'audit_gates.py type_sm' in text, text
    plan = V12.plan_check()
    assert plan['type_sm']['kind'] == 'type_sm_audit', plan['type_sm']
    assert plan['type_sm']['status'] != 'fail', plan['type_sm']
