"""CPU checks for design-stage 2x2 interaction power (part 39).

The existing interaction SESOI/TOST layer can separate practical additivity
from "not detected", but it cannot rescue a panel whose seed count is below
the exact sign-flip resolution floor.  These checks pin the interaction
estimand, the family-Holm resolution limit, the SESOI power/MDE readout, the
fail-closed input handling and the audit/CLI wiring.  No test reads the
held-out split or changes a frozen manifest.
"""
import contextlib
import inspect
import io
import json
import math
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _cells_from_deltas(deltas, *, base=50.0):
    """Exact 2x2 cells whose log ratio-of-ratios equals ``deltas`` per seed.

    Setting 10 = 01 = 00 = base and 11 = base*exp(d) gives
    log(11/10) - log(01/00) = d, so the test does not depend on rounding in
    four separate PPL values.
    """
    cells = {k: [] for k in L.FACTORIAL_POWER_CELL_KEYS}
    for i, d in enumerate(deltas):
        b = float(base) + 0.1 * i
        cells['11'].append(b * math.exp(float(d)))
        cells['10'].append(b)
        cells['01'].append(b)
        cells['00'].append(b)
    return cells


_STRONG = [0.030, 0.028, 0.032, 0.029, 0.031, 0.027, 0.033, 0.030]
_WEAK = [0.001 + x for x in
         (0.000, 0.020, -0.010, 0.015, -0.020, 0.010, -0.005, 0.000)]


def test_interaction_estimand_and_family_resolution_floor():
    r = L.factorial_interaction_power(
        _cells_from_deltas(_STRONG), m_family=5, relative_pct=1.0,
        expected_sign=1)
    assert r['status'] == 'ok', r
    assert r['n_seeds'] == 8
    assert np.allclose(r['interaction']['deltas'], _STRONG, atol=1e-12)
    # With n=8 and m=5 the exact two-sided Holm floor is 5*2/2**8.
    assert math.isclose(r['resolution']['p_floor'], 2.0 / 2 ** 8,
                        rel_tol=1e-15)
    assert math.isclose(r['resolution']['min_attainable_p_holm'],
                        5.0 * 2.0 / 2 ** 8, rel_tol=1e-15)
    assert r['resolution']['resolution_ok'] is True
    assert r['resolution']['min_n_for_alpha'] == 8
    # The per-interaction power must be the same planning projection the
    # frozen sensitivity gate uses, with the SESOI converted from percent to
    # the natural-log ratio-of-ratios scale.
    base = L.paired_design_sensitivity(
        8, 5, alpha=0.05, target_power=0.80, pilot_deltas=_STRONG,
        expected_sign='positive', sesoi=math.log1p(0.01))
    assert math.isclose(r['sensitivity']['pilot']['mean'],
                        base['pilot']['mean'], rel_tol=1e-12)
    assert math.isclose(r['sensitivity']['sesoi']['power_at_sesoi'],
                        base['sesoi']['power_at_sesoi'], rel_tol=1e-12)
    assert r['mde_relative_pct'] is not None and r['mde_relative_pct'] > 0.0


def test_below_family_floor_is_resolution_limited_not_a_zero_interaction():
    # n=6, m=5: the smallest attainable Holm p is 0.15625, so no effect size
    # can reject.  This is a design limitation, not evidence of additivity.
    for n in (5, 6, 7):
        r = L.factorial_interaction_power(
            _cells_from_deltas(_STRONG[:n]), m_family=5, relative_pct=1.0,
            expected_sign=1)
        assert r['status'] == 'resolution_limited', (n, r)
        assert r['resolution']['resolution_ok'] is False
        assert r['resolution']['min_n_for_alpha'] == 8
        assert r['resolution']['min_attainable_p_holm'] > 0.05
        assert 'no effect size' in r['note']
    a = L.factorial_power_audit(n_seeds=6)
    assert a['status'] == 'warnings', a
    assert a['family_resolution']['min_n_for_alpha'] == 8
    assert len(a['resolution_limited']) == len(L.FACTORIAL_DESIGN)
    # At n=8 the family is mathematically resolvable, but without pilot cells
    # the power/SESOI readout is still unverified (never a green pass).
    b = L.factorial_power_audit(n_seeds=8)
    assert b['status'] == 'warnings', b
    assert b['resolution_limited'] == []
    assert set(b['not_measured']) == set(L.FACTORIAL_DESIGN)
    # Neither a pilot nor a declared n is not a warning about a design; it is
    # the absence of a verifiable design readout.
    c = L.factorial_power_audit()
    assert c['status'] == 'not_verifiable_locally', c


def test_sesoi_power_mde_and_required_n_are_reported_and_gated():
    r = L.factorial_interaction_power(
        _cells_from_deltas(_WEAK), m_family=5, relative_pct=1.0,
        expected_sign=1)
    assert r['status'] == 'underpowered', r
    ses = r['sensitivity']['sesoi']
    assert ses['status'] == 'underpowered'
    assert ses['power_at_sesoi'] < 0.80
    assert ses['required_n_for_target_power'] > 8
    assert r['mde_relative_pct'] > 1.0
    assert any('cannot resolve the margin' in w for w in r['warnings'])
    pilot = {'factorials': {
        'dynamic_blocking_x_learned_selection': {
            'source': 'synthetic validation pilot',
            'cells': _cells_from_deltas(_WEAK),
            'relative_pct': 1.0,
        }}}
    a = L.factorial_power_audit(pilot, n_seeds=8)
    assert a['status'] == 'warnings', a
    assert 'dynamic_blocking_x_learned_selection' in a['underpowered']
    strict = L.factorial_power_audit(pilot, n_seeds=8, require_power=True)
    assert strict['status'] == 'fail', strict
    assert any('require_power=true' in v for v in strict['violations'])


def test_direction_and_ppl_scale_do_not_change_the_interaction_power():
    cells = _cells_from_deltas(_STRONG)
    mirrored = dict(cells)
    mirrored['11'], mirrored['10'] = list(cells['10']), list(cells['11'])
    pos = L.factorial_interaction_power(
        cells, m_family=5, relative_pct=1.0, expected_sign=1)
    neg = L.factorial_interaction_power(
        mirrored, m_family=5, relative_pct=1.0, expected_sign=-1)
    assert pos['status'] == neg['status'] == 'ok'
    assert math.isclose(pos['sensitivity']['sesoi']['power_at_sesoi'],
                        neg['sensitivity']['sesoi']['power_at_sesoi'],
                        rel_tol=1e-12)
    scaled = {k: [2.0 * x for x in v] for k, v in cells.items()}
    sc = L.factorial_interaction_power(
        scaled, m_family=5, relative_pct=1.0, expected_sign=1)
    assert np.allclose(sc['interaction']['deltas'], pos['interaction']['deltas'],
                       atol=1e-12)
    assert math.isclose(sc['sensitivity']['sesoi']['power_at_sesoi'],
                        pos['sensitivity']['sesoi']['power_at_sesoi'],
                        rel_tol=1e-12)


def test_malformed_interaction_inputs_fail_closed():
    good = _cells_from_deltas(_STRONG)
    bad_cases = []
    missing = dict(good)
    missing.pop('00')
    bad_cases.append(missing)
    extra = dict(good)
    extra['99'] = [1.0] * 8
    bad_cases.append(extra)
    ragged = dict(good)
    ragged['11'] = list(good['11'])[:-1]
    bad_cases.append(ragged)
    mapping = dict(good)
    mapping['11'] = {'seed0': 50.0, 'seed1': 51.0}
    bad_cases.append(mapping)
    for key in ('11', '10', '01', '00'):
        for bad in (0.0, -1.0, float('nan'), float('inf'), True, '50'):
            case = {k: list(v) for k, v in good.items()}
            case[key][0] = bad
            bad_cases.append(case)
    for case in bad_cases:
        r = L.factorial_interaction_power(
            case, m_family=5, relative_pct=1.0, expected_sign=1)
        assert r['status'] == 'fail', (case, r)
        assert r['violations']
    assert L.factorial_interaction_power(
        good, n_seeds=7, m_family=5, relative_pct=1.0,
        expected_sign=1)['status'] == 'fail'
    for kwargs in ({'m_family': 0}, {'m_family': True}, {'alpha': 1.2},
                   {'alpha': 0.0}, {'target_power': 0}, {'target_power': 1.0},
                   {'relative_pct': 0.0}, {'relative_pct': -1.0},
                   {'expected_sign': 2}):
        args = dict(m_family=5, relative_pct=1.0, expected_sign=1)
        args.update(kwargs)
        r = L.factorial_interaction_power(good, **args)
        assert r['status'] == 'fail', (kwargs, r)
    assert L.factorial_interaction_power(
        {'11': [1.0], '10': [1.0], '01': [1.0], '00': [1.0]},
        m_family=5, relative_pct=1.0, expected_sign=1)['status'] == 'fail'
    assert L.factorial_power_audit('not-an-object')['status'] == 'fail'
    assert L.factorial_power_audit(design={})['status'] == 'fail'
    # A pilot name that is not in the declared design is visible, not fatal.
    extra_pilot = {'factorials': {
        'not_a_factorial': {'cells': _cells_from_deltas(_STRONG)}}}
    ep = L.factorial_power_audit(extra_pilot, n_seeds=8)
    assert ep['status'] in ('warnings', 'not_verifiable_locally')
    assert any('not present in the design' in w for w in ep['warnings'])


def test_attrition_projection_is_optional_and_complete_coverage_aware():
    r = L.factorial_interaction_power(
        _cells_from_deltas(_STRONG), m_family=5, relative_pct=1.0,
        expected_sign=1, pair_retention=0.95)
    att = r['attrition']
    assert att['status'] != 'not_verifiable', att
    assert att['require_complete_coverage'] is True
    assert 0.0 <= att['unconditional_power'] <= att['conditional_power'] + 1e-12
    assert 0.0 < att['probability_runnable'] < 1.0
    bad = L.factorial_interaction_power(
        _cells_from_deltas(_STRONG), m_family=5, relative_pct=1.0,
        expected_sign=1, pair_retention=1.5)
    assert bad['status'] == 'fail'


def test_audit_gate_cli_and_report_are_wired(tmp_path):
    pilot = {'kind': 'factorial_power_pilot', 'factorials': {
        name: {'source': 'synthetic validation pilot',
               'cells': _cells_from_deltas(_STRONG), 'relative_pct': 1.0}
        for name in L.FACTORIAL_DESIGN}}
    p = tmp_path / 'factorial_power_pilot.json'
    p.write_text(json.dumps(pilot), encoding='utf-8')
    ok = A.audit_interaction_power(str(p))
    assert ok['status'] == 'ok', ok
    assert ok['pilot_available'] is True
    assert len(ok['powered']) == len(L.FACTORIAL_DESIGN)
    strict = A.audit_interaction_power(str(p), require_power=True)
    assert strict['status'] == 'ok'
    missing = A.audit_interaction_power(str(tmp_path / 'absent.json'))
    assert missing['status'] == 'not_verifiable_locally'
    assert missing['pilot_available'] is False
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    broken = A.audit_interaction_power(str(bad))
    assert broken['status'] == 'not_verifiable_locally'
    assert any('not readable JSON' in w for w in broken['warnings'])
    cli = inspect.getsource(A._run_cli)
    for alias in ('interaction_power', 'factorial_power', 'power_2x2'):
        assert alias in cli
    assert 'interaction_power' in inspect.getsource(A.run_report)
    with contextlib.redirect_stdout(io.StringIO()):
        dp = A.audit_design_preflight()
    ip = dp.get('interaction_power') or {}
    assert (ip.get('family_resolution') or {}).get('min_n_for_alpha') == 8, ip
    assert any('at least 8 seeds' in w for w in dp['warnings']), dp['warnings']
