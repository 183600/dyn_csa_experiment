"""Regression checks for the prospective group-sequential design fragment.

The exact sign-test plan is a design gate, so two properties matter:

* the reported tail probabilities must be exact for the actual parity of the
  sign sum (a boundary belongs to the attainable lattice W = -n, -n+2, ...);
* an untrusted, edited artifact or declaration must fail closed with a
  strict-JSON diagnostic, including when a scale becomes large enough to hit
  CPython's integer-to-decimal conversion limit or a declared look count is
  enormous.

These are CPU-only, zero-experiment checks.
"""
import copy
import json
import math
import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


HUGE = 10 ** 10000


def _exact_tail(n, b, sides):
    total = 0
    for w in range(-n, n + 1, 2):
        hit = ((sides == 'two' and abs(w) >= b)
               or (sides == 'greater' and w >= b)
               or (sides == 'less' and w <= -b))
        if hit:
            total += math.comb(n, (n + w) // 2)
    return float(Fraction(total, 1 << n))


def test_nominal_tail_probability_matches_the_attainable_parity():
    for n in range(1, 41):
        for b in range(1, n + 2):
            for sides in ('two', 'greater', 'less'):
                got = L._seq_tail_prob(n, b, sides)
                want = _exact_tail(n, b, sides)
                assert got == pytest.approx(want, abs=1e-15), (n, b, sides)


def test_tail_probability_handles_zero_and_negative_thresholds():
    # A one-sided threshold at 0 is ~1/2, not the two-sided |W|>=0 = 1.
    for n in range(1, 21):
        for b in range(-n - 1, 1):
            for sides in ('two', 'greater', 'less'):
                got = L._seq_tail_prob(n, b, sides)
                want = _exact_tail(n, b, sides)
                assert got == pytest.approx(want, abs=1e-15), (n, b, sides)


def test_boundary_reports_exact_nominal_p_even_for_parity_mismatched_b():
    # n=4 has W in {-4,-2,0,2,4}; the smallest integer boundary that fits
    # alpha=0.2 is b=3, whose exact two-sided tail is 2/16 (W=+-4), not the
    # 8/16 that a naive even-stride walk from b=3 would produce.
    plan = L.sequential_boundaries([4], alpha=0.2, spending='none')
    row = plan['boundaries'][0]
    assert row['b'] == 3
    assert row['nominal_p_boundary'] == pytest.approx(2 / 16, abs=1e-15)
    assert plan['fwer'] == pytest.approx(2 / 16, abs=1e-15)


def test_tampered_artifact_scalars_fail_closed_and_stay_strict_json():
    artifact = L.build_sequential_design(
        n_max=12, look_sizes=[6, 12], alpha=0.2,
        pilot_deltas=[0.1] * 8 + [-0.1] * 4, pilot_ppl=30.0)
    assert L.validate_sequential_design(artifact)['status'] == 'ok'

    cases = [
        (('kind',), HUGE),
        (('schema',), HUGE),
        (('result', 'status'), HUGE),
        (('result', 'status'), 'tampered'),
        (('result', 'power'), HUGE),
        (('result', 'power'), '0.9'),
        (('result', 'n_max'), HUGE),
        (('result', 'n_max'), 11),
        (('result', 'look_sizes'), [6]),
        (('result', 'look_sizes'), [6, HUGE]),
        (('result', 'boundaries'), []),
        (('result', 'boundaries'),
         [{'n': 6, 'b': HUGE}, {'n': 12, 'b': 5}]),
        (('result', 'fwer'), HUGE),
        (('result', 'fwer'), 'x'),
        (('inputs', 'n_max'), HUGE),
        (('inputs', 'n_max'), 11),
        (('inputs', 'alpha'), 'x'),
        (('inputs', 'alpha'), 0.3),
        (('revision',), HUGE),
        (('source',), HUGE),
        (('created_at',), HUGE),
        (('supersedes',), [HUGE]),
        (('reason',), HUGE),
    ]
    for path, value in cases:
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        bad['content_sha256'] = L._seq_content_sha(bad)
        verdict = L.validate_sequential_design(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        json.dumps(verdict, allow_nan=False)

    # Huge integers used as mapping keys/paths are diagnostics, not crashes.
    for where in ('artifact', 'inputs'):
        bad = copy.deepcopy(artifact)
        if where == 'artifact':
            bad[HUGE] = {'x': 1}
        else:
            bad['inputs'][HUGE] = 1
        bad['content_sha256'] = L._seq_content_sha(bad)
        verdict = L.validate_sequential_design(bad, replay=False)
        assert verdict['status'] == 'fail', (where, verdict)
        json.dumps(verdict, allow_nan=False)
        assert any('unrepresentable int' in problem
                   for problem in verdict['problems']), (where, verdict)

    # A huge kind/schema/status is named with the unrepresentable-int
    # fallback rather than raising while formatting the diagnostic.
    for path in (('kind',), ('schema',), ('result', 'status')):
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = HUGE
        bad['content_sha256'] = L._seq_content_sha(bad)
        verdict = L.validate_sequential_design(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert any('unrepresentable int' in problem
                   for problem in verdict['problems']), (path, verdict)


def test_huge_declared_look_count_fails_before_materialising_the_schedule():
    manifest = {
        'seeds': list(range(12)),
        'statistics': {'sequential': {'n_looks': HUGE}},
    }
    with pytest.raises(ValueError, match='look cap'):
        L.sequential_design_from_manifest(manifest)

    protocol = {
        'sample_size': {'planned_seeds': 12},
        'stopping': {'sequential': {'n_looks': HUGE}},
    }
    with pytest.raises(ValueError, match='look cap'):
        L.sequential_design_from_protocol(protocol)


def test_conditional_power_rejects_out_of_range_and_oversized_boundaries():
    plan = L.sequential_boundaries([6, 12], alpha=0.2, spending='linear')
    with pytest.raises(ValueError):
        L.sequential_conditional_power(
            plan, look=1, observed_w=HUGE, q=0.7)
    bad = copy.deepcopy(plan['boundaries'])
    bad[1]['n'] = HUGE
    with pytest.raises(ValueError):
        L.sequential_conditional_power(
            bad, look=1, observed_w=0, q=0.7)
    with pytest.raises(ValueError):
        L.sequential_conditional_power(
            plan['boundaries'] * 4, look=1, observed_w=0, q=0.7)


def test_extreme_or_degenerate_pilot_spread_is_withheld_not_raised():
    # statistics.stdev([1.7e308, -1.7e308]) raises OverflowError even though
    # both deltas are finite; the design must still replay with no fake MDE.
    summary = L._seq_pilot_summary([1.7e308, -1.7e308])
    assert summary['pilot_sd'] is None
    assert summary['pilot_sd_withheld'] is True
    assert L._seq_pilot_summary([1e308, 1e308])['pilot_sd'] is None

    for deltas in ([1.7e308, -1.7e308], [1e308, 1e308]):
        artifact = L.build_sequential_design(
            n_max=12, look_sizes=[12], pilot_deltas=deltas, pilot_ppl=30.0)
        assert artifact['result']['mde']['log_ppl'] is None
        assert artifact['result']['pilot']['pilot_sd'] is None
        assert any('no usable pilot SD' in warning
                   for warning in artifact['result']['warnings'])
        verdict = L.validate_sequential_design(artifact)
        assert verdict['status'] == 'ok', verdict
        json.dumps(artifact, allow_nan=False)
        json.dumps(verdict, allow_nan=False)

    # A representable large spread is still reported, not discarded.
    finite = L.build_sequential_design(
        n_max=12, look_sizes=[12], pilot_deltas=[1e308, 1.5e308])
    assert finite['result']['pilot']['pilot_sd'] is not None
    assert finite['result']['mde']['log_ppl'] is not None
    assert L.validate_sequential_design(finite)['status'] == 'ok'


def test_oversized_power_curve_is_refused_before_enumeration():
    with pytest.raises(ValueError, match='supported cap'):
        L.build_sequential_design(
            n_max=12, look_sizes=[12],
            power_curve_q=[0.5] * (L.SEQUENTIAL_MAX_POWER_CURVE + 1))


def test_audit_unusable_path_values_fail_closed_and_stay_strict_json():
    for kwargs in (
            {'path': HUGE},
            {'manifest_path': HUGE},
            {'protocol_path': HUGE},
            {'look_history_path': HUGE},
            {'events_path': HUGE}):
        report = L.sequential_design_audit(**kwargs)
        assert report['status'] == 'fail', (kwargs, report)
        assert report['problems'], (kwargs, report)
        json.dumps(report, allow_nan=False)
        assert any('unrepresentable int' in problem
                   for problem in report['problems']), (kwargs, report)


def test_large_but_admissible_plan_remains_exact_and_bounded():
    looks = [32 * i for i in range(1, 33)]
    plan = L.sequential_boundaries(looks, alpha=0.05,
                                   spending='linear')
    assert plan['status'] == 'ok'
    assert plan['n_max'] == 1024
    assert plan['n_looks'] == 32
    assert plan['fwer_ok'] is True
    assert plan['fwer'] <= 0.05 + 1e-15
    for row in plan['boundaries']:
        assert isinstance(row['b'], int) and 1 <= row['b'] <= row['n']
        assert 0.0 < row['nominal_p_boundary'] <= 1.0
        # The reported p is an exact attainable-tail probability.
        assert row['nominal_p_boundary'] == pytest.approx(
            _exact_tail(row['n'], row['b'], 'two'), abs=1e-15)
