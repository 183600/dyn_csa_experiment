"""Scale/tamper regressions for the split-half consistency fragment.

The split-half readout is O(seeds * units) and an artifact is untrusted JSON.
Two properties therefore matter at scale:

* finite per-unit effects/token counts near float64's limit must not turn a
  representable weighted mean into ``inf``/``nan``;
* an edited archive must fail closed with a strict-JSON diagnostic, including
  CPython's integer-to-decimal conversion limit and huge mapping keys.

These are CPU-only, zero-experiment checks.
"""
import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
HUGE = 10 ** 10000
UNITS = [f'u{i:03d}' for i in range(16)]
TOKENS = [100.0 + 7.0 * i for i in range(16)]
SEEDS = list(range(12))


def _rows(scale=1.0):
    return [[scale * (-0.20 + 0.01 * math.sin(i + k)) for i in range(16)]
            for k in range(12)]


def _strict_json(value):
    json.dumps(value, allow_nan=False)
    return value


def test_huge_effects_and_token_totals_stay_representable_and_replay():
    for scale in (1e100, 1e200, 1e300, 1.7e308):
        artifact = L.build_unit_split_replication(
            _rows(scale), TOKENS, unit_ids=UNITS, seeds=SEEDS,
            direction='negative', n_salts=0)
        result = artifact['result']
        assert result['status'] in ('ok', 'warnings'), (scale, result)
        for half in result['halves']:
            for key in ('mean', 'sd', 'sem'):
                value = half['stats'][key]
                assert value is None or math.isfinite(value), (scale, key, value)
        assert result['iut']['p_iut'] == pytest.approx(max(
            result['halves'][0]['test']['p_one_sided'],
            result['halves'][1]['test']['p_one_sided']))
        for key in ('mean', 'sd', 'sem'):
            value = result['half_difference'][key]
            assert value is None or math.isfinite(value), (scale, key)
        _strict_json(artifact)
        verdict = L.validate_unit_split_replication(artifact)
        assert verdict['status'] in ('ok', 'warnings'), (scale, verdict)
        assert verdict['problems'] == [], (scale, verdict)

    # Per-unit counts are finite individually but their total overflows.
    huge_tokens = [1e308] * 16
    assignment = L.unit_split_assignment(UNITS, huge_tokens)
    assert assignment['n_units'] == 16
    assert all(h['n_tokens'] is None for h in assignment['halves'])
    shares = [h['token_share'] for h in assignment['halves']]
    assert all(math.isfinite(s) for s in shares)
    assert sum(shares) == pytest.approx(1.0)
    _strict_json(assignment)

    artifact = L.build_unit_split_replication(
        _rows(), huge_tokens, unit_ids=UNITS, seeds=SEEDS,
        direction='negative', n_salts=0)
    assert artifact['result']['token_total'] is None
    assert artifact['result']['status'] == 'warnings'
    _strict_json(artifact)
    verdict = L.validate_unit_split_replication(artifact)
    assert verdict['status'] == 'warnings', verdict
    assert verdict['problems'] == [], verdict


def test_tampered_inputs_and_results_fail_closed_without_replay():
    artifact = L.build_unit_split_replication(
        _rows(), TOKENS, unit_ids=UNITS, seeds=SEEDS,
        direction='negative', n_salts=2)
    assert L.validate_unit_split_replication(artifact)['status'] == 'ok'

    cases = [
        (('inputs', 'alpha'), 0.3),
        (('inputs', 'min_token_share'), 0.4),
        (('inputs', 'salt'), 'tampered'),
        (('inputs', 'tokens'), [1.0] * 16),
        (('inputs', 'unit_ids'), [f'x{i}' for i in range(16)]),
        (('inputs', 'log_ratios'), [[None] * 16] * 12),
        (('result', 'n_units'), 99),
        (('result', 'status'), 'tampered'),
        (('result', 'alpha'), 0.3),
        (('result', 'iut', 'claim'), 'tampered'),
        (('result', 'iut', 'reject'), False),
        (('result', 'seeds'), [HUGE] + SEEDS[1:]),
        (('result', 'split_sensitivity', 'n_pass'), 999),
    ]
    for path, value in cases:
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        bad['content_sha256'] = L._esci_content_sha(bad)
        verdict = L.validate_unit_split_replication(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        _strict_json(verdict)

    # Huge integer keys are diagnostics, not crashes.
    for where in ('artifact', 'inputs', 'result'):
        bad = copy.deepcopy(artifact)
        if where == 'artifact':
            bad[HUGE] = {'x': 1}
        elif where == 'inputs':
            bad['inputs'][HUGE] = 1
        else:
            bad['result'][HUGE] = 1
        bad['content_sha256'] = L._esci_content_sha(bad)
        verdict = L.validate_unit_split_replication(bad, replay=False)
        assert verdict['status'] == 'fail', (where, verdict)
        assert any('unrepresentable int' in p for p in verdict['problems']), (
            where, verdict)
        _strict_json(verdict)


def test_huge_knobs_and_seeds_fail_closed_with_bounded_diagnostics():
    for key in ('max_n', 'min_n', 'min_units', 'min_units_per_half',
                'n_salts'):
        kwargs = {key: HUGE}
        if key == 'n_salts':
            kwargs['n_salts'] = HUGE
        else:
            kwargs['n_salts'] = 0
        with pytest.raises(ValueError, match=key):
            L.unit_split_replication(_rows(), TOKENS, unit_ids=UNITS,
                                     seeds=SEEDS, direction='negative',
                                     **kwargs)
    with pytest.raises(ValueError, match='seeds'):
        L.unit_split_replication(_rows(), TOKENS, unit_ids=UNITS,
                                 seeds=[HUGE] + SEEDS[1:],
                                 direction='negative', n_salts=0)

    # Provenance arguments are untrusted too; each is bounded text, not an
    # integer-formatting crash.
    artifact = L.build_unit_split_replication(
        _rows(), TOKENS, unit_ids=UNITS, seeds=SEEDS, direction='negative',
        n_salts=0, created_at=HUGE, source=HUGE)
    assert isinstance(artifact['created_at'], str)
    assert isinstance(artifact['source'], str)
    _strict_json(artifact)
    assert L.validate_unit_split_replication(artifact)['status'] in (
        'ok', 'warnings')


def test_audit_gate_unusable_path_fails_closed_and_stays_strict_json():
    report = A.audit_unit_split_replication(HUGE, verbose=False)
    assert report['status'] == 'fail', report
    assert report['problems'], report
    assert any('unrepresentable int' in problem
               for problem in report['problems']), report
    _strict_json(report)

    report = A.audit_unit_split_replication(3.14, verbose=False)
    assert report['status'] == 'fail', report
    assert report['problems'], report
    _strict_json(report)


def test_empty_unit_panel_builds_a_not_estimable_artifact():
    artifact = L.build_unit_split_replication(
        [[]], [], direction='negative', seeds=[0], n_salts=0)
    assert artifact['result']['status'] == 'not_estimable'
    for replay in (False, True):
        verdict = L.validate_unit_split_replication(artifact, replay=replay)
        assert verdict['status'] == 'not_estimable', (replay, verdict)
        assert verdict['problems'] == [], (replay, verdict)
        _strict_json(verdict)
