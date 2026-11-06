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


def test_skewed_token_weights_cannot_violate_min_units_per_half():
    """A dominant unit must not collapse one half below the declared floor.

    The old assignment only checked ``len(units) >= 2 * min_units_per_half``
    before the greedy balance.  With weights ``[1000, 1, 1, 1]`` the other
    three units all chase the dominant one, leaving a 3/1 split and a
    half with fewer units than requested.  At scale, one very long document
    can do the same to a much larger panel.  The repair moves the lightest
    units, never the dominant one, so the floor is met without rejecting the
    panel.
    """
    units = ['u0', 'u1', 'u2', 'u3']
    assignment = L.unit_split_assignment(
        units, [1000.0, 1.0, 1.0, 1.0], salt='x', min_units_per_half=2)
    counts = [h['n_units'] for h in assignment['halves']]
    assert min(counts) >= 2, assignment
    assert sorted(assignment['halves'][0]['indices']
                  + assignment['halves'][1]['indices']) == list(range(4))
    _strict_json(assignment)

    result = L.unit_split_replication(
        [[0.0] * 4 for _ in SEEDS], [1000.0, 1.0, 1.0, 1.0],
        unit_ids=units, seeds=SEEDS, direction='negative',
        min_units=4, min_units_per_half=2, salt='x', n_salts=0)
    assert result['status'] in ('ok', 'warnings'), result
    assert all(h['n_units'] >= 2 for h in result['assignment']['halves'])
    _strict_json(result)


def test_discordant_halves_survive_subnormal_effect_underflow():
    """Tiny opposite half effects are discordant, not a zero half.

    ``mean0 * mean1`` underflows to ``-0.0`` when the two means are around
    +/-1e-200, so the product test called a genuinely opposite pair
    ``zero_in_one_half``.  Comparing the two signs directly is exact at any
    representable magnitude.
    """
    assignment = L.unit_split_assignment(UNITS, TOKENS)
    i0 = assignment['halves'][0]['indices']
    i1 = assignment['halves'][1]['indices']
    row = [0.0] * len(UNITS)
    for i in i0:
        row[i] = -1e-200
    for i in i1:
        row[i] = 1e-200
    result = L.unit_split_replication(
        [row[:] for _ in SEEDS], TOKENS, unit_ids=UNITS, seeds=SEEDS,
        direction='negative', n_salts=0)
    assert result['status'] in ('ok', 'warnings'), result
    assert result['halves'][0]['stats']['mean'] < 0.0
    assert result['halves'][1]['stats']['mean'] > 0.0
    assert result['sign_agreement'] == 'discordant', result
    assert result['iut']['claim'] == 'halves_discordant', result
    _strict_json(result)


def test_stamped_writer_preserves_unreadable_or_nonobject_artifacts(tmp_path):
    """A corrupt existing artifact must conflict, not be silently replaced.

    ``json.load`` can fail on a crash-truncated file, very deep JSON, or a
    multi-GB archive; valid-but-non-object JSON also has no comparable content
    digest.  Previously all of those looked like an absent path to the writer,
    so ``overwrite=False`` destroyed the previous bytes.
    """
    artifact = L.build_unit_split_replication(
        _rows(), TOKENS, unit_ids=UNITS, seeds=SEEDS, direction='negative',
        n_salts=0)
    path = tmp_path / 'unit_split.json'
    assert L.write_unit_split_replication(path, artifact)['status'] == 'written'

    for body in ('{truncated', '[1, 2, 3]', 'true'):
        path.write_text(body, encoding='utf-8')
        result = L.write_unit_split_replication(
            path, artifact, overwrite=False)
        assert result['status'] == 'conflict', (body, result)
        assert path.read_text(encoding='utf-8') == body, body
        assert L.write_unit_split_replication(
            path, artifact, overwrite=True)['status'] == 'written'
        _strict_json(json.loads(path.read_text(encoding='utf-8')))

    # Two objects with no content digest are not the same artifact.  The old
    # ``None == None`` shortcut skipped the write and reported a no-op.
    no_hash = copy.deepcopy(artifact)
    no_hash.pop('content_sha256', None)
    path.write_text('{"other": 1}', encoding='utf-8')
    result = L._write_stamped_artifact(
        path, no_hash, validator=lambda _a: {'status': 'ok'},
        name='unit_split.json')
    assert result['status'] == 'conflict', result
    assert path.read_text(encoding='utf-8') == '{"other": 1}'
