"""Replay/report paths must fail closed on exact integers beyond float64.

``10**400`` is a finite Python integer but has no float64 spelling.  A stored
measurement of that size is malformed for a float-precision audit, so the
audit must report a failure or withhold the field -- never abort with
``OverflowError`` while merely validating the artifact.
"""
import json

import exp_lib as L


HUGE = 10 ** 400


def test_arm_rank_unit_audit_fails_closed_on_huge_endpoint_value(tmp_path, capsys):
    import audit_gates as A

    arm_units = {
        'a': {s: {'nll': [1.0, 2.0, 3.0, 4.0], 'tokens': [10] * 4}
              for s in range(5)},
        'b': {s: {'nll': [2.0, 2.0, 3.0, 4.0], 'tokens': [10] * 4}
              for s in range(5)},
    }
    block = L.arm_rank_unit_sensitivity(
        arm_units, n_boot=10, min_seeds=5, min_units=2)
    block['per_arm_seed_values'] = {
        arm: {str(s): (HUGE if arm == 'a' and s == 0 else 1.0)
              for s in range(5)}
        for arm in ('a', 'b')}
    block['endpoint_values_match'] = 'checked'
    path = tmp_path / 'test_stats.json'
    path.write_text(json.dumps({'finaltest': {'arm_rank_units': block}}),
                    encoding='utf-8')

    out = A.audit_arm_rank_units(str(path))
    capsys.readouterr()
    assert out['status'] == 'fail'
    assert out['problems']


def test_winner_curse_audit_fails_closed_on_huge_policy_values(tmp_path, capsys):
    import audit_gates as A

    values = {arm: {seed: float(seed + ord(arm)) for seed in range(5)}
              for arm in ('a', 'b', 'c')}
    block = L.winner_curse_correction(
        values, n_boot=200, split_seed=1, max_partitions=16, min_seeds=5)
    assert block['status'] == 'selection_robust'

    for mutate in ('tie_tol', 'gap'):
        stored = json.loads(json.dumps(block))
        if mutate == 'tie_tol':
            stored['policy']['tie_tol'] = HUGE
        else:
            stored['observed_gap'] = HUGE
        path = tmp_path / f'{mutate}.json'
        path.write_text(json.dumps({'finaltest': {'winner_curse': stored}}),
                        encoding='utf-8')
        out = A.audit_winner_curse(str(path))
        capsys.readouterr()
        assert out['status'] == 'fail'
        assert out['problems']


def test_effect_sensitivity_audit_fails_closed_on_huge_ppl(tmp_path, capsys):
    import audit_gates as A

    block = L.effect_sensitivity(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        [1.2, 1.3, 1.4, 1.5, 1.6, 1.7], seeds=list(range(6)))
    block['per_seed'][0]['with'] = HUGE
    path = tmp_path / 'effect.json'
    path.write_text(
        json.dumps({'finaltest': {'comparisons': {
            'c': {'effect_sensitivity': block}}}}),
        encoding='utf-8')

    out = A.audit_effect_sensitivity(str(path))
    capsys.readouterr()
    assert out['status'] == 'fail'
    assert out['problems']


def test_unit_quantile_audit_fails_closed_on_huge_nll_difference(
        tmp_path, capsys):
    import audit_gates as A

    n_units = 8
    rows = [[1.0 + 0.1 * seed + 0.01 * unit for unit in range(n_units)]
            for seed in range(6)]
    block = L.unit_quantile_profile(
        rows, [10] * n_units,
        unit_ids=[f'u{unit}' for unit in range(n_units)],
        seeds=list(range(6)), quantiles=[0.1, 0.5, 0.9])
    assert block['status'] == 'ok'
    block['sufficient_statistics']['per_seed_nll_diff'][0][0] = HUGE
    path = tmp_path / 'quantiles.json'
    path.write_text(
        json.dumps({'finaltest': {'comparisons': {
            'c': {'doc_sampling': {'unit_quantile_profile': block}}}}}),
        encoding='utf-8')

    out = A.audit_unit_quantiles(str(path))
    capsys.readouterr()
    assert out['status'] == 'fail'
    assert out['problems']


def test_efficiency_audit_withholds_a_huge_prefill_time(monkeypatch, capsys):
    import audit_gates as A

    summary = {
        'a::seed0': {
            'variant': 'a',
            'efficiency': {'status': 'ok', 'prefill_ms_p50': HUGE,
                           'n_tokens': 512, 'batch': 1,
                           'decode_has_kv_cache': False,
                           'device': 'cpu', 'torch': 'test'},
        },
        'b::seed0': {
            'variant': 'b',
            'efficiency': {'status': 'ok', 'prefill_ms_p50': 2.0,
                           'n_tokens': 512, 'batch': 1,
                           'decode_has_kv_cache': False,
                           'device': 'cpu', 'torch': 'test'},
        },
    }
    monkeypatch.setattr(A, '_load', lambda path: summary)

    out = A.audit_efficiency_pairs(outdirs=['fake'])
    capsys.readouterr()
    assert out['status'] == 'ok'
    assert out['claims_allowed'] == []
    assert out['refused'] == ['fake: a vs b']


def test_efficiency_report_withholds_unrepresentable_numbers(monkeypatch):
    import _build_report as BR

    monkeypatch.setattr(
        BR, '_result_summaries',
        lambda: {'panel': {'cell': {
            'train_tokens_per_s': HUGE, 'params': HUGE,
            'tokens_seen': 10, 'efficiency': {'status': 'ok'},
        }}})
    out = BR.sec_efficiency()
    assert out.startswith('## Measured efficiency')
    # Both the parameter count and the training throughput are withheld as
    # unavailable instead of being printed from an overflowing conversion.
    assert out.count('n/a') >= 2
