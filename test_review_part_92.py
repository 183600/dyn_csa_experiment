"""Curve-level simultaneous inference: a row of p-values is not a curve.

These checks exercise the exact sign-flip max-T band, the range test of
flatness, the intersection-union (IUT) one-sided/TOST claims, complete-seed
handling, the artifact/replay gate and the confirmation/CLI/Makefile wiring.
No training, no split read, and no frozen p/Holm/confirmed change.
"""
import inspect
import itertools
import json
import math
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(6))


def _curve(**overrides):
    effects = {
        '512': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
        '1024': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
        '2048': [-0.05, -0.10, -0.02, -0.08, -0.04, -0.06],
        '4096': [0.10, 0.20, 0.15, 0.12, 0.18, 0.16],
    }
    kwargs = dict(seeds=SEEDS, alpha=0.05, min_n=6)
    kwargs.update(overrides)
    return L.curve_simultaneous_inference(list(effects), effects, **kwargs)


def _brute_one_sided(values):
    values = [float(v) for v in values]
    n = len(values)
    obs = sum(values)
    lower = upper = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        total = sum(s * v for s, v in zip(signs, values))
        if total <= obs + 1e-12:
            lower += 1
        if total >= obs - 1e-12:
            upper += 1
    total = 1 << n
    return lower / total, upper / total


def _brute_tost(values, margin):
    values = [float(v) for v in values]
    n = len(values)
    low = [v + margin for v in values]
    high = [margin - v for v in values]
    total = 1 << n
    low_p = high_p = 0
    obs_low = sum(low)
    obs_high = sum(high)
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        if sum(s * v for s, v in zip(signs, low)) >= obs_low - 1e-12:
            low_p += 1
        if sum(s * v for s, v in zip(signs, high)) >= obs_high - 1e-12:
            high_p += 1
    return max(low_p / total, high_p / total)


def test_exact_one_sided_and_tost_match_brute_force():
    effects = {
        'a': [-0.20, -0.32, -0.11, -0.24, -0.17, -0.28],
        'b': [-0.03, -0.08, 0.01, -0.05, -0.02, -0.04],
        'c': [0.12, 0.20, 0.16, 0.10, 0.18, 0.14],
    }
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=6,
        margin=0.05)
    for label, values in effects.items():
        row = result['per_condition'][label]
        lower, upper = _brute_one_sided(values)
        assert row['p_pointwise_lower'] == pytest.approx(lower)
        assert row['p_pointwise_upper'] == pytest.approx(upper)
        assert row['equivalence']['p_tost'] == pytest.approx(
            _brute_tost(values, 0.05))


def test_range_test_matches_brute_force_and_ignores_a_common_shift():
    effects = {
        'a': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
        'b': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
        'c': [0.10, 0.20, 0.15, 0.12, 0.18, 0.16],
    }
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=6)
    means = [sum(v) / len(v) for v in effects.values()]
    obs_range = max(means) - min(means)
    total = 1 << len(SEEDS)
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(SEEDS)):
        shifted = [sum(s * v for s, v in zip(signs, values)) / len(SEEDS)
                   for values in effects.values()]
        if max(shifted) - min(shifted) >= obs_range - 1e-12:
            count += 1
    assert result['flatness']['p_range'] == pytest.approx(count / total)
    assert result['flatness']['observed_range'] == pytest.approx(obs_range)
    # Add the same constant to every condition: the range is unchanged.
    shifted_effects = {k: [v + 0.7 for v in vals]
                       for k, vals in effects.items()}
    shifted = L.curve_simultaneous_inference(
        list(shifted_effects), shifted_effects, seeds=SEEDS,
        alpha=0.05, min_n=6)
    assert shifted['flatness']['p_range'] == pytest.approx(
        result['flatness']['p_range'])
    assert shifted['flatness']['observed_range'] == pytest.approx(obs_range)


def test_max_t_adjusted_p_is_family_level_and_interval_is_studentised():
    result = _curve()
    assert result['status'] == 'ok'
    assert result['method'] == 'exact_signflip_maxt_iut'
    assert result['family']['m'] == 4
    assert result['family']['n_patterns'] == 64
    for label, row in result['per_condition'].items():
        assert row['p_adj_single'] >= row['p_pointwise_two_sided'] - 1e-12
        assert row['p_adj_stepdown'] >= row['p_pointwise_two_sided'] - 1e-12
        sim = row['ci95_simultaneous']
        assert sim is not None
        half = (sim[1] - sim[0]) / 2.0
        assert half == pytest.approx(
            result['family']['critical_value'] * row['sem'], rel=1e-9)
        assert row['direction'] in ('positive', 'negative',
                                    'positive_unresolved',
                                    'negative_unresolved', 'zero_unresolved')


def test_uniform_negative_iut_is_an_all_conditions_claim():
    improving = {
        '512': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
        '1024': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
        '2048': [-0.05, -0.10, -0.02, -0.08, -0.04, -0.06],
    }
    result = L.curve_simultaneous_inference(
        list(improving), improving, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['uniform']['negative']['supported'] is True
    assert result['uniform']['negative']['p_iut'] == pytest.approx(1.0 / 64.0)
    # One condition with the opposite sign closes the intersection claim.
    mixed = dict(improving)
    mixed['4096'] = [0.10, 0.15, 0.05, 0.12, 0.08, 0.11]
    result = L.curve_simultaneous_inference(
        list(mixed), mixed, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['uniform']['negative']['supported'] is False
    assert result['uniform']['negative']['p_iut'] > 0.5
    assert result['mixed_direction'] is True


def test_uniform_equivalence_needs_every_condition_inside_the_margin():
    inside = {
        '512': [0.01, -0.01, 0.00, 0.02, -0.02, 0.00],
        '1024': [0.00, 0.01, -0.01, 0.00, 0.01, -0.01],
        '2048': [0.005, -0.005, 0.005, -0.005, 0.005, -0.005],
    }
    result = L.curve_simultaneous_inference(
        list(inside), inside, seeds=SEEDS, alpha=0.05, min_n=6,
        margin=0.05)
    assert result['uniform']['equivalence']['supported'] is True
    assert result['uniform']['equivalence']['conditions_not_equivalent'] == []
    outside = dict(inside)
    outside['2048'] = [0.10, 0.08, 0.12, 0.09, 0.11, 0.07]
    result = L.curve_simultaneous_inference(
        list(outside), outside, seeds=SEEDS, alpha=0.05, min_n=6,
        margin=0.05)
    assert result['uniform']['equivalence']['supported'] is False
    assert result['uniform']['equivalence']['conditions_not_equivalent'] == ['2048']


def test_flatness_equivalence_is_an_iut_over_all_pairs():
    inside = {
        'a': [0.01, -0.01, 0.00, 0.02, -0.02, 0.00],
        'b': [0.00, 0.01, -0.01, 0.00, 0.01, -0.01],
        'c': [0.005, -0.005, 0.005, -0.005, 0.005, -0.005],
    }
    result = L.curve_simultaneous_inference(
        list(inside), inside, seeds=SEEDS, alpha=0.05, min_n=6,
        flatness_margin=0.05)
    eq = result['flatness']['equivalence']
    assert eq['supported'] is True
    assert eq['n_pairs'] == 3
    assert eq['p_tost'] == pytest.approx(
        max(_brute_tost([inside[a][i] - inside[b][i] for i in range(6)], 0.05)
            for a, b in itertools.combinations(inside, 2)))
    outside = dict(inside)
    outside['c'] = [0.20, 0.18, 0.22, 0.19, 0.21, 0.17]
    result = L.curve_simultaneous_inference(
        list(outside), outside, seeds=SEEDS, alpha=0.05, min_n=6,
        flatness_margin=0.05)
    assert result['flatness']['equivalence']['supported'] is False


def test_flatness_detects_a_monotone_curve_and_not_a_constant_one():
    increasing = {
        'a': [-0.20, -0.22, -0.18, -0.21, -0.19, -0.20],
        'b': [-0.05, -0.06, -0.04, -0.05, -0.06, -0.05],
        'c': [0.10, 0.12, 0.08, 0.11, 0.09, 0.10],
        'd': [0.25, 0.27, 0.23, 0.26, 0.24, 0.25],
    }
    result = L.curve_simultaneous_inference(
        list(increasing), increasing, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['flatness']['p_range'] < 0.05
    constant = {k: [0.10, 0.11, 0.09, 0.10, 0.12, 0.08]
                for k in ('a', 'b', 'c')}
    result = L.curve_simultaneous_inference(
        list(constant), constant, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['flatness']['p_range'] > 0.5


def test_incomplete_seed_is_dropped_from_the_whole_curve_and_named():
    effects = {
        '512': [-0.20, -0.30, None, -0.25, -0.15, -0.22],
        '1024': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
    }
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=3)
    assert result['status'] == 'warnings'
    assert result['n'] == 5
    assert result['n_input'] == 6
    assert result['dropped_seeds'] == [
        {'seed': 2, 'missing_conditions': ['512']}]
    assert result['per_condition']['512']['n'] == 5


def test_underpowered_and_resolution_limited_are_named():
    effects = {'a': [-0.2, -0.3, -0.1], 'b': [-0.1, -0.2, -0.15]}
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=[0, 1, 2], alpha=0.05, min_n=6)
    assert result['status'] == 'warnings'
    assert any('underpowered' in text for text in result['warnings'])
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=[0, 1, 2], alpha=0.01, min_n=3)
    assert any('resolution-limited' in text for text in result['warnings'])
    assert result['family']['p_floor_two_sided'] == pytest.approx(0.25)


def test_all_zero_curve_is_not_estimable():
    effects = {'a': [0.0] * 6, 'b': [0.0] * 6}
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['status'] == 'not_estimable'
    assert 'exactly zero' in result['reason']


def test_one_condition_has_no_flatness_claim():
    effects = {'only': [-0.2, -0.3, -0.1, -0.25, -0.15, -0.22]}
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=6)
    assert result['status'] == 'warnings'
    assert result['flatness']['status'] == 'not_applicable'


def test_declaration_errors_fail_closed():
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a', 'a'], {'a': [0.0] * 6},
                                       seeds=SEEDS)
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a', 'b'], {'a': [0.0] * 6},
                                       seeds=SEEDS)
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a'], {'a': [0.0] * 6},
                                       seeds=[0, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a'], {'a': [True] * 6},
                                       seeds=SEEDS)
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a'], {'a': [0.0] * 6},
                                       seeds=SEEDS, alpha=0.0)
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a'], {'a': [0.0] * 6},
                                       seeds=SEEDS, min_n=0)
    with pytest.raises(ValueError):
        L.curve_simultaneous_inference(['a'], {'a': [0.0] * 6},
                                       seeds=SEEDS, margin=-0.1)


def test_artifact_replay_tamper_and_forbidden_fields():
    artifact = L.build_curve_inference(
        list(_curve()['per_condition']), {
            '512': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
            '1024': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
            '2048': [-0.05, -0.10, -0.02, -0.08, -0.04, -0.06],
            '4096': [0.10, 0.20, 0.15, 0.12, 0.18, 0.16],
        }, seeds=SEEDS, min_n=6)
    good = L.validate_curve_inference(artifact)
    assert good['status'] in ('ok', 'warnings')
    assert good['replayed'] is True
    bad = json.loads(json.dumps(artifact))
    bad['result']['per_condition']['512']['mean'] = 99.0
    assert L.validate_curve_inference(bad)['status'] == 'fail'
    frozen = json.loads(json.dumps(artifact))
    frozen['result']['confirmed'] = True
    assert L.validate_curve_inference(frozen)['status'] == 'fail'


def test_block_validation_and_tamper():
    block = L.curve_inference_block(
        ['a', 'b'],
        {'a': [-0.2, -0.3, -0.1, -0.25, -0.15, -0.22],
         'b': [-0.1, -0.2, -0.15, -0.18, -0.12, -0.16]},
        seeds=SEEDS, min_n=6)
    assert L.validate_curve_inference_block(block)['status'] in ('ok', 'warnings')
    bad = json.loads(json.dumps(block))
    bad['result']['uniform']['negative']['supported'] = True
    bad['result']['uniform']['negative']['p_iut'] = 1e-9
    assert L.validate_curve_inference_block(bad)['status'] == 'fail'


def test_writer_noop_conflict_overwrite():
    artifact = L.build_curve_inference(
        ['a', 'b'],
        {'a': [-0.2, -0.3, -0.1, -0.25, -0.15, -0.22],
         'b': [-0.1, -0.2, -0.15, -0.18, -0.12, -0.16]},
        seeds=SEEDS, min_n=6)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'curve_inference.json')
        assert L.write_curve_inference(path, artifact)['status'] == 'written'
        assert L.write_curve_inference(path, artifact)['status'] == 'noop'
        different = L.build_curve_inference(
            ['a', 'b'],
            {'a': [-0.1, -0.2, -0.15, -0.18, -0.12, -0.16],
             'b': [-0.2, -0.3, -0.1, -0.25, -0.15, -0.22]},
            seeds=SEEDS, min_n=6)
        assert L.write_curve_inference(path, different)['status'] == 'conflict'
        assert L.write_curve_inference(
            path, different, overwrite=True)['status'] == 'written'


def test_markdown_contains_curve_level_fields():
    result = _curve()
    markdown = L.curve_inference_markdown(result)
    assert 'Curve-level simultaneous inference' in markdown
    assert 'adjusted p' in markdown
    assert 'all conditions improve' in markdown
    assert 'flatness' in markdown


def test_audit_gate_artifact_block_test_stats_and_tamper():
    effects = {
        '512': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
        '1024': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
    }
    artifact = L.build_curve_inference(
        list(effects), effects, seeds=SEEDS, min_n=6)
    block = L.curve_inference_block(
        list(effects), effects, seeds=SEEDS, min_n=6)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'curve_inference.json')
        L.write_curve_inference(path, artifact)
        gate = A.audit_curve_inference(path, verbose=False)
        assert gate['status'] in ('ok', 'warnings')
        assert gate['n_blocks'] == 1
        block_path = os.path.join(tmp, 'block.json')
        with open(block_path, 'w', encoding='utf-8') as fh:
            json.dump(block, fh)
        assert A.audit_curve_inference(
            block_path, verbose=False)['status'] in ('ok', 'warnings')
        ts_path = os.path.join(tmp, 'test_stats.json')
        with open(ts_path, 'w', encoding='utf-8') as fh:
            json.dump({'comparisons': {'cmp': {
                'length_curve_inference': block}}}, fh)
        gate = A.audit_curve_inference(ts_path, verbose=False)
        assert gate['status'] in ('ok', 'warnings')
        assert gate['n_blocks'] == 1
        bad = json.loads(json.dumps(block))
        bad['result']['flatness']['p_range'] = 1e-9
        with open(block_path, 'w', encoding='utf-8') as fh:
            json.dump(bad, fh)
        assert A.audit_curve_inference(
            block_path, verbose=False)['status'] == 'fail'
    assert A.audit_curve_inference(
        os.path.join(tmp, 'missing.json'), verbose=False)['status'] == (
            'not_verifiable_locally')


def test_audit_gate_missing_and_not_applicable():
    with tempfile.TemporaryDirectory() as tmp:
        ts_path = os.path.join(tmp, 'test_stats.json')
        with open(ts_path, 'w', encoding='utf-8') as fh:
            json.dump({'comparisons': {'cmp': {'mean': 0.0}}}, fh)
        gate = A.audit_curve_inference(ts_path, verbose=False)
        assert gate['status'] == 'not_applicable'
    assert A.audit_curve_inference(
        os.path.join(tempfile.gettempdir(), 'no-such-curve-inference.json'),
        verbose=False)['status'] == 'not_verifiable_locally'


def test_confirmation_helper_complete_seeds_and_markdown():
    recs_a, recs_b = {}, {}
    base = {'512': 20.0, '1024': 19.0, '2048': 18.0, '4096': 17.0}
    for seed in SEEDS:
        recs_a[seed] = {'by_len': {
            str(L): {'ppl': base[str(L)] * (1.0 - 0.01 * (seed + 1)),
                     'truncated': False}
            for L in (512, 1024, 2048, 4096)}}
        recs_b[seed] = {'by_len': {
            str(L): {'ppl': base[str(L)] * (1.0 + 0.005 * (seed + 1)),
                     'truncated': False}
            for L in (512, 1024, 2048, 4096)}}
    recs_a[0]['by_len']['1024']['ppl'] = None
    block = V12._length_curve_inference(  # noqa: F821
        recs_a, recs_b, SEEDS, [512, 1024, 2048, 4096],
        alpha=0.05, min_n=3)
    assert block['kind'] == 'curve_inference_block'
    assert block['result']['n'] == 5
    assert block['result']['n_input'] == 6
    assert block['result']['dropped_seeds'] == [
        {'seed': 0, 'missing_conditions': ['1024']}]
    lines = V12._length_curve_inference_markdown('cmp', block)  # noqa: F821
    assert any('全长度曲线' in line for line in lines)
    assert any('simultaneous' in line for line in lines)


def test_normal_fallback_above_exact_cap():
    effects = {
        label: [-0.20 + 0.01 * math.sin(i + k)
                for i in range(21)]
        for k, label in enumerate(('a', 'b', 'c'))
    }
    result = L.curve_simultaneous_inference(
        list(effects), effects, seeds=list(range(21)), alpha=0.05, min_n=6,
        margin=0.05, flatness_margin=0.05)
    assert result['method'] == 'normal_approximation_fallback'
    assert result['exact'] is False
    assert result['status'] == 'warnings'
    assert any('normal-approximation' in text for text in result['warnings'])
    assert result['per_condition']['a']['p_pointwise_lower'] is not None
    assert result['uniform']['negative']['supported'] is True


def test_report_cli_and_makefile_wiring_and_protected_paths():
    src = inspect.getsource(A.run_report)
    assert 'curve_inference' in src
    cli_src = inspect.getsource(A._run_cli)
    assert 'curve_inference' in cli_src
    makefile = open(os.path.join(REPO, 'Makefile'), encoding='utf-8').read()
    assert 'curve-inference:' in makefile
    assert 'audit_gates.py curve_inference' in makefile
    for relative in ('.github', 'README.md', 'README_zh.md'):
        assert os.path.exists(os.path.join(REPO, relative))
