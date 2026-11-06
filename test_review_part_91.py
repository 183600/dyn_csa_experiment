"""Budget x ablation interaction: a single-budget effect is not budget-invariant.

A leave-one-out / on-off ablation summarised at one training budget is a
property of that stopping point, not of the component.  These checks exercise
the paired log-PPL budget ladder, the exact sign-flip slope test, the declared
Holm family, the optional exact TOST stability companion, the raw-scale
sensitivity, the artifact replay and the audit/CLI/Makefile wiring.  No
training, no split read, and no frozen p/Holm/confirmed change.
"""
import inspect
import json
import math
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [1500, 3000, 8000]
SEEDS = list(range(6))


def _cells(effects, *, arms=('with', 'without'), budgets=BUDGETS,
           seeds=SEEDS, noise=1e-4):
    """Cells whose paired log-PPL effect is ``effects[i]`` plus seed noise."""
    rows = []
    for index, budget in enumerate(budgets):
        for seed in seeds:
            base = 30.0 * math.exp(-0.10 * index)
            wiggle = noise * math.sin((seed + 1) * (index + 2))
            with_ppl = base * math.exp(-effects[index] / 2.0)
            without_ppl = base * math.exp(effects[index] / 2.0 + wiggle)
            rows.append({'arm': arms[0], 'seed': seed, 'budget': budget,
                         'ppl': with_ppl})
            rows.append({'arm': arms[1], 'seed': seed, 'budget': budget,
                         'ppl': without_ppl})
    return rows


def _design(component='component', *, budgets=BUDGETS, seeds=SEEDS,
            with_arm='with', without_arm='without'):
    return {
        'budget_key': 'steps',
        'budgets': budgets,
        'components': [{'name': component, 'with_arm': with_arm,
                        'without_arm': without_arm}],
        'seeds': seeds,
    }


def test_growing_effect_is_detected_and_not_extrapolable():
    result = L.budget_ablation_analysis(
        _design(), _cells([0.0, 0.20, 0.40]))
    row = result['components']['component']
    assert result['status'] == 'warnings'
    assert row['classification'] == 'interaction_detected'
    assert row['budget_sensitive'] is True
    assert row['interaction_detected'] is True
    assert row['sign_reversal'] is False
    assert row['single_budget_extrapolation_allowed'] is False
    assert row['log_slope']['mean'] > 0.0
    assert row['log_slope']['p_exact_signflip'] == pytest.approx(2.0 / 64.0)
    assert row['p_family'] == pytest.approx(2.0 / 64.0)
    assert any('interaction is detected' in text for text in row['warnings'])


def test_stable_effect_needs_a_margin_before_stability_is_claimed():
    no_margin = L.budget_ablation_analysis(
        _design(), _cells([0.0, 0.0, 0.0]), margin=None)
    row = no_margin['components']['component']
    assert row['classification'] == 'no_interaction_detected'
    assert row['single_budget_extrapolation_allowed'] is False
    assert row['equivalence'] is None
    assert row['stability_test']['status'] == 'not_tested'
    assert any('no interaction SESOI' in text for text in no_margin['warnings'])

    with_margin = L.budget_ablation_analysis(
        _design(), _cells([0.0, 0.0, 0.0]), margin=0.05)
    row = with_margin['components']['component']
    assert row['classification'] == 'equivalence_supported'
    assert row['single_budget_extrapolation_allowed'] is True
    assert row['equivalence']['equivalent'] is True
    assert row['stability_test']['status'] == 'equivalence_supported'
    assert with_margin['status'] == 'ok'
    assert not row['warnings']


def test_sign_reversal_is_flagged_even_when_the_slope_is_significant():
    result = L.budget_ablation_analysis(
        _design(), _cells([0.40, 0.0, -0.40]))
    row = result['components']['component']
    assert row['sign_reversal'] is True
    assert row['classification'] == 'interaction_detected'
    assert row['single_budget_extrapolation_allowed'] is False
    assert any('changes sign' in text for text in row['warnings'])


def test_small_n_is_resolution_limited_not_read_as_stability():
    result = L.budget_ablation_analysis(
        _design(seeds=list(range(4))),
        _cells([0.0, 0.2, 0.4], seeds=list(range(4))))
    row = result['components']['component']
    assert row['log_slope']['underpowered'] is False  # min_n defaults to 3
    assert row['log_slope']['resolution_limited'] is True
    assert row['budget_sensitive'] is False
    assert row['classification'] == 'inconclusive'
    assert row['single_budget_extrapolation_allowed'] is False
    assert any('resolution-limited' in text for text in row['warnings'])


def test_below_min_n_is_underpowered():
    result = L.budget_ablation_analysis(
        _design(seeds=[0, 1]), _cells([0.0, 0.2, 0.4], seeds=[0, 1]),
        min_n=3)
    row = result['components']['component']
    assert row['log_slope']['underpowered'] is True
    assert row['budget_sensitive'] is False
    assert row['classification'] == 'inconclusive'
    assert any('below min_n' in text for text in row['warnings'])


def test_missing_cell_is_named_and_never_imputed():
    cells = _cells([0.0, 0.2, 0.4])
    cells = [c for c in cells if not (c['arm'] == 'without'
                                      and c['seed'] == 2
                                      and c['budget'] == 3000)]
    result = L.budget_ablation_analysis(_design(), cells)
    assert result['status'] == 'not_estimable'
    assert result['n_estimable'] == 0
    row = result['components']['component']
    assert row['status'] == 'not_estimable'
    assert 'missing' in row['reason']


def test_duplicate_and_nonpositive_ppl_are_refused():
    cells = _cells([0.0, 0.2, 0.4])
    dup = list(cells) + [dict(cells[0])]
    result = L.budget_ablation_analysis(_design(), dup)
    assert result['status'] == 'not_estimable'
    assert any('duplicate' in problem
               for problem in result.get('cell_problems') or [])

    bad = [dict(c) for c in cells]
    bad[0]['ppl'] = 0.0
    result = L.budget_ablation_analysis(_design(), bad)
    assert result['status'] == 'not_estimable'
    row = result['components']['component']
    assert row['status'] == 'not_estimable'
    assert 'non-positive' in row['reason']


def test_invalid_designs_fail_closed():
    with pytest.raises(ValueError):
        L.budget_ablation_analysis(
            {'budget_key': 'steps', 'budgets': [3000, 1500],
             'components': [{'name': 'x', 'with_arm': 'a',
                             'without_arm': 'b'}]},
            _cells([0.0, 0.0] , budgets=[1500, 3000]))
    with pytest.raises(ValueError):
        L.budget_ablation_analysis(
            {'budget_key': 'steps', 'budgets': BUDGETS,
             'components': [{'name': 'x', 'with_arm': 'a',
                             'without_arm': 'a'}]},
            _cells([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        L.budget_ablation_analysis(
            {'budget_key': 'steps', 'budgets': BUDGETS,
             'components': [{'name': 'x', 'with_arm': 'a',
                             'without_arm': 'b'},
                            {'name': 'x', 'with_arm': 'c',
                             'without_arm': 'd'}]},
            _cells([0.0, 0.0, 0.0]))


def test_family_holm_includes_every_declared_component():
    design = {
        'budget_key': 'steps',
        'budgets': BUDGETS,
        'components': [
            {'name': 'grow', 'with_arm': 'w1', 'without_arm': 'o1'},
            {'name': 'stable', 'with_arm': 'w2', 'without_arm': 'o2'},
        ],
        'seeds': SEEDS,
    }
    cells = (_cells([0.0, 0.2, 0.4], arms=('w1', 'o1'))
             + _cells([0.0, 0.0, 0.0], arms=('w2', 'o2')))
    result = L.budget_ablation_analysis(design, cells)
    grow = result['components']['grow']
    stable = result['components']['stable']
    assert result['family']['m'] == 2
    assert grow['log_slope']['p_exact_signflip'] == pytest.approx(2.0 / 64.0)
    assert grow['p_family'] == pytest.approx(4.0 / 64.0)
    assert grow['p_family'] >= grow['log_slope']['p_exact_signflip']
    assert grow['rank'] == 1
    assert stable['rank'] == 2
    assert grow['budget_sensitive'] is False  # Holm-adjusted, not raw
    assert grow['interaction_detected_raw'] is True
    assert grow['classification'] == 'interaction_detected_raw_only'
    assert any('nominal interaction' in text for text in grow['warnings'])


def test_not_estimable_component_still_occupies_a_family_slot():
    design = {
        'budget_key': 'steps',
        'budgets': BUDGETS,
        'components': [
            {'name': 'ok', 'with_arm': 'w1', 'without_arm': 'o1'},
            {'name': 'missing', 'with_arm': 'w2', 'without_arm': 'o2'},
        ],
        'seeds': SEEDS,
    }
    cells = _cells([0.0, 0.2, 0.4], arms=('w1', 'o1'))
    result = L.budget_ablation_analysis(design, cells)
    assert result['family']['m'] == 2
    assert result['components']['missing']['status'] == 'not_estimable'
    assert result['components']['missing']['p_family'] == 1.0
    assert result['components']['ok']['rank'] == 1


def test_raw_and_log_scales_can_disagree():
    # Absolute PPL gaps shrink while log gaps grow: the raw-PPL slope is
    # negative and the log-PPL slope is positive.  Neither scale may be chosen
    # after seeing the result.
    base = [100.0, 1.0, 0.5]
    raw_gap = [1.0, 0.05, 0.02]
    cells = []
    for index, budget in enumerate(BUDGETS):
        for seed in SEEDS:
            wiggle = 1.0 + 0.001 * math.sin((seed + 1) * (index + 2))
            with_ppl = base[index]
            without_ppl = base[index] + raw_gap[index] * wiggle
            cells.append({'arm': 'with', 'seed': seed, 'budget': budget,
                          'ppl': with_ppl})
            cells.append({'arm': 'without', 'seed': seed, 'budget': budget,
                          'ppl': without_ppl})
    result = L.budget_ablation_analysis(_design(), cells)
    row = result['components']['component']
    assert row['log_slope']['mean'] > 0.0
    assert row['raw_slope']['mean'] < 0.0
    assert row['scale_sensitive'] is True
    assert any('raw-PPL' in text for text in row['warnings'])


def test_artifact_hash_round_trip_and_created_at_identity():
    artifact = L.build_budget_ablation(
        _design(), _cells([0.0, 0.2, 0.4]), source='synthetic')
    verdict = L.validate_budget_ablation(artifact)
    assert verdict['status'] in ('ok', 'warnings')
    assert verdict['problems'] == []
    again = L.build_budget_ablation(
        _design(), _cells([0.0, 0.2, 0.4]), source='synthetic',
        created_at='2000-01-01T00:00:00')
    assert again['content_sha256'] == artifact['content_sha256']
    # A changed result is a different content identity.
    bad = json.loads(json.dumps(artifact))
    bad['result']['components']['component']['log_slope']['mean'] = 99.0
    assert L.validate_budget_ablation(bad)['status'] == 'fail'
    # Frozen-decision fields are refused even when the hash is recomputed.
    frozen = json.loads(json.dumps(artifact))
    frozen['result']['components']['component']['p_holm'] = 0.01
    assert L.validate_budget_ablation(frozen)['status'] == 'fail'


def test_writer_noop_conflict_overwrite():
    artifact = L.build_budget_ablation(
        _design(), _cells([0.0, 0.2, 0.4]), source='synthetic')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'budget_ablation.json')
        assert L.write_budget_ablation(path, artifact)['status'] == 'written'
        assert L.write_budget_ablation(path, artifact)['status'] == 'noop'
        different = L.build_budget_ablation(
            _design(), _cells([0.0, 0.0, 0.0]), source='synthetic')
        assert L.write_budget_ablation(path, different)['status'] == 'conflict'
        assert L.write_budget_ablation(
            path, different, overwrite=True)['status'] == 'written'


def test_audit_gate_missing_valid_tampered_and_not_estimable():
    assert A.audit_budget_ablation(
        os.path.join(tempfile.gettempdir(), 'no-such-budget-ablation.json'),
        verbose=False)['status'] == 'not_verifiable_locally'
    artifact = L.build_budget_ablation(
        _design(), _cells([0.0, 0.2, 0.4]), source='synthetic')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'budget_ablation.json')
        L.write_budget_ablation(path, artifact)
        good = A.audit_budget_ablation(path, verbose=False)
        assert good['status'] in ('ok', 'warnings')
        assert good['interaction_detected_components'] == ['component']
        bad = json.loads(json.dumps(artifact))
        bad['result']['components']['component']['classification'] = 'stable'
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(bad, fh)
        assert A.audit_budget_ablation(path, verbose=False)['status'] == 'fail'

    missing = [c for c in _cells([0.0, 0.2, 0.4])
               if not (c['arm'] == 'without' and c['seed'] == 0
                       and c['budget'] == 3000)]
    not_est = L.build_budget_ablation(
        _design(), missing, source='synthetic')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'budget_ablation.json')
        L.write_budget_ablation(path, not_est)
        gate = A.audit_budget_ablation(path, verbose=False)
        assert gate['status'] == 'not_verifiable_locally'
        assert gate['result_status'] == 'not_estimable'


def test_records_adapter_and_markdown():
    records = []
    for cell in _cells([0.0, 0.2, 0.4]):
        records.append({'variant': cell['arm'], 'seed': cell['seed'],
                        'steps': cell['budget'], 'ppl': cell['ppl']})
    records.append({'variant': 'other', 'seed': 0, 'steps': 1500, 'ppl': 1.0})
    cells = L.budget_ablation_cells_from_records(records, _design())
    assert len(cells) == len(_cells([0.0, 0.2, 0.4]))
    assert all(set(row) == {'arm', 'seed', 'budget', 'ppl'} for row in cells)
    result = L.budget_ablation_analysis(_design(), cells)
    markdown = L.budget_ablation_markdown(result)
    assert 'Budget x ablation interaction' in markdown
    assert 'p_family' in markdown
    assert 'interaction_detected' in markdown


def test_report_cli_and_makefile_wiring_and_protected_paths():
    src = inspect.getsource(A.run_report)
    assert 'budget_ablation' in src
    cli_src = inspect.getsource(A._run_cli)
    assert 'budget_ablation' in cli_src
    makefile = open(os.path.join(REPO, 'Makefile'), encoding='utf-8').read()
    assert 'budget-ablation:' in makefile
    assert 'audit_gates.py budget_ablation' in makefile
    # The protected documentation/CI paths must still exist.
    for relative in ('.github', 'README.md', 'README_zh.md'):
        assert os.path.exists(os.path.join(REPO, relative))
