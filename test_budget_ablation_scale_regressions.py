"""Scale/tamper regressions for the budget x ablation interaction fragment.

The ladder is a design declaration, so two properties matter at scale:

* an exact integer budget above float64's range must still produce a finite
  log-budget slope (a ``float(budget)`` intermediate used to overflow);
* an untrusted, edited artifact must fail closed with a strict-JSON diagnostic,
  including CPython 3.12's integer-to-decimal conversion limit and huge mapping
  keys.

These are CPU-only, zero-experiment checks.
"""
import copy
import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


HUGE = 10 ** 10000


def _design(budgets=(1500, 3000, 8000), seeds=range(4)):
    return {
        'budget_key': 'steps',
        'budgets': list(budgets),
        'components': [{'name': 'component', 'with_arm': 'with',
                        'without_arm': 'without'}],
        'seeds': list(seeds),
    }


def _cells(budgets, seeds, effects):
    rows = []
    for index, budget in enumerate(budgets):
        for seed in seeds:
            base = 30.0 * math.exp(-0.1 * index)
            effect = effects[index]
            rows.append({'arm': 'with', 'seed': seed, 'budget': budget,
                         'ppl': base * math.exp(-effect / 2.0)})
            rows.append({'arm': 'without', 'seed': seed, 'budget': budget,
                         'ppl': base * math.exp(effect / 2.0)})
    return rows


def test_huge_budget_ladder_is_representable_and_replays():
    budgets = [10 ** 3998, 10 ** 3999, 10 ** 4000]
    design = _design(budgets=budgets)
    artifact = L.build_budget_ablation(
        design, _cells(budgets, list(range(4)), [0.0, 0.2, 0.4]))
    row = artifact['result']['components']['component']
    assert row['log_slope']['mean'] is not None
    assert math.isfinite(row['log_slope']['mean'])
    verdict = L.validate_budget_ablation(artifact)
    assert verdict['status'] in ('ok', 'warnings'), verdict
    assert verdict['problems'] == []
    json.dumps(artifact, allow_nan=False)
    json.dumps(verdict, allow_nan=False)


def test_finite_ppls_near_float_max_keep_a_representable_mean():
    """The per-budget mean PPL must not be lost to an overflowing sum.

    Four finite PPLs near 1e308 have an ordinary mean near 1e308, but their
    naive total overflows float64.  The summary used to turn both arms into
    ``null`` even though the reported geometric mean was present; a scale-up
    panel must keep the arithmetic readout too.
    """
    scale = 1e308 / 30.0
    budgets = [1500, 3000, 8000]
    design = _design(budgets=budgets)
    rows = []
    expected = {}
    for index, budget in enumerate(budgets):
        base = 30.0 * math.exp(-0.1 * index)
        with_ppl = base * scale * math.exp(0.01)
        without_ppl = base * scale * math.exp(-0.01)
        expected[budget] = (with_ppl, without_ppl)
        for seed in design['seeds']:
            rows.append({'arm': 'with', 'seed': seed, 'budget': budget,
                         'ppl': with_ppl})
            rows.append({'arm': 'without', 'seed': seed, 'budget': budget,
                         'ppl': without_ppl})
    assert sum(row['ppl'] for row in rows[:4]) == float('inf')
    artifact = L.build_budget_ablation(design, rows)
    component = artifact['result']['components']['component']
    assert component['status'] == 'ok', component
    for row in component['per_budget']:
        with_ppl, without_ppl = expected[row['budget']]
        assert row['with_mean_ppl'] == pytest.approx(with_ppl)
        assert row['without_mean_ppl'] == pytest.approx(without_ppl)
        assert math.isfinite(row['with_mean_ppl'])
        assert math.isfinite(row['without_mean_ppl'])
    assert L.validate_budget_ablation(artifact)['status'] in ('ok', 'warnings')
    json.dumps(artifact, allow_nan=False)


def test_tampered_scalars_and_keys_fail_closed_and_stay_strict_json():
    artifact = L.build_budget_ablation(
        _design(), _cells([1500, 3000, 8000], list(range(4)),
                          [0.0, 0.2, 0.4]))
    assert L.validate_budget_ablation(artifact)['status'] in ('ok', 'warnings')

    cases = [
        (('kind',), HUGE),
        (('schema',), HUGE),
        (('result', 'status'), HUGE),
        (('result', 'status'), 'tampered'),
        (('result', 'n_estimable'), HUGE),
        (('result', 'n_estimable'), 99),
        (('result', 'family'), {'m': HUGE}),
        (('result', 'family'), {'m': 99}),
        (('result', 'budgets'), [HUGE]),
        (('result', 'budgets'), [1, 2, 3]),
        (('result', 'components', 'component', 'n'), HUGE),
        (('result', 'components', 'component', 'n'), 99),
        (('inputs', 'alpha'), 'x'),
        (('inputs', 'alpha'), 0.3),
        (('created_at',), HUGE),
        (('source',), HUGE),
    ]
    for path, value in cases:
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        bad['content_sha256'] = L._ba_content_sha(bad)
        verdict = L.validate_budget_ablation(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        json.dumps(verdict, allow_nan=False)

    # Huge integers used as mapping keys are diagnostics, not crashes.
    for where in ('artifact', 'inputs', 'result'):
        bad = copy.deepcopy(artifact)
        if where == 'artifact':
            bad[HUGE] = {'x': 1}
        elif where == 'inputs':
            bad['inputs'][HUGE] = 1
        else:
            bad['result'][HUGE] = 1
        bad['content_sha256'] = L._ba_content_sha(bad)
        verdict = L.validate_budget_ablation(bad, replay=False)
        assert verdict['status'] == 'fail', (where, verdict)
        json.dumps(verdict, allow_nan=False)
        assert any('unrepresentable int' in problem
                   for problem in verdict['problems']), (where, verdict)


def test_duplicate_adapter_cell_is_refused():
    design = _design()
    records = [
        {'variant': 'with', 'seed': 0, 'steps': 1500, 'ppl': 1.0},
        {'variant': 'with', 'seed': 0, 'steps': 1500, 'ppl': 2.0},
    ]
    with pytest.raises(ValueError, match='duplicate'):
        L.budget_ablation_cells_from_records(records, design)


def test_audit_unusable_path_fails_closed_and_stays_strict_json():
    report = A.audit_budget_ablation(HUGE, verbose=False)
    assert report['status'] == 'fail', report
    assert report['problems'], report
    json.dumps(report, allow_nan=False)
    assert any('unrepresentable int' in problem
               for problem in report['problems']), report


def test_written_huge_ladder_audits_cleanly():
    budgets = [10 ** 3998, 10 ** 3999, 10 ** 4000]
    artifact = L.build_budget_ablation(
        _design(budgets=budgets),
        _cells(budgets, list(range(4)), [0.0, 0.2, 0.4]))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'budget_ablation.json')
        assert L.write_budget_ablation(path, artifact)['status'] == 'written'
        report = A.audit_budget_ablation(path, verbose=False)
        assert report['status'] in ('ok', 'warnings'), report
        assert report['problems'] == [], report
        json.dumps(report, allow_nan=False)
