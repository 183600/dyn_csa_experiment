"""The raw-PPL sensitivity slope must survive finite effects near float64 max.

A constant raw effect of 1e308 has a true OLS slope of exactly zero, but the
historical ``sum(values)`` intermediate overflowed to inf and the component's
raw slope became unavailable (with the samples silently dropped by the
downstream paired summary).  The log-scale primary already had a scaled path;
the raw sensitivity scale needs the same guard.  CPU-only, zero-experiment.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_constant_huge_raw_effects_have_a_zero_slope():
    assert L._ba_slope([1e308, 1e308, 1e308], [1, 2, 3]) == 0.0
    got = L._ba_slope([1e308, 9e307, 8e307], [1, 2, 3])
    assert got is not None and math.isfinite(got) and got < 0.0
    # Ordinary magnitudes keep the historical result.
    y = [1.0, 2.0, 4.0]
    x = [0.0, math.log(2.0), math.log(3.0)]
    xbar = sum(x) / 3.0
    want = (sum((xi - xbar) * (yi - sum(y) / 3.0)
                for xi, yi in zip(x, y))
            / sum((xi - xbar) ** 2 for xi in x))
    assert L._ba_slope(y, [1, 2, 3]) == want


def test_artifact_keeps_the_raw_slope_for_extreme_ppl_cells():
    budgets = [1, 2, 3]
    seeds = list(range(4))
    design = {
        'budget_key': 'steps',
        'budgets': budgets,
        'components': [{'name': 'c', 'with_arm': 'w',
                        'without_arm': 'wo'}],
        'seeds': seeds,
    }
    rows = []
    for budget in budgets:
        for seed in seeds:
            rows.append({'arm': 'w', 'seed': seed, 'budget': budget,
                         'ppl': 1e-300})
            rows.append({'arm': 'wo', 'seed': seed, 'budget': budget,
                         'ppl': 1e308})
    art = L.build_budget_ablation(design, rows)
    component = art['result']['components']['c']
    assert component['status'] == 'ok', component
    assert component['raw_slope']['n'] == len(seeds)
    assert component['raw_slope']['mean'] == 0.0
    assert L.validate_budget_ablation(art)['status'] in ('ok', 'warnings')
    json.dumps(art, allow_nan=False)
