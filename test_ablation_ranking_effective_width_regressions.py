"""The ablation-ranking resolution floor must use informative seeds.

The lattice can be stored with one row per planned seed, including seeds whose
Shapley attribution is exactly zero (a padded/unused slot or an exact tie).
A zero vector entry is sign-neutral for that contrast: it duplicates every sign
pattern without changing a p-value, but it does lower the attainable
resolution.  The declared ``2/2**n`` floor therefore understated the floor and
could label a panel resolvable when its informative width could not reach
alpha.  These checks pin the effective-width floor and its replay.
"""

import os
import sys
import itertools
import math

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


ACTIVE_SEEDS = 4
PAD_SEEDS = 16
EFFECTS = {'a': -0.6, 'b': -0.3}


def _lattice(seeds, comps=('a', 'b'), active=ACTIVE_SEEDS):
    cells = {}
    for size in range(len(comps) + 1):
        for combo in itertools.combinations(comps, size):
            cells[frozenset(combo)] = {
                seed: (math.exp(3.0 + sum(EFFECTS[c] for c in combo))
                       if seed < active else 1.0)
                for seed in seeds}
    return cells


def test_zero_padded_lattice_uses_its_informative_width():
    base = L.ablation_ranking(
        _lattice(range(ACTIVE_SEEDS)), components=['a', 'b'],
        min_n=2, scales=('log_ppl',))
    padded = L.ablation_ranking(
        _lattice(range(ACTIVE_SEEDS + PAD_SEEDS)), components=['a', 'b'],
        min_n=2, scales=('log_ppl',))
    assert padded['n_seeds'] == ACTIVE_SEEDS + PAD_SEEDS
    assert padded['n_seeds_effective'] == ACTIVE_SEEDS
    assert padded['n_seeds_effective_components'] == ACTIVE_SEEDS
    assert padded['n_seeds_effective_pairs'] == ACTIVE_SEEDS
    # 2 / 2**4 = 0.125 > alpha=0.05: the informative panel is resolution
    # limited even though the declared n=20 has a theoretical floor ~1.9e-6.
    assert padded['p_floor'] == pytest.approx(2.0 ** (1 - ACTIVE_SEEDS))
    assert padded['p_floor'] == pytest.approx(base['p_floor'])
    assert padded['resolution_limited'] is True
    assert padded['joint']['p_floor_single'] == pytest.approx(
        base['joint']['p_floor_single'])
    for name in ('a', 'b'):
        assert padded['component_stats']['log_ppl'][name]['p_floor'] == (
            pytest.approx(base['component_stats']['log_ppl'][name]['p_floor']))


def test_component_and_pair_widths_are_counted_separately():
    component, pair = L._arb_effective_widths(
        {'a': [1.0, 0.0, 0.0, 0.0], 'b': [0.0, 1.0, 0.0, 0.0]}, ['a', 'b'])
    assert component == 1
    assert pair == 2
    # The family floor is the best any declared test can attain, while the
    # joint max-T floor is set by the component vectors only.
    assert L._arb_p_floor(component) == 1.0
    assert L._arb_p_floor(pair) == 2.0 ** -1


def test_ordinary_lattice_floor_is_unchanged_and_replays():
    out = L.ablation_ranking(
        _lattice(range(8), active=8), components=['a', 'b'], min_n=2,
        scales=('log_ppl',))
    assert out['n_seeds_effective'] == 8
    assert out['p_floor'] == pytest.approx(2.0 ** -7)
    artifact = L.build_ablation_ranking(
        _lattice(range(8), active=8), components=['a', 'b'], min_n=2,
        scales=('log_ppl',))
    assert L.validate_ablation_ranking(artifact)['status'] == 'ok'
