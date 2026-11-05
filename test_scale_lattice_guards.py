"""Scale guards for the exact ablation-lattice analysis.

The exact Shapley estimator materialises ``2**k`` subsets, so a caller must
not be able to lift the hard component cap, a stored artifact must not be able
to carry more cells than its declared lattice, and the design audit must not
walk an arbitrarily oversized ``arms_by_subset`` mapping.  These are CPU-only
checks (no training, no split read).
"""
import copy
import json
import math

from test_support import *  # noqa: F401,F403


def _ppl_cells(components, *, seeds=(0, 1, 2)):
    cells = {}
    for mask in range(1 << len(components)):
        subset = frozenset(c for i, c in enumerate(components) if mask >> i & 1)
        cells[subset] = {seed: math.exp(-0.1 * mask) for seed in seeds}
    return cells


def test_max_components_cannot_lift_the_exact_lattice_cap():
    # Seven components would need 128 cells; the hard cap is six (64 cells).
    comps = [f'c{i}' for i in range(7)]
    cells = _ppl_cells(comps)
    out = L.shapley_lattice_attribution(
        cells, components=comps, max_components=10 ** 6, min_n=1)
    assert out['status'] == 'not_estimable'
    assert out['max_components'] == L.AL_LATTICE_MAX_COMPONENTS
    assert any('max_components' in p for p in out['problems'])


def test_design_max_components_cannot_lift_the_exact_lattice_cap():
    comps = [f'c{i}' for i in range(7)]
    out = L.ablation_lattice_design(
        components=comps, arms_by_subset={}, max_components=10 ** 6)
    assert out['status'] == 'not_estimable'
    assert out['max_components'] == L.AL_LATTICE_MAX_COMPONENTS


def test_oversized_arms_mapping_is_rejected_before_it_is_walked():
    # A two-component lattice has exactly four cells.  The extra entries are
    # structurally impossible and must fail closed before the per-key loop.
    arms = {
        frozenset(): 'a0',
        frozenset({'a'}): 'a1',
        frozenset({'b'}): 'b1',
        frozenset({'a', 'b'}): 'ab',
    }
    oversized = dict(arms)
    oversized[('a', 'a', 'b')] = 'not-a-subset'
    out = L.ablation_lattice_design(
        components=['a', 'b'], arms_by_subset=oversized, present=set(arms.values()))
    assert out['status'] == 'fail'
    assert any('arms_by_subset has' in p for p in out['problems'])


def test_oversized_stored_cell_list_fails_closed():
    cells = _ppl_cells(['a', 'b'])
    art = L.build_ablation_lattice(cells, components=['a', 'b'])
    bad = copy.deepcopy(art)
    bad['inputs']['cells'].append(copy.deepcopy(bad['inputs']['cells'][0]))
    verdict = L.validate_ablation_lattice(bad)
    assert verdict['status'] == 'fail'
    assert any('inputs' in p or 'content_sha256' in p
               for p in verdict['problems'])


def test_huge_numeric_tamper_does_not_abort_replay():
    cells = _ppl_cells(['a', 'b'])
    art = L.build_ablation_lattice(cells, components=['a', 'b'])
    bad = copy.deepcopy(art)
    bad['total_effect']['mean'] = 10 ** 10000
    verdict = L.validate_ablation_lattice(bad)
    assert verdict['status'] == 'fail'
    # The verdict itself must stay serialisable.
    json.dumps(verdict, allow_nan=False)


def test_unreportable_lattice_cell_count_is_withheld_not_serialised():
    comps = [f'c{i}' for i in range(20000)]
    out = L.ablation_lattice_design(components=comps, arms_by_subset={})
    assert out['status'] == 'not_estimable', out
    assert out['global']['full_lattice_cells'] is None, out
    assert out['global']['full_lattice_cells_bits'] == 20001, out
    assert out['global']['full_lattice_cells_available'] is False, out
    json.dumps(out, allow_nan=False)

    shapley = L.shapley_lattice_attribution({}, components=comps, min_n=1)
    assert shapley['status'] == 'not_estimable', shapley
    json.dumps(shapley, allow_nan=False)
