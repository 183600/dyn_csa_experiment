"""Malformed-artifact validators must fail closed, never raise.

A stored JSON artifact is untrusted input.  A hand-edited value can replace a
mapping/sequence with a scalar, a huge integer, or a NumPy value (an in-memory
caller is not guaranteed to have round-tripped through JSON first).  The
validators must report that as a structured ``fail`` and stay strict-JSON
serialisable; they must not surface ``IndexError``/``AttributeError`` or
NumPy's ambiguous-truth ``ValueError``.

These are CPU-only, zero-experiment checks.
"""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
HUGE = 10 ** 10000


def _strict(value):
    json.dumps(value, allow_nan=False)
    return value


def _unit_artifact():
    units = [f'u{i:03d}' for i in range(8)]
    tokens = [100.0 + 7.0 * i for i in range(8)]
    seeds = list(range(6))
    rows = [[-0.20 + 0.01 * math.sin(i + k) for i in range(8)]
            for k in range(6)]
    return L.build_unit_split_replication(
        rows, tokens, unit_ids=units, seeds=seeds,
        direction='negative', n_salts=0)


def test_unit_split_scalar_inputs_are_reported_not_indexed():
    art = _unit_artifact()
    for scalar in (np.bool_(True), True, 1, [1, 2], 'x'):
        bad = copy.deepcopy(art)
        bad['inputs'] = scalar
        verdict = L.validate_unit_split_replication(bad)
        assert verdict['status'] == 'fail', (scalar, verdict)
        assert verdict['problems'], (scalar, verdict)
        _strict(verdict)


def test_unit_split_scalar_assignment_halves_is_a_mismatch_not_a_crash():
    art = _unit_artifact()
    bad = copy.deepcopy(art)
    bad['result']['assignment']['halves'] = np.bool_(True)
    verdict = L.validate_unit_split_replication(bad)
    assert verdict['status'] == 'fail', verdict
    assert any('assignment' in p for p in verdict['problems']), verdict
    _strict(verdict)


def _ablation_cells(seeds, comps=('a', 'b')):
    import itertools
    effects = {'a': -0.6, 'b': -0.3}
    out = {}
    for size in range(len(comps) + 1):
        for combo in itertools.combinations(comps, size):
            out[frozenset(combo)] = {
                seed: math.exp(3.0 + sum(effects[c] for c in combo))
                for seed in seeds}
    return out


def test_lattice_and_ranking_scalar_components_fail_closed():
    lattice = L.build_ablation_lattice(
        _ablation_cells(range(4)), components=['a', 'b'], scale='ppl')
    for value in (np.bool_(True), True, [1, 2], 'x'):
        bad = copy.deepcopy(lattice)
        bad['components'] = value
        verdict = L.validate_ablation_lattice(bad)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)

    ranking = L.build_ablation_ranking(
        _ablation_cells(range(4)), components=['a', 'b'])
    for value in (np.bool_(True), True, [1, 2], 'x'):
        bad = copy.deepcopy(ranking)
        bad['result']['components'] = value
        verdict = L.validate_ablation_ranking(bad)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)


def test_global_fdr_scalar_policy_methods_fails_closed():
    art = L.build_global_fdr([
        {'name': 'a', 'p': 0.01, 'n_nonzero': 12},
        {'name': 'b', 'p': 0.03, 'n_nonzero': 12},
    ])
    for value in (np.bool_(True), True, [1, 2], {'x': 1}, 'x'):
        bad = copy.deepcopy(art)
        bad['result']['policy']['methods'] = value
        verdict = L.validate_global_fdr(bad)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)


def test_contrast_interaction_scalar_design_factor_fails_closed():
    design = L.CONFIRMATION_SINK_2X2
    cells = {cell: {seed: 10.0 + 0.01 * seed for seed in range(4)}
             for cell in design['cells']}
    art = L.build_contrast_interaction(design, cells, min_n=2)
    for value in (np.bool_(True), True, [1, 2], {'x': 1}, 'x'):
        bad = copy.deepcopy(art)
        bad['inputs']['design']['factors'] = value
        verdict = L.validate_contrast_interaction(bad)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)


def test_budget_truncation_scalar_design_arms_fails_closed():
    with open(os.path.join(REPO, 'confirmation_frozen.json'),
              encoding='utf-8') as fh:
        manifest = json.load(fh)
    art = L.budget_truncation_from_manifest(manifest)
    for value in (np.bool_(True), True, [1, 2], {'x': 1}, 'x'):
        bad = copy.deepcopy(art)
        bad['design']['arms'] = value
        verdict = L.validate_budget_truncation(bad)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)


def test_validation_transfer_scalar_comparison_fails_closed():
    import test_review_part_76 as VT
    manifest, block = VT._block()
    key = next(iter(block['comparisons']))
    for value in (1, np.bool_(True), [1, 2], 'x', None):
        bad = copy.deepcopy(block)
        bad['comparisons'][key] = value
        verdict = L.replay_validation_transfer(bad, manifest=manifest)
        assert verdict['status'] == 'fail', (value, verdict)
        assert verdict['problems'], (value, verdict)
        _strict(verdict)
    # A huge integer in a stored scalar used to raise OverflowError while
    # comparing it with the finite replay value.
    bad = copy.deepcopy(block)
    bad['comparisons'][key]['validation_mean_log_ratio'] = HUGE
    verdict = L.replay_validation_transfer(bad, manifest=manifest)
    assert verdict['status'] == 'fail', verdict
    _strict(verdict)


def test_multiverse_accepts_numpy_numeric_scalars_but_not_numpy_bools():
    specs = []
    for k in range(3):
        specs.append({
            'name': f's{k}', 'family': 'f',
            'deltas': {np.int64(i): np.float64(1.0 + 0.01 * i + 0.001 * k)
                       for i in range(6)},
            'features': {'level': np.int64(k),
                         'offset': np.float32(0.5)},
        })
    art = L.build_analysis_multiverse(specs, primary='s0')
    assert art['result']['status'] in ('ok', 'warnings'), art['result']
    assert art['inputs']['specs'][0]['deltas'][0][0] == 0
    assert L.validate_analysis_multiverse(art)['status'] in ('ok', 'warnings')
    for flag in (np.bool_(True), np.bool_(False)):
        for fn in (L._am_number, L._am_int, L._am_seed):
            with pytest.raises((ValueError, TypeError)):
                fn(flag, where='flag')
