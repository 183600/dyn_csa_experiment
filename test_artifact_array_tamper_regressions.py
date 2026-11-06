"""Array-valued artifact fields must fail closed, never raise.

A stored artifact is untrusted, and an in-memory caller has not necessarily
round-tripped through JSON.  A NumPy array in a scalar slot makes ``a != b``
return an array whose truth value is ambiguous, so a validator that compares
that slot directly raises ``ValueError`` instead of returning its structured
``fail`` verdict.  These checks cover the replayed validators whose comparisons
were hardened for this case.

CPU-only, zero-experiment checks.
"""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403
import test_review_part_76 as VT

REPO = os.path.dirname(os.path.abspath(__file__))


def _strict(value):
    json.dumps(value, allow_nan=False)
    return value


def _set_path(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _assert_fails(fn, artifact, path):
    bad = copy.deepcopy(artifact)
    _set_path(bad, path, np.array([1, 2]))
    verdict = fn(bad)
    assert verdict['status'] == 'fail', (path, verdict)
    assert verdict['problems'], (path, verdict)
    _strict(verdict)


def test_validation_transfer_array_fields_fail_closed():
    manifest, block = VT._block()
    key = block['family'][0]
    for path in (
            ('kind',),
            ('comparisons', key, 'name'),
            ('comparisons', key, 'n'),
            ('comparisons', key, 'classification'),
            ('holm', key, 'reject'),
            ('summary', 'class_counts'),
            ('summary', 'family_holm_rejections', 0),
    ):
        bad = copy.deepcopy(block)
        _set_path(bad, path, np.array([1, 2]))
        verdict = L.replay_validation_transfer(bad, manifest=manifest)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        _strict(verdict)


def test_exact_signflip_ci_array_schema_fields_fail_closed():
    art = L.build_exact_signflip_ci([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
    for path in (('kind',), ('schema',), ('result', 'kind'),
                 ('result', 'version'), ('result', 'status')):
        _assert_fails(L.validate_exact_signflip_ci, art, path)


def test_curve_inference_array_kind_and_status_fail_closed():
    conds = ['a', 'b', 'c']
    effects = {c: [0.1 * (i + 1) + 0.01 * k for i in range(6)]
               for k, c in enumerate(conds)}
    art = L.build_curve_inference(conds, effects)
    for path in (('kind',), ('result', 'kind'), ('result', 'status')):
        _assert_fails(L.validate_curve_inference, art, path)


def test_sequential_and_budget_ablation_array_schema_fail_closed():
    seq = L.build_sequential_design(
        n_max=20, look_sizes=[10, 20], pilot_sign_rate=0.7,
        target_power=0.6)
    for path in (('kind',), ('schema',)):
        _assert_fails(L.validate_sequential_design, seq, path)

    design = {'budgets': [1, 2],
              'components': [{'name': 'c', 'with_arm': 'a',
                              'without_arm': 'b'}]}
    cells = [{'arm': arm, 'seed': s, 'budget': budget,
              'ppl': 10.0 + 0.1 * s * (1.0 if arm == 'a' else 1.1)}
             for arm in ('a', 'b') for s in range(4) for budget in (1, 2)]
    ba = L.build_budget_ablation(design, cells, min_n=2)
    for path in (('kind',), ('schema',)):
        _assert_fails(L.validate_budget_ablation, ba, path)


def test_unit_split_array_nested_fields_fail_closed():
    units = [f'u{i:03d}' for i in range(8)]
    tokens = [100.0 + 7.0 * i for i in range(8)]
    rows = [[-0.20 + 0.01 * math.sin(i + k) for i in range(8)]
            for k in range(6)]
    art = L.build_unit_split_replication(
        rows, tokens, unit_ids=units, seeds=list(range(6)),
        direction='negative', n_salts=0)
    for path in (
            ('kind',),
            ('inputs', 'unit_ids', 0),
            ('inputs', 'n_salts'),
            ('result', 'sign_agreement'),
            ('result', 'assignment', 'halves', 0, 'name'),
            ('result', 'iut', 'claim'),
    ):
        _assert_fails(L.validate_unit_split_replication, art, path)
