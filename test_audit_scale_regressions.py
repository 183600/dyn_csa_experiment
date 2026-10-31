"""Scale/robustness regressions for the audit summaries.

Three summaries recomputed a per-item scan once per arm, so a large arm grid
turned a linear report into O(n_arms * n_cells).  The two preparation-clearance
cases cover untrusted JSON shapes that used to abort the launch gate instead of
recording a named failure.

These are zero-GPU checks.
"""
import copy
import json
import os
import sys as _sys

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import audit_gates as A
import test_review_part_72 as T


def _disposition_events(cells):
    return [{
        'kind': 'cell', 'key': cell, 'variant': cell.split('::seed')[0],
        'seed': 0, 'status': 'ok', 'run_id': f'r{cell}',
        'started_at': '2026-01-01T00:00:00',
        'finished_at': '2026-01-01T00:00:00',
    } for cell in cells]


def test_disposition_large_arm_grid_keeps_the_exact_attempt_map():
    n = 5000
    cells = [f'arm{i}::seed0' for i in range(n)]
    plan = L.build_run_disposition_plan(
        cells, created_at='2020-01-01T00:00:00')
    kept = {cell: {'run_id': f'r{cell}'} for cell in cells}
    rep = L.run_disposition_audit(
        _disposition_events(cells), plan=plan, retained=kept)
    assert rep['status'] == 'ok', rep
    attempts = rep['differential_attempts']['max_attempts_per_cell_by_arm']
    assert len(attempts) == n, len(attempts)
    assert set(attempts.values()) == {1}, set(attempts.values())
    # Arm keys keep the historical lexicographic order.
    assert list(attempts)[:3] == ['arm0', 'arm1', 'arm10'], list(attempts)[:3]


def _confounded_pairs_exhaustive(records, min_per_arm=None):
    policy = L.ENVIRONMENT_BALANCE_POLICY
    minimum = int(policy['min_records_per_arm']
                  if min_per_arm is None else min_per_arm)
    counts = {}
    for _label, rec in L._iter_allocation_records(records):
        variant = str(rec.get('variant'))
        counts.setdefault(variant, {})
        ident = L.environment_identity(L._record_environment(rec))
        if ident is None:
            continue
        key = L._environment_key(ident)
        counts[variant][key] = counts[variant].get(key, 0) + 1
    measured = [v for v, c in counts.items() if sum(c.values()) >= minimum]
    pairs = []
    for i, a in enumerate(measured):
        ea = set(counts[a])
        for b in measured[i + 1:]:
            eb = set(counts[b])
            if ea and eb and ea.isdisjoint(eb):
                pairs.append([a, b])
    return pairs


def test_environment_balance_pairs_match_the_exhaustive_reference():
    for shift in range(6):
        records = {}
        for a in range(12):
            for s in range(2):
                env = 'same' if a % 3 else f'only-{(a + shift) % 4}'
                records[f'a{a}::seed{s}'] = {
                    'variant': f'a{a}', 'seed': s,
                    'environment': {
                        'hostname': env, 'cuda_available': False,
                        'torch': '2', 'cuda_version': '12',
                    },
                }
        got = L.environment_balance_audit(records)['confounded_pairs']
        assert got == _confounded_pairs_exhaustive(records), (shift, got)


def test_environment_balance_large_shared_grid_has_no_confounded_pair():
    n = 5000
    records = {}
    for a in range(n):
        for s in range(2):
            records[f'arm{a}::seed{s}'] = {
                'variant': f'arm{a}', 'seed': s,
                'environment': {
                    'hostname': 'worker', 'cuda_available': True,
                    'gpus': ['A', 'B'], 'torch': '2', 'cuda_version': '12',
                },
            }
    rep = L.environment_balance_audit(records)
    assert rep['status'] == 'ok', rep
    assert rep['confounded_pairs'] == [], rep['confounded_pairs']
    assert rep['n_arms'] == n and rep['n_environments'] == 1, rep


def test_controlled_variables_report_counts_every_record_once():
    n = 3000
    recs = {}
    for a in range(n):
        for s in range(3):
            recs[(f'arm{a}', '', s)] = {
                'variant': f'arm{a}', 'seed': s, 'run_cfg': 'cfg',
                'train_knobs': None, 'hyperparams': {'steps': 10},
            }
    rep = L.controlled_variables_report(recs, [f'arm{a}' for a in range(n)])
    assert rep['n_records'] == {f'arm{a}': 3 for a in range(n)}
    assert rep['run_cfg_equal'] is True, rep


def test_preparation_clearance_tolerates_non_list_planned_cells():
    p = T._protocol()
    manifest = T._manifest()
    base = dict(
        run_order=T._order(manifest), disposition=T._disposition(manifest),
        power=T._power(manifest), source_tree=T._tree(),
        rehearsal={'ok': True}, test_access_state=False,
        dependency_contract=T._dependency())
    for field, bad in (('run_order', 3), ('run_order', True),
                       ('randomization', {'planned_cells': 3})):
        kwargs = copy.deepcopy(base)
        if field == 'run_order':
            kwargs['run_order']['planned_cells'] = bad
            proto = p
        else:
            proto = copy.deepcopy(p)
            proto['randomization'] = bad
        rep = L.build_preparation_clearance(proto, **kwargs)
        assert rep['status'] == 'fail', (field, bad, rep['status'])
        json.dumps(rep, allow_nan=False)


def test_dependency_contract_with_malformed_detail_fails_closed():
    p = T._protocol()
    m = T._manifest()
    kwargs = dict(
        run_order=T._order(m), disposition=T._disposition(m),
        power=T._power(m), source_tree=T._tree(), rehearsal={'ok': True},
        test_access_state=False)
    for dependency in (
            {'status': 'ok', 'lock_sha256': 'a' * 64,
             'mismatches': {'torch': '1'}},
            {'status': 'ok', 'lock_sha256': 'a' * 64, 'problems': 7},
            {'status': 'fail', 'lock_sha256': 'a' * 64,
             'mismatches': {'torch': '1'}}):
        rep = L.build_preparation_clearance(
            p, dependency_contract=dependency, **kwargs)
        assert rep['status'] == 'fail', (dependency, rep['status'])
        assert rep['checks']['dependency']['status'] == 'fail', dependency
        json.dumps(rep, allow_nan=False)


def test_unique_int_set_matches_set_semantics_for_sidecars():
    arr = np.array([[3, 1], [1, 3], [2, 2]], dtype=np.uint64)
    assert A._unique_int_set(arr) == {1, 2, 3}
    assert A._unique_int_set(np.array([], dtype=np.int64)) == set()
    # Object arrays keep the historical int() conversion path.
    obj = np.array([[1, '2']], dtype=object)
    assert A._unique_int_set(obj) == {1, 2}


def test_clearance_audit_tolerates_non_object_json_artifacts():
    # A stored artifact is untrusted JSON.  A top-level list/scalar is a
    # malformed clearance/protocol, not an object the replay helpers may call
    # ``.get()`` on: the audit must report a failed validation instead of
    # raising AttributeError.
    import tempfile

    proto = T._protocol()
    clearance = T._clearance(proto)
    for bad_clearance in ([], 'not-an-object', 3, True):
        for bad_protocol in (proto, [], 'not-an-object', 3, True):
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, 'preparation_clearance.json'),
                          'w', encoding='utf-8') as fh:
                    json.dump(bad_clearance, fh)
                with open(os.path.join(td, 'design_protocol.json'),
                          'w', encoding='utf-8') as fh:
                    json.dump(bad_protocol, fh)
                rep = L.preparation_clearance_audit(
                    path=os.path.join(td, 'preparation_clearance.json'),
                    protocol_path=os.path.join(td, 'design_protocol.json'),
                    events=[])
                assert rep['status'] == 'fail', (bad_clearance,
                                                 bad_protocol, rep)
                json.dumps(rep, allow_nan=False)


def test_aggregate_fallback_keeps_per_variant_order(monkeypatch):
    # The pairing-failure fallback used to re-filter the whole sub-panel for
    # every variant.  It must still emit the same per-arm keys and preserve
    # the admitted records when the strict cross-variant gate refuses.
    import contextlib
    import io

    base = {'run_cfg': 'cfg1', 'steps': 10, 'tokens_seen': 1000,
            'params': 100, '_code': 'v1'}
    summary = {}
    for variant, ppl in (('a', 10.0), ('b', 13.0)):
        for seed in (0, 1):
            summary[f'{variant}::s{seed}'] = dict(
                base, variant=variant, seed=seed, ppl=ppl + seed,
                nll_sum=(ppl + seed) * 1000, n_tokens=1000)
    real_aggregate = L.aggregate

    def _fake_aggregate(sub):
        if len({r['variant'] for r in sub.values()}) > 1:
            raise ValueError('forced cross-variant pairing failure')
        return real_aggregate(sub)

    monkeypatch.setattr(L, 'aggregate', _fake_aggregate)
    with contextlib.redirect_stdout(io.StringIO()):
        agg, report, admitted = L.aggregate_with_exclusions(summary)
    assert list(agg) == ['a@arma', 'b@armb'], list(agg)
    assert len(admitted) == 4, admitted
    assert report['pairing_split_groups'][0]['variants'] == ['a', 'b'], report
    json.dumps(report, allow_nan=False)
