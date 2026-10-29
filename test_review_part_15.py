"""CPU review checks (split group 15): prospective run allocation.

The run-order design already randomizes *when* a cell runs.  These checks cover
the two remaining preparation variables: the plan must be written before the
first cell (not reconstructed afterwards), and the arm must not be collinear
with the physical device/host it ran on.
"""
import inspect
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _env(device, *, host='node-a', torch='2.8.0', cuda='12.4'):
    return {
        'hostname': host,
        'cuda_available': True,
        'cuda_visible_devices': device,
        'gpu_identity': [{
            'name': 'NVIDIA A100-SXM4-80GB', 'major': 8, 'minor': 0,
            'total_memory_bytes': 80 * (1 << 30),
        }],
        'torch': torch,
        'cuda_version': cuda,
        'cudnn_version': '9.0.0',
        'determinism': 'strict',
        'tf32_matmul': False,
        'float32_matmul_precision': 'highest',
    }


def _record(variant, seed, device, *, host='node-a', torch='2.8.0'):
    return {
        'variant': variant, 'seed': seed,
        'provenance': {'environment': _env(device, host=host, torch=torch)},
    }


def test_environment_slots_are_balanced_and_cover_the_plan():
    variants = ['a', 'b', 'c', 'd']
    cells = [(v, s) for s in range(4) for v in variants]
    plan = L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=11,
        environment_slots=['gpu0', 'gpu1'])
    assert plan['environment_slots'] == ['gpu0', 'gpu1']
    assert set(plan['cell_environments']) == set(plan['planned_cells'])
    assert plan['design']['environment_balance'] == 'cyclic_slot_rotation'
    # Every arm visits both slots equally often, so slot is not collinear with
    # arm.  The old "rotate the block and the position together" idea would
    # cancel and put each arm on one slot; this asserts the cancellation is
    # gone.
    for v in variants:
        counts = {s: 0 for s in plan['environment_slots']}
        for lab, slot in plan['cell_environments'].items():
            if plan['cell_variants'][lab] == v:
                counts[slot] += 1
        assert sorted(counts.values()) == [2, 2], (v, counts)
    rep = L.environment_plan_audit(plan)
    assert rep['status'] == 'ok' and rep['slot_balanced'] is True, rep
    assert all(c == 2 for c in rep['arm_slot_counts']['a'].values()), rep
    # No slots means no assignment, and the audit says "not declared" rather
    # than inventing a balanced device design.
    plain = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=11)
    assert 'cell_environments' not in plain and 'environment_slots' not in plain
    assert L.environment_plan_audit(plain)['status'] == 'not_declared'
    # Malformed slots fail closed.
    for bad in ([], ['gpu0', 'gpu0'], [''], [None], [True], 'gpu0'):
        try:
            L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=1,
                                   environment_slots=bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'bad environment slots accepted: {bad!r}')
    # A resume rebuilds the same plan with a fresh timestamp.  The first file
    # is the precommitment and must not be overwritten by that rebuild.
    original = L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=11,
        created_at='2026-01-01T00:00:00', environment_slots=['gpu0', 'gpu1'])
    rebuilt = L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=11,
        created_at='2026-02-01T00:00:00', environment_slots=['gpu0', 'gpu1'])
    with tempfile.TemporaryDirectory() as d:
        path = L.write_run_order(d, original)
        L.write_run_order(d, rebuilt)
        with open(path, encoding='utf-8') as fh:
            kept = json.load(fh)
        assert kept['created_at'] == '2026-01-01T00:00:00', kept
        assert kept['cell_environments'] == original['cell_environments']
        # A genuinely different design is not silently frozen out.
        other = L.build_run_order_plan(
            [('a', 0), ('b', 0)], block_key=lambda c: c[1], seed=99,
            created_at='2026-03-01T00:00:00')
        L.write_run_order(d, other)
        with open(path, encoding='utf-8') as fh:
            replaced = json.load(fh)
        assert replaced['created_at'] == '2026-03-01T00:00:00', replaced
        assert replaced['planned_cells'] == other['planned_cells']


def test_environment_plan_audit_detects_incomplete_or_unbalanced_assignment():
    variants = ['a', 'b']
    cells = [(v, s) for s in range(2) for v in variants]
    plan = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=3,
                                  environment_slots=['gpu0'])
    assert L.environment_plan_audit(plan)['status'] == 'ok'
    broken = json.loads(json.dumps(plan))
    first = broken['planned_cells'][0]
    broken['cell_environments'].pop(first)
    rep = L.environment_plan_audit(broken)
    assert rep['status'] == 'fail' and any(
        'no environment assignment' in p for p in rep['problems']), rep
    wrong = json.loads(json.dumps(plan))
    wrong['cell_environments'][first] = 'gpu9'
    rep = L.environment_plan_audit(wrong)
    assert rep['status'] == 'fail' and any(
        'undeclared slot' in p for p in rep['problems']), rep
    # An arm concentrated on one slot (here both a-cells are forced to gpu0
    # while b-cell counts are unchanged) must not be called balanced.
    skewed = json.loads(json.dumps(plan))
    slots = ['gpu0', 'gpu1']
    skewed['environment_slots'] = slots
    for i, lab in enumerate(skewed['planned_cells']):
        skewed['cell_environments'][lab] = slots[i % 2]
    for lab in skewed['planned_cells']:
        if skewed['cell_variants'][lab] == 'a':
            skewed['cell_environments'][lab] = 'gpu0'
    rep = L.environment_plan_audit(skewed)
    assert rep['status'] == 'fail' and rep['slot_balanced'] is False, rep
    orphan = json.loads(json.dumps(plan))
    orphan.pop('environment_slots')
    rep = L.environment_plan_audit(orphan)
    assert rep['status'] == 'fail' and any(
        'cell_environments is present' in p for p in rep['problems']), rep


def test_environment_balance_audit_detects_complete_confounding():
    # Balanced: each arm sees both devices across the seeds.
    balanced = {}
    for s in range(4):
        balanced[f'a::s{s}'] = _record('a', s, '0' if s % 2 == 0 else '1')
        balanced[f'b::s{s}'] = _record('b', s, '1' if s % 2 == 0 else '0')
    rep = L.environment_balance_audit(balanced)
    assert rep['status'] == 'ok', rep
    assert rep['n_environments'] == 2 and rep['confounded_pairs'] == [], rep
    assert set(rep['arm_environment_counts']['a'].values()) == {2}, rep

    # Complete confound: arm a only on device 0, arm b only on device 1.
    confounded = {}
    for s in range(4):
        confounded[f'a::s{s}'] = _record('a', s, '0')
        confounded[f'b::s{s}'] = _record('b', s, '1')
    rep = L.environment_balance_audit(confounded)
    assert rep['status'] == 'fail', rep
    assert rep['confounded_pairs'] == [['a', 'b']], rep
    assert any('perfectly confounded' in p for p in rep['problems']), rep

    # A single observation per arm is not enough to call a confound.
    sparse = {'a::s0': _record('a', 0, '0'), 'b::s0': _record('b', 0, '1')}
    rep = L.environment_balance_audit(sparse)
    assert rep['status'] == 'not_verifiable_locally', rep

    # Missing environment metadata is never a green pass.
    missing = {'a::s0': {'variant': 'a', 'seed': 0},
               'b::s0': {'variant': 'b', 'seed': 0}}
    rep = L.environment_balance_audit(missing)
    assert rep['status'] == 'not_verifiable_locally', rep

    # A different host/driver is a different environment, so two arms on the
    # same device token but different hosts are also completely confounded.
    hosts = {}
    for s in range(3):
        hosts[f'a::s{s}'] = _record('a', s, '0', host='node-a')
        hosts[f'b::s{s}'] = _record('b', s, '0', host='node-b')
    rep = L.environment_balance_audit(hosts)
    assert rep['status'] == 'fail' and rep['confounded_pairs'] == [['a', 'b']], rep


def test_run_order_plan_timing_requires_a_prospective_plan():
    cells = [('a', 0), ('b', 0)]
    plan = L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=1,
                                  created_at='2026-01-01T00:00:00')
    ok = L.run_order_plan_timing(
        plan, [{'kind': 'cell', 'started_at': '2026-01-01T00:01:00'},
               {'kind': 'cell', 'started_at': '2026-01-01T00:02:00'}])
    assert ok['status'] == 'ok' and ok['plan_created_at'] == '2026-01-01T00:00:00', ok
    late = L.run_order_plan_timing(
        plan, [{'kind': 'cell', 'started_at': '2025-12-31T23:59:00'}])
    assert late['status'] == 'fail', late
    assert any('after execution' in p for p in late['problems']), late
    # A plan without a timestamp is not a precommitment; it is not green.
    no_ts = json.loads(json.dumps(plan))
    no_ts.pop('created_at')
    rep = L.run_order_plan_timing(no_ts, [{'kind': 'cell', 'started_at': '2026-01-01T00:01:00'}])
    assert rep['status'] == 'not_verifiable_locally', rep
    # Malformed timestamps fail closed; no matching events is unverifiable.
    bad = json.loads(json.dumps(plan))
    bad['created_at'] = 'not-a-date'
    assert L.run_order_plan_timing(
        bad, [{'kind': 'cell', 'started_at': '2026-01-01T00:01:00'}])['status'] == 'fail'
    assert L.run_order_plan_timing(plan, [])['status'] == 'not_verifiable_locally'
    # A different timezone convention is not silently compared.
    aware = json.loads(json.dumps(plan))
    aware['created_at'] = '2026-01-01T00:00:00+00:00'
    rep = L.run_order_plan_timing(
        aware, [{'kind': 'cell', 'started_at': '2026-01-01T00:01:00'}])
    assert rep['status'] == 'not_verifiable_locally', rep


def test_run_allocation_audit_wiring_and_panel_status():
    variants = ['a', 'b']
    cells = [(v, s) for s in range(2) for v in variants]
    plan = L.build_run_order_plan(
        cells, block_key=lambda c: c[1], seed=5,
        created_at='2026-01-01T00:00:00', environment_slots=['0', '1'])
    summary = {}
    for lab in plan['planned_cells']:
        v, s = lab.rsplit('::seed', 1)
        summary[lab] = _record(v, int(s), plan['cell_environments'][lab])
    events = [
        {'kind': 'cell', 'status': 'ok', 'variant': lab.rsplit('::seed', 1)[0],
         'seed': int(lab.rsplit('::seed', 1)[1]),
         'started_at': f'2026-01-01T00:0{i + 1}:00'}
        for i, lab in enumerate(plan['planned_cells'])
    ]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'run_order.json'), 'w', encoding='utf-8') as f:
            json.dump(plan, f)
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        with open(os.path.join(d, 'experiments.jsonl'), 'w', encoding='utf-8') as f:
            for e in events:
                f.write(json.dumps(e) + '\n')
        rep = A.audit_run_allocation([d])
        rec = rep['panels'][os.path.basename(d)]
        assert rep['status'] == 'ok' and rec['status'] == 'ok', rec
        assert rec['timing']['status'] == 'ok'
        assert rec['environment_plan']['status'] == 'ok'
        assert rec['environment_actual']['status'] == 'ok'
        # A post-hoc plan is a hard failure even when the order/env are valid.
        late = json.loads(json.dumps(plan))
        late['created_at'] = '2026-01-02T00:00:00'
        with open(os.path.join(d, 'run_order.json'), 'w', encoding='utf-8') as f:
            json.dump(late, f)
        rep = A.audit_run_allocation([d])
        assert rep['status'] == 'fail', rep
        assert any('after execution' in p
                   for p in rep['panels'][os.path.basename(d)]['problems']), rep
        # A confounded environment fails even when the plan is prospective.
        summary = {}
        for lab in plan['planned_cells']:
            v, s = lab.rsplit('::seed', 1)
            summary[lab] = _record(v, int(s), '0' if v == 'a' else '1')
        with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        with open(os.path.join(d, 'run_order.json'), 'w', encoding='utf-8') as f:
            json.dump(plan, f)
        rep = A.audit_run_allocation([d])
        assert rep['status'] == 'fail', rep
        assert rep['panels'][os.path.basename(d)]['environment_actual'][
            'confounded_pairs'] == [['a', 'b']]
    # A directory with neither a plan nor records is unverifiable, not green.
    with tempfile.TemporaryDirectory() as empty:
        rep = A.audit_run_allocation([empty])
        assert rep['status'] == 'not_verifiable_locally', rep
    assert 'run_allocation' in (A.__doc__ or '')
    assert 'run_allocation' in inspect.getsource(A._run_cli)
    assert "'run_allocation': audit_run_allocation()" in \
        inspect.getsource(A.run_report)
