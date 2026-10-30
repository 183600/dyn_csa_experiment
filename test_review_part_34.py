"""CPU checks for pre-registered run disposition and the attempt ledger.

A planned ``(arm, seed)`` cell is not the same object as a measured cell.  These
checks pin the design-time contract (which failures may be replaced, how many
attempts a cell gets, which success is binding) and the audit that joins it to
``experiments.jsonl``.  No test trains a model, reads a result panel or a
held-out split.
"""
import contextlib
import io
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _ev(key, status, rid, *, cfg='c1', error=None, variant=None, seed=None,
        failure_class=None, started_at=None):
    arm, _, seed_text = key.rpartition('::seed')
    out = {'kind': 'cell', 'key': key,
           'variant': variant if variant is not None else arm,
           'seed': int(seed_text) if seed is None else seed,
           'status': status, 'run_id': rid, 'run_cfg': cfg}
    if error is not None:
        out['error'] = error
    if failure_class is not None:
        out['failure_class'] = failure_class
    if started_at is not None:
        out['started_at'] = started_at
    return out


def _plan(cells, **policy):
    return L.build_run_disposition_plan(cells, policy=policy or None)


def _retained(**rows):
    return {k: {'run_id': v[0], 'run_cfg': v[1] if len(v) > 1 else 'c1'}
            for k, v in rows.items()}


def _write_plan(td, cells, **policy):
    return L.write_run_disposition(
        td, L.build_run_disposition_plan(cells, policy=policy or None))


def _log(td, event):
    L.log_experiment_event(td, event)


def _write_summary(td, mapping):
    L.atomic_write_json(os.path.join(td, 'summary.json'), mapping)


def _raises(fn, why):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(why)


CELLS = ['a::seed0', 'b::seed0', 'a::seed1', 'b::seed1']


def test_clean_single_attempt_panel_is_ok():
    plan = _plan(CELLS)
    events = [_ev('a::seed0', 'ok', 'r1'), _ev('b::seed0', 'ok', 'r2'),
              _ev('a::seed1', 'ok', 'r3'), _ev('b::seed1', 'ok', 'r4')]
    ret = _retained(**{'a::seed0': ('r1',), 'b::seed0': ('r2',),
                       'a::seed1': ('r3',), 'b::seed1': ('r4',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    assert r['status'] == 'ok', r['problems'] + r['warnings']
    assert r['frame'] == 'precommitted'
    assert r['n_planned'] == 4 and r['n_attempted'] == 4 and r['n_measured'] == 4
    assert r['n_attempts'] == 4 and r['n_cells_retried'] == 0
    assert r['n_unplanned_measurements'] == 0
    assert r['differential_attempts']['status'] == 'no_retries'
    assert r['cells']['a::seed0']['retained_is_first_success'] is True
    assert r['cells']['a::seed0']['status'] == 'measured'


def test_infrastructure_retry_is_allowed_but_named():
    plan = _plan(['a::seed0', 'b::seed0'])
    events = [_ev('a::seed0', 'error', 'r1', error='CUDA out of memory'),
              _ev('a::seed0', 'ok', 'r2'),
              _ev('b::seed0', 'ok', 'r3')]
    ret = _retained(**{'a::seed0': ('r2',), 'b::seed0': ('r3',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    assert r['status'] == 'warn', r
    assert r['problems'] == []
    assert r['n_cells_retried'] == 1
    assert r['n_non_replaceable_retried'] == 0
    fc = r['failure_classes']['resource_exhausted']
    assert fc['n_attempts'] == 1 and fc['n_cells'] == 1 and fc['n_retried'] == 1
    assert fc['replaceable'] is True
    assert r['by_arm']['a']['n_attempts'] == 2
    assert r['by_arm']['b']['n_attempts'] == 1
    assert r['cells']['a::seed0']['retried_after_failure'] is True
    # One retry across two arms is necessarily concentrated; it must be named
    # as a descriptive imbalance, not silently treated as balanced.
    assert r['differential_attempts']['status'] == 'concentrated'
    assert any('concentrated' in w for w in r['warnings'])


def test_retained_second_success_is_post_hoc_selection():
    plan = _plan(['a::seed0', 'b::seed0'])
    events = [_ev('a::seed0', 'ok', 'r1'), _ev('a::seed0', 'ok', 'r2'),
              _ev('b::seed0', 'ok', 'r3')]
    ret = _retained(**{'a::seed0': ('r2',), 'b::seed0': ('r3',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    assert r['status'] == 'fail', r
    assert r['n_retained_second_success'] == 1
    assert r['cells']['a::seed0']['retained_is_first_success'] is False
    assert any('post-hoc selection' in p for p in r['problems'])


def test_retained_first_success_with_a_repeat_is_not_a_selection():
    plan = _plan(['a::seed0', 'b::seed0'])
    events = [_ev('a::seed0', 'ok', 'r1'), _ev('a::seed0', 'ok', 'r2'),
              _ev('b::seed0', 'ok', 'r3')]
    ret = _retained(**{'a::seed0': ('r1',), 'b::seed0': ('r3',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    assert r['status'] == 'warn', r
    assert r['problems'] == []
    assert r['n_multiple_success_cells'] == 1
    assert r['cells']['a::seed0']['retained_is_first_success'] is True
    assert any('first success was retained' in w for w in r['warnings'])


def test_unplanned_measurement_and_unplanned_attempt_fail():
    plan = _plan(['a::seed0', 'b::seed0'])
    events = [_ev('a::seed0', 'ok', 'r1'), _ev('b::seed0', 'ok', 'r2'),
              _ev('c::seed1', 'ok', 'r3')]
    ret = _retained(**{'a::seed0': ('r1',), 'b::seed0': ('r2',),
                       'c::seed1': ('r3',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    assert r['status'] == 'fail'
    assert r['n_unplanned_measurements'] == 1
    assert r['n_unplanned_attempts'] == 1
    assert any('c::seed1' in p and 'not in the pre-registered plan' in p
               for p in r['problems'])


def test_seed_key_mismatch_and_duplicate_run_id_fail():
    with tempfile.TemporaryDirectory() as td:
        plan = _plan(['a::seed0'])
        _write_plan(td, ['a::seed0'])
    # A key/seed mismatch is a malformed attempt, not a second seed.
    _raises(lambda: L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'r1', seed=1)], plan=plan),
        'key/seed mismatch must be rejected')
    # A run_id cannot identify two attempts.
    r = L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'dup'), _ev('a::seed0', 'ok', 'dup')],
        plan=_plan(['a::seed0']))
    assert r['status'] == 'fail'
    assert any('duplicate run_id' in p for p in r['problems'])


def test_non_replaceable_and_unclassified_failures_cannot_be_retried():
    plan = _plan(['a::seed0'])
    nan = L.run_disposition_audit(
        [_ev('a::seed0', 'error', 'r1', error='loss is NaN'),
         _ev('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': ('r2',)}))
    assert nan['status'] == 'fail'
    assert nan['n_non_replaceable_retried'] == 1
    assert any('non-replaceable' in p for p in nan['problems'])
    unknown = L.run_disposition_audit(
        [_ev('a::seed0', 'error', 'r1', error='kaboom'),
         _ev('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': ('r2',)}))
    assert unknown['status'] == 'fail'
    assert unknown['n_unclassified_failures'] == 1
    assert any('unknown class' in p for p in unknown['problems'])


def test_attempt_budget_and_total_budget_fail():
    events = [_ev('a::seed0', 'error', 'r1', error='timeout'),
              _ev('a::seed0', 'error', 'r2', error='timeout'),
              _ev('a::seed0', 'ok', 'r3')]
    r = L.run_disposition_audit(events, plan=_plan(['a::seed0']),
                                retained=_retained(**{'a::seed0': ('r3',)}))
    assert r['status'] == 'fail'
    assert r['n_attempt_budget_exceeded'] == 1
    assert any('maximum is 2' in p for p in r['problems'])
    r2 = L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'r1'), _ev('b::seed0', 'ok', 'r2')],
        plan=_plan(['a::seed0', 'b::seed0'], max_total_attempts=1),
        retained=_retained(**{'a::seed0': ('r1',), 'b::seed0': ('r2',)}))
    assert r2['status'] == 'fail'
    assert any('at most 1' in p for p in r2['problems'])


def test_superseding_a_measured_cell_needs_an_explicit_policy():
    events = [_ev('a::seed0', 'ok', 'r1', cfg='codeA'),
              _ev('a::seed0', 'ok', 'r2', cfg='codeB')]
    ret = _retained(**{'a::seed0': ('r2', 'codeB')})
    strict = L.run_disposition_audit(events, plan=_plan(['a::seed0']),
                                     retained=ret)
    assert strict['status'] == 'fail'
    assert strict['n_superseded_cells'] == 1
    assert any('does not allow superseding' in p for p in strict['problems'])
    loose = L.run_disposition_audit(
        events,
        plan=_plan(['a::seed0'], allow_supersede=True,
                   retention='last_success'),
        retained=ret)
    assert loose['status'] in ('warn', 'ok'), loose
    assert loose['n_superseded_cells'] == 1
    assert any('allows supersession' in w for w in loose['warnings'])


def test_plan_writer_binds_the_first_precommitment():
    with tempfile.TemporaryDirectory() as td:
        first = L.build_run_disposition_plan(['a::seed0', 'b::seed0'])
        out = L.write_run_disposition(td, first)
        assert out['written'] is True and out['conflict'] is False
        same = L.write_run_disposition(
            td, L.build_run_disposition_plan(['a::seed0', 'b::seed0']))
        assert same['written'] is False and same['conflict'] is False
        changed = L.build_run_disposition_plan(['a::seed0'])
        keep = L.write_run_disposition(td, changed, keep_existing=True)
        assert keep['conflict'] is True and keep['written'] is False
        _raises(lambda: L.write_run_disposition(td, changed),
                'a silent precommitment rewrite must be refused')
        old = L.build_run_disposition_plan(['a::seed0', 'b::seed0'])
        revised = L.build_run_disposition_plan(
            ['a::seed0'], revision=1,
            supersedes=[old['content_sha256']], reason='drop a duplicated arm')
        out2 = L.write_run_disposition(td, revised)
        assert out2['written'] is True and out2['revision'] == 1
        with open(os.path.join(td, 'run_disposition.json'), encoding='utf-8') as f:
            stored = json.load(f)
        assert stored['revision'] == 1
        assert stored['supersedes'] == [old['content_sha256']]
        assert stored['reason'] == 'drop a duplicated arm'


def test_retry_concentration_is_descriptive_and_leaves_ok_alone():
    cells = ['a::seed0', 'b::seed0', 'c::seed0']
    plan = _plan(cells)
    events = [_ev('a::seed0', 'ok', 'r1'),
              _ev('b::seed0', 'error', 'r2', error='network reset'),
              _ev('b::seed0', 'ok', 'r3'),
              _ev('c::seed0', 'ok', 'r4')]
    ret = _retained(**{'a::seed0': ('r1',), 'b::seed0': ('r3',),
                       'c::seed0': ('r4',)})
    r = L.run_disposition_audit(events, plan=plan, retained=ret)
    d = r['differential_attempts']
    assert d['status'] == 'concentrated'
    assert d['arms_with_retries'] == ['b']
    assert d['max_attempts_per_cell_by_arm'] == {'a': 1, 'b': 2, 'c': 1}
    assert 'role' in d and 'p-value' in d['role']


def test_missing_plan_is_not_verifiable_not_a_pass():
    events = [_ev('a::seed0', 'ok', 'r1')]
    r = L.run_disposition_audit(events, retained=_retained(**{'a::seed0': ('r1',)}))
    assert r['status'] == 'not_verifiable_locally'
    assert r['frame'] == 'observed_only'
    assert any('cannot detect a silently added cell' in w for w in r['warnings'])
    supplied = L.run_disposition_audit(
        events, planned_cells=['a::seed0'],
        retained=_retained(**{'a::seed0': ('r1',)}))
    assert supplied['frame'] == 'caller_supplied'
    assert supplied['status'] == 'warn'


def test_ignored_probe_cells_are_counted_not_forced_into_seed_cells():
    plan = _plan(['a::seed0'])
    events = [_ev('a::seed0', 'ok', 'r1'),
              {'kind': 'cell', 'key': 'a::s0', 'variant': 'a', 'seed': 0,
               'status': 'ok', 'run_id': 'probe1'},
              {'kind': 'cell', 'key': 'a::seed0::niah_train', 'variant': 'a',
               'seed': 0, 'status': 'ok', 'run_id': 'probe2'}]
    r = L.run_disposition_audit(events, plan=plan,
                                retained=_retained(**{'a::seed0': ('r1',)}))
    assert r['n_ignored_cell_events'] == 2
    assert r['status'] == 'warn'
    assert any('non-training key' in w for w in r['warnings'])


def test_cap_statuses_are_classified_without_a_pattern_match():
    plan = _plan(['a::seed0'])
    for status in ('skipped_cap', 'truncated'):
        events = [_ev('a::seed0', status, 'r1'),
                  _ev('a::seed0', 'ok', 'r2')]
        r = L.run_disposition_audit(events, plan=plan,
                                    retained=_retained(**{'a::seed0': ('r2',)}))
        assert r['status'] == 'warn', (status, r)
        assert 'runtime_cap' in r['failure_classes'], (status, r['failure_classes'])
        assert r['failure_classes']['runtime_cap']['replaceable'] is True
        assert r['n_non_replaceable_retried'] == 0


def test_corrupt_or_tampered_plan_is_fail_not_missing_evidence():
    with tempfile.TemporaryDirectory() as td:
        _write_plan(td, ['a::seed0'])
        _log(td, _ev('a::seed0', 'ok', 'r1'))
        _write_summary(td, {'a::seed0': {'run_id': 'r1', 'run_cfg': 'c1',
                                         'ppl': 9.0}})
        plan_path = os.path.join(td, 'run_disposition.json')
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_run_disposition([td])
        assert rep['status'] == 'fail', rep
        assert rep['panels'][td]['plan_unreadable'] is True
    # Valid JSON with a mismatched content hash is tampering, not a plan.
    with tempfile.TemporaryDirectory() as td2:
        plan = L.build_run_disposition_plan(['a::seed0'])
        plan['planned_cells'] = ['b::seed0']
        L.atomic_write_json(os.path.join(td2, 'run_disposition.json'), plan)
        _log(td2, _ev('a::seed0', 'ok', 'r1'))
        _write_summary(td2, {'a::seed0': {'run_id': 'r1', 'run_cfg': 'c1',
                                          'ppl': 9.0}})
        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = A.audit_run_disposition([td2])
        assert rep2['status'] == 'fail', rep2
        assert any('could not run' in p or 'content_sha256' in p
                   for p in rep2['problems'])


def test_plan_written_after_the_first_attempt_is_a_reconstruction():
    cells = ['a::seed0']
    ret = _retained(**{'a::seed0': ('r1',)})
    event = _ev('a::seed0', 'ok', 'r1', started_at='2026-01-01T00:10:00')
    before = L.build_run_disposition_plan(cells, created_at='2026-01-01T00:00:00')
    r = L.run_disposition_audit([event], plan=before, retained=ret)
    assert r['status'] == 'ok', r
    assert r['plan_before_first_attempt'] is True
    after = L.build_run_disposition_plan(cells, created_at='2026-01-01T00:20:00')
    late = L.run_disposition_audit([event], plan=after, retained=ret)
    assert late['status'] == 'fail'
    assert late['plan_before_first_attempt'] is False
    assert any('reconstruction, not a precommitment' in p
               for p in late['problems'])
    forgiving = L.build_run_disposition_plan(
        cells, created_at='2026-01-01T00:20:00',
        policy={'require_plan_before_first_attempt': False})
    soft = L.run_disposition_audit([event], plan=forgiving, retained=ret)
    assert soft['status'] == 'warn', soft
    assert any('reconstruction' in w for w in soft['warnings'])
    # An attempt without a timestamp leaves the relation unverifiable, not
    # silently true and not a failure.
    mixed = L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'r1')], plan=before, retained=ret)
    assert mixed['plan_before_first_attempt'] is None
    assert mixed['status'] == 'ok'


def test_design_preflight_reads_a_disposition_policy():
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with tempfile.TemporaryDirectory() as td:
        good = json.loads(json.dumps(manifest))
        good.setdefault('statistics', {})['disposition_policy'] = {
            'retention': 'first_success', 'replacement': 'same_cell',
            'allow_supersede': False, 'max_total_attempts': 96}
        good_path = os.path.join(td, 'confirmation_frozen.json')
        L.atomic_write_json(good_path, good)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = A.audit_design_preflight(manifest_path=good_path)
        assert ok['status'] == 'ok', ok['violations']
        assert ok['frozen']['disposition_policy']['status'] == 'ok'
        bad = json.loads(json.dumps(manifest))
        bad.setdefault('statistics', {})['disposition_policy'] = {
            'retention': 'last_success', 'allow_supersede': True}
        bad_path = os.path.join(td, 'confirmation_frozen.json')
        L.atomic_write_json(bad_path, bad)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_design_preflight(manifest_path=bad_path)
        assert rep['status'] == 'fail', rep
        assert rep['frozen']['disposition_policy']['status'] == 'fail'
        assert any('first_success' in v for v in rep['violations'])
        assert any('supersed' in v for v in rep['violations'])
        nonobject = json.loads(json.dumps(manifest))
        nonobject.setdefault('statistics', {})['disposition_policy'] = 'first_success'
        L.atomic_write_json(bad_path, nonobject)
        with contextlib.redirect_stdout(io.StringIO()):
            rep2 = A.audit_design_preflight(manifest_path=bad_path)
        assert rep2['status'] == 'fail', rep2
        assert any('must be an object' in v for v in rep2['violations'])


def test_malformed_inputs_fail_closed():
    _raises(lambda: L.build_run_disposition_plan(['a']),
            'a cell must be arm::seedN')
    _raises(lambda: L.build_run_disposition_plan(['a::seed0', 'a::seed0']),
            'duplicate planned cells must be rejected')
    _raises(lambda: L.build_run_disposition_plan(['a::seedx']),
            'a non-integer seed must be rejected')
    _raises(lambda: L.build_run_disposition_plan(
        ['a::seed0'], policy={'max_attempts_per_cell': True}),
        'a boolean attempt budget must be rejected')
    _raises(lambda: L.build_run_disposition_plan(
        ['a::seed0'], policy={'max_attempts_per_cell': 2.0}),
        'a float attempt budget must be rejected')
    _raises(lambda: L.build_run_disposition_plan(
        ['a::seed0'], policy={'replaceable_failure_classes': ['x'],
                              'non_replaceable_failure_classes': ['x']}),
        'overlapping failure classes must be rejected')
    _raises(lambda: L.build_run_disposition_plan(
        ['a::seed0'], policy={'not_a_rule': 1}),
        'an unknown policy key must be rejected')
    _raises(lambda: L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'r1', seed=True)], plan=_plan(['a::seed0'])),
        'a boolean seed must be rejected')
    _raises(lambda: L.run_disposition_audit(
        [_ev('a::seed0', 'mystery', 'r1')], plan=_plan(['a::seed0'])),
        'an unknown status must be rejected')
    _raises(lambda: L.run_disposition_audit([42], plan=_plan(['a::seed0'])),
            'a non-object event must be rejected')
    _raises(lambda: L.run_disposition_audit(
        [_ev('a::seed0', 'ok', 'r1')], plan=_plan(['a::seed0']),
        planned_cells=['b::seed0']),
        'a caller frame that disagrees with the plan must be rejected')


def test_gate_wiring_and_cli_alias():
    with tempfile.TemporaryDirectory() as td:
        _write_plan(td, ['a::seed0', 'b::seed0'])
        _log(td, _ev('a::seed0', 'error', 'r1', error='watchdog timeout'))
        _log(td, _ev('a::seed0', 'ok', 'r2'))
        _log(td, _ev('b::seed0', 'ok', 'r3'))
        _write_summary(td, {
            'a::seed0': {'run_id': 'r2', 'run_cfg': 'c1', 'ppl': 10.0},
            'b::seed0': {'run_id': 'r3', 'run_cfg': 'c1', 'ppl': 11.0}})
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_run_disposition([td])
        assert rep['status'] == 'warn', rep
        panel = rep['panels'][td]
        assert panel['ledger'] == 'ok'
        assert panel['plan_present'] is True
        assert panel['retained_source'] == 'summary.json'
        assert panel['audit']['n_cells_retried'] == 1
        # Retaining the failed attempt is not a measurement: it must fail.
        _write_summary(td, {
            'a::seed0': {'run_id': 'r1', 'run_cfg': 'c1', 'ppl': 10.0},
            'b::seed0': {'run_id': 'r3', 'run_cfg': 'c1', 'ppl': 11.0}})
        with contextlib.redirect_stdout(io.StringIO()):
            bad = A.audit_run_disposition([td])
        assert bad['status'] == 'fail'
        # The CLI alias accepts the panel path and exits non-zero on fail.
        old_argv = _sys.argv
        try:
            _sys.argv = ['audit_gates.py', 'disposition', td]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    A._run_cli()
            except SystemExit as e:
                assert e.code == 1
            else:
                raise AssertionError('a failed disposition must exit non-zero')
            _sys.argv = ['audit_gates.py', 'attempts']
            with contextlib.redirect_stdout(io.StringIO()):
                A._run_cli()  # no local panel -> not_verifiable, exit 0
        finally:
            _sys.argv = old_argv


def test_tampered_ledger_fails_the_disposition_gate():
    with tempfile.TemporaryDirectory() as td:
        _write_plan(td, ['a::seed0'])
        _log(td, _ev('a::seed0', 'ok', 'r1'))
        _write_summary(td, {'a::seed0': {'run_id': 'r1', 'run_cfg': 'c1',
                                         'ppl': 9.0}})
        path = os.path.join(td, 'experiments.jsonl')
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        rec = json.loads(lines[0])
        rec['status'] = 'error'
        lines[0] = json.dumps(rec, sort_keys=True) + '\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_run_disposition([td])
        assert rep['status'] == 'fail', rep
        assert any('hash-chain' in p or 'ledger' in p.lower()
                   for p in rep['problems'])
